import csv
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from typing import Dict, List

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from werkzeug.utils import secure_filename

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

REALDATA_IMAGES_DIR = os.path.join(PROJECT_ROOT, "realdata", "images")
PREDICTION_CSV = os.path.join(PROJECT_ROOT, "realdata_predictions.csv")
WEBAPP_RUNS_DIR = os.path.join(PROJECT_ROOT, "temp", "webapp_runs")
ALLOWED_EXTENSIONS = {"jpg", "jpeg"}

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
if not cors_origins:
    cors_origins = ["*"]

CORS(app, resources={r"/api/*": {"origins": cors_origins}})

_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()
_backend_logs: List[str] = []
_backend_logs_lock = threading.Lock()


def _append_backend_log(line: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _backend_logs_lock:
        _backend_logs.append(f"[{timestamp}] {line}")
        if len(_backend_logs) > 5000:
            _backend_logs[:] = _backend_logs[-5000:]


_append_backend_log("backend process initialized")


def _set_job_fields(job_id: str, updates: Dict) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


def _append_job_log(job_id: str, line: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["logs"].append(line)
        if len(job["logs"]) > 300:
            job["logs"] = job["logs"][-300:]
        job["lastUpdated"] = time.time()
        job["message"] = line

    _append_backend_log(f"job={job_id} {line}")


def _maybe_update_percent_from_log(job_id: str, line: str) -> None:
    pair_match = re.search(r"\[PAIR\s+(\d+)/(\d+)\]", line, re.IGNORECASE)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        current = int(job.get("percent", 0))

        if pair_match:
            current_pair = int(pair_match.group(1))
            total_pairs = max(int(pair_match.group(2)), 1)
            # Reserve 20% for mask generation and 80% for pair predictions.
            percent = 20 + int((current_pair / total_pairs) * 80)
            job["percent"] = max(current, min(percent, 99))
            return

        if "Generating masks for realdata" in line:
            percent_match = re.search(r"(\d{1,3})%", line)
            if percent_match:
                mask_percent = min(int(percent_match.group(1)), 100)
                estimated = int(mask_percent * 0.2)
                job["percent"] = max(current, estimated)
                return


def _start_prediction_job(job_id: str) -> None:
    try:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return
            run_root = job.get("runRoot")

        if not run_root:
            raise RuntimeError("Missing run directory for prediction job.")

        _append_backend_log(f"job={job_id} prediction started")

        _set_job_fields(job_id, {"status": "running", "percent": 1, "message": "Starting prediction pipeline..."})

        cmd = [
            sys.executable,
            "-c",
            "from src.predict_realdata import predict_realdata; predict_realdata()",
        ]

        process = subprocess.Popen(
            cmd,
            cwd=run_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "PYTHONPATH": PROJECT_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
            },
        )

        _set_job_fields(job_id, {"process": process})

        if process.stdout is not None:
            for line in process.stdout:
                stripped = line.strip()
                if stripped:
                    _append_job_log(job_id, stripped)
                    _maybe_update_percent_from_log(job_id, stripped)

        return_code = process.wait()
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job and job.get("status") == "cancelled":
                return

        if return_code != 0:
            raise RuntimeError(f"Prediction process exited with code {return_code}")

        csv_path = os.path.join(run_root, "realdata_predictions.csv")
        rows = _read_prediction_csv(csv_path)
        _set_job_fields(
            job_id,
            {
                "status": "completed",
                "percent": 100,
                "message": "Prediction completed successfully.",
                "rows": rows,
                "csvPath": csv_path,
                "process": None,
                "completedAt": time.time(),
            },
        )
        _append_backend_log(f"job={job_id} prediction completed")
    except Exception as exc:  # noqa: BLE001
        _append_backend_log(f"job={job_id} prediction failed: {exc}")
        _set_job_fields(
            job_id,
            {
                "status": "failed",
                "percent": 100,
                "message": f"Prediction failed: {exc}",
                "error": traceback.format_exc(),
                "process": None,
                "completedAt": time.time(),
            },
        )


def _ensure_run_images_directory(run_root: str) -> str:
    images_dir = os.path.join(run_root, "realdata", "images")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def _has_active_job() -> bool:
    with _jobs_lock:
        for job in _jobs.values():
            if job.get("status") in {"queued", "running"}:
                return True
    return False


def _is_allowed_file(filename: str) -> bool:
    if not filename:
        return False

    # Some clients may submit path-like filenames; always validate only the leaf name.
    leaf_name = os.path.basename(filename.replace("\\", "/")).strip()
    if not leaf_name:
        return False

    _, ext = os.path.splitext(leaf_name)
    extension = ext.lstrip(".").lower()
    return extension in ALLOWED_EXTENSIONS


def _extract_patient_eye(filename: str):
    leaf_name = os.path.basename(filename.replace("\\", "/")).strip()
    match = re.match(r"^(\d+)([LR])(?:\.|$)", leaf_name, re.IGNORECASE)
    if not match:
        return None
    return match.group(1), match.group(2).upper()


@app.route("/api/download-results-csv", methods=["GET"])
def download_results_csv():
    requested_job_id = (request.args.get("jobId") or "").strip()
    csv_path = None

    with _jobs_lock:
        if requested_job_id and requested_job_id in _jobs:
            csv_path = _jobs[requested_job_id].get("csvPath")
        else:
            completed_jobs = [j for j in _jobs.values() if j.get("status") == "completed" and j.get("csvPath")]
            if completed_jobs:
                completed_jobs.sort(key=lambda j: j.get("completedAt", 0), reverse=True)
                csv_path = completed_jobs[0].get("csvPath")

    if not csv_path or not os.path.exists(csv_path):
        return jsonify({"ok": False, "message": "Prediction CSV not found. Run prediction first."}), 404

    return send_file(
        csv_path,
        as_attachment=True,
        download_name="realdata_predictions.csv",
        mimetype="text/csv",
    )


def _read_prediction_csv(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def _label_to_int(label_text: str):
    normalized = (label_text or "").strip().lower()
    if normalized == "diabetic":
        return 1
    if normalized == "control":
        return 0
    return None


def _safe_round(value, digits: int = 4):
    if value is None:
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return round(float(value), digits)
    except Exception:  # noqa: BLE001
        return None


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "message": "Backend is running."})


@app.route("/api/backend-logs", methods=["GET"])
def backend_logs():
    limit_text = (request.args.get("limit") or "0").strip()
    try:
        limit = int(limit_text) if limit_text else 0
    except ValueError:
        limit = 0

    with _backend_logs_lock:
        logs = _backend_logs[-limit:] if limit > 0 else list(_backend_logs)
        total = len(_backend_logs)

    return jsonify(
        {
            "ok": True,
            "count": len(logs),
            "total": total,
            "logs": logs,
        }
    )


@app.route("/", methods=["GET"])
def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)

    return jsonify(
        {
            "ok": True,
            "message": "Iris Diabetes backend is running.",
            "health": "/api/health",
            "predict": "/api/run-prediction",
        }
    )


@app.route("/api/run-prediction", methods=["POST"])
def run_prediction():
    if _has_active_job():
        return jsonify({"ok": False, "message": "A prediction job is already running. Please wait for it to finish."}), 409

    uploaded_files = request.files.getlist("files")

    if not uploaded_files:
        return jsonify({"ok": False, "message": "Please select image files first."}), 400

    invalid_files = []
    valid_files = []
    upload_entries = []

    for file_storage in uploaded_files:
        original_name = (file_storage.filename or "").strip()
        original_name = os.path.basename(original_name.replace("\\", "/"))
        if not original_name:
            continue
        if not _is_allowed_file(original_name):
            invalid_files.append(original_name)
        else:
            valid_files.append(file_storage)
            parsed = _extract_patient_eye(original_name)
            upload_entries.append(
                {
                    "name": original_name,
                    "storage": file_storage,
                    "patient": parsed[0] if parsed else None,
                    "eye": parsed[1] if parsed else None,
                }
            )

    if not valid_files:
        return jsonify(
            {
                "ok": False,
                "message": "No valid files found. Upload JPG/JPEG files only. File name pairing is handled by your prediction script.",
                "invalidFiles": invalid_files,
            }
        ), 400

    patient_to_eyes = {}
    for entry in upload_entries:
        if not entry["patient"]:
            continue
        patient = entry["patient"]
        if patient not in patient_to_eyes:
            patient_to_eyes[patient] = {"L": [], "R": []}
        patient_to_eyes[patient][entry["eye"]].append(entry)

    paired_patients = set()
    skipped_patients = []
    skipped_files = []

    for patient, eyes in patient_to_eyes.items():
        if eyes["L"] and eyes["R"]:
            paired_patients.add(patient)
        else:
            missing_eye = "R" if eyes["L"] else "L"
            skipped_patients.append(f"{patient} (missing {missing_eye})")
            skipped_files.extend([entry["name"] for entry in eyes["L"] + eyes["R"]])

    paired_entries = [
        entry for entry in upload_entries if entry["patient"] in paired_patients
    ]

    unparsable_files = [entry["name"] for entry in upload_entries if not entry["patient"]]
    if unparsable_files:
        skipped_files.extend(unparsable_files)

    if not paired_entries:
        return jsonify(
            {
                "ok": False,
                "message": "No complete left/right pairs found. Unpaired patients were skipped.",
                "invalidFiles": invalid_files,
                "skippedPatients": skipped_patients,
                "skippedFiles": sorted(set(skipped_files)),
            }
        ), 400

    job_id = uuid.uuid4().hex
    run_root = os.path.join(WEBAPP_RUNS_DIR, job_id)

    try:
        if os.path.exists(run_root):
            shutil.rmtree(run_root)
        images_dir = _ensure_run_images_directory(run_root)

        for entry in paired_entries:
            file_storage = entry["storage"]
            leaf_name = os.path.basename((file_storage.filename or "").replace("\\", "/")).strip()
            safe_name = secure_filename(leaf_name)
            save_path = os.path.join(images_dir, safe_name)
            file_storage.save(save_path)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "message": f"Failed to save uploaded files: {exc}"}), 500

    initial_logs = ["Queued for prediction..."]
    if skipped_patients:
        initial_logs.append(f"[WARN] Unpaired patients skipped: {', '.join(skipped_patients)}")
    if unparsable_files:
        initial_logs.append(f"[WARN] Skipped files with unrecognized pattern: {', '.join(unparsable_files)}")

    with _jobs_lock:
        _jobs[job_id] = {
            "jobId": job_id,
            "status": "queued",
            "percent": 0,
            "message": "Queued for prediction...",
            "logs": initial_logs,
            "rows": [],
            "invalidFiles": invalid_files,
            "uploadedCount": len(valid_files),
            "pairedUploadCount": len(paired_entries),
            "skippedPatients": skipped_patients,
            "skippedFiles": sorted(set(skipped_files)),
            "runRoot": run_root,
            "createdAt": time.time(),
            "lastUpdated": time.time(),
        }

    worker = threading.Thread(target=_start_prediction_job, args=(job_id,), daemon=True)
    worker.start()

    return (
        jsonify(
            {
                "ok": True,
                "jobId": job_id,
                "message": "Prediction started.",
                "invalidFiles": invalid_files,
                "uploadedCount": len(valid_files),
                "pairedUploadCount": len(paired_entries),
                "skippedPatients": skipped_patients,
                "skippedFiles": sorted(set(skipped_files)),
            }
        ),
        202,
    )


@app.route("/api/progress/<job_id>", methods=["GET"])
def get_progress(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"ok": False, "message": "Job not found."}), 404

        response = {
            "ok": True,
            "jobId": job["jobId"],
            "status": job["status"],
            "percent": job["percent"],
            "message": job.get("message", ""),
            "logs": job.get("logs", []),
            "rows": job.get("rows", []),
            "invalidFiles": job.get("invalidFiles", []),
            "uploadedCount": job.get("uploadedCount", 0),
            "pairedUploadCount": job.get("pairedUploadCount", 0),
            "skippedPatients": job.get("skippedPatients", []),
            "skippedFiles": job.get("skippedFiles", []),
            "error": job.get("error", ""),
        }

    return jsonify(response)


@app.route("/api/cancel-active-job", methods=["POST"])
def cancel_active_job():
    payload = request.get_json(silent=True) or {}
    reason = (payload.get("reason") or "Cancelled by user refresh.").strip()
    cancelled_count = 0

    with _jobs_lock:
        active_jobs = [
            job for job in _jobs.values() if job.get("status") in {"queued", "running"}
        ]

        for job in active_jobs:
            proc = job.get("process")
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    if proc.poll() is None:
                        proc.kill()
                except Exception:  # noqa: BLE001
                    pass

            job["status"] = "cancelled"
            job["percent"] = 100
            job["message"] = reason
            job["process"] = None
            job["completedAt"] = time.time()
            logs = job.get("logs", [])
            logs.append(f"[INFO] {reason}")
            job["logs"] = logs[-300:]
            cancelled_count += 1

                _append_backend_log(f"job={job['jobId']} cancelled: {reason}")

    return jsonify({"ok": True, "cancelled": cancelled_count})


@app.route("/api/calculate-metrics", methods=["POST"])
def calculate_metrics():
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows")

    if not isinstance(rows, list) or not rows:
        return jsonify({"ok": False, "message": "No rows provided for metrics calculation."}), 400

    y_true = []
    y_pred = []
    y_score = []
    skipped_rows = []

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            skipped_rows.append(f"Row {idx}: invalid row format")
            continue

        gt_label = row.get("groundTruth", "")
        pred_label = row.get("Prediction", "")
        prob_text = row.get("Probability", "")

        gt_value = _label_to_int(gt_label)
        pred_value = _label_to_int(pred_label)

        if gt_value is None:
            skipped_rows.append(f"Row {idx}: missing/invalid ground truth")
            continue
        if pred_value is None:
            skipped_rows.append(f"Row {idx}: missing/invalid predicted label")
            continue

        y_true.append(gt_value)
        y_pred.append(pred_value)

        try:
            y_score.append(float(prob_text))
        except Exception:  # noqa: BLE001
            y_score.append(float(pred_value))

    if not y_true:
        return jsonify({"ok": False, "message": "No valid rows available after validation.", "skippedRows": skipped_rows}), 400

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else None
    npv = (tn / (tn + fn)) if (tn + fn) > 0 else None

    auc_roc = None
    if len(set(y_true)) > 1:
        try:
            auc_roc = roc_auc_score(y_true, y_score)
        except Exception:  # noqa: BLE001
            auc_roc = None

    result = {
        "ok": True,
        "usedSamples": len(y_true),
        "skippedRows": skipped_rows,
        "confusionMatrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "metrics": {
            "accuracy": _safe_round(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": _safe_round(balanced_accuracy_score(y_true, y_pred)),
            "precision": _safe_round(precision_score(y_true, y_pred, zero_division=0)),
            "recall_sensitivity": _safe_round(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": _safe_round(specificity),
            "f1_score": _safe_round(f1_score(y_true, y_pred, zero_division=0)),
            "npv": _safe_round(npv),
            "mcc": _safe_round(matthews_corrcoef(y_true, y_pred)) if len(set(y_true + y_pred)) > 1 else None,
            "auc_roc": _safe_round(auc_roc),
        },
    }

    return jsonify(result)


if __name__ == "__main__":
    _append_backend_log("backend process started")
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
