#!/usr/bin/env bash
set -euo pipefail

cd /app

mkdir -p models
mkdir -p temp/webapp_runs
mkdir -p realdata/images realdata/masks realdata/pancreatic_masks

# Optional model bootstrap from remote storage.
# Set MODEL_BASE_URL and (optionally) MODEL_FILES in Render env vars.
if [[ -n "${MODEL_BASE_URL:-}" ]]; then
  IFS=',' read -r -a MODEL_LIST <<< "${MODEL_FILES:-best_f1_model_fold_1.pth,best_f1_model_fold_2.pth,best_f1_model_fold_3.pth,best_f1_model_fold_4.pth,best_f1_model_fold_5.pth,best_iris_model_3class.pth,best_iris_model_2class.pth}"

  echo "[INFO] MODEL_BASE_URL detected. Ensuring model files are present in /app/models..."
  for f in "${MODEL_LIST[@]}"; do
    file_trimmed="$(echo "$f" | xargs)"
    [[ -z "$file_trimmed" ]] && continue

    if [[ ! -f "/app/models/$file_trimmed" ]]; then
      echo "[INFO] Downloading $file_trimmed"
      curl -fL "${MODEL_BASE_URL%/}/$file_trimmed" -o "/app/models/$file_trimmed"
    fi
  done
fi

PORT_VALUE="${PORT:-10000}"
WORKERS="${GUNICORN_WORKERS:-1}"
THREADS="${GUNICORN_THREADS:-4}"
TIMEOUT="${GUNICORN_TIMEOUT:-900}"

exec gunicorn \
  --workers "$WORKERS" \
  --threads "$THREADS" \
  --worker-class gthread \
  --bind "0.0.0.0:${PORT_VALUE}" \
  --timeout "$TIMEOUT" \
  webapp.app:app
