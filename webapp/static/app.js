const fileInput = document.getElementById("fileInput");
const fileList = document.getElementById("fileList");
const runButton = document.getElementById("runButton");
const statusBox = document.getElementById("status");
const resultsBody = document.querySelector("#resultsTable tbody");
const logsBox = document.getElementById("logs");
const dropzone = document.getElementById("dropzone");
const clearFilesButton = document.getElementById("clearFilesButton");
const progressCard = document.getElementById("progressCard");
const progressPercent = document.getElementById("progressPercent");
const progressFill = document.getElementById("progressFill");
const progressLogs = document.getElementById("progressLogs");
const downloadCsvButton = document.getElementById("downloadCsvButton");
const evaluationPanel = document.getElementById("evaluationPanel");
const groundTruthBody = document.querySelector("#groundTruthTable tbody");
const calculateMetricsButton = document.getElementById("calculateMetricsButton");
const metricsStatus = document.getElementById("metricsStatus");
const metricsResult = document.getElementById("metricsResult");
const metricsCards = document.getElementById("metricsCards");
const metricsNotes = document.getElementById("metricsNotes");
const cmTn = document.getElementById("cm-tn");
const cmFp = document.getElementById("cm-fp");
const cmFn = document.getElementById("cm-fn");
const cmTp = document.getElementById("cm-tp");

let selectedFiles = [];
let progressTimer = null;
let latestPredictionRows = [];

async function readApiPayload(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }

  const text = await response.text();
  return {
    ok: false,
    message: text ? `Server returned non-JSON response: ${text.slice(0, 140)}` : "Server returned an empty response.",
  };
}

function setStatus(message, level = "") {
  statusBox.textContent = message;
  statusBox.className = `status ${level}`.trim();
}

function renderSelectedFiles() {
  fileList.innerHTML = "";

  if (!selectedFiles.length) {
    const li = document.createElement("li");
    li.textContent = "No files selected.";
    fileList.appendChild(li);
    return;
  }

  selectedFiles
    .slice()
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
    .forEach((file) => {
      const li = document.createElement("li");
      li.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
      fileList.appendChild(li);
    });
}

function updateFiles(fileListObject) {
  const incomingFiles = Array.from(fileListObject || []);
  const existingByKey = new Map(
    selectedFiles.map((file) => [`${file.name}::${file.size}::${file.lastModified}`, file])
  );

  incomingFiles.forEach((file) => {
    const key = `${file.name}::${file.size}::${file.lastModified}`;
    if (!existingByKey.has(key)) {
      existingByKey.set(key, file);
    }
  });

  selectedFiles = Array.from(existingByKey.values());
  renderSelectedFiles();
  if (selectedFiles.length) {
    setStatus(`${selectedFiles.length} file(s) selected. Ready to run prediction.`);
  } else {
    setStatus("Select files to begin.");
  }
}

function clearSelectedFiles() {
  selectedFiles = [];
  fileInput.value = "";
  renderSelectedFiles();
  setStatus("All selected files cleared.");
}

function renderResults(rows) {
  latestPredictionRows = Array.isArray(rows) ? rows : [];
  resultsBody.innerHTML = "";

  if (!rows || !rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = "<td colspan=\"5\">No result rows were generated.</td>";
    resultsBody.appendChild(tr);
    return;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.Left_Image || ""}</td>
      <td>${row.Right_Image || ""}</td>
      <td>${row.Prediction || ""}</td>
      <td>${row.Probability || ""}</td>
      <td>${row.Confidence || ""}</td>
    `;
    resultsBody.appendChild(tr);
  });

  renderGroundTruthTable(latestPredictionRows);
}

function setMetricsStatus(message, level = "") {
  metricsStatus.textContent = message;
  metricsStatus.className = `status ${level}`.trim();
}

function renderGroundTruthTable(rows) {
  groundTruthBody.innerHTML = "";

  if (!rows || !rows.length) {
    evaluationPanel.classList.add("hidden");
    return;
  }

  evaluationPanel.classList.remove("hidden");
  metricsResult.classList.add("hidden");
  setMetricsStatus("Select ground truth labels and click Calculate Measures.");

  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.Left_Image || ""}</td>
      <td>${row.Right_Image || ""}</td>
      <td>${row.Prediction || ""}</td>
      <td>${row.Probability || ""}</td>
      <td>
        <select class="gt-select" data-row-index="${index}">
          <option value="">Select</option>
          <option value="Control">Control</option>
          <option value="Diabetic">Diabetic</option>
        </select>
      </td>
    `;
    groundTruthBody.appendChild(tr);
  });
}

function formatMetricName(name) {
  return name
    .replaceAll("_", " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function metricValueLabel(value) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  return `${value}`;
}

function renderMetrics(payload) {
  const metrics = payload.metrics || {};
  const cm = payload.confusionMatrix || { tn: 0, fp: 0, fn: 0, tp: 0 };

  metricsCards.innerHTML = "";
  Object.entries(metrics).forEach(([key, value]) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `<span class="metric-name">${formatMetricName(key)}</span><strong class="metric-value">${metricValueLabel(value)}</strong>`;
    metricsCards.appendChild(card);
  });

  cmTn.textContent = `${cm.tn ?? 0}`;
  cmFp.textContent = `${cm.fp ?? 0}`;
  cmFn.textContent = `${cm.fn ?? 0}`;
  cmTp.textContent = `${cm.tp ?? 0}`;

  const notes = [];
  notes.push(`Used samples: ${payload.usedSamples ?? 0}`);
  if (Array.isArray(payload.skippedRows) && payload.skippedRows.length) {
    notes.push("Skipped rows:");
    payload.skippedRows.forEach((line) => notes.push(`- ${line}`));
  }
  metricsNotes.textContent = notes.join("\n");

  metricsResult.classList.remove("hidden");
}

async function calculateMeasures() {
  if (!latestPredictionRows.length) {
    setMetricsStatus("Run prediction first.", "error");
    return;
  }

  const gtSelects = Array.from(document.querySelectorAll(".gt-select"));
  const rowsPayload = latestPredictionRows.map((row, index) => {
    const select = gtSelects.find((item) => Number(item.dataset.rowIndex) === index);
    return {
      ...row,
      groundTruth: select ? select.value : "",
    };
  });

  const unselectedCount = rowsPayload.filter((row) => !row.groundTruth).length;
  if (unselectedCount > 0) {
    setMetricsStatus(`Please select ground truth for all rows. Missing: ${unselectedCount}.`, "error");
    return;
  }

  calculateMetricsButton.disabled = true;
  setMetricsStatus("Calculating evaluation measures...", "");

  try {
    const response = await fetch("/api/calculate-metrics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: rowsPayload }),
    });
    const payload = await readApiPayload(response);

    if (!response.ok || !payload.ok) {
      setMetricsStatus(payload.message || "Could not calculate measures.", "error");
      return;
    }

    renderMetrics(payload);
    setMetricsStatus("Metrics calculated successfully.", "success");
  } catch (error) {
    setMetricsStatus(`Failed to calculate metrics: ${error.message}`, "error");
  } finally {
    calculateMetricsButton.disabled = false;
  }
}

function showProgressBox() {
  progressCard.classList.remove("hidden");
}

function setProgress(percent, logs) {
  const safePercent = Math.max(0, Math.min(100, Number(percent) || 0));
  progressPercent.textContent = `${safePercent}%`;
  progressFill.style.width = `${safePercent}%`;
  if (Array.isArray(logs)) {
    progressLogs.textContent = logs.slice(-18).join("\n");
    progressLogs.scrollTop = progressLogs.scrollHeight;
  }
}

async function pollProgress(jobId) {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }

  return new Promise((resolve) => {
    progressTimer = setInterval(async () => {
      try {
        const response = await fetch(`/api/progress/${jobId}`);
        const payload = await readApiPayload(response);

        if (!response.ok || !payload.ok) {
          clearInterval(progressTimer);
          progressTimer = null;
          setStatus(payload.message || "Could not read progress.", "error");
          runButton.disabled = false;
          resolve(null);
          return;
        }

        setProgress(payload.percent, payload.logs || []);
        setStatus(payload.message || "Running prediction...", "");

        if (payload.status === "completed") {
          clearInterval(progressTimer);
          progressTimer = null;
          renderResults(payload.rows || []);
          logsBox.textContent = (payload.logs || []).join("\n");
          downloadCsvButton.href = `/api/download-results-csv?jobId=${encodeURIComponent(payload.jobId)}`;
          downloadCsvButton.classList.remove("hidden");
          const skippedPatientsInfo = payload.skippedPatients?.length
            ? ` Skipped unpaired patients: ${payload.skippedPatients.join(", ")}.`
            : "";
          if (payload.invalidFiles?.length) {
            setStatus(`Prediction completed. Some files were ignored: ${payload.invalidFiles.join(", ")}.${skippedPatientsInfo}`, "success");
          } else {
            setStatus(`Prediction completed successfully.${skippedPatientsInfo}`, "success");
          }
          runButton.disabled = false;
          resolve(payload);
          return;
        }

        if (payload.status === "failed") {
          clearInterval(progressTimer);
          progressTimer = null;
          const fullLogs = [...(payload.logs || [])];
          if (payload.error) {
            fullLogs.push("", payload.error);
          }
          logsBox.textContent = fullLogs.join("\n");
          setStatus(payload.message || "Prediction failed.", "error");
          runButton.disabled = false;
          resolve(payload);
        }

        if (payload.status === "cancelled") {
          clearInterval(progressTimer);
          progressTimer = null;
          logsBox.textContent = (payload.logs || []).join("\n");
          setStatus(payload.message || "Prediction cancelled.", "error");
          runButton.disabled = false;
          resolve(payload);
        }
      } catch (error) {
        clearInterval(progressTimer);
        progressTimer = null;
        setStatus(`Progress polling failed: ${error.message}`, "error");
        runButton.disabled = false;
        resolve(null);
      }
    }, 1000);
  });
}

function cancelActiveJob(reason) {
  try {
    const body = JSON.stringify({ reason });
    if (navigator.sendBeacon) {
      const blob = new Blob([body], { type: "application/json" });
      navigator.sendBeacon("/api/cancel-active-job", blob);
      return;
    }

    fetch("/api/cancel-active-job", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true,
    }).catch(() => {});
  } catch (_) {
    // Ignore cancellation best-effort failures on unload.
  }
}

async function runPrediction() {
  if (!selectedFiles.length) {
    setStatus("Please select files before running prediction.", "error");
    return;
  }

  const formData = new FormData();
  selectedFiles.forEach((file) => formData.append("files", file));

  runButton.disabled = true;
  downloadCsvButton.classList.add("hidden");
  evaluationPanel.classList.add("hidden");
  latestPredictionRows = [];
  showProgressBox();
  setProgress(0, ["Starting prediction..."]);
  setStatus("Running prediction pipeline. This may take a while...", "");

  try {
    const response = await fetch("/api/run-prediction", {
      method: "POST",
      body: formData,
    });

    const payload = await readApiPayload(response);

    if (!response.ok || !payload.ok) {
      const invalidInfo = payload.invalidFiles?.length
        ? ` Invalid files: ${payload.invalidFiles.join(", ")}`
        : "";
      const skippedInfo = payload.skippedPatients?.length
        ? ` Skipped unpaired patients: ${payload.skippedPatients.join(", ")}.`
        : "";
      setStatus((payload.message || "Prediction request failed.") + invalidInfo + skippedInfo, "error");
      runButton.disabled = false;
      return;
    }

    if (payload.skippedPatients?.length) {
      setStatus(`Prediction started. Skipping unpaired patients: ${payload.skippedPatients.join(", ")}.`, "");
    }

    await pollProgress(payload.jobId);
  } catch (error) {
    setStatus(`Failed to reach backend: ${error.message}`, "error");
    runButton.disabled = false;
  }
}

fileInput.addEventListener("change", (event) => {
  updateFiles(event.target.files);
});

runButton.addEventListener("click", runPrediction);
clearFilesButton.addEventListener("click", clearSelectedFiles);
calculateMetricsButton.addEventListener("click", calculateMeasures);

window.addEventListener("beforeunload", () => {
  cancelActiveJob("Prediction cancelled due to page refresh.");
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.add("drag-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.remove("drag-active");
  });
});

dropzone.addEventListener("drop", (event) => {
  updateFiles(event.dataTransfer.files);
});

renderSelectedFiles();
setStatus("Select files to begin.");
