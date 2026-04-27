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
const backendConnectionBadge = document.getElementById("backendConnectionBadge");
const loadBackendLogsButton = document.getElementById("loadBackendLogsButton");
const backendLogsStatus = document.getElementById("backendLogsStatus");
const backendLogs = document.getElementById("backendLogs");
const loginGate = document.getElementById("loginGate");
const appContent = document.getElementById("appContent");
const loginBackendState = document.getElementById("loginBackendState");
const loginForm = document.getElementById("loginForm");
const loginUser = document.getElementById("loginUser");
const loginPass = document.getElementById("loginPass");
const loginButton = document.getElementById("loginButton");
const retryBackendButton = document.getElementById("retryBackendButton");
const loginStatus = document.getElementById("loginStatus");
const toastContainer = document.getElementById("toastContainer");

const API_BASE = (window.__API_BASE__ || "").replace(/\/$/, "");

function apiUrl(path) {
  if (!API_BASE) {
    return path;
  }
  return `${API_BASE}${path}`;
}

let selectedFiles = [];
let progressTimer = null;
let latestPredictionRows = [];
let backendReady = false;
let backendAlertShown = false;
let isAuthenticated = false;

const VALID_USERNAME = "decoders123";
const VALID_PASSWORD = "decoders123";

function setLoginStatus(message, level = "") {
  loginStatus.textContent = message;
  loginStatus.className = `status ${level}`.trim();
}

function showToast(message, type = "info", timeoutMs = 3600) {
  if (!toastContainer) {
    return;
  }

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.setAttribute("role", type === "error" ? "alert" : "status");
  toast.innerHTML = `
    <div class="toast-content">
      <strong>${type === "error" ? "Error" : type === "success" ? "Success" : "Notice"}</strong>
      <span>${message}</span>
    </div>
    <button type="button" class="toast-close" aria-label="Dismiss notification">&times;</button>
  `;

  const closeToast = () => {
    toast.classList.add("closing");
    window.setTimeout(() => toast.remove(), 180);
  };

  toast.querySelector(".toast-close").addEventListener("click", closeToast);
  toastContainer.appendChild(toast);

  window.setTimeout(closeToast, timeoutMs);
}

function setLoginEnabled(enabled) {
  loginButton.disabled = !enabled;
  loginUser.disabled = !enabled;
  loginPass.disabled = !enabled;
}

function updateAccessGate() {
  const canEnter = backendReady && isAuthenticated;

  if (appContent) {
    appContent.classList.toggle("locked", !canEnter);
    appContent.setAttribute("aria-hidden", canEnter ? "false" : "true");
  }

  if (loginGate) {
    loginGate.classList.toggle("hidden", canEnter);
  }

  document.body.classList.toggle("app-gated", !canEnter);
  setUiEnabled(canEnter);
}

function setBackendConnectionState(state, message) {
  if (backendConnectionBadge) {
    backendConnectionBadge.classList.remove("connected", "disconnected", "checking");
    backendConnectionBadge.classList.add(state);
    backendConnectionBadge.textContent = message;
  }

  if (loginBackendState) {
    loginBackendState.classList.remove("connected", "disconnected", "checking");
    loginBackendState.classList.add(state);
    loginBackendState.textContent = message;
  }
}

function setUiEnabled(enabled) {
  fileInput.disabled = !enabled;
  runButton.disabled = !enabled;
  clearFilesButton.disabled = !enabled;
  calculateMetricsButton.disabled = !enabled;
  loadBackendLogsButton.disabled = !enabled;
  dropzone.style.pointerEvents = enabled ? "" : "none";
  dropzone.style.opacity = enabled ? "" : "0.55";
}

function blockWithBackendAlert(message) {
  backendReady = false;
  setUiEnabled(false);
  setLoginEnabled(false);
  setBackendConnectionState("disconnected", "Backend: Disconnected");
  setLoginStatus("Backend not connected. Click Retry Connection.", "error");
  updateAccessGate();
  setStatus(message, "error");
  if (!backendAlertShown) {
    showToast("Backend not connected. Please try again later.", "error", 5000);
    backendAlertShown = true;
  }
}

function requireBackendConnection() {
  if (backendReady) {
    return true;
  }
  blockWithBackendAlert("Backend not connected. Please try again later.");
  return false;
}

async function checkBackendConnection() {
  setUiEnabled(false);
  setLoginEnabled(false);
  setBackendConnectionState("checking", "Backend: Checking...");
  setLoginStatus("Checking backend connection...", "");
  setStatus("Checking backend connection...", "");

  let timeoutId = null;
  try {
    const controller = new AbortController();
    timeoutId = setTimeout(() => controller.abort(), 8000);

    const response = await fetch(apiUrl("/api/health"), {
      method: "GET",
      signal: controller.signal,
    });
    const payload = await readApiPayload(response);

    if (!response.ok || !payload.ok) {
      const errorMessage = payload.message || `Health check failed with status ${response.status}.`;
      throw new Error(errorMessage);
    }

    backendReady = true;
    backendAlertShown = false;
    setLoginEnabled(true);
    setBackendConnectionState("connected", "Backend: Connected");
    setLoginStatus("Backend connected. Enter valid credentials.", "success");
    setStatus("Backend connected. Login required to continue.", "success");
    updateAccessGate();
  } catch (error) {
    backendReady = false;
    blockWithBackendAlert(`Backend not connected: ${error.message}`);
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
}

function handleLoginSubmit(event) {
  event.preventDefault();

  if (!backendReady) {
    setLoginStatus("Backend not connected. Please retry connection first.", "error");
    showToast("Backend not connected. Please try again later.", "error", 5000);
    return;
  }

  const username = (loginUser.value || "").trim();
  const password = loginPass.value || "";

  if (username !== VALID_USERNAME || password !== VALID_PASSWORD) {
    isAuthenticated = false;
    updateAccessGate();
    setLoginStatus("Invalid ID or password.", "error");
    showToast("Invalid login credentials.", "error", 4200);
    return;
  }

  isAuthenticated = true;
  setLoginStatus("Login successful. Entering website...", "success");
  setStatus("Login successful. You can now use the website.", "success");
  showToast("Login successful. Welcome.", "success", 2800);
  updateAccessGate();
}

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

async function readXhrPayload(xhr) {
  const contentType = xhr.getResponseHeader("content-type") || "";
  const responseText = xhr.responseText || "";

  if (contentType.includes("application/json")) {
    try {
      return JSON.parse(responseText || "{}");
    } catch (error) {
      return {
        ok: false,
        message: `Server returned invalid JSON: ${error.message}`,
      };
    }
  }

  return {
    ok: false,
    message: responseText ? `Server returned non-JSON response: ${responseText.slice(0, 140)}` : "Server returned an empty response.",
  };
}

function sendFormDataWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);

    xhr.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) {
        return;
      }

      const percent = Math.round((event.loaded / event.total) * 100);
      onProgress(percent, event.loaded, event.total);
    });

    xhr.addEventListener("load", async () => {
      try {
        resolve({
          status: xhr.status,
          payload: await readXhrPayload(xhr),
        });
      } catch (error) {
        reject(error);
      }
    });

    xhr.addEventListener("error", () => {
      reject(new Error("Upload failed due to a network error."));
    });

    xhr.addEventListener("abort", () => {
      reject(new Error("Upload was cancelled."));
    });

    xhr.send(formData);
  });
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
  if (!requireBackendConnection()) {
    return;
  }

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
  if (!requireBackendConnection()) {
    return;
  }

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

function setBackendLogsStatus(message, level = "") {
  backendLogsStatus.textContent = message;
  backendLogsStatus.className = `status ${level}`.trim();
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
  if (!requireBackendConnection()) {
    return;
  }

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
    const response = await fetch(apiUrl("/api/calculate-metrics"), {
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

async function loadBackendLogs() {
  if (!requireBackendConnection()) {
    return;
  }

  loadBackendLogsButton.disabled = true;
  setBackendLogsStatus("Loading backend logs...", "");

  try {
    const response = await fetch(apiUrl("/api/backend-logs"), {
      method: "GET",
    });
    const payload = await readApiPayload(response);

    if (!response.ok || !payload.ok) {
      setBackendLogsStatus(payload.message || "Failed to load backend logs.", "error");
      return;
    }

    const lines = Array.isArray(payload.logs) ? payload.logs : [];
    backendLogs.textContent = lines.length ? lines.join("\n") : "No backend logs available yet.";
    setBackendLogsStatus(`Loaded ${lines.length} backend log lines.`, "success");
  } catch (error) {
    setBackendLogsStatus(`Failed to load backend logs: ${error.message}`, "error");
  } finally {
    loadBackendLogsButton.disabled = false;
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
        const response = await fetch(apiUrl(`/api/progress/${jobId}`));
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
          downloadCsvButton.href = apiUrl(`/api/download-results-csv?jobId=${encodeURIComponent(payload.jobId)}`);
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
      navigator.sendBeacon(apiUrl("/api/cancel-active-job"), blob);
      return;
    }

    fetch(apiUrl("/api/cancel-active-job"), {
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
  if (!requireBackendConnection()) {
    return;
  }

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
  setProgress(0, ["Preparing upload..."]);
  setStatus("Uploading selected images...", "");

  try {
    const fileCount = selectedFiles.length;
    const { status, payload } = await sendFormDataWithProgress(apiUrl("/api/run-prediction"), formData, (percent, loadedBytes, totalBytes) => {
      setProgress(percent, [
        `Uploading ${fileCount} file(s)...`,
        `${percent}% complete (${loadedBytes}/${totalBytes} bytes)`,
      ]);
      setStatus(`Uploading selected images... ${percent}%`, "");
    });

    setProgress(100, ["Upload complete. Waiting for prediction response..."]);

    if (status < 200 || status >= 300 || !payload.ok) {
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

    setStatus("Upload complete. Running prediction pipeline...", "");

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
loadBackendLogsButton.addEventListener("click", loadBackendLogs);
loginForm.addEventListener("submit", handleLoginSubmit);
retryBackendButton.addEventListener("click", checkBackendConnection);

window.addEventListener("beforeunload", () => {
  if (!backendReady) {
    return;
  }
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
setLoginEnabled(false);
setLoginStatus("Checking backend connection...", "");
setBackendConnectionState("checking", "Backend: Checking...");
updateAccessGate();
checkBackendConnection();
