/**
 * batch.js — CSV batch upload, processing, and results rendering.
 *
 * Handles drag-and-drop file selection, batch API calls, progress
 * simulation, results table rendering, and CSV download.
 */

import { EMOTION_CONFIG, BATCH_KEY } from "./config.js";
import { showError, hideError, escapeHtml, alphaSlider } from "./ui.js";
import { fetchBatchPrediction } from "./api.js";

// ─── DOM references ───────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const dropZone           = $("dropZone");
const csvFileInput       = $("csvFileInput");
const batchFileInfo      = $("batchFileInfo");
const batchFileName      = $("batchFileName");
const batchFileRows      = $("batchFileRows");
const batchClearFile     = $("batchClearFile");
const batchRunBtn        = $("batchRunBtn");
const batchBtnText       = $("batchBtnText");
const batchBtnLoader     = $("batchBtnLoader");
const batchProgress      = $("batchProgress");
const batchProgressLabel = $("batchProgressLabel");
const batchProgressPct   = $("batchProgressPct");
const batchProgressFill  = $("batchProgressFill");
const batchResults       = $("batchResults");
const batchResultsCount  = $("batchResultsCount");
const batchTime          = $("batchTime");
const batchDownloadBtn   = $("batchDownloadBtn");
const batchTableBody     = $("batchTableBody");

// ─── State ────────────────────────────────────────────────────────────
let uploadedFile     = null;
let batchResultsData = null;

/**
 * Initialise all batch-related event listeners.
 * Called once from app.js at startup.
 */
export function initBatch() {
  if (!dropZone) return;

  // Click to open file picker
  dropZone.addEventListener("click", () => csvFileInput.click());
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") csvFileInput.click();
  });

  // Drag and drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  });

  // File input change
  csvFileInput.addEventListener("change", () => {
    const file = csvFileInput.files[0];
    if (file) handleFileSelect(file);
  });

  // Clear file
  batchClearFile.addEventListener("click", () => {
    uploadedFile = null;
    csvFileInput.value = "";
    batchFileInfo.hidden = true;
    dropZone.hidden = false;
    batchRunBtn.disabled = true;
    batchProgress.hidden = true;
    batchResults.hidden = true;
  });

  // Run batch
  batchRunBtn.addEventListener("click", runBatchPrediction);

  // Download CSV
  batchDownloadBtn.addEventListener("click", downloadBatchCSV);
}

// ════════════════════════════════════════════════════════════════════════
// FILE SELECTION
// ════════════════════════════════════════════════════════════════════════
function handleFileSelect(file) {
  if (!file.name.endsWith(".csv")) {
    showError("Please upload a .csv file.");
    return;
  }

  uploadedFile = file;

  // Count rows
  const reader = new FileReader();
  reader.onload = (e) => {
    const lines = e.target.result.split("\n").filter(l => l.trim());
    const rowCount = Math.max(0, lines.length - 1);  // subtract header

    batchFileName.textContent = file.name;
    batchFileRows.textContent = `${rowCount} rows`;
    batchFileInfo.hidden = false;
    dropZone.hidden = true;
    batchRunBtn.disabled = false;
    batchProgress.hidden = true;
    batchResults.hidden = true;
  };
  reader.readAsText(file);
}

// ════════════════════════════════════════════════════════════════════════
// RUN BATCH
// ════════════════════════════════════════════════════════════════════════
async function runBatchPrediction() {
  if (!uploadedFile) return;

  hideError();
  batchRunBtn.disabled = true;
  batchBtnText.textContent = "Processing...";
  batchBtnLoader.style.display = "block";
  batchProgress.hidden = false;
  batchResults.hidden = true;
  batchProgressFill.style.width = "0%";
  batchProgressPct.textContent = "0%";
  batchProgressLabel.textContent = "Uploading & processing...";

  // Simulate progress (since it's a single request)
  let fakeProgress = 0;
  const progressInterval = setInterval(() => {
    fakeProgress = Math.min(fakeProgress + Math.random() * 8, 92);
    batchProgressFill.style.width = `${fakeProgress}%`;
    batchProgressPct.textContent = `${Math.round(fakeProgress)}%`;
  }, 300);

  try {
    const alpha = alphaSlider ? parseFloat(alphaSlider.value) : 0.7;
    const data = await fetchBatchPrediction(uploadedFile, alpha);
    batchResultsData = data;

    clearInterval(progressInterval);

    // Complete progress
    batchProgressFill.style.width = "100%";
    batchProgressPct.textContent = "100%";
    batchProgressLabel.textContent = "Complete!";

    // Render results
    renderBatchResults(data);

  } catch (err) {
    clearInterval(progressInterval);
    batchProgressFill.style.width = "0%";
    batchProgress.hidden = true;

    if (err.name === "TimeoutError") {
      showError("Batch request timed out. Try fewer rows or reduce file size.");
    } else {
      showError(err.message || "Batch prediction failed.");
    }
  } finally {
    batchRunBtn.disabled = false;
    batchBtnText.textContent = "Run Batch Prediction";
    batchBtnLoader.style.display = "";
  }
}

// ════════════════════════════════════════════════════════════════════════
// RENDER RESULTS
// ════════════════════════════════════════════════════════════════════════
function renderBatchResults(data) {
  batchResults.hidden = false;
  batchResultsCount.textContent = `${data.total_rows} rows`;
  batchTime.textContent = `⏱ ${data.batch_processing_time_ms} ms`;

  // Persist latest batch run in sessionStorage for Insight Dashboard Batch Analytics
  try {
    sessionStorage.setItem(BATCH_KEY, JSON.stringify({
      filename: uploadedFile ? uploadedFile.name : "batch.csv",
      timestamp: new Date().toISOString(),
      ...data,
    }));
  } catch (_) {}

  batchTableBody.innerHTML = "";

  data.predictions.forEach((pred) => {
    const cfg = EMOTION_CONFIG[pred.predicted_emotion] ?? EMOTION_CONFIG.neutral;
    const row = document.createElement("tr");

    const badgeBg = cfg.color.replace(")", ", 0.15)").replace("hsl(", "hsla(");
    row.innerHTML = `
      <td>${pred.row}</td>
      <td title="${escapeHtml(pred.text || "")}">${escapeHtml((pred.text || "").substring(0, 80))}${(pred.text || "").length > 80 ? "…" : ""}</td>
      <td>
        <span class="batch-emotion-badge" style="background: ${badgeBg}; color: ${cfg.color};">
          ${cfg.emoji} ${pred.predicted_emotion}
        </span>
      </td>
      <td>${pred.confidence_pct || "—"}</td>
    `;
    batchTableBody.appendChild(row);
  });
}

// ════════════════════════════════════════════════════════════════════════
// DOWNLOAD CSV
// ════════════════════════════════════════════════════════════════════════
function downloadBatchCSV() {
  if (!batchResultsData) return;

  const header = "Row,Text,Predicted_Emotion,Confidence,Processing_Time_ms\n";
  const rows = batchResultsData.predictions.map((p) => {
    const text = `"${(p.text || "").replace(/"/g, '""')}"`;
    return `${p.row},${text},${p.predicted_emotion},${p.confidence_pct || "0%"},${p.processing_time_ms || 0}`;
  });

  const csv = header + rows.join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url  = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = `emosense_batch_results_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
