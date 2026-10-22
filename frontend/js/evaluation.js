/**
 * evaluation.js — EmoSense AI Analytics & Insights Controller
 * ============================================================
 * Manages all 3 analytical scopes:
 *  - Scope A: Verified Model Evaluation Benchmarks (Held-out Test Dataset)
 *  - Scope B & C: Prediction Analytics (Stored History & Current Session)
 *  - Scope D: Batch Processing Analytics (Latest CSV Run)
 *
 * Adheres strictly to data integrity:
 *  • Zero fabricated metrics or mock placeholder values.
 *  • Never calculates accuracy/F1 on unlabeled user predictions.
 *  • Distinguishes between zero data, empty state, and API failure.
 */

"use strict";

import { API_BASE, EMOTION_CONFIG, HISTORY_KEY, SESSION_KEY, BATCH_KEY } from "./config.js";
import { checkApiStatus } from "./api.js";

// ─── DOM References ───────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// Header & Status
const apiStatus    = $("apiStatus");
const statusLabel  = $("statusLabel");
const themeToggle  = $("themeToggle");
const themeIcon    = $("themeIcon");

// Scope Tabs
const tabBenchmark   = $("tabBenchmark");
const tabPredictions = $("tabPredictions");
const tabBatch       = $("tabBatch");
const panelBenchmark   = $("panelBenchmark");
const panelPredictions = $("panelPredictions");
const panelBatch       = $("panelBatch");

// Benchmark elements
const benchmarkBannerText = $("benchmarkBannerText");
const kpiEnsAcc           = $("kpi-ens-acc");
const kpiEnsF1            = $("kpi-ens-f1");
const kpiSampleCount      = $("kpi-sample-count");
const kpiEnsLat           = $("kpi-ens-lat");
const classMetricsBody    = $("classMetricsBody");
const confusionMatrixGrid = $("confusionMatrixGrid");
const highestPerformerLabel = $("highestPerformerLabel");
const lowestPerformerLabel  = $("lowestPerformerLabel");

// Prediction Analytics filters & elements
const filterScope               = $("filterScope");
const filterDate                = $("filterDate");
const filterEmotion             = $("filterEmotion");
const btnRefreshPredictions     = $("btnRefreshPredictions");
const btnClearHistoryAnalytics  = $("btnClearHistoryAnalytics");
const kpiPredTotal              = $("kpi-pred-total");
const kpiPredTop                = $("kpi-pred-top");
const kpiPredConf               = $("kpi-pred-conf");
const kpiPredLat                = $("kpi-pred-lat");
const bandVeryHigh              = $("band-very-high");
const bandHigh                  = $("band-high");
const bandModerate              = $("band-moderate");
const bandLow                   = $("band-low");
const predictionsTableBody      = $("predictionsTableBody");
const predictionsTableWrapper   = $("predictionsTableWrapper");
const predictionsEmptyState     = $("predictionsEmptyState");
const donutEmptyState           = $("donutEmptyState");
const trendsEmptyState          = $("trendsEmptyState");

// Batch Analytics elements
const batchContent             = $("batchContent");
const batchEmptyState          = $("batchEmptyState");
const kpiBatchTotal            = $("kpi-batch-total");
const kpiBatchSuccess          = $("kpi-batch-success");
const kpiBatchTime             = $("kpi-batch-time");
const kpiBatchTop              = $("kpi-batch-top");
const batchDetailFilename      = $("batchDetailFilename");
const batchDetailTimestamp     = $("batchDetailTimestamp");
const batchDetailAlpha         = $("batchDetailAlpha");
const batchDetailThroughput    = $("batchDetailThroughput");
const batchTableBody           = $("batchTableBody");
const btnDownloadBatchResults  = $("btnDownloadBatchResults");

// ─── Chart Instances ──────────────────────────────────────────────────
let comparisonChartInstance = null;
let predictionDonutInstance = null;
let predictionTrendsInstance = null;
let batchDonutInstance = null;

// ════════════════════════════════════════════════════════════════════════
// THEME MANAGEMENT
// ════════════════════════════════════════════════════════════════════════
function initTheme() {
  const saved = localStorage.getItem("emosense_theme");
  const isDark = saved !== "light";
  document.body.classList.toggle("dark", isDark);
  document.body.classList.toggle("light", !isDark);
  if (themeIcon) themeIcon.textContent = isDark ? "☀️" : "🌙";

  themeToggle?.addEventListener("click", () => {
    const nextDark = !document.body.classList.contains("dark");
    document.body.classList.toggle("dark", nextDark);
    document.body.classList.toggle("light", !nextDark);
    if (themeIcon) themeIcon.textContent = nextDark ? "☀️" : "🌙";
    localStorage.setItem("emosense_theme", nextDark ? "dark" : "light");
    // Update chart colors on theme switch
    refreshAllCharts();
  });
}

// ════════════════════════════════════════════════════════════════════════
// TAB NAVIGATION
// ════════════════════════════════════════════════════════════════════════
function initTabs() {
  const tabs = [
    { btn: tabBenchmark, panel: panelBenchmark, name: "benchmark" },
    { btn: tabPredictions, panel: panelPredictions, name: "predictions" },
    { btn: tabBatch, panel: panelBatch, name: "batch" },
  ];

  tabs.forEach(({ btn, panel, name }) => {
    btn?.addEventListener("click", () => {
      tabs.forEach((t) => {
        t.btn?.classList.remove("active");
        t.panel?.classList.remove("active");
      });
      btn.classList.add("active");
      panel.classList.add("active");

      // Trigger specific scope render when switched
      if (name === "predictions") {
        renderPredictionAnalytics();
      } else if (name === "batch") {
        renderBatchAnalytics();
      } else if (name === "benchmark" && comparisonChartInstance) {
        comparisonChartInstance.resize();
      }
    });
  });

  // Check URL hash (e.g. #predictions or #batch)
  const hash = window.location.hash.toLowerCase();
  if (hash === "#predictions") {
    tabPredictions?.click();
  } else if (hash === "#batch") {
    tabBatch?.click();
  }
}

// ════════════════════════════════════════════════════════════════════════
// SCOPE A: MODEL EVALUATION (HELD-OUT BENCHMARKS)
// ════════════════════════════════════════════════════════════════════════
async function loadBenchmarkMetrics() {
  let data = null;

  try {
    // 1. Try dedicated API endpoint
    const res = await fetch(`${API_BASE}/analytics/evaluation`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      data = await res.json();
    }
  } catch (_) {
    // Fallback to static metrics.json
  }

  if (!data) {
    try {
      const res = await fetch("./metrics.json", { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        data = await res.json();
      }
    } catch (err) {
      console.warn("Could not load metrics.json:", err);
    }
  }

  if (!data) {
    benchmarkBannerText.innerHTML = `<strong>⚠️ Evaluation Offline:</strong> Unable to load empirical validation benchmarks. Please ensure the backend is running.`;
    return;
  }

  renderBenchmarkDashboard(data);
}

function renderBenchmarkDashboard(data) {
  const { summary, branches, class_metrics, confusion_matrix } = data;

  // Banner
  if (benchmarkBannerText) {
    benchmarkBannerText.innerHTML = `
      <strong>Verified Empirical Benchmark:</strong> Evaluated on ${summary.num_samples} stratified test cases (${summary.device} · ${summary.timestamp}) from <code>clean_emotion_dataset.csv</code>.
    `;
  }

  // KPIs
  if (kpiEnsAcc && branches.ensemble) kpiEnsAcc.textContent = `${branches.ensemble.accuracy}%`;
  if (kpiEnsF1 && branches.ensemble) kpiEnsF1.textContent = `${branches.ensemble.macro_f1}%`;
  if (kpiSampleCount && summary) kpiSampleCount.textContent = summary.num_samples;
  if (kpiEnsLat && branches.ensemble) kpiEnsLat.textContent = `~${branches.ensemble.avg_latency_ms} ms`;

  // Branch Comparison Table
  if (branches.xgb) {
    $("xgb-acc").textContent = `${branches.xgb.accuracy}%`;
    $("xgb-f1").textContent = `${branches.xgb.macro_f1}%`;
    $("xgb-lat").textContent = `~${branches.xgb.avg_latency_ms} ms`;
  }
  if (branches.roberta) {
    $("rob-acc").textContent = `${branches.roberta.accuracy}%`;
    $("rob-f1").textContent = `${branches.roberta.macro_f1}%`;
    $("rob-lat").textContent = `~${branches.roberta.avg_latency_ms} ms`;
  }
  if (branches.ensemble) {
    $("ens-acc").textContent = `${branches.ensemble.accuracy}%`;
    $("ens-f1").textContent = `${branches.ensemble.macro_f1}%`;
    $("ens-lat").textContent = `~${branches.ensemble.avg_latency_ms} ms`;
  }

  // Dual-Axis Comparison Chart
  drawComparisonChart(branches);

  // Class-Specific Metrics Table
  if (classMetricsBody && class_metrics) {
    classMetricsBody.innerHTML = "";
    let highestF1 = -1, highestCls = "";
    let lowestF1 = 999, lowestCls = "";

    Object.entries(class_metrics).forEach(([cls, stats]) => {
      const cfg = EMOTION_CONFIG[cls] ?? EMOTION_CONFIG.neutral;
      if (stats.f1 > highestF1) { highestF1 = stats.f1; highestCls = cls; }
      if (stats.f1 < lowestF1) { lowestF1 = stats.f1; lowestCls = cls; }

      const row = document.createElement("tr");
      row.innerHTML = `
        <td><span style="margin-right: 6px;">${cfg.emoji}</span><strong>${cfg.label}</strong></td>
        <td>${stats.precision}%</td>
        <td>${stats.recall}%</td>
        <td><strong style="color: ${cfg.color};">${stats.f1}%</strong></td>
        <td>${stats.support}</td>
      `;
      classMetricsBody.appendChild(row);
    });

    const highCfg = EMOTION_CONFIG[highestCls] ?? EMOTION_CONFIG.neutral;
    const lowCfg = EMOTION_CONFIG[lowestCls] ?? EMOTION_CONFIG.neutral;
    if (highestPerformerLabel) highestPerformerLabel.innerHTML = `${highCfg.emoji} <strong>${highCfg.label}</strong> (${highestF1}% F1)`;
    if (lowestPerformerLabel) lowestPerformerLabel.innerHTML = `${lowCfg.emoji} <strong>${lowCfg.label}</strong> (${lowestF1}% F1)`;
  }

  // Dynamic Confusion Matrix Heatmap
  if (confusionMatrixGrid && confusion_matrix) {
    renderConfusionMatrixHeatmap(confusion_matrix);
  }
}

function drawComparisonChart(branches) {
  const canvas = $("comparisonChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  if (comparisonChartInstance) {
    comparisonChartInstance.destroy();
  }

  comparisonChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["XGBoost (Semantic)", "RoBERTa (Contextual)", "Hybrid Ensemble"],
      datasets: [
        {
          label: "Accuracy (%)",
          data: [
            branches.xgb?.accuracy ?? 0,
            branches.roberta?.accuracy ?? 0,
            branches.ensemble?.accuracy ?? 0,
          ],
          backgroundColor: [
            "rgba(245, 158, 11, 0.4)",
            "rgba(59, 130, 246, 0.4)",
            "rgba(139, 92, 246, 0.65)"
          ],
          borderColor: [
            "rgba(245, 158, 11, 0.8)",
            "rgba(59, 130, 246, 0.8)",
            "rgba(139, 92, 246, 1)"
          ],
          borderWidth: 2,
          yAxisID: "yAccuracy",
          order: 2,
        },
        {
          label: "Latency (ms)",
          data: [
            branches.xgb?.avg_latency_ms ?? 0,
            branches.roberta?.avg_latency_ms ?? 0,
            branches.ensemble?.avg_latency_ms ?? 0,
          ],
          type: "line",
          borderColor: "#ef4444",
          backgroundColor: "#ef4444",
          borderWidth: 3,
          pointRadius: 5,
          fill: false,
          yAxisID: "yLatency",
          order: 1,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: { color: getTextColor(0.8), font: { family: "Inter", size: 12 } }
        },
        tooltip: {
          callbacks: {
            label(ctx) {
              if (ctx.dataset.label.includes("Accuracy")) {
                return ` Accuracy: ${ctx.raw}%`;
              }
              return ` Latency: ${ctx.raw} ms`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: getTextColor(0.7), font: { family: "Outfit", size: 11 } }
        },
        yAccuracy: {
          type: "linear",
          position: "left",
          min: 60,
          max: 100,
          title: { display: true, text: "Accuracy (%)", color: getTextColor(0.7), font: { family: "Outfit" } },
          ticks: { color: getTextColor(0.6) },
          grid: { color: getGridColor() }
        },
        yLatency: {
          type: "linear",
          position: "right",
          min: 0,
          max: 320,
          title: { display: true, text: "Latency (ms)", color: "#ef4444", font: { family: "Outfit" } },
          ticks: { color: "#ef4444" },
          grid: { display: false }
        }
      }
    }
  });
}

function renderConfusionMatrixHeatmap(cmData) {
  confusionMatrixGrid.innerHTML = "";
  const { labels, matrix } = cmData;

  // Header row: top-left corner is empty
  const corner = document.createElement("div");
  corner.className = "matrix-cell matrix-header-cell";
  corner.textContent = "Act \\ Pred";
  confusionMatrixGrid.appendChild(corner);

  labels.forEach((lbl) => {
    const colHeader = document.createElement("div");
    colHeader.className = "matrix-cell matrix-header-cell";
    const cfg = EMOTION_CONFIG[lbl] ?? EMOTION_CONFIG.neutral;
    colHeader.textContent = lbl.substring(0, 3).toUpperCase();
    colHeader.title = `Predicted: ${cfg.label}`;
    confusionMatrixGrid.appendChild(colHeader);
  });

  // Data rows
  labels.forEach((actLbl, rIdx) => {
    // Row header (actual label)
    const rowHeader = document.createElement("div");
    rowHeader.className = "matrix-cell matrix-header-cell y-label";
    const actCfg = EMOTION_CONFIG[actLbl] ?? EMOTION_CONFIG.neutral;
    rowHeader.textContent = actLbl.substring(0, 3).toUpperCase();
    rowHeader.title = `Actual: ${actCfg.label}`;
    confusionMatrixGrid.appendChild(rowHeader);

    // Matrix cells for this row
    labels.forEach((predLbl, cIdx) => {
      const val = matrix[rIdx] ? matrix[rIdx][cIdx] : 0.0;
      const cell = document.createElement("div");
      cell.className = "matrix-cell";
      cell.textContent = `${val}%`;
      cell.title = `Actual ${actCfg.label} predicted as ${predLbl}: ${val}%`;

      const opacity = Math.min(val / 100, 1);
      if (rIdx === cIdx) {
        // Diagonal: Correct prediction
        const hue = actCfg.hue;
        const sat = actLbl === "neutral" ? "20%" : "80%";
        cell.style.background = `hsla(${hue}, ${sat}, 50%, ${Math.max(opacity * 0.9, 0.2)})`;
        cell.style.color = opacity > 0.4 ? (actLbl === "joy" ? "#000" : "#fff") : "var(--text)";
      } else {
        // Off-diagonal: Misclassification
        if (val > 0.0) {
          cell.style.background = `rgba(239, 68, 68, ${opacity * 0.75})`;
          cell.style.color = "var(--text-2)";
        } else {
          cell.style.background = "var(--surface-2)";
          cell.style.color = "var(--text-3)";
        }
      }

      confusionMatrixGrid.appendChild(cell);
    });
  });
}

// ════════════════════════════════════════════════════════════════════════
// SCOPE B & C: PREDICTION ANALYTICS (STORED & SESSION)
// ════════════════════════════════════════════════════════════════════════
function getStoredPredictions() {
  const scope = filterScope ? filterScope.value : "saved";
  try {
    if (scope === "session") {
      return JSON.parse(sessionStorage.getItem(SESSION_KEY) || "[]");
    }
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch (_) {
    return [];
  }
}

function renderPredictionAnalytics() {
  const rawRecords = getStoredPredictions();

  // Apply filters
  const dateFilter = filterDate ? filterDate.value : "all";
  const emotionFilter = filterEmotion ? filterEmotion.value : "all";

  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;

  const filtered = rawRecords.filter((rec) => {
    // Emotion filter
    if (emotionFilter !== "all" && rec.predicted_emotion?.toLowerCase() !== emotionFilter.toLowerCase()) {
      return false;
    }

    // Date filter
    if (dateFilter !== "all" && rec.timestamp) {
      const recTime = new Date(rec.timestamp).getTime();
      if (dateFilter === "today" && now - recTime > dayMs) return false;
      if (dateFilter === "week" && now - recTime > 7 * dayMs) return false;
    }

    return true;
  });

  const total = filtered.length;

  // 1. Overview KPIs
  if (kpiPredTotal) kpiPredTotal.textContent = total;

  if (total === 0) {
    if (kpiPredTop) kpiPredTop.textContent = "—";
    if (kpiPredConf) kpiPredConf.textContent = "—";
    if (kpiPredLat) kpiPredLat.textContent = "—";
    if (bandVeryHigh) bandVeryHigh.textContent = "0";
    if (bandHigh) bandHigh.textContent = "0";
    if (bandModerate) bandModerate.textContent = "0";
    if (bandLow) bandLow.textContent = "0";

    // Show empty states
    if (donutEmptyState) donutEmptyState.style.display = "flex";
    if ($("predictionEmotionChart")) $("predictionEmotionChart").style.display = "none";
    if (trendsEmptyState) trendsEmptyState.style.display = "flex";
    if ($("predictionTrendsChart")) $("predictionTrendsChart").style.display = "none";
    if (predictionsEmptyState) predictionsEmptyState.style.display = "flex";
    if (predictionsTableWrapper) predictionsTableWrapper.style.display = "none";

    if (predictionDonutInstance) { predictionDonutInstance.destroy(); predictionDonutInstance = null; }
    if (predictionTrendsInstance) { predictionTrendsInstance.destroy(); predictionTrendsInstance = null; }
    return;
  }

  // 2. Counts and distribution
  const counts = { joy: 0, sadness: 0, anger: 0, fear: 0, surprise: 0, neutral: 0 };
  let totalConfidence = 0;
  let totalLatency = 0;
  let latencyCount = 0;
  const bands = { veryHigh: 0, high: 0, moderate: 0, low: 0 };

  filtered.forEach((r) => {
    const e = r.predicted_emotion?.toLowerCase();
    if (counts[e] !== undefined) counts[e]++;

    const conf = parseFloat(r.confidence) || 0;
    totalConfidence += conf;

    if (conf >= 0.90) bands.veryHigh++;
    else if (conf >= 0.80) bands.high++;
    else if (conf >= 0.60) bands.moderate++;
    else bands.low++;

    if (r.processing_time_ms !== undefined && r.processing_time_ms !== null) {
      totalLatency += parseFloat(r.processing_time_ms);
      latencyCount++;
    }
  });

  // Top emotion
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const [topEmo, topCount] = sorted[0];
  const topPct = ((topCount / total) * 100).toFixed(1);
  const topCfg = EMOTION_CONFIG[topEmo] ?? EMOTION_CONFIG.neutral;
  if (kpiPredTop) kpiPredTop.innerHTML = `${topCfg.emoji} ${topCfg.label} <span style="font-size: 1rem; color: var(--text-2); font-weight: 500;">(${topPct}%)</span>`;

  // Avg Confidence
  const avgConf = ((totalConfidence / total) * 100).toFixed(1);
  if (kpiPredConf) kpiPredConf.textContent = `${avgConf}%`;

  // Avg Latency
  if (kpiPredLat) {
    if (latencyCount > 0) {
      kpiPredLat.textContent = `${(totalLatency / latencyCount).toFixed(1)} ms`;
    } else {
      kpiPredLat.textContent = "—";
    }
  }

  // Confidence Bands
  if (bandVeryHigh) bandVeryHigh.textContent = bands.veryHigh;
  if (bandHigh) bandHigh.textContent = bands.high;
  if (bandModerate) bandModerate.textContent = bands.moderate;
  if (bandLow) bandLow.textContent = bands.low;

  // 3. Emotion Distribution Chart (Donut)
  if (donutEmptyState) donutEmptyState.style.display = "none";
  if ($("predictionEmotionChart")) $("predictionEmotionChart").style.display = "block";
  drawPredictionDonut(counts, total);

  // 4. Trends Chart (Daily Aggregation)
  drawPredictionTrends(filtered);

  // 5. Prediction Records Table
  if (predictionsEmptyState) predictionsEmptyState.style.display = "none";
  if (predictionsTableWrapper) predictionsTableWrapper.style.display = "block";
  renderPredictionsTable(filtered);
}

function drawPredictionDonut(counts, total) {
  const canvas = $("predictionEmotionChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  if (predictionDonutInstance) {
    predictionDonutInstance.destroy();
  }

  const labels = Object.keys(counts).map((k) => EMOTION_CONFIG[k]?.label || k);
  const data = Object.values(counts);
  const bgColors = Object.keys(counts).map((k) => EMOTION_CONFIG[k]?.color || "#999");

  predictionDonutInstance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        {
          data,
          backgroundColor: bgColors,
          borderWidth: 2,
          borderColor: "var(--surface-1)",
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "right",
          labels: { color: getTextColor(0.8), font: { family: "Outfit", size: 12 } }
        },
        tooltip: {
          callbacks: {
            label(ctx) {
              const val = ctx.raw;
              const pct = ((val / total) * 100).toFixed(1);
              return ` ${ctx.label}: ${val} (${pct}%)`;
            }
          }
        }
      }
    }
  });
}

function drawPredictionTrends(records) {
  const canvas = $("predictionTrendsChart");
  if (!canvas) return;

  // Aggregate by date (YYYY-MM-DD)
  const dateMap = {};
  records.forEach((r) => {
    if (r.timestamp) {
      const dateKey = r.timestamp.substring(0, 10);
      dateMap[dateKey] = (dateMap[dateKey] || 0) + 1;
    }
  });

  const dates = Object.keys(dateMap).sort();

  if (dates.length < 1) {
    if (trendsEmptyState) trendsEmptyState.style.display = "flex";
    canvas.style.display = "none";
    if (predictionTrendsInstance) { predictionTrendsInstance.destroy(); predictionTrendsInstance = null; }
    return;
  }

  if (trendsEmptyState) trendsEmptyState.style.display = "none";
  canvas.style.display = "block";

  const ctx = canvas.getContext("2d");
  if (predictionTrendsInstance) {
    predictionTrendsInstance.destroy();
  }

  const counts = dates.map((d) => dateMap[d]);

  predictionTrendsInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: dates,
      datasets: [
        {
          label: "Predictions Recorded",
          data: counts,
          backgroundColor: "rgba(139, 92, 246, 0.5)",
          borderColor: "rgba(139, 92, 246, 1)",
          borderWidth: 2,
          borderRadius: 4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label(ctx) {
              return ` ${ctx.raw} prediction${ctx.raw === 1 ? "" : "s"}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: getTextColor(0.7), font: { family: "Inter", size: 11 } }
        },
        y: {
          beginAtZero: true,
          ticks: { stepSize: 1, color: getTextColor(0.6) },
          grid: { color: getGridColor() }
        }
      }
    }
  });
}

function renderPredictionsTable(records) {
  if (!predictionsTableBody) return;
  predictionsTableBody.innerHTML = "";

  records.slice(0, 50).forEach((item) => {
    const cfg = EMOTION_CONFIG[item.predicted_emotion] ?? EMOTION_CONFIG.neutral;
    const timeStr = item.timestamp ? formatTimestamp(item.timestamp) : "—";
    const confStr = item.confidence_pct || (item.confidence ? `${(item.confidence * 100).toFixed(1)}%` : "—");
    const latStr = item.processing_time_ms ? `${item.processing_time_ms} ms` : "—";
    const textSnippet = escapeHtml(item.text || "—");

    const badgeBg = cfg.color.replace(")", ", 0.15)").replace("hsl(", "hsla(");

    const row = document.createElement("tr");
    row.innerHTML = `
      <td><span style="font-size: 0.8rem; color: var(--text-2);">${timeStr}</span></td>
      <td>
        <span class="badge-tag" style="background: ${badgeBg}; color: ${cfg.color};">
          ${cfg.emoji} ${cfg.label}
        </span>
      </td>
      <td><strong>${confStr}</strong></td>
      <td>${latStr}</td>
      <td title="${textSnippet}">${textSnippet.substring(0, 65)}${textSnippet.length > 65 ? "…" : ""}</td>
    `;
    predictionsTableBody.appendChild(row);
  });
}

// ════════════════════════════════════════════════════════════════════════
// SCOPE D: BATCH PROCESSING ANALYTICS
// ════════════════════════════════════════════════════════════════════════
function renderBatchAnalytics() {
  let batchData = null;
  try {
    batchData = JSON.parse(sessionStorage.getItem(BATCH_KEY));
  } catch (_) {}

  if (!batchData || !batchData.predictions || batchData.predictions.length === 0) {
    if (batchContent) batchContent.style.display = "none";
    if (batchEmptyState) batchEmptyState.style.display = "flex";
    return;
  }

  if (batchContent) batchContent.style.display = "block";
  if (batchEmptyState) batchEmptyState.style.display = "none";

  const total = batchData.total_rows || batchData.predictions.length;
  const timeMs = batchData.batch_processing_time_ms || 0;

  if (kpiBatchTotal) kpiBatchTotal.textContent = total;
  if ($("kpi-batch-file")) $("kpi-batch-file").textContent = batchData.filename || "Uploaded CSV";
  if (kpiBatchSuccess) kpiBatchSuccess.textContent = batchData.predictions.length;
  if (kpiBatchTime) kpiBatchTime.textContent = `${timeMs} ms`;

  // Details
  if (batchDetailFilename) batchDetailFilename.textContent = batchData.filename || "batch.csv";
  if (batchDetailTimestamp) batchDetailTimestamp.textContent = batchData.timestamp ? formatTimestamp(batchData.timestamp) : "Recent";
  if (batchDetailAlpha) batchDetailAlpha.textContent = batchData.alpha ? parseFloat(batchData.alpha).toFixed(2) : "0.70";

  const throughput = timeMs > 0 ? ((total / (timeMs / 1000))).toFixed(1) : "—";
  if (batchDetailThroughput) batchDetailThroughput.textContent = `${throughput} rows/sec`;

  // Batch Emotion Distribution
  const counts = { joy: 0, sadness: 0, anger: 0, fear: 0, surprise: 0, neutral: 0 };
  batchData.predictions.forEach((p) => {
    const e = p.predicted_emotion?.toLowerCase();
    if (counts[e] !== undefined) counts[e]++;
  });

  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const [topEmo, topCount] = sorted[0];
  const topCfg = EMOTION_CONFIG[topEmo] ?? EMOTION_CONFIG.neutral;
  if (kpiBatchTop) kpiBatchTop.innerHTML = `${topCfg.emoji} ${topCfg.label}`;
  if ($("kpi-batch-top-pct")) $("kpi-batch-top-pct").textContent = `${((topCount / total) * 100).toFixed(1)}% of batch rows`;

  drawBatchDonut(counts, total);

  // Batch Table Body
  if (batchTableBody) {
    batchTableBody.innerHTML = "";
    batchData.predictions.slice(0, 100).forEach((pred) => {
      const cfg = EMOTION_CONFIG[pred.predicted_emotion] ?? EMOTION_CONFIG.neutral;
      const badgeBg = cfg.color.replace(")", ", 0.15)").replace("hsl(", "hsla(");
      const text = escapeHtml(pred.text || "");

      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${pred.row}</td>
        <td title="${text}">${text.substring(0, 70)}${text.length > 70 ? "…" : ""}</td>
        <td>
          <span class="badge-tag" style="background: ${badgeBg}; color: ${cfg.color};">
            ${cfg.emoji} ${cfg.label}
          </span>
        </td>
        <td><strong>${pred.confidence_pct || "—"}</strong></td>
      `;
      batchTableBody.appendChild(row);
    });
  }

  // Hook up download button
  if (btnDownloadBatchResults) {
    btnDownloadBatchResults.onclick = () => downloadBatchCSV(batchData);
  }
}

function drawBatchDonut(counts, total) {
  const canvas = $("batchEmotionChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  if (batchDonutInstance) {
    batchDonutInstance.destroy();
  }

  const labels = Object.keys(counts).map((k) => EMOTION_CONFIG[k]?.label || k);
  const data = Object.values(counts);
  const bgColors = Object.keys(counts).map((k) => EMOTION_CONFIG[k]?.color || "#999");

  batchDonutInstance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        {
          data,
          backgroundColor: bgColors,
          borderWidth: 2,
          borderColor: "var(--surface-1)",
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "right",
          labels: { color: getTextColor(0.8), font: { family: "Outfit", size: 12 } }
        },
        tooltip: {
          callbacks: {
            label(ctx) {
              const val = ctx.raw;
              const pct = ((val / total) * 100).toFixed(1);
              return ` ${ctx.label}: ${val} (${pct}%)`;
            }
          }
        }
      }
    }
  });
}

function downloadBatchCSV(batchData) {
  if (!batchData || !batchData.predictions) return;

  const header = "Row,Text,Predicted_Emotion,Confidence,Processing_Time_ms\n";
  const rows = batchData.predictions.map((p) => {
    const text = `"${(p.text || "").replace(/"/g, '""')}"`;
    return `${p.row},${text},${p.predicted_emotion},${p.confidence_pct || "0%"},${p.processing_time_ms || 0}`;
  });

  const csv = header + rows.join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = `emosense_batch_results_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ════════════════════════════════════════════════════════════════════════
// HELPERS
// ════════════════════════════════════════════════════════════════════════
function getTextColor(opacity = 1) {
  const isDark = document.body.classList.contains("dark");
  return isDark ? `rgba(255, 255, 255, ${opacity})` : `rgba(15, 23, 42, ${opacity})`;
}

function getGridColor() {
  const isDark = document.body.classList.contains("dark");
  return isDark ? "rgba(255, 255, 255, 0.06)" : "rgba(0, 0, 0, 0.06)";
}

function refreshAllCharts() {
  if (comparisonChartInstance) comparisonChartInstance.update();
  if (predictionDonutInstance) predictionDonutInstance.update();
  if (predictionTrendsInstance) predictionTrendsInstance.update();
  if (batchDonutInstance) batchDonutInstance.update();
}

function formatTimestamp(iso) {
  try {
    const d = new Date(iso);
    return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  } catch (_) {
    return iso;
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ════════════════════════════════════════════════════════════════════════
// EVENT LISTENERS & INITIALIZATION
// ════════════════════════════════════════════════════════════════════════
function initEventListeners() {
  // Filters
  filterScope?.addEventListener("change", renderPredictionAnalytics);
  filterDate?.addEventListener("change", renderPredictionAnalytics);
  filterEmotion?.addEventListener("change", renderPredictionAnalytics);
  btnRefreshPredictions?.addEventListener("click", renderPredictionAnalytics);

  // Clear History
  btnClearHistoryAnalytics?.addEventListener("click", () => {
    if (confirm("Are you sure you want to clear your saved prediction analytics history?")) {
      localStorage.removeItem(HISTORY_KEY);
      sessionStorage.removeItem(SESSION_KEY);
      renderPredictionAnalytics();
    }
  });
}

// Startup
initTheme();
initTabs();
initEventListeners();

// Initial API Status
checkApiStatus(apiStatus, statusLabel);
setInterval(() => checkApiStatus(apiStatus, statusLabel), 30_000);

// Load Benchmark Data initially
loadBenchmarkMetrics();
