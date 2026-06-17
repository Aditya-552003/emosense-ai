/**
 * app.js — EmoSense AI Frontend Entry Point
 * ===========================================
 * Wires together all modules: config, api, ui, chart, probBars,
 * history, and batch.  Registers event listeners and initialises
 * the application on page load.
 *
 * Features:
 *  • Startup health-check with live API status indicator
 *  • Real-time character counter with colour warnings
 *  • Sample pill autofill
 *  • Full async predict flow with loading states
 *  • Chart.js donut chart (live updates)
 *  • Animated probability bars
 *  • Dynamic emotion-driven UI colour theming
 *  • localStorage prediction history (persistent)
 *  • Copy-to-clipboard with visual feedback
 *  • Clear input + clear history
 *  • Friendly error messages for all failure modes
 */

"use strict";

import { EMOTION_CONFIG } from "./config.js";
import { checkApiStatus, fetchPrediction } from "./api.js";
import {
  textInput, analyzeBtn, clearBtn, charCounter,
  emptyState, resultContent,
  emotionEmoji, emotionName, confidenceValue,
  donutPct, probBars, procTime,
  copyBtn, copyIcon,
  clearHistoryBtn,
  statusLabel, apiStatus,
  themeToggle, alphaSlider, alphaDisplay,
  applyTheme, toggleTheme, setActiveEmotion,
  updateCounter, showError, hideError, setLoading,
  isDark,
} from "./ui.js";
import { updateDonut } from "./chart.js";
import { renderProbBars } from "./probBars.js";
import { loadHistory, addToHistory, clearHistory } from "./history.js";
import { initBatch } from "./batch.js";

// ─── State ────────────────────────────────────────────────────────────
let lastResult = null;

// ════════════════════════════════════════════════════════════════════════
// THEME INIT
// ════════════════════════════════════════════════════════════════════════
applyTheme(isDark);
themeToggle.addEventListener("click", toggleTheme);

// ════════════════════════════════════════════════════════════════════════
// ALPHA SLIDER
// ════════════════════════════════════════════════════════════════════════
const savedAlpha = localStorage.getItem("emosense_alpha");
if (savedAlpha !== null && alphaSlider) {
  alphaSlider.value = savedAlpha;
}
if (alphaSlider) {
  alphaDisplay.textContent = parseFloat(alphaSlider.value).toFixed(2);
  alphaSlider.addEventListener("input", () => {
    alphaDisplay.textContent = parseFloat(alphaSlider.value).toFixed(2);
    localStorage.setItem("emosense_alpha", alphaSlider.value);
  });
}

// ════════════════════════════════════════════════════════════════════════
// API STATUS CHECK
// ════════════════════════════════════════════════════════════════════════
checkApiStatus(apiStatus, statusLabel);
setInterval(() => checkApiStatus(apiStatus, statusLabel), 30_000);

// ════════════════════════════════════════════════════════════════════════
// CHARACTER COUNTER
// ════════════════════════════════════════════════════════════════════════
textInput.addEventListener("input", updateCounter);
updateCounter();

// ════════════════════════════════════════════════════════════════════════
// SAMPLE PILLS
// ════════════════════════════════════════════════════════════════════════
document.querySelectorAll(".pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    textInput.value = pill.dataset.sample;
    updateCounter();
    textInput.focus();
    hideError();
  });
});

// ════════════════════════════════════════════════════════════════════════
// CLEAR BUTTON
// ════════════════════════════════════════════════════════════════════════
clearBtn.addEventListener("click", () => {
  textInput.value = "";
  updateCounter();
  hideError();
  textInput.focus();
});

// ════════════════════════════════════════════════════════════════════════
// ANALYSE BUTTON → POST /predict
// ════════════════════════════════════════════════════════════════════════
analyzeBtn.addEventListener("click", runPrediction);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runPrediction();
});

async function runPrediction() {
  const text = textInput.value.trim();
  if (!text) return;

  hideError();
  setLoading(true);

  try {
    const alpha = alphaSlider ? parseFloat(alphaSlider.value) : 0.7;
    const data  = await fetchPrediction(text, alpha);
    lastResult  = { text, ...data };

    displayResult(data);
    addToHistory({ text, ...data });

  } catch (err) {
    if (err.name === "TimeoutError") {
      showError("Request timed out. The backend may be overloaded — please try again.");
    } else if (err.name === "TypeError" || err.message.includes("fetch")) {
      showError("Cannot reach the backend. Make sure the server is running at http://127.0.0.1:8080");
    } else {
      showError(err.message || "An unexpected error occurred.");
    }
    // Revert API status indicator
    checkApiStatus(apiStatus, statusLabel);
  } finally {
    setLoading(false);
  }
}

// ════════════════════════════════════════════════════════════════════════
// DISPLAY RESULT
// ════════════════════════════════════════════════════════════════════════
function displayResult(data) {
  const { predicted_emotion, confidence, confidence_pct, probabilities, processing_time_ms } = data;
  const cfg = EMOTION_CONFIG[predicted_emotion] ?? EMOTION_CONFIG.neutral;

  // 1. Drive emotion theming across the entire UI
  setActiveEmotion(predicted_emotion);

  // 2. Swap empty state → result content
  emptyState.hidden    = true;
  resultContent.hidden = false;

  // 3. Emotion hero
  emotionEmoji.textContent    = cfg.emoji;
  emotionName.textContent     = cfg.label;
  confidenceValue.textContent = confidence_pct;

  // 4. Donut chart
  const pctNum = Math.round(confidence * 100);
  donutPct.textContent = `${pctNum}%`;
  updateDonut(probabilities, predicted_emotion);

  // 5. Probability bars
  renderProbBars(probBars, probabilities, predicted_emotion);

  // 6. Processing time
  procTime.textContent = `⏱ ${processing_time_ms} ms`;
}

// ════════════════════════════════════════════════════════════════════════
// COPY RESULT
// ════════════════════════════════════════════════════════════════════════
copyBtn.addEventListener("click", () => {
  if (!lastResult) return;
  const { predicted_emotion, confidence_pct, probabilities, text } = lastResult;

  const lines = [
    `EmoSense AI — Emotion Detection Result`,
    `─────────────────────────────────────`,
    `Input    : "${text}"`,
    `Emotion  : ${predicted_emotion.toUpperCase()} (${confidence_pct})`,
    ``,
    `Breakdown:`,
    ...Object.entries(probabilities)
      .sort((a, b) => b[1] - a[1])
      .map(([em, p]) => `  ${em.padEnd(10)}: ${(p * 100).toFixed(1)}%`),
  ];

  navigator.clipboard.writeText(lines.join("\n")).then(() => {
    copyBtn.classList.add("copied");
    copyIcon.textContent = "✅";
    setTimeout(() => {
      copyBtn.classList.remove("copied");
      copyIcon.textContent = "📋";
    }, 2000);
  });
});

// ════════════════════════════════════════════════════════════════════════
// HISTORY
// ════════════════════════════════════════════════════════════════════════
clearHistoryBtn.addEventListener("click", clearHistory);
loadHistory();

// ════════════════════════════════════════════════════════════════════════
// BATCH CSV
// ════════════════════════════════════════════════════════════════════════
initBatch();

// ════════════════════════════════════════════════════════════════════════
// INIT — set default emotion theme
// ════════════════════════════════════════════════════════════════════════
setActiveEmotion("neutral");
