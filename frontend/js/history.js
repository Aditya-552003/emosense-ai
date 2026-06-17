/**
 * history.js — Prediction history management (localStorage).
 *
 * Provides CRUD operations and rendering for the persistent
 * prediction history sidebar.
 */

import { EMOTION_CONFIG, HISTORY_KEY, MAX_HISTORY } from "./config.js";
import { escapeHtml, updateCounter, hideError, textInput, historyCount, historyEmpty, historyList } from "./ui.js";

// ─── State ────────────────────────────────────────────────────────────
let history = [];

// ════════════════════════════════════════════════════════════════════════
// LOAD / SAVE
// ════════════════════════════════════════════════════════════════════════
export function loadHistory() {
  try {
    history = JSON.parse(localStorage.getItem(HISTORY_KEY)) ?? [];
  } catch (_) {
    history = [];
  }
  renderHistory();
}

export function addToHistory(item) {
  history.unshift({
    ...item,
    timestamp: new Date().toISOString(),
  });
  if (history.length > MAX_HISTORY) history.pop();
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  renderHistory();
}

export function clearHistory() {
  history = [];
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
}

// ════════════════════════════════════════════════════════════════════════
// RENDER
// ════════════════════════════════════════════════════════════════════════
function renderHistory() {
  historyCount.textContent = history.length;

  if (history.length === 0) {
    historyEmpty.hidden = false;
    return;
  }
  historyEmpty.hidden = true;

  // Remove old items (keep empty placeholder)
  historyList.querySelectorAll(".history-item").forEach((el) => el.remove());

  history.forEach((item) => {
    const cfg     = EMOTION_CONFIG[item.predicted_emotion] ?? EMOTION_CONFIG.neutral;
    const timeStr = formatRelativeTime(item.timestamp);

    const card = document.createElement("div");
    card.className = "history-item";
    card.style.setProperty("--item-h", cfg.hue);
    card.setAttribute("title", "Click to re-analyse this text");
    card.setAttribute("role", "button");
    card.setAttribute("tabindex", "0");

    card.innerHTML = `
      <div class="history-item-header">
        <span class="history-emoji" aria-hidden="true">${cfg.emoji}</span>
        <span class="history-emotion-badge">${cfg.label}</span>
        <span class="history-conf">${item.confidence_pct}</span>
      </div>
      <div class="history-text">${escapeHtml(item.text)}</div>
      <div class="history-time">${timeStr}</div>
    `;

    // Click to re-fill input
    card.addEventListener("click", () => {
      textInput.value = item.text;
      updateCounter();
      hideError();
      textInput.scrollIntoView({ behavior: "smooth", block: "center" });
      textInput.focus();
    });
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") card.click();
    });

    historyList.appendChild(card);
  });
}

// ─── Helpers ──────────────────────────────────────────────────────────
function formatRelativeTime(iso) {
  const delta = (Date.now() - new Date(iso).getTime()) / 1000;
  if (delta < 60)  return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}
