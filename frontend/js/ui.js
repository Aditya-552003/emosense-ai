/**
 * ui.js — DOM helpers, theme management, and UI state functions.
 *
 * Manages theme toggling, emotion theming via CSS custom properties,
 * character counter, loading states, and error banner visibility.
 */

import { EMOTION_CONFIG, MAX_CHARS } from "./config.js";

// ─── DOM references ───────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// Exported DOM elements used by other modules
export const textInput       = $("textInput");
export const analyzeBtn      = $("analyzeBtn");
export const clearBtn        = $("clearBtn");
export const charCounter     = $("charCounter");
export const btnIcon         = $("btnIcon");
export const btnText         = $("btnText");
export const btnLoader       = $("btnLoader");
export const errorBanner     = $("errorBanner");
export const errorMsg        = $("errorMsg");
export const resultCard      = $("resultCard");
export const emptyState      = $("emptyState");
export const resultContent   = $("resultContent");
export const emotionEmoji    = $("emotionEmoji");
export const emotionName     = $("emotionName");
export const confidenceValue = $("confidenceValue");
export const donutPct        = $("donutPct");
export const probBars        = $("probBars");
export const procTime        = $("procTime");
export const copyBtn         = $("copyBtn");
export const copyIcon        = $("copyIcon");
export const historyList     = $("historyList");
export const historyCount    = $("historyCount");
export const historyEmpty    = $("historyEmpty");
export const clearHistoryBtn = $("clearHistoryBtn");
export const statusDot       = $("statusDot");
export const statusLabel     = $("statusLabel");
export const apiStatus       = $("apiStatus");
export const heroEmotion     = $("heroEmotion");
export const themeToggle     = $("themeToggle");
export const themeIcon       = $("themeIcon");
export const alphaSlider     = $("alphaSlider");
export const alphaDisplay    = $("alphaDisplay");

// ─── State ────────────────────────────────────────────────────────────
export let isDark = localStorage.getItem("emosense_theme") !== "light";

// ════════════════════════════════════════════════════════════════════════
// THEME
// ════════════════════════════════════════════════════════════════════════
export function applyTheme(dark) {
  document.body.classList.toggle("dark",  dark);
  document.body.classList.toggle("light", !dark);
  themeIcon.textContent = dark ? "☀️" : "🌙";
  localStorage.setItem("emosense_theme", dark ? "dark" : "light");
}

export function toggleTheme() {
  isDark = !isDark;
  applyTheme(isDark);
}

// ════════════════════════════════════════════════════════════════════════
// EMOTION THEMING  (drive CSS custom properties from JS)
// ════════════════════════════════════════════════════════════════════════
export function setActiveEmotion(emotion) {
  const cfg = EMOTION_CONFIG[emotion] ?? EMOTION_CONFIG.neutral;
  const root = document.documentElement;

  // Update CSS variables
  root.style.setProperty("--emo-h", cfg.hue);
  root.style.setProperty("--emo-s", emotion === "neutral" ? "15%" : "80%");
  root.style.setProperty("--emo-l", "55%");

  // Hero text
  heroEmotion.textContent = cfg.label;

  // Body data attribute (for any attribute-based CSS rules)
  document.body.setAttribute("data-emotion", emotion);
}

// ════════════════════════════════════════════════════════════════════════
// CHARACTER COUNTER
// ════════════════════════════════════════════════════════════════════════
export function updateCounter() {
  const len = textInput.value.length;
  charCounter.textContent = `${len} / ${MAX_CHARS}`;
  charCounter.className   = "char-counter";

  if (len > MAX_CHARS * 0.9) charCounter.classList.add("error");
  else if (len > MAX_CHARS * 0.7) charCounter.classList.add("warn");

  // Enable/disable analyse button
  analyzeBtn.disabled = len === 0 || !textInput.value.trim();
}

// ════════════════════════════════════════════════════════════════════════
// ERROR HELPERS
// ════════════════════════════════════════════════════════════════════════
export function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.hidden   = false;
}

export function hideError() {
  errorBanner.hidden = true;
}

// ════════════════════════════════════════════════════════════════════════
// LOADING STATE
// ════════════════════════════════════════════════════════════════════════
export function setLoading(on) {
  analyzeBtn.classList.toggle("loading", on);
  analyzeBtn.disabled = on;
  btnLoader.style.display = on ? "block" : "";
}

// ════════════════════════════════════════════════════════════════════════
// UTILITY
// ════════════════════════════════════════════════════════════════════════
export function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
