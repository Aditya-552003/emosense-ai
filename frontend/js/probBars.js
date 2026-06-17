/**
 * probBars.js — Probability bar rendering.
 *
 * Renders the animated horizontal bars showing per-emotion probabilities.
 */

import { EMOTION_CONFIG } from "./config.js";

/**
 * Render probability bars into the given container.
 * @param {HTMLElement} container - The DOM element to render into.
 * @param {Object} probabilities - Emotion→probability mapping.
 * @param {string} activeEmotion - The predicted emotion key.
 */
export function renderProbBars(container, probabilities, activeEmotion) {
  container.innerHTML = "";

  // Sort descending
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([emotion, prob], idx) => {
    const cfg   = EMOTION_CONFIG[emotion] ?? EMOTION_CONFIG.neutral;
    const pct   = (prob * 100).toFixed(1);
    const isTop = emotion === activeEmotion;

    const row = document.createElement("div");
    row.className = `prob-row${isTop ? " active" : ""}`;
    row.style.animationDelay = `${idx * 60}ms`;
    row.innerHTML = `
      <div class="prob-label">
        <span class="prob-emoji" aria-hidden="true">${cfg.emoji}</span>
        ${cfg.label}
      </div>
      <div class="prob-track" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${cfg.label} probability">
        <div class="prob-fill" style="background:${cfg.color};" data-width="${prob}"></div>
      </div>
      <div class="prob-pct">${pct}%</div>
    `;
    container.appendChild(row);
  });

  // Trigger bar animations on next frame
  requestAnimationFrame(() => {
    container.querySelectorAll(".prob-fill").forEach((fill) => {
      fill.style.transform = `scaleX(${fill.dataset.width})`;
    });
  });
}
