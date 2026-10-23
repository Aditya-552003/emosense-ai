/**
 * api.js — API communication layer.
 *
 * Handles all fetch requests to the backend, including health checks,
 * single predictions, and batch CSV uploads.
 */

import { API_BASE } from "./config.js";

/**
 * Check if the backend API is reachable.
 * @param {HTMLElement} apiStatus - The status container element.
 * @param {HTMLElement} statusLabel - The label text element.
 * @returns {Promise<boolean>}
 */
export async function checkApiStatus(apiStatus, statusLabel) {
  for (const path of ["/gradio_api/api/health/ready", "/api/health/ready", "/health/ready", "/"]) {
    try {
      const res = await fetch(`${API_BASE}${path}`, { signal: AbortSignal.timeout(4000) });
      if (res.ok) {
        const ct = res.headers.get("content-type") || "";
        if (ct.includes("application/json") || path.includes("health")) {
          apiStatus.className = "api-status online";
          statusLabel.textContent = "API Online";
          return true;
        }
      }
    } catch (_) { /* continue */ }
  }

  apiStatus.className = "api-status offline";
  statusLabel.textContent = "API Offline";
  return false;
}

/**
 * Send a single text for emotion prediction.
 * @param {string} text - The text to analyze.
 * @param {number} alpha - Blend weight (0–1).
 * @returns {Promise<Object>} The prediction result.
 */
export async function fetchPrediction(text, alpha = 0.7) {
  let url = `${API_BASE}/predict`;
  let res = await fetch(url, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ text, alpha }),
    signal:  AbortSignal.timeout(30_000),
  });

  // Seamless fallback if backend routes are mounted under /gradio_api/api or /api
  if (res.status === 404) {
    for (const prefix of ["/gradio_api/api", "/api"]) {
      url = `${API_BASE}${prefix}/predict`;
      res = await fetch(url, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text, alpha }),
        signal:  AbortSignal.timeout(30_000),
      });
      if (res.ok) break;
    }
  }

  if (!res.ok) {
    let detail = `Server error ${res.status}`;
    try {
      const err = await res.json();
      detail = err.detail ?? detail;
    } catch (_) {}
    throw new Error(detail);
  }

  return res.json();
}

/**
 * Upload a CSV file for batch emotion prediction.
 * @param {File} file - The CSV file to upload.
 * @param {number} alpha - Blend weight (0–1).
 * @returns {Promise<Object>} The batch prediction results.
 */
export async function fetchBatchPrediction(file, alpha = 0.7) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("alpha", alpha);

  let url = `${API_BASE}/predict/batch`;
  let res = await fetch(url, {
    method: "POST",
    body:   formData,
    signal: AbortSignal.timeout(120_000),  // 2 min timeout for batch
  });

  // Seamless fallback if backend routes are mounted under /gradio_api/api or /api
  if (res.status === 404) {
    for (const prefix of ["/gradio_api/api", "/api"]) {
      url = `${API_BASE}${prefix}/predict/batch`;
      res = await fetch(url, {
        method: "POST",
        body:   formData,
        signal: AbortSignal.timeout(120_000),
      });
      if (res.ok) break;
    }
  }

  if (!res.ok) {
    let detail = `Server error ${res.status}`;
    try {
      const err = await res.json();
      detail = err.detail ?? detail;
    } catch (_) {}
    throw new Error(detail);
  }

  return res.json();
}

