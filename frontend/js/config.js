/**
 * config.js — Centralized constants and emotion configuration.
 *
 * All shared configuration values live here so they can be imported
 * by any module without circular dependencies.
 *
 * API_BASE resolution order:
 *  1. Same origin (when frontend is served by FastAPI on 8080 / 7860 / remote domain)
 *  2. Local backend fallback (http://127.0.0.1:8080) for file:// or local static dev servers (e.g. port 5500)
 *  3. Explicit <meta name="api-base" content="..."> override if provided
 */

function resolveApiBase() {
  const meta = document.querySelector('meta[name="api-base"]')?.content?.trim();
  
  // Optional URL param override (e.g. ?api=https://my-space.hf.space) or localStorage override
  let queryOverride = null;
  let storedOverride = null;
  try {
    queryOverride = new URLSearchParams(window.location.search).get("api")?.trim();
    if (queryOverride) localStorage.setItem("emosense_api_base", queryOverride);
    storedOverride = localStorage.getItem("emosense_api_base")?.trim();
  } catch (_) { /* ignore cross-origin/sandbox storage errors */ }

  const customBase = queryOverride || storedOverride || (meta && meta !== "" ? meta : null);

  // If running via HTTP/HTTPS
  if (window.location.protocol.startsWith("http")) {
    const isLocal = window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost";
    const isHfSpace = window.location.hostname.endsWith("hf.space");

    // If served directly by backend on HF Space or local uvicorn (port 8080 / 7860)
    if (isHfSpace || window.location.port === "8080" || window.location.port === "7860") {
      return customBase || window.location.origin;
    }

    // If served by a local static dev server (e.g. Live Server on port 5500, 3000)
    if (isLocal) {
      return customBase || "http://127.0.0.1:8080";
    }

    // If hosted remotely on Vercel / Netlify / custom static CDN
    return customBase || window.location.origin;
  }

  // Fallback for file:/// protocol
  return customBase || "http://127.0.0.1:8080";
}

export const API_BASE = resolveApiBase();
export const MAX_CHARS   = 2000;
export const HISTORY_KEY = "emosense_history";
export const BATCH_KEY   = "emosense_last_batch";
export const SESSION_KEY = "emosense_session_predictions";
export const MAX_HISTORY = 100;

/**
 * Emotion configuration — colours, emojis, and display labels.
 * Keyed by the emotion class name returned by the API.
 */
export const EMOTION_CONFIG = {
  joy:      { emoji: "😊", hue: 45,  label: "Joy",      color: "hsl(45,95%,58%)" },
  sadness:  { emoji: "😢", hue: 220, label: "Sadness",   color: "hsl(220,80%,55%)" },
  anger:    { emoji: "😡", hue: 5,   label: "Anger",     color: "hsl(5,85%,52%)" },
  fear:     { emoji: "😨", hue: 270, label: "Fear",      color: "hsl(270,70%,55%)" },
  surprise: { emoji: "😲", hue: 190, label: "Surprise",  color: "hsl(190,85%,50%)" },
  neutral:  { emoji: "😐", hue: 215, label: "Neutral",   color: "hsl(215,15%,55%)" },
};
