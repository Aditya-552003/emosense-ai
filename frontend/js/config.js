/**
 * config.js — Centralized constants and emotion configuration.
 *
 * All shared configuration values live here so they can be imported
 * by any module without circular dependencies.
 */

export const API_BASE    = "http://127.0.0.1:8080";
export const MAX_CHARS   = 500;
export const HISTORY_KEY = "emosense_history";
export const MAX_HISTORY = 20;

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
