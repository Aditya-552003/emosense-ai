"""
config.py — Centralized application settings.

All configuration is loaded from environment variables with sensible
defaults.  Values can be overridden via a `.env` file in the project
root or by setting the corresponding environment variables directly.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout (resolved relative to *this* file so the server can
# be started from any working directory)
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent          # backend/app/
BACKEND_DIR = _APP_DIR.parent                       # backend/
ROOT_DIR = BACKEND_DIR.parent                       # project root
MODEL_DIR = BACKEND_DIR / os.getenv("MODEL_DIR", "model")
NRC_PATH = MODEL_DIR / "nrc_lexicon.txt"
ROBERTA_DIR = MODEL_DIR / "roberta_best"
FRONTEND_DIR = ROOT_DIR / "frontend"

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8080"))

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT: str = "%H:%M:%S"

# ---------------------------------------------------------------------------
# API metadata
# ---------------------------------------------------------------------------
API_TITLE = "Emotion Detection API"
API_DESCRIPTION = (
    "Hybrid stacking ensemble: fine-tuned RoBERTa (contextual) "
    "+ TF-IDF / NRC / XGBoost (semantic) -> Logistic Regression meta-learner."
)
API_VERSION = "1.0.0"
