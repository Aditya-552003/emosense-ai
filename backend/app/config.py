"""
config.py — Centralized application settings.

All configuration is loaded from environment variables with safe, sensible
defaults. Values can be overridden via a `.env` file in the project
root or by setting the corresponding environment variables directly.
"""

import os
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Directory layout (resolved relative to this file so the server can
# be started from any working directory)
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent          # backend/app/
BACKEND_DIR = _APP_DIR.parent                       # backend/
ROOT_DIR = BACKEND_DIR.parent                       # project root

_model_dir_env = os.getenv("MODEL_DIR", "model")
_model_dir_path = Path(_model_dir_env)
MODEL_DIR = _model_dir_path if _model_dir_path.is_absolute() else (BACKEND_DIR / _model_dir_path)

NRC_PATH = MODEL_DIR / "nrc_lexicon.txt"
ROBERTA_DIR = MODEL_DIR / "roberta_best"
FRONTEND_DIR = ROOT_DIR / "frontend"

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8080"))

# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------
# Safe defaults: local UI and Gradio origins.
_raw_cors = os.getenv(
    "CORS_ORIGINS",
    "http://127.0.0.1:8080,http://localhost:8080,http://127.0.0.1:7860,http://localhost:7860",
)
CORS_ORIGINS: List[str] = [origin.strip() for origin in _raw_cors.split(",") if origin.strip()]

# Browsers disallow allow_credentials=True when origin is wildcard "*"
CORS_ALLOW_ALL: bool = "*" in CORS_ORIGINS
CORS_ALLOW_CREDENTIALS: bool = not CORS_ALLOW_ALL

# ---------------------------------------------------------------------------
# Security & Upload Limits
# ---------------------------------------------------------------------------
MAX_CHARS: int = int(os.getenv("MAX_CHARS", "2000"))
MAX_BATCH_ROWS: int = int(os.getenv("MAX_BATCH_ROWS", "500"))
MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(5 * 1024 * 1024)))  # 5 MB
MAX_CONCURRENT_INFERENCES: int = int(os.getenv("MAX_CONCURRENT_INFERENCES", "4"))
DEFAULT_ALPHA: float = float(os.getenv("DEFAULT_ALPHA", "0.7"))

# ---------------------------------------------------------------------------
# Hugging Face Model Repository
# ---------------------------------------------------------------------------
HF_MODEL_REPO: str = os.getenv("HF_MODEL_REPO", "DSAditya552003/emosense-ai-model")
HF_MODEL_REVISION: str = os.getenv("HF_MODEL_REVISION", "main")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

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
    "Hybrid ensemble emotion detection combining fine-tuned RoBERTa (contextual) "
    "and XGBoost on TF-IDF + NRC lexicon features with dynamic alpha-weighted soft voting "
    "and stacking meta-classifier capabilities."
)
API_VERSION = "1.0.0"
