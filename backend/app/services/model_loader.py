"""
services/model_loader.py — ML model lifecycle management.

Loads all model artifacts (sklearn, XGBoost, RoBERTa) at startup and
provides a module-level accessor for the shared model registry.
"""

import logging
import os
import pickle
from typing import Dict

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import MODEL_DIR, NRC_PATH, ROBERTA_DIR
from app.utils.preprocessing import download_nrc_lexicon, load_nrc_lexicon

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model registry (singleton, populated during startup)
# ---------------------------------------------------------------------------
_models: Dict = {}


def get_models() -> Dict:
    """Return the shared model registry.  Must be called after load_models()."""
    return _models


def load_models() -> None:
    """
    Load all ML artifacts into the module-level registry.

    Called once during the FastAPI lifespan startup event.
    """
    log.info("=" * 55)
    log.info("  Emotion Detection API  --  starting up")
    log.info("=" * 55)

    # 1. NRC Emotion Lexicon
    download_nrc_lexicon(str(NRC_PATH))
    nrc_dict, nrc_emotions = load_nrc_lexicon(str(NRC_PATH))
    _models["nrc_dict"] = nrc_dict
    _models["nrc_emotions"] = nrc_emotions
    log.info(f"NRC Lexicon loaded  ({len(nrc_emotions)} emotion categories)")

    # 2. Sklearn / joblib artifacts
    log.info("Loading sklearn artifacts ...")
    _models["word_tfidf"] = joblib.load(os.path.join(MODEL_DIR, "word_tfidf.pkl"))
    _models["char_tfidf"] = joblib.load(os.path.join(MODEL_DIR, "char_tfidf.pkl"))
    _models["nrc_scaler"] = joblib.load(os.path.join(MODEL_DIR, "nrc_scaler.pkl"))
    _models["xgb"] = joblib.load(os.path.join(MODEL_DIR, "xgb.pkl"))
    _models["meta_clf"] = joblib.load(os.path.join(MODEL_DIR, "meta_clf.pkl"))
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        _models["label_encoder"] = pickle.load(f)
    log.info("Sklearn artifacts loaded.")

    # 3. RoBERTa
    roberta_dir = str(ROBERTA_DIR)
    log.info(f"Loading RoBERTa from {roberta_dir} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _models["device"] = device
    _models["tokenizer"] = AutoTokenizer.from_pretrained(roberta_dir)
    roberta = AutoModelForSequenceClassification.from_pretrained(roberta_dir)
    roberta.to(device)
    roberta.eval()
    # Warm-up pass to pre-compile kernels
    _warmup(roberta, _models["tokenizer"], device)
    _models["roberta"] = roberta
    log.info(f"RoBERTa loaded on {device.upper()}")

    log.info("All models ready  [OK]")
    log.info("=" * 55)


def unload_models() -> None:
    """Release model references on shutdown."""
    _models.clear()
    log.info("Models unloaded.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _warmup(model, tokenizer, device: str) -> None:
    """Run a single dummy inference to warm up PyTorch JIT / cuDNN."""
    enc = tokenizer(
        ["warm up"],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        model(**enc)
