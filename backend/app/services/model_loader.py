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
    from huggingface_hub import hf_hub_download
    repo_id = os.getenv("HF_MODEL_REPO", "DSAditya552003/emosense-ai-model")
    token = os.getenv("HF_TOKEN")

    def _load_artifact(name: str, is_pickle: bool = False):
        local_path = os.path.join(MODEL_DIR, name)
        if os.path.exists(local_path):
            log.info(f"Loading {name} from local storage...")
            path_to_load = local_path
        else:
            log.info(f"Downloading {name} from Hugging Face Hub ({repo_id})...")
            try:
                path_to_load = hf_hub_download(repo_id=repo_id, filename=name, token=token)
            except Exception as e:
                log.error(f"Failed to download {name} from HF Hub: {e}")
                raise FileNotFoundError(f"Model artifact {name} not found locally or on HF Hub.") from e

        if is_pickle:
            with open(path_to_load, "rb") as f:
                return pickle.load(f)
        else:
            return joblib.load(path_to_load)

    _models["word_tfidf"] = _load_artifact("word_tfidf.pkl")
    _models["char_tfidf"] = _load_artifact("char_tfidf.pkl")
    _models["nrc_scaler"] = _load_artifact("nrc_scaler.pkl")
    _models["xgb"] = _load_artifact("xgb.pkl")
    _models["meta_clf"] = _load_artifact("meta_clf.pkl")
    _models["label_encoder"] = _load_artifact("label_encoder.pkl", is_pickle=True)
    log.info("Sklearn artifacts loaded.")

    # 3. RoBERTa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _models["device"] = device

    roberta_dir = str(ROBERTA_DIR)
    if os.path.isdir(roberta_dir) and os.path.exists(os.path.join(roberta_dir, "model.safetensors")):
        log.info(f"Loading RoBERTa from local directory: {roberta_dir} ...")
        model_source = roberta_dir
    else:
        log.info(f"Loading RoBERTa from Hugging Face Hub repo: {repo_id} ...")
        model_source = repo_id

    _models["tokenizer"] = AutoTokenizer.from_pretrained(model_source, token=token)
    roberta = AutoModelForSequenceClassification.from_pretrained(model_source, token=token)
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
