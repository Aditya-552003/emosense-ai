"""
services/model_loader.py — ML model lifecycle management.

Loads all model artifacts (sklearn, XGBoost, RoBERTa) at startup,
verifies artifact consistency and class alignment, and provides a
thread-safe model registry and readiness checking.
"""

import gc
import logging
import os
import pickle
from enum import Enum
from typing import Any, Dict, Optional

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import (
    HF_MODEL_REPO,
    HF_MODEL_REVISION,
    HF_TOKEN,
    MODEL_DIR,
    NRC_PATH,
    ROBERTA_DIR,
)
from app.utils.preprocessing import download_nrc_lexicon, load_nrc_lexicon

log = logging.getLogger(__name__)


class ModelState(str, Enum):
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Module-level model registry (singleton)
# ---------------------------------------------------------------------------
_models: Dict[str, Any] = {}
_state: ModelState = ModelState.UNINITIALIZED
_state_error: Optional[str] = None


def get_model_state() -> ModelState:
    """Return the current model lifecycle state."""
    return _state


def is_model_ready() -> bool:
    """Return True if all models are loaded and verified."""
    return _state == ModelState.READY


def get_model_error() -> Optional[str]:
    """Return the last startup error message, if any."""
    return _state_error


def get_models() -> Dict[str, Any]:
    """
    Return the shared model registry.
    Raises RuntimeError if models have not completed initialization.
    """
    if _state != ModelState.READY:
        raise RuntimeError(
            f"Model registry is not ready. Current state: {_state.value}. "
            f"Details: {_state_error or 'Startup in progress or not triggered.'}"
        )
    return _models


def load_models() -> None:
    """
    Load and verify all ML artifacts into the module-level registry.
    Atomically transitions state from UNINITIALIZED -> LOADING -> READY (or FAILED).
    """
    global _state, _state_error
    _state = ModelState.LOADING
    _state_error = None

    loaded: Dict[str, Any] = {}

    log.info("=" * 60)
    log.info("  Loading ML artifacts (Model Directory: %s)", MODEL_DIR)
    log.info("=" * 60)

    try:
        # 1. NRC Emotion Lexicon
        nrc_file = str(NRC_PATH)
        download_nrc_lexicon(nrc_file)
        nrc_dict, nrc_emotions = load_nrc_lexicon(nrc_file)
        loaded["nrc_dict"] = nrc_dict
        loaded["nrc_emotions"] = nrc_emotions
        log.info("Loaded NRC Lexicon (%d emotion categories)", len(nrc_emotions))

        # 2. Sklearn / joblib / pickle artifacts
        token = HF_TOKEN if HF_TOKEN else None

        def _load_artifact(name: str, is_pickle: bool = False) -> Any:
            local_path = os.path.join(MODEL_DIR, name)
            if os.path.exists(local_path):
                log.info("Loading artifact '%s' from local storage...", name)
                path_to_load = local_path
            else:
                log.info(
                    "Downloading artifact '%s' from HF Hub (%s @ %s)...",
                    name,
                    HF_MODEL_REPO,
                    HF_MODEL_REVISION,
                )
                from huggingface_hub import hf_hub_download

                try:
                    path_to_load = hf_hub_download(
                        repo_id=HF_MODEL_REPO,
                        filename=name,
                        revision=HF_MODEL_REVISION,
                        token=token,
                    )
                except Exception as exc:
                    log.error("Failed to download %s from HF Hub: %s", name, exc)
                    raise FileNotFoundError(
                        f"Model artifact '{name}' could not be found locally at {local_path} "
                        f"or downloaded from Hugging Face Hub ({HF_MODEL_REPO})."
                    ) from exc

            if is_pickle:
                with open(path_to_load, "rb") as f:
                    return pickle.load(f)
            return joblib.load(path_to_load)

        loaded["word_tfidf"] = _load_artifact("word_tfidf.pkl")
        loaded["char_tfidf"] = _load_artifact("char_tfidf.pkl")
        loaded["nrc_scaler"] = _load_artifact("nrc_scaler.pkl")
        loaded["xgb"] = _load_artifact("xgb.pkl")
        loaded["meta_clf"] = _load_artifact("meta_clf.pkl")
        loaded["label_encoder"] = _load_artifact("label_encoder.pkl", is_pickle=True)

        # 3. RoBERTa Transformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded["device"] = device

        roberta_dir = str(ROBERTA_DIR)
        has_local_roberta = os.path.isdir(roberta_dir) and (
            os.path.exists(os.path.join(roberta_dir, "model.safetensors"))
            or os.path.exists(os.path.join(roberta_dir, "pytorch_model.bin"))
        )

        if has_local_roberta:
            log.info("Loading RoBERTa from local directory: %s", roberta_dir)
            model_source = roberta_dir
            subfolder = None
        else:
            log.info(
                "Loading RoBERTa from HF Hub (%s @ %s, subfolder: roberta_best)...",
                HF_MODEL_REPO,
                HF_MODEL_REVISION,
            )
            model_source = HF_MODEL_REPO
            subfolder = "roberta_best"

        load_kwargs: Dict[str, Any] = {"revision": HF_MODEL_REVISION}
        if token:
            load_kwargs["token"] = token
        if subfolder:
            load_kwargs["subfolder"] = subfolder

        tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)
        roberta = AutoModelForSequenceClassification.from_pretrained(model_source, **load_kwargs)
        roberta.to(device)
        roberta.eval()

        # Warm-up pass
        _warmup(roberta, tokenizer, device)

        loaded["tokenizer"] = tokenizer
        loaded["roberta"] = roberta
        log.info("RoBERTa loaded and warmed up on %s", device.upper())

        # 4. Consistency & class alignment verification
        classes = loaded["label_encoder"].classes_.tolist()
        expected_classes = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]
        if classes != expected_classes:
            log.warning("Label encoder classes %s differ from documented %s", classes, expected_classes)

        meta_features = getattr(loaded["meta_clf"], "n_features_in_", None)
        if meta_features is not None and meta_features != 12:
            raise ValueError(f"meta_clf expected 12 input features, found {meta_features}")

        # Swap atomically into registry
        _models.clear()
        _models.update(loaded)
        _state = ModelState.READY
        log.info("All model artifacts loaded and verified [OK]")
        log.info("=" * 60)

    except Exception as exc:
        _state = ModelState.FAILED
        _state_error = str(exc)
        _models.clear()
        log.error("Model loading failed: %s", exc, exc_info=True)
        raise


def unload_models() -> None:
    """Release all model references and free PyTorch / GPU memory."""
    global _state, _state_error
    _models.clear()
    _state = ModelState.UNINITIALIZED
    _state_error = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("All models unloaded successfully.")


def _warmup(model: Any, tokenizer: Any, device: str) -> None:
    """Run a single warmup forward pass to pre-compile execution kernels."""
    enc = tokenizer(
        ["warmup"],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        model(**enc)
