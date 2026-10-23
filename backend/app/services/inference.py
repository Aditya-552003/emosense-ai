"""
services/inference.py — Ensemble emotion inference pipeline.

Implements the multi-branch emotion prediction architecture:
  1. Semantic branch  : word TF-IDF + char TF-IDF + NRC lexicon -> XGBoost
  2. Contextual branch: fine-tuned RoBERTa
  3. Ensemble:
     - Default / Deployed: Alpha-weighted soft voting:
         P = alpha * P(RoBERTa) + (1 - alpha) * P(XGBoost)
     - Stacking Meta-Classifier:
         P = meta_clf.predict_proba([P(RoBERTa), P(XGBoost)])
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.sparse import hstack

from app.services.model_loader import get_models
from app.utils.preprocessing import clean_text, nrc_features


def _normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """Ensure probabilities are finite, non-negative, and sum to 1.0 per sample."""
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 0.0, 1.0)
    row_sums = probs.sum(axis=-1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    normalized = probs / safe_sums
    n_classes = probs.shape[-1]
    uniform = np.full_like(probs, 1.0 / n_classes)
    return np.where(row_sums > 0, normalized, uniform)


def run_inference(
    text: str,
    alpha: float = 0.7,
    ensemble_method: str = "soft_voting",
) -> Dict[str, Any]:
    """
    Run full multi-branch emotion prediction on a single text string.

    Args:
        text: Input text to classify.
        alpha: Blend weight for soft voting (alpha * RoBERTa + (1-alpha) * XGBoost).
        ensemble_method: 'soft_voting' (default) or 'stacking'.

    Returns:
        Dict conforming to the PredictResponse schema.
    """
    results = run_batch_inference([text], alpha=alpha, ensemble_method=ensemble_method, batch_size=1)
    return results[0]


def run_batch_inference(
    texts: List[str],
    alpha: float = 0.7,
    ensemble_method: str = "soft_voting",
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Vectorized batch inference over multiple texts.

    Transforms TF-IDF, NRC, and XGBoost features in bulk, and executes RoBERTa
    inference in mini-batches to optimize GPU/CPU throughput.

    Args:
        texts: List of input strings.
        alpha: Blend weight for soft voting in [0.0, 1.0].
        ensemble_method: 'soft_voting' or 'stacking'.
        batch_size: Mini-batch size for transformer tokenization & forward pass.

    Returns:
        List of result dictionaries.
    """
    t0 = time.perf_counter()
    models = get_models()
    n_samples = len(texts)

    if n_samples == 0:
        return []

    classes = models["label_encoder"].classes_.tolist()
    n_classes = len(classes)

    # ── 1. Semantic branch (XGBoost) ─────────────────────────────────────────
    # Training-time clean_text applied to semantic branch
    cleaned_texts = [clean_text(t) for t in texts]

    x_w = models["word_tfidf"].transform(cleaned_texts)
    x_c = models["char_tfidf"].transform(cleaned_texts)
    raw_nrc = nrc_features(cleaned_texts, models["nrc_dict"], models["nrc_emotions"])
    x_n = models["nrc_scaler"].transform(raw_nrc)

    x_sem = hstack([x_w, x_c, x_n])
    sem_probs = models["xgb"].predict_proba(x_sem)  # (n_samples, 6)

    device = "cuda" if torch.cuda.is_available() else models.get("device", "cpu")
    tokenizer = models["tokenizer"]
    roberta = models["roberta"]
    if hasattr(roberta, "parameters"):
        try:
            param_dev = next(roberta.parameters()).device.type
            if param_dev != device:
                roberta.to(device)
        except Exception as _dev_err:
            log.warning("Could not set RoBERTa device to %s: %s", device, _dev_err)
            device = "cpu"

    ctx_probs_list = []
    with torch.inference_mode():
        for i in range(0, n_samples, batch_size):
            chunk = texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            if hasattr(enc, "to"):
                enc = enc.to(device)
            elif isinstance(enc, dict):
                enc = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}

            logits = roberta(**enc).logits
            p = torch.softmax(logits, dim=1).cpu().numpy()
            ctx_probs_list.append(p)

    ctx_probs = np.vstack(ctx_probs_list) if ctx_probs_list else np.zeros((0, n_classes))

    # ── 3. Ensemble Computation ──────────────────────────────────────────────
    if ensemble_method == "stacking" and "meta_clf" in models:
        meta_X = np.hstack([ctx_probs, sem_probs])  # (n_samples, 12)
        raw_final = models["meta_clf"].predict_proba(meta_X)
    else:
        # Default: Alpha-weighted soft voting
        raw_final = alpha * ctx_probs + (1.0 - alpha) * sem_probs

    final_probs = _normalize_probabilities(raw_final)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    per_item_ms = round(elapsed_ms / max(n_samples, 1), 2)

    results: List[Dict[str, Any]] = []
    for i in range(n_samples):
        probs_row = final_probs[i]
        pred_idx = int(np.argmax(probs_row))
        pred_emotion = classes[pred_idx]
        confidence = float(probs_row[pred_idx])

        prob_dict = {cls_name: float(round(probs_row[c_idx], 6)) for c_idx, cls_name in enumerate(classes)}

        results.append(
            {
                "success": True,
                "predicted_emotion": pred_emotion,
                "confidence": round(confidence, 6),
                "confidence_pct": f"{confidence:.1%}",
                "probabilities": prob_dict,
                "processing_time_ms": per_item_ms,
                "alpha": round(float(alpha), 2),
            }
        )

    return results
