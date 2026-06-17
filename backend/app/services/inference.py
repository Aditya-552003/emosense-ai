"""
services/inference.py — Ensemble emotion inference pipeline.

Implements the three-branch prediction:
  1. Semantic branch  : word TF-IDF + char TF-IDF + NRC lexicon → XGBoost
  2. Contextual branch: fine-tuned RoBERTa
  3. Hybrid blend     : alpha * RoBERTa + (1-alpha) * XGBoost
"""

import time
from typing import Dict

import numpy as np
import torch
from scipy.sparse import hstack

from app.services.model_loader import get_models
from app.utils.preprocessing import clean_text, nrc_features


def run_inference(text: str, alpha: float = 0.7) -> Dict:
    """
    Full three-branch prediction pipeline.

    Args:
        text:  Input text to classify.
        alpha: Blend weight — alpha * RoBERTa + (1-alpha) * XGBoost.
               0.0 = pure XGBoost, 1.0 = pure RoBERTa, 0.7 = default.

    Returns a dict with keys matching PredictResponse.
    """
    t0 = time.perf_counter()
    models = get_models()

    clean = clean_text(text)

    # ── Semantic branch (XGBoost) ────────────────────────────────────────
    x_w = models["word_tfidf"].transform([clean])
    x_c = models["char_tfidf"].transform([clean])
    x_n = models["nrc_scaler"].transform(
        nrc_features([clean], models["nrc_dict"], models["nrc_emotions"])
    )
    x_sem = hstack([x_w, x_c, x_n])
    sem_probs = models["xgb"].predict_proba(x_sem)  # (1, 6)

    # ── Contextual branch (RoBERTa) ──────────────────────────────────────
    device = models["device"]
    enc = models["tokenizer"](
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = models["roberta"](**enc).logits
    ctx_probs = torch.softmax(logits, dim=1).cpu().numpy()  # (1, 6)

    # ── Hybrid Ensemble: Alpha-Weighted Soft Voting ──────────────────────
    # alpha = 1.0 → pure RoBERTa  |  alpha = 0.0 → pure XGBoost
    final_probs = alpha * ctx_probs[0] + (1.0 - alpha) * sem_probs[0]

    classes = models["label_encoder"].classes_.tolist()
    pred_idx = int(np.argmax(final_probs))
    pred_emotion = classes[pred_idx]
    confidence = float(final_probs[pred_idx])

    prob_dict = {cls: float(p) for cls, p in zip(classes, final_probs)}

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "success": True,
        "predicted_emotion": pred_emotion,
        "confidence": round(confidence, 6),
        "confidence_pct": f"{confidence:.1%}",
        "probabilities": prob_dict,
        "processing_time_ms": round(elapsed_ms, 2),
        "alpha": round(alpha, 2),
    }
