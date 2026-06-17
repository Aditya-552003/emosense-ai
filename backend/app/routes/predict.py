"""
routes/predict.py — Prediction endpoints (single + batch).
"""

import io
import logging
import time

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.schemas.predict import PredictRequest, PredictResponse
from app.services.inference import run_inference

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse, summary="Predict emotion")
async def predict(payload: PredictRequest):
    """
    Detect the emotion in the provided text using the hybrid stacking ensemble.

    - **text**: The sentence or paragraph to analyze (max 2000 chars).

    Returns the predicted emotion label, confidence score, and per-class
    probability distribution.
    """
    try:
        result = run_inference(payload.text, alpha=payload.alpha)
        return result
    except Exception as exc:
        log.exception("Inference error")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(exc)}",
        )


@router.post("/predict/batch", summary="Batch predict emotions from CSV")
async def predict_batch(
    file: UploadFile = File(...),
    alpha: float = Form(0.7),
):
    """
    Upload a CSV file with a 'text' column. Returns emotion predictions for
    every row. Max 500 rows per batch.

    - **file**: CSV file with a column named 'text' (case-insensitive)
    - **alpha**: Blend weight (0.0 = pure XGBoost, 1.0 = pure RoBERTa)
    """
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, detail="Only .csv files are accepted.")

    alpha = max(0.0, min(1.0, alpha))

    try:
        raw = await file.read()
        text_content = raw.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text_content))
    except Exception as exc:
        raise HTTPException(400, detail=f"Failed to parse CSV: {str(exc)}")

    # Find text column (case-insensitive)
    text_col = None
    for col in df.columns:
        if col.strip().lower() == "text":
            text_col = col
            break

    if text_col is None:
        raise HTTPException(
            400,
            detail=f"CSV must contain a 'text' column. Found columns: {list(df.columns)}",
        )

    # Limit rows
    if len(df) > 500:
        raise HTTPException(400, detail=f"Too many rows ({len(df)}). Maximum is 500.")

    texts = df[text_col].fillna("").astype(str).tolist()
    results = []

    t0_batch = time.perf_counter()
    for idx, text in enumerate(texts):
        text = text.strip()
        if not text:
            results.append(
                {
                    "row": idx + 1,
                    "text": "",
                    "predicted_emotion": "unknown",
                    "confidence": 0.0,
                    "confidence_pct": "0.0%",
                    "probabilities": {},
                    "processing_time_ms": 0.0,
                }
            )
            continue

        try:
            result = run_inference(text, alpha=alpha)
            result["row"] = idx + 1
            result["text"] = text[:200]  # Truncate for response size
            results.append(result)
        except Exception as exc:
            log.warning(f"Batch row {idx + 1} failed: {exc}")
            results.append(
                {
                    "row": idx + 1,
                    "text": text[:200],
                    "predicted_emotion": "error",
                    "confidence": 0.0,
                    "confidence_pct": "0.0%",
                    "probabilities": {},
                    "processing_time_ms": 0.0,
                }
            )

    batch_time = (time.perf_counter() - t0_batch) * 1000

    return {
        "success": True,
        "total_rows": len(results),
        "alpha": round(alpha, 2),
        "batch_processing_time_ms": round(batch_time, 2),
        "predictions": results,
    }
