"""
routes/predict.py — Prediction endpoints (single + batch).

Implements hardened file upload processing, threadpool concurrency offloading,
safe error handling, and vectorized batch inference.
"""

import asyncio
import io
import logging
import math
import time
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app import config
from app.schemas.predict import BatchPredictResponse, PredictRequest, PredictResponse
from app.services.inference import run_batch_inference, run_inference
from app.services.model_loader import is_model_ready

log = logging.getLogger(__name__)

router = APIRouter()

# Bounded concurrency semaphore to prevent CPU/GPU memory exhaustion
_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_INFERENCES)


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict emotion",
    status_code=status.HTTP_200_OK,
)
async def predict(payload: PredictRequest) -> PredictResponse:
    """
    Detect the emotion in the provided text using the hybrid ensemble.

    - **text**: The sentence or paragraph to analyze (1 to 2000 chars).
    - **alpha**: Blend weight between 0.0 (XGBoost) and 1.0 (RoBERTa).
    - **ensemble_method**: 'soft_voting' (default) or 'stacking'.

    Returns the predicted emotion label, confidence score, and per-class probabilities.
    """
    if not is_model_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready to serve predictions. Please check /health/ready.",
        )

    try:
        async with _semaphore:
            result = await asyncio.to_thread(
                run_inference,
                payload.text,
                alpha=payload.alpha,
                ensemble_method=payload.ensemble_method,
            )
        return PredictResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Inference failed unexpectedly for payload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed due to an internal server error.",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch predict emotions from CSV",
    status_code=status.HTTP_200_OK,
)
async def predict_batch(
    file: UploadFile = File(...),
    alpha: float = Form(0.7),
    ensemble_method: str = Form("soft_voting"),
) -> BatchPredictResponse:
    """
    Upload a CSV file containing a 'text' column.
    Returns emotion predictions for every row (maximum 500 rows per batch).

    - **file**: CSV file with a column named 'text' (case-insensitive).
    - **alpha**: Blend weight (0.0 = pure XGBoost, 1.0 = pure RoBERTa).
    - **ensemble_method**: 'soft_voting' or 'stacking'.
    """
    if not is_model_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready to serve predictions. Please check /health/ready.",
        )

    # 1. Validate filename
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only valid .csv files are accepted.",
        )

    # 2. Validate alpha
    if not isinstance(alpha, (int, float)) or math.isnan(alpha) or math.isinf(alpha):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Alpha must be a finite numeric value.",
        )
    if alpha < 0.0 or alpha > 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Alpha must be between 0.0 and 1.0 (got {alpha}).",
        )

    # 3. Validate ensemble_method
    if ensemble_method not in ("soft_voting", "stacking"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ensemble_method must be either 'soft_voting' or 'stacking'.",
        )

    # 4. Stream and bound upload size (prevent OOM / DoS)
    chunks: List[bytes] = []
    total_bytes = 0
    chunk_size = 64 * 1024  # 64 KB

    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > config.MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE),
                    detail=(
                        f"Uploaded file exceeds the maximum size limit of "
                        f"{config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
                    ),
                )
            chunks.append(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV file is empty.",
            )

        raw_data = b"".join(chunks)

        # 5. Decode CSV safely
        try:
            text_content = raw_data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text_content = raw_data.decode("latin-1")
            except Exception as dec_exc:
                log.warning("Encoding error in CSV upload: %s", dec_exc)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is not a valid text-encoded CSV.",
                )

        try:
            df = pd.read_csv(io.StringIO(text_content), skip_blank_lines=False)
        except Exception as parse_exc:
            log.warning("CSV parse error: %s", parse_exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to parse CSV. Please verify that the file format is valid.",
            )

        # 6. Locate 'text' column
        text_col: Optional[str] = None
        for col in df.columns:
            if str(col).strip().lower() == "text":
                text_col = str(col)
                break

        if text_col is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV must contain a 'text' column. Found columns: {list(df.columns)}",
            )

        # 7. Check row limit
        if len(df) > config.MAX_BATCH_ROWS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV contains {len(df)} rows. Maximum allowed is {config.MAX_BATCH_ROWS}.",
            )

        raw_texts = df[text_col].fillna("").astype(str).tolist()

        # 8. Separate non-empty rows for batch inference
        valid_indices: List[int] = []
        valid_texts: List[str] = []

        for idx, t in enumerate(raw_texts):
            clean_t = t.strip()
            if clean_t:
                valid_indices.append(idx)
                # Enforce per-cell length bound
                valid_texts.append(clean_t[: config.MAX_CHARS])

        t0_batch = time.perf_counter()

        # Run vectorized batch inference on all valid texts in thread pool
        if valid_texts:
            async with _semaphore:
                batch_outputs = await asyncio.to_thread(
                    run_batch_inference,
                    valid_texts,
                    alpha=alpha,
                    ensemble_method=ensemble_method,
                )
        else:
            batch_outputs = []

        # Re-assemble results with exact 1-based row numbering
        results = [None] * len(raw_texts)
        valid_map = dict(zip(valid_indices, batch_outputs))

        for idx, t in enumerate(raw_texts):
            clean_t = t.strip()
            if idx in valid_map:
                res = valid_map[idx]
                results[idx] = {
                    "row": idx + 1,
                    "text": clean_t[:200],
                    "predicted_emotion": res["predicted_emotion"],
                    "confidence": res["confidence"],
                    "confidence_pct": res["confidence_pct"],
                    "probabilities": res["probabilities"],
                    "processing_time_ms": res["processing_time_ms"],
                }
            else:
                # Blank row representation
                results[idx] = {
                    "row": idx + 1,
                    "text": "",
                    "predicted_emotion": "unknown",
                    "confidence": 0.0,
                    "confidence_pct": "0.0%",
                    "probabilities": {},
                    "processing_time_ms": 0.0,
                }

        batch_time = (time.perf_counter() - t0_batch) * 1000

        return BatchPredictResponse(
            success=True,
            total_rows=len(results),
            alpha=round(alpha, 2),
            batch_processing_time_ms=round(batch_time, 2),
            predictions=results,
        )

    finally:
        await file.close()
