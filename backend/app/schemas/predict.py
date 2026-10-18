"""
schemas/predict.py — Pydantic models for the /predict endpoints.
"""

import math
from typing import Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Payload for single-text emotion prediction."""

    text: str = Field(
        ...,
        description="The sentence or paragraph to classify (1 to 2000 characters).",
        examples=["I am absolutely thrilled about this breakthrough!"],
    )
    alpha: float = Field(
        default=0.7,
        description="Blend weight between 0.0 (pure XGBoost) and 1.0 (pure RoBERTa).",
        examples=[0.7],
    )
    ensemble_method: Literal["soft_voting", "stacking"] = Field(
        default="soft_voting",
        description="Ensemble strategy: 'soft_voting' (default) or 'stacking'.",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Input text must be a string.")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Input text must not be empty or whitespace-only.")
        if len(stripped) > 2000:
            raise ValueError(f"Input text exceeds maximum allowed length of 2000 characters (got {len(stripped)}).")
        return stripped

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError("Alpha must be a finite numeric value.")
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Alpha must be between 0.0 and 1.0 (got {v}).")
        return float(v)


class PredictResponse(BaseModel):
    """Response payload for a single emotion prediction."""

    success: bool
    predicted_emotion: str
    confidence: float
    confidence_pct: str
    probabilities: Dict[str, float]
    processing_time_ms: float
    alpha: float


class BatchItemResult(BaseModel):
    """Individual row outcome inside a batch prediction response."""

    row: int
    text: str
    predicted_emotion: str
    confidence: float
    confidence_pct: str
    probabilities: Dict[str, float]
    processing_time_ms: float


class BatchPredictResponse(BaseModel):
    """Response payload for a batch CSV prediction."""

    success: bool
    total_rows: int
    alpha: float
    batch_processing_time_ms: float
    predictions: List[BatchItemResult]
