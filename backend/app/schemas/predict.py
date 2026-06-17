"""
schemas/predict.py — Pydantic models for the /predict endpoints.
"""

from typing import Dict

from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    """Payload for single-text emotion prediction."""

    text: str
    alpha: float = 0.7  # Blend weight: alpha * RoBERTa + (1-alpha) * XGBoost

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Input text must not be empty.")
        if len(v) > 2000:
            raise ValueError("Input text must not exceed 2000 characters.")
        return v.strip()

    @field_validator("alpha")
    @classmethod
    def alpha_in_range(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class EmotionProbability(BaseModel):
    """Individual emotion probability entry."""

    emotion: str
    probability: float
    percentage: str


class PredictResponse(BaseModel):
    """Response payload for a single emotion prediction."""

    success: bool
    predicted_emotion: str
    confidence: float
    confidence_pct: str
    probabilities: Dict[str, float]
    processing_time_ms: float
    alpha: float
