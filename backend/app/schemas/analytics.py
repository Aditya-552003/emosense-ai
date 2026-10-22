"""
schemas/analytics.py — Pydantic models for evaluation and analytics endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BranchStats(BaseModel):
    """Evaluation summary metrics for a specific model branch."""
    accuracy: float = Field(..., description="Accuracy percentage (0–100).", ge=0.0, le=100.0)
    macro_f1: float = Field(..., description="Macro F1-score percentage (0–100).", ge=0.0, le=100.0)
    avg_latency_ms: float = Field(..., description="Average inference latency in milliseconds.", ge=0.0)


class ClassMetric(BaseModel):
    """Class-specific performance metrics on labeled evaluation data."""
    precision: float = Field(..., description="Precision percentage.", ge=0.0, le=100.0)
    recall: float = Field(..., description="Recall percentage.", ge=0.0, le=100.0)
    f1: float = Field(..., description="F1-score percentage.", ge=0.0, le=100.0)
    support: int = Field(..., description="Number of true instances in the evaluation split.", ge=0)


class ConfusionMatrixData(BaseModel):
    """Normalized confusion matrix for evaluation."""
    labels: List[str] = Field(..., description="Ordered list of emotion class labels.")
    matrix: List[List[float]] = Field(..., description="Normalized percentage values (rows=actual, cols=predicted).")


class DatasetSummary(BaseModel):
    """Metadata describing the held-out validation dataset and evaluation environment."""
    num_samples: int = Field(..., description="Number of evaluation samples evaluated.")
    timestamp: str = Field(..., description="Timestamp when evaluation was executed.")
    device: str = Field(..., description="Hardware compute device (e.g. CPU, CUDA).")
    dataset_name: str = Field(default="clean_emotion_dataset.csv", description="Name of source dataset.")
    split_description: str = Field(
        default="Stratified sampling (50 samples per class, seed=42) from held-out validation data.",
        description="Dataset split and sampling methodology.",
    )


class EvaluationMetricsResponse(BaseModel):
    """Full empirical model evaluation response."""
    success: bool = True
    summary: DatasetSummary
    branches: Dict[str, BranchStats]
    class_metrics: Dict[str, ClassMetric]
    confusion_matrix: ConfusionMatrixData


class PredictionRecordInput(BaseModel):
    """A single stored prediction record submitted for aggregation."""
    predicted_emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: Optional[float] = Field(default=None, ge=0.0)
    timestamp: Optional[str] = None
    source: Optional[str] = "single"


class PredictionStatsRequest(BaseModel):
    """Request payload to calculate analytics over a set of prediction records."""
    records: List[PredictionRecordInput] = Field(default_factory=list)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    emotion_filter: Optional[str] = None


class TimeBucket(BaseModel):
    """Time-series aggregation bucket for prediction trends."""
    bucket: str
    count: int
    dominant_emotion: Optional[str] = None


class PredictionStatsResponse(BaseModel):
    """Aggregated statistics derived from genuine prediction records."""
    success: bool = True
    total_predictions: int
    top_emotion: Optional[str] = None
    top_emotion_pct: float = 0.0
    emotion_counts: Dict[str, int]
    emotion_percentages: Dict[str, float]
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    confidence_bands: Dict[str, int] = Field(
        description="Counts in bands: very_high (>=90%), high (80-89%), moderate (60-79%), low (<60%)."
    )
    low_confidence_count: int = 0
    time_series: List[TimeBucket] = Field(default_factory=list)
    scope_description: str = "Client prediction records"
