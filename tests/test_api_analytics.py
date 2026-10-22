"""
tests/test_api_analytics.py — Tests for analytics and evaluation endpoints.
"""

from datetime import datetime, timezone
import pytest


def test_get_evaluation_metrics(client):
    """Verify /analytics/evaluation returns valid schema and metrics."""
    res = client.get("/analytics/evaluation")
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert "summary" in data
    assert data["summary"]["num_samples"] >= 50
    assert "branches" in data
    assert "xgb" in data["branches"]
    assert "roberta" in data["branches"]
    assert "ensemble" in data["branches"]

    for b_name in ["xgb", "roberta", "ensemble"]:
        branch = data["branches"][b_name]
        assert 0 <= branch["accuracy"] <= 100
        assert 0 <= branch["macro_f1"] <= 100
        assert branch["avg_latency_ms"] >= 0

    assert "class_metrics" in data
    for cls in ["anger", "fear", "joy", "neutral", "sadness", "surprise"]:
        assert cls in data["class_metrics"]
        cm = data["class_metrics"][cls]
        assert 0 <= cm["precision"] <= 100
        assert 0 <= cm["recall"] <= 100
        assert 0 <= cm["f1"] <= 100
        assert cm["support"] >= 0

    assert "confusion_matrix" in data
    labels = data["confusion_matrix"]["labels"]
    matrix = data["confusion_matrix"]["matrix"]
    assert len(labels) == 6
    assert len(matrix) == 6
    for row in matrix:
        assert len(row) == 6


def test_get_model_info(client):
    """Verify /analytics/model-info returns architecture details and classes."""
    res = client.get("/analytics/model-info")
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert "branches" in data
    assert len(data["emotion_classes"]) == 6
    assert data["default_alpha"] == 0.7


def test_analytics_summary_empty_records(client):
    """Verify summary with zero records returns safe empty zero state without errors."""
    payload = {"records": []}
    res = client.post("/analytics/summary", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["total_predictions"] == 0
    assert data["top_emotion"] is None
    assert data["top_emotion_pct"] == 0.0
    assert data["avg_confidence"] == 0.0
    assert data["avg_latency_ms"] == 0.0
    assert data["low_confidence_count"] == 0
    assert len(data["time_series"]) == 0
    for cls in ["anger", "fear", "joy", "neutral", "sadness", "surprise"]:
        assert data["emotion_counts"][cls] == 0
        assert data["emotion_percentages"][cls] == 0.0


def test_analytics_summary_known_distribution(client):
    """Verify exact count, percentage, confidence, and latency calculations."""
    records = [
        {"predicted_emotion": "joy", "confidence": 0.95, "processing_time_ms": 100.0, "timestamp": "2026-10-18T10:00:00Z"},
        {"predicted_emotion": "joy", "confidence": 0.85, "processing_time_ms": 120.0, "timestamp": "2026-10-18T11:00:00Z"},
        {"predicted_emotion": "sadness", "confidence": 0.65, "processing_time_ms": 140.0, "timestamp": "2026-10-19T10:00:00Z"},
        {"predicted_emotion": "anger", "confidence": 0.55, "processing_time_ms": 160.0, "timestamp": "2026-10-19T11:00:00Z"},
    ]

    res = client.post("/analytics/summary", json={"records": records})
    assert res.status_code == 200
    data = res.json()

    assert data["total_predictions"] == 4
    assert data["top_emotion"] == "joy"
    assert data["top_emotion_pct"] == 50.0

    assert data["emotion_counts"]["joy"] == 2
    assert data["emotion_counts"]["sadness"] == 1
    assert data["emotion_counts"]["anger"] == 1
    assert data["emotion_counts"]["neutral"] == 0

    assert data["emotion_percentages"]["joy"] == 50.0
    assert data["emotion_percentages"]["sadness"] == 25.0
    assert data["emotion_percentages"]["anger"] == 25.0

    # (0.95 + 0.85 + 0.65 + 0.55) / 4 = 3.0 / 4 = 0.75
    assert data["avg_confidence"] == 0.75
    # (100 + 120 + 140 + 160) / 4 = 130.0
    assert data["avg_latency_ms"] == 130.0

    # Confidence bands: 0.95 -> very_high, 0.85 -> high, 0.65 -> moderate, 0.55 -> low
    assert data["confidence_bands"]["very_high"] == 1
    assert data["confidence_bands"]["high"] == 1
    assert data["confidence_bands"]["moderate"] == 1
    assert data["confidence_bands"]["low"] == 1
    assert data["low_confidence_count"] == 1

    # Time series (2 buckets: 2026-10-18 and 2026-10-19)
    assert len(data["time_series"]) == 2
    b1 = data["time_series"][0]
    assert b1["bucket"] == "2026-10-18"
    assert b1["count"] == 2
    assert b1["dominant_emotion"] == "joy"


def test_analytics_summary_emotion_filter(client):
    """Verify emotion filtering isolates only matching prediction records."""
    records = [
        {"predicted_emotion": "joy", "confidence": 0.90},
        {"predicted_emotion": "fear", "confidence": 0.85},
        {"predicted_emotion": "joy", "confidence": 0.80},
    ]

    res = client.post("/analytics/summary", json={"records": records, "emotion_filter": "joy"})
    assert res.status_code == 200
    data = res.json()

    assert data["total_predictions"] == 2
    assert data["top_emotion"] == "joy"
    assert data["emotion_counts"]["joy"] == 2
    assert data["emotion_counts"]["fear"] == 0


def test_analytics_summary_date_filter(client):
    """Verify date filtering properly respects boundaries."""
    records = [
        {"predicted_emotion": "joy", "confidence": 0.90, "timestamp": "2026-10-10T00:00:00+00:00"},
        {"predicted_emotion": "anger", "confidence": 0.80, "timestamp": "2026-10-15T00:00:00+00:00"},
        {"predicted_emotion": "sadness", "confidence": 0.70, "timestamp": "2026-10-20T00:00:00+00:00"},
    ]

    # Filter between Oct 12 and Oct 18
    res = client.post(
        "/analytics/summary",
        json={
            "records": records,
            "date_from": "2026-10-12T00:00:00+00:00",
            "date_to": "2026-10-18T00:00:00+00:00",
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["total_predictions"] == 1
    assert data["top_emotion"] == "anger"


def test_analytics_summary_missing_optional_fields(client):
    """Verify records missing latency or timestamps do not cause crashes."""
    records = [
        {"predicted_emotion": "joy", "confidence": 0.90},
        {"predicted_emotion": "neutral", "confidence": 0.75, "processing_time_ms": 50.0},
    ]

    res = client.post("/analytics/summary", json={"records": records})
    assert res.status_code == 200
    data = res.json()

    assert data["total_predictions"] == 2
    assert data["avg_latency_ms"] == 50.0  # Only computed over records with latency
