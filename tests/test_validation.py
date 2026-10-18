"""
tests/test_validation.py — Unit tests for Pydantic input/output validation.
"""

import pytest
from pydantic import ValidationError

from app.schemas.predict import PredictRequest, PredictResponse


def test_predict_request_valid():
    req = PredictRequest(text="I am thrilled about this!", alpha=0.7)
    assert req.text == "I am thrilled about this!"
    assert req.alpha == 0.7
    assert req.ensemble_method == "soft_voting"


def test_predict_request_stacking_method():
    req = PredictRequest(text="Great news!", alpha=0.5, ensemble_method="stacking")
    assert req.ensemble_method == "stacking"


def test_predict_request_empty_text():
    with pytest.raises(ValidationError) as exc:
        PredictRequest(text="")
    assert "Input text must not be empty" in str(exc.value)


def test_predict_request_whitespace_only():
    with pytest.raises(ValidationError) as exc:
        PredictRequest(text="   \n\t   ")
    assert "Input text must not be empty" in str(exc.value)


def test_predict_request_oversized_text():
    oversized = "a" * 2001
    with pytest.raises(ValidationError) as exc:
        PredictRequest(text=oversized)
    assert "exceeds maximum allowed length" in str(exc.value)


def test_predict_request_max_chars_exact():
    exact = "a" * 2000
    req = PredictRequest(text=exact)
    assert len(req.text) == 2000


def test_predict_request_alpha_bounds():
    # Valid boundaries
    req0 = PredictRequest(text="hello", alpha=0.0)
    assert req0.alpha == 0.0

    req1 = PredictRequest(text="hello", alpha=1.0)
    assert req1.alpha == 1.0

    # Negative alpha must be rejected
    with pytest.raises(ValidationError) as exc:
        PredictRequest(text="hello", alpha=-0.1)
    assert "Alpha must be between 0.0 and 1.0" in str(exc.value)

    # Alpha > 1.0 must be rejected
    with pytest.raises(ValidationError) as exc:
        PredictRequest(text="hello", alpha=1.05)
    assert "Alpha must be between 0.0 and 1.0" in str(exc.value)


def test_predict_request_alpha_nan_and_inf():
    with pytest.raises(ValidationError) as exc:
        PredictRequest(text="hello", alpha=float("nan"))
    assert "finite numeric value" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        PredictRequest(text="hello", alpha=float("inf"))
    assert "finite numeric value" in str(exc.value)


def test_predict_request_invalid_ensemble_method():
    with pytest.raises(ValidationError):
        PredictRequest(text="hello", ensemble_method="unsupported_method")


def test_predict_response_schema():
    resp = PredictResponse(
        success=True,
        predicted_emotion="joy",
        confidence=0.95,
        confidence_pct="95.0%",
        probabilities={"joy": 0.95, "neutral": 0.05},
        processing_time_ms=12.5,
        alpha=0.7,
    )
    assert resp.success is True
    assert resp.predicted_emotion == "joy"
    assert resp.confidence == 0.95
