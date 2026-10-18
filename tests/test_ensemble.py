"""
tests/test_ensemble.py — Unit tests for probability normalization, soft voting, stacking, and batch inference.
"""

import numpy as np
import pytest

from app.services.inference import _normalize_probabilities, run_batch_inference, run_inference


def test_normalize_probabilities_basic():
    raw = np.array([[0.1, 0.2, 0.7], [0.0, 0.0, 0.0], [np.nan, 0.5, 0.5]])
    norm = _normalize_probabilities(raw)

    assert norm.shape == (3, 3)
    # Check rows sum to 1.0
    for r in norm:
        assert np.isclose(np.sum(r), 1.0)

    # Row 1 (all zeros) becomes uniform
    assert np.allclose(norm[1], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def test_single_inference_soft_voting(mock_models):
    res = run_inference("I am so delighted!", alpha=0.7, ensemble_method="soft_voting")

    assert res["success"] is True
    assert res["predicted_emotion"] in mock_models["label_encoder"].classes_
    assert "probabilities" in res
    assert len(res["probabilities"]) == 6
    assert np.isclose(sum(res["probabilities"].values()), 1.0, atol=1e-4)
    assert res["alpha"] == 0.7


def test_single_inference_stacking(mock_models):
    res = run_inference("I am so delighted!", alpha=0.7, ensemble_method="stacking")

    assert res["success"] is True
    assert res["predicted_emotion"] in mock_models["label_encoder"].classes_
    assert len(res["probabilities"]) == 6
    assert np.isclose(sum(res["probabilities"].values()), 1.0, atol=1e-4)


def test_batch_inference_equivalence(mock_models):
    texts = [
        "I love this project!",
        "I am so terrified and scared.",
        "Today is an ordinary Monday.",
    ]

    # Run batch inference
    batch_res = run_batch_inference(texts, alpha=0.8, batch_size=2)
    assert len(batch_res) == 3

    # Compare each against single run_inference
    for i, t in enumerate(texts):
        single_res = run_inference(t, alpha=0.8)
        assert batch_res[i]["predicted_emotion"] == single_res["predicted_emotion"]
        for cls_name in single_res["probabilities"]:
            assert np.isclose(
                batch_res[i]["probabilities"][cls_name],
                single_res["probabilities"][cls_name],
                atol=1e-5,
            )


def test_emotion_class_order(mock_models):
    res = run_inference("Test text")
    expected_classes = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]
    assert list(res["probabilities"].keys()) == expected_classes
