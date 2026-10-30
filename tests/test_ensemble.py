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


def test_dynamic_blend_probabilities():
    from app.services.inference import dynamic_blend_probabilities

    # 3 samples:
    # 1. RoBERTa is highly confident (0.90 joy) >= threshold 0.65 -> unchanged
    # 2. RoBERTa is uncertain (0.45 anger, 0.40 fear) < threshold 0.65 -> blended
    # 3. RoBERTa is moderately confident (0.70 sadness) >= 0.65 -> unchanged
    roberta_p = np.array([
        [0.02, 0.02, 0.90, 0.02, 0.02, 0.02],
        [0.45, 0.40, 0.05, 0.05, 0.03, 0.02],
        [0.05, 0.05, 0.10, 0.05, 0.70, 0.05],
    ])
    xgb_p = np.array([
        [0.10, 0.10, 0.10, 0.60, 0.05, 0.05],
        [0.10, 0.70, 0.10, 0.05, 0.03, 0.02],
        [0.20, 0.20, 0.20, 0.20, 0.10, 0.10],
    ])

    blended = dynamic_blend_probabilities(roberta_p, xgb_p, threshold=0.65, alpha=0.85)
    assert blended.shape == (3, 6)

    # Sample 0 should be exactly roberta_p normalized
    assert np.allclose(blended[0], roberta_p[0] / roberta_p[0].sum(), atol=1e-5)

    # Sample 1 should be weighted blend
    expected_1 = 0.85 * roberta_p[1] + 0.15 * xgb_p[1]
    expected_1 = expected_1 / expected_1.sum()
    assert np.allclose(blended[1], expected_1, atol=1e-5)

    # Sample 2 should be exactly roberta_p normalized
    assert np.allclose(blended[2], roberta_p[2] / roberta_p[2].sum(), atol=1e-5)


def test_single_inference_dynamic_blend(mock_models):
    res = run_inference("I am so delighted!", alpha=0.85, ensemble_method="dynamic_blend")

    assert res["success"] is True
    assert res["predicted_emotion"] in mock_models["label_encoder"].classes_
    assert len(res["probabilities"]) == 6
    assert np.isclose(sum(res["probabilities"].values()), 1.0, atol=1e-4)
