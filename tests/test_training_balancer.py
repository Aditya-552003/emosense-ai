"""
tests/test_training_balancer.py
===============================
Unit tests for the model training rebalancing and class weighting utilities.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from app.utils.training_balancer import (
    compute_pytorch_class_weights,
    compute_xgboost_sample_weights,
    rebalance_dataset,
)


def test_compute_pytorch_class_weights_basic():
    labels = ["neutral"] * 70 + ["anger"] * 10 + ["fear"] * 5 + ["joy"] * 5 + ["sadness"] * 5 + ["surprise"] * 5
    classes = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]

    weights = compute_pytorch_class_weights(labels, classes=classes)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (6,)
    assert (weights > 0).all()

    # Minority classes (fear=5) must have higher weight than majority class (neutral=70)
    neutral_idx = classes.index("neutral")
    fear_idx = classes.index("fear")
    assert weights[fear_idx] > weights[neutral_idx]


def test_compute_xgboost_sample_weights():
    labels = ["neutral"] * 100 + ["fear"] * 10
    sample_weights = compute_xgboost_sample_weights(labels)

    assert isinstance(sample_weights, np.ndarray)
    assert len(sample_weights) == 110
    # Weights for minority items must be higher than for majority items
    assert sample_weights[-1] > sample_weights[0]


def test_rebalance_dataset_undersampling():
    df = pd.DataFrame({
        "text": [f"text_{i}" for i in range(120)],
        "emotion": ["neutral"] * 100 + ["anger"] * 10 + ["joy"] * 10,
    })

    rebalanced = rebalance_dataset(df, max_majority_ratio=2.0, min_samples_per_class=10)
    assert isinstance(rebalanced, pd.DataFrame)
    assert "text" in rebalanced.columns
    assert "emotion" in rebalanced.columns

    # Median minority count is 10, so neutral should be capped at 10 * 2.0 = 20
    counts = rebalanced["emotion"].value_counts()
    assert counts["neutral"] <= 20
    assert counts["anger"] >= 10
    assert counts["joy"] >= 10


def test_rebalance_dataset_oversampling_minority():
    df = pd.DataFrame({
        "text": [f"text_{i}" for i in range(25)],
        "emotion": ["neutral"] * 20 + ["fear"] * 5,
    })

    rebalanced = rebalance_dataset(df, min_samples_per_class=15)
    counts = rebalanced["emotion"].value_counts()
    # Fear was 5, should be elevated to at least 15
    assert counts["fear"] >= 15
