"""
app/utils/training_balancer.py
==============================
Utilities to mitigate severe class imbalance during model training
and fine-tuning across all 6 emotion categories.

Provides:
  1. compute_pytorch_class_weights: Balanced tensor weights for CrossEntropyLoss.
  2. compute_xgboost_sample_weights: Sample weight array for XGBClassifier.fit.
  3. rebalance_dataset: Stratified down-sampling of majority classes (neutral)
     and up-sampling of minority classes (fear, surprise).
  4. augment_minority_texts: Lightweight rule-based text augmentation
     for low-frequency emotion categories.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


def compute_pytorch_class_weights(
    labels: Union[List[str], np.ndarray, pd.Series],
    classes: Optional[List[str]] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute balanced class weights for PyTorch CrossEntropyLoss.
    Penalizes misclassifications on minority classes (e.g. fear, surprise)
    proportionally to their inverse frequency.

    Args:
        labels: 1D array-like of true class labels.
        classes: Ordered list of all class names.
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        torch.Tensor of shape (n_classes,) suitable for nn.CrossEntropyLoss(weight=...)
    """
    labels_arr = np.array(labels)
    if classes is None:
        unique_classes = np.unique(labels_arr)
    else:
        unique_classes = np.array(classes)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels_arr,
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_xgboost_sample_weights(
    labels: Union[List[str], np.ndarray, pd.Series],
) -> np.ndarray:
    """
    Compute sample weights for XGBoost training:
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    Args:
        labels: 1D array-like of class labels for the training set.

    Returns:
        np.ndarray of shape (len(labels),)
    """
    labels_arr = np.array(labels)
    return compute_sample_weight("balanced", labels_arr)


def rebalance_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    emotion_col: str = "emotion",
    max_majority_ratio: float = 2.5,
    min_samples_per_class: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Rebalance an imbalanced dataset:
      - Caps the dominant class (e.g. neutral) to at most `max_majority_ratio`
        times the median minority class size.
      - Stratified bootstrap sampling to elevate minority classes (e.g. fear)
        to at least `min_samples_per_class`.

    Args:
        df: Input DataFrame containing text and emotion columns.
        text_col: Name of the text column.
        emotion_col: Name of the emotion column.
        max_majority_ratio: Max multiplier for majority class relative to median.
        min_samples_per_class: Floor count for minority classes.
        random_state: Random seed for reproducibility.

    Returns:
        Rebalanced pd.DataFrame.
    """
    rng = np.random.RandomState(random_state)
    counts = df[emotion_col].value_counts()
    median_count = int(counts.median())
    majority_cap = int(median_count * max_majority_ratio)

    balanced_subsets = []
    for emo, group in df.groupby(emotion_col):
        n = len(group)
        if n > majority_cap:
            # Under-sample majority
            sampled = group.sample(n=majority_cap, random_state=rng)
        elif n < min_samples_per_class:
            # Over-sample minority with replacement
            sampled = group.sample(n=min_samples_per_class, replace=True, random_state=rng)
        else:
            sampled = group.copy()
        balanced_subsets.append(sampled)

    result = pd.concat(balanced_subsets, ignore_index=True)
    return result.sample(frac=1.0, random_state=rng).reset_index(drop=True)
