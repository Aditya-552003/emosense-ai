"""
tests/conftest.py — Pytest configuration and shared fixtures.
"""

import os
import sys
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure backend/ is in sys.path
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app import create_app
from app.services import model_loader


class DummyLabelEncoder:
    def __init__(self):
        self.classes_ = np.array(["anger", "fear", "joy", "neutral", "sadness", "surprise"])

    def inverse_transform(self, indices):
        return [self.classes_[i] for i in indices]


class DummyVectorizer:
    def __init__(self, n_features: int):
        self.n_features = n_features

    def transform(self, texts):
        from scipy.sparse import csr_matrix
        return csr_matrix(np.zeros((len(texts), self.n_features), dtype=np.float32))


class DummyScaler:
    def transform(self, feats):
        return feats


class DummyXGB:
    def __init__(self):
        self.classes_ = np.array([0, 1, 2, 3, 4, 5])
        self.n_features_in_ = 23010

    def predict_proba(self, x):
        n = x.shape[0]
        # Return deterministic mock probabilities: highest for 'joy' (class 2)
        probs = np.tile([0.05, 0.05, 0.70, 0.10, 0.05, 0.05], (n, 1))
        return probs


class DummyMetaClf:
    def __init__(self):
        self.classes_ = np.array([0, 1, 2, 3, 4, 5])
        self.n_features_in_ = 12

    def predict_proba(self, x):
        n = x.shape[0]
        probs = np.tile([0.02, 0.02, 0.85, 0.05, 0.03, 0.03], (n, 1))
        return probs


class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        import torch
        n = len(texts)
        return {
            "input_ids": torch.ones((n, 10), dtype=torch.long),
            "attention_mask": torch.ones((n, 10), dtype=torch.long),
        }


class DummyRoBERTa:
    def __init__(self):
        self.config = MagicMock()
        self.config.num_labels = 6

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        import torch
        input_ids = kwargs.get("input_ids")
        n = input_ids.shape[0] if input_ids is not None else 1
        # Logits giving highest probability to 'joy' (index 2)
        logits = torch.tile(torch.tensor([[0.0, 0.0, 5.0, 1.0, 0.0, 0.0]]), (n, 1))
        out = MagicMock()
        out.logits = logits
        return out


@pytest.fixture
def mock_models(monkeypatch):
    """Provide a mock model registry populated with verified dummy models."""
    registry = {
        "device": "cpu",
        "nrc_dict": {"happy": [4], "sad": [7]},
        "nrc_emotions": [
            "anger", "anticipation", "disgust", "fear", "joy",
            "negative", "positive", "sadness", "surprise", "trust",
        ],
        "word_tfidf": DummyVectorizer(10000),
        "char_tfidf": DummyVectorizer(13000),
        "nrc_scaler": DummyScaler(),
        "xgb": DummyXGB(),
        "meta_clf": DummyMetaClf(),
        "label_encoder": DummyLabelEncoder(),
        "tokenizer": DummyTokenizer(),
        "roberta": DummyRoBERTa(),
    }
    monkeypatch.setattr(model_loader, "_models", registry)
    monkeypatch.setattr(model_loader, "_state", model_loader.ModelState.READY)
    monkeypatch.setattr(model_loader, "_state_error", None)
    return registry


@pytest.fixture
def client(mock_models):
    """FastAPI TestClient with mock models initialized."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
