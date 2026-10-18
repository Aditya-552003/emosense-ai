"""
tests/test_real_models.py — Real-model smoke tests that run against actual local artifacts.
"""

import os
import pytest

from app.config import MODEL_DIR
from app.services.model_loader import load_models, unload_models, is_model_ready
from app.services.inference import run_inference, run_batch_inference

has_artifacts = (
    os.path.exists(os.path.join(MODEL_DIR, "xgb.pkl"))
    and os.path.exists(os.path.join(MODEL_DIR, "word_tfidf.pkl"))
    and os.path.exists(os.path.join(MODEL_DIR, "meta_clf.pkl"))
    and os.path.exists(os.path.join(MODEL_DIR, "roberta_best", "model.safetensors"))
)


@pytest.mark.skipif(not has_artifacts, reason="Local model artifacts not present")
def test_real_model_smoke_test():
    """Verify loading real artifacts and running inference."""
    try:
        load_models()
        assert is_model_ready() is True

        res = run_inference("I am so genuinely pleased with these results!")
        assert res["success"] is True
        assert res["predicted_emotion"] == "joy"
        assert res["confidence"] > 0.5
        assert len(res["probabilities"]) == 6

        # Test batch inference equivalence
        batch = run_batch_inference(["I am so genuinely pleased with these results!"])
        assert batch[0]["predicted_emotion"] == res["predicted_emotion"]
    finally:
        unload_models()
