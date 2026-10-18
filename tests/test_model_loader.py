"""
tests/test_model_loader.py — Unit tests for model lifecycle management.
"""

import pytest

from app.services import model_loader


def test_model_state_initial(monkeypatch):
    monkeypatch.setattr(model_loader, "_models", {})
    monkeypatch.setattr(model_loader, "_state", model_loader.ModelState.UNINITIALIZED)
    monkeypatch.setattr(model_loader, "_state_error", None)

    assert model_loader.is_model_ready() is False
    assert model_loader.get_model_state() == model_loader.ModelState.UNINITIALIZED

    with pytest.raises(RuntimeError) as exc:
        model_loader.get_models()
    assert "not ready" in str(exc.value)


def test_model_unload(monkeypatch):
    monkeypatch.setattr(model_loader, "_models", {"dummy": 1})
    monkeypatch.setattr(model_loader, "_state", model_loader.ModelState.READY)

    model_loader.unload_models()

    assert model_loader.is_model_ready() is False
    assert model_loader.get_model_state() == model_loader.ModelState.UNINITIALIZED
    assert len(model_loader._models) == 0
