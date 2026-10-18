"""
tests/test_api_health.py — API tests for health, liveness, and readiness endpoints.
"""

from app.services import model_loader


def test_root_endpoint_ready(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["api"] == "Emotion Detection API"
    assert "emotion_classes" in data
    assert len(data["emotion_classes"]) == 6
    assert "endpoints" in data


def test_liveness_probe(client):
    res = client.get("/health/live")
    assert res.status_code == 200
    assert res.json() == {"status": "alive"}


def test_readiness_probe_ready(client):
    res = client.get("/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["state"] == "ready"


def test_readiness_probe_unready(client, monkeypatch):
    monkeypatch.setattr(model_loader, "_state", model_loader.ModelState.LOADING)
    res = client.get("/health/ready")
    assert res.status_code == 503
    data = res.json()
    assert data["status"] == "not_ready"
    assert data["state"] == "loading"
