"""
tests/test_api_predict.py — API tests for the POST /predict endpoint.
"""


def test_predict_success(client):
    res = client.post(
        "/predict",
        json={"text": "I am so happy and excited today!", "alpha": 0.7},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "predicted_emotion" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 6
    assert data["alpha"] == 0.7


def test_predict_empty_text(client):
    res = client.post("/predict", json={"text": ""})
    assert res.status_code == 422
    assert "empty" in str(res.json()).lower()


def test_predict_whitespace_text(client):
    res = client.post("/predict", json={"text": "   \n\t  "})
    assert res.status_code == 422


def test_predict_oversized_text(client):
    res = client.post("/predict", json={"text": "a" * 2001})
    assert res.status_code == 422
    assert "exceeds" in str(res.json()).lower()


def test_predict_invalid_alpha(client):
    res = client.post("/predict", json={"text": "valid text", "alpha": -0.5})
    assert res.status_code == 422

    res2 = client.post("/predict", json={"text": "valid text", "alpha": 1.5})
    assert res2.status_code == 422


def test_predict_invalid_json(client):
    res = client.post(
        "/predict",
        content="malformed json{",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 422


def test_cors_headers_allowed_origin(client):
    # Default allowed origins include http://localhost:8080
    res = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.headers.get("access-control-allow-origin") == "http://localhost:8080"


def test_cors_headers_disallowed_origin(client):
    res = client.options(
        "/predict",
        headers={
            "Origin": "http://evil-attacker.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.headers.get("access-control-allow-origin") is None
