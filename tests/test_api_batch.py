"""
tests/test_api_batch.py — API tests for the POST /predict/batch CSV upload endpoint.
"""

import io
from app import config


def test_batch_valid_csv(client):
    csv_content = "text\nI am thrilled!\nI feel so sad and lonely.\nJust a normal Tuesday."
    res = client.post(
        "/predict/batch",
        files={"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
        data={"alpha": "0.7"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["total_rows"] == 3
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["row"] == 1
    assert data["predictions"][0]["predicted_emotion"] != ""


def test_batch_missing_text_column(client):
    csv_content = "message,id\nhello,1\nworld,2"
    res = client.post(
        "/predict/batch",
        files={"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
    )
    assert res.status_code == 400
    assert "must contain a 'text' column" in res.json()["detail"]


def test_batch_non_csv_extension(client):
    res = client.post(
        "/predict/batch",
        files={"file": ("test.txt", io.BytesIO(b"text\nhello"), "text/plain")},
    )
    assert res.status_code == 400
    assert "Only valid .csv files are accepted" in res.json()["detail"]


def test_batch_empty_file(client):
    res = client.post(
        "/predict/batch",
        files={"file": ("test.csv", io.BytesIO(b""), "text/csv")},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_batch_too_many_rows(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_BATCH_ROWS", 5)
    rows = ["text"] + [f"sample text row {i}" for i in range(10)]
    csv_content = "\n".join(rows)

    res = client.post(
        "/predict/batch",
        files={"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
    )
    assert res.status_code == 400
    assert "Maximum allowed is 5" in res.json()["detail"]


def test_batch_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_BYTES", 50)  # 50 bytes ceiling
    csv_content = "text\n" + ("very long text content exceeding fifty bytes " * 10)

    res = client.post(
        "/predict/batch",
        files={"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
    )
    assert res.status_code == 413
    assert "exceeds the maximum size limit" in res.json()["detail"]


def test_batch_blank_rows_handling(client):
    csv_content = "text\nI am happy!\n   \n\nI am scared."
    res = client.post(
        "/predict/batch",
        files={"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total_rows"] == 4
    # Row 2 was blank
    assert data["predictions"][1]["predicted_emotion"] == "unknown"
    assert data["predictions"][1]["confidence"] == 0.0
    # Row 4 was valid
    assert data["predictions"][3]["predicted_emotion"] != "unknown"


def test_batch_invalid_alpha(client):
    csv_content = "text\nHello"
    res = client.post(
        "/predict/batch",
        files={"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
        data={"alpha": "1.5"},
    )
    assert res.status_code == 400
    assert "Alpha must be between 0.0 and 1.0" in res.json()["detail"]
