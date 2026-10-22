# API Specification — EmoSense AI Emotion Detection API

**Base URL (local):** `http://localhost:8080` (or `http://localhost:7860/` in HF Spaces / Docker)

---

## 1. Endpoints Overview

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `GET` | `/` | Capability discovery & general health status (backward compatible) | `200` |
| `GET` | `/health/live` | Process liveness probe | `200` |
| `GET` | `/health/ready` | Model readiness probe | `200`, `503` |
| `POST` | `/predict` | Classify emotion in a single text string | `200`, `422`, `500`, `503` |
| `POST` | `/predict/batch` | Batch emotion prediction from an uploaded CSV file | `200`, `400`, `413`, `503` |
| `GET` | `/analytics/evaluation` | Empirical model evaluation metrics on held-out test data | `200` |
| `GET` | `/analytics/model-info` | Model architecture details and supported emotion classes | `200` |
| `POST` | `/analytics/summary` | Aggregate prediction records into KPI stats and trends | `200`, `422` |

---

## 2. Health & Readiness

### `GET /`
Returns general status, loaded emotion classes, and active inference device.

#### Response `200 OK`
```json
{
  "status": "online",
  "api": "Emotion Detection API",
  "version": "1.0.0",
  "device": "cpu",
  "emotion_classes": ["anger", "fear", "joy", "neutral", "sadness", "surprise"],
  "frontend": "/ui/index.html",
  "evaluation": "/ui/evaluation.html",
  "endpoints": {
    "health": "GET  /",
    "liveness": "GET  /health/live",
    "readiness": "GET  /health/ready",
    "predict": "POST /predict  { text: string }",
    "predict_batch": "POST /predict/batch  [multipart/form-data CSV]"
  }
}
```

### `GET /health/live`
Kubernetes / Docker liveness probe confirming the server process is responsive.

#### Response `200 OK`
```json
{ "status": "alive" }
```

### `GET /health/ready`
Readiness probe verifying that all ML models (RoBERTa, XGBoost, TF-IDF vectorizers, NRC scaler) are fully initialized and ready to serve traffic.

#### Response `200 OK`
```json
{
  "status": "ready",
  "state": "ready",
  "device": "cpu"
}
```

#### Response `503 Service Unavailable`
```json
{
  "status": "not_ready",
  "state": "loading",
  "detail": "Model artifacts are still initializing or failed to load."
}
```

---

## 3. Single Prediction

### `POST /predict`
Analyzes the emotion expressed in a text string.

#### Request Headers
```
Content-Type: application/json
```

#### Request Body
```json
{
  "text": "I am so happy and excited today! Everything is going perfectly!",
  "alpha": 0.7,
  "ensemble_method": "soft_voting"
}
```

| Field | Type | Required | Default | Bounds | Description |
|---|---|---|---|---|---|
| `text` | string | ✓ | — | 1 – 2000 chars | The text to classify. Non-empty, non-whitespace. |
| `alpha` | float | Optional | `0.7` | `0.0` – `1.0` | Soft voting weight: $\alpha \cdot P_{\text{RoBERTa}} + (1 - \alpha) \cdot P_{\text{XGBoost}}$. |
| `ensemble_method` | string | Optional | `"soft_voting"` | `"soft_voting"`, `"stacking"` | Ensemble strategy. |

#### Response `200 OK`
```json
{
  "success": true,
  "predicted_emotion": "joy",
  "confidence": 0.9401,
  "confidence_pct": "94.0%",
  "probabilities": {
    "anger": 0.0077,
    "fear": 0.0218,
    "joy": 0.9401,
    "neutral": 0.0024,
    "sadness": 0.0137,
    "surprise": 0.0142
  },
  "processing_time_ms": 78.4,
  "alpha": 0.7
}
```

#### Validation Error `422 Unprocessable Entity`
Returned when text is empty, exceeds 2000 characters, or alpha is outside $[0.0, 1.0]$.
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "alpha"],
      "msg": "Value error, Alpha must be between 0.0 and 1.0 (got 1.5).",
      "input": 1.5
    }
  ]
}
```

---

## 4. Batch Prediction

### `POST /predict/batch`
Uploads a CSV file containing a `text` column. Returns predictions for all rows (up to 500 rows).

#### Request Headers
```
Content-Type: multipart/form-data
```

#### Form Parameters
| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | file (`.csv`) | ✓ | — | CSV file with a column named `text` (case-insensitive). Max 5 MB. |
| `alpha` | float | Optional | `0.7` | Blend weight between 0.0 and 1.0. |
| `ensemble_method` | string | Optional | `"soft_voting"` | `"soft_voting"` or `"stacking"`. |

#### Response `200 OK`
```json
{
  "success": true,
  "total_rows": 3,
  "alpha": 0.7,
  "batch_processing_time_ms": 112.5,
  "predictions": [
    {
      "row": 1,
      "text": "I am so incredibly happy today!",
      "predicted_emotion": "joy",
      "confidence": 0.937,
      "confidence_pct": "93.7%",
      "probabilities": { "joy": 0.937, ... },
      "processing_time_ms": 37.5
    },
    {
      "row": 2,
      "text": "",
      "predicted_emotion": "unknown",
      "confidence": 0.0,
      "confidence_pct": "0.0%",
      "probabilities": {},
      "processing_time_ms": 0.0
    }
  ]
}
```

#### Error Responses
- **`400 Bad Request`**: Malformed CSV, missing `text` column, empty file, or row count exceeds 500.
- **`413 Content Too Large`**: Uploaded file exceeds the 5 MB ceiling.
- **`503 Service Unavailable`**: Models not initialized or currently loading.

---

## 5. Analytics & Insights

### `GET /analytics/evaluation`
Retrieves empirical evaluation metrics calculated directly from the held-out validation dataset (300 stratified test cases from `clean_emotion_dataset.csv`).

#### Response `200 OK`
```json
{
  "success": true,
  "summary": {
    "num_samples": 300,
    "timestamp": "2026-10-18 22:03:11",
    "device": "CPU",
    "dataset_name": "clean_emotion_dataset.csv",
    "split_description": "Stratified held-out validation sample (50 per class, seed=42, 300 total samples)."
  },
  "branches": {
    "xgb": { "accuracy": 91.3, "macro_f1": 91.5, "avg_latency_ms": 13.41 },
    "roberta": { "accuracy": 98.0, "macro_f1": 98.0, "avg_latency_ms": 275.59 },
    "ensemble": { "accuracy": 98.0, "macro_f1": 98.0, "avg_latency_ms": 275.72 }
  },
  "class_metrics": {
    "anger": { "precision": 100.0, "recall": 98.0, "f1": 99.0, "support": 50 },
    "fear": { "precision": 98.0, "recall": 96.0, "f1": 97.0, "support": 50 },
    "joy": { "precision": 98.0, "recall": 100.0, "f1": 99.0, "support": 50 },
    "neutral": { "precision": 96.0, "recall": 96.0, "f1": 96.0, "support": 50 },
    "sadness": { "precision": 96.2, "recall": 100.0, "f1": 98.0, "support": 50 },
    "surprise": { "precision": 100.0, "recall": 98.0, "f1": 99.0, "support": 50 }
  },
  "confusion_matrix": {
    "labels": ["anger", "fear", "joy", "neutral", "sadness", "surprise"],
    "matrix": [
      [98.0, 0.0, 0.0, 2.0, 0.0, 0.0],
      [0.0, 96.0, 0.0, 2.0, 2.0, 0.0],
      [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
      [0.0, 2.0, 0.0, 96.0, 2.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 100.0, 0.0],
      [0.0, 0.0, 2.0, 0.0, 0.0, 98.0]
    ]
  }
}
```

### `GET /analytics/model-info`
Returns architectural metadata, blend configurations, and standard emotion classes.

#### Response `200 OK`
```json
{
  "success": true,
  "model_name": "EmoSense AI Hybrid Ensemble",
  "branches": {
    "roberta": "Fine-tuned roberta-base sequence classification (contextual)",
    "xgboost": "XGBoost on word/char TF-IDF and 10 NRC Emotion Lexicon density features (semantic)",
    "ensemble": "Dynamic alpha-weighted soft voting (alpha * RoBERTa + (1-alpha) * XGBoost)"
  },
  "default_alpha": 0.7,
  "emotion_classes": ["anger", "fear", "joy", "neutral", "sadness", "surprise"]
}
```

### `POST /analytics/summary`
Calculates real aggregate KPI metrics, emotion distribution, confidence bands, and time-series trends from client prediction records.

#### Request Body
```json
{
  "records": [
    {
      "predicted_emotion": "joy",
      "confidence": 0.95,
      "processing_time_ms": 110.2,
      "timestamp": "2026-10-19T10:00:00Z"
    }
  ],
  "date_from": "2026-10-01T00:00:00Z",
  "date_to": "2026-10-31T23:59:59Z",
  "emotion_filter": "all"
}
```

#### Response `200 OK`
```json
{
  "success": true,
  "total_predictions": 1,
  "top_emotion": "joy",
  "top_emotion_pct": 100.0,
  "emotion_counts": {
    "anger": 0, "fear": 0, "joy": 1, "neutral": 0, "sadness": 0, "surprise": 0
  },
  "emotion_percentages": {
    "anger": 0.0, "fear": 0.0, "joy": 100.0, "neutral": 0.0, "sadness": 0.0, "surprise": 0.0
  },
  "avg_confidence": 0.95,
  "avg_latency_ms": 110.2,
  "confidence_bands": {
    "very_high": 1,
    "high": 0,
    "moderate": 0,
    "low": 0
  },
  "low_confidence_count": 0,
  "time_series": [
    { "bucket": "2026-10-19", "count": 1, "dominant_emotion": "joy" }
  ],
  "scope_description": "Aggregated 1 predictions across 6 emotion categories."
}
```

---

## 6. Emotion Classes & Encodings

| Index | Class | Emoji | Hex / HSL Color | Description |
|---|---|---|---|---|
| 0 | `anger` | 😡 | `#EF4444` / `hsl(5, 85%, 52%)` | Outrage, frustration, annoyance |
| 1 | `fear` | 😨 | `#A855F7` / `hsl(270, 70%, 55%)` | Panic, anxiety, dread |
| 2 | `joy` | 😄 | `#F59E0B` / `hsl(45, 95%, 58%)` | Happiness, delight, gratitude |
| 3 | `neutral` | 😐 | `#6B7280` / `hsl(215, 15%, 55%)` | Objective statements, matter-of-fact |
| 4 | `sadness` | 😢 | `#3B82F6` / `hsl(220, 80%, 55%)` | Grief, sorrow, disappointment |
| 5 | `surprise` | 😲 | `#10B981` / `hsl(190, 85%, 50%)` | Astonishment, unexpected shock |

---

## 6. Example cURL Commands

```bash
# Health check
curl http://localhost:8080/

# Liveness probe
curl http://localhost:8080/health/live

# Readiness probe
curl http://localhost:8080/health/ready

# Single Prediction (Soft Voting)
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I am so happy and excited today!", "alpha": 0.7}'

# Batch Prediction
curl -X POST http://localhost:8080/predict/batch \
     -F "file=@test_data.csv" \
     -F "alpha=0.7"
```
