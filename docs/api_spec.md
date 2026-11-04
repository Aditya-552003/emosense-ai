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
| `ensemble_method` | string | Optional | `"soft_voting"` | `"soft_voting"`, `"stacking"`, `"dynamic_blend"` | Ensemble strategy (`dynamic_blend` uses confidence-gating). |
| `confidence_threshold` | float | Optional | `0.65` | `0.0` – `1.0` | Cutoff for `dynamic_blend` (trusts RoBERTa if $\ge$ threshold). |

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
| `ensemble_method` | string | Optional | `"soft_voting"` | `"soft_voting"`, `"stacking"`, or `"dynamic_blend"`. |

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
Retrieves empirical evaluation metrics calculated directly from the validation dataset (17,484 eligible test cases from `clean_emotion_dataset.csv`).

#### Response `200 OK`
```json
{
  "success": true,
  "summary": {
    "num_samples": 17484,
    "timestamp": "2026-10-30 12:09:14",
    "device": "CPU",
    "dataset_name": "clean_emotion_dataset.csv",
    "split_description": "Empirical evaluation on 17,484 eligible samples across 6 emotion classes."
  },
  "branches": {
    "xgb": { "accuracy": 84.8, "macro_f1": 78.9, "avg_latency_ms": 0.25 },
    "roberta": { "accuracy": 96.3, "macro_f1": 93.9, "avg_latency_ms": 41.37 },
    "ensemble": { "accuracy": 95.5, "macro_f1": 92.6, "avg_latency_ms": 41.62 }
  },
  "class_metrics": {
    "anger": { "precision": 85.4, "recall": 97.9, "f1": 91.2, "support": 1544 },
    "fear": { "precision": 79.8, "recall": 98.4, "f1": 88.1, "support": 510 },
    "joy": { "precision": 89.6, "recall": 98.6, "f1": 93.9, "support": 1010 },
    "neutral": { "precision": 99.6, "recall": 94.4, "f1": 96.9, "support": 12798 },
    "sadness": { "precision": 89.2, "recall": 98.9, "f1": 93.8, "support": 873 },
    "surprise": { "precision": 85.3, "recall": 99.7, "f1": 91.9, "support": 749 }
  },
  "confusion_matrix": {
    "labels": ["anger", "fear", "joy", "neutral", "sadness", "surprise"],
    "matrix": [
      [97.9, 0.1, 0.1, 1.8, 0.1, 0.1],
      [0.0, 98.4, 0.0, 1.0, 0.6, 0.0],
      [0.2, 0.0, 98.6, 1.0, 0.0, 0.2],
      [2.0, 1.0, 0.9, 94.4, 0.8, 1.0],
      [0.2, 0.0, 0.0, 0.9, 98.9, 0.0],
      [0.0, 0.0, 0.1, 0.1, 0.0, 99.7]
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
