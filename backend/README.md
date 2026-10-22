# 🧠 Emotion Detection API — Backend

A production-hardened **FastAPI** backend serving a hybrid multi-branch emotion classifier combining:
- 🤗 **RoBERTa** (fine-tuned) — deep contextual text semantics
- 🌲 **XGBoost** on word TF-IDF, character TF-IDF, and NRC Emotion Lexicon density features — semantic lexicon signal
- 📊 **Dynamic Alpha-Weighted Soft Voting** ($P = \alpha P_{\text{RoBERTa}} + (1 - \alpha) P_{\text{XGBoost}}$) with optional **Stacking Meta-Classifier** support

**Detects 6 standard emotion categories:** `anger` · `fear` · `joy` · `neutral` · `sadness` · `surprise`

---

## 📁 Folder Structure

```
backend/
├── app/                        # Application package
│   ├── __init__.py             # App factory (create_app), lifespan, CORS, error handling
│   ├── config.py               # Centralized configuration & environment variables
│   ├── routes/                 # API endpoint handlers
│   │   ├── __init__.py
│   │   ├── analytics.py        # GET /analytics/evaluation, GET /analytics/model-info, POST /analytics/summary
│   │   ├── health.py           # GET /, GET /health/live, GET /health/ready
│   │   └── predict.py          # POST /predict, POST /predict/batch
│   ├── schemas/                # Pydantic request & response validation schemas
│   │   ├── __init__.py
│   │   ├── analytics.py        # EvaluationMetricsResponse, PredictionStatsRequest, PredictionStatsResponse
│   │   └── predict.py          # PredictRequest, PredictResponse, BatchPredictResponse
│   ├── services/               # Core business & ML logic
│   │   ├── __init__.py
│   │   ├── inference.py        # Vectorized batch & single inference, probability normalization
│   │   └── model_loader.py     # Thread-safe model lifecycle, readiness, & HF Hub downloader
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       └── preprocessing.py    # Training-compatible clean_text & NRC lexicon feature extractor
├── model/                      # ML model artifacts (local or auto-downloaded)
│   ├── xgb.pkl                 # XGBoost classifier
│   ├── word_tfidf.pkl          # Word-level TF-IDF vectorizer
│   ├── char_tfidf.pkl          # Character-level TF-IDF vectorizer
│   ├── nrc_scaler.pkl          # StandardScaler for NRC density features
│   ├── label_encoder.pkl       # LabelEncoder mapping 6 classes
│   ├── meta_clf.pkl            # Stacking LogisticRegression meta-learner
│   ├── nrc_lexicon.txt         # NRC Emotion Lexicon v0.92
│   └── roberta_best/           # Fine-tuned RoBERTa transformer weights
├── scripts/                    # CLI evaluation & smoke-test scripts
│   ├── evaluate_model.py       # Empirical model evaluation generating frontend/metrics.json
│   └── test_inference.py       # End-to-end smoke test script
├── requirements.txt            # Constrained Python runtime dependencies
├── start_server.bat            # Windows startup script
└── README.md
```

---

## ⚡ Quick Start (Local)

### 1. Install Dependencies
```bash
# Recommended Python version: 3.10
python -m pip install -r backend/requirements.txt
```

> **Note on Dependencies:** `transformers<5.0.0` is pinned to maintain binary compatibility with PyTorch 2.x. `numpy<2.0.0` is constrained for PyTorch tensor compatibility.

### 2. Start the Server
```bash
# Windows bat runner
backend\start_server.bat

# Or manual execution via uvicorn factory
uvicorn app:create_app --factory --host 127.0.0.1 --port 8080
```

### 3. Verify Health & Readiness
```bash
# Liveness probe
curl http://127.0.0.1:8080/health/live

# Readiness probe
curl http://127.0.0.1:8080/health/ready
```

---

## 🔧 Configuration & Environment Variables

Settings can be customized via `.env` or direct environment variables:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind host address |
| `PORT` | `8080` | Bind port number |
| `CORS_ORIGINS` | `http://127.0.0.1:8080,...` | Comma-separated allowed origins (wildcard `*` disables credentials) |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MODEL_DIR` | `model` | Directory for model artifacts (relative to `backend/` or absolute) |
| `MAX_CHARS` | `2000` | Maximum allowed character length per input text |
| `MAX_BATCH_ROWS` | `500` | Maximum rows processed per CSV batch upload |
| `MAX_UPLOAD_SIZE_BYTES` | `5242880` (5 MB) | Maximum uploaded CSV file size before rejection |
| `MAX_CONCURRENT_INFERENCES` | `4` | Concurrency semaphore limit protecting CPU/GPU memory |
| `HF_MODEL_REPO` | `DSAditya552003/emosense-ai-model` | Hugging Face Hub repository for missing artifacts |
| `HF_MODEL_REVISION` | `main` | Pinned commit hash or branch on Hugging Face Hub |
| `HF_TOKEN` | `""` | Optional Hugging Face auth token for private repositories |

---

## 📡 API Reference Summary

- `GET /`: API overview and capability metadata.
- `GET /health/live`: Lightweight process liveness check (`200 OK`).
- `GET /health/ready`: Lifecycle readiness probe (`200 OK` or `503 Service Unavailable`).
- `POST /predict`: Single emotion analysis. Accepts JSON `{ "text": "...", "alpha": 0.7 }`. Returns predicted label, confidence, per-class probabilities, and latency.
- `POST /predict/batch`: Bounded multipart CSV batch classification. Vectorized inference across up to 500 rows.
- `GET /analytics/evaluation`: Verified empirical model evaluation metrics evaluated on held-out test data.
- `GET /analytics/model-info`: Architectural specifications, default blend parameters, and emotion classes.
- `POST /analytics/summary`: Aggregate stored prediction records into KPI cards, emotion distribution, confidence bands, and trend time-series.

---

## 🧪 Testing

The repository includes a comprehensive 52-test automated pytest suite:

```bash
# Run all unit, API, and analytics tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v
```

---

## 📄 Citation

NRC Emotion Lexicon:
> Mohammad, S. & Turney, P. (2013). *Crowdsourcing a Word-Emotion Association Lexicon.* Computational Intelligence, 29(3), 436–465.
