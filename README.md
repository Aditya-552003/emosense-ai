---
title: EmoSense AI
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
---

# 🧠 EmoSense AI — Emotion Detection System

A production-hardened AI emotion detection platform powered by a **hybrid ensemble** combining **fine-tuned RoBERTa** (deep contextual understanding) with **XGBoost + NRC Emotion Lexicon** (semantic signal & word/char TF-IDF), blended via dynamic alpha-weighted soft voting or stacking meta-classification.

**Detects 6 standard emotion categories:** `anger` · `fear` · `joy` · `neutral` · `sadness` · `surprise`

---

## 📁 Project Structure

```
emotion_model/
├── app.py                      # HF Spaces Gradio & FastAPI entry point
├── requirements.txt            # Root dependencies (Hugging Face Spaces)
├── Dockerfile                  # Production container definition (non-root)
├── clean_emotion_dataset.csv   # Validation dataset (17,518 samples)
├── backend/                    # FastAPI backend
│   ├── app/                    # Application package
│   │   ├── __init__.py         # App factory (create_app), lifespan, CORS, error handling
│   │   ├── config.py           # Configuration & environment variable resolution
│   │   ├── routes/             # API endpoint routers
│   │   │   ├── health.py       # GET /, GET /health/live, GET /health/ready
│   │   │   └── predict.py      # POST /predict, POST /predict/batch
│   │   ├── schemas/            # Strict Pydantic input/output schemas
│   │   │   └── predict.py      # PredictRequest, PredictResponse, BatchPredictResponse
│   │   ├── services/           # ML & business logic
│   │   │   ├── inference.py    # Vectorized batch & single inference, probability normalization
│   │   │   └── model_loader.py # Thread-safe model lifecycle & artifact manager
│   │   └── utils/              # Shared NLP utilities
│   │       └── preprocessing.py# clean_text & NRC lexicon feature extractor
│   ├── model/                  # Trained ML artifacts
│   │   ├── xgb.pkl             # Trained XGBoost model
│   │   ├── word_tfidf.pkl      # Word-level TF-IDF vectorizer
│   │   ├── char_tfidf.pkl      # Char-level TF-IDF vectorizer
│   │   ├── nrc_scaler.pkl      # StandardScaler for NRC features
│   │   ├── label_encoder.pkl   # LabelEncoder (6 emotion classes)
│   │   ├── meta_clf.pkl        # Stacking LogisticRegression meta-classifier
│   │   ├── nrc_lexicon.txt     # NRC Word-Level Emotion Lexicon
│   │   └── roberta_best/       # Fine-tuned RoBERTa transformer weights
│   ├── scripts/                # Evaluation & CLI test tools
│   │   ├── evaluate_model.py   # Empirical evaluation generating frontend/metrics.json
│   │   └── test_inference.py   # Smoke test script
│   ├── requirements.txt        # Backend dependencies
│   └── start_server.bat        # Windows server launcher
├── frontend/                   # Browser UI
│   ├── css/                    # Stylesheets (modern responsive dark/light mode)
│   ├── js/                     # Modular ES6 JavaScript
│   ├── index.html              # Interactive single & batch analyzer
│   ├── evaluation.html         # Live empirical performance dashboard
│   └── metrics.json            # Empirical model metrics
├── tests/                      # Automated pytest suite (45 tests)
│   ├── conftest.py             # Fixtures and mock components
│   ├── test_validation.py      # Pydantic schema validation tests
│   ├── test_preprocessing.py   # Preprocessing & NRC extraction unit tests
│   ├── test_ensemble.py        # Soft voting, stacking, & batch equivalence
│   ├── test_model_loader.py    # Model lifecycle & error states
│   ├── test_api_health.py      # Health, liveness, & readiness probes
│   ├── test_api_predict.py     # Single prediction API & CORS tests
│   ├── test_api_batch.py       # CSV upload bounds, parsing, & row limits
│   └── test_real_models.py     # Real artifact smoke tests
├── docs/                       # Technical documentation
│   ├── CODE_AUDIT.md           # Full audit report & baseline findings
│   ├── TEST_REPORT.md          # Complete test execution & empirical report
│   └── api_spec.md             # REST API specification
├── .github/workflows/
│   ├── ci.yml                  # Automated CI test pipeline
│   └── hf_sync.yml             # Automatic sync to Hugging Face Spaces
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Python Dependencies
```bash
# Recommended Python version: 3.10
python -m pip install -r backend/requirements.txt
```

> **Note on Compatibility:** `transformers<5.0.0` is pinned to maintain binary compatibility with PyTorch 2.x without triggering runtime backend disables.

### 2. Start the Backend Server
```bash
# On Windows
backend\start_server.bat

# Or manually via uvicorn factory
cd backend
uvicorn app:create_app --factory --host 127.0.0.1 --port 8080
```

### 3. Open the Frontend
Open **http://127.0.0.1:8080/ui/index.html** in your browser.

---

## 🏗️ Architecture

```
                       Raw Text Input
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌───────────────────────┐   ┌────────────────────────┐
   │    Semantic Branch    │   │   Contextual Branch    │
   │  clean_text()         │   │  Raw Text Tokenization │
   │  Word TF-IDF (10,000) │   │  Fine-Tuned RoBERTa    │
   │  Char TF-IDF (13,000) │   │  torch.inference_mode()│
   │  NRC Lexicon (10-dim) │   │  Softmax Probabilities │
   │  → XGBClassifier      │   │  (6 Classes)           │
   └──────────┬────────────┘   └───────────┬────────────┘
              │ P(XGB)                     │ P(RoBERTa)
              └──────────────┬─────────────┘
                             ▼
              ┌─────────────────────────────┐
              │    Hybrid Ensemble Layer    │
              │                             │
              │  Mode 1 (Deployed Default): │
              │    Alpha-Weighted Soft Vote │
              │    P = α·P_rob + (1-α)·P_xgb│
              │                             │
              │  Mode 2 (Stacking Meta):    │
              │    LogisticRegression(12)   │
              │    consumed via meta_clf    │
              └──────────────┬──────────────┘
                             ▼
                    Probability Norm &
                    Argmax Emotion Label
```

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Capability discovery & API status |
| `GET` | `/health/live` | Process liveness probe (`200 OK`) |
| `GET` | `/health/ready` | Model lifecycle readiness probe (`200 OK` or `503 Unavailable`) |
| `POST` | `/predict` | Single emotion prediction (max 2000 chars, alpha 0.0–1.0) |
| `POST` | `/predict/batch` | Batch CSV prediction (max 500 rows, max 5 MB upload) |

Full documentation: [docs/api_spec.md](docs/api_spec.md)

---

## 🧪 Automated Testing

Run the full pytest suite (45 tests covering unit, API, security, and integration):

```bash
# Run complete test suite
python -m pytest tests/ -v

# Run without loading heavy real model weights
python -m pytest tests/ -v -m "not real_model"
```

---

## 🐳 Docker Deployment

Build and run the hardened production container (running as non-root `appuser` with container healthchecks):

```bash
docker build -t emosense-ai .
docker run -p 7860:7860 emosense-ai
```

Access the UI at `http://localhost:7860/ui/index.html` and Gradio at `http://localhost:7860/gradio`.

---

## 📄 License & Attribution

NRC Emotion Lexicon:
> Mohammad, S. & Turney, P. (2013). *Crowdsourcing a Word-Emotion Association Lexicon.* Computational Intelligence, 29(3), 436–465.
