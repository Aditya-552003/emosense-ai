# 🧠 Emotion Detection API — Backend

A production-ready **FastAPI** backend that serves a hybrid stacking ensemble emotion classifier, combining:
- 🤗 **RoBERTa** (fine-tuned) — deep contextual understanding
- 🌲 **XGBoost** on TF-IDF + NRC Emotion Lexicon features — semantic signal
- 📊 **Dynamic alpha-weighted soft voting** blending both branches

**Detects 6 emotions:** `anger` · `fear` · `joy` · `neutral` · `sadness` · `surprise`

---

## 📁 Folder Structure

```
backend/
├── app/                        # Application package
│   ├── __init__.py             # App factory (create_app)
│   ├── config.py               # Environment-based settings
│   ├── routes/                 # API endpoint handlers
│   │   ├── __init__.py
│   │   ├── health.py           # GET /
│   │   └── predict.py          # POST /predict, POST /predict/batch
│   ├── schemas/                # Pydantic request/response models
│   │   ├── __init__.py
│   │   └── predict.py          # PredictRequest, PredictResponse
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── inference.py        # Full ensemble inference pipeline
│   │   └── model_loader.py     # Model lifecycle management
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       └── preprocessing.py    # Text cleaning + NRC features
├── model/                      # ML model artifacts
│   ├── xgb.pkl                 # XGBoost model
│   ├── word_tfidf.pkl          # Word-level TF-IDF vectorizer
│   ├── char_tfidf.pkl          # Character-level TF-IDF vectorizer
│   ├── nrc_scaler.pkl          # StandardScaler for NRC features
│   ├── label_encoder.pkl       # LabelEncoder (6 classes)
│   ├── meta_clf.pkl            # Stacking LogisticRegression
│   ├── nrc_lexicon.txt         # Auto-downloaded on first run
│   └── roberta_best/           # Fine-tuned RoBERTa weights
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── scripts/                    # Standalone CLI utilities
│   ├── evaluate_model.py       # Generate metrics.json from dataset
│   └── test_inference.py       # Quick smoke-test
├── requirements.txt
├── start_server.bat
└── README.md
```

---

## ⚡ Quick Start (Local)

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note:** `numpy<2` is pinned intentionally — NumPy 2.x breaks PyTorch tensor→numpy conversion.

### 2. Start the server

```bash
# Windows
start_server.bat

# Or manually
uvicorn app:create_app --factory --host 127.0.0.1 --port 8080
```

The NRC Emotion Lexicon is downloaded automatically on the first startup (~2.5 MB).

### 3. Verify it's running

```bash
curl http://localhost:8080/
```

---

## 📡 API Reference

### `GET /`

Health check and status endpoint.

**Response:**
```json
{
  "status": "online",
  "api": "Emotion Detection API",
  "version": "1.0.0",
  "device": "cpu",
  "emotion_classes": ["anger","fear","joy","neutral","sadness","surprise"],
  "endpoints": { ... }
}
```

---

### `POST /predict`

Analyze the emotion in a text string.

**Request body:**
```json
{ "text": "I am absolutely thrilled about this project!" }
```

**Response:**
```json
{
  "success": true,
  "predicted_emotion": "joy",
  "confidence": 0.9423,
  "confidence_pct": "94.2%",
  "probabilities": {
    "anger":   0.0121,
    "fear":    0.0087,
    "joy":     0.9423,
    "neutral": 0.0098,
    "sadness": 0.0204,
    "surprise": 0.0067
  },
  "processing_time_ms": 312.5
}
```

**Validation errors:**
```json
{ "detail": [{ "msg": "Input text must not be empty." }] }
```

---

### `POST /predict/batch`

Upload a CSV file with a `text` column. Returns predictions for every row (max 500).

---

## 🔧 Configuration

Settings are loaded from environment variables (or a `.env` file in the project root).

| Variable       | Default     | Description                                       |
|----------------|-------------|---------------------------------------------------|
| `HOST`         | `127.0.0.1` | Server bind address                               |
| `PORT`         | `8080`      | Server port                                       |
| `CORS_ORIGINS` | `*`         | Comma-separated allowed origins                   |
| `LOG_LEVEL`    | `INFO`      | Logging level (DEBUG, INFO, WARNING, ERROR)        |
| `MODEL_DIR`    | `model`     | Model artifacts directory (relative to backend/)   |

---

## 🚀 Scripts

```bash
# Generate live evaluation metrics
cd backend
python -m scripts.evaluate_model

# Quick smoke-test
python -m scripts.test_inference
```

---

## 🌐 Deployment

### Render (recommended free tier)

1. Push the `backend/` folder to a GitHub repo.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:create_app --factory --host 0.0.0.0 --port $PORT`
4. Upload the `model/` directory as a Render **Persistent Disk** (or pre-bake into the image).

### Railway

```bash
railway init
railway up
```

Set `START_COMMAND = uvicorn app:create_app --factory --host 0.0.0.0 --port $PORT`.

---

## 📄 Citation

If you use the NRC Emotion Lexicon in your research:
> Mohammad, S. & Turney, P. (2013). *Crowdsourcing a Word-Emotion Association Lexicon.* Computational Intelligence, 29(3), 436–465.
