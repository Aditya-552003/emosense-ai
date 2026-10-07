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

A full-stack AI emotion detection system powered by a **hybrid stacking ensemble** that combines **fine-tuned RoBERTa** (contextual understanding) with **XGBoost + NRC Emotion Lexicon** (semantic features), blended via dynamic alpha-weighted soft voting.

**Detects 6 emotions:** `anger` · `fear` · `joy` · `neutral` · `sadness` · `surprise`

---

## 📁 Project Structure

```
emotion_model/
├── app.py                      # HF Spaces Gradio entry point
├── requirements.txt            # Root deps (for HF Spaces)
├── Dockerfile                  # Docker deployment (optional)
├── backend/                    # FastAPI backend
│   ├── app/                    # Application package
│   │   ├── __init__.py         # App factory (create_app)
│   │   ├── config.py           # Environment-based settings
│   │   ├── routes/             # API endpoint handlers
│   │   │   ├── health.py       # GET /
│   │   │   └── predict.py      # POST /predict, POST /predict/batch
│   │   ├── schemas/            # Pydantic request/response models
│   │   │   └── predict.py
│   │   ├── services/           # Business logic
│   │   │   ├── inference.py    # Ensemble inference pipeline
│   │   │   └── model_loader.py # Model lifecycle management
│   │   └── utils/              # Shared utilities
│   │       └── preprocessing.py# Text cleaning & NRC features
│   ├── model/                  # ML model artifacts
│   ├── scripts/                # Standalone evaluation & test scripts
│   ├── requirements.txt        # Backend-only deps
│   └── start_server.bat
├── frontend/                   # Static web UI
│   ├── css/                    # Stylesheets
│   │   ├── style.css           # Main design system
│   │   └── evaluation.css      # Evaluation page styles
│   ├── js/                     # ES6 modules
│   │   ├── app.js              # Entry point
│   │   ├── config.js           # Constants & emotion config
│   │   ├── api.js              # API communication layer
│   │   ├── ui.js               # DOM helpers & theme
│   │   ├── chart.js            # Donut chart
│   │   ├── probBars.js         # Probability bar rendering
│   │   ├── history.js          # Prediction history
│   │   ├── batch.js            # CSV batch upload
│   │   └── evaluation.js       # Evaluation dashboard
│   ├── index.html              # Main analyzer page
│   └── evaluation.html         # Model insights page
├── docs/                       # Documentation
│   └── api_spec.md             # REST API specification
├── .env.example                # Environment variable template
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Python dependencies

```bash
cd backend
python -m pip install -r requirements.txt
```

> **⚠️ Windows users:** Always use `python -m pip` instead of `pip` directly. Application Control policies may block `pip.exe` from running.

> **Note:** `numpy<2` is pinned — NumPy 2.x breaks PyTorch tensor→numpy conversion.

### 2. Start the backend server

```bash
# Windows
start_server.bat

# Or manually
cd backend
uvicorn app:create_app --factory --host 127.0.0.1 --port 8080
```

### 3. Open the frontend

Navigate to **http://127.0.0.1:8080/ui/index.html** in your browser.

---

## 🏗️ Architecture

```
Raw Text Input
       │
       ├─────────────────────────┐
       ▼                         ▼
┌─────────────────┐    ┌──────────────────┐
│  XGBoost Branch │    │  RoBERTa Branch  │
│  (Semantic)     │    │  (Contextual)    │
│  TF-IDF + NRC   │    │  Fine-tuned      │
│  → XGBClassifier│    │  Transformer     │
└────────┬────────┘    └────────┬─────────┘
         │                      │
         └──────┬───────────────┘
                ▼
    ┌────────────────────────┐
    │  Alpha-Weighted Blend  │
    │  α·RoBERTa + (1-α)·XGB│
    └────────────┬───────────┘
                 ▼
         Final Prediction
```

---

## 📡 API Endpoints

| Method | Path             | Description                    |
|--------|------------------|--------------------------------|
| GET    | `/`              | Health check & API status      |
| POST   | `/predict`       | Single text emotion prediction |
| POST   | `/predict/batch` | CSV batch emotion prediction   |

Full API specification: [docs/api_spec.md](docs/api_spec.md)

---

## 📄 License

See individual component licenses. NRC Emotion Lexicon citation:
> Mohammad, S. & Turney, P. (2013). *Crowdsourcing a Word-Emotion Association Lexicon.* Computational Intelligence, 29(3), 436–465.
