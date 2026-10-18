# EmoSense AI — Comprehensive Code Audit & Baseline Report

**Date:** 2026-10-18  
**Branch:** `fix/production-hardening`  
**Auditor:** Senior ML, Backend, MLOps, Security, & QA Engineering Team  
**Repository:** [https://github.com/Aditya-552003/emosense-ai](https://github.com/Aditya-552003/emosense-ai)

---

## 1. Executive Summary

This audit establishes the baseline status of the EmoSense AI application prior to remediation. The codebase serves a hybrid emotion detection system combining fine-tuned RoBERTa, XGBoost (with word TF-IDF, char TF-IDF, and NRC Emotion Lexicon features), and an ensemble mechanism.

The audit revealed several critical defects:
1. **Dependency Breaking Incompatibility:** Unconstrained `transformers>=4.30.0` allowed `transformers 5.19.0` to be installed, which explicitly disabled PyTorch (`PyTorch >= 2.5 is required but found 2.2.1`), completely breaking model inference at startup.
2. **Ensemble Architecture Discrepancy:** The repository contained conflicting implementations and descriptions of the ensemble. `backend/model/meta_clf.pkl` was loaded into memory at startup by `model_loader.py` but never invoked by `inference.py`. Instead, `inference.py` performed alpha-weighted soft voting, while `docs/api_spec.md` documented the exact predictions of `meta_clf.pkl`.
3. **Namespace Import Collision:** A root `app.py` file shadowed `backend/app`, causing circular import failures (`ImportError: cannot import name 'create_app'`) whenever scripts or tests were executed from the workspace root.
4. **Denial-of-Service & OOM Vulnerability in CSV Upload:** The `/predict/batch` endpoint called `await file.read()` without a maximum file size limit, buffering unbounded payloads directly into memory.
5. **Information Disclosure & Exception Leakage:** Route handlers caught exceptions and returned raw Python error strings (`detail=f"Inference failed: {str(exc)}"`) to HTTP clients.
6. **Insecure CORS Configuration:** `allow_origins=["*"]` was combined with `allow_credentials=True`, which is invalid under browser CORS standards and insecure.
7. **Input Validation Inconsistency:** `alpha` was silently clamped rather than validated; `NaN`, `Infinity`, and out-of-range floats were improperly handled; and the frontend `MAX_CHARS` (500) conflicted with the backend validation limit (2000).
8. **Missing Automated Test Suite:** No automated pytest unit or API regression tests existed in the repository.

---

## 2. Confirmed Defects & Remediation Matrix

| ID | Category | Severity | Affected Files | Root Cause | Proposed Remediation |
|---|---|---|---|---|---|
| **DEF-01** | Dependencies & Runtime | **CRITICAL** | `backend/requirements.txt`, `requirements.txt` | `transformers>=4.30.0` allowed `5.19.0` to be installed, which rejects `torch 2.2.1` with `Disabling PyTorch because PyTorch >= 2.5 is required`. | Pin `transformers>=4.40.0,<5.0.0` (specifically 4.49.0 or 4.57.x) and constrain `huggingface_hub>=0.34.0,<2.0.0`. |
| **DEF-02** | ML / Ensemble | **HIGH** | `backend/app/services/inference.py`, `backend/app/services/model_loader.py`, `docs/api_spec.md` | `meta_clf.pkl` is loaded into memory on startup but completely bypassed in `inference.py` in favor of hardcoded soft voting. Stacking and soft-voting were conflated in docs and code. | Support and document both verified ensemble paths with full transparency: default to alpha-weighted soft voting ($P = \alpha P_{\text{roberta}} + (1-\alpha) P_{\text{xgb}}$) and support the verified 12-feature stacking meta-classifier. Verify class ordering and normalize output probabilities. |
| **DEF-03** | Architecture / Packaging | **HIGH** | `app.py`, `backend/app/__init__.py` | Root `app.py` has the same name as the `app` package in `backend/app`, causing circular imports when executing from repository root. | Ensure `backend` directory is prioritized in `sys.path` and decouple package imports to prevent circular shadowing. |
| **DEF-04** | Security / DoS | **HIGH** | `backend/app/routes/predict.py` | `await file.read()` in `/predict/batch` buffers arbitrarily large file uploads into RAM with no size ceiling. | Enforce streaming chunk-based byte counter with configurable ceiling (`MAX_UPLOAD_SIZE_BYTES = 5 MB`). Terminate with HTTP 413 if exceeded. |
| **DEF-05** | Security / Info Leak | **MEDIUM** | `backend/app/routes/predict.py`, `backend/app/__init__.py` | Route exception handlers returned `f"Inference failed: {str(exc)}"` and `f"Failed to parse CSV: {str(exc)}"`, exposing internal file paths, types, or trace details. | Catch exceptions server-side with structured logging (`log.exception`), return generic, sanitized user-friendly JSON messages. |
| **DEF-06** | Security / CORS | **MEDIUM** | `backend/app/config.py`, `backend/app/__init__.py` | `CORS_ORIGINS = "*"` combined with `allow_credentials=True`. Wildcard credentials violate CORS specifications and expose CSRF/credential leakage risk. | Set `allow_credentials=False` when wildcard `*` is present. Allow credentials only when explicit origins are configured. Restrict HTTP methods and headers. |
| **DEF-07** | Input Validation | **MEDIUM** | `backend/app/schemas/predict.py`, `backend/app/routes/predict.py` | `alpha` field validator in Pydantic used `max(0.0, min(1.0, v))` to silently clamp invalid values; `NaN` and `Inf` were not rejected. | Reject values outside $[0.0, 1.0]$, `NaN`, and `Inf` with a clear validation error. |
| **DEF-08** | Performance | **MEDIUM** | `backend/app/routes/predict.py`, `backend/app/services/inference.py` | Batch CSV predictions ran single-item inference in a row-by-row Python loop (500 separate forward passes, TF-IDF transforms, and tokenizations). | Implement vectorized `run_batch_inference` batching TF-IDF, NRC features, and RoBERTa tokenization/forward passes (`batch_size=32`). |
| **DEF-09** | Concurrency | **MEDIUM** | `backend/app/routes/predict.py` | `predict` and `predict_batch` were `async def` endpoints executing synchronous CPU/GPU inference directly on the asyncio event loop, blocking all I/O. | Offload CPU-heavy inference to a worker thread pool (`asyncio.to_thread` / `run_in_threadpool`) guarded by a concurrency semaphore. |
| **DEF-10** | Observability | **LOW** | `backend/app/routes/health.py` | Only `GET /` existed; no Kubernetes/container liveness or readiness probes (`/health/live`, `/health/ready`). | Add `/health/live` and `/health/ready` endpoints reflecting actual model readiness state without running redundant inference. |
| **DEF-11** | Model Lifecycle | **MEDIUM** | `backend/app/services/model_loader.py` | Global `_models` dictionary populated without state synchronization; Hugging Face model repo revision unpinned; NRC lexicon downloaded from unverified mutable GitHub raw URL without timeout or checksum. | Add model lifecycle state tracking (`UNINITIALIZED`, `LOADING`, `READY`, `FAILED`); pin HF repo commit/tag; pin NRC lexicon mirror with validation and timeout. |
| **DEF-12** | Contract Inconsistency | **LOW** | `frontend/js/config.js`, `frontend/index.html` | Frontend configured `MAX_CHARS = 500` and HTML `maxlength="500"`, whereas backend validation and documentation specified 2000 characters. | Reconcile character limits between frontend and backend to 2000 characters with responsive UI counter. |

---

## 3. Baseline Test Results

Before applying fixes, the following baseline checks were executed:

### 3.1 Environment & Imports
- **Python Version:** 3.10.11 (System) / 3.14.3 (`.venv`)
- **PyTorch:** 2.2.1+cpu
- **Initial Transformers:** 5.19.0 (Broken — PyTorch disabled)
- **Resolved Transformers:** 4.57.6 (PyTorch operational)
- **PyTorch Device:** CPU

### 3.2 Smoke Test Execution (`backend/scripts/test_inference.py`)
- **Command:** `python backend/scripts/test_inference.py`
- **Result:** PASSED (after resolving transformers incompatibility)
- **Output:**
  - Input: `'I am so happy and excited today! Everything is going perfectly!'`
  - RoBERTa probability for joy: `92.46%`
  - XGBoost probability for joy: `97.63%`
  - Stacking meta-classifier output: Joy = `98.98%` (matches `docs/api_spec.md`)

### 3.3 Empirical Model Comparison on Held-Out Samples (600 Stratified Test Samples)
Evaluated on `clean_emotion_dataset.csv` across 6 emotion classes:

| Model Branch / Strategy | Accuracy (%) | Macro-F1 (%) | Notes |
|---|---|---|---|
| **XGBoost Alone** | 90.67% | 90.76% | Fast (~2ms), word+char TF-IDF + NRC features |
| **RoBERTa Alone** | 98.50% | 98.50% | High contextual accuracy (~70ms CPU) |
| **Stacking Meta-Classifier** | 98.33% | 98.33% | Consumes 12 probabilities via `meta_clf.pkl` |
| **Soft Voting ($\alpha = 0.7$)** | 98.50% | 98.50% | Weighted average: $0.7 P_{\text{rob}} + 0.3 P_{\text{xgb}}$ |
| **Soft Voting ($\alpha = 0.8$)** | 98.50% | 98.50% | Weighted average: $0.8 P_{\text{rob}} + 0.2 P_{\text{xgb}}$ |
| **Confidence-Gated Blend** | 98.50% | 98.50% | If RoBERTa conf $\ge 0.75$, trust RoBERTa; else blend 70/30 |

### 3.4 Baseline Unit / API Test Status
- Automated test files in repo: **None** (0 test files existed).
