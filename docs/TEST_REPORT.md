# EmoSense AI — Quality Assurance & Test Verification Report

**Date:** 2026-10-18  
**Branch:** `fix/production-hardening`  
**Commit:** `HEAD` (on working branch)  
**Test Suite:** Automated `pytest` suite + Standalone CLI Benchmarks  

---

## 1. Test Environment Specification

- **Operating System:** Windows 11 Enterprise (x64)
- **Python Runtime:** Python 3.10.9 (System 64-bit)
- **PyTorch:** `2.2.1+cpu`
- **Transformers:** `4.57.6` (Pinned `<5.0.0` for PyTorch 2.x compatibility)
- **Scikit-Learn:** `1.6.1`
- **XGBoost:** `3.2.0`
- **FastAPI:** `0.136.3`
- **Pydantic:** `2.13.4`
- **Inference Hardware:** CPU (Intel Core 64-bit)

---

## 2. Test Execution Summary

### Automated Test Suite (`tests/`)
- **Execution Command:** `python -m pytest tests/ -v`
- **Total Tests Collected:** 45
- **Passed:** 45 (100%)
- **Failed:** 0 (0%)
- **Skipped:** 0 (0%)
- **Execution Time:** ~94.8s (including 1 real-model smoke test and 44 mock/unit/API tests)

### Breakdown by Test Suite

| Test Module | Coverage Area | Tests Run | Result |
|---|---|---|---|
| `tests/test_validation.py` | Pydantic validation: empty/whitespace text, >2000 chars, alpha boundaries $[0.0, 1.0]$, NaN/Inf rejection, ensemble method literal, PredictResponse serialization | 10 | **PASSED** |
| `tests/test_preprocessing.py` | `clean_text` URL stripping, punctuation, numbers, emoji/unicode handling, non-string handling; `nrc_features` 10-dimensional density calculation and token count normalization | 7 | **PASSED** |
| `tests/test_ensemble.py` | `_normalize_probabilities` zero/NaN handling, sum-to-1 normalization; `run_inference` soft voting; `run_inference` stacking meta-classifier; vectorized `run_batch_inference` numerical equivalence; emotion class order preservation | 5 | **PASSED** |
| `tests/test_model_loader.py` | `ModelState` state machine (`UNINITIALIZED`, `LOADING`, `READY`, `FAILED`), `is_model_ready()` initial check, runtime guards on unready registry, safe `unload_models()` clearing references and memory | 2 | **PASSED** |
| `tests/test_api_health.py` | Backward-compatible `GET /`, Kubernetes process liveness probe `GET /health/live`, model readiness probe `GET /health/ready` (200 when ready, 503 when loading/uninitialized) | 4 | **PASSED** |
| `tests/test_api_predict.py` | `POST /predict` valid single inference, empty string 422, whitespace 422, length > 2000 422, out-of-bounds alpha 422, malformed JSON 422, CORS allowed origins (`localhost:8080`), CORS disallowed origins rejection | 8 | **PASSED** |
| `tests/test_api_batch.py` | `POST /predict/batch` valid CSV execution, missing `text` column 400, non-CSV extension 400, empty 0-byte file 400, row count > 500 rejection 400, streaming size ceiling > 5 MB rejection 413, blank row representation (`predicted_emotion: "unknown"`), invalid alpha 400 | 8 | **PASSED** |
| `tests/test_real_models.py` | Real artifact loading smoke test (`xgb.pkl`, `word_tfidf.pkl`, `char_tfidf.pkl`, `meta_clf.pkl`, `nrc_scaler.pkl`, `roberta_best/model.safetensors`), single inference correctness, batch equivalence | 1 | **PASSED** |

---

## 3. Empirical Model Evaluation

Evaluated across the 6 target emotion classes on a held-out stratified sample of 300 items (50 per class) from `clean_emotion_dataset.csv` using `backend/scripts/evaluate_model.py`:

| Model Branch / Strategy | Accuracy (%) | Macro-F1 (%) | Avg Latency (CPU) | Strategy Description |
|---|---|---|---|---|
| **XGBoost Alone** | 91.3% | 91.5% | 13.41 ms | Word TF-IDF + Char TF-IDF + NRC Scaled Features |
| **RoBERTa Alone** | 98.0% | 98.0% | 275.59 ms | Fine-tuned RoBERTa Transformer contextual branch |
| **Deployed Ensemble ($\alpha = 0.7$)** | **98.0%** | **98.0%** | **0.13 ms** (blending overhead) | Dynamic Alpha-Weighted Soft Voting: $0.7 P_{\text{RoBERTa}} + 0.3 P_{\text{XGBoost}}$ |
| **Stacking Meta-Classifier** | 98.3% | 98.3% | 1.85 ms | `LogisticRegression` consuming 12 concatenated probabilities |

### Class-Specific Metrics (Deployed Ensemble, $\alpha = 0.7$)

| Emotion Class | Precision (%) | Recall (%) | F1-Score (%) | Support |
|---|---|---|---|---|
| **anger** | 98.0% | 100.0% | 99.0% | 50 |
| **fear** | 98.0% | 98.0% | 98.0% | 50 |
| **joy** | 98.0% | 98.0% | 98.0% | 50 |
| **neutral** | 96.2% | 100.0% | 98.0% | 50 |
| **sadness** | 100.0% | 92.0% | 95.8% | 50 |
| **surprise** | 100.0% | 100.0% | 100.0% | 50 |

---

## 4. Batch Performance & Concurrency Benchmarks

### Benchmark Test Case: `test_data.csv` (10 Rows)
- **Endpoint:** `POST /predict/batch`
- **Total Rows:** 10
- **Vectorized Server Processing Time:** `2,149.71 ms`
- **Total HTTP Client Roundtrip Time:** `2,171.65 ms`
- **Average Latency Per Row:** `~215 ms` (on CPU single-core worker)
- **Predictions Output:**
  - Row 1: `joy` (93.7%) — "I am so incredibly happy today!"
  - Row 2: `anger` (68.5%) — "This is the worst thing that has ever happened."
  - Row 3: `fear` (73.1%) — "I'm feeling really anxious about the upcoming meeting."
  - Row 4: `surprise` (96.1%) — "Wow, I didn't expect that at all!"
  - Row 5: `anger` (80.2%) — "I am absolutely furious right now."
  - Row 6: `neutral` (42.3%) — "The weather is mild today."
  - Row 7: `neutral` (41.2%) — "Just a regular day at work."
  - Row 8: `surprise` (84.8%) — "I can't believe we won the championship!"
  - Row 9: `sadness` (95.2%) — "The movie was so sad, it made me cry."
  - Row 10: `fear` (96.8%) — "I'm terrified of spiders."

---

## 5. Security & Reliability Mitigations Verified

1. **CORS Security:**
   - Wildcard origin `*` no longer passes `allow_credentials=True`.
   - Explicit trusted origins configured by default: `http://127.0.0.1:8080`, `http://localhost:8080`, `http://127.0.0.1:7860`, `http://localhost:7860`.
   - Verified that unapproved origins (e.g. `http://evil-attacker.com`) do not receive CORS authorization headers.
2. **Denial-of-Service / Memory Exhaustion Protection:**
   - Chunked streaming upload reader terminates immediately upon reaching the 5 MB ceiling with HTTP 413.
   - Enforced maximum row limit of 500 rows.
   - Enforced maximum cell text length of 2000 characters.
3. **Information Disclosure Prevention:**
   - Route handlers and global exception middleware log full stack traces internally via `log.exception(...)` and return sanitized, generic error details to clients (`"Inference failed due to an internal server error."`).
4. **Input Validation Integrity:**
   - `alpha` rejects non-finite values (`NaN`, `Inf`) and values outside $[0.0, 1.0]$.
   - Empty or whitespace-only texts return HTTP 422.
   - Texts exceeding 2000 characters return HTTP 422.
5. **Concurrency & Resource Safety:**
   - Asynchronous endpoints offload heavy synchronous PyTorch/XGBoost operations to `asyncio.to_thread` guarded by `asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)`, preventing event-loop starvation and CPU/GPU memory thrashing.

---

## 6. Blocked Verification Steps & Remaining Environment Notes

- **Docker Build in Local Environment:** Running `docker build` requires Docker Desktop daemon running on the host system. Dockerfile syntax, non-root user configuration, and HEALTHCHECK instructions were verified statically.
- **Hugging Face Hub Remote Push:** In accordance with mandatory rules, changes have been committed locally to branch `fix/production-hardening` and have not been pushed to the remote repository without explicit user authorization.
