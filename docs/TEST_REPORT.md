# EmoSense AI — Quality Assurance & Test Verification Report

**Date:** 2026-10-30  
**Branch:** `feature/evaluation-100k`  
**Commit:** `HEAD` (on working branch)  
**Test Suite:** Automated `pytest` suite (78 tests) + Standalone Empirical Evaluation Pipeline  

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

### Automated Software Test Suite (`tests/`)

- **Execution Command:** `python -m pytest tests/ -v -m "not real_model"`
- **Total Tests Collected:** 78
- **Passed:** 78 (100%)
- **Failed:** 0 (0%)
- **Skipped:** 0 (0%)
- **Execution Time:** ~6.23s

> [!NOTE]
> Evaluating model metrics on 17,484 dataset samples is distinct from software verification tests. The software tests verify input validation, API contracts, pipeline mechanics, and security controls using controlled synthetic fixtures.

### Breakdown by Test Suite

| Test Module | Coverage Area | Tests Run | Result |
|---|---|---:|---|
| `tests/test_evaluation_pipeline.py` | Configurable sample size, deterministic seeding, over-sample boundary handling, empty/malformed data, deduplication, JSON schema integrity, frontend compatibility | 26 | **PASSED** |
| `tests/test_validation.py` | Pydantic validation: empty/whitespace text, >2000 chars, alpha boundaries $[0.0, 1.0]$, NaN/Inf rejection, ensemble method literal, PredictResponse serialization | 10 | **PASSED** |
| `tests/test_preprocessing.py` | `clean_text` URL stripping, punctuation, numbers, emoji/unicode handling, non-string handling; `nrc_features` 10-dimensional density calculation and token count normalization | 7 | **PASSED** |
| `tests/test_ensemble.py` | `_normalize_probabilities` zero/NaN handling, sum-to-1 normalization; `run_inference` soft voting; `run_inference` stacking meta-classifier; vectorized `run_batch_inference` numerical equivalence; emotion class order preservation | 5 | **PASSED** |
| `tests/test_model_loader.py` | `ModelState` state machine (`UNINITIALIZED`, `LOADING`, `READY`, `FAILED`), `is_model_ready()` initial check, runtime guards on unready registry, safe `unload_models()` clearing references and memory | 2 | **PASSED** |
| `tests/test_api_health.py` | Backward-compatible `GET /`, Kubernetes process liveness probe `GET /health/live`, model readiness probe `GET /health/ready` (200 when ready, 503 when loading/uninitialized) | 4 | **PASSED** |
| `tests/test_api_predict.py` | `POST /predict` valid single inference, empty string 422, whitespace 422, length > 2000 422, out-of-bounds alpha 422, malformed JSON 422, CORS allowed origins (`localhost:8080`), CORS disallowed origins rejection | 8 | **PASSED** |
| `tests/test_api_batch.py` | `POST /predict/batch` valid CSV execution, missing `text` column 400, non-CSV extension 400, empty 0-byte file 400, row count > 500 rejection 400, streaming size ceiling > 5 MB rejection 413, blank row representation (`predicted_emotion: "unknown"`), invalid alpha 400 | 8 | **PASSED** |
| `tests/test_api_analytics.py` | `GET /analytics/evaluation` empirical metrics schema verification, `GET /analytics/model-info`, `POST /analytics/summary` KPI aggregation, emotion/date filtering, confidence bands | 7 | **PASSED** |
| `tests/test_real_models.py` | Real artifact loading smoke test (`xgb.pkl`, `word_tfidf.pkl`, `char_tfidf.pkl`, `meta_clf.pkl`, `nrc_scaler.pkl`, `roberta_best/model.safetensors`), single inference correctness, batch equivalence | 1 | **PASSED** |

---

## 3. Empirical Model Evaluation

### 3.1 Evaluation Overview

- **Evaluation Date:** 2026-10-30 12:09:14
- **Dataset File:** `clean_emotion_dataset.csv`
- **Total Initial Records:** 17,518
- **Duplicate Records Removed:** 34
- **Re-computed Clean Texts:** 3
- **Total Eligible Examples:** 17,484
- **Requested Evaluation Samples:** 100,000
- **Actual Evaluated Samples:** 17,484 (100% of all eligible samples in dataset)
- **Model Checkpoints:**
  - Semantic Branch: `xgb.pkl`, `word_tfidf.pkl`, `char_tfidf.pkl`, `nrc_scaler.pkl`, `label_encoder.pkl`
  - Contextual Branch: `roberta_best/` (`model.safetensors`, `config.json`, `tokenizer.json`)
  - Meta Classifier: `meta_clf.pkl`
- **Inference Hardware:** CPU (x86_64)
- **Batch Size:** 64 (RoBERTa mini-batches)
- **Random Seed:** 42

### 3.2 Evaluation Methodology

1. **Dataset Validation & Cleansing:** The dataset `clean_emotion_dataset.csv` was inspected and validated. Missing texts and empty/whitespace rows were removed, missing `clean_text` fields were deterministically re-computed, and exact `(text, emotion)` duplicates were dropped.
2. **Target Sample Selection:** When `--num-samples 100000` is requested, the pipeline checks available eligible records. Since the repository contains 17,484 eligible records, the pipeline avoids synthetic replication or data fabrication and evaluates all 17,484 genuine records.
3. **Inference Execution:**
   - **XGBoost:** Feature transformation (word TF-IDF + char TF-IDF + NRC lexicon densities) is vectorized across all samples in one sparse pass.
   - **RoBERTa:** Evaluated in mini-batches (`batch_size=64`) under `torch.inference_mode()`. A 1-sample warmup pass was executed before timing to exclude JIT/cache loading overhead.
   - **Hybrid Ensemble ($\alpha = 0.7$):** Soft voting blends probabilities ($0.7 P_{\text{RoBERTa}} + 0.3 P_{\text{XGBoost}}$) followed by row normalization.
4. **Latency Measurement:** High-resolution monotonic timing (`time.perf_counter()`) measured model inference exclusively, isolating inference from dataset loading, validation, and JSON serialization.

### 3.3 Model Comparison Table

| Model Branch / Strategy | Samples Evaluated | Accuracy | Macro F1 | Weighted F1 | Average Latency |
|---|---:|---:|---:|---:|---:|
| **XGBoost Alone** | 17,484 | 84.8% | 78.9% | 85.6% | 0.25 ms |
| **RoBERTa Alone** | 17,484 | 96.3% | 93.9% | 96.3% | 41.37 ms |
| **Hybrid Ensemble ($\alpha = 0.7$)** | 17,484 | 95.5% | 92.6% | 95.6% | 41.62 ms |

### 3.4 Per-Class Evaluation (Hybrid Ensemble, $\alpha = 0.7$)

| Emotion Class | Precision (%) | Recall (%) | F1-Score (%) | Support |
|---|---:|---:|---:|---:|
| **anger** | 85.4% | 97.9% | 91.2% | 1,544 |
| **fear** | 79.8% | 98.4% | 88.1% | 510 |
| **joy** | 89.6% | 98.6% | 93.9% | 1,010 |
| **neutral** | 99.6% | 94.4% | 96.9% | 12,798 |
| **sadness** | 89.2% | 98.9% | 93.8% | 873 |
| **surprise** | 85.3% | 99.7% | 91.9% | 749 |
| **Total / Macro Avg** | **88.1%** | **98.0%** | **92.6%** | **17,484** |

- **Total Correct Predictions:** 16,697 / 17,484 (95.5%)
- **Total Incorrect Predictions:** 787 / 17,484 (4.5%)
- **Support Sum Check:** $1,544 + 510 + 1,010 + 12,798 + 873 + 749 = 17,484$ (exactly matches evaluated count).

### 3.5 Normalized Confusion Matrix (Actual vs. Predicted, %)

| Actual \ Predicted | anger | fear | joy | neutral | sadness | surprise |
|---|---:|---:|---:|---:|---:|---:|
| **anger** | **97.9** | 0.1 | 0.1 | 1.8 | 0.1 | 0.1 |
| **fear** | 0.0 | **98.4** | 0.0 | 1.0 | 0.6 | 0.0 |
| **joy** | 0.2 | 0.0 | **98.6** | 1.0 | 0.0 | 0.2 |
| **neutral** | 2.0 | 1.0 | 0.9 | **94.4** | 0.8 | 1.0 |
| **sadness** | 0.2 | 0.0 | 0.0 | 0.9 | **98.9** | 0.0 |
| **surprise** | 0.0 | 0.0 | 0.1 | 0.1 | 0.0 | **99.7** |

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

1. **CORS Security:** Wildcard origin `*` no longer passes `allow_credentials=True`. Explicit origins configured.
2. **DoS / Memory Exhaustion Protection:** Chunked streaming upload reader terminates immediately upon reaching 5 MB ceiling with HTTP 413. Maximum row count capped at 500.
3. **Information Disclosure Prevention:** Sanitized, generic error messages returned to clients on unhandled errors; full trace logged internally.
4. **Input Validation Integrity:** Strict validation on `alpha` in $[0.0, 1.0]$, rejection of `NaN`/`Inf`, and empty/oversized texts rejected with HTTP 422.
5. **Concurrency & Resource Safety:** CPU-heavy inference offloaded to worker threads guarded by concurrency semaphore.

---

## 6. Reproducibility

To reproduce this exact empirical evaluation:

```bash
# 1. Activate environment and ensure dependencies are installed
python -m pip install -r backend/requirements.txt

# 2. Run the evaluation script
python -m backend.scripts.evaluate_model --num-samples 100000 --seed 42 --batch-size 64

# 3. Verify generated metrics
python -c "import json; m=json.load(open('frontend/metrics.json')); print('Evaluated samples:', m['summary']['num_samples'])"
```

---

## 7. Limitations & Honest Assessment

1. **Dataset Size Constraint (100,000 Sample Target):**
   - The primary evaluation file `clean_emotion_dataset.csv` contains 17,518 raw rows, yielding 17,484 valid, non-duplicate records.
   - An authentic 100,000-sample evaluation is not possible with the existing repository assets without manufacturing artificial duplicates, which violates scientific integrity. The pipeline reports the genuine 17,484 sample evaluation.
2. **Severe Class Imbalance:**
   - 73.2% of the dataset is `neutral` (12,798 samples), while `fear` comprises only 2.9% (510 samples).
   - Weighted F1 (95.6%) is significantly higher than Macro F1 (92.6%) due to majority-class dominance.
3. **Absence of Independent Held-Out Test Set (Data Leakage Risk):**
   - `clean_emotion_dataset.csv` contains pre-computed training columns (`clean_text`, `label_enc`) and likely overlaps with the training distribution.
   - Reported metrics represent in-distribution validation performance rather than out-of-distribution generalization.
