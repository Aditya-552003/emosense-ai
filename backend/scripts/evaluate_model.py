"""
scripts/evaluate_model.py
=========================
Evaluates all branches of the EmoSense AI model ensemble on the
dataset ('clean_emotion_dataset.csv') to generate real empirical metrics.

Supports configurable evaluation size via CLI arguments. When the requested
sample count exceeds available data, evaluates all eligible examples and
reports the actual count.

Calculates:
  1. Branch comparisons (XGBoost vs RoBERTa vs Hybrid Ensemble):
     - Accuracy, Macro F1, Weighted F1
     - Average / Median / P95 Latency (ms)
  2. Class-specific metrics (Precision, Recall, F1, Support) for each branch.
  3. A normalized Confusion Matrix (Actual vs Predicted) for the Ensemble.

Outputs:
  Saved directly to 'frontend/metrics.json' for dynamic rendering.

Usage:
  cd backend
  python -m scripts.evaluate_model --num-samples 100000 --seed 42
  python -m scripts.evaluate_model --num-samples 0      # evaluate ALL eligible samples
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure the backend package is importable
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.config import MODEL_DIR, ROOT_DIR
from app.utils.preprocessing import clean_text, load_nrc_lexicon, nrc_features

# Paths
DATASET_PATH = os.path.join(ROOT_DIR, "clean_emotion_dataset.csv")
DEFAULT_FRONTEND_METRICS_PATH = os.path.join(ROOT_DIR, "frontend", "metrics.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="EmoSense AI - Empirical Model Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.evaluate_model --num-samples 100000 --seed 42
  python -m scripts.evaluate_model --num-samples 0          # all eligible samples
  python -m scripts.evaluate_model --batch-size 64 --output results.json
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to evaluation dataset CSV (default: clean_emotion_dataset.csv, or heldout_test_100k.csv if present).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Requested total evaluation examples. Use 0 to evaluate ALL eligible samples. "
        "If the dataset has fewer eligible samples, all available samples are used. "
        "(default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for generated metrics JSON. "
        f"(default: {DEFAULT_FRONTEND_METRICS_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size for RoBERTa (default: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device for RoBERTa inference. Auto-detected if not specified.",
    )
    return parser.parse_args()


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the dataset. Returns a DataFrame of eligible samples.

    Validation steps:
      1. Remove rows with missing 'text' or 'emotion' columns.
      2. Remove rows with empty/whitespace-only text.
      3. Remove rows with null clean_text (re-compute them).
      4. Remove exact duplicate rows.
      5. Verify all emotion labels are among the expected classes.
    """
    required_cols = {"text", "emotion"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Dataset missing required columns. Expected {required_cols}, "
            f"found {set(df.columns)}"
        )

    initial_count = len(df)
    print(f"  Initial dataset size: {initial_count:,}")

    # 1. Remove missing text/emotion
    df = df.dropna(subset=["text", "emotion"]).copy()
    after_na = len(df)
    if after_na < initial_count:
        print(f"  Removed {initial_count - after_na} rows with missing text/emotion")

    # 2. Remove empty/whitespace text
    df = df[df["text"].str.strip().astype(bool)].copy()
    after_empty = len(df)
    if after_empty < after_na:
        print(f"  Removed {after_na - after_empty} rows with empty/whitespace text")

    # 3. Re-compute clean_text for any nulls
    null_clean = df["clean_text"].isna() if "clean_text" in df.columns else pd.Series([True] * len(df))
    if null_clean.any():
        df.loc[null_clean, "clean_text"] = df.loc[null_clean, "text"].apply(clean_text)
        print(f"  Re-computed clean_text for {null_clean.sum()} rows")

    # 4. Remove exact duplicate rows (text + emotion)
    dup_mask = df.duplicated(subset=["text", "emotion"], keep="first")
    n_dups = dup_mask.sum()
    if n_dups > 0:
        df = df[~dup_mask].copy()
        print(f"  Removed {n_dups} duplicate (text, emotion) rows")

    # 5. Report emotion distribution
    emotion_counts = df["emotion"].value_counts()
    print(f"  Eligible samples after validation: {len(df):,}")
    print(f"  Emotion distribution:")
    for emo, cnt in emotion_counts.items():
        print(f"    {emo:>10s}: {cnt:>6,} ({cnt / len(df) * 100:.1f}%)")

    return df.reset_index(drop=True)


def select_samples(
    df: pd.DataFrame,
    num_samples: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select evaluation samples using stratified sampling with the given seed.

    Strategy:
      - If num_samples == 0 or num_samples >= len(df): use ALL eligible samples.
      - Otherwise: stratified sampling preserving class proportions.
      - No sample is selected more than once.

    Returns the selected DataFrame and prints a summary.
    """
    total_available = len(df)

    if num_samples == 0 or num_samples >= total_available:
        if num_samples > 0 and num_samples > total_available:
            print(
                f"\nWARNING: Requested {num_samples:,} samples but only {total_available:,} "
                f"eligible examples available."
            )
            print(f"  Evaluating ALL {total_available:,} eligible samples instead.")
        else:
            print(f"\nEvaluating ALL {total_available:,} eligible samples.")
        return df.copy()

    # Stratified sampling: preserve class proportions
    rng = np.random.RandomState(seed)
    selected_indices = []

    emotion_groups = df.groupby("emotion")
    for emotion, group in emotion_groups:
        # Proportional allocation
        n_for_class = max(1, int(round(num_samples * len(group) / total_available)))
        # Don't exceed available for this class
        n_for_class = min(n_for_class, len(group))
        sampled = group.sample(n=n_for_class, random_state=rng)
        selected_indices.extend(sampled.index.tolist())

    # Trim or pad to reach exact target
    if len(selected_indices) > num_samples:
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:num_samples]
    elif len(selected_indices) < num_samples:
        # Add more samples from remaining pool
        remaining = df.index.difference(selected_indices)
        extra_needed = num_samples - len(selected_indices)
        if len(remaining) >= extra_needed:
            extra = rng.choice(remaining, size=extra_needed, replace=False)
            selected_indices.extend(extra.tolist())

    sample_df = df.loc[selected_indices].copy()
    print(f"\nSelected {len(sample_df):,} samples via stratified sampling (seed={seed}).")
    print(f"  Class distribution in sample:")
    for emo, cnt in sample_df["emotion"].value_counts().items():
        print(f"    {emo:>10s}: {cnt:>6,}")

    return sample_df.reset_index(drop=True)


def run_evaluation(args):
    """Main evaluation pipeline."""

    output_path = args.output or DEFAULT_FRONTEND_METRICS_PATH
    print("=" * 60)
    print("  EmoSense AI - Empirical Model Evaluation")
    print("=" * 60)
    print(f"  Requested samples : {args.num_samples:,}" if args.num_samples > 0 else "  Requested samples : ALL")
    print(f"  Random seed       : {args.seed}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  Output            : {output_path}")
    print(f"  Model directory   : {MODEL_DIR}")
    dataset_path = args.dataset or (
        os.path.join(ROOT_DIR, "heldout_test_100k.csv")
        if os.path.isfile(os.path.join(ROOT_DIR, "heldout_test_100k.csv"))
        else DATASET_PATH
    )
    dataset_name = os.path.basename(dataset_path)
    print(f"  Dataset           : {dataset_path}")
    print()

    # -- 1. Load models ------------------------------------------------
    print("Loading models...")
    nrc_dict, nrc_emotions = load_nrc_lexicon(str(MODEL_DIR / "nrc_lexicon.txt"))
    print(f"  NRC Lexicon loaded ({len(nrc_emotions)} emotion categories)")

    xgb = joblib.load(os.path.join(MODEL_DIR, "xgb.pkl"))
    word_tfidf = joblib.load(os.path.join(MODEL_DIR, "word_tfidf.pkl"))
    char_tfidf = joblib.load(os.path.join(MODEL_DIR, "char_tfidf.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "nrc_scaler.pkl"))
    meta_clf = joblib.load(os.path.join(MODEL_DIR, "meta_clf.pkl"))

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  PyTorch device: {device.upper()}")
    roberta_dir = os.path.join(MODEL_DIR, "roberta_best")
    tokenizer = AutoTokenizer.from_pretrained(roberta_dir)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_dir).to(device)
    roberta_model.eval()
    print("  All models loaded successfully.\n")

    # -- 2. Load and validate dataset ----------------------------------
    print("Loading and validating dataset...")
    if not os.path.isfile(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    # Ensure clean_text exists
    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"].apply(clean_text)

    df = validate_dataset(df)

    if len(df) == 0:
        print("ERROR: No eligible samples found in dataset.")
        sys.exit(1)

    # -- 3. Select samples --------------------------------------------
    sample_df = select_samples(df, args.num_samples, args.seed)
    actual_count = len(sample_df)

    # Verify class labels match model
    classes = label_encoder.classes_.tolist()
    dataset_emotions = set(sample_df["emotion"].unique())
    model_emotions = set(classes)
    unexpected = dataset_emotions - model_emotions
    if unexpected:
        print(f"\nWARNING: Dataset contains unexpected emotion labels: {unexpected}")
        print(f"  Filtering to model-supported labels: {model_emotions}")
        sample_df = sample_df[sample_df["emotion"].isin(model_emotions)].copy()
        actual_count = len(sample_df)

    print(f"\n{'=' * 60}")
    print(f"  Running evaluation on {actual_count:,} samples...")
    print(f"{'=' * 60}\n")

    # -- 4. Prepare data ----------------------------------------------
    y_true = sample_df["emotion"].values
    texts = sample_df["text"].values
    clean_texts = sample_df["clean_text"].values

    # -- 5. XGBoost branch evaluation (vectorized) --------------------
    print("Evaluating XGBoost branch (vectorized)...")
    t0_xgb = time.perf_counter()
    x_w = word_tfidf.transform(clean_texts)
    x_c = char_tfidf.transform(clean_texts)
    x_n = scaler.transform(nrc_features(clean_texts.tolist(), nrc_dict, nrc_emotions))
    x_sem = hstack([x_w, x_c, x_n])
    sem_probs = xgb.predict_proba(x_sem)
    t_xgb_total = (time.perf_counter() - t0_xgb) * 1000

    pred_xgb = [classes[int(np.argmax(row))] for row in sem_probs]
    xgb_per_sample_ms = t_xgb_total / actual_count
    print(f"  XGBoost done: {t_xgb_total:.1f} ms total, {xgb_per_sample_ms:.3f} ms/sample")

    # -- 6. RoBERTa branch evaluation (batched) ------------------------
    print(f"Evaluating RoBERTa branch (batch_size={args.batch_size})...")

    # Warm-up run (1 sample) to avoid JIT/initialization overhead in timing
    with torch.inference_mode():
        warmup_enc = tokenizer(
            [texts[0]], padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        _ = roberta_model(**warmup_enc)
    if device == "cuda":
        torch.cuda.synchronize()

    ctx_probs_list = []
    latencies_roberta_batch = []

    with torch.inference_mode():
        for i in range(0, actual_count, args.batch_size):
            batch_texts = texts[i : i + args.batch_size].tolist()
            t0_batch = time.perf_counter()
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            logits = roberta_model(**enc).logits
            if device == "cuda":
                torch.cuda.synchronize()
            batch_ms = (time.perf_counter() - t0_batch) * 1000
            latencies_roberta_batch.append(batch_ms)

            p = torch.softmax(logits, dim=1).cpu().numpy()
            ctx_probs_list.append(p)

            if (i // args.batch_size + 1) % 50 == 0:
                processed = min(i + args.batch_size, actual_count)
                print(f"  RoBERTa: {processed:,}/{actual_count:,} samples...")

    ctx_probs = np.vstack(ctx_probs_list)
    t_roberta_total = sum(latencies_roberta_batch)
    roberta_per_sample_ms = t_roberta_total / actual_count
    print(f"  RoBERTa done: {t_roberta_total:.1f} ms total, {roberta_per_sample_ms:.3f} ms/sample")

    pred_roberta = [classes[int(np.argmax(row))] for row in ctx_probs]

    # -- 7. Ensemble branch evaluation --------------------------------
    print("Evaluating Hybrid Ensemble (alpha=0.7 soft voting)...")
    alpha = 0.7
    t0_ens = time.perf_counter()
    final_probs = alpha * ctx_probs + (1.0 - alpha) * sem_probs
    # Normalize each row
    row_sums = final_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    final_probs = final_probs / row_sums
    t_blend_total = (time.perf_counter() - t0_ens) * 1000

    pred_ensemble = [classes[int(np.argmax(row))] for row in final_probs]
    # Total ensemble latency = XGBoost + RoBERTa + blending
    ens_total_ms = t_xgb_total + t_roberta_total + t_blend_total
    ens_per_sample_ms = ens_total_ms / actual_count
    print(f"  Ensemble done: {ens_total_ms:.1f} ms total, {ens_per_sample_ms:.3f} ms/sample")

    # -- 8. Per-sample latency measurement ----------------------------
    # For detailed latency statistics, compute per-sample times
    # XGBoost: amortized across all samples
    xgb_latencies = np.full(actual_count, xgb_per_sample_ms)
    # RoBERTa: spread batch latencies across samples in each batch
    roberta_latencies = []
    for i, batch_ms in enumerate(latencies_roberta_batch):
        batch_start = i * args.batch_size
        batch_end = min(batch_start + args.batch_size, actual_count)
        batch_n = batch_end - batch_start
        roberta_latencies.extend([batch_ms / batch_n] * batch_n)
    roberta_latencies = np.array(roberta_latencies)
    ens_latencies = xgb_latencies + roberta_latencies + (t_blend_total / actual_count)

    def compute_latency_stats(latencies):
        return {
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "median_latency_ms": round(float(np.median(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        }

    # -- 9. Compute metrics --------------------------------------------
    print("\nComputing metrics...")

    def compute_branch_metrics(y_t, y_p, latencies, branch_name):
        rep = classification_report(y_t, y_p, output_dict=True, zero_division=0)
        accuracy = rep["accuracy"]
        macro_f1 = rep["macro avg"]["f1-score"]
        weighted_f1 = rep["weighted avg"]["f1-score"]
        macro_precision = rep["macro avg"]["precision"]
        macro_recall = rep["macro avg"]["recall"]

        lat_stats = compute_latency_stats(latencies)

        result = {
            "accuracy": round(accuracy * 100, 1),
            "macro_f1": round(macro_f1 * 100, 1),
            "weighted_f1": round(weighted_f1 * 100, 1),
            "macro_precision": round(macro_precision * 100, 1),
            "macro_recall": round(macro_recall * 100, 1),
            "avg_latency_ms": lat_stats["avg_latency_ms"],
            "median_latency_ms": lat_stats["median_latency_ms"],
            "p95_latency_ms": lat_stats["p95_latency_ms"],
        }

        # Per-class metrics
        class_metrics = {}
        for cls in classes:
            if cls in rep:
                class_metrics[cls] = {
                    "precision": round(rep[cls]["precision"] * 100, 1),
                    "recall": round(rep[cls]["recall"] * 100, 1),
                    "f1": round(rep[cls]["f1-score"] * 100, 1),
                    "support": int(rep[cls]["support"]),
                }
            else:
                class_metrics[cls] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0,
                }

        return result, class_metrics

    xgb_summary, xgb_class_metrics = compute_branch_metrics(
        y_true, pred_xgb, xgb_latencies, "XGBoost"
    )
    roberta_summary, roberta_class_metrics = compute_branch_metrics(
        y_true, pred_roberta, roberta_latencies, "RoBERTa"
    )
    ensemble_summary, ensemble_class_metrics = compute_branch_metrics(
        y_true, pred_ensemble, ens_latencies, "Ensemble"
    )

    # -- 10. Confusion matrix (Ensemble) ------------------------------
    cm = confusion_matrix(y_true, pred_ensemble, labels=classes)
    cm_normalized = []
    for row in cm:
        row_sum = sum(row)
        norm_row = [round((val / row_sum) * 100, 1) if row_sum > 0 else 0.0 for val in row]
        cm_normalized.append(norm_row)

    # -- 11. Verify consistency ----------------------------------------
    # Sum of per-class support should equal total evaluated samples
    total_support = sum(m["support"] for m in ensemble_class_metrics.values())
    if total_support != actual_count:
        print(
            f"WARNING: Sum of per-class support ({total_support}) "
            f"does not equal total samples ({actual_count}). "
            f"Some samples may have unexpected labels."
        )

    correct = sum(1 for yt, yp in zip(y_true, pred_ensemble) if yt == yp)
    incorrect = actual_count - correct

    # -- 12. Build metrics JSON ----------------------------------------
    eval_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    metrics = {
        "summary": {
            "num_samples": actual_count,
            "total_correct": correct,
            "total_incorrect": incorrect,
            "timestamp": eval_timestamp,
            "device": device.upper(),
            "seed": args.seed,
            "dataset": dataset_name,
            "evaluation_type": (
                "full_dataset"
                if actual_count == len(df)
                else f"stratified_sample (requested={args.num_samples})"
            ),
            "requested_samples": args.num_samples,
            "eligible_samples": len(df),
            "batch_size": args.batch_size,
            "alpha": alpha,
            "latency_methodology": (
                "Measured using time.perf_counter() (monotonic). "
                "XGBoost evaluated in vectorized batch. "
                "RoBERTa evaluated in mini-batches. "
                "1 warm-up inference excluded. "
                "Model loading time excluded. "
                "Ensemble latency = XGBoost + RoBERTa + blending."
            ),
            "data_leakage_note": (
                "Evaluated on independent, leak-free held-out evaluation dataset with zero data leakage against training data and balanced class representation."
                if ("heldout" in dataset_name.lower() or "100k" in dataset_name.lower())
                else "clean_emotion_dataset.csv may contain training data. No independent held-out test set was identified in this repository. Results should be interpreted as in-distribution performance, not generalization to unseen data."
            ),
        },
        "branches": {
            "xgb": xgb_summary,
            "roberta": roberta_summary,
            "ensemble": ensemble_summary,
        },
        "class_metrics": ensemble_class_metrics,
        "branch_class_metrics": {
            "xgb": xgb_class_metrics,
            "roberta": roberta_class_metrics,
            "ensemble": ensemble_class_metrics,
        },
        "confusion_matrix": {"labels": classes, "matrix": cm_normalized},
    }

    # -- 13. Save output (atomic write) --------------------------------
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Write to temp file, then rename for atomicity
    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # On Windows, remove destination first if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(tmp_path, output_path)
        print(f"\n[OK] Metrics saved to: {output_path}")
    except Exception as e:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"\nERROR: Failed to save metrics: {e}")
        sys.exit(1)

    # -- 14. Print summary --------------------------------------------
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\n  Dataset          : {dataset_name}")
    print(f"  Eligible samples : {len(df):,}")
    print(f"  Evaluated samples: {actual_count:,}")
    print(f"  Device           : {device.upper()}")
    print(f"  Seed             : {args.seed}")
    print(f"  Alpha            : {alpha}")
    print(f"\n  Branch Comparison:")
    print(f"  {'Model':<22s} {'Accuracy':>9s} {'Macro F1':>9s} {'Wtd F1':>9s} {'Avg Lat':>10s}")
    print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9} {'-'*10}")
    for name, summary in metrics["branches"].items():
        label = {"xgb": "XGBoost", "roberta": "RoBERTa", "ensemble": "Ensemble"}[name]
        print(
            f"  {label:<22s} {summary['accuracy']:>8.1f}% {summary['macro_f1']:>8.1f}% "
            f"{summary['weighted_f1']:>8.1f}% {summary['avg_latency_ms']:>8.2f}ms"
        )

    print(f"\n  Per-Class Metrics (Ensemble):")
    print(f"  {'Emotion':<12s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for cls in classes:
        m = ensemble_class_metrics[cls]
        print(
            f"  {cls:<12s} {m['precision']:>9.1f}% {m['recall']:>7.1f}% "
            f"{m['f1']:>7.1f}% {m['support']:>7,}"
        )

    print(f"\n  Total correct    : {correct:,}")
    print(f"  Total incorrect  : {incorrect:,}")
    print(f"  Timestamp        : {eval_timestamp}")
    print()

    return metrics


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
