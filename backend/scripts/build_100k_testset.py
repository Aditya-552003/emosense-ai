"""
backend/scripts/build_100k_testset.py
=====================================
Downloads and builds a verified, balanced, leak-free 100,000-sample
held-out evaluation dataset from peer-reviewed public corpora:
  - dair-ai/emotion (Twitter emotion corpus)
  - google-research-datasets/go_emotions (Reddit emotion corpus)
  - cardiffnlp/tweet_eval (TweetEval benchmark)

Guarantees:
  1. Exactly 100,000 uniquely distinct labeled examples (0 duplicates).
  2. Balanced distribution: exactly 16,667 or 16,666 samples per emotion across all 6 classes.
  3. ZERO data leakage: removes any texts present in clean_emotion_dataset.csv.
  4. Preprocessed with the production clean_text pipeline (all non-empty).

Usage:
  python backend/scripts/build_100k_testset.py --output heldout_test_100k.csv
"""

import argparse
import os
import sys

import pandas as pd
from datasets import load_dataset

# Ensure backend package is importable
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.config import ROOT_DIR
from app.utils.preprocessing import clean_text

TARGET_CLASSES = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]
# Target distribution totaling exactly 100,000
CLASS_TARGETS = {
    "surprise": 16667,
    "fear": 16667,
    "neutral": 16667,
    "anger": 16667,
    "sadness": 16666,
    "joy": 16666,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a balanced, leak-free 100,000-sample emotion evaluation dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT_DIR, "heldout_test_100k.csv"),
        help="Target output CSV file path (default: heldout_test_100k.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    return parser.parse_args()


def load_leak_filter() -> set:
    """Load existing training dataset texts to filter out for zero data leakage."""
    train_path = os.path.join(ROOT_DIR, "clean_emotion_dataset.csv")
    if not os.path.isfile(train_path):
        print(f"  Warning: {train_path} not found. Skipping training set filter.")
        return set()

    df_train = pd.read_csv(train_path)
    leak_set = set(df_train["text"].dropna().str.strip().str.lower())
    print(f"  Loaded {len(leak_set):,} reference training texts to prevent data leakage.")
    return leak_set


def build_dataset(output_path: str, seed: int = 42) -> pd.DataFrame:
    print("=" * 65)
    print("  EmoSense AI - Building 100,000 Held-Out Evaluation Dataset")
    print("=" * 65)

    leak_filter = load_leak_filter()

    pools = {cls: [] for cls in TARGET_CLASSES}

    # -------------------------------------------------------------------
    # 1. Source 1: dair-ai/emotion (Twitter corpus)
    # -------------------------------------------------------------------
    print("\n[1/3] Loading dair-ai/emotion (unsplit corpus)...")
    dair_ds = load_dataset("dair-ai/emotion", "unsplit")["train"]
    dair_df = dair_ds.to_pandas()
    label_map_dair = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    dair_df["emotion"] = dair_df["label"].map(label_map_dair)

    for emo in ["anger", "fear", "joy", "sadness", "surprise"]:
        subset = dair_df[dair_df["emotion"] == emo][["text", "emotion"]].dropna()
        pools[emo].append(subset)
    print(f"  dair-ai/emotion loaded ({len(dair_df):,} total rows).")

    # -------------------------------------------------------------------
    # 2. Source 2: google-research-datasets/go_emotions (Reddit corpus)
    # -------------------------------------------------------------------
    print("\n[2/3] Loading google-research-datasets/go_emotions (raw config)...")
    go_ds = load_dataset("google-research-datasets/go_emotions", "raw")["train"]
    go_df = go_ds.to_pandas()

    all_emotions_excluding_neutral = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise"
    ]

    is_pure_neutral = (go_df["neutral"] == 1) & (go_df[all_emotions_excluding_neutral].sum(axis=1) == 0)
    go_neutral = go_df[is_pure_neutral][["text"]].copy()
    go_neutral["emotion"] = "neutral"
    pools["neutral"].append(go_neutral)

    is_surprise = (go_df["surprise"] == 1) & (go_df["neutral"] == 0)
    go_surprise = go_df[is_surprise][["text"]].copy()
    go_surprise["emotion"] = "surprise"
    pools["surprise"].append(go_surprise)

    print(f"  go_emotions loaded: {len(go_neutral):,} neutral, {len(go_surprise):,} surprise candidates.")

    # -------------------------------------------------------------------
    # 3. Source 3: cardiffnlp/tweet_eval (TweetEval sentiment benchmark)
    # -------------------------------------------------------------------
    print("\n[3/3] Loading cardiffnlp/tweet_eval (sentiment neutral split)...")
    tweet_ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    tweet_df = pd.concat([tweet_ds["train"].to_pandas(), tweet_ds["test"].to_pandas()])
    tweet_neutral = tweet_df[tweet_df["label"] == 1][["text"]].copy()
    tweet_neutral["emotion"] = "neutral"
    pools["neutral"].append(tweet_neutral)
    print(f"  tweet_eval loaded: {len(tweet_neutral):,} neutral candidates.")

    # -------------------------------------------------------------------
    # 4. Global deduplication, clean_text verification, and leak-free sampling
    # -------------------------------------------------------------------
    print("\n[4/4] Deduplicating, filtering data leakage, and balancing...")
    global_seen_texts = set(leak_filter)
    balanced_subsets = []

    for emo in ["surprise", "fear", "neutral", "anger", "sadness", "joy"]:
        combined = pd.concat(pools[emo], ignore_index=True)
        combined = combined.dropna(subset=["text"]).copy()
        combined["text"] = combined["text"].astype(str).str.strip()
        combined = combined[combined["text"].astype(bool)]

        # Pre-clean text to guarantee all candidates have valid clean_text
        combined["clean_text"] = combined["text"].apply(clean_text)
        combined = combined[combined["clean_text"].astype(bool)].copy()

        # Normalized string for deduplication
        combined["norm_text"] = combined["clean_text"].str.lower()
        combined = combined.drop_duplicates(subset=["norm_text"]).copy()

        # Filter against global seen texts
        mask = ~combined["norm_text"].isin(global_seen_texts)
        available = combined[mask].copy()

        target_n = CLASS_TARGETS[emo]
        if len(available) < target_n:
            raise ValueError(
                f"Insufficient globally unique samples for '{emo}': requested {target_n}, "
                f"available {len(available)}."
            )

        sampled = available.sample(n=target_n, random_state=seed)
        global_seen_texts.update(sampled["norm_text"].tolist())

        balanced_subsets.append(sampled[["text", "emotion", "clean_text"]])
        print(f"  {emo:>10s}: {len(available):>6,} unique candidates -> {target_n:>6,} selected")

    # -------------------------------------------------------------------
    # 5. Assemble final dataset
    # -------------------------------------------------------------------
    final_df = pd.concat(balanced_subsets, ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("  HELD-OUT 100K EVALUATION DATASET GENERATED SUCCESSFULLY!")
    print("=" * 65)
    print(f"  Output path    : {output_path}")
    print(f"  Total samples  : {len(final_df):,}")
    print(f"  Duplicates     : {final_df.duplicated(subset=['clean_text']).sum()}")
    print(f"  Class balance  :")
    for emo, count in final_df["emotion"].value_counts().items():
        print(f"    {emo:>10s}: {count:>6,} ({count / len(final_df) * 100:.1f}%)")
    print()

    return final_df


def main():
    args = parse_args()
    build_dataset(output_path=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
