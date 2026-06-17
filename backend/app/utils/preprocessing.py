"""
preprocessing.py
----------------
Text cleaning and NRC Emotion Lexicon feature engineering pipeline.
Mirrors the exact preprocessing used during model training.
"""

import os
import re
import urllib.request
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# NRC Lexicon download mirror (public, non-gated GitHub mirror)
# ---------------------------------------------------------------------------
NRC_MIRROR_URL = (
    "https://raw.githubusercontent.com/aditeyabaral/"
    "lok-sabha-election-twitter-analysis/master/"
    "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
)


def download_nrc_lexicon(dest_path: str) -> None:
    """Download the NRC Emotion Lexicon from a public mirror if not present."""
    if os.path.exists(dest_path):
        return
    print(f"[startup] Downloading NRC Emotion Lexicon -> {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(NRC_MIRROR_URL, dest_path)
    print("[startup] NRC Lexicon downloaded successfully.")


def load_nrc_lexicon(lexicon_path: str) -> Tuple[Dict, List[str]]:
    """
    Parse the NRC word-level lexicon file.

    Returns
    -------
    nrc_dict     : dict mapping word -> list of emotion indices
    nrc_emotions : sorted list of emotion category names (length 10)
    """
    nrc = pd.read_csv(lexicon_path, sep="\t", names=["word", "emotion", "association"])
    nrc = nrc[nrc["association"] == 1]
    nrc_emotions: List[str] = sorted(nrc["emotion"].unique().tolist())
    emotion2idx: Dict[str, int] = {e: i for i, e in enumerate(nrc_emotions)}

    nrc_dict: Dict[str, List[int]] = defaultdict(list)
    for _, row in nrc.iterrows():
        nrc_dict[row["word"]].append(emotion2idx[row["emotion"]])

    return dict(nrc_dict), nrc_emotions


def clean_text(text: str) -> str:
    """
    Lowercase, strip URLs, and keep only ASCII letters and spaces.
    Must match the training-time clean_text() function exactly.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def nrc_features(
    texts: List[str],
    nrc_dict: Dict[str, List[int]],
    nrc_emotions: List[str],
) -> np.ndarray:
    """
    Compute NRC emotion density features for a list of pre-cleaned texts.

    Each feature = count of emotion tokens / total token count,
    matching the training-time nrc_features() function.

    Returns
    -------
    np.ndarray of shape (len(texts), len(nrc_emotions))
    """
    feats = np.zeros((len(texts), len(nrc_emotions)), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = text.split()
        n_tokens = max(len(tokens), 1)
        for token in set(tokens):
            for emo_idx in nrc_dict.get(token, []):
                feats[i, emo_idx] += 1.0 / n_tokens
    return feats
