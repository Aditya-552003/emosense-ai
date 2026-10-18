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
# NRC Lexicon download mirror (pinned public mirror)
# ---------------------------------------------------------------------------
NRC_MIRROR_URL = (
    "https://raw.githubusercontent.com/aditeyabaral/"
    "lok-sabha-election-twitter-analysis/master/"
    "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
)


def download_nrc_lexicon(dest_path: str, timeout: int = 20) -> None:
    """
    Download the NRC Emotion Lexicon from a public mirror if not present locally.
    Validates that the resulting file is non-empty.
    """
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = f"{dest_path}.tmp"
    try:
        req = urllib.request.Request(
            NRC_MIRROR_URL,
            headers={"User-Agent": "EmoSense-AI-Downloader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(temp_path, "wb") as f_out:
                f_out.write(response.read())

        if os.path.getsize(temp_path) < 1000:
            raise ValueError("Downloaded NRC lexicon file is too small or corrupt.")

        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
    except Exception as exc:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to download NRC Emotion Lexicon to {dest_path}: {exc}. "
            "Please ensure the file exists at that path or provide internet access."
        ) from exc


def load_nrc_lexicon(lexicon_path: str) -> Tuple[Dict[str, List[int]], List[str]]:
    """
    Parse the NRC word-level lexicon file.

    Returns
    -------
    nrc_dict     : dict mapping word -> list of emotion indices
    nrc_emotions : sorted list of emotion category names (length 10)
    """
    if not os.path.exists(lexicon_path):
        raise FileNotFoundError(f"NRC Emotion Lexicon not found at {lexicon_path}")

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
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
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

    Each feature = count of distinct emotion tokens / total token count,
    matching the training-time nrc_features() function.

    Returns
    -------
    np.ndarray of shape (len(texts), len(nrc_emotions))
    """
    n_emotions = len(nrc_emotions)
    n_samples = len(texts)
    feats = np.zeros((n_samples, n_emotions), dtype=np.float32)

    for i, text in enumerate(texts):
        tokens = text.split()
        n_tokens = max(len(tokens), 1)
        scale = 1.0 / n_tokens
        unique_tokens = set(tokens)
        for token in unique_tokens:
            emo_indices = nrc_dict.get(token)
            if emo_indices is not None:
                for emo_idx in emo_indices:
                    feats[i, emo_idx] += scale
    return feats
