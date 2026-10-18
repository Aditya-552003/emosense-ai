"""
tests/test_preprocessing.py — Unit tests for text cleaning and NRC feature engineering.
"""

import numpy as np
import pytest

from app.utils.preprocessing import clean_text, nrc_features


def test_clean_text_basic():
    assert clean_text("Hello World!") == "hello world"


def test_clean_text_urls():
    text = "Check this link http://example.com/test?a=1 for details."
    assert clean_text(text) == "check this link  for details"


def test_clean_text_numbers_and_punctuation():
    text = "I won $500 in 2026! That's awesome :)"
    assert clean_text(text) == "i won  in  thats awesome"


def test_clean_text_emojis_and_unicode():
    text = "So excited! 😄🎉 Let's go!"
    assert clean_text(text) == "so excited  lets go"


def test_clean_text_whitespace_and_newlines():
    text = "   Multi \n\n line \t string   "
    assert clean_text(text) == "multi \n\n line \t string"


def test_clean_text_empty_and_non_string():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""


def test_nrc_features_computation():
    nrc_dict = {
        "happy": [4],   # joy (index 4)
        "sad": [7],     # sadness (index 7)
    }
    nrc_emotions = [
        "anger", "anticipation", "disgust", "fear", "joy",
        "negative", "positive", "sadness", "surprise", "trust",
    ]

    texts = [
        "happy day",       # 2 tokens -> happy: 1/2 = 0.5
        "sad day",         # 2 tokens -> sad: 1/2 = 0.5
        "happy and sad",   # 3 tokens -> happy: 1/3, sad: 1/3
        "",                # 0 tokens -> empty (n_tokens=1)
    ]

    feats = nrc_features(texts, nrc_dict, nrc_emotions)
    assert feats.shape == (4, 10)

    # Row 0: happy (idx 4) = 0.5
    assert np.isclose(feats[0, 4], 0.5)
    assert np.isclose(feats[0, 7], 0.0)

    # Row 1: sad (idx 7) = 0.5
    assert np.isclose(feats[1, 7], 0.5)

    # Row 2: both happy and sad = 1/3
    assert np.isclose(feats[2, 4], 1.0 / 3.0)
    assert np.isclose(feats[2, 7], 1.0 / 3.0)

    # Row 3: empty
    assert np.all(feats[3] == 0.0)
