"""
scripts/test_inference.py
=========================
Quick smoke-test for the full inference pipeline.

Loads all models and runs a test prediction to verify that the
XGBoost, RoBERTa, Soft-Voting, and Stacking ensemble branches produce valid output.

Usage:
  cd backend
  python -m scripts.test_inference
"""

import os
import pickle
import sys

import joblib
import numpy as np
import torch
from scipy.sparse import hstack
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure the backend package is importable
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.config import MODEL_DIR
from app.utils.preprocessing import clean_text, download_nrc_lexicon, load_nrc_lexicon, nrc_features


def main():
    # 1. Download NRC Lexicon if not exists
    nrc_path = str(MODEL_DIR / "nrc_lexicon.txt")
    download_nrc_lexicon(nrc_path)

    # 2. Parse NRC Lexicon
    print("Parsing NRC Emotion Lexicon...")
    nrc_dict, nrc_emotions = load_nrc_lexicon(nrc_path)
    print("NRC Emotions:", nrc_emotions)

    # 3. Load Models
    print("Loading semantic models...")
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgb.pkl"))
    word_tfidf = joblib.load(os.path.join(MODEL_DIR, "word_tfidf.pkl"))
    char_tfidf = joblib.load(os.path.join(MODEL_DIR, "char_tfidf.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "nrc_scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    meta_clf = joblib.load(os.path.join(MODEL_DIR, "meta_clf.pkl"))

    print("Loading contextual model (RoBERTa)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    roberta_dir = os.path.join(MODEL_DIR, "roberta_best")
    tokenizer = AutoTokenizer.from_pretrained(roberta_dir)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_dir).to(device)
    roberta_model.eval()

    # 4. Test Text
    test_text = "I am so happy and excited today! Everything is going perfectly!"
    print(f"\nInput: '{test_text}'")

    clean_txt = clean_text(test_text)
    print(f"Cleaned Input: '{clean_txt}'")

    # --- XGBoost Branch
    x_w = word_tfidf.transform([clean_txt])
    x_c = char_tfidf.transform([clean_txt])
    x_n = scaler.transform(nrc_features([clean_txt], nrc_dict, nrc_emotions))
    x_sem = hstack([x_w, x_c, x_n])
    sem_probs = xgb.predict_proba(x_sem)
    print("XGBoost probabilities:", sem_probs)

    # --- RoBERTa Branch
    enc = tokenizer([test_text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = roberta_model(**enc).logits
        ctx_probs = torch.softmax(logits, dim=1).cpu().numpy()
    print("RoBERTa probabilities:", ctx_probs)

    # --- Soft-Voting Branch (Alpha = 0.7)
    soft_probs = 0.7 * ctx_probs[0] + 0.3 * sem_probs[0]
    soft_probs = soft_probs / np.sum(soft_probs)
    soft_idx = int(np.argmax(soft_probs))
    soft_label = label_encoder.classes_[soft_idx]
    print("\n--- Soft-Voting Prediction (Alpha = 0.7) ---")
    print(f"Emotion: {soft_label} (Confidence: {soft_probs[soft_idx]:.2%})")

    # --- Stacking Meta-Learner Branch
    meta_X = np.hstack([ctx_probs, sem_probs])
    meta_final_probs = meta_clf.predict_proba(meta_X)
    pred_idx = int(np.argmax(meta_final_probs[0]))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = meta_final_probs[0][pred_idx]

    print("\n--- Stacking Meta-Learner Prediction ---")
    print(f"Emotion: {pred_label} (Confidence: {confidence:.2%})")
    print("Probabilities per class:")
    for name, p in zip(label_encoder.classes_, meta_final_probs[0]):
        print(f"  {name}: {p:.2%}")


if __name__ == "__main__":
    main()
