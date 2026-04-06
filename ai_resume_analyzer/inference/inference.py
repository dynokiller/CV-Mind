"""
inference/inference.py

Singleton inference layer for the V4 XGBoost + Embeddings domain classifier.

The models are loaded ONCE when the module is first imported and reused
for every subsequent call — no per-request reloading.

Usage:
    from inference.inference import predict_domain
    result = predict_domain("Experienced Python developer with Django...")
"""

import os
import pickle
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from training.advanced_text_cleaner import clean_and_lemmatize

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR          = "models/v4_xgboost"
LABEL_ENCODER_PATH = f"{MODEL_DIR}/label_encoder.pkl"
XGB_PATH           = f"{MODEL_DIR}/xgb_model.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  Singleton: load ONCE at module import time
# ═══════════════════════════════════════════════════════════════════════════════
class _ModelSingleton:
    """
    Internal singleton that holds all loaded V4 model artefacts.
    Instantiated once when the module is first loaded.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._load()
            cls._instance = obj
        return cls._instance

    def _load(self):
        print(f"[Inference] Loading V4 XGBoost model from: {MODEL_DIR}")

        if not os.path.exists(XGB_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
            raise FileNotFoundError(
                f"V4 Models not found in '{MODEL_DIR}'. "
                f"Run 'python -m training.train_advanced' first."
            )

        # 1. Load Sentence Transformer for embedding
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Load XGBoost Classifier
        self.model = xgb.XGBClassifier()
        self.model.load_model(XGB_PATH)

        # 3. Load Label Encoder
        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.class_names = list(self.label_encoder.classes_)
        print(f"[Inference] V4 Model loaded. Classes: {len(self.class_names)}")


# ── Module-level singleton — created on first import ─────────────────────────
_model = _ModelSingleton()


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════
def predict_domain(resume_text: str) -> dict:
    """
    Run domain classification on raw resume text utilizing the V4 architecture.

    Args:
        resume_text: Plain text extracted from a resume.

    Returns:
        {
            "predicted_domain": "Data Science",
            "confidence": 0.94,
            "all_probabilities": {
                "Data Science": 0.94,
                "HR": 0.03,
                ...
            }
        }
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text cannot be empty.")

    # 1. Deep NLP Spacy Cleaning
    cleaned_text = clean_and_lemmatize(resume_text)
    
    # 2. Dense Semantic Embedding
    embedding_vector = _model.embedder.encode([cleaned_text], device="cpu")

    # 3. XGBoost Prediction Probabilities
    probs = _model.model.predict_proba(embedding_vector)[0]

    # Build output
    pred_idx       = int(np.argmax(probs))
    predicted_domain = _model.class_names[pred_idx]
    confidence       = float(probs[pred_idx])

    all_probs = {
        name: round(float(p), 4)
        for name, p in zip(_model.class_names, probs)
    }

    return {
        "predicted_domain":  predicted_domain,
        "confidence":        round(confidence, 4),
        "all_probabilities": all_probs,
    }


def get_model():
    """Return the singleton — used by SHAP explainer to access model internals."""
    return _model
