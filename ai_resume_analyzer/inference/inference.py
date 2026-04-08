"""
inference/inference.py

Singleton inference layer for the V4 XGBoost + TF-IDF domain classifier.

The models are loaded ONCE when the module is first imported and reused
for every subsequent call — no per-request reloading.

Feature flags (environment variables):
    USE_OCR                    = true/false  (default: true)
    USE_LINKEDIN               = true/false  (default: false)
    USE_TRANSFORMER_CLASSIFIER = true/false  (default: false)

Usage:
    from inference.inference import predict_domain
    result = predict_domain("Experienced Python developer with Django...")
"""

import os
import re
import pickle
import logging
import numpy as np
import xgboost as xgb
from training.advanced_text_cleaner import clean_and_lemmatize

logger = logging.getLogger("resume_analyzer.inference")

# ── Feature flags ─────────────────────────────────────────────────────────────
USE_OCR = os.getenv("USE_OCR", "true").lower() == "true"
USE_LINKEDIN = os.getenv("USE_LINKEDIN", "false").lower() == "true"
USE_TRANSFORMER_CLASSIFIER = (
    os.getenv("USE_TRANSFORMER_CLASSIFIER", "false").lower() == "true"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR          = "models/v4_xgboost"
LABEL_ENCODER_PATH = f"{MODEL_DIR}/label_encoder.pkl"
TFIDF_PATH         = f"{MODEL_DIR}/tfidf_vectorizer.pkl"
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
        logger.info(f"[Inference] Loading V4 XGBoost model from: {MODEL_DIR}")

        if not (
            os.path.exists(XGB_PATH)
            and os.path.exists(LABEL_ENCODER_PATH)
            and os.path.exists(TFIDF_PATH)
        ):
            raise FileNotFoundError(
                f"V4 Models not found in '{MODEL_DIR}'. "
                f"Run 'python -m training.train_advanced' first."
            )

        # 1. Load TF-IDF Vectorizer
        with open(TFIDF_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

        # 2. Load XGBoost Classifier
        self.model = xgb.XGBClassifier()
        self.model.load_model(XGB_PATH)

        # 3. Load Label Encoder
        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.class_names = list(self.label_encoder.classes_)
        logger.info(
            f"[Inference] V4 Model loaded. "
            f"Classes: {len(self.class_names)} | "
            f"USE_OCR={USE_OCR} | USE_LINKEDIN={USE_LINKEDIN} | "
            f"USE_TRANSFORMER_CLASSIFIER={USE_TRANSFORMER_CLASSIFIER}"
        )


# ── Module-level singleton — created on first import ─────────────────────────
_model = _ModelSingleton()


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain Override Rules
# ═══════════════════════════════════════════════════════════════════════════════
def _count_signals(text_lower: str, keywords: list) -> int:
    """Count how many keywords appear in the text."""
    return sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))


def _apply_domain_override(text_lower: str, predicted: str, confidence: float, all_probs: dict):
    """
    Rule-based post-processing that overrides XGBoost predictions for specific
    subdomains not well represented in the training dataset.

    A rule fires only if enough strong signals are present (score threshold >= 3).
    This prevents false overrides on resumes that merely mention IT in passing.
    """
    # ── IT / Cyber Security / Software ─────────────────────────────────────
    it_signals = [
        'cybersecurity', 'cyber security', 'penetration testing', 'ethical hacking',
        'owasp', 'burp suite', 'nmap', 'metasploit', 'kali linux', 'wireshark',
        'cissp', 'ceh', 'siem', 'splunk', 'qradar', 'firewall', 'vulnerability',
        'network security', 'ccna', 'information security', 'infosec',
        'ansible', 'openid', 'oauth', 'jwt', 'active directory',
        'software developer', 'software engineer', 'full stack', 'backend',
        'frontend', 'web developer', 'react', 'node.js', 'django', 'flask',
        'computer science', 'btech', 'b.tech', 'bachelor technology',
        'deep learning', 'machine learning', 'artificial intelligence',
        'pytorch', 'tensorflow', 'transformers', 'llm',
        'cloud computing', 'aws', 'docker', 'kubernetes', 'devops',
        'data science', 'data engineer', 'data analyst',
    ]
    it_score = _count_signals(text_lower, it_signals)

    # ── Finance / Banking ───────────────────────────────────────────────────
    finance_signals = [
        'chartered accountant', 'cpa', 'financial analyst', 'investment banker',
        'portfolio management', 'equity research', 'derivatives', 'fixed income',
        'hedge fund', 'balance sheet', 'profit loss', 'auditor', 'taxation', 'tally',
    ]
    finance_score = _count_signals(text_lower, finance_signals)

    # ── HR ──────────────────────────────────────────────────────────────────
    hr_signals = [
        'recruitment', 'talent acquisition', 'onboarding', 'employee relations',
        'payroll management', 'performance management', 'hris', 'human resources business',
    ]
    hr_score = _count_signals(text_lower, hr_signals)

    # ── Apply overrides ──────────────────────────────────────────────────────
    if it_score >= 3 and predicted not in ('INFORMATION-TECHNOLOGY', 'ENGINEERING'):
        it_prob  = all_probs.get('INFORMATION-TECHNOLOGY', 0)
        eng_prob = all_probs.get('ENGINEERING', 0)
        if it_prob >= eng_prob:
            return 'INFORMATION-TECHNOLOGY', max(confidence, it_prob + 0.05)
        return 'ENGINEERING', max(confidence, eng_prob + 0.05)

    if finance_score >= 3 and predicted not in ('FINANCE', 'BANKING', 'ACCOUNTANT'):
        return 'FINANCE', max(confidence, all_probs.get('FINANCE', 0) + 0.05)

    if hr_score >= 3 and predicted != 'HR':
        return 'HR', max(confidence, all_probs.get('HR', 0) + 0.05)

    return predicted, confidence


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════
def predict_domain(resume_text: str) -> dict:
    """
    Run domain classification on raw resume text.

    Routing:
      - USE_TRANSFORMER_CLASSIFIER=true  → attempts DistilBERT zero-shot first,
                                           falls back to XGBoost on any error.
      - USE_TRANSFORMER_CLASSIFIER=false → XGBoost V4 (default, fast, lightweight).

    Args:
        resume_text: Plain text extracted from a resume.

    Returns:
        {
            "predicted_domain": "INFORMATION-TECHNOLOGY",
            "confidence": 0.94,
            "all_probabilities": { ... },
            "backend": "xgboost" | "transformers"
        }
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text cannot be empty.")

    # ── Route to transformer backend if requested ──────────────────────────
    if USE_TRANSFORMER_CLASSIFIER:
        try:
            from inference.domain_classifier import classify_domain_transformer
            result = classify_domain_transformer(resume_text)
            logger.info(
                f"[Inference] Transformer: {result['predicted_domain']} "
                f"({result['confidence']:.2f})"
            )
            return result
        except Exception as e:
            logger.warning(
                f"[Inference] Transformer backend failed ({e}). "
                "Falling back to XGBoost."
            )

    # ── XGBoost V4 pipeline ────────────────────────────────────────────────
    # 1. SpaCy NLP cleaning + skill injection
    cleaned_text = clean_and_lemmatize(resume_text)

    # 2. Sparse TF-IDF transform
    embedding_vector = _model.vectorizer.transform([cleaned_text])

    # 3. XGBoost Prediction
    probs = _model.model.predict_proba(embedding_vector)[0]

    pred_idx         = int(np.argmax(probs))
    predicted_domain = _model.class_names[pred_idx]
    confidence       = float(probs[pred_idx])

    all_probs = {
        name: round(float(p), 4)
        for name, p in zip(_model.class_names, probs)
    }

    # 4. Rule-based override for subdomains missing from training data
    predicted_domain, confidence = _apply_domain_override(
        resume_text.lower(), predicted_domain, confidence, all_probs
    )

    return {
        "predicted_domain":  predicted_domain,
        "confidence":        round(confidence, 4),
        "all_probabilities": all_probs,
        "backend":           "xgboost",
    }


def get_model():
    """Return the singleton — used by SHAP explainer to access model internals."""
    return _model


def get_feature_flags() -> dict:
    """Return the currently active feature flags."""
    return {
        "USE_OCR":                    USE_OCR,
        "USE_LINKEDIN":               USE_LINKEDIN,
        "USE_TRANSFORMER_CLASSIFIER": USE_TRANSFORMER_CLASSIFIER,
    }
