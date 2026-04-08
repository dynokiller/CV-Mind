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

# ═══════════════════════════════════════════════════════════════════════════════
#  Hugging Face Space Integration
# ═══════════════════════════════════════════════════════════════════════════════
HF_API_URL = "https://dyno0126-resume.hf.space/predict"

def get_model():
    """Dummy fallback for SHAP if it expects local model."""
    return None


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

    import requests

    # ── Remote API Routing to Hugging Face ──────────────────────────────────────
    try:
        response = requests.post(
            HF_API_URL, 
            data={"inputs": resume_text}, 
            timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            predicted_domain = result.get("predicted_domain", "Unknown")
            confidence = result.get("confidence", 0.0)
            all_probs = result.get("all_probabilities", {})
        else:
            raise Exception(f"HF API returned {response.status_code}")
    except Exception as e:
        logger.error(f"[Inference] Failed to reach HF Space: {e}")
        predicted_domain = "Unknown"
        confidence = 0.0
        all_probs = {}

    # 4. Rule-based override for subdomains missing from training data
    predicted_domain, confidence = _apply_domain_override(
        resume_text.lower(), predicted_domain, confidence, all_probs
    )

    return {
        "predicted_domain":  predicted_domain,
        "confidence":        round(confidence, 4),
        "all_probabilities": all_probs,
        "backend":           "huggingface_api",
    }

def get_feature_flags() -> dict:
    """Return the currently active feature flags."""
    return {
        "USE_OCR":                    USE_OCR,
        "USE_LINKEDIN":               USE_LINKEDIN,
        "USE_TRANSFORMER_CLASSIFIER": USE_TRANSFORMER_CLASSIFIER,
    }
