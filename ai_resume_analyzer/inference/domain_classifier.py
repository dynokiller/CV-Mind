"""
inference/domain_classifier.py

Unified Domain Classifier — two pluggable backends.

Backend A (default): XGBoost + TF-IDF  (existing V4 pipeline, <10 MB, fast)
Backend B (opt-in) : DistilBERT zero-shot classification via HuggingFace
                     Transformers pipeline.  No fine-tuning, no large checkpoints
                     stored in the repo.  The ~260 MB model is cached in
                     ~/.cache/huggingface/ on first use.

Environment flags:
    USE_TRANSFORMER_CLASSIFIER=true   → use DistilBERT zero-shot
    USE_TRANSFORMER_CLASSIFIER=false  → use XGBoost (default)

Both backends return the same dict shape:
    {
        "predicted_domain":  str,
        "confidence":        float,
        "all_probabilities": {domain: float, ...},
        "backend":           "xgboost" | "transformers"
    }
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger("resume_analyzer.domain_classifier")

# ── Feature flag ───────────────────────────────────────────────────────────────
USE_TRANSFORMER_CLASSIFIER = (
    os.getenv("USE_TRANSFORMER_CLASSIFIER", "false").lower() == "true"
)

# ── Canonical domain labels (used by both backends) ──────────────────────────
DOMAIN_LABELS = [
    "INFORMATION-TECHNOLOGY",
    "ENGINEERING",
    "FINANCE",
    "BANKING",
    "ACCOUNTANT",
    "HR",
    "HEALTHCARE",
    "SALES",
    "BUSINESS-DEVELOPMENT",
    "DESIGNER",
    "ARTS",
    "AVIATION",
    "DIGITAL-MEDIA",
    "PUBLIC-RELATIONS",
    "CONSULTANT",
    "CONSTRUCTION",
    "AUTOMOBILE",
    "AGRICULTURE",
    "APPAREL",
    "FITNESS",
    "ADVOCATE",
    "CHEF",
    "BPO",
    "TEACHER",
]

# ── Human-readable descriptions fed to the zero-shot model ───────────────────
_DOMAIN_DESCRIPTIONS = {
    "INFORMATION-TECHNOLOGY": "software engineering, coding, cybersecurity, cloud, data science, machine learning, networking",
    "ENGINEERING":            "mechanical, electrical, civil, hardware engineering, systems design",
    "FINANCE":                "financial analysis, investment, wealth management, portfolio, equity research",
    "BANKING":                "bank, teller, loan officer, credit, mortgage, retail banking",
    "ACCOUNTANT":             "accounting, auditing, taxation, bookkeeping, CPA, tally",
    "HR":                     "human resources, recruitment, talent acquisition, employee relations, payroll",
    "HEALTHCARE":             "doctor, nurse, physician, medical, clinical, hospital, pharmacy",
    "SALES":                  "sales, business development, account executive, retail, client relations",
    "BUSINESS-DEVELOPMENT":   "business strategy, partnerships, market development, growth",
    "DESIGNER":               "ui/ux design, graphic design, product design, visual design",
    "ARTS":                   "art, music, animation, creative, film, photography",
    "AVIATION":               "pilot, aviation, aircraft, flight, air traffic",
    "DIGITAL-MEDIA":          "social media, content creation, journalism, video production, writing",
    "PUBLIC-RELATIONS":       "public relations, PR, communications, press, media relations",
    "CONSULTANT":             "consulting, management consulting, advisory, strategy",
    "CONSTRUCTION":           "construction, building, architecture, contractor, site engineer",
    "AUTOMOBILE":             "automotive, car, vehicle, motor, manufacturing",
    "AGRICULTURE":            "farming, agriculture, crop, agri, horticulture",
    "APPAREL":                "fashion, clothing, garment, textile, apparel",
    "FITNESS":                "fitness, gym, personal training, sports coaching, wellness",
    "ADVOCATE":               "law, legal, advocate, attorney, paralegal, litigation",
    "CHEF":                   "chef, cooking, culinary, kitchen, restaurant, food",
    "BPO":                    "customer service, call centre, BPO, support, helpdesk",
    "TEACHER":                "teaching, education, professor, tutor, instructor, academic",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Backend A — XGBoost (delegates to existing inference.py singleton)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_domain_xgboost(text: str) -> Dict[str, Any]:
    """
    Delegate to the existing V4 XGBoost inference pipeline.
    Adds a 'backend' key to the output.
    """
    from inference.inference import predict_domain  # local import avoids circular
    result = predict_domain(text)
    result["backend"] = "xgboost"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Backend B — DistilBERT zero-shot
# ═══════════════════════════════════════════════════════════════════════════════

_transformer_pipeline = None   # lazy-loaded singleton


def _get_transformer_pipeline():
    """
    Lazy-load the zero-shot classification pipeline (singleton).
    Model: facebook/bart-large-mnli (MNLI fine-tuned, ~1.2 GB)  ← too large
    Better: cross-encoder/nli-MiniLM2-L6-H768  (~90 MB) — compact and fast.
    """
    global _transformer_pipeline
    if _transformer_pipeline is None:
        try:
            from transformers import pipeline
            logger.info(
                "[Classifier] Loading zero-shot pipeline "
                "(cross-encoder/nli-MiniLM2-L6-H768). First run may download ~90 MB."
            )
            _transformer_pipeline = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-MiniLM2-L6-H768",
                device=-1,   # CPU
            )
            logger.info("[Classifier] Transformer pipeline ready.")
        except Exception as e:
            logger.error(f"[Classifier] Transformer pipeline failed to load: {e}")
    return _transformer_pipeline


def classify_domain_transformer(text: str) -> Dict[str, Any]:
    """
    Classify domain using DistilBERT/MiniLM zero-shot classification.

    Sends candidate descriptions for each domain label so the model can
    match the resume content to the most relevant one without any fine-tuning.
    """
    pipe = _get_transformer_pipeline()
    if pipe is None:
        raise RuntimeError("Transformer pipeline is not available.")

    # Use at most 512 tokens worth of text (truncate to ~1800 chars)
    truncated_text = text[:1800]

    candidate_labels = list(_DOMAIN_DESCRIPTIONS.values())
    label_keys       = list(_DOMAIN_DESCRIPTIONS.keys())

    output = pipe(
        truncated_text,
        candidate_labels=candidate_labels,
        multi_label=False,
    )

    # Map description → domain key
    desc_to_key = {v: k for k, v in _DOMAIN_DESCRIPTIONS.items()}
    all_probs: Dict[str, float] = {}
    for label, score in zip(output["labels"], output["scores"]):
        domain_key = desc_to_key.get(label, label)
        all_probs[domain_key] = round(float(score), 4)

    # Best prediction
    best_desc  = output["labels"][0]
    best_key   = desc_to_key.get(best_desc, best_desc)
    confidence = round(float(output["scores"][0]), 4)

    return {
        "predicted_domain":  best_key,
        "confidence":        confidence,
        "all_probabilities": all_probs,
        "backend":           "transformers",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Unified public API
# ═══════════════════════════════════════════════════════════════════════════════

def predict_domain_unified(text: str) -> Dict[str, Any]:
    """
    Route domain classification through the active backend.

    Fallback chain:
      1. If USE_TRANSFORMER_CLASSIFIER=true  → try transformer → fall back to XGBoost
      2. Otherwise                           → use XGBoost directly

    Args:
        text: Plain text resume or job description.

    Returns:
        {predicted_domain, confidence, all_probabilities, backend}
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    if USE_TRANSFORMER_CLASSIFIER:
        try:
            return classify_domain_transformer(text)
        except Exception as e:
            logger.warning(
                f"[Classifier] Transformer backend failed ({e}). "
                "Falling back to XGBoost."
            )

    return classify_domain_xgboost(text)


def get_available_domains() -> list:
    """Return the list of all supported domain labels."""
    return DOMAIN_LABELS.copy()
