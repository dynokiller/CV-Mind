"""
inference/similarity.py

Resume ↔ Job Description similarity using SentenceTransformers.

Model: all-MiniLM-L6-v2  (fast, accurate, 80MB)
Method: Cosine similarity → normalised to [0, 100] percentage

The SentenceTransformer model is loaded ONCE as a module-level singleton.

Usage:
    from inference.similarity import compute_match_score
    result = compute_match_score(resume_text, job_description)
"""

import re
from sentence_transformers import SentenceTransformer, util

# ── Singleton model load ───────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

print(f"[Similarity] Loading SentenceTransformer: {EMBED_MODEL_NAME}")
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("[Similarity] Model ready.")


def _extract_keywords(text: str, top_n: int = 30) -> list[str]:
    """
    Lightweight keyword extraction: strips stopwords and returns
    the most frequent meaningful tokens.
    """
    stopwords = {
        "a", "an", "the", "and", "or", "but", "for", "in", "on", "at",
        "to", "of", "is", "are", "was", "were", "with", "this", "that",
        "from", "by", "as", "it", "we", "our", "your", "be", "will",
        "have", "has", "had", "not", "can", "you", "he", "she", "they",
    }
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#\-\.]+\b", text.lower())
    freq  = {}
    for w in words:
        if w not in stopwords and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq, key=freq.get, reverse=True)
    return sorted_words[:top_n]


def compute_match_score(resume_text: str, job_description: str) -> dict:
    """
    Compute semantic similarity between resume and job description.

    Args:
        resume_text:     Plain text of the resume.
        job_description: Plain text of the job description.

    Returns:
        {
            "match_score_percent": 82.3,
            "matched_keywords": ["Python", "Django", "REST"],
            "missing_keywords": ["Kubernetes", "Docker"],
            "similarity_method": "cosine",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    """
    if not resume_text.strip() or not job_description.strip():
        raise ValueError("Both resume_text and job_description must be non-empty.")

    # ── Semantic similarity ───────────────────────────────────────────────────
    resume_emb = _embed_model.encode(resume_text,      convert_to_tensor=True)
    jd_emb     = _embed_model.encode(job_description,  convert_to_tensor=True)

    cosine_sim        = float(util.cos_sim(resume_emb, jd_emb).item())
    match_score_pct   = round(max(0.0, min(cosine_sim, 1.0)) * 100, 2)

    # ── Keyword-level diff ────────────────────────────────────────────────────
    resume_keywords = set(_extract_keywords(resume_text))
    jd_keywords     = set(_extract_keywords(job_description))

    matched  = sorted(resume_keywords & jd_keywords)
    missing  = sorted(jd_keywords - resume_keywords)

    return {
        "match_score_percent": match_score_pct,
        "matched_keywords":    matched[:20],   # top 20 to keep response compact
        "missing_keywords":    missing[:20],
        "similarity_method":   "cosine",
        "embedding_model":     EMBED_MODEL_NAME,
    }
