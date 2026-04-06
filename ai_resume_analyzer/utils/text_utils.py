"""
utils/text_utils.py

Shared text cleaning utilities used by both the training pipeline
and at inference time. Keeps preprocessing consistent across both stages.
"""

import re
import string


# ── Common English stopwords (lightweight, no NLTK dependency) ────────────────
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "for", "in", "on", "at", "to",
    "of", "is", "are", "was", "were", "with", "this", "that", "from", "by",
    "as", "it", "we", "our", "your", "be", "will", "have", "has", "had",
    "not", "can", "you", "he", "she", "they", "i", "my", "am", "do", "did",
    "so", "up", "if", "than", "then", "while", "about", "which",
}


def clean_text(text: str) -> str:
    """
    Standard text cleaning pipeline:
      1. Lowercase
      2. Strip HTML tags
      3. Remove URLs
      4. Remove non-ASCII characters
      5. Remove punctuation (except hyphens, dots, +, # for tech terms)
      6. Collapse extra whitespace

    Returns cleaned string.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)             # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)       # URLs
    text = re.sub(r"[^\x00-\x7f]", " ", text)         # Non-ASCII
    text = re.sub(r"[^a-z0-9\s\.,\-\+#]", " ", text)  # Keep only safe chars
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_keywords(text: str, top_n: int = 30) -> list[str]:
    """
    Simple frequency-based keyword extraction.

    Removes stopwords and short tokens (<3 chars).
    Returns list of top_n highest-frequency words.
    """
    text  = clean_text(text)
    words = re.findall(r"\b[a-z][a-z0-9\+#\-\.]{2,}\b", text)

    freq: dict[str, int] = {}
    for w in words:
        if w not in _STOPWORDS:
            freq[w] = freq.get(w, 0) + 1

    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)][:top_n]


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to a maximum character count at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated
