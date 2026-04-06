from typing import Tuple

from .utils.advanced_parser import extract_text_from_pdf as _adv_extract_pdf
from .utils.pdf_parser import extract_text_from_pdf as _plumber_extract_pdf
from .utils.text_cleaner import clean_text as _basic_clean


def extract_raw_text(file_bytes: bytes) -> str:
    """
    Robust PDF text extraction with fast fallbacks.

    Strategy:
    - Try PyMuPDF-based extractor (fast, layout-aware text)
    - If result is empty/very short, fall back to pdfplumber-based extractor
    - Return raw text (with newlines preserved) for downstream section parsing
    """
    text = _adv_extract_pdf(file_bytes)

    # If almost nothing was extracted, try the alternate implementation
    if not text or len(text.strip()) < 50:
        fallback = _plumber_extract_pdf(file_bytes)
        if fallback and len(fallback.strip()) > len(text.strip()):
            text = fallback

    return text or ""


def normalize_pdf_text(text: str) -> str:
    """
    Light-weight normalization for PDF text before section parsing.

    - Normalize line breaks and whitespace
    - Remove obvious control characters
    - Keep case and punctuation (section parser relies on them)
    """
    if not text:
        return ""

    # Replace Windows line endings and carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printable control chars
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())

    # Collapse multiple blank lines but keep paragraph boundaries
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned_lines = []
    blank_streak = 0
    for line in lines:
        if line.strip():
            cleaned_lines.append(line)
            blank_streak = 0
        else:
            if blank_streak == 0:
                cleaned_lines.append("")
            blank_streak += 1

    normalized = "\n".join(cleaned_lines).strip()
    return normalized


def extract_and_normalize(file_bytes: bytes) -> Tuple[str, str]:
    """
    Convenience function used by the API:

    Returns:
        (raw_text, normalized_text)
    """
    raw = extract_raw_text(file_bytes)
    normalized = normalize_pdf_text(raw)
    return raw, normalized

