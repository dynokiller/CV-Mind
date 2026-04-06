"""
Service wrapper around MODEL 2 (OCR) and existing text extractors.

Responsibilities:
- Detect scanned vs digital PDFs
- Merge normal PDF/DOCX text with OCR text (including images in DOCX)
- Provide a simple `extract_text_from_file` API for the unified pipeline.
"""

from typing import Tuple

from resume_ai_model.utils.advanced_parser import (
    extract_text_from_pdf as extract_pdf_text_digital,
    extract_text_from_docx as extract_docx_text,
)
from models.ocr_model import ocr_image_bytes, ocr_scanned_pdf, ocr_images_in_docx


def extract_from_pdf(file_bytes: bytes) -> Tuple[str, str]:
    """
    Returns (combined_text, ocr_only_text)
    """
    digital_text = extract_pdf_text_digital(file_bytes)
    ocr_text = ocr_scanned_pdf(file_bytes)

    if digital_text and ocr_text:
        combined = digital_text.strip() + "\n\n" + ocr_text.strip()
    else:
        combined = (digital_text or "") + "\n\n" + (ocr_text or "")

    return combined.strip(), (ocr_text or "").strip()


def extract_from_docx(file_bytes: bytes) -> Tuple[str, str]:
    """
    Returns (combined_text, ocr_only_text) for DOCX files,
    including images extracted via zip (word/media/*).
    """
    text = extract_docx_text(file_bytes) or ""
    ocr_text = ocr_images_in_docx(file_bytes)
    if ocr_text:
        combined = text.strip() + "\n\n" + ocr_text.strip()
    else:
        combined = text
    return combined.strip(), (ocr_text or "").strip()


def extract_from_image(file_bytes: bytes) -> str:
    """Run OCR on an image-only resume."""
    return ocr_image_bytes(file_bytes)

