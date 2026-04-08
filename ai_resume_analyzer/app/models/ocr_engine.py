"""
app/models/ocr_engine.py

Robust OCR engine for the AI Resume Analyzer.

Strategy:
  1. Primary engine  : pytesseract (lightweight, always available)
  2. Fallback engine : easyocr     (better accuracy, opt-in via USE_EASYOCR=true)
  3. PDF OCR fallback: If PyMuPDF extracts < MIN_TEXT_LEN chars, render each page
                       as an image and OCR it.

Environment flags:
  USE_EASYOCR=true   — enable EasyOCR as a fallback / secondary engine
  OCR_DPI=200        — DPI used when rasterising PDF pages (default 200)
"""

import os
import re
import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger("resume_analyzer.ocr")

# ── Environment flags ──────────────────────────────────────────────────────────
USE_EASYOCR  = os.getenv("USE_EASYOCR", "false").lower() == "true"
OCR_DPI      = int(os.getenv("OCR_DPI", "200"))
MIN_TEXT_LEN = 50   # Minimum chars from direct PDF extraction before OCR fallback

# ── Lazy loaders ───────────────────────────────────────────────────────────────
_tesseract_available: Optional[bool] = None
_easyocr_reader = None


def _get_tesseract():
    """Return pytesseract module or None if not installed."""
    global _tesseract_available
    if _tesseract_available is None:
        try:
            import pytesseract  # noqa: F401
            _ = pytesseract.get_tesseract_version()
            _tesseract_available = True
            logger.info("[OCR] pytesseract is available.")
        except Exception as e:
            _tesseract_available = False
            logger.warning(f"[OCR] pytesseract not available: {e}")
    if _tesseract_available:
        import pytesseract
        return pytesseract
    return None


def _get_easyocr():
    """Return an EasyOCR reader (singleton) or None."""
    global _easyocr_reader
    if _easyocr_reader is None and USE_EASYOCR:
        try:
            import easyocr
            # gpu=False keeps it CPU-only and Render-compatible
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("[OCR] EasyOCR reader initialised.")
        except Exception as e:
            logger.warning(f"[OCR] EasyOCR not available: {e}")
    return _easyocr_reader


# ═══════════════════════════════════════════════════════════════════════════════
#  Image preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Preprocess a PIL image for best OCR accuracy:
      1. Convert to greyscale
      2. Auto-contrast (handles dark / low-contrast scans)
      3. Moderate sharpening
      4. Otsu-like binarisation via point()
    """
    # Grayscale
    img = img.convert("L")

    # Auto-contrast (stretch histogram)
    img = ImageOps.autocontrast(img, cutoff=1)

    # Sharpen slightly to improve edge clarity
    img = img.filter(ImageFilter.SHARPEN)

    # Binarise: pixels below 128 → 0, above → 255
    img = img.point(lambda x: 0 if x < 128 else 255, "L")

    # Upscale small images: Tesseract works best at ~300 DPI
    w, h = img.size
    if w < 1200:
        scale = 1200 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img


def _deskew(img: Image.Image) -> Image.Image:
    """
    Simple deskew using numpy projection profile.
    Only applied when the image's aspect ratio suggests it's a scanned page.
    Falls back gracefully if scipy is not available.
    """
    try:
        from scipy.ndimage import rotate

        arr = np.array(img)
        # Try angles from -5 to +5 degrees in 0.5 steps
        scores = []
        for angle in np.arange(-5, 5, 0.5):
            rotated = rotate(arr, angle, reshape=False, cval=255)
            # Measure variance of horizontal projections (peaks = aligned text)
            proj = rotated.sum(axis=1)
            scores.append((proj.var(), angle))

        best_angle = max(scores)[1]
        if abs(best_angle) > 0.5:
            arr = rotate(arr, best_angle, reshape=False, cval=255).astype(np.uint8)
            img = Image.fromarray(arr)
    except ImportError:
        pass  # scipy not installed — skip deskew
    except Exception as e:
        logger.debug(f"[OCR] Deskew skipped: {e}")
    return img


# ═══════════════════════════════════════════════════════════════════════════════
#  OCR engines
# ═══════════════════════════════════════════════════════════════════════════════

def ocr_image_pytesseract(img: Image.Image) -> str:
    """
    Run Tesseract OCR on a preprocessed PIL image.
    Config: PSM 6 (assume uniform block of text), OEM 3 (default LSTM + legacy).
    """
    tess = _get_tesseract()
    if tess is None:
        raise RuntimeError("pytesseract is not available.")

    custom_config = r"--oem 3 --psm 6"
    text = tess.image_to_string(img, config=custom_config, lang="eng")
    return text.strip()


def ocr_image_easyocr(img: Image.Image) -> str:
    """
    Run EasyOCR on a PIL image.
    Returns joined text from all detected boxes.
    """
    reader = _get_easyocr()
    if reader is None:
        raise RuntimeError("EasyOCR is not available.")

    arr = np.array(img)
    results = reader.readtext(arr, detail=0, paragraph=True)
    return " ".join(results).strip()


def _ocr_single_image(path: str) -> str:
    """
    OCR a single image file.
    Falls back: pytesseract → easyocr → raises.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        raise IOError(f"Cannot open image '{path}': {e}")

    img = _deskew(img)
    img = preprocess_image(img)

    errors = []

    # --- Attempt 1: pytesseract -----------------------------------------------
    try:
        text = ocr_image_pytesseract(img)
        if text and len(text.strip()) >= 20:
            return clean_ocr_output(text)
    except Exception as e:
        errors.append(f"pytesseract: {e}")

    # --- Attempt 2: easyocr (if enabled) ---------------------------------------
    if USE_EASYOCR:
        try:
            text = ocr_image_easyocr(img)
            if text:
                return clean_ocr_output(text)
        except Exception as e:
            errors.append(f"easyocr: {e}")

    if errors:
        logger.warning(f"[OCR] All engines failed for '{path}': {errors}")
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF OCR fallback
# ═══════════════════════════════════════════════════════════════════════════════

def ocr_pdf_pages(pdf_path: str) -> str:
    """
    Rasterise each PDF page via PyMuPDF and OCR it.
    Used when direct text extraction yields insufficient content (scanned PDFs).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF OCR fallback.")

    all_text = []
    zoom = OCR_DPI / 72.0  # 72 DPI is PyMuPDF's default render DPI
    mat  = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        try:
            pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img  = Image.frombytes("L", [pix.width, pix.height], pix.samples)
            img  = preprocess_image(img)
            text = ocr_image_pytesseract(img)
            if text:
                all_text.append(text)
                logger.debug(f"[OCR] Page {page_num + 1}: extracted {len(text)} chars")
        except Exception as e:
            logger.warning(f"[OCR] OCR failed on page {page_num + 1}: {e}")
    doc.close()

    return clean_ocr_output("\n".join(all_text))


# ═══════════════════════════════════════════════════════════════════════════════
#  Text post-processing
# ═══════════════════════════════════════════════════════════════════════════════

def clean_ocr_output(text: str) -> str:
    """
    Clean raw OCR output:
      - Fix common OCR encoding artifacts (e.g., ligatures fi → fi)
      - Remove lone single characters on their own lines (OCR noise)
      - Normalise whitespace
      - Remove excessive blank lines (> 2 consecutive)
    """
    if not text:
        return ""

    # Common OCR ligature fixes
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "\x0c": "\n",   # form-feed → newline
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Remove lines that are just single non-alphabetic characters (OCR noise)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) == 1 and not stripped.isalpha():
            continue
        cleaned_lines.append(line)

    # Collapse more than 2 consecutive blank lines
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalise spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_ocr(file_path: str) -> str:
    """
    Main entry point for OCR-based text extraction.

    Supports:
      - .jpg / .jpeg / .png / .bmp / .tiff  → direct image OCR
      - .pdf                                → PyMuPDF text first, then OCR fallback

    Args:
        file_path: Absolute path to the file.

    Returns:
        Cleaned extracted text string. Returns "" on failure.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        # Try PyMuPDF direct text first
        try:
            import fitz
            doc  = fitz.open(file_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            text = clean_ocr_output(text)
            if len(text.strip()) >= MIN_TEXT_LEN:
                logger.info(f"[OCR] PDF direct extraction: {len(text)} chars.")
                return text
            logger.info("[OCR] PDF text too short — falling back to page OCR.")
        except Exception as e:
            logger.warning(f"[OCR] PyMuPDF direct failed: {e}")

        # OCR fallback for scanned PDFs
        return ocr_pdf_pages(file_path)

    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        return _ocr_single_image(file_path)

    else:
        raise ValueError(f"Unsupported file type for OCR: '{ext}'")
