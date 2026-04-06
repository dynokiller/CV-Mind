import io
from typing import Optional

import pytesseract
from PIL import Image, ImageFilter, ImageOps

try:
    import easyocr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    easyocr = None  # type: ignore

try:
    import layoutparser as lp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    lp = None  # type: ignore


def _load_image(file_bytes: bytes) -> Image.Image:
    """Load image from raw bytes and convert to RGB."""
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _preprocess_image(image: Image.Image) -> Image.Image:
    """
    Apply light-weight preprocessing to improve OCR accuracy while keeping it fast.

    Steps:
    - Convert to grayscale
    - Auto contrast
    - Mild sharpening
    - Binarization
    """
    # Grayscale
    gray = ImageOps.grayscale(image)

    # Auto contrast to improve text visibility
    gray = ImageOps.autocontrast(gray)

    # Mild sharpen
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

    # Simple binarization
    # Using point operation is fast and keeps us under the 5s budget
    threshold = 180
    bw = gray.point(lambda x: 255 if x > threshold else 0, mode="1")

    return bw


def _ocr_tesseract(image: Image.Image) -> str:
    """Run Tesseract OCR with sane defaults for resumes."""
    # Assume English resumes by default; caller can override later if needed
    custom_oem_psm_config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(image, lang="eng", config=custom_oem_psm_config)
    return text


def _ocr_easyocr(image: Image.Image) -> Optional[str]:
    """Optional EasyOCR-based recognition for challenging images."""
    if easyocr is None:
        return None

    reader = easyocr.Reader(["en"], gpu=False)
    # EasyOCR expects numpy array
    import numpy as np  # Local import to avoid hard dependency at import time

    np_img = np.array(image)
    results = reader.readtext(np_img, detail=0, paragraph=True)
    return "\n".join(results)


def _layout_aware_ocr(image: Image.Image) -> Optional[str]:
    """
    Optional LayoutParser-based OCR.

    This lets us preserve reading order on complex multi-column resumes when
    layoutparser and a compatible detection model are available.
    """
    if lp is None:
        return None

    try:
        import numpy as np  # Local import to avoid mandatory dependency

        np_img = np.array(image)

        # Use a fast, general text detection model if available
        model = lp.Detectron2LayoutModel(
            config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            label_map={0: "Text"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        )

        layout = model.detect(np_img)
        # Sort blocks top-to-bottom, then left-to-right to approximate reading order
        layout = lp.Layout(sorted(layout, key=lambda b: (b.block.y_1, b.block.x_1)))

        full_text = []
        for block in layout:
            x_1, y_1, x_2, y_2 = map(int, block.block.coordinates)
            crop = image.crop((x_1, y_1, x_2, y_2))
            block_text = _ocr_tesseract(crop)
            if block_text.strip():
                full_text.append(block_text.strip())

        return "\n".join(full_text)
    except Exception:
        # If anything goes wrong, fall back to standard OCR
        return None


def extract_text_from_image(file_bytes: bytes) -> str:
    """
    High-level API for image OCR used by the resume extraction pipeline.

    Strategy (within ~5s budget on typical resumes):
    - Load and preprocess image
    - Try layout-aware OCR if available (best accuracy on complex layouts)
    - Fallback to Tesseract on preprocessed image
    - Optionally blend with EasyOCR output if available
    """
    image = _load_image(file_bytes)
    preprocessed = _preprocess_image(image)

    # 1) Try layout-aware OCR first if available
    layout_text = _layout_aware_ocr(image)
    if layout_text and layout_text.strip():
        base_text = layout_text
    else:
        # 2) Fallback to standard Tesseract on preprocessed image
        try:
            base_text = _ocr_tesseract(preprocessed)
        except Exception as e:
            print(f"Tesseract failed: {e}")
            base_text = ""

    # 3) Optionally augment/validate with EasyOCR
    try:
        easy_text = _ocr_easyocr(preprocessed)
    except Exception as e:
        print(f"EasyOCR failed: {e}")
        easy_text = None

    if easy_text and len(easy_text) > len(base_text) * 0.5:
        # If EasyOCR recovers significantly more content, prefer it
        # This heuristic keeps us robust without double-counting duplicates
        base_text = easy_text

    return base_text.strip()

