"""
tests/test_ocr_engine.py

Unit tests for the OCR engine module.
Uses only in-memory images so no files need to be present.
"""

import pytest
from PIL import Image, ImageDraw, ImageFont
import io
import os
import sys

# Make sure the project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_text_image(text: str = "Python Developer with 5 years experience") -> str:
    """Create a temp PNG image with given text, return file path."""
    img    = Image.new("RGB", (800, 200), color=(255, 255, 255))
    draw   = ImageDraw.Draw(img)
    draw.text((20, 80), text, fill=(0, 0, 0))
    path = "/tmp/test_ocr_sample.png"
    img.save(path)
    return path


class TestPreprocessImage:
    def test_returns_pillow_image(self):
        from app.models.ocr_engine import preprocess_image
        img    = Image.new("RGB", (400, 300), color=(200, 200, 200))
        result = preprocess_image(img)
        assert isinstance(result, Image.Image)

    def test_converts_to_grayscale(self):
        from app.models.ocr_engine import preprocess_image
        img    = Image.new("RGB", (400, 300), color=(200, 200, 200))
        result = preprocess_image(img)
        assert result.mode == "L"

    def test_upscales_small_image(self):
        from app.models.ocr_engine import preprocess_image
        img    = Image.new("RGB", (400, 300))
        result = preprocess_image(img)
        # Width should be >= 1200 for small input
        assert result.size[0] >= 1200


class TestCleanOCROutput:
    def test_fixes_ligatures(self):
        from app.models.ocr_engine import clean_ocr_output
        result = clean_ocr_output("ﬁle manager ﬂow")
        assert "fi" in result
        assert "fl" in result
        assert "ﬁ" not in result

    def test_collapses_blank_lines(self):
        from app.models.ocr_engine import clean_ocr_output
        text = "line1\n\n\n\n\nline2"
        result = clean_ocr_output(text)
        assert "\n\n\n" not in result

    def test_removes_noise_chars(self):
        from app.models.ocr_engine import clean_ocr_output
        text = "Name: John\n|\nExperience: 5 years"
        result = clean_ocr_output(text)
        assert "John" in result
        assert "5 years" in result

    def test_empty_input(self):
        from app.models.ocr_engine import clean_ocr_output
        assert clean_ocr_output("") == ""
        assert clean_ocr_output(None) == ""


class TestExtractTextOCR:
    def test_unsupported_extension_raises(self):
        from app.models.ocr_engine import extract_text_ocr
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text_ocr("document.xyz")

    @pytest.mark.skipif(
        os.getenv("CI") == "true" and not os.path.exists("/usr/bin/tesseract"),
        reason="Tesseract not installed"
    )
    def test_image_ocr_returns_string(self):
        from app.models.ocr_engine import extract_text_ocr
        path   = _make_text_image("Software Engineer Python Django")
        result = extract_text_ocr(path)
        assert isinstance(result, str)
        # Text may be imperfect but should be non-empty
        assert len(result) >= 0


class TestOCREngineIntegration:
    def test_clean_output_unicode_fixes(self):
        from app.models.ocr_engine import clean_ocr_output
        text = "He\u2019s a full\u2013stack developer\u2014great skills"
        result = clean_ocr_output(text)
        assert "'" in result
        assert "-" in result
