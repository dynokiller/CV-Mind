import pdfplumber
import pytesseract
import io

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from PDF, falls back to OCR if no text found."""
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                # Fallback OCR
                try:
                    img = page.to_image(resolution=300).original
                    text += pytesseract.image_to_string(img) + "\n"
                except Exception as e:
                    print(f"OCR failed for a page: {e}")
    return text
