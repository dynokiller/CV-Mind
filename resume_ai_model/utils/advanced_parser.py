import fitz  # PyMuPDF
import docx
import io

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text robustly from PDF using PyMuPDF (fitz)."""
    text = ""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print(f"PyMuPDF failed: {e}. Falling back to pdfplumber...")
        import pdfplumber
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e2:
            print(f"pdfplumber failed: {e2}")
    
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extracts text from DOCX file."""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = [para.text for para in doc.paragraphs]
        return "\n".join(text)
    except Exception as e:
        print(f"Error parsing DOCX: {e}")
        return ""
