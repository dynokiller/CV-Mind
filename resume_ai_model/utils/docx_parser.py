import docx
import io

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extracts text from DOCX file."""
    doc = docx.Document(io.BytesIO(file_bytes))
    text = [para.text for para in doc.paragraphs]
    return "\n".join(text)
