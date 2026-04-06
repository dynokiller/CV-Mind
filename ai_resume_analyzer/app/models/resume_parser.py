import fitz  # PyMuPDF
import docx
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from app.utils.text_cleaner import clean_text

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return clean_text(text)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    try:
        doc = docx.Document(docx_path)
        text = " ".join([para.text for para in doc.paragraphs])
        return clean_text(text)
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
        return ""

def extract_text_from_image(image_path: str) -> str:
    try:
        image = cv2.imread(image_path)
        if image is None:
            return ""
        
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        text = pytesseract.image_to_string(thresh)
        return clean_text(text)
    except Exception as e:
        print(f"Error reading IMAGE {image_path}: {e}")
        return ""

def parse_resume(file_path: str) -> str:
    ext = file_path.lower().split(".")[-1]
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['doc', 'docx']:
        return extract_text_from_docx(file_path)
    elif ext in ['png', 'jpg', 'jpeg']:
        return extract_text_from_image(file_path)
    else:
        return ""
