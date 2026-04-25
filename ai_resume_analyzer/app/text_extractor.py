import fitz  # PyMuPDF
import docx
import io

def extract_text(file_path, filename):
    text = ""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    try:
        if ext == 'pdf':
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text("text") + "\n"
        elif ext in ['doc', 'docx']:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            # Fallback for plain text or unsupported types
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        
    return text.strip()

import re

def extract_candidate_name(text):
    """
    Very basic heuristic to extract a candidate's name from resume text.
    It looks at the first 10 non-empty lines and finds the first one that 
    looks like a name (2-3 words, mostly alphabetic, Title Case or UPPERCASE).
    """
    if not text:
        return None
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Check the first 10 lines
    for line in lines[:10]:
        # Remove common resume headers if they appear
        if line.lower() in ['resume', 'curriculum vitae', 'cv']:
            continue
            
        # A name usually has 1 to 4 words
        words = line.split()
        if 1 <= len(words) <= 4:
            # Check if all words consist of letters (or dot for initials)
            is_name = True
            for word in words:
                clean_word = re.sub(r'[^A-Za-z]', '', word)
                if not clean_word:
                    is_name = False
                    break
                    
            if is_name:
                # Capitalize it nicely
                return line.title()
                
    return None
