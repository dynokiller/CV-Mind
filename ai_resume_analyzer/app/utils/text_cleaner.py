import re

def clean_text(text: str) -> str:
    """
    Cleans raw resume text:
    - Removes HTML tags
    - Removes non-ascii characters
    - Removes extra whitespace
    - Removes special characters
    - Converts newlines to spaces
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove non-ascii
    text = text.encode("ascii", "ignore").decode()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,@:-]', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()
