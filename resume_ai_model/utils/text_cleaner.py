import re

def clean_text(text: str) -> str:
    """Cleans text for transformer input."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    text = re.sub(r'http\S+', '', text) # remove URLs
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text) # remove punctuation
    return text.strip()
