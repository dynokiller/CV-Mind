import re
import spacy
import nltk
from nltk.corpus import stopwords

# Download stopwords securely once
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load spacy 
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    nlp = None

stop_words = set(stopwords.words('english'))

def advanced_clean(text: str) -> str:
    """Cleans text, lemmatizes, removes stopwords and noise."""
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs, emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove isolated numbers and dates (keeping contextual numbers if alpha-numeric like "b2b")
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Remove punctuation & special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenize and lemmatize using spaCy
    if nlp:
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if str(token.lemma_).strip() not in stop_words and len(str(token.lemma_).strip()) > 1]
    else:
        # Fallback if spacy is not loaded
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    text = " ".join(tokens)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
