import spacy
import re
from typing import Dict, List, Any

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Predefined Skill Dictionary (can be expanded)
SKILL_DB = [
    "Python", "Java", "C++", "JavaScript", "React", "Angular", "Vue", "Node.js",
    "FastAPI", "Flask", "Django", "SQL", "NoSQL", "Machine Learning", "Deep Learning",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy", "AWS", "Azure",
    "GCP", "Docker", "Kubernetes", "Git", "CI/CD", "NLP", "Computer Vision",
    "Data Analysis", "Tableau", "Power BI", "Excel", "Communication", "Leadership",
    "Problem Solving", "Agile", "Scrum", "HTML", "CSS", "TypeScript"
]

def extract_name(doc) -> str:
    """Extracts the first Person entity found."""
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Unknown"

def extract_contact_info(text: str) -> Dict[str, str]:
    """Extracts email and phone number using Regex."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    
    email = re.search(email_pattern, text)
    phone = re.search(phone_pattern, text)
    
    return {
        "email": email.group(0) if email else "Unknown",
        "phone": phone.group(0) if phone else "Unknown"
    }

def extract_skills(text: str) -> List[str]:
    """Extracts skills based on the predefined SKILL_DB."""
    found_skills = []
    text_lower = text.lower()
    for skill in SKILL_DB:
        # Match whole words only to avoid partial matches (e.g., 'C' in 'Clean')
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found_skills.append(skill)
    return list(set(found_skills))

def extract_education(text: str) -> List[str]:
    """Extracts education keywords (Simple matching)."""
    education_keywords = ["B.Sc", "M.Sc", "B.Tech", "M.Tech", "PhD", "Bachelor", "Master", "Diploma", "Associate", "University", "College"]
    found_education = []
    text_lower = text.lower()
    for edu in education_keywords:
        if edu.lower() in text_lower:
            found_education.append(edu)
    return list(set(found_education))

def extract_experience(text: str) -> str:
    """Estimates years of experience (Naive approach)."""
    # Look for patterns like "5+ years", "3 years of experience"
    exp_pattern = r'(\d+)\+?\s*years?'
    matches = re.findall(exp_pattern, text.lower())
    if matches:
        return max(matches, key=int)  # Return the highest number found
    return "0"

def extract_features(text: str) -> Dict[str, Any]:
    """Main function to extract all features from resume text."""
    doc = nlp(text)
    
    contact = extract_contact_info(text)
    
    return {
        "name": extract_name(doc),
        "email": contact["email"],
        "phone": contact["phone"],
        "skills": extract_skills(text),
        "education": extract_education(text),
        "experience": extract_experience(text)
    }
