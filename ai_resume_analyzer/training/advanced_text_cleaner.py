"""
training/advanced_text_cleaner.py

V4 ML Pipeline: NLP Preprocessing & Data Cleaning
- Spacy based Lemmatization & Stopword removal
- Duplicate / Noise reduction
- Extracts structured keywords (Skills, Tech) 
- Provides Synthetic Templating for Minority classes (AGRICULTURE, AUTOMOBILE, BPO)
"""

import re
import pandas as pd
import spacy

# Ensure Spacy runs; will prompt download if missing in runtime
try:
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

def clean_and_lemmatize(text: str) -> str:
    """
    Advanced NLP Text Cleaner:
    1. Removes HTML, Links, Special Chars.
    2. Uses Spacy for Lemmatization ('managed' -> 'manage').
    3. Removes Stop Words & Punctuation.
    """
    text = str(text).lower()
    
    # Remove HTML and CSS
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'(?:id|style|class)=["\'][^"\']*["\']', " ", text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # Remove extremely long repetitive blocks or non-ascii
    text = re.sub(r'[^a-zA-Z0-9\s.,+#\-]', ' ', text)
    
    # SpaCy NLP Pipeline (Max char limit guard to prevent OOM)
    # 6000 chars is ~1000 words. SentenceTransformers truncates at ~512 tokens anyway.
    if len(text) > 6000:
         text = text[:6000] 
         
    doc = nlp(text)
    
    cleaned_tokens = []
    for token in doc:
        # Keep alphabetic or specific tech symbols (c#, c++)
        if (token.is_alpha or token.text in ['c++', 'c#', '.net']) and not token.is_stop and not token.is_punct:
            if len(token.lemma_) > 1: # remove single stray letters
                cleaned_tokens.append(token.lemma_)
                
    return " ".join(cleaned_tokens)

def extract_structured_skills(text: str) -> str:
    """
    Extracts explicit hard-skills or tech to inject heavily back into the top of the text.
    Ensures embeddings heavily weigh these terms.
    """
    # Simplified list for demo; in production this is driven by a massive dictionary
    tech_keywords = {
        'python', 'java', 'sql', 'react', 'node', 'aws', 'docker', 'kubernetes', 'ml', 
        'machine learning', 'data analysis', 'finance', 'accounting', 'sales', 
        'marketing', 'seo', 'design', 'photoshop', 'figma', 'agriculture', 'bpo', 
        'manufacturing', 'cad', 'autocad', 'hr', 'recruitment'
    }
    
    found_skills = []
    text_lower = text.lower()
    for skill in tech_keywords:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill)
            
    # Return formatted block like "SKILLS: python java docker"
    if found_skills:
        return "EXTRACTED_SKILLS_BLOCK: " + " ".join(found_skills) + " | "
    return ""

def generate_synthetic_resumes(category: str, count: int = 5) -> list:
    """
    Generates realistic variations of resumes for critically underrepresented categories
    to give SMOTE a wider variance baseline before mathematical interpolation.
    """
    templates = {
        "AGRICULTURE": [
            "Experienced agriculture specialist with expertise in crop management, soil science, irrigation systems, and sustainable farming. Track record of improving yield by 20%.",
            "Agricultural Engineer skilled in farm machinery, precision agriculture tech, hydroponics, and agribusiness logistics.",
            "Farm Manager handling daily agribusiness operations, livestock management, agricultural supply chain, and pest control methodologies."
        ],
        "AUTOMOBILE": [
            "Automotive Engineer handling powertrain design, CAD modeling, vehicle dynamics, and OEM manufacturing protocols.",
            "Automobile Technician experienced in engine diagnostics, mechanical repairs, HVAC systems, and fleet maintenance.",
            "Senior Automotive Designer utilizing AutoCAD and CATIA for chassis design, aerodynamics, and structural crash testing compliance."
        ],
        "BPO": [
            "Customer Service Representative with 5 years explicitly in BPO operations, handling inbound calls, resolving tickets in Salesforce, and maintaining 95% CSAT.",
            "BPO Team Lead managing call center metrics, SLA compliance, workforce management, and agent QA coaching.",
            "Technical Support Executive within a major BPO, solving tier-1 IT issues remotely, high volume call handling, CRM utilization."
        ]
    }
    
    if category not in templates:
        return []
        
    base_templates = templates[category]
    synthetics = []
    
    for i in range(count):
        # Slightly alter the template round-robin to simulate unique resumes
        base = base_templates[i % len(base_templates)]
        synthetics.append(f"Synthetic Template Variation {i}: " + base)
        
    return synthetics
