"""
training/advanced_text_cleaner.py

V4 ML Pipeline: NLP Preprocessing & Data Cleaning
- Spacy based Lemmatization & Stopword removal
- Duplicate / Noise reduction
- Extracts structured keywords (Skills, Tech) 
- Provides Synthetic Templating for Minority classes (AGRICULTURE, AUTOMOBILE, BPO)
"""

import re
import re

# Dependency-free cleaning (replaces spaCy lemmatization)

# Hard max chars for SpaCy processing — keeps inference fast
SPACY_CHAR_LIMIT = 3000

def clean_and_lemmatize(text: str) -> str:
    """
    Simplified cleaning using regex (No spaCy dependency).
    Focuses on tokenization, noise removal, and skill injection.
    (Kept name for backward compatibility with inference.py).
    """
    text = str(text).lower()
    
    # Remove HTML and CSS
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'(?:id|style|class)=["\'][^"\']*["\']', " ", text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # Keep alphanumeric, separators, and special technical chars
    text = re.sub(r'[^a-zA-Z0-9\s.,+#\-]', ' ', text)
    
    # Tokenize and filter short noise/digits
    tokens = text.split()
    cleaned_tokens = [
        t for t in tokens 
        if (len(t) > 1 or t in ['c', 'r', 'c++', 'c#', '.net']) 
        and not t.isdigit()
    ]
                
    # Prepend structured skill signal (domain-specific keywords only)
    structured_skills = extract_structured_skills(text)
    
    return structured_skills + " " + " ".join(cleaned_tokens)

def extract_structured_skills(text: str) -> str:
    """
    Injects explicitly found hard-skills at the START of the text so TF-IDF
    heavily weights them. 

    IMPORTANT: Only include PROFESSIONAL SKILLS here — NOT domain names like
    'agriculture' or 'aviation' that could confuse the classifier when they 
    appear in non-domain contexts (e.g., a CS student's research project 
    mentioning 'sustainable agriculture').
    """
    tech_keywords = {
        # Programming
        'python', 'java', 'javascript', 'typescript', 'sql', 'react', 'node',
        'aws', 'docker', 'kubernetes', 'pytorch', 'tensorflow', 'keras',
        # IT / Cyber Security — these are professional skills, not domain names
        'cyber', 'security', 'network', 'penetration testing', 'vulnerability',
        'cissp', 'siem', 'firewall', 'wireshark', 'metasploit', 'kali',
        'owasp', 'burp suite', 'nmap', 'cryptography', 'ethical hacking',
        'ccna', 'ansible', 'openid', 'oauth', 'jwt',
        # Data/AI
        'machine learning', 'deep learning', 'nlp', 'data science',
        'computer vision', 'llm', 'generative ai', 'transformers',
        # Web
        'html', 'css', 'rest', 'api',
        # Finance/Business
        'finance', 'accounting', 'sales', 'marketing',
        # Design
        'figma', 'photoshop', 'ui', 'ux',
        # HR
        'recruitment', 'hr', 'human resources',
        # Manufacturing
        'cad', 'autocad', 'catia',
        # BPO / Customer
        'bpo', 'crm', 'customer service',
    }
    
    found_skills = []
    text_lower = text.lower()
    for skill in sorted(tech_keywords):  # sorted for determinism
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill)
            
    if found_skills:
        return "SKILLS: " + " ".join(found_skills) + " | "
    return ""

def generate_synthetic_resumes(category: str, count: int = 5) -> list:
    """
    Generates realistic variations of resumes for critically underrepresented categories.
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
        base = base_templates[i % len(base_templates)]
        synthetics.append(f"Synthetic Template Variation {i}: " + base)
        
    return synthetics
