import os
import re
import pickle
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ── 1. Sample Dataset ─────────────────────────────────────────────────────────
data = [
    ("Python developer with experience in Django, Flask, and AWS. Skilled in Docker and Kubernetes.", "INFORMATION-TECHNOLOGY"),
    ("Java Software Engineer specialized in Spring Boot and Microservices. SQL and NoSQL expert.", "INFORMATION-TECHNOLOGY"),
    ("Cyber security analyst with CISSP and CEH certifications. SIEM and Network security specialist.", "INFORMATION-TECHNOLOGY"),
    ("Front-end React developer with UI/UX design skills. Expert in HTML, CSS, and Figma.", "INFORMATION-TECHNOLOGY"),
    
    ("Chartered Accountant with 10 years in taxation and auditing. Expert in Tally and ERP.", "FINANCE"),
    ("Financial Analyst with expertise in investment banking and portfolio management.", "FINANCE"),
    ("Banking professional skilled in credit analysis and risk management.", "FINANCE"),
    
    ("Human Resources Manager with experience in recruitment, onboarding, and payroll.", "HR"),
    ("Talent Acquisition specialist focusing on IT hiring and employee relations.", "HR"),
    ("HR Generalist with knowledge of labor laws and performance management.", "HR"),
    
    ("Mechanical Engineer with CAD and AutoCAD proficiency. Experience in automotive design.", "ENGINEERING"),
    ("Civil Engineer specialized in structural design and project management.", "ENGINEERING"),
    
    ("Sales representative with a track record in pharmaceutical sales and CRM utilization.", "SALES"),
    ("Marketing manager specialized in digital marketing and social media strategy.", "MARKETING")
]

texts, labels = zip(*data)

# ── 2. Preprocessing Logic (Must match app.py) ──────────────────────────────
def extract_structured_skills(text: str) -> str:
    tech_keywords = {
        'python', 'java', 'javascript', 'typescript', 'sql', 'react', 'node',
        'aws', 'docker', 'kubernetes', 'pytorch', 'tensorflow', 'keras',
        'cyber', 'security', 'network', 'penetration testing', 'vulnerability',
        'cissp', 'siem', 'firewall', 'wireshark', 'metasploit', 'kali',
        'owasp', 'burp suite', 'nmap', 'cryptography', 'ethical hacking',
        'ccna', 'ansible', 'openid', 'oauth', 'jwt',
        'machine learning', 'deep learning', 'nlp', 'data science',
        'computer vision', 'llm', 'generative ai', 'transformers',
        'html', 'css', 'rest', 'api', 'finance', 'accounting', 'sales', 'marketing',
        'figma', 'photoshop', 'ui', 'ux', 'recruitment', 'hr', 'human resources',
        'cad', 'autocad', 'catia', 'bpo', 'crm', 'customer service',
    }
    found_skills = []
    text_lower = text.lower()
    for skill in sorted(tech_keywords):
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill)
    if found_skills:
        return "SKILLS: " + " ".join(found_skills) + " | "
    return ""

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'(?:id|style|class)=["\'][^"\']*["\']', " ", text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,+#\-]', ' ', text)
    
    tokens = text.split()
    cleaned_tokens = [
        t for t in tokens 
        if (len(t) > 1 or t in ['c', 'r', 'c++', 'c#', '.net']) 
        and not t.isdigit()
    ]
    
    structured_skills = extract_structured_skills(text)
    return structured_skills + " " + " ".join(cleaned_tokens)

# ── 3. Training Pipeline ──────────────────────────────────────────────────────
print("[Build] Cleaning texts...")
cleaned_texts = [clean_text(t) for t in texts]

print("[Build] Fitting TF-IDF Vectorizer...")
# CRITICAL FIX: Use fit_transform to ensure idf_ is populated
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(cleaned_texts)

print("[Build] Fitting Label Encoder...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

print("[Build] Training XGBoost Classifier...")
model = xgb.XGBClassifier(objective='multi:softprob', num_class=len(label_encoder.classes_))
model.fit(X, y)

# ── 4. Saving Artifacts ───────────────────────────────────────────────────────
print("[Build] Saving artifacts...")
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

model.save_model("xgb_model.json")

# ── 5. Validation ─────────────────────────────────────────────────────────────
print("\n[Validation] Checking artifacts...")

# Verify Vectorizer
if hasattr(vectorizer, 'idf_'):
    print("✅ TF-IDF Vectorizer is FITTED (idf_ exists).")
else:
    print("❌ TF-IDF Vectorizer is NOT FITTED.")

# Verify End-to-End Prediction
test_text = "Full-stack developer with React and Node.js expertise."
test_cleaned = clean_text(test_text)
test_vector = vectorizer.transform([test_cleaned])
test_pred_idx = model.predict(test_vector)[0]
test_domain = label_encoder.inverse_transform([test_pred_idx])[0]

print(f"✅ Sample Prediction: '{test_text}' -> {test_domain}")
print("\n[FINISH] Model artifacts regenerated successfully.")
