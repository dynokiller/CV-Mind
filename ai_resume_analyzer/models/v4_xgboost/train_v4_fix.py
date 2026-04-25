import os
import re
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ── 1. Dataset ─────────────────────────────────────────────────────────
print("[Build] Loading dataset...")
csv_path = r"C:\Users\Admin\Downloads\archive\Resume\Resume.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=['Resume_str', 'Category'])
texts = df['Resume_str'].tolist()
labels = df['Category'].tolist()

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
        'doctor', 'medical', 'healthcare', 'surgery', 'surgeon', 'nurse', 'nursing', 'clinical',
        'cardiology', 'pediatrics', 'hospital', 'patient', 'md', 'mbbs', 'bls', 'acls', 'pharmacy',
        'anatomy', 'physiology', 'pathology', 'medicine',
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
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    n_estimators=100,
    tree_method='hist',
    device='cuda'
)
model.fit(X.toarray(), y)


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
    print("[SUCCESS] TF-IDF Vectorizer is FITTED (idf_ exists).")
else:
    print("[ERROR] TF-IDF Vectorizer is NOT FITTED.")

# Verify End-to-End Prediction
test_text = "Experienced medical doctor and surgeon specializing in cardiology."
test_cleaned = clean_text(test_text)
test_vector = vectorizer.transform([test_cleaned])
test_pred_idx = model.predict(test_vector.toarray())[0]
test_domain = label_encoder.inverse_transform([test_pred_idx])[0]

print(f"[SUCCESS] Sample Prediction: '{test_text}' -> {test_domain}")
print("\n[FINISH] Model artifacts regenerated successfully.")
