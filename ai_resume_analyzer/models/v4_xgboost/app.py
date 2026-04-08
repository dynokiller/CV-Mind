import os
import re
import pickle
import warnings
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, Form, HTTPException
from typing import Dict, Any

# ── 🔧 FIX 4: Handle sklearn Version Warning ────────────────────────────────
# Solution A: Ignore safely for deployment (Prevents startup errors)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = FastAPI(title="AI Resume Classifier API (Fixed)")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "xgb_model.json"
TFIDF_PATH = "tfidf_vectorizer.pkl"
LABEL_PATH = "label_encoder.pkl"

# ── Load Core ML Components ────────────────────────────────────────────────────
def load_models():
    if not (os.path.exists(TFIDF_PATH) and os.path.exists(LABEL_PATH) and os.path.exists(MODEL_PATH)):
        raise FileNotFoundError("Model artifacts missing in container root.")

    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(LABEL_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # Safety check for TF-IDF fitting
    if not hasattr(vectorizer, 'idf_'):
        print("❌ CRITICAL ERROR: Loaded TF-IDF vectorizer is NOT fitted!")
        raise Exception("TF-IDF vectorizer is not fitted. idf_ attribute missing.")
    
    return vectorizer, label_encoder, model

print("[Server] Initializing AI Resume Classifier...")
vectorizer, label_encoder, model = load_models()

# Final Verification
print(f"[Server] Vectorizer fitted: {hasattr(vectorizer, 'idf_')}")
class_names = list(label_encoder.classes_)
print(f"[Server] Models loaded successfully. Classes: {len(class_names)}")

# ── Preprocessing Logic ───────────────────────────────────────────────────────
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

# ── 🔧 5. Validate app.py (Endpoints) ──────────────────────────────────────────
@app.get("/")
def read_root():
    """Health check for Hugging Face Spaces."""
    return {"status": "online", "model": "V4.5 XGBoost", "engine": "FastAPI/Docker"}

@app.post("/predict")
async def predict(inputs: str = Form(...)):
    """
    Predict domain for a given resume text.
    Handles 'inputs' as form data for compatibility.
    """
    if not inputs:
        raise HTTPException(status_code=400, detail="inputs field is empty")
    
    try:
        # 1. Clean
        cleaned = clean_text(inputs)
        
        # 2. Vectorize
        vector = vectorizer.transform([cleaned])
        
        # 3. Predict
        probs = model.predict_proba(vector)[0]
        pred_idx = int(np.argmax(probs))
        
        domain = class_names[pred_idx]
        confidence = float(probs[pred_idx])
        
        all_probs = {
            name: round(float(p), 4)
            for name, p in zip(class_names, probs)
        }
        
        return {
            "predicted_domain": domain,
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
            "backend": "docker_fastapi"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces port is 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
