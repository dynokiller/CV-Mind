import numpy as np
import pickle
import xgboost as xgb
from sentence_transformers import SentenceTransformer

# Load Artifacts
print("Loading model and artifacts...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model = xgb.XGBClassifier()
model.load_model("xgb_model_bert.json")

CONFIDENCE_THRESHOLD = 0.40

def predict_resume(text):
    # Clean text like in training
    cleaned_text = ' '.join(str(text).split())
    # Embed
    embedding = embedder.encode([cleaned_text])
    # Predict
    probs = model.predict_proba(embedding)
    max_prob = np.max(probs, axis=1)[0]
    best_class = np.argmax(probs, axis=1)[0]
    
    if max_prob >= CONFIDENCE_THRESHOLD:
        domain = label_encoder.inverse_transform([best_class])[0]
    else:
        domain = "UNKNOWN/REVIEW"
        
    return domain, max_prob

sample_resumes = [
    "Experienced pediatric nurse with 5 years of clinical experience in a fast-paced hospital environment. Skilled in patient care, administering medication, and BLS certification.",
    "Senior Java backend developer. Expert in Spring Boot, microservices architecture, AWS, and Docker. 10 years of software engineering experience.",
    "Creative Graphic Designer with a strong portfolio in branding and UI/UX. Proficient in Adobe Creative Suite, Figma, and typography.",
    "Certified Public Accountant (CPA) with expertise in corporate tax, auditing, and financial forecasting. Managed portfolios for Fortune 500 companies.",
    "Construction Project Manager overseeing multi-million dollar commercial building projects. Skilled in blueprint reading, site safety, and contractor management.",
    "Short uninformative text that shouldn't match anything perfectly. I like to read books and watch movies.",
    "A fitness instructor specialized in personal training, yoga, and nutrition planning."
]

print("\n=== Resume Prediction Test ===")
for i, resume in enumerate(sample_resumes, 1):
    domain, conf = predict_resume(resume)
    print(f"\nSample {i}: {resume}")
    print(f"--> Predicted Category: {domain} (Confidence: {conf*100:.1f}%)")
