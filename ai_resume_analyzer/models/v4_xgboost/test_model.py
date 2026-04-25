import pickle
import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import re

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

print("[Test] Loading artifacts...")
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

print("[Test] Loading dataset for evaluation...")
csv_path = r"C:\Users\Admin\Downloads\archive\Resume\Resume.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=['Resume_str', 'Category'])

df_sample = df.sample(n=min(500, len(df)), random_state=42)

texts = df_sample['Resume_str'].tolist()
y_true_labels = df_sample['Category'].tolist()

print("[Test] Cleaning texts...")
cleaned_texts = [clean_text(t) for t in texts]

print("[Test] Vectorizing...")
X = vectorizer.transform(cleaned_texts)

print("[Test] Predicting...")
y_pred_idx = model.predict(X.toarray())
y_pred_labels = label_encoder.inverse_transform(y_pred_idx)

print("\n=== Model Performance on 500 Random Samples ===")
print(f"Overall Accuracy: {accuracy_score(y_true_labels, y_pred_labels) * 100:.2f}%")

print("\n=== Custom Sample Predictions ===")
examples = [
    "Experienced medical doctor and surgeon specializing in cardiology. Extensive clinical and hospital management experience.",
    "Python developer with 5 years experience in Django, React, and AWS. Strong background in Kubernetes.",
    "Chartered Accountant skilled in tax audit, finance, and portfolio management.",
    "HR Manager with 10 years experience in recruitment, payroll, and employee relations.",
    "Civil engineer with experience in structural design and AutoCAD."
]

for ex in examples:
    cl = clean_text(ex)
    vec = vectorizer.transform([cl])
    pred = model.predict(vec.toarray())[0]
    domain = label_encoder.inverse_transform([pred])[0]
    print(f"Input: {ex}\nPredicted Category: {domain}\n")
