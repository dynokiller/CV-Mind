"""
training/train_advanced.py

V4.5 ML Orchestrator (Massive Dataset Upgrade):
1. Loads original Kaggle dataset.
2. Downloads massive 15,000+ Hugging Face Job Description dataset.
3. Automatically maps 850 granular job titles back to the 24 macro target domains.
4. Cleans & Appends structured synthetics.
5. Builds TF-IDF sparse matrix (max_features=10000 to handle massive vocab).
6. Applies SMOTE to balance everything.
7. Trains hyper-optimized XGBoost model on the giant corpus.
"""

import os
import pickle
import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score
from datasets import load_dataset

from training.advanced_text_cleaner import clean_and_lemmatize, generate_synthetic_resumes

DATA_PATH = "data/resume_dataset.csv"
MODEL_DIR = "models/v4_xgboost"
ENCODER_PATH = f"{MODEL_DIR}/label_encoder.pkl"
TFIDF_PATH = f"{MODEL_DIR}/tfidf_vectorizer.pkl"
XGB_PATH = f"{MODEL_DIR}/xgb_model.json"

SMOTE_RANDOM_STATE = 42
TEST_SIZE = 0.2

def map_hf_title_to_macro(title: str):
    t = str(title).lower()
    
    # 1. IT & Cyber
    if any(x in t for x in ['cyber', 'security', 'software', 'developer', 'data', 'cloud', 'network', 'it ', 'web', 'programmer', 'systems analyst', 'information technology']):
        return 'INFORMATION-TECHNOLOGY'
        
    # 2. Engineering
    if any(x in t for x in ['engineer', 'mechanic', 'electrical', 'civil', 'hardware']):
        return 'ENGINEERING'
        
    # 3. Finance & Accountant & Banking
    if any(x in t for x in ['account', 'audit', 'tax']): return 'ACCOUNTANT'
    if any(x in t for x in ['bank', 'teller', 'loan']): return 'BANKING'
    if any(x in t for x in ['finance', 'financial', 'invest', 'wealth']): return 'FINANCE'
    
    # 4. HR
    if any(x in t for x in ['hr', 'human resources', 'recruit', 'talent']): return 'HR'
    
    # 5. Teacher
    if any(x in t for x in ['teach', 'tutor', 'professor', 'instructor', 'faculty']): return 'TEACHER'
    
    # 6. Healthcare
    if any(x in t for x in ['nurse', 'doctor', 'medical', 'health', 'clinic', 'therapist', 'physician', 'pharm', 'caregiver', 'surgery']): return 'HEALTHCARE'
    
    # 7. Sales & Business Dev
    if any(x in t for x in ['sale', 'account executive', 'retail']): return 'SALES'
    if any(x in t for x in ['business development', 'strategy']): return 'BUSINESS-DEVELOPMENT'
    
    # 8. Designer & Arts
    if any(x in t for x in ['design', 'ui', 'ux', 'graphic']): return 'DESIGNER'
    if any(x in t for x in ['art', 'music', 'animat', 'creative']): return 'ARTS'
    
    # 9. Aviation
    if any(x in t for x in ['aviat', 'pilot', 'flight', 'aircraft']): return 'AVIATION'
    
    # 10. Digital Media / PR
    if any(x in t for x in ['media', 'social', 'video', 'content', 'journal', 'writer']): return 'DIGITAL-MEDIA'
    if any(x in t for x in ['public relations', 'pr ', 'communication']): return 'PUBLIC-RELATIONS'
    
    # 11. Consultant
    if any(x in t for x in ['consult']): return 'CONSULTANT'
    
    # 12. Construction
    if any(x in t for x in ['construct', 'build', 'site', 'architect', 'contractor']): return 'CONSTRUCTION'
    
    # 13. Automobile
    if any(x in t for x in ['auto', 'car', 'vehicle', 'motor']): return 'AUTOMOBILE'
    
    # 14. Agriculture
    if any(x in t for x in ['farm', 'agricultur', 'crop']): return 'AGRICULTURE'
    
    # 15. Apparel
    if any(x in t for x in ['apparel', 'fashion', 'clothing', 'garment', 'tailor']): return 'APPAREL'
    
    # 16. Fitness
    if any(x in t for x in ['fit', 'train', 'gym', 'coach', 'sport']): return 'FITNESS'
    
    # 17. Advocate / Legal
    if any(x in t for x in ['law', 'advocate', 'legal', 'attorney', 'counsel', 'paralegal']): return 'ADVOCATE'
    
    # 18. Chef
    if any(x in t for x in ['chef', 'cook', 'culinary', 'kitchen']): return 'CHEF'
    
    # 19. BPO
    if any(x in t for x in ['bpo', 'call center', 'customer service']): return 'BPO'

    return None

def load_and_preprocess_data():
    print(f"1. Loading Base Raw Data from {DATA_PATH}...")
    df_base = pd.read_csv(DATA_PATH)
    
    col_map = {}
    for col in df_base.columns:
        if col.lower() in ("resume_str", "resume", "resume_text"):
            col_map[col] = "Resume_str"
        elif col.lower() in ("category", "label", "domain"):
            col_map[col] = "Category"
    df_base.rename(columns=col_map, inplace=True)
    df_base.dropna(subset=["Resume_str", "Category"], inplace=True)

    print("2. Downloading Massive Hugging Face Job Description Dataset...")
    try:
        ds = load_dataset('jacob-hugging-face/job-descriptions', split='train')
    except Exception as e:
        print(f"Failed to load Hugging Face dataset. Falling back to Kaggle only... {e}")
        ds = []

    hf_texts = []
    hf_labels = []
    
    mapped_count = 0
    for row in ds:
        title = row.get("position_title", "")
        desc = row.get("model_response", "")
        if not title or not desc:
            continue
            
        macro_domain = map_hf_title_to_macro(title)
        if macro_domain:
            hf_texts.append(desc)
            hf_labels.append(macro_domain)
            mapped_count += 1
            
    if hf_texts:
        df_hf = pd.DataFrame({"Resume_str": hf_texts, "Category": hf_labels})
        df = pd.concat([df_base, df_hf], ignore_index=True)
        print(f" -> Successfully fused {mapped_count} Hugging Face job descriptions into the core framework!")
    else:
        df = df_base
        
    print(f"Total Combined Corpus Size: {len(df)}")
    
    print("3. Spacy NLP Preprocessing (Lemmatization, Stop-Words) [Warning: Takes time for large corpus]...")
    df["Cleaned"] = df["Resume_str"].apply(clean_and_lemmatize)
    
    counts = df["Category"].value_counts()
    synthetic_texts = []
    synthetic_labels = []

    for cat, count in counts.items():
        if count < 10:
            print(f" -> Injecting synthetics for critical minority: {cat} ({count})")
            for var in generate_synthetic_resumes(cat, count=10):
                synthetic_texts.append(clean_and_lemmatize(var))
                synthetic_labels.append(cat)
                
    if synthetic_texts:
        synth_df = pd.DataFrame({"Resume_str": synthetic_texts, "Category": synthetic_labels, "Cleaned": synthetic_texts})
        df = pd.concat([df, synth_df], ignore_index=True)
        
    return df

def build_tfidf_and_smote(df):
    print("4. Generating Expanded TF-IDF Sparse Matrices...")
    # Increase max_features to 10000 to harness the larger 15k corpus vocab
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
    
    X_texts = df["Cleaned"].tolist()
    y_raw = df["Category"].tolist()
    
    t0 = time.time()
    X_tfidf = vectorizer.fit_transform(X_texts)
    print(f" -> TF-IDF took {time.time() - t0:.2f} seconds. Shape: {X_tfidf.shape}")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y_enc, test_size=TEST_SIZE, random_state=SMOTE_RANDOM_STATE, stratify=y_enc
    )
    
    print("5. Applying SMOTE to Balance Massive Matrix Mathematically...")
    # k_neighbors=2 to accommodate absolute minority thresholds
    smote = SMOTE(k_neighbors=2, random_state=SMOTE_RANDOM_STATE) 
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f" -> Orig Train size: {X_train.shape[0]}. SMOTE Balanced size: {X_train_res.shape[0]}")
    return X_train_res, X_test, y_train_res, y_test, le

def train_xgboost(X_train, y_train):
    print("6. Training XGBoost Classifier on Massive Corpus...")
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=300, # Increased for larger data
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    print("7. Saving V4.5 Hyper-XGBoost Model...")
    clf.save_model(XGB_PATH)
    return clf

def evaluate_pipeline(clf, X_test, y_test, le):
    print("8. Evaluating V4.5 Architecture...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_wght = f1_score(y_test, y_pred, average="weighted")
    
    print("=" * 60)
    print(f" V4.5 PIPELINE RESULTS ")
    print("=" * 60)
    print(f" Accuracy : {acc:.4f}")
    print(f" F1 Macro : {f1_mac:.4f}")
    print(f" F1 Weight: {f1_wght:.4f}")
    print("=" * 60)
    print(" Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test, le = build_tfidf_and_smote(df)
    model = train_xgboost(X_train, y_train)
    evaluate_pipeline(model, X_test, y_test, le)
