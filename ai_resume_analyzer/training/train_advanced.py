"""
training/train_advanced.py

V4 ML Orchestrator:
1. Loads dataset & Cleans using Spacy
2. Identifies critically sparse classes & injects synthetic texts
3. Embeds everything into 384D Space (SentenceTransformer)
4. Applies SMOTE to mathematically balance 24 clusters
5. Trains an XGBoost classifier on the embeddings
6. Saves models for the API endpoint
"""

import os
import pickle
import pandas as pd
import numpy as np
import time

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score

from training.advanced_text_cleaner import clean_and_lemmatize, generate_synthetic_resumes

DATA_PATH = "data/resume_dataset.csv"
MODEL_DIR = "models/v4_xgboost"
ENCODER_PATH = f"{MODEL_DIR}/label_encoder.pkl"
XGB_PATH = f"{MODEL_DIR}/xgb_model.json"

# Fixed hyperparameters for stability on 500 records
SMOTE_RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_and_preprocess_data():
    print(f"1. Loading Raw Data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Normalize col names
    col_map = {}
    for col in df.columns:
        if col.lower() in ("resume_str", "resume", "resume_text"):
            col_map[col] = "Resume_str"
        elif col.lower() in ("category", "label", "domain"):
            col_map[col] = "Category"
    df.rename(columns=col_map, inplace=True)
    df.dropna(subset=["Resume_str", "Category"], inplace=True)
    
    print("2. Spacy NLP Preprocessing (Lemmatization, Stop-Words)...")
    # Clean text
    df["Cleaned"] = df["Resume_str"].apply(clean_and_lemmatize)
    
    # Count occurrences to flag critical minorities
    counts = df["Category"].value_counts()
    synthetic_texts = []
    synthetic_labels = []
    
    # If a class has fewer than 10 samples, inject synthetic variants before SMOTE
    # to create enough base nodes for SMOTE to interpolate properly without collapsing.
    for cat, count in counts.items():
        if count < 10:
            print(f" -> Injecting synthetic templates for critical minority: {cat} (count: {count})")
            variants = generate_synthetic_resumes(cat, count=5)
            for var in variants:
                synthetic_texts.append(clean_and_lemmatize(var))
                synthetic_labels.append(cat)
                
    if synthetic_texts:
        synth_df = pd.DataFrame({"Resume_str": synthetic_texts, "Category": synthetic_labels, "Cleaned": synthetic_texts})
        df = pd.concat([df, synth_df], ignore_index=True)
        print(f" -> Added {len(synthetic_texts)} synthetic templates to baseline.")
        
    return df

def build_embeddings_and_smote(df):
    print("3. Generating Dense Semantic Vectors (SentenceTransformers)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # We embed the whole set before splitting to apply SMOTE correctly on train-only later
    X_texts = df["Cleaned"].tolist()
    y_raw = df["Category"].tolist()
    
    t0 = time.time()
    X_embed = embedder.encode(X_texts, show_progress_bar=True, device="cpu") # enforce CPU for standard envs
    print(f" -> Encoding took {time.time() - t0:.2f} seconds. Shape: {X_embed.shape}")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    
    # Save encoder immediately
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
        
    # SPLIT BEFORE SMOTE to prevent data leakage (never oversample validation data!)
    X_train, X_test, y_train, y_test = train_test_split(
        X_embed, y_enc, test_size=TEST_SIZE, random_state=SMOTE_RANDOM_STATE, stratify=y_enc
    )
    
    print("4. Applying SMOTE to Balance 24 Domains Mathematically...")
    # Oversample to equalize all classes in the training set
    smote = SMOTE(k_neighbors=2, random_state=SMOTE_RANDOM_STATE) # k=2 since some classes only have 4+5 temps = 9 total
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f" -> Orig Train size: {len(X_train)}. SMOTE Balanced size: {len(X_train_res)}")
    return X_train_res, X_test, y_train_res, y_test, le

def train_xgboost(X_train, y_train):
    print("5. Training XGBoost Classifier...")
    # XGBoost configuration highly optimized for embeddings (tree depth 4 prevents overfitting)
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1 # use all CPU cores
    )
    
    clf.fit(X_train, y_train)
    
    # Save Model
    print("6. Saving V4 XGBoost Model...")
    clf.save_model(XGB_PATH)
    return clf

def evaluate_pipeline(clf, X_test, y_test, le):
    print("7. Evaluating V4 Architecture...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_wght = f1_score(y_test, y_pred, average="weighted")
    
    print("=" * 60)
    print(f" V4 PIPELINE RESULTS ")
    print("=" * 60)
    print(f" Accuracy : {acc:.4f}")
    print(f" F1 Macro : {f1_mac:.4f}")
    print(f" F1 Weight: {f1_wght:.4f}")
    print("=" * 60)
    print(" Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test, le = build_embeddings_and_smote(df)
    model = train_xgboost(X_train, y_train)
    evaluate_pipeline(model, X_test, y_test, le)
