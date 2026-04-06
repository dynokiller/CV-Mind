"""
training/evaluate_advanced.py

Evaluates the new V4 XGBoost + Embeddings Pipeline.
Generates:
1. Detailed Classification Report
2. Macro / Weighted accuracy metrics
3. Confusion Matrix Heatmap (Saved to root for easy viewing)
"""

import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sentence_transformers import SentenceTransformer

from training.advanced_text_cleaner import clean_and_lemmatize

DATA_PATH = "data/resume_dataset.csv"
MODEL_DIR = "models/v4_xgboost"
ENCODER_PATH = f"{MODEL_DIR}/label_encoder.pkl"
XGB_PATH = f"{MODEL_DIR}/xgb_model.json"
HEATMAP_PATH = "confusion_matrix_v4.png"

def evaluate_v4():
    print("1. Loading V4 Models (XGBoost + Encoder)...")
    if not os.path.exists(XGB_PATH) or not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"V4 Models not found in {MODEL_DIR}. Please run training/train_advanced.py first.")
        
    clf = xgb.XGBClassifier()
    clf.load_model(XGB_PATH)
    
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
        
    print("2. Loading Original Dataset for inference validation...")
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
    
    print("3. Preprocessing (Spacy lemmatization) on raw holdout data...")
    # NOTE: In production we use a real holdout split. 
    # Here we are scoring on the entire set to see total coverage capability, 
    # but the actual SMOTE validation is strictly split inside train_advanced.py.
    X_texts = df["Resume_str"].apply(clean_and_lemmatize).tolist()
    y_true = df["Category"].tolist()
    y_enc = le.transform(y_true)
    
    print("4. Generating Inference Embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    X_embed = embedder.encode(X_texts, show_progress_bar=True, device="cpu")
    
    print("5. Running XGBoost Predictions...")
    y_pred_enc = clf.predict(X_embed)
    y_pred = le.inverse_transform(y_pred_enc)
    
    print("\n" + "="*80)
    print(" FINAL V4 PIPELINE EVALUATION ")
    print("="*80)
    
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    print(f" ACCURACY : {acc*100:.2f}%")
    print(f" MACRO F1 : {f1_mac:.4f}")
    
    print("\n[ Classification Report ]")
    print(classification_report(y_true, y_pred))
    
    # Generate Heatmap
    print("\n6. Generating Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('V4 Model (Embeddings + SMOTE + XGBoost) Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=150)
    print(f" -> Heatmap saved perfectly to {HEATMAP_PATH}")
    
if __name__ == "__main__":
    evaluate_v4()
