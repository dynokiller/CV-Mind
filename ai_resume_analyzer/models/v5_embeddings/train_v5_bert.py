import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer

# 1. Load Data
print("[1/5] Loading Data...")
csv_path = r"C:\Users\Admin\Downloads\archive\Resume\Resume.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=['Resume_str', 'Category'])

# Basic text cleaning (removing excessive whitespace)
df['Cleaned_Text'] = df['Resume_str'].apply(lambda x: ' '.join(str(x).split()))

# 2. Data Augmentation (Optional but recommended for short-text robustness)
# Create short versions of 20% of the data to help the model learn short summaries
print("[2/5] Augmenting data for short-text robustness...")
short_df = df.sample(frac=0.2, random_state=42).copy()
short_df['Cleaned_Text'] = short_df['Cleaned_Text'].apply(lambda x: ' '.join(x.split()[:30]))
df = pd.concat([df, short_df], ignore_index=True)

# 3. Stratified Train/Test Split (Fixes Data Leakage)
X_raw = df['Cleaned_Text'].tolist()
y_raw = df['Category'].tolist()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Generate Embeddings
print("[3/5] Generating Sentence-BERT Embeddings... (This may take a minute)")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X_train = embedder.encode(X_train_raw, show_progress_bar=True)
X_test = embedder.encode(X_test_raw, show_progress_bar=False)

# 5. Train XGBoost on Dense Embeddings
print("[4/5] Training XGBoost Classifier on GPU...")
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    tree_method='hist',
    device='cuda'
)
model.fit(X_train, y_train)

# 6. Evaluation & Thresholding
print("[5/5] Evaluating Model...")
# Get probabilities instead of direct class predictions
y_pred_proba = model.predict_proba(X_test)

# Custom Predict Function with Confidence Threshold
CONFIDENCE_THRESHOLD = 0.40

def predict_with_fallback(probs, threshold=CONFIDENCE_THRESHOLD):
    max_probs = np.max(probs, axis=1)
    best_classes = np.argmax(probs, axis=1)
    
    # If confidence is below threshold, return -1 (which we will map to UNKNOWN)
    return [cls if prob >= threshold else -1 for cls, prob in zip(best_classes, max_probs)]

y_pred_thresholded = predict_with_fallback(y_pred_proba)

# Map back to labels, handling the -1 fallback
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = [label_encoder.inverse_transform([idx])[0] if idx != -1 else "UNKNOWN/REVIEW" for idx in y_pred_thresholded]

print("\n=== Classification Report ===")
print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

# 7. Save Artifacts
print("Saving artifacts...")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
model.save_model("xgb_model_bert.json")
print("Done! Note: The SentenceTransformer model is downloaded dynamically, no need to pickle it.")
