import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

def train_pipeline():
    print("Loading Kaggle Dataset...")
    df = pd.read_csv(r"C:\Users\Admin\Downloads\archive\Resume\Resume.csv").dropna(subset=['Resume_str', 'Category'])
    df['Cleaned_Text'] = df['Resume_str'].apply(lambda x: ' '.join(str(x).split()))

    # Stratified Split
    X_raw, y_raw = df['Cleaned_Text'].tolist(), df['Category'].tolist()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    
    X_train_str, X_test_str, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Generating Sentence-BERT Embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    X_train = embedder.encode(X_train_str, show_progress_bar=True)
    X_test = embedder.encode(X_test_str, show_progress_bar=False)

    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(label_encoder.classes_),
        n_estimators=150,
        tree_method='hist',
        device='cuda'
    )
    model.fit(X_train, y_train)

    print("Saving Core Artifacts...")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    model.save_model("xgb_model_analyzer.json")
    print("Training complete.")

if __name__ == "__main__":
    train_pipeline()
