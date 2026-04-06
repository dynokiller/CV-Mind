
import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.text_cleaner import clean_text

# Paths
DATASET_PATH = r'c:\Users\Admin\Desktop\Resume\ai_resume_analyzer\dataset\resumes_dataset.csv'
MODEL_PATH = r'c:\Users\Admin\Desktop\Resume\ai_resume_analyzer\app\ml\model.pkl'
VECTORIZER_PATH = r'c:\Users\Admin\Desktop\Resume\ai_resume_analyzer\app\ml\vectorizer.pkl'

def evaluate():
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found.")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    if 'Resume_str' not in df.columns:
        if 'resume_text' in df.columns:
             df.rename(columns={'resume_text': 'Resume_str'}, inplace=True)
    if 'Category' not in df.columns:
        if 'category' in df.columns:
             df.rename(columns={'category': 'Category'}, inplace=True)

    print(f"Dataset Shape: {df.shape}")
    
    print("Cleaning text...")
    df['Cleaned_Resume'] = df['Resume_str'].apply(clean_text)
    
    # Split data (Same as training)
    _, X_test, _, y_test = train_test_split(
        df['Cleaned_Resume'], 
        df['Category'], 
        test_size=0.2, 
        random_state=42
    )
    
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    print("Vectorizing test data...")
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Predicting...")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    with open('evaluation_results.txt', 'w') as f:
        sys.stdout = f
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix (Summary):")
        # Just printing the shape or first few rows to confirm
        print(confusion_matrix(y_test, y_pred))
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    print("Evaluation complete. Results written to evaluation_results.txt")
    
if __name__ == "__main__":
    evaluate()
