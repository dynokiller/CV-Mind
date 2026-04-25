import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
from tqdm import tqdm
from resume_analyzer import ResumeAnalyzer

def evaluate():
    print("Loading Kaggle Dataset...")
    df = pd.read_csv(r"C:\Users\Admin\Downloads\archive\Resume\Resume.csv").dropna(subset=['Resume_str', 'Category'])
    df['Cleaned_Text'] = df['Resume_str'].apply(lambda x: ' '.join(str(x).split()))

    X_raw, y_raw = df['Cleaned_Text'].tolist(), df['Category'].tolist()
    
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
        
    y_encoded = label_encoder.transform(y_raw)
    
    X_train_str, X_test_str, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Testing on {len(X_test_str)} unseen resumes...")
    
    analyzer = ResumeAnalyzer()
    
    y_pred_labels = []
    # Map true labels to handle the fallback logic
    y_true_labels = label_encoder.inverse_transform(y_test)
    y_true_mapped = ["Software Engineer" if l == "INFORMATION-TECHNOLOGY" else l for l in y_true_labels]

    print("Running inference... (This might take a minute)")
    for text in tqdm(X_test_str):
        result = analyzer.analyze(text)
        y_pred_labels.append(result['predicted_domain'])

    print("\n=== Intelligent Analyzer Evaluation ===")
    print(f"Overall Accuracy: {accuracy_score(y_true_mapped, y_pred_labels) * 100:.2f}%")
    
    # We will get warnings if some labels (like Cyber Security) are in pred but not true
    print("\n=== Classification Report ===")
    print(classification_report(y_true_mapped, y_pred_labels, zero_division=0))

if __name__ == "__main__":
    evaluate()
