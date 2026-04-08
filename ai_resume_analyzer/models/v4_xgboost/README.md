---
title: AI Resume Domain Classifier (V4.5)
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
private: false
---

# AI Resume Domain Classifier (V4.5)

This space provides professional domain classification for resumes using an **XGBoost** classifier paired with a **TF-IDF vectorizer** and custom NLP preprocessing.

## Model Performance (Holdout Set)

- **Accuracy**: 93.52%
- **Macro F1-Score**: 0.9309

---

## 🛠️ Usage (Free API)

This space provides a free REST API. You can call it from any application (like your Render-hosted frontend).

### 1. API Endpoint
`https://dyno0126-cv.hf.space/run/predict`

### 2. cURL Example
```bash
curl -X POST https://dyno0126-cv.hf.space/run/predict -H 'Content-Type: application/json' -d '{
  "data": ["Pasted Resume Text..."]
}'
```

### 3. Python Example
```python
from gradio_client import Client

client = Client("dyno0126/CV")
result = client.predict(
		"Experienced Python developer with 5 years in AWS and Docker...",
		api_name="/predict"
)
print(result)
```

## 🏗️ Model Architecture
- **Training Corpus**: ~17,000 records (Kaggle Resumes + Hugging Face Job Descriptions).
- **Preprocessing**: SpaCy lemmatization (`en_core_web_sm`) + automated skill signal injection.
- **Vectorizer**: TF-IDF (10,000 max features, n-grams 1-2).
- **Classifier**: Multi-class XGBoost with SMOTE balancing.
