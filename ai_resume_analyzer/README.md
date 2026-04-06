# AI Resume Analyzer — ML System v2.0

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your dataset
Place your Kaggle resume CSV at:
```
data/resume_dataset.csv
```
Column requirements: `Resume_str` (text) + `Category` (label)

### 3. Train the RoBERTa model
```bash
python -m training.train
# Optional args: --epochs 5 --batch_size 8 --lr 2e-5
```
Model saves to `models/roberta-domain/`

### 4. Evaluate
```bash
python -m training.evaluate
```
Results printed + saved to `evaluation_results.txt`

### 5. Run the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Or:
```bash
python -m api.main
```

### 6. Docker
```bash
docker build -t resume-analyzer .
docker run -p 8000:8000 resume-analyzer
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Classify domain + match score + SHAP keywords |
| `POST` | `/feedback` | Submit domain correction (active learning) |
| `POST` | `/bulk-analyze` | Rank N resumes by match score |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/docs` | Swagger UI |

---

## Example: `/analyze`

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python ML engineer with TensorFlow, NLP and deep learning...",
    "job_description": "Looking for a Python ML Engineer with 3+ years..."
  }'
```

**Response:**
```json
{
  "request_id": "7f3a9b12",
  "predicted_domain": "Data Science",
  "confidence": 0.94,
  "match_score": 82.3,
  "top_keywords": [
    {"word": "tensorflow", "impact": 0.38},
    {"word": "nlp",        "impact": 0.27}
  ],
  "matched_skills": ["python", "nlp", "deep"],
  "missing_skills": ["kubernetes", "docker"],
  "latency_ms": 312.5
}
```

---

## Folder Structure

```
ai_resume_analyzer/
├── api/
│   └── main.py          # FastAPI app — 3 endpoints
├── inference/
│   ├── inference.py     # RoBERTa singleton predict
│   ├── explain.py       # SHAP XAI
│   └── similarity.py    # SentenceTransformer match score
├── training/
│   ├── data_loader.py   # CSV → tokenized HuggingFace Dataset
│   ├── train.py         # Fine-tune RoBERTa with Trainer API
│   └── evaluate.py      # Accuracy, F1, confusion matrix
├── utils/
│   └── text_utils.py    # Text cleaning helpers
├── models/              # Saved model weights (git-ignored large files)
├── data/                # Dataset + feedback CSV
├── requirements.txt
└── Dockerfile
```
