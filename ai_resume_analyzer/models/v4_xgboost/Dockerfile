# ─────────────────────────────────────────────────────────────────────
#  🔧 2. Fix Dockerfile (FINAL Version)
# ─────────────────────────────────────────────────────────────────────

# Use Python 3.10 as requested
FROM python:3.10-slim

# Force Hugging Face to ignore cache and fetch fresh artifacts
ENV CACHE_BUSTER=20240408_0955

# Set up user and working directory for Hugging Face Spaces
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

# Install system dependencies (build-essential for xgboost/shap if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY app.py .
COPY xgb_model.json .
COPY label_encoder.pkl .
COPY tfidf_vectorizer.pkl .

# Set correct ownership
RUN chown -R appuser:appuser /home/appuser/app
USER appuser

# Hugging Face Spaces default port
EXPOSE 7860

# 🔧 1. Correct startup command using uvicorn (app:app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
