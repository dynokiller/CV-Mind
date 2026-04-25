"""
CV Mind — HuggingFace Space FastAPI Backend
Endpoints:
  GET  /                   → API info
  GET  /health             → health check
  POST /upload-analyze     → file upload (PDF/DOCX) → full analysis JSON
  POST /analyze-text       → raw text JSON → analysis JSON
"""

import io
import os
import time
import json

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Text Extraction ────────────────────────────────────────────────────────────
def extract_text_from_pdf(data: bytes) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc).strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF extraction failed: {e}")


def extract_text_from_docx(data: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"DOCX extraction failed: {e}")


def extract_text(filename: str, data: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(data)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(data)
    else:
        # Try raw decode for plain text uploads
        try:
            return data.decode("utf-8", errors="ignore").strip()
        except Exception:
            raise HTTPException(status_code=422, detail="Unsupported file format")


# ── Analyzer (lazy-loaded singleton) ──────────────────────────────────────────
_analyzer = None


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from resume_analyzer import ResumeAnalyzer
        _analyzer = ResumeAnalyzer()
    return _analyzer


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CV Mind Analyzer API",
    description="Resume & LinkedIn profile analysis powered by XGBoost + SentenceTransformer",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "CV Mind Analyzer API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "upload_file":  "POST /upload-analyze  (multipart: file=<PDF|DOCX>)",
            "analyze_text": "POST /analyze-text    (JSON: {\"text\": \"...\"})",
            "health":       "GET  /health",
            "docs":         "GET  /docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload-analyze")
async def upload_analyze(file: UploadFile = File(...)):
    """
    Accept a PDF or DOCX resume file, extract its text, run the ML analyzer,
    and return a JSON payload compatible with the CV Mind Flask frontend.
    """
    t0 = time.time()

    # 1. Read & validate file size (28 MB max)
    data = await file.read()
    if len(data) > 28 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 28 MB)")

    # 2. Extract text
    text = extract_text(file.filename or "resume.pdf", data)

    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Could not extract sufficient text from file")

    # 3. Run analyzer
    analyzer = get_analyzer()
    result = analyzer.analyze(text)

    latency_ms = round((time.time() - t0) * 1000)

    # 4. Map to schema expected by Flask frontend
    matched   = result.get("matched_keywords", [])
    missing   = result.get("missing_keywords", [])
    confidence = result.get("confidence", 0.0)

    # Derive a 0–100 final score from match ratio + model confidence
    domain_size = len(matched) + len(missing)
    match_ratio = len(matched) / max(domain_size, 1)
    final_score = round((match_ratio * 0.6 + confidence * 0.4) * 100)

    return {
        "predicted_domain": result.get("predicted_domain", "Unknown"),
        "confidence":        confidence,          # 0.0–1.0  (Flask multiplies ×100)
        "skills_found":      matched,             # Flask reads as "strengths"
        "missing_skills":    missing,             # Flask reads as missing keywords
        "suggestions":       result.get("suggestions", []),
        "keywords":          matched[:10],        # top keywords for display
        "final_score":       final_score,         # 0–100
        "full_resume_text":  text[:3000],         # trimmed; used for domain refinement
        "strength_score":    result.get("strength_score", "0/10"),
        "latency_ms":        latency_ms,
    }


class TextPayload(BaseModel):
    text: str


@app.post("/analyze-text")
def analyze_text(payload: TextPayload):
    """
    Accept raw text (LinkedIn profile or pasted resume) and return analysis.
    Used by Flask's /analyze-linkedin route.
    """
    t0 = time.time()

    if not payload.text or len(payload.text.strip()) < 30:
        raise HTTPException(status_code=422, detail="Text too short — please provide more content")

    analyzer = get_analyzer()
    result = analyzer.analyze(payload.text)

    latency_ms = round((time.time() - t0) * 1000)

    matched   = result.get("matched_keywords", [])
    missing   = result.get("missing_keywords", [])
    confidence = result.get("confidence", 0.0)

    domain_size = len(matched) + len(missing)
    match_ratio = len(matched) / max(domain_size, 1)
    final_score = round((match_ratio * 0.6 + confidence * 0.4) * 100)

    return {
        "predicted_domain": result.get("predicted_domain", "Unknown"),
        "confidence":        confidence,
        "skills_found":      matched,
        "missing_skills":    missing,
        "suggestions":       result.get("suggestions", []),
        "keywords":          matched[:10],
        "final_score":       final_score,
        "full_resume_text":  payload.text[:3000],
        "strength_score":    result.get("strength_score", "0/10"),
        "latency_ms":        latency_ms,
    }
