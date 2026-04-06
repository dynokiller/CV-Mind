"""
Unified AI Resume Analyzer API

Single endpoint:
  POST /analyze_resume

Inputs:
  - PDF / DOCX / image resume (UploadFile)
  - OR LinkedIn URL

Pipeline:
  1. Detect input type
  2. Use MODEL 2 (OCR) or standard extractors to get text
  3. For LinkedIn, use MODEL 3 to get structured profile
  4. Send clean resume text to MODEL 1 (intelligence)
  5. Return a unified JSON response
"""

from __future__ import annotations

import mimetypes
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from services.resume_intelligence_service import analyze_resume_text
from services.ocr_service import extract_from_pdf, extract_from_docx, extract_from_image
from services.linkedin_service import analyze_linkedin_profile


# ── Config (simplified for demo; move to env/config in production) ─────────────

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
VALID_API_KEYS = {"dev-api-key"}  # override via env/config in real deployments
RATE_LIMIT_PER_MINUTE = 30  # per API key


app = FastAPI(
    title="Unified AI Resume Analyzer",
    version="1.0.0",
    description="Single-entry resume analysis pipeline over three core models.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Security: API key auth + rate limiting ─────────────────────────────────────

_rate_limit_store: Dict[str, Deque[float]] = defaultdict(deque)


def _check_rate_limit(api_key: str) -> None:
    now = time.time()
    window_start = now - 60.0
    dq = _rate_limit_store[api_key]
    while dq and dq[0] < window_start:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    dq.append(now)


def verify_api_key(x_api_key: str = Header(...)) -> str:
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    _check_rate_limit(x_api_key)
    return x_api_key


# ── Request / response models ──────────────────────────────────────────────────


class AnalyzeResponse(BaseModel):
    name: str
    email: str
    skills: list[str]
    predicted_domain: str
    domain_confidence: float
    resume_score: float
    missing_skills: list[str]
    suggestions: list[str]
    full_resume_text: str


# ── Helpers ────────────────────────────────────────────────────────────────────


def _detect_file_type(upload: UploadFile) -> str:
    content_type = (upload.content_type or "").lower()
    filename = upload.filename or ""
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    if content_type == "application/pdf" or ext == "pdf":
        return "pdf"
    if ext == "docx" or content_type in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }:
        return "docx"
    if content_type.startswith("image/") or ext in {"png", "jpg", "jpeg"}:
        return "image"

    guessed, _ = mimetypes.guess_type(filename)
    if guessed == "application/pdf":
        return "pdf"
    if guessed and guessed.startswith("image/"):
        return "image"
    if ext == "docx":
        return "docx"
    return "unknown"


async def _read_and_validate_file(upload: UploadFile) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 5MB).")
    return data


# ── Global error handler & logging middleware ──────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # In production, plug into proper logging/observability
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": {"code": "INTERNAL_SERVER_ERROR", "message": "Unexpected error."},
        },
    )


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000.0, 2)
    response.headers["X-Latency-ms"] = str(elapsed)
    return response


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze_resume", response_model=AnalyzeResponse)
async def analyze_resume(
    api_key: str = Depends(verify_api_key),
    resume_file: Optional[UploadFile] = File(None),
    linkedin_url: Optional[HttpUrl] = Form(None),
):
    """
    Unified pipeline over:
      - MODEL 1 (resume intelligence)
      - MODEL 2 (OCR)
      - MODEL 3 (LinkedIn extraction)
    """
    # Input validation: exactly one of (file, linkedin_url)
    if (resume_file is None and linkedin_url is None) or (
        resume_file is not None and linkedin_url is not None
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide either a resume_file OR a linkedin_url, not both.",
        )

    # LinkedIn path → MODEL 3 then MODEL 1
    if linkedin_url is not None:
        result = analyze_linkedin_profile(str(linkedin_url))
        return AnalyzeResponse(
            name=result.get("name", ""),
            email=result.get("email", ""),
            skills=result.get("skills", []),
            predicted_domain=result.get("predicted_domain", "Unknown"),
            domain_confidence=float(result.get("domain_confidence", 0.0)),
            resume_score=float(result.get("resume_score", 0.0)),
            missing_skills=result.get("missing_skills", []),
            suggestions=result.get("suggestions", []),
            full_resume_text=result.get("full_resume_text", ""),
        )

    # File-based path
    assert resume_file is not None
    file_bytes = await _read_and_validate_file(resume_file)
    file_type = _detect_file_type(resume_file)

    if file_type == "unknown":
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use PDF, DOCX, or image formats.",
        )

    if file_type == "pdf":
        combined_text, _ = extract_from_pdf(file_bytes)
        raw_text = combined_text
    elif file_type == "docx":
        combined_text, _ = extract_from_docx(file_bytes)
        raw_text = combined_text
    else:  # image
        raw_text = extract_from_image(file_bytes)

    if not raw_text or len(raw_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Could not extract sufficient text from document.")

    intelligence = analyze_resume_text(raw_text)

    return AnalyzeResponse(
        name=intelligence.get("name", ""),
        email=intelligence.get("email", ""),
        skills=intelligence.get("skills", []),
        predicted_domain=intelligence.get("predicted_domain", "Unknown"),
        domain_confidence=float(intelligence.get("domain_confidence", 0.0)),
        resume_score=float(intelligence.get("resume_score", 0.0)),
        missing_skills=intelligence.get("missing_skills", []),
        suggestions=intelligence.get("suggestions", []),
        full_resume_text=intelligence.get("full_resume_text", raw_text),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main_api:app", host="0.0.0.0", port=8100, reload=False)

