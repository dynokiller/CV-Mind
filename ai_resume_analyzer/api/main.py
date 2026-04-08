"""
api/main.py

Production FastAPI application for the AI Resume Analyzer.

Endpoints:
  POST /analyze          — domain classification + match score + SHAP keywords
  POST /feedback         — active learning: store domain corrections
  POST /bulk-analyze     — rank N resumes by match score (recruiter mode)
  POST /upload-analyze   — unified file parsing → V4 ML prediction flow
  POST /linkedin-analyze — LinkedIn profile text → structured JSON + domain
  GET  /flags            — inspect active feature flags

All ML models are loaded ONCE at startup (lifespan hook).
No ML objects are ever recreated per request.
"""

import csv
import os
import uuid
import time
import logging
import tempfile
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Set up logging before importing ML modules ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("resume_analyzer")


# ── Mail Service Imports ──────────────────────────────────────────────────────
import smtplib
from email.message import EmailMessage
import sys
# Add frontend mail handlers to path to reuse logic
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CV_MIND-main", "CV_MIND-main"))
try:
    from mail_service.handlers import HANDLERS
except ImportError:
    HANDLERS = {}
    logger.warning("[API] Could not import mail handlers from frontend directory.")

# ── Import inference & app modules ───────────────────────────────────────────
from inference.inference import predict_domain, get_feature_flags
from inference.similarity import compute_match_score
from inference.explain import explain_resume
from inference.scoring_engine import calculate_score
from app.models.resume_parser import parse_resume as extract_text
from app.models.feature_extractor import extract_features
from app.models.feedback_generator import generate_feedback

# ── LinkedIn analyzer (lazy — only imported when USE_LINKEDIN=true) ───────────
import os as _os
_USE_LINKEDIN = _os.getenv("USE_LINKEDIN", "false").lower() == "true"
if _USE_LINKEDIN:
    try:
        from app.models.linkedin_analyzer import parse_linkedin_profile
        logger_tmp = logging.getLogger("resume_analyzer")
        logger_tmp.info("[API] LinkedIn analyzer module loaded.")
    except Exception as _e:
        logging.getLogger("resume_analyzer").warning(f"[API] LinkedIn analyzer failed to import: {_e}")
        parse_linkedin_profile = None
else:
    parse_linkedin_profile = None

# ── Feedback CSV path ─────────────────────────────────────────────────────────
FEEDBACK_CSV = "data/feedback_data.csv"
os.makedirs("data", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pydantic Schemas
# ═══════════════════════════════════════════════════════════════════════════════
class AnalyzeRequest(BaseModel):
    resume_text:     str = Field(..., min_length=50,  description="Full plain-text resume")
    job_description: str = Field(..., min_length=20,  description="Target job description")


class AnalyzeResponse(BaseModel):
    request_id:       str
    predicted_domain: str
    confidence:       float
    match_score:      float          # percentage 0-100
    top_keywords:     List[dict]     # [{"word":..., "impact":...}, ...]
    matched_skills:   List[str]
    missing_skills:   List[str]
    latency_ms:       float


# ── Mail Service Schemas ──────────────────────────────────────────────────────
class MailRequest(BaseModel):
    email: str
    type: str
    otp: str = None
    redirect: str = None


class MailResponse(BaseModel):
    success: bool
    message: str = None


class FeedbackRequest(BaseModel):
    resume_text:    str = Field(..., min_length=50)
    correct_domain: str = Field(..., min_length=2)
    user_id:        str = Field(default="anonymous")


class FeedbackResponse(BaseModel):
    correction_id: str
    message:       str


class BulkAnalyzeRequest(BaseModel):
    resumes:         List[str] = Field(..., min_items=1, max_items=50, description="List of resume plain texts")
    job_description: str       = Field(..., min_length=20)


class BulkCandidateResult(BaseModel):
    rank:             int
    resume_index:     int
    predicted_domain: str
    confidence:       float
    match_score:      float
    matched_skills:   List[str]
    missing_skills:   List[str]


class BulkAnalyzeResponse(BaseModel):
    request_id:         str
    total_resumes:      int
    ranked_candidates:  List[BulkCandidateResult]
    latency_ms:         float


class UploadAnalyzeResponse(BaseModel):
    predicted_domain: str
    confidence: float
    skills_found: List[str]
    missing_skills: List[str]
    suggestions: List[str]
    keywords: List[str]
    final_score: float


# ── LinkedIn Schemas ──────────────────────────────────────────────────────────
class LinkedInAnalyzeRequest(BaseModel):
    linkedin_text:   str = Field(..., min_length=30, description="Pasted LinkedIn profile text")
    job_description: str = Field(default="",          description="Optional JD for match scoring")


class LinkedInExperience(BaseModel):
    title:       str = ""
    company:     str = ""
    duration:    str = ""
    description: str = ""


class LinkedInEducation(BaseModel):
    institution: str = ""
    degree:      str = ""
    field:       str = ""
    year:        str = ""


class LinkedInAnalyzeResponse(BaseModel):
    name:             str
    headline:         str
    summary:          str
    location:         str
    skills:           List[str]
    experience:       List[dict]
    education:        List[dict]
    certifications:   List[str]
    languages:        List[str]
    profile_strength: str
    predicted_domain: str
    domain_confidence: float
    match_score:      float
    latency_ms:       float


# ═══════════════════════════════════════════════════════════════════════════════
#  Lifespan — startup / shutdown hooks & Motor client
# ═══════════════════════════════════════════════════════════════════════════════

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
db_client = None
db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client, db
    logger.info("=" * 55)
    logger.info("  AI Resume Analyzer API — STARTING")
    logger.info("  All ML models loaded (singleton pattern).")
    try:
        db_client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = db_client["resume_db"]
        logger.info(f"Connected to MongoDB at {MONGO_URI}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        db = None
    logger.info("=" * 55)
    yield
    if db_client:
        db_client.close()
    logger.info("API shutting down.")


# ═══════════════════════════════════════════════════════════════════════════════
#  App factory
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title        = "AI Resume Analyzer API",
    version      = "2.0.0",
    description  = "Unified RoBERTa classifier + Upload Flow + SentenceTransformer match score",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
    lifespan     = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": {
                "code":    "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again.",
            },
        },
    )


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    elapsed  = round((time.time() - start) * 1000, 2)
    response.headers["X-Latency-ms"] = str(elapsed)
    return response


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
def root():
    return {"message": "AI Resume Analyzer API v2.0 — visit /docs"}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/upload-analyze", response_model=UploadAnalyzeResponse)
@app.post("/parse", response_model=UploadAnalyzeResponse)
async def upload_analyze(
    file: UploadFile = File(...),
    job_description: str = Form(default="")
):
    """
    Unified Pipeline Route:
    1. Upload File & temporarily save
    2. Parse PDF/DOCX/Images
    3. Pass text to V4 NLP prediction
    4. Provide Domain, Skills, and Suggestions
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx', '.png', '.jpg', '.jpeg']:
        raise HTTPException(status_code=400, detail="Unsupported file extension. Supported: pdf, docx, png, jpg, jpeg.")
    
    # Secure temporary space
    fd, temp_filename = tempfile.mkstemp(suffix=file_ext)
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Parse text (OCR implicitly handled inside extract_text/parse_resume)
        text = extract_text(temp_filename)
        if not text or len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract sufficient text or OCR failed.")
            
        print("Extracted Text:", text[:500])
        
        # ML Inference: Domain classification
        clf_result = predict_domain(text)
        prediction_domain = clf_result["predicted_domain"]
        confidence = clf_result["confidence"]
        
        print("Prediction:", clf_result)
        
        # Skill matching and suggestions
        skills_found = []
        missing_skills = []
        match_score = 0.0
        
        features = extract_features(text)
        
        if job_description and len(job_description.strip()) > 20:
            sim_result = compute_match_score(text, job_description)
            skills_found = sim_result["matched_keywords"]
            missing_skills = sim_result["missing_keywords"]
            match_score = sim_result["match_score_percent"]
        else:
            skills_found = features.get("skills", [])
            
        # Extract SHAP top keywords
        try:
            xai_result = explain_resume(text, top_n=10)
            keywords = [kw["word"] for kw in xai_result.get("top_keywords", [])]
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            keywords = []
            
        # Compute dynamic final scoring
        final_score = calculate_score(
            skills_data={"match_score": match_score, "resume_skills": skills_found, "jd_skills": skills_found + missing_skills},
            experience=features.get("experience", "0"),
            education=features.get("education", []),
            domain_confidence=confidence,
            shap_features=keywords
        )
        
        # Generate actionable advice
        suggestions = generate_feedback(final_score, missing_skills)
        
        # Mongo Async Database persistence
        if db is not None:
            doc = {
                "resume_text": text,
                "predicted_domain": prediction_domain,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "skills": skills_found,
                "suggestions": suggestions
            }
            try:
                await db["resumes"].insert_one(doc)
            except Exception as mongo_err:
                logger.error(f"MongoDB save failed: {mongo_err}")
                
        # Return properly structured JSON 
        return UploadAnalyzeResponse(
            predicted_domain = prediction_domain,
            confidence       = confidence,
            skills_found     = skills_found,
            missing_skills   = missing_skills,
            suggestions      = suggestions,
            keywords         = keywords,
            final_score      = final_score
        )
        
    except Exception as e:
        logger.error(f"Upload analyze failed: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="Analysis failed due to an internal error.")
    finally:
        os.close(fd)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest):
    req_id = str(uuid.uuid4())[:8]
    t0     = time.time()
    logger.info(f"[{req_id}] /analyze — text length={len(body.resume_text)}")

    try:
        clf_result = predict_domain(body.resume_text)
        sim_result = compute_match_score(body.resume_text, body.job_description)

        try:
            xai_result = explain_resume(body.resume_text, top_n=10)
            top_keywords = xai_result["top_keywords"]
        except Exception as shap_err:
            logger.warning(f"[{req_id}] SHAP failed (non-fatal): {shap_err}")
            top_keywords = []

        latency = round((time.time() - t0) * 1000, 2)
        logger.info(f"[{req_id}] /analyze done in {latency}ms")

        return AnalyzeResponse(
            request_id       = req_id,
            predicted_domain = clf_result["predicted_domain"],
            confidence       = clf_result["confidence"],
            match_score      = sim_result["match_score_percent"],
            top_keywords     = top_keywords,
            matched_skills   = sim_result["matched_keywords"],
            missing_skills   = sim_result["missing_keywords"],
            latency_ms       = latency,
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[{req_id}] /analyze failed: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(body: FeedbackRequest):
    correction_id = str(uuid.uuid4())[:12]
    logger.info(f"Feedback received — correct_domain='{body.correct_domain}' user='{body.user_id}'")

    row = {
        "correction_id":  correction_id,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "user_id":        body.user_id,
        "resume_text":    body.resume_text.replace("\n", " "),
        "correct_domain": body.correct_domain,
    }

    file_exists = os.path.exists(FEEDBACK_CSV)
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return FeedbackResponse(
        correction_id = correction_id,
        message       = "Thank you! Your correction has been saved and will improve the model.",
    )


@app.post("/bulk-analyze", response_model=BulkAnalyzeResponse)
async def bulk_analyze(body: BulkAnalyzeRequest):
    req_id = str(uuid.uuid4())[:8]
    t0     = time.time()
    n      = len(body.resumes)
    logger.info(f"[{req_id}] /bulk-analyze — {n} resumes")

    if n == 0:
        raise HTTPException(status_code=400, detail="resumes list cannot be empty.")

    results = []
    for idx, resume_text in enumerate(body.resumes):
        try:
            clf = predict_domain(resume_text)
            sim = compute_match_score(resume_text, body.job_description)

            results.append({
                "resume_index":     idx,
                "predicted_domain": clf["predicted_domain"],
                "confidence":       clf["confidence"],
                "match_score":      sim["match_score_percent"],
                "matched_skills":   sim["matched_keywords"],
                "missing_skills":   sim["missing_keywords"],
            })
        except Exception as e:
            logger.warning(f"[{req_id}] Resume index {idx} failed: {e}")
            results.append({
                "resume_index":     idx,
                "predicted_domain": "Error",
                "confidence":       0.0,
                "match_score":      0.0,
                "matched_skills":   [],
                "missing_skills":   [],
            })

    results.sort(key=lambda r: r["match_score"], reverse=True)

    ranked = [
        BulkCandidateResult(rank=i + 1, **r)
        for i, r in enumerate(results)
    ]

    latency = round((time.time() - t0) * 1000, 2)
    logger.info(f"[{req_id}] /bulk-analyze done in {latency}ms")

    return BulkAnalyzeResponse(
        request_id        = req_id,
        total_resumes     = n,
        ranked_candidates = ranked,
        latency_ms        = latency,
    )


# ── Mail Service Dispatcher ────────────────────────────────────────────────────
@app.post("/mail_service/index", response_model=MailResponse)
async def mail_dispatcher(request: Request):
    """
    Unified mail dispatcher ported from Flask mail service.
    Handles OTP_VERIFY and RESET_PASSWORD types.
    """
    try:
        data = await request.json()
    except Exception:
        return MailResponse(success=False, message="Invalid JSON payload")

    email = data.get("email")
    mail_type = data.get("type")

    if not email or not mail_type:
        return MailResponse(success=False, message="email and type required")

    handler = HANDLERS.get(mail_type)
    if not handler:
        return MailResponse(success=False, message=f"Invalid mail type: {mail_type}")

    # SMTP Config from ENV
    mail_user = os.getenv("MAIL_EMAIL")
    mail_pass = os.getenv("MAIL_APP_PASSWORD")
    smtp_host = "smtp.gmail.com"
    smtp_port = 465

    if not mail_user or not mail_pass:
        return MailResponse(success=False, message="Mail credentials not configured in backend ENV")

    try:
        # Build message using existing handler logic
        msg = handler(data)
        
        # Dispatch via SMTP_SSL
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
            smtp.login(mail_user, mail_pass)
            smtp.send_message(msg)
            
        logger.info(f"[MAIL] Sent {mail_type} to {email}")
        return MailResponse(success=True, message="Email sent")

    except Exception as e:
        logger.error(f"[MAIL] Failed to send {mail_type} to {email}: {e}")
        return MailResponse(success=False, message=str(e))


# ── LinkedIn Analyze Endpoint ─────────────────────────────────────────────────
@app.post("/linkedin-analyze", response_model=LinkedInAnalyzeResponse)
async def linkedin_analyze(body: LinkedInAnalyzeRequest):
    """
    Parse a pasted LinkedIn profile and return structured data + domain prediction.

    - Requires USE_LINKEDIN=true env var to enable LinkedIn parsing.
    - Falls back to basic skill extraction from raw text if module unavailable.
    - Optional: include job_description for match scoring.
    """
    t0 = time.time()

    # ── LinkedIn parsing ──────────────────────────────────────────────────
    if parse_linkedin_profile is not None:
        try:
            profile = parse_linkedin_profile(body.linkedin_text)
            if "error" in profile:
                raise HTTPException(status_code=400, detail=profile["error"])
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"[/linkedin-analyze] Profile parse error: {e}")
            profile = {}
    else:
        logger.warning(
            "[/linkedin-analyze] LinkedIn module not loaded. "
            "Set USE_LINKEDIN=true to enable full parsing."
        )
        profile = {}

    # ── Domain classification ─────────────────────────────────────────────
    try:
        clf_result = predict_domain(body.linkedin_text)
        predicted_domain   = clf_result["predicted_domain"]
        domain_confidence  = clf_result["confidence"]
    except Exception as e:
        logger.warning(f"[/linkedin-analyze] Domain prediction error: {e}")
        predicted_domain  = "UNKNOWN"
        domain_confidence = 0.0

    # ── Match scoring (optional) ──────────────────────────────────────────
    match_score = 0.0
    if body.job_description and len(body.job_description.strip()) > 20:
        try:
            sim = compute_match_score(body.linkedin_text, body.job_description)
            match_score = sim.get("match_score_percent", 0.0)
        except Exception as e:
            logger.warning(f"[/linkedin-analyze] Match scoring error: {e}")

    latency = round((time.time() - t0) * 1000, 2)

    return LinkedInAnalyzeResponse(
        name             = profile.get("name", "Unknown"),
        headline         = profile.get("headline", ""),
        summary          = profile.get("summary", ""),
        location         = profile.get("location", ""),
        skills           = profile.get("skills", []),
        experience       = profile.get("experience", []),
        education        = profile.get("education", []),
        certifications   = profile.get("certifications", []),
        languages        = profile.get("languages", []),
        profile_strength = profile.get("profile_strength", "basic"),
        predicted_domain = predicted_domain,
        domain_confidence = domain_confidence,
        match_score      = match_score,
        latency_ms       = latency,
    )


# ── Feature Flags Introspection ───────────────────────────────────────────────
@app.get("/flags")
def flags():
    """Return the current ML feature flags and their active state."""
    f = get_feature_flags()
    f["USE_LINKEDIN"] = _USE_LINKEDIN
    return {
        "flags": f,
        "description": {
            "USE_OCR":                    "Enable OCR for image/scanned-PDF resumes",
            "USE_LINKEDIN":               "Enable /linkedin-analyze endpoint",
            "USE_TRANSFORMER_CLASSIFIER": "Use DistilBERT zero-shot instead of XGBoost",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
