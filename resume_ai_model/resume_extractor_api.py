from __future__ import annotations

import mimetypes
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from .pdf_parser import extract_and_normalize as extract_pdf_text
from .utils.resume_ocr import extract_text_from_image
from .resume_section_parser import parse_resume_text, clean_for_model
from .linkedin_scraper import scrape_linkedin_profile_json

# Optional: Selenium imports for LinkedIn scraping
from selenium import webdriver  # type: ignore
from selenium.webdriver.chrome.options import Options as ChromeOptions  # type: ignore


app = FastAPI(title="Resume Extraction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LinkedInRequest(BaseModel):
    linkedin_url: HttpUrl


def _detect_file_type(upload: UploadFile) -> str:
    """
    Very fast and simple file type detection.

    Returns: "pdf" | "image" | "unknown"
    """
    content_type = (upload.content_type or "").lower()
    filename = upload.filename or ""
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    if content_type == "application/pdf" or ext == "pdf":
        return "pdf"

    if content_type.startswith("image/") or ext in {"png", "jpg", "jpeg"}:
        return "image"

    # Fallback guess via mimetypes
    guessed, _ = mimetypes.guess_type(filename)
    if guessed == "application/pdf":
        return "pdf"
    if guessed and guessed.startswith("image/"):
        return "image"

    return "unknown"


def _build_response(structured: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """
    Attach full raw text and an additional cleaned field for downstream ML models.
    """
    response = dict(structured)
    response["full_resume_text"] = raw_text.strip()
    # Provide a model-ready cleaned variant without changing the primary schema
    response["clean_text_for_model"] = clean_for_model(raw_text)
    return response


@app.post("/extract_resume")
async def extract_resume(resume_file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Unified endpoint for:
    - PDF resumes
    - Image resumes (PNG/JPG)

    Automatically detects the file type and runs the appropriate pipeline.
    """
    file_type = _detect_file_type(resume_file)
    if file_type == "unknown":
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF or image (PNG/JPG).",
        )

    file_bytes = await resume_file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    if file_type == "pdf":
        raw, normalized = extract_pdf_text(file_bytes)
        if not raw.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from PDF resume.")
        structured = parse_resume_text(normalized or raw)
        return _build_response(structured, raw)

    # Image-based resume
    raw_text = extract_text_from_image(file_bytes)
    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from image resume.")
    structured = parse_resume_text(raw_text)
    return _build_response(structured, raw_text)


def _create_headless_chrome() -> webdriver.Chrome:
    """Create a headless Chrome instance for LinkedIn scraping."""
    options = ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)
    return driver


@app.post("/extract_linkedin")
def extract_linkedin(payload: LinkedInRequest) -> Dict[str, Any]:
    """
    Extract a LinkedIn profile and convert it into the unified resume JSON format.

    NOTE:
    - In production you should manage Selenium/WebDriver lifecycle and authentication
      (e.g., cookies / logged-in session) outside this endpoint for reliability.
    """
    driver = None
    try:
        driver = _create_headless_chrome()
        result = scrape_linkedin_profile_json(payload.linkedin_url, driver=driver)
        # Ensure the same additional fields as /extract_resume
        full_text = result.get("full_resume_text", "")
        return _build_response(result, full_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LinkedIn scraping failed: {e}")
    finally:
        if driver is not None:
            driver.quit()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

