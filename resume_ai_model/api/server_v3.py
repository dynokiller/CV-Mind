import time
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from utils.advanced_parser import extract_text_from_pdf, extract_text_from_docx
from utils.resume_ocr import extract_text_from_image
from utils.nlp_cleaner import advanced_clean
from models.resume_scorer import score_resume
from models.inference_v3 import classifier_v3

app = FastAPI(title="AI Resume Analyzer V3", description="Advanced Domain Classifier & Resume Scorer", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class KeywordImpact(BaseModel):
    word: str
    impact: float

class ResumeStrength(BaseModel):
    score: int
    strengths: List[str]
    improvements: List[str]

class AnalyzeResponse(BaseModel):
    request_id: str
    predicted_domain: str
    confidence: float
    top_keywords: List[KeywordImpact]
    resume_strength: ResumeStrength
    latency_ms: float

@app.post("/analyze_resume", response_model=AnalyzeResponse)
async def analyze_resume(file: UploadFile = File(...)):
    start_time = time.time()
    req_id = str(uuid.uuid4())
    
    content = await file.read()
    filename = file.filename.lower()
    
    try:
        # 1. Full Resume Parsing
        if filename.endswith(".pdf"):
            raw_text = await asyncio.to_thread(extract_text_from_pdf, content)
        elif filename.endswith(".docx"):
            raw_text = await asyncio.to_thread(extract_text_from_docx, content)
        elif filename.endswith(".txt"):
            raw_text = content.decode("utf-8", errors="ignore")
        elif filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            raw_text = await asyncio.to_thread(extract_text_from_image, content)
        else:
            raise HTTPException(400, "Unsupported file type. Use PDF, DOCX, TXT, PNG, or JPG.")
            
        # 2. Extract Sections & Text Clean
        cleaned_text = advanced_clean(raw_text)
        
        if not cleaned_text or len(cleaned_text) < 10:
            raise HTTPException(400, "Could not extract sufficient text from document.")
            
        # 3. Model Inference (DistilBERT + SHAP)
        domain, confidence, keywords = await asyncio.to_thread(classifier_v3.predict, cleaned_text)
        
        # 4. Resume Strength Analysis
        scorer_results = score_resume(raw_text) # Use raw text so verbs and grammar exist for rules
        
        latency = round((time.time() - start_time) * 1000, 2)
        
        return AnalyzeResponse(
            request_id=req_id,
            predicted_domain=domain,
            confidence=round(confidence, 4),
            top_keywords=keywords,
            resume_strength=scorer_results,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(500, f"Error processing resume: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server_v3:app", host="0.0.0.0", port=8080)
