import time
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from utils.pdf_parser import extract_text_from_pdf
from utils.docx_parser import extract_text_from_docx
from utils.text_cleaner import clean_text
from models.inference import classifier

app = FastAPI(title="AI Resume Analyzer API", description="High-Accuracy Resume Domain Classifier", version="1.0.0")

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

class AnalyzeResponse(BaseModel):
    request_id: str
    predicted_domain: str
    confidence: float
    top_keywords: List[KeywordImpact]
    latency_ms: float

@app.post("/analyze_resume", response_model=AnalyzeResponse)
async def analyze_resume(file: UploadFile = File(...)):
    start_time = time.time()
    req_id = str(uuid.uuid4())
    
    content = await file.read()
    filename = file.filename.lower()
    
    try:
        if filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(content)
        elif filename.endswith(".docx"):
            raw_text = extract_text_from_docx(content)
        elif filename.endswith(".txt"):
            raw_text = content.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(400, "Unsupported file type. Use PDF, DOCX, or TXT.")
            
        cleaned_text = clean_text(raw_text)
        
        if not cleaned_text or len(cleaned_text) < 10:
            raise HTTPException(400, "Could not extract sufficient text from document.")
            
        # Model Inference
        domain, confidence, keywords = classifier.predict(cleaned_text)
        
        latency = round((time.time() - start_time) * 1000, 2)
        
        return AnalyzeResponse(
            request_id=req_id,
            predicted_domain=domain,
            confidence=round(confidence, 4),
            top_keywords=keywords,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(500, f"Error processing resume: {str(e)}")

# Add a health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000)
