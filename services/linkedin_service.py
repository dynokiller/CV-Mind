"""
Service wrapper for MODEL 3 — LinkedIn Profile Extraction.
"""

from typing import Any, Dict

from models.linkedin_model import extract_linkedin_profile
from services.resume_intelligence_service import analyze_resume_text


def analyze_linkedin_profile(url: str) -> Dict[str, Any]:
    """
    Fetch LinkedIn profile, convert to resume-shaped JSON, then
    run MODEL 1 on the full text to get domain, score, gaps, suggestions.
    """
    profile = extract_linkedin_profile(url)
    full_text = profile.get("full_resume_text", "") or ""

    intelligence = analyze_resume_text(full_text) if full_text.strip() else {
        "predicted_domain": "Unknown",
        "domain_confidence": 0.0,
        "resume_score": 0.0,
        "missing_skills": [],
        "suggestions": [],
        "name": profile.get("name", ""),
        "email": "",
        "skills": profile.get("skills", []),
        "full_resume_text": full_text,
    }

    # Merge structural profile info with intelligence output
    result: Dict[str, Any] = {
        "name": intelligence.get("name") or profile.get("name", ""),
        "email": intelligence.get("email", ""),
        "skills": intelligence.get("skills", profile.get("skills", [])),
        "predicted_domain": intelligence.get("predicted_domain", "Unknown"),
        "domain_confidence": intelligence.get("domain_confidence", 0.0),
        "resume_score": intelligence.get("resume_score", 0.0),
        "missing_skills": intelligence.get("missing_skills", []),
        "suggestions": intelligence.get("suggestions", []),
        "full_resume_text": intelligence.get("full_resume_text", full_text),
        "experience": profile.get("experience", []),
        "education": profile.get("education", []),
        "projects": profile.get("projects", []),
    }
    return result

