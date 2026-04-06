"""
MODEL 1 — Resume Intelligence Model.

Responsibilities:
- Domain classification (SentenceTransformer + XGBoost via v4 model)
- Resume scoring (0–100)
- Skill gap detection
- Human-readable suggestions
"""

from typing import Any, Dict, List

from ai_resume_analyzer.app.models.feature_extractor import extract_features
from ai_resume_analyzer.inference.inference import predict_domain
from resume_ai_model.models.resume_scorer import score_resume

from .skill_gap_engine import detect_skill_gaps


def _compute_component_scores(
    extracted: Dict[str, Any],
    predicted_domain: str,
    domain_confidence: float,
    missing_skills: List[str],
) -> Dict[str, float]:
    """
    Derive sub-scores for skills / experience / education / keywords / structure.
    """
    skills = extracted.get("skills", [])
    education = extracted.get("education", [])
    years_str = extracted.get("experience", "0")

    # Skills score based on coverage vs missing skills
    if skills or missing_skills:
        total = len(skills) + len(missing_skills)
        skills_score = max(0.0, 100.0 * (len(skills) / total)) if total else 50.0
    else:
        skills_score = 50.0

    # Experience score from naive years-of-experience
    try:
        years = int(years_str)
    except (ValueError, TypeError):
        years = 0
    if years >= 5:
        exp_score = 100.0
    elif years >= 3:
        exp_score = 80.0
    elif years >= 1:
        exp_score = 60.0
    else:
        exp_score = 40.0

    # Education score
    edu_text = " ".join(education).lower()
    if any(k in edu_text for k in ["phd", "doctorate"]):
        edu_score = 100.0
    elif any(k in edu_text for k in ["master", "m.sc", "m.tech", "mba"]):
        edu_score = 90.0
    elif any(k in edu_text for k in ["bachelor", "b.sc", "b.tech"]):
        edu_score = 80.0
    else:
        edu_score = 60.0

    # Keyword match score: proportion of non-missing skills
    if skills or missing_skills:
        total_kw = len(skills) + len(missing_skills)
        keyword_score = max(0.0, 100.0 * (len(skills) / total_kw))
    else:
        keyword_score = 50.0

    # Structure score from `score_resume`
    structure_raw = score_resume(extracted.get("raw_text", ""))["score"]
    structure_score = float(structure_raw)

    # Domain confidence scaled
    domain_score = float(domain_confidence) * 100.0

    return {
        "skills_score": round(skills_score, 2),
        "experience_score": round(exp_score, 2),
        "education_score": round(edu_score, 2),
        "keyword_match_score": round(keyword_score, 2),
        "structure_score": round(structure_score, 2),
        "domain_score": round(domain_score, 2),
    }


def _aggregate_resume_score(components: Dict[str, float]) -> float:
    """
    Weighted combination into a single 0-100 ATS style score.
    """
    final_score = (
        components["skills_score"] * 0.30
        + components["experience_score"] * 0.20
        + components["education_score"] * 0.10
        + components["keyword_match_score"] * 0.20
        + components["structure_score"] * 0.10
        + components["domain_score"] * 0.10
    )
    return round(max(0.0, min(100.0, final_score)), 2)


def _build_suggestions(
    final_score: float,
    missing_skills: List[str],
    weak_sections: List[str],
) -> List[str]:
    suggestions: List[str] = []

    if final_score < 50:
        suggestions.append(
            "Your resume currently has a low match to typical requirements. "
            "Tailor it more closely to your target domain and job description."
        )
    elif final_score < 75:
        suggestions.append(
            "Your profile is relevant but can be strengthened with more targeted skills and clearer impact."
        )
    else:
        suggestions.append("Strong overall profile. You can still refine phrasing and quantify more impact.")

    if missing_skills:
        suggestions.append(
            "Consider adding or demonstrating experience with: " + ", ".join(missing_skills[:8])
        )

    if "skills" in weak_sections:
        suggestions.append("Add a clearly labeled 'Skills' section listing your core technologies and tools.")

    suggestions.append("Include quantified achievements (e.g., 'improved latency by 30%', 'reduced costs by 15%').")
    suggestions.append("Use clear headings like 'Experience', 'Education', and 'Projects' for better ATS parsing.")
    return suggestions


def analyze_resume_text(resume_text: str) -> Dict[str, Any]:
    """
    High-level entry point used by the unified API.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text cannot be empty.")

    # 1) Extract basic entities & skills
    features = extract_features(resume_text)
    features["raw_text"] = resume_text

    # 2) Domain classification (SentenceTransformer + XGBoost)
    domain_result = predict_domain(resume_text)
    predicted_domain = domain_result["predicted_domain"]
    confidence = float(domain_result["confidence"])

    # 3) Skill gaps
    gaps = detect_skill_gaps(predicted_domain, features.get("skills", []))
    missing_skills = gaps["missing_skills"]
    weak_sections = gaps["weak_sections"]

    # 4) Component scores + final score
    component_scores = _compute_component_scores(
        extracted=features,
        predicted_domain=predicted_domain,
        domain_confidence=confidence,
        missing_skills=missing_skills,
    )
    resume_score = _aggregate_resume_score(component_scores)

    # 5) Suggestions
    suggestions = _build_suggestions(resume_score, missing_skills, weak_sections)

    return {
        "name": features.get("name", "Unknown"),
        "email": features.get("email", "Unknown"),
        "skills": features.get("skills", []),
        "predicted_domain": predicted_domain,
        "domain_confidence": round(confidence, 4),
        "resume_score": resume_score,
        "missing_skills": missing_skills,
        "suggestions": suggestions,
        "full_resume_text": resume_text.strip(),
        "component_scores": component_scores,
    }

