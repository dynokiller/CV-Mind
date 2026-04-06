from typing import Dict, Any, List

def calculate_score(
    skills_data: Dict[str, Any],
    experience: str,
    education: list,
    domain_confidence: float = 0.0,
    shap_features: List[str] = None
) -> float:
    """
    Calculates the final ATS score (0-100) based on loaded weights.
    
    Weights:
    - Skill Similarity: 40%
    - Experience: 20%
    - Education: 10%
    - Keyword Match: 20%
    - Domain Confidence: 10%
    """
    
    # Robust type handling for domain_confidence
    try:
        domain_confidence = float(domain_confidence)
    except (ValueError, TypeError):
        domain_confidence = 0.0
        
    # 1. Skill Similarity (0-100)
    skill_sim_score = skills_data.get('match_score', 0)
    
    # 2. Experience Score (0-100)
    try:
        years = int(experience)
    except (ValueError, TypeError):
        years = 0
        
    if years >= 5:
        exp_score = 100
    elif years >= 3:
        exp_score = 80
    elif years >= 1:
        exp_score = 60
    else:
        exp_score = 40
        
    # 3. Education Score (0-100)
    # Simple heuristic: higher education = higher score
    edu_str = str(education).lower() if education else ""
    if any(deg in edu_str for deg in ['phd', 'doctorate']):
        edu_score = 100
    elif any(deg in edu_str for deg in ['master', 'm.sc', 'm.tech', 'mba']):
        edu_score = 90
    elif any(deg in edu_str for deg in ['bachelor', 'b.sc', 'b.tech']):
        edu_score = 80
    else:
        edu_score = 60
        
    # 4. Keyword Match Score (0-100)
    # Based on % of found skills vs total JD skills
    resume_skills = set(skills_data.get('resume_skills', []))
    jd_skills = set(skills_data.get('jd_skills', []))
    
    total_jd_skills = len(jd_skills)
    
    if total_jd_skills > 0:
        keyword_score = min((len(resume_skills) / total_jd_skills) * 100, 100)
    else:
        keyword_score = 0  # If no skills in JD, this metric is irrelevant or 0
        
    # SHAP Feature Importance Boost: 
    # Boost keyword score if SHAP top words overlap with JD skills
    if shap_features and total_jd_skills > 0:
        shap_words = set(str(word).lower() for word in shap_features)
        jd_skills_lower = set(str(skill).lower() for skill in jd_skills)
        shap_overlap = len(shap_words.intersection(jd_skills_lower))
        
        # Boost up to 15 points
        boost = min(shap_overlap * 5, 15)
        keyword_score = min(keyword_score + boost, 100)
        
    # 5. Domain Confidence Score (0-100)
    domain_score = domain_confidence * 100
    
    # Final Weighted Calculation
    final_score = (
        (skill_sim_score * 0.40) +
        (exp_score * 0.20) +
        (edu_score * 0.10) +
        (keyword_score * 0.20) +
        (domain_score * 0.10)
    )
    
    return round(final_score, 2)
