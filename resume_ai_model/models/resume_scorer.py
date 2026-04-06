import re

def score_resume(resume_text: str) -> dict:
    """Evaluates the structural and content strength of a resume providing detailed, actionable improvement suggestions."""
    score = 100
    strengths = []
    improvements = []
    
    text_lower = resume_text.lower()
    
    # 1. Word Count Density
    word_count = len(text_lower.split())
    if word_count < 250:
        score -= 20
        improvements.append("Resume is too brief. Expand your 'Experience' and 'Projects' sections to detail your technical impact and daily tasks.")
    elif word_count > 1200:
        score -= 10
        improvements.append("Resume is somewhat dense. Consider trimming boilerplate text to highlight the most relevant points clearly for recruiters.")
    else:
        strengths.append("Excellent text density and overall resume length.")

    # 2. Measurable Metrics / Quantified Achievements (Crucial for ATS)
    # Checks for percentages, dollars, and numbers corresponding to metrics
    has_metrics = bool(re.search(r'\b\d{1,3}%\b|\$\d+([kKmMbB]?)|increased by|decreased by|saved \d+', text_lower))
    if has_metrics:
        strengths.append("Contains quantified achievements (metrics/impact) which strongly appeals to hiring managers.")
    else:
        score -= 15
        improvements.append("Actionable insight: Add measurable achievements (e.g., 'improved efficiency by 35%', 'managed budget of $50k'). Don't just list responsibilities.")

    # 3. Action Verb Analysis
    weak_verbs = ['responsible for', 'worked on', 'helped with', 'managed', 'did', 'assisted', 'participated']
    strong_verbs = ['orchestrated', 'engineered', 'spearheaded', 'pioneered', 'implemented', 'designed', 'optimized', 'accelerated']
    
    found_weak = [verb for verb in weak_verbs if verb in text_lower]
    found_strong = [verb for verb in strong_verbs if verb in text_lower]
    
    if found_weak:
        score -= 10
        improvements.append(f"Replace weak phrasing like '{found_weak[0]}' with impact-driven action verbs (e.g., 'Architected', 'Engineered', 'Spearheaded').")
    
    if len(found_strong) >= 2:
        strengths.append("Strong usage of impactful action verbs detected.")
    elif not found_weak and len(found_strong) < 2:
        improvements.append("Start your bullet points with stronger past-tense action verbs to convey leadership and execution.")

    # 4. Fundamental Sections Validation
    if any(edu in text_lower for edu in ['education', 'university', 'bachelor', 'master', 'degree']):
        strengths.append("Clear education credentials and academic background established.")
    else:
        score -= 10
        improvements.append("Ensure your 'Education' section is clearly labeled and detailed.")

    if any(skills in text_lower for skills in ['skills', 'technologies', 'core competencies', 'proficient']):
        strengths.append("Dedicated skills section or list of competencies is present.")
    else:
        score -= 5
        improvements.append("Add a dedicated 'Skills' or 'Technologies' section to highlight core competencies for ATS matching.")

    if any(exp in text_lower for exp in ['experience', 'employment', 'work history']):
        strengths.append("Professional experience section identified.")
    else:
        score -= 10
        improvements.append("Clearly demarcate your 'Professional Experience' section so parsers can parse your career history.")

    # Cap bounds cleanly
    score = max(0, min(100, score))
    
    return {
        "score": score,
        "strengths": strengths,
        "improvements": improvements
    }
