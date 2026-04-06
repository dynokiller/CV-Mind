from typing import List

def generate_feedback(score: float, missing_skills: List[str]) -> List[str]:
    """Generates feedback based on the final score."""
    feedback = []
    
    # Score-based feedback
    if score < 50:
        feedback.append("Major Improvement Required: Your resume has a low match with the job description.")
        feedback.append("Tip: Tailor your resume specifically to the keywords in the job description.")
    elif 50 <= score <= 75:
        feedback.append("Good Match: Your profile is relevant, but there are some missing key skills.")
    else:
        feedback.append("Strong Candidate: Your resume is a great match for this role!")
        
    # Missing Skills
    if missing_skills:
        feedback.append(f"Consider adding these missing skills: {', '.join(missing_skills[:5])}.")
    else:
        feedback.append("Your skill set aligns well with the requirements.")
        
    # General Advice
    feedback.append("Ensure your achievements are quantified (e.g., 'Improved efficiency by 20%').")
    feedback.append("Use standard headings like 'Experience', 'Education', and 'Skills' for better ATS parsing.")
    
    return feedback
