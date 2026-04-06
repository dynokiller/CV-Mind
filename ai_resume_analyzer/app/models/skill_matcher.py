from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
from app.models.feature_extractor import extract_skills
import numpy as np

class SkillMatcher:
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading SentenceTransformer: {e}")
            self.model = None

    def match_skills(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        if not self.model:
            return {"score": 0, "missing_skills": []}

        # 1. Semantic Similarity using Embeddings
        embeddings1 = self.model.encode(resume_text, convert_to_tensor=True)
        embeddings2 = self.model.encode(jd_text, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        similarity_score = float(cosine_scores[0][0]) * 100
        
        # 2. Key Skill Extraction & Missing Skills (Keyword based)
        resume_skills = set(extract_skills(resume_text))
        jd_skills = set(extract_skills(jd_text))
        
        missing_skills = list(jd_skills - resume_skills)
        
        return {
            "match_score": round(similarity_score, 2),
            "missing_skills": missing_skills,
            "resume_skills": list(resume_skills),
            "jd_skills": list(jd_skills)
        }
