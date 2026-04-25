import numpy as np
import pickle
import xgboost as xgb
import json
import re
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- DOMAIN KNOWLEDGE BASE ---
DOMAIN_KNOWLEDGE = {
    "Cyber Security": ["siem", "soc", "splunk", "wireshark", "nmap", "metasploit", "threat hunting", "incident response", "malware analysis", "dfir", "edr", "xdr", "penetration testing", "firewall", "cissp"],
    "Data Science": ["python", "machine learning", "deep learning", "nlp", "sql", "tensorflow", "pytorch", "pandas", "data visualization", "statistics", "scikit-learn", "xgboost"],
    "Software Engineer": ["java", "python", "javascript", "react", "node", "aws", "docker", "kubernetes", "microservices", "spring boot", "git", "ci/cd"],
    "HR": ["recruitment", "onboarding", "payroll", "employee relations", "talent acquisition", "labor laws", "performance management", "benefits administration"]
}

# --- RULE-BASED OVERRIDES ---
STRONG_SIGNALS = {
    "Cyber Security": ["siem", "soc", "penetration testing", "incident response", "metasploit", "threat hunting"],
    "Data Science": ["machine learning", "deep learning", "nlp", "pytorch", "tensorflow"]
}

class ResumeAnalyzer:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = xgb.XGBClassifier()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "xgb_model_analyzer.json")
        encoder_path = os.path.join(base_dir, "label_encoder.pkl")
        
        self.model.load_model(model_path)
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
            
    def _extract_keywords(self, text, top_n=15):
        # Lightweight TF-IDF for single document keyword extraction
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            # Sort by score descending
            top_indices = scores.argsort()[-top_n:][::-1]
            return [feature_names[i] for i in top_indices]
        except ValueError:
            return [] # In case text is empty or only stop words

    def _check_rules(self, text_lower):
        for domain, signals in STRONG_SIGNALS.items():
            for signal in signals:
                if re.search(r'\b' + re.escape(signal) + r'\b', text_lower):
                    return domain
        return None

    def analyze(self, text):
        text_lower = text.lower()
        
        # 1. Check Rule-Based Overrides
        rule_domain = self._check_rules(text_lower)
        
        if rule_domain:
            predicted_domain = rule_domain
            confidence = 0.95
        else:
            # 2. Semantic Prediction
            embedding = self.embedder.encode([text])
            probs = self.model.predict_proba(embedding)[0]
            max_idx = np.argmax(probs)
            confidence = float(probs[max_idx])
            
            # Map standard model domains
            predicted_domain = self.label_encoder.inverse_transform([max_idx])[0]
            
            if predicted_domain == "INFORMATION-TECHNOLOGY":
                predicted_domain = "Software Engineer" # Default IT fallback

        # 3. Keyword Extraction
        resume_keywords = self._extract_keywords(text_lower)
        
        # 4. Gap Analysis
        domain_skills = DOMAIN_KNOWLEDGE.get(predicted_domain, [])
        matched = [skill for skill in domain_skills if skill in text_lower]
        missing = [skill for skill in domain_skills if skill not in text_lower]
        
        # 5. Strength Score
        score = len(matched) / max(len(domain_skills), 1) * 10
        strength_score = f"{round(score, 1)}/10"
        
        # 6. Actionable Suggestions
        suggestions = []
        if missing:
            suggestions.append(f"Consider adding missing core skills like: {', '.join(missing[:3])}")
        if score < 5:
            suggestions.append(f"Your resume lacks strong signals for {predicted_domain}. Detail your specific projects.")
        else:
            suggestions.append(f"Strong match for {predicted_domain}! Ensure your impact is quantified.")

        return {
            "predicted_domain": predicted_domain,
            "confidence": round(confidence, 2),
            "matched_keywords": matched,
            "missing_keywords": missing,
            "strength_score": strength_score,
            "suggestions": suggestions
        }

if __name__ == "__main__":
    print("Initializing Analyzer...")
    analyzer = ResumeAnalyzer()
    
    print("\n--- Testing Analyzer ---")
    
    sample_cyber = "Experienced IT professional specializing in penetration testing, threat hunting, and vulnerability assessments. Familiar with Wireshark and Python."
    print("\nSample 1: Cyber Security Professional")
    result_cyber = analyzer.analyze(sample_cyber)
    print(json.dumps(result_cyber, indent=2))

    sample_ds = "Data Analyst with 3 years of experience. Proficient in SQL, Python, Pandas, and basic data visualization."
    print("\nSample 2: Data Analyst (Should map to Data Science)")
    result_ds = analyzer.analyze(sample_ds)
    print(json.dumps(result_ds, indent=2))
