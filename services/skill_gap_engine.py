"""
Skill gap detection and domain skill database for MODEL 1.
"""

from typing import Dict, List, Set


DOMAIN_SKILLS: Dict[str, Set[str]] = {
    "Cybersecurity": {
        "network security",
        "penetration testing",
        "siem",
        "ids",
        "ips",
        "firewalls",
        "incident response",
        "vulnerability assessment",
        "nmap",
        "wireshark",
        "linux",
        "cloud security",
    },
    "Data Science": {
        "python",
        "pandas",
        "numpy",
        "scikit-learn",
        "statistics",
        "machine learning",
        "sql",
        "data visualization",
        "matplotlib",
        "seaborn",
        "feature engineering",
    },
    "Web Development": {
        "javascript",
        "react",
        "node.js",
        "html",
        "css",
        "typescript",
        "rest api",
        "django",
        "flask",
        "fastapi",
    },
    "Cloud": {
        "aws",
        "azure",
        "gcp",
        "cloudformation",
        "terraform",
        "iam",
        "cloud security",
        "serverless",
        "kubernetes",
    },
    "DevOps": {
        "docker",
        "kubernetes",
        "ci/cd",
        "jenkins",
        "github actions",
        "helm",
        "ansible",
        "monitoring",
        "prometheus",
        "grafana",
    },
    "AI/ML": {
        "pytorch",
        "tensorflow",
        "transformers",
        "nlp",
        "computer vision",
        "mle",
        "model deployment",
        "mlops",
        "feature store",
    },
}


def normalise_skill(s: str) -> str:
    return s.strip().lower()


def detect_skill_gaps(
    predicted_domain: str,
    resume_skills: List[str],
) -> Dict[str, List[str]]:
    """
    Compare extracted resume skills against the domain skill database.
    """
    domain_key = predicted_domain
    if domain_key not in DOMAIN_SKILLS:
        # Best effort: if label from model differs (e.g. "Data Science / ML")
        for k in DOMAIN_SKILLS:
            if k.lower() in predicted_domain.lower():
                domain_key = k
                break

    target_skills = DOMAIN_SKILLS.get(domain_key, set())
    if not target_skills:
        return {"missing_skills": [], "weak_sections": []}

    resume_norm = {normalise_skill(s) for s in resume_skills}
    missing = sorted(
        {s for s in target_skills if s not in resume_norm}
    )

    weak_sections: List[str] = []
    coverage = 1.0 - (len(missing) / max(1, len(target_skills)))
    if coverage < 0.5:
        weak_sections.append("skills")
    # simple placeholders for extension:
    # experience/education sections could be flagged via heuristics later

    return {"missing_skills": missing, "weak_sections": weak_sections}

