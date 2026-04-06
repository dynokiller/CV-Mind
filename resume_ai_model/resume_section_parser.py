import re
from typing import Any, Dict, List, Optional, Tuple

import spacy

from .utils.nlp_cleaner import advanced_clean

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None  # graceful degradation – regex/heuristics only


SECTION_HEADERS = {
    "skills": ["skills", "technical skills", "key skills", "skills & tools"],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
    ],
    "education": ["education", "academic background", "qualifications"],
    "projects": ["projects", "personal projects", "academic projects"],
    "certifications": ["certifications", "licenses", "certificates"],
}


HEADER_PATTERN = re.compile(
    r"^\s*(?P<header>[A-Z][A-Z\s/&]+)\s*$"
)


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def detect_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into coarse sections based on headings.

    We operate on a line-by-line basis to keep things fast and robust.
    """
    if not text:
        return {}

    lines = text.splitlines()
    sections: Dict[str, List[str]] = {k: [] for k in SECTION_HEADERS.keys()}
    current_key: Optional[str] = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            # always include blank lines where a section is active
            if current_key:
                sections[current_key].append("")
            continue

        # Detect explicit ALL-CAPS headers
        m = HEADER_PATTERN.match(line)
        if m:
            header_text = m.group("header").lower()
            matched_key = None
            for key, aliases in SECTION_HEADERS.items():
                for alias in aliases:
                    if alias in header_text:
                        matched_key = key
                        break
                if matched_key:
                    break
            if matched_key:
                current_key = matched_key
                continue

        # Otherwise, append to current section if one is active
        if current_key:
            sections[current_key].append(line)

    # Join lines back to text
    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def extract_contact_info(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract candidate name and email from full resume text."""
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    email = email_match.group(0) if email_match else None

    # Name heuristics:
    # - First non-empty line that is not obviously a label
    # - Prefer spacy PERSON entity if model is available
    candidate_name: Optional[str] = None

    # Use top 5 lines to search for name
    top_lines = [l.strip() for l in text.splitlines() if l.strip()][:5]
    if nlp:
        doc = nlp("\n".join(top_lines))
        persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
        if persons:
            candidate_name = persons[0]

    if not candidate_name and top_lines:
        # Fallback: first line that is 2–5 tokens, mostly alphabetic
        for line in top_lines:
            tokens = line.split()
            if 1 < len(tokens) <= 5 and all(re.match(r"^[A-Za-z.\-]+$", t) for t in tokens):
                candidate_name = line
                break

    return candidate_name, email


def extract_skills(skills_text: str) -> List[str]:
    if not skills_text:
        return []

    # Split on commas, semicolons, bullets, and newlines
    raw_tokens = re.split(r"[,;\n•\-–]\s*", skills_text)
    skills: List[str] = []
    for token in raw_tokens:
        token = _normalize_line(token)
        if not token:
            continue
        # Filter out obviously non-skill phrases
        if len(token) > 64:
            continue
        skills.append(token)

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for s in skills:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped


def extract_experience(exp_text: str) -> List[Dict[str, Any]]:
    if not exp_text:
        return []

    entries: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {"role": "", "company": "", "duration": ""}

    for line in exp_text.splitlines():
        norm = _normalize_line(line)
        if not norm:
            # blank line -> finalize current entry if it has content
            if any(current.values()):
                entries.append(current)
                current = {"role": "", "company": "", "duration": ""}
            continue

        # Look for "Role at Company" or "Role - Company"
        m = re.match(
            r"^(?P<role>[^@\-\|•]+)\s+(?:at|@|\-|–|\|)\s+(?P<company>.+)$",
            norm,
            flags=re.IGNORECASE,
        )
        if m:
            if any(current.values()):
                entries.append(current)
                current = {"role": "", "company": "", "duration": ""}
            current["role"] = m.group("role").strip()
            current["company"] = m.group("company").strip()
            continue

        # Duration patterns like "Jan 2020 - Present"
        if re.search(
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}).*(present|current|\d{4})",
            norm,
            flags=re.IGNORECASE,
        ):
            current["duration"] = norm
            continue

        # If neither pattern matched and we don't yet have a role, treat as role
        if not current["role"]:
            current["role"] = norm
        elif not current["company"]:
            current["company"] = norm
        else:
            # Append to duration or role if looks like additional detail
            if current["duration"]:
                current["duration"] += " | " + norm
            else:
                current["role"] += " | " + norm

    if any(current.values()):
        entries.append(current)

    return entries


def extract_education(edu_text: str) -> List[Dict[str, Any]]:
    if not edu_text:
        return []

    entries: List[Dict[str, Any]] = []
    for line in edu_text.splitlines():
        norm = _normalize_line(line)
        if not norm:
            continue

        # Look for "Degree, University" or "Degree - University"
        m = re.match(
            r"^(?P<degree>[^,\-•]+)[,\-–]\s*(?P<university>.+)$",
            norm,
            flags=re.IGNORECASE,
        )
        degree = university = None
        if m:
            degree = m.group("degree").strip()
            university = m.group("university").strip()
        else:
            # Fallback: heuristic for degree keywords
            if re.search(
                r"(b\.?tech|bachelor|master|b\.sc|m\.sc|phd|b\.e\.|m\.e\.)",
                norm,
                flags=re.IGNORECASE,
            ):
                degree = norm
            else:
                university = norm

        entries.append(
            {
                "degree": degree or "",
                "university": university or "",
            }
        )

    return entries


def extract_projects(projects_text: str) -> List[str]:
    if not projects_text:
        return []

    projects: List[str] = []
    for line in projects_text.splitlines():
        norm = _normalize_line(line.lstrip("•-– "))
        if not norm:
            continue
        if len(norm) > 200:
            continue
        projects.append(norm)
    return projects


def clean_for_model(text: str) -> str:
    """
    Final text cleaning pipeline before sending to ML models.

    Uses the existing advanced cleaner, which handles:
    - lowercasing
    - URL/email removal
    - punctuation/number cleanup
    - stopword removal and lemmatization (when spaCy is available)
    """
    return advanced_clean(text)


def parse_resume_text(text: str) -> Dict[str, Any]:
    """
    Main entry point: convert raw resume text (from PDF or OCR)
    into the structured JSON format required by the system.
    """
    text = text or ""
    sections = detect_sections(text)

    name, email = extract_contact_info(text)
    skills = extract_skills(sections.get("skills", ""))
    experience = extract_experience(sections.get("experience", ""))
    education = extract_education(sections.get("education", ""))
    projects = extract_projects(sections.get("projects", ""))

    return {
        "name": name or "",
        "email": email or "",
        "skills": skills,
        "experience": experience,
        "education": education,
        "projects": projects,
        "full_resume_text": text.strip(),
        # Optionally, callers can also pass clean_for_model(text) onward
    }

