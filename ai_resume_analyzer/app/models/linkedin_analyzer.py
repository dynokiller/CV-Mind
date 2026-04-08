"""
app/models/linkedin_analyzer.py

LinkedIn Profile Analyzer for the AI Resume Analyzer.

Accepts pasted LinkedIn profile text and extracts structured information using
spaCy NLP + regex heuristics.  No scraping — operates on text input only.

Usage:
    from app.models.linkedin_analyzer import parse_linkedin_profile

    json_result = parse_linkedin_profile(linkedin_text)

Output JSON schema:
    {
        "name":       str,
        "headline":   str,
        "summary":    str,
        "location":   str,
        "skills":     List[str],
        "experience": List[{title, company, duration, description}],
        "education":  List[{degree, institution, year}],
        "certifications": List[str],
        "languages":  List[str],
        "profile_strength": "basic" | "intermediate" | "strong"
    }
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("resume_analyzer.linkedin")

# ── Lazy spaCy load ───────────────────────────────────────────────────────────
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            logger.info("[LinkedIn] spaCy model loaded.")
        except OSError:
            try:
                from spacy.cli import download
                download("en_core_web_sm")
                import spacy
                _nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.error(f"[LinkedIn] spaCy load failed: {e}")
                _nlp = None
    return _nlp


# ═══════════════════════════════════════════════════════════════════════════════
#  Section detection helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Section header patterns (case insensitive) that appear in copy-pasted profiles
_SECTION_PATTERNS = {
    "experience":      re.compile(r"^(experience|work experience|employment history|professional experience)\s*$", re.I | re.M),
    "education":       re.compile(r"^(education|academic background|qualifications)\s*$", re.I | re.M),
    "skills":          re.compile(r"^(skills|top skills|technical skills|skill set|core competencies)\s*$", re.I | re.M),
    "summary":         re.compile(r"^(summary|about|about me|profile|professional summary|objective)\s*$", re.I | re.M),
    "certifications":  re.compile(r"^(certifications?|licenses? & certifications?|credentials?)\s*$", re.I | re.M),
    "languages":       re.compile(r"^(languages?)\s*$", re.I | re.M),
}

def _split_into_sections(text: str) -> Dict[str, str]:
    """
    Split the raw LinkedIn profile text into labelled sections by detecting
    common section headers. Returns a dict of {section_name: section_text}.
    """
    # Find all section header positions
    positions: List[Tuple[int, str]] = []
    for section, pattern in _SECTION_PATTERNS.items():
        for m in pattern.finditer(text):
            positions.append((m.start(), section, m.end()))

    if not positions:
        return {"raw": text}

    positions.sort(key=lambda x: x[0])

    sections: Dict[str, str] = {}
    # Text before first section header is the "header" (name / headline)
    first_pos = positions[0][0]
    sections["header"] = text[:first_pos].strip()

    for i, (start, section, end) in enumerate(positions):
        next_start = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[end:next_start].strip()
        # Allow multiple occurrences (keep last)
        sections[section] = content

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
#  Field extractors
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_name_headline(header_text: str) -> Tuple[str, str]:
    """
    First line → name, second (if not a date/email) → headline.
    """
    lines = [l.strip() for l in header_text.splitlines() if l.strip()]
    name     = lines[0] if lines else "Unknown"
    headline = ""
    if len(lines) > 1:
        second = lines[1]
        # Skip if it looks like a location or connection count line
        if not re.match(r"^\d+", second) and "@" not in second:
            headline = second
    return name, headline


def _extract_location(text: str) -> str:
    """
    Try to extract a location from the first few lines (City, Country / City, State).
    """
    loc_pattern = re.compile(
        r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*))\b"
    )
    match = loc_pattern.search(text[:500])
    return match.group(1) if match else ""


_TECH_SKILLS = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "r", "scala", "php", "ruby", "perl", "matlab",
    # Web
    "react", "angular", "vue", "node.js", "html", "css", "sass", "graphql",
    "rest", "fastapi", "flask", "django", "express", "spring boot",
    # Data / ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "huggingface",
    "pandas", "numpy", "matplotlib", "seaborn", "spark", "hadoop",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "tableau", "power bi", "excel",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "github actions", "ci/cd", "linux", "bash",
    # Project management / soft
    "agile", "scrum", "kanban", "jira", "confluence", "git",
    # Finance
    "financial analysis", "accounting", "tally", "sap", "bloomberg",
    "equity research", "portfolio management",
    # Healthcare
    "emr", "ehr", "hipaa", "clinical research", "icd-10", "nursing",
}

_DEGREE_WORDS = {
    "b.sc", "bsc", "b.s.", "bs", "b.tech", "btech", "be", "b.e.",
    "m.sc", "msc", "m.s.", "ms", "m.tech", "mtech", "me", "m.e.",
    "mba", "m.b.a.", "phd", "ph.d.", "md", "m.d.",
    "bachelor", "master", "doctorate", "associate", "diploma",
    "b.com", "m.com", "b.a.", "ma", "llb", "llm",
}

_DURATION_PATTERN = re.compile(
    r"""
    (?:
        (?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|
           jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)
        [\s,]*\d{4}
    |
        \d{4}
    )
    \s*[-–—to]+\s*
    (?:
        (?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|
           jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)
        [\s,]*\d{4}
        |
        \d{4}
        |
        present|current|now
    )
    """,
    re.I | re.VERBOSE,
)


def extract_linkedin_skills(text: str) -> List[str]:
    """
    Extract skills from the skills section text.
    Strategy:
      1. Line-by-line matching against TECH_SKILLS dict
      2. spaCy noun-chunk extraction as supplementary
    """
    found = set()
    text_lower = text.lower()

    for skill in _TECH_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            # Preserve original casing as best we can
            m = re.search(pattern, text_lower)
            start = m.start()
            found.add(text[start: start + len(skill)])

    # spaCy noun-chunks for custom skills in the skills blob
    nlp = _get_nlp()
    if nlp:
        # Limit to 2000 chars for speed
        doc = nlp(text[:2000])
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip().lower()
            if 2 <= len(chunk_text) <= 40 and chunk_text not in {"i", "we", "you"}:
                # Only keep if it looks like a skill (not a full sentence)
                if len(chunk_text.split()) <= 4:
                    found.add(chunk.text.strip())

    # Also split comma-separated skills lists
    for line in text.splitlines():
        line = line.strip()
        if "," in line and len(line) < 200:
            parts = [p.strip() for p in line.split(",")]
            for p in parts:
                if 2 <= len(p) <= 40:
                    found.add(p)

    return sorted(found, key=lambda x: x.lower())


def extract_linkedin_experience(text: str) -> List[Dict[str, str]]:
    """
    Parse the experience section into a list of job entries.

    Expected format per entry (as LinkedIn exports / copy-paste produce):
        Job Title
        Company Name
        Duration (e.g. Jan 2020 – Present)
        Description lines...
    """
    entries = []
    if not text.strip():
        return entries

    # Split into blocks separated by blank lines
    blocks = re.split(r"\n{2,}", text.strip())

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue

        entry: Dict[str, str] = {
            "title":       "",
            "company":     "",
            "duration":    "",
            "description": "",
        }

        description_start = 0

        # First line → title (usually all-caps or title-case)
        entry["title"] = lines[0]

        # Find duration line (contains year pattern)
        for i, line in enumerate(lines):
            dur_m = _DURATION_PATTERN.search(line)
            if dur_m:
                entry["duration"] = line.strip()
                # Company is the line before the duration
                if i > 1:
                    entry["company"] = lines[i - 1]
                elif i == 1:
                    entry["company"] = lines[1]
                description_start = i + 1
                break
        else:
            # No duration found — use second line as company
            if len(lines) > 1:
                entry["company"] = lines[1]
            description_start = 2

        # Remaining lines → description
        entry["description"] = " ".join(lines[description_start:]).strip()

        if entry["title"] or entry["company"]:
            entries.append(entry)

    return entries


def extract_linkedin_education(text: str) -> List[Dict[str, str]]:
    """
    Parse the education section into degree entries.

    Expected format:
        University / College Name
        Degree, Field of Study
        Start Year – End Year
    """
    entries = []
    if not text.strip():
        return entries

    blocks = re.split(r"\n{2,}", text.strip())

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        entry: Dict[str, str] = {
            "institution": "",
            "degree":      "",
            "field":       "",
            "year":        "",
        }

        entry["institution"] = lines[0]

        # Look for degree keyword
        for line in lines[1:]:
            line_lower = line.lower()
            for kw in _DEGREE_WORDS:
                if kw in line_lower:
                    entry["degree"] = line
                    break

            # Year / duration
            dur_m = re.search(r"\b\d{4}\b", line)
            if dur_m and not entry["year"]:
                entry["year"] = dur_m.group(0)

        if entry["institution"]:
            entries.append(entry)

    return entries


def _extract_certifications(text: str) -> List[str]:
    """Extract certification names, one per line."""
    certs = []
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) > 5:
            certs.append(line)
    return certs[:20]  # Limit to 20


def _extract_languages(text: str) -> List[str]:
    """Parse language proficiency lines."""
    langs = []
    common = {
        "english", "hindi", "french", "german", "spanish", "mandarin",
        "arabic", "japanese", "portuguese", "russian", "italian",
        "dutch", "korean", "bengali", "tamil", "telugu",
    }
    for line in text.splitlines():
        line_lower = line.strip().lower()
        for lang in common:
            if lang in line_lower and lang not in [l.lower() for l in langs]:
                langs.append(line.strip().split()[0].capitalize())
    return langs


def _profile_strength(data: Dict[str, Any]) -> str:
    """Return a simple profile strength rating."""
    score = 0
    if data.get("name") and data["name"] != "Unknown":   score += 1
    if data.get("headline"):                               score += 1
    if data.get("summary"):                               score += 1
    if len(data.get("skills", [])) >= 5:                   score += 2
    if len(data.get("experience", [])) >= 1:               score += 2
    if len(data.get("education", [])) >= 1:                score += 1
    if len(data.get("certifications", [])) >= 1:           score += 1

    if score >= 7:  return "strong"
    if score >= 4:  return "intermediate"
    return "basic"


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def parse_linkedin_profile(text: str) -> Dict[str, Any]:
    """
    Parse pasted LinkedIn profile text into a structured JSON dict.

    Args:
        text: Raw copy-pasted LinkedIn profile text (plain text).

    Returns:
        Structured dict with keys: name, headline, summary, location,
        skills, experience, education, certifications, languages,
        profile_strength.
        Returns a dict with an "error" key if text is empty/too short.
    """
    if not text or len(text.strip()) < 30:
        return {"error": "Insufficient profile text. Please paste more content."}

    logger.info(f"[LinkedIn] Parsing profile text ({len(text)} chars).")
    sections = _split_into_sections(text)

    # ── Header extraction ──────────────────────────────────────────────────────
    header_text = sections.get("header", text[:300])
    name, headline = _extract_name_headline(header_text)
    location       = _extract_location(header_text + "\n" + text[:500])

    # ── Summary ────────────────────────────────────────────────────────────────
    summary_raw = sections.get("summary", "")
    summary     = " ".join(summary_raw.splitlines()).strip()[:600]

    # ── Skills ────────────────────────────────────────────────────────────────
    skills_text = sections.get("skills", "")
    # Also scan full text for skills if section is empty
    if not skills_text:
        skills_text = text
    skills = extract_linkedin_skills(skills_text)

    # ── Experience ────────────────────────────────────────────────────────────
    exp_text   = sections.get("experience", "")
    experience = extract_linkedin_experience(exp_text) if exp_text else []

    # ── Education ─────────────────────────────────────────────────────────────
    edu_text  = sections.get("education", "")
    education = extract_linkedin_education(edu_text) if edu_text else []

    # ── Certifications ────────────────────────────────────────────────────────
    cert_text      = sections.get("certifications", "")
    certifications = _extract_certifications(cert_text)

    # ── Languages ─────────────────────────────────────────────────────────────
    lang_text = sections.get("languages", text)
    languages = _extract_languages(lang_text)

    result = {
        "name":             name,
        "headline":         headline,
        "summary":          summary,
        "location":         location,
        "skills":           skills,
        "experience":       experience,
        "education":        education,
        "certifications":   certifications,
        "languages":        languages,
    }
    result["profile_strength"] = _profile_strength(result)

    logger.info(
        f"[LinkedIn] Parsed: {len(skills)} skills, "
        f"{len(experience)} jobs, {len(education)} edu entries."
    )
    return result
