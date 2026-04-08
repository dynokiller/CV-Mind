"""
tests/test_linkedin_analyzer.py

Unit tests for the LinkedIn profile analyzer.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.linkedin_analyzer import (
    parse_linkedin_profile,
    extract_linkedin_skills,
    extract_linkedin_experience,
    extract_linkedin_education,
    _split_into_sections,
)

# ── Sample profile text ────────────────────────────────────────────────────────
SAMPLE_PROFILE = """
John Doe
Senior Software Engineer at TechCorp | Python | AWS | Machine Learning

Mumbai, India · 500+ connections

About
Experienced software engineer with 8 years of experience in building scalable
web applications and ML pipelines. Passionate about open-source and cloud technologies.

Experience

Senior Software Engineer
TechCorp
Jan 2021 – Present
Led backend development using Python, FastAPI, and AWS. Built ML pipeline serving 1M+ users.

Software Developer
StartupABC
Jun 2018 – Dec 2020
Developed REST APIs using Django and PostgreSQL. Implemented CI/CD pipelines with Jenkins.

Education

Indian Institute of Technology
B.Tech, Computer Science
2014 – 2018

Skills

Python
AWS
Machine Learning
Docker
Kubernetes
FastAPI
PostgreSQL
TensorFlow
Git

Languages

English
Hindi
"""

SAMPLE_MINIMAL = "John Smith\nSoftware developer with Python skills"


class TestSplitIntoSections:
    def test_detects_experience_section(self):
        sections = _split_into_sections(SAMPLE_PROFILE)
        assert "experience" in sections

    def test_detects_education_section(self):
        sections = _split_into_sections(SAMPLE_PROFILE)
        assert "education" in sections

    def test_detects_skills_section(self):
        sections = _split_into_sections(SAMPLE_PROFILE)
        assert "skills" in sections

    def test_header_extracted(self):
        sections = _split_into_sections(SAMPLE_PROFILE)
        assert "header" in sections or "raw" in sections


class TestExtractSkills:
    def test_returns_list(self):
        skills = extract_linkedin_skills("Python AWS Docker Machine Learning")
        assert isinstance(skills, list)

    def test_finds_known_tech_skills(self):
        skills = extract_linkedin_skills("I work with Python, AWS, and Docker.")
        skill_lower = [s.lower() for s in skills]
        assert any("python" in s for s in skill_lower)
        assert any("aws" in s for s in skill_lower)

    def test_empty_text(self):
        assert extract_linkedin_skills("") == []

    def test_deduplication(self):
        text    = "Python Python Python machine learning deep learning"
        skills  = extract_linkedin_skills(text)
        # Should not have massive duplicates
        assert len(skills) == len(set(s.lower() for s in skills))


class TestExtractExperience:
    def test_returns_list_of_dicts(self):
        exp = extract_linkedin_experience(
            "Senior Engineer\nTechCorp\nJan 2020 – Present\nBuilt scalable systems."
        )
        assert isinstance(exp, list)
        if exp:
            assert "title" in exp[0]
            assert "company" in exp[0]

    def test_empty_text(self):
        assert extract_linkedin_experience("") == []

    def test_detects_duration(self):
        text = "Software Developer\nABC Corp\n2019 – 2022\nDeveloped APIs."
        exp  = extract_linkedin_experience(text)
        if exp:
            assert exp[0].get("duration", "") != ""


class TestExtractEducation:
    def test_returns_list_of_dicts(self):
        edu = extract_linkedin_education(
            "MIT\nB.Tech, Computer Science\n2015 – 2019"
        )
        assert isinstance(edu, list)
        if edu:
            assert "institution" in edu[0]

    def test_empty_text(self):
        assert extract_linkedin_education("") == []


class TestParseLinkedInProfile:
    def test_full_profile_parse(self):
        result = parse_linkedin_profile(SAMPLE_PROFILE)
        assert isinstance(result, dict)
        assert "name" in result
        assert "skills" in result
        assert "experience" in result
        assert "education" in result

    def test_finds_skills_in_full_profile(self):
        result = parse_linkedin_profile(SAMPLE_PROFILE)
        assert len(result["skills"]) > 0

    def test_profile_strength_present(self):
        result = parse_linkedin_profile(SAMPLE_PROFILE)
        assert result["profile_strength"] in ("basic", "intermediate", "strong")

    def test_strong_profile_gets_high_strength(self):
        result = parse_linkedin_profile(SAMPLE_PROFILE)
        # A full profile should be intermediate or strong
        assert result["profile_strength"] in ("intermediate", "strong")

    def test_minimal_profile(self):
        result = parse_linkedin_profile(SAMPLE_MINIMAL)
        assert isinstance(result, dict)
        assert "name" in result

    def test_empty_text_returns_error(self):
        result = parse_linkedin_profile("")
        assert "error" in result

    def test_too_short_text_returns_error(self):
        result = parse_linkedin_profile("hi")
        assert "error" in result
