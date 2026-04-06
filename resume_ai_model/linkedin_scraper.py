from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup  # type: ignore
from selenium.webdriver.remote.webdriver import WebDriver  # type: ignore
from selenium.webdriver.common.by import By  # type: ignore
from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
from selenium.webdriver.support import expected_conditions as EC  # type: ignore


@dataclass
class LinkedInProfile:
    name: Optional[str] = None
    headline: Optional[str] = None
    experiences: List[Dict[str, Any]] = None
    educations: List[Dict[str, Any]] = None
    skills: List[str] = None
    projects: List[str] = None
    full_profile_text: Optional[str] = None

    def to_resume_json(self) -> Dict[str, Any]:
        """Map scraped profile into the resume JSON format expected by downstream models."""
        return {
            "name": self.name or "",
            "email": "",  # LinkedIn rarely exposes email; keep empty
            "skills": self.skills or [],
            "experience": self.experiences or [],
            "education": self.educations or [],
            "projects": self.projects or [],
            "full_resume_text": (self.full_profile_text or "").strip(),
        }


def _scroll_page(driver: WebDriver, pauses: int = 3, delay: float = 0.8) -> None:
    """Scroll the page a few times to trigger dynamic loading."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(pauses):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def _wait_for_profile_header(driver: WebDriver, timeout: int = 15) -> None:
    """Wait until the main profile header is present (best-effort)."""
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located(
            (
                By.CSS_SELECTOR,
                "section.artdeco-card, div.ph5, main.scaffold-layout__main",
            )
        )
    )


def _extract_text_or_none(element) -> Optional[str]:
    if not element:
        return None
    text = element.get_text(separator=" ", strip=True)
    return text or None


def _parse_profile_html(html: str) -> LinkedInProfile:
    """Parse LinkedIn profile HTML using robust, class-insensitive heuristics."""
    soup = BeautifulSoup(html, "html.parser")

    # Name and headline (top card)
    name = None
    headline = None

    # LinkedIn changes classes frequently, so use structural hints
    top_card = soup.find("section", {"class": lambda c: c and "top-card-layout" in c}) or soup.find(
        "div", {"class": lambda c: c and "pv-text-details__left-panel" in c}
    )
    if top_card:
        h1 = top_card.find("h1")
        name = _extract_text_or_none(h1)
        headline_el = top_card.find("div", {"class": lambda c: c and "headline" in c}) or top_card.find(
            "div", {"class": lambda c: c and "text-body-medium" in c}
        )
        headline = _extract_text_or_none(headline_el)

    # Experience section
    experiences: List[Dict[str, Any]] = []
    exp_section = soup.find("section", {"id": lambda x: x and "experience" in x}) or soup.find(
        "section", string=lambda t: t and "Experience" in t
    )
    if exp_section:
        for li in exp_section.find_all("li"):
            role = company = duration = None
            title_el = li.find("div", {"class": lambda c: c and "t-bold" in c}) or li.find("span", {"class": "mr1"})
            company_el = li.find("span", {"class": lambda c: c and "t-normal" in c})
            duration_el = li.find("span", string=lambda t: t and ("Present" in t or "–" in t or "-" in t))

            role = _extract_text_or_none(title_el)
            company = _extract_text_or_none(company_el)
            duration = _extract_text_or_none(duration_el)

            if role or company or duration:
                experiences.append(
                    {
                        "role": role or "",
                        "company": company or "",
                        "duration": duration or "",
                    }
                )

    # Education section
    educations: List[Dict[str, Any]] = []
    edu_section = soup.find("section", {"id": lambda x: x and "education" in x}) or soup.find(
        "section", string=lambda t: t and "Education" in t
    )
    if edu_section:
        for li in edu_section.find_all("li"):
            degree = university = None
            uni_el = li.find("span", {"class": lambda c: c and "t-bold" in c})
            degree_el = li.find("span", {"class": lambda c: c and "t-normal" in c})
            university = _extract_text_or_none(uni_el)
            degree = _extract_text_or_none(degree_el)
            if degree or university:
                educations.append(
                    {
                        "degree": degree or "",
                        "university": university or "",
                    }
                )

    # Skills
    skills: List[str] = []
    skills_section = soup.find("section", {"id": lambda x: x and "skills" in x}) or soup.find(
        "section", string=lambda t: t and "Skills" in t
    )
    if skills_section:
        for span in skills_section.find_all("span"):
            label = _extract_text_or_none(span)
            if label and len(label) < 64:  # avoid long sentences
                skills.append(label)
    # Deduplicate skills while preserving order
    seen = set()
    deduped_skills = []
    for s in skills:
        if s.lower() not in seen:
            seen.add(s.lower())
            deduped_skills.append(s)

    # Projects (often under "Projects" / "Featured" / "Accomplishments")
    projects: List[str] = []
    for section_label in ["projects", "accomplishments", "featured"]:
        sec = soup.find("section", {"id": lambda x: x and section_label in x})
        if sec:
            for li in sec.find_all("li"):
                title = _extract_text_or_none(li)
                if title and len(title) < 150:
                    projects.append(title)

    # Entire visible profile text for downstream ML
    full_text = soup.get_text(separator=" ", strip=True)

    return LinkedInProfile(
        name=name,
        headline=headline,
        experiences=experiences,
        educations=educations,
        skills=deduped_skills,
        projects=projects,
        full_profile_text=full_text,
    )


def scrape_linkedin_profile(url: str, driver: WebDriver, timeout: int = 20) -> LinkedInProfile:
    """
    Scrape a LinkedIn public profile using Selenium.

    Notes:
    - Assumes the caller has already configured & authenticated the driver if needed.
    - Handles dynamic loading by scrolling and waiting for main sections to render.
    - Returns a LinkedInProfile dataclass that can be converted to the resume JSON format.
    """
    driver.get(url)
    _wait_for_profile_header(driver, timeout=timeout)
    _scroll_page(driver, pauses=4, delay=0.8)

    html = driver.page_source
    profile = _parse_profile_html(html)
    return profile


def scrape_linkedin_profile_json(url: str, driver: WebDriver, timeout: int = 20) -> Dict[str, Any]:
    """Helper that directly returns the downstream JSON-compatible resume structure."""
    profile = scrape_linkedin_profile(url, driver=driver, timeout=timeout)
    return profile.to_resume_json()

