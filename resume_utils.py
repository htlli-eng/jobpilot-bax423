from __future__ import annotations

import re
from typing import BinaryIO


SKILL_VOCABULARY = [
    "Python",
    "SQL",
    "R",
    "Excel",
    "Tableau",
    "Power BI",
    "pandas",
    "NumPy",
    "scikit-learn",
    "PyTorch",
    "TensorFlow",
    "Spark",
    "PySpark",
    "Kafka",
    "AWS",
    "GCP",
    "Azure",
    "machine learning",
    "deep learning",
    "NLP",
    "computer vision",
    "data analysis",
    "dashboarding",
    "A/B testing",
    "regression",
    "optimization",
    "statistics",
    "forecasting",
    "ETL",
    "data engineering",
    "business intelligence",
]
RESUME_SECTION_HEADINGS = [
    "Profile",
    "Summary",
    "Objective",
    "Skills",
    "Technical Skills",
    "Work Experience",
    "Professional Experience",
    "Experience",
    "Education",
    "Projects",
    "Certifications",
]
SUMMARY_HEADINGS = ["Summary", "Profile", "Objective"]
SKILLS_HEADINGS = ["Skills", "Technical Skills"]
EXPERIENCE_HEADINGS = ["Experience", "Work Experience", "Professional Experience"]
PROJECT_EDUCATION_HEADINGS = ["Projects", "Education", "Certifications"]


def extract_text_from_pdf(uploaded_pdf: BinaryIO) -> str:
    """Extract plain text from an uploaded PDF file-like object."""
    from pypdf import PdfReader

    uploaded_pdf.seek(0)
    reader = PdfReader(uploaded_pdf)
    page_text: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            page_text.append(text.strip())

    return "\n\n".join(page_text)


def extract_skills(text: str) -> list[str]:
    """Extract known skills from text with simple keyword matching."""
    if not text:
        return []

    found_skills: list[str] = []
    for skill in SKILL_VOCABULARY:
        pattern = build_skill_pattern(skill)
        if re.search(pattern, text, flags=re.IGNORECASE):
            found_skills.append(skill)

    return found_skills


def extract_candidate_name(text: str) -> str:
    """Use the first non-empty resume line as a simple candidate name guess."""
    if not text:
        return "Unknown"

    for line in text.splitlines():
        clean_line = re.sub(r"\s+", " ", line).strip()
        if clean_line:
            return clean_line[:80]

    return "Unknown"


def format_resume_preview(text: str, max_chars: int = 1_500) -> str:
    """Clean extracted resume text and add readable section breaks for preview."""
    if not text:
        return ""

    preview = text.replace("\r\n", "\n").replace("\r", "\n")
    preview = re.sub(r"[ \t]+", " ", preview)
    preview = re.sub(r"\n\s*\n+", "\n\n", preview)

    for heading in RESUME_SECTION_HEADINGS:
        preview = re.sub(
            rf"(?<!\n)(\b{heading}\b\s*:?)",
            rf"\n\n\1",
            preview,
            flags=re.IGNORECASE,
        )

    lines = [line.strip() for line in preview.splitlines()]
    preview = "\n".join(line for line in lines if line)
    preview = re.sub(r"\n{3,}", "\n\n", preview).strip()

    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip() + "\n\n[Preview trimmed]"

    return preview


def create_resume_preview_sections(text: str) -> dict[str, str]:
    """Create short, readable resume preview sections from extracted text."""
    cleaned_text = format_resume_preview(text, max_chars=8_000)
    if not cleaned_text:
        return {}

    section_map = split_resume_sections(cleaned_text)
    sections: dict[str, str] = {}

    header = extract_header_preview(cleaned_text)
    if header:
        sections["Header / Candidate Info"] = header

    summary = first_section_text(section_map, SUMMARY_HEADINGS)
    if summary:
        sections["Summary"] = trim_words(summary, max_words=70)

    skills = first_section_text(section_map, SKILLS_HEADINGS)
    if skills:
        sections["Skills"] = trim_words(skills, max_words=80)

    experience = first_section_text(section_map, EXPERIENCE_HEADINGS)
    if experience:
        sections["Experience Preview"] = trim_words(experience, max_words=100)
    else:
        sections["Experience Preview"] = trim_words(cleaned_text, max_words=90)

    project_or_education = first_section_text(section_map, PROJECT_EDUCATION_HEADINGS)
    if project_or_education:
        sections["Projects / Education"] = trim_words(project_or_education, max_words=80)

    return sections


def split_resume_sections(text: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections: dict[str, list[str]] = {}
    current_heading = ""

    for line in lines:
        heading = normalize_section_heading(line)
        if heading:
            current_heading = heading
            sections.setdefault(current_heading, [])
            remainder = remove_heading_prefix(line, heading)
            if remainder:
                sections[current_heading].append(remainder)
            continue

        if current_heading:
            sections[current_heading].append(line)

    return {heading: "\n".join(values).strip() for heading, values in sections.items()}


def normalize_section_heading(line: str) -> str:
    normalized_line = re.sub(r"[^a-z ]+", "", line.lower()).strip()
    for heading in sorted(RESUME_SECTION_HEADINGS, key=len, reverse=True):
        normalized_heading = heading.lower()
        if normalized_line == normalized_heading:
            return heading
        if normalized_line.startswith(f"{normalized_heading} "):
            return heading
    return ""


def remove_heading_prefix(line: str, heading: str) -> str:
    return re.sub(
        rf"^\s*{re.escape(heading)}\s*:?\s*",
        "",
        line,
        flags=re.IGNORECASE,
    ).strip()


def first_section_text(section_map: dict[str, str], headings: list[str]) -> str:
    for heading in headings:
        text = section_map.get(heading, "")
        if text:
            return text
    return ""


def extract_header_preview(text: str, max_lines: int = 4) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_lines: list[str] = []

    for line in lines:
        if normalize_section_heading(line):
            break
        header_lines.append(line)
        if len(header_lines) >= max_lines:
            break

    return "\n".join(header_lines).strip()


def trim_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + "..."


def build_skill_pattern(skill: str) -> str:
    escaped = re.escape(skill)
    escaped = escaped.replace(r"\ ", r"\s+")
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
