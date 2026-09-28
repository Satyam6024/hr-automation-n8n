from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

from scraper import scrape_company


load_dotenv()


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROFILE_FILE = BASE_DIR / "candidate_profile.txt"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# Current stable Gemini Flash models.
# 3.8 is preferred; lower models are fallbacks for temporary
# capacity/rate-limit problems.
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

MAX_COMPANY_TEXT = 60000
MAX_RETRIES_PER_MODEL = 3


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="HR Company Research API",
    version="1.0.0",
)


# ============================================================
# Request schema
# ============================================================

class AnalyzeRequest(BaseModel):
    email: str
    hr_name: str | None = None


# ============================================================
# Email / name helpers
# ============================================================

def extract_domain(email: str) -> str:
    """
    Extract company domain from an email address.

    Example:
        mownikareddy.a@primefocus.com
        -> primefocus.com
    """

    email = email.strip().lower()

    if "@" not in email:
        raise ValueError("Invalid email address.")

    local_part, domain = email.rsplit("@", 1)

    if not local_part or not domain:
        raise ValueError("Invalid email address.")

    return domain


def get_first_name(name: str | None) -> str:
    """
    Normalize a recruiter name to first name only.

    Examples:
        "Mownika Reddy A" -> "Mownika"
        "Mownika Reddy"   -> "Mownika"
        "Mownika"         -> "Mownika"
        "Unknown"         -> "Unknown"
    """

    if not name:
        return "Unknown"

    name = str(name).strip()

    if not name:
        return "Unknown"

    if name.lower() in {
        "unknown",
        "not supplied",
        "not available",
        "n/a",
        "na",
        "none",
    }:
        return "Unknown"

    # Split on whitespace and common name separators.
    parts = [
        part
        for part in re.split(r"[\s._\-+]+", name)
        if part
    ]

    if not parts:
        return "Unknown"

    first_name = parts[0].strip()

    # Keep alphabetic characters, allowing apostrophes.
    first_name = re.sub(
        r"[^A-Za-zÀ-ÖØ-öø-ÿ']",
        "",
        first_name,
    )

    if not first_name:
        return "Unknown"

    return first_name.capitalize()


def infer_first_name_with_gemini(
    email: str,
    company_text: str,
) -> str:
    """
    Try to infer a recruiter's first name from the email address
    and company website content.

    This is an inference, not verified identity information.
    Generic addresses return Unknown.
    """

    local_part = email.split("@", 1)[0].strip().lower()

    generic_prefixes = {
        "hr",
        "careers",
        "career",
        "jobs",
        "job",
        "recruiter",
        "recruitment",
        "talent",
        "talentacquisition",
        "info",
        "contact",
        "admin",
        "support",
        "hello",
    }

    if local_part in generic_prefixes:
        return "Unknown"

    prompt = f"""
Extract the most likely FIRST NAME of the person associated
with this recruiter email address.

Email:
{email}

Email local part:
{local_part}

Company website text:
{company_text[:15000]}

Rules:
- Return ONLY the first name.
- Do not return surname.
- Do not return initials.
- Do not return email address.
- Do not explain your answer.
- If the address is a generic mailbox such as hr@, careers@,
  jobs@, recruitment@, info@, return Unknown.
- If the person's first name can be reasonably inferred from
  the email address or supplied company content, return it.
- For example:
  mownikareddy.a@primefocus.com -> Mownika
  rahul.sharma@company.com -> Rahul
  hr@company.com -> Unknown
"""

    try:
        response = client.models.generate_content(
            model=MODELS[0],
            contents=prompt,
        )

        name = response.text.strip()

        # Remove accidental markdown/code formatting.
        name = re.sub(r"`+", "", name).strip()

        # Keep only the first token.
        name = name.split()[0] if name else ""

        return get_first_name(name)

    except Exception as exc:
        print(
            f"[Name Inference] Failed for {email}: {exc}"
        )
        return "Unknown"


# ============================================================
# File helpers
# ============================================================

def load_candidate_profile() -> str:
    """
    Load the candidate profile from candidate_profile.txt.
    """

    if not PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"Candidate profile not found: {PROFILE_FILE}"
        )

    profile = PROFILE_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not profile:
        raise ValueError(
            "candidate_profile.txt is empty."
        )

    return profile


# ============================================================
# Gemini response helpers
# ============================================================

def clean_json_response(text: str) -> dict[str, Any]:
    """
    Clean and parse Gemini's JSON response.

    Handles accidental markdown code fences.
    """

    if not text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    text = text.strip()

    # Remove accidental ```json ... ``` wrapping.
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned invalid JSON: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            "Gemini response must be a JSON object."
        )

    return result


def is_retryable_error(exc: Exception) -> bool:
    """
    Identify temporary Gemini/API failures.

    Retryable:
        408
        429
        500
        502
        503
        504
        RESOURCE_EXHAUSTED
        UNAVAILABLE
        TIMEOUT
    """

    error_text = str(exc).upper()

    retryable_patterns = [
        "408",
        "429",
        "500",
        "502",
        "503",
        "504",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "SERVICE_UNAVAILABLE",
        "TOO_MANY_REQUESTS",
        "HIGH DEMAND",
        "TIMEOUT",
        "TIMED OUT",
    ]

    return any(
        pattern in error_text
        for pattern in retryable_patterns
    )


def validate_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize Gemini output so n8n receives predictable fields.
    """

    defaults: dict[str, Any] = {
        "company_name": "",
        "company_summary": "",
        "industry": "",
        "ai_ml_relevance": "",
        "relevant_technologies": [],
        "potential_roles": [],
        "fit_score": 0,
        "fit_reason": "",
        "hr_name": "Unknown",
        "email_subject": "",
        "email_body": "",
        "source_urls": [],
    }

    normalized: dict[str, Any] = {}

    for field, default_value in defaults.items():
        value = result.get(
            field,
            default_value,
        )

        if field in {
            "relevant_technologies",
            "potential_roles",
            "source_urls",
        }:

            if value is None:
                value = []

            if not isinstance(value, list):
                value = [str(value)]

        elif field == "fit_score":

            try:
                value = int(float(value))
            except (TypeError, ValueError):
                value = 0

            value = max(0, min(100, value))

        elif field == "hr_name":

            value = get_first_name(
                str(value) if value is not None else None
            )

        else:

            if value is None:
                value = ""

            value = str(value).strip()

        normalized[field] = value

    return normalized


# ============================================================
# Gemini prompt
# ============================================================

def build_prompt(
    *,
    email: str,
    hr_name: str | None,
    company_text: str,
    source_urls: list[str],
    candidate_profile: str,
) -> str:

    supplied_name = (
        get_first_name(hr_name)
        if hr_name
        else "Not supplied"
    )

    return f"""
You are a recruitment research and personalization engine.

You are given:
1. A recruiter email address.
2. Public company website content.
3. A candidate profile.

Your task is to research the company and compare the
candidate's documented skills against the company's publicly
described work.

============================================================
CANDIDATE PROFILE
============================================================

{candidate_profile}

============================================================
RECRUITER INFORMATION
============================================================

Recruiter email:
{email}

Recruiter name supplied by user:
{supplied_name}

IMPORTANT NAME RULE:
- The JSON field "hr_name" must contain ONLY the recruiter's
  first name.
- Example: "Mownika Reddy A" -> "Mownika"
- Example: "Rahul Sharma" -> "Rahul"
- Never include surname or initials.
- If the recruiter's name cannot be determined confidently,
  return "Unknown".
- Do not infer a personal name for generic addresses such as
  hr@, careers@, jobs@, recruitment@, recruiter@, info@, etc.

============================================================
COMPANY SOURCE
============================================================

Source URLs:
{json.dumps(source_urls, ensure_ascii=False)}

Company website content:

{company_text}

============================================================
COMPANY ANALYSIS
============================================================

Determine:

1. Company name.
2. What the company does.
3. Primary industry.
4. AI/ML/GenAI/Python relevance.
5. Relevant technologies or technical areas.
6. Plausible role categories based on the company's actual work.
7. Candidate-company fit score.
8. Explanation of the fit.
9. Personalized outreach email.

============================================================
FACTUALITY RULES
============================================================

Use only information supported by the supplied company content.

Never invent:

- Products
- Services
- Clients
- Technologies
- Partnerships
- Offices
- Company achievements
- Job openings
- Recruiter responsibilities
- Candidate experience

Do not claim that a company is hiring unless the supplied
source explicitly confirms it.

Do not claim that the candidate used a technology unless it
exists in the candidate profile.

If the company information is insufficient, state that clearly.

============================================================
FIT SCORE
============================================================

Calculate 0-100 using:

AI/ML relevance       = 30 points
Python/backend        = 15 points
GenAI/LLM/RAG         = 20 points
Cloud/MLOps           = 15 points
Role relevance        = 10 points
Experience alignment  = 10 points

Base the score on actual overlap between company information
and candidate profile.

Do not inflate the score because the company is famous.

============================================================
EMAIL RULES
============================================================

Write a professional, concise, personalized outreach email.

Requirements:

- Approximately 100-140 words.
- Use the recruiter's first name when confidently known.
- Otherwise use "Hello" or "Hi there".
- Mention one specific factual reason the company is relevant.
- Connect it to relevant candidate experience.
- Mention the candidate's relevant AI/ML/Python background.
- Mention the resume.
- Ask whether the profile could be considered for a relevant
  opportunity.
- Ask for a 5-10 minute call to walk through relevant projects.
- Never claim that a job is open unless the source confirms it.
- Never fabricate previous contact.
- Never use:
  "I hope this email finds you well."
- Don't excessively praise the company.
- Don't sound like a mass-generated email.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

No markdown.
No code fences.
No commentary.

Use exactly this structure:

{{
  "company_name": "",
  "company_summary": "",
  "industry": "",
  "ai_ml_relevance": "",
  "relevant_technologies": [],
  "potential_roles": [],
  "fit_score": 0,
  "fit_reason": "",
  "hr_name": "",
  "email_subject": "",
  "email_body": "",
  "source_urls": []
}}
"""


# ============================================================
# Gemini call
# ============================================================

def analyze_with_gemini(
    *,
    email: str,
    hr_name: str | None,
    company_text: str,
    source_urls: list[str],
) -> dict[str, Any]:

    candidate_profile = load_candidate_profile()

    company_text = company_text[:MAX_COMPANY_TEXT]

    prompt = build_prompt(
        email=email,
        hr_name=hr_name,
        company_text=company_text,
        source_urls=source_urls,
        candidate_profile=candidate_profile,
    )

    last_error: Exception | None = None

    for model in MODELS:

        for attempt in range(
            1,
            MAX_RETRIES_PER_MODEL + 1,
        ):

            try:

                print(
                    f"[Gemini] "
                    f"model={model} "
                    f"attempt={attempt}/"
                    f"{MAX_RETRIES_PER_MODEL}"
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                result = clean_json_response(
                    response.text
                )

                result = validate_result(result)

                print(
                    f"[Gemini] SUCCESS "
                    f"model={model}"
                )

                return result

            except Exception as exc:

                last_error = exc

                print(
                    f"[Gemini] ERROR "
                    f"model={model} "
                    f"attempt={attempt}: {exc}"
                )

                if not is_retryable_error(exc):
                    raise

                if attempt < MAX_RETRIES_PER_MODEL:

                    wait_seconds = 2 ** (
                        attempt - 1
                    )

                    print(
                        f"[Gemini] Retrying in "
                        f"{wait_seconds}s..."
                    )

                    time.sleep(
                        wait_seconds
                    )

        print(
            f"[Gemini] Model {model} unavailable. "
            f"Trying fallback."
        )

    raise RuntimeError(
        "All Gemini models failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# API routes
# ============================================================

@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "HR Company Research API is running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
    }


@app.post("/analyze")
def analyze(
    request: AnalyzeRequest,
) -> dict[str, Any]:

    try:

        # ----------------------------------------------------
        # 1. Clean input
        # ----------------------------------------------------

        email = request.email.strip()

        if not email:
            raise ValueError(
                "Email address cannot be empty."
            )

        # Recruiter name is optional.
        # n8n currently sends only the email.
        supplied_hr_name = (
            request.hr_name.strip()
            if request.hr_name
            else None
        )

        print(
            f"\n[Analyze] Email: {email}"
        )

        # ----------------------------------------------------
        # 2. Extract company domain
        # ----------------------------------------------------

        domain = extract_domain(email)

        print(
            f"[Analyze] Domain: {domain}"
        )

        # ----------------------------------------------------
        # 3. Scrape company website
        # ----------------------------------------------------

        company = scrape_company(domain)

        if company.get("status") != "SUCCESS":

            print(
                f"[Scraper] FAILED: {domain}"
            )

            return {
                "email": email,
                "hr_name": (
                    get_first_name(supplied_hr_name)
                    if supplied_hr_name
                    else "Unknown"
                ),
                "company": domain,
                "website": company.get(
                    "website",
                    f"https://{domain}",
                ),
                "scrape_status": company.get(
                    "status",
                    "FAILED",
                ),
                "company_name": "",
                "company_summary": "",
                "industry": "",
                "ai_ml_relevance": "",
                "relevant_technologies": [],
                "potential_roles": [],
                "fit_score": 0,
                "fit_reason": (
                    "Company website could not be "
                    "retrieved or parsed."
                ),
                "email_subject": "",
                "email_body": "",
                "source_urls": company.get(
                    "pages",
                    [],
                ),
                "error": (
                    "No usable company website "
                    "content was found."
                ),
            }

        # ----------------------------------------------------
        # 4. Validate scraped text
        # ----------------------------------------------------

        pages = company.get(
            "pages",
            [],
        )

        company_text = company.get(
            "text",
            "",
        )

        if not company_text.strip():

            print(
                f"[Scraper] PARTIAL: {domain}"
            )

            return {
                "email": email,
                "hr_name": (
                    get_first_name(supplied_hr_name)
                    if supplied_hr_name
                    else "Unknown"
                ),
                "company": domain,
                "website": company.get(
                    "website",
                    f"https://{domain}",
                ),
                "scrape_status": "PARTIAL",
                "company_name": "",
                "company_summary": "",
                "industry": "",
                "ai_ml_relevance": "",
                "relevant_technologies": [],
                "potential_roles": [],
                "fit_score": 0,
                "fit_reason": (
                    "Website was reachable but "
                    "no usable text was extracted."
                ),
                "email_subject": "",
                "email_body": "",
                "source_urls": pages,
                "error": (
                    "No usable text extracted "
                    "from company website."
                ),
            }

        print(
            f"[Scraper] SUCCESS: {domain}"
        )

        # ----------------------------------------------------
        # 5. Gemini analysis
        # ----------------------------------------------------

        analysis = analyze_with_gemini(
            email=email,
            hr_name=supplied_hr_name,
            company_text=company_text,
            source_urls=pages,
        )

        # ----------------------------------------------------
        # 6. Normalize recruiter first name
        # ----------------------------------------------------

        first_name = get_first_name(
        analysis.get("hr_name")
        )

        # Fallback name inference.
        if first_name == "Unknown":
            first_name = infer_first_name_with_gemini(
                email=email,
                company_text=company_text,
            )

        # ----------------------------------------------------
        # 7. Final response for n8n
        # ----------------------------------------------------

        result = {
            "email": email,
            "hr_name": first_name,
            "company": analysis.get(
                "company_name",
                domain,
            ),
            "website": company.get(
                "website",
                f"https://{domain}",
            ),
            "scrape_status": "SUCCESS",
            "company_name": analysis.get(
                "company_name",
                "",
            ),
            "company_summary": analysis.get(
                "company_summary",
                "",
            ),
            "industry": analysis.get(
                "industry",
                "",
            ),
            "ai_ml_relevance": analysis.get(
                "ai_ml_relevance",
                "",
            ),
            "relevant_technologies": analysis.get(
                "relevant_technologies",
                [],
            ),
            "potential_roles": analysis.get(
                "potential_roles",
                [],
            ),
            "fit_score": analysis.get(
                "fit_score",
                0,
            ),
            "fit_reason": analysis.get(
                "fit_reason",
                "",
            ),
            "email_subject": analysis.get(
                "email_subject",
                "",
            ),
            "email_body": analysis.get(
                "email_body",
                "",
            ),
            "source_urls": pages,
        }

        print(
            f"[Analyze] Complete: "
            f"{result['company_name']} | "
            f"HR: {result['hr_name']} | "
            f"Fit Score: {result['fit_score']}"
        )

        return result

    except ValueError as exc:

        print(
            f"[Validation Error] {exc}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        print(
            f"[File Error] {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print(
            f"[Server Error] {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc