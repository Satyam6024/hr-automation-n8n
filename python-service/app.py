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

from database import (
    begin_research,
    get_contact,
    save_research,
    mark_research_failed,
    claim_send,
    mark_sent,
    mark_send_failed,
)


# ============================================================
# Environment
# ============================================================

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

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# Gemini fallback chain.
# These are current stable Gemini Flash models.
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

MAX_COMPANY_TEXT = 60000
MAX_RETRIES_PER_MODEL = 3


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="HR Company Research API",
    version="2.0.0",
)


# ============================================================
# Request models
# ============================================================

class AnalyzeRequest(BaseModel):
    email: str
    hr_name: str | None = None


class SendRequest(BaseModel):
    email: str


class MarkSentRequest(BaseModel):
    email: str
    gmail_message_id: str | None = None


# ============================================================
# Email helpers
# ============================================================

def extract_domain(email: str) -> str:
    """
    Extract the company domain from an email address.

    Example:
        mownika@primefocus.com
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
    Normalize a full recruiter name to first name only.

    Examples:
        Mownika Reddy A -> Mownika
        Rahul Sharma    -> Rahul
        Mownika         -> Mownika
        Unknown         -> Unknown
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
        "null",
    }:
        return "Unknown"

    parts = [
        part
        for part in re.split(
            r"[\s._\-+]+",
            name,
        )
        if part
    ]

    if not parts:
        return "Unknown"

    first_name = parts[0].strip()

    first_name = re.sub(
        r"[^A-Za-zÀ-ÖØ-öø-ÿ']",
        "",
        first_name,
    )

    if not first_name:
        return "Unknown"

    return first_name.capitalize()


# ============================================================
# Candidate profile
# ============================================================

def load_candidate_profile() -> str:
    """
    Load candidate_profile.txt.
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
# Gemini helpers
# ============================================================

def clean_json_response(
    text: str,
) -> dict[str, Any]:
    """
    Parse Gemini's JSON response.

    Removes accidental markdown code fences.
    """

    if not text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    text = text.strip()

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


def is_retryable_error(
    exc: Exception,
) -> bool:
    """
    Identify temporary API failures.
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
    Normalize Gemini output into a predictable structure.
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

            except (
                TypeError,
                ValueError,
            ):
                value = 0

            value = max(
                0,
                min(100, value),
            )

        elif field == "hr_name":

            value = get_first_name(
                value
            )

        else:

            if value is None:
                value = ""

            value = str(value).strip()

        normalized[field] = value

    return normalized


# ============================================================
# Recruiter name inference
# ============================================================

def infer_first_name_with_gemini(
    email: str,
    company_text: str,
) -> str:
    """
    Infer the recruiter's first name when the main company
    analysis returns Unknown.

    Generic mailbox addresses return Unknown.
    """

    local_part = (
        email
        .split("@", 1)[0]
        .strip()
        .lower()
    )

    generic_addresses = {
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
        "office",
    }

    if local_part in generic_addresses:
        return "Unknown"

    prompt = f"""
Extract the most likely FIRST NAME of the recruiter
associated with this email address.

Email:
{email}

Email local part:
{local_part}

Relevant company website content:
{company_text[:15000]}

Rules:
- Return ONLY the first name.
- Never return the surname.
- Never return initials.
- Never return the email address.
- Never explain the answer.
- Generic addresses such as hr@, careers@, jobs@,
  recruitment@, info@ must return Unknown.
- If a first name can be reasonably inferred from the
  email address, return it.
- Example:
  mownikareddy.a@primefocus.com -> Mownika
  rahul.sharma@company.com -> Rahul
  hr@company.com -> Unknown
"""

    try:

        response = client.models.generate_content(
            model=MODELS[0],
            contents=prompt,
        )

        name = (
            response.text
            .strip()
        )

        name = re.sub(
            r"`+",
            "",
            name,
        ).strip()

        if not name:
            return "Unknown"

        return get_first_name(name)

    except Exception as exc:

        print(
            f"[Name Inference] "
            f"Failed for {email}: {exc}"
        )

        return "Unknown"


# ============================================================
# Prompt
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

Research the company and compare the company's actual
publicly described work against the candidate's documented
skills.

============================================================
CANDIDATE PROFILE
============================================================

{candidate_profile}

============================================================
RECRUITER
============================================================

Recruiter email:
{email}

Recruiter name supplied by user:
{supplied_name}

NAME RULES:
- The JSON field "hr_name" must contain ONLY the first name.
- "Mownika Reddy A" -> "Mownika"
- "Rahul Sharma" -> "Rahul"
- Never include surname.
- Never include initials.
- Generic addresses such as hr@, careers@, jobs@,
  recruitment@, recruiter@, info@ should return Unknown.
- If uncertain, return Unknown.

============================================================
COMPANY SOURCE
============================================================

Source URLs:
{json.dumps(source_urls, ensure_ascii=False)}

Company website content:

{company_text}

============================================================
TASK
============================================================

Determine:

1. Company name.
2. What the company does.
3. Primary industry.
4. AI/ML/GenAI/Python relevance.
5. Relevant technologies or technical areas.
6. Plausible role categories for this candidate.
7. Candidate-company fit score.
8. Evidence-based fit reason.
9. Personalized outreach email.

============================================================
FACTUALITY RULES
============================================================

Use ONLY information supported by the supplied company
content and candidate profile.

Never invent:

- Products
- Services
- Clients
- Technologies
- Partnerships
- Offices
- Achievements
- Job openings
- Recruiter responsibilities
- Candidate experience

Do not claim that the company is hiring unless the
supplied source explicitly confirms it.

Do not claim the candidate used a technology unless it
exists in the candidate profile.

If information is insufficient, say so.

============================================================
FIT SCORE
============================================================

Calculate a score from 0 to 100.

AI/ML relevance       = 30
Python/backend        = 15
GenAI/LLM/RAG        = 20
Cloud/MLOps          = 15
Role relevance        = 10
Experience alignment  = 10

Use actual evidence.

Do not inflate the score because the company is famous.

============================================================
EMAIL
============================================================

Generate a personalized professional outreach email.

Requirements:

- Approximately 100-140 words.
- Address the recruiter by first name when confidently known.
- Otherwise use "Hello".
- Mention one concrete company-specific reason.
- Connect it to relevant candidate experience.
- Mention relevant AI/ML/Python experience.
- Mention the resume.
- Ask whether the profile could be considered for a relevant
  opportunity.
- Ask for a 5-10 minute call to walk through relevant projects.
- Never claim a job is open without evidence.
- Never fabricate previous communication.
- Never use:
  "I hope this email finds you well."
- Avoid excessive praise.
- Do not sound like a mass email.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Do not use markdown.
Do not use code fences.
Do not add commentary.

Use exactly:

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
# Gemini company analysis
# ============================================================

def analyze_with_gemini(
    *,
    email: str,
    hr_name: str | None,
    company_text: str,
    source_urls: list[str],
) -> dict[str, Any]:

    candidate_profile = (
        load_candidate_profile()
    )

    company_text = company_text[
        :MAX_COMPANY_TEXT
    ]

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

                response = (
                    client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                )

                result = clean_json_response(
                    response.text
                )

                result = validate_result(
                    result
                )

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
                    f"attempt={attempt}: "
                    f"{exc}"
                )

                if not is_retryable_error(
                    exc
                ):
                    raise

                if attempt < MAX_RETRIES_PER_MODEL:

                    wait_seconds = 2 ** (
                        attempt - 1
                    )

                    print(
                        f"[Gemini] "
                        f"Retrying in "
                        f"{wait_seconds}s..."
                    )

                    time.sleep(
                        wait_seconds
                    )

        print(
            f"[Gemini] {model} unavailable. "
            f"Trying next fallback model."
        )

    raise RuntimeError(
        "All Gemini models failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# Cached response helper
# ============================================================

def build_cached_response(
    record: dict[str, Any],
    processing_status: str,
) -> dict[str, Any]:
    """
    Convert a database record into the same response shape
    used by new research.
    """

    return {
        "email": record.get(
            "email",
            "",
        ),

        "hr_name": record.get(
            "hr_name",
            "Unknown",
        ),

        "company": record.get(
            "company_name",
            "",
        ),

        "website": record.get(
            "website",
            "",
        ),

        "scrape_status": record.get(
            "scrape_status",
            "",
        ),

        "company_name": record.get(
            "company_name",
            "",
        ),

        "company_summary": record.get(
            "company_summary",
            "",
        ),

        "industry": record.get(
            "industry",
            "",
        ),

        "ai_ml_relevance": record.get(
            "ai_ml_relevance",
            "",
        ),

        "relevant_technologies": record.get(
            "relevant_technologies",
            [],
        ),

        "potential_roles": record.get(
            "potential_roles",
            [],
        ),

        "fit_score": record.get(
            "fit_score",
            0,
        ),

        "fit_reason": record.get(
            "fit_reason",
            "",
        ),

        "email_subject": record.get(
            "email_subject",
            "",
        ),

        "email_body": record.get(
            "email_body",
            "",
        ),

        "source_urls": record.get(
            "source_urls",
            [],
        ),

        "research_status": record.get(
            "research_status",
            "",
        ),

        "research_error": record.get(
            "research_error",
            "",
        ),

        "email_status": record.get(
            "email_status",
            "NOT_SENT",
        ),

        "sent_at": record.get(
            "sent_at",
        ),

        "gmail_message_id": record.get(
            "gmail_message_id",
        ),

        "processing_status": processing_status,

        "from_cache": True,
    }


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": (
            "HR Company Research API "
            "is running"
        ),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
    }


# ============================================================
# Company research / deduplication
# ============================================================

@app.post("/analyze")
def analyze(
    request: AnalyzeRequest,
) -> dict[str, Any]:

    email = ""

    try:

        # ----------------------------------------------------
        # 1. Normalize email
        # ----------------------------------------------------

        email = request.email.strip().lower()

        if not email:
            raise ValueError(
                "Email address cannot be empty."
            )

        supplied_hr_name = (
            request.hr_name.strip()
            if request.hr_name
            else None
        )

        print(
            f"\n[Analyze] Email: {email}"
        )

        # ----------------------------------------------------
        # 2. Persistent deduplication
        # ----------------------------------------------------

        research_state = begin_research(
            email
        )

        print(
            f"[Database] "
            f"{email} -> "
            f"{research_state}"
        )

        # ----------------------------------------------------
        # Already processed
        # ----------------------------------------------------

        if research_state == "ALREADY_PROCESSED":

            cached = get_contact(
                email
            )

            if cached:

                print(
                    f"[Database] Returning "
                    f"cached research for "
                    f"{email}"
                )

                return build_cached_response(
                    cached,
                    "ALREADY_PROCESSED",
                )

        # ----------------------------------------------------
        # Another request is currently processing it
        # ----------------------------------------------------

        if research_state == "IN_PROGRESS":

            cached = get_contact(
                email
            )

            if cached:

                return build_cached_response(
                    cached,
                    "IN_PROGRESS",
                )

            return {
                "email": email,
                "hr_name": "Unknown",
                "company": "",
                "website": "",
                "scrape_status": "IN_PROGRESS",
                "company_name": "",
                "company_summary": "",
                "industry": "",
                "ai_ml_relevance": "",
                "relevant_technologies": [],
                "potential_roles": [],
                "fit_score": 0,
                "fit_reason": "",
                "email_subject": "",
                "email_body": "",
                "source_urls": [],
                "processing_status": (
                    "IN_PROGRESS"
                ),
                "from_cache": False,
                "email_status": "NOT_SENT",
            }

        # ----------------------------------------------------
        # 3. Extract domain
        # ----------------------------------------------------

        domain = extract_domain(
            email
        )

        print(
            f"[Analyze] Domain: {domain}"
        )

        # ----------------------------------------------------
        # 4. Scrape company
        # ----------------------------------------------------

        company = scrape_company(
            domain
        )

        if company.get("status") != "SUCCESS":

            error_message = (
                "Company website could not "
                "be retrieved or parsed."
            )

            mark_research_failed(
                email,
                error_message,
            )

            print(
                f"[Scraper] FAILED: {domain}"
            )

            return {
                "email": email,
                "hr_name": (
                    get_first_name(
                        supplied_hr_name
                    )
                    if supplied_hr_name
                    else "Unknown"
                ),
                "company": domain,
                "website": company.get(
                    "website",
                    f"https://{domain}",
                ),
                "scrape_status": "FAILED",
                "company_name": "",
                "company_summary": "",
                "industry": "",
                "ai_ml_relevance": "",
                "relevant_technologies": [],
                "potential_roles": [],
                "fit_score": 0,
                "fit_reason": "",
                "email_subject": "",
                "email_body": "",
                "source_urls": company.get(
                    "pages",
                    [],
                ),
                "research_status": "FAILED",
                "research_error": error_message,
                "email_status": "NOT_SENT",
                "processing_status": "NEW",
                "from_cache": False,
            }

        pages = company.get(
            "pages",
            [],
        )

        company_text = company.get(
            "text",
            "",
        )

        # ----------------------------------------------------
        # 5. Validate scraped content
        # ----------------------------------------------------

        if not company_text.strip():

            error_message = (
                "Website was reachable but "
                "no usable text was extracted."
            )

            mark_research_failed(
                email,
                error_message,
            )

            print(
                f"[Scraper] PARTIAL: {domain}"
            )

            return {
                "email": email,
                "hr_name": (
                    get_first_name(
                        supplied_hr_name
                    )
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
                "fit_reason": "",
                "email_subject": "",
                "email_body": "",
                "source_urls": pages,
                "research_status": "FAILED",
                "research_error": error_message,
                "email_status": "NOT_SENT",
                "processing_status": "NEW",
                "from_cache": False,
            }

        print(
            f"[Scraper] SUCCESS: {domain}"
        )

        # ----------------------------------------------------
        # 6. Gemini company analysis
        # ----------------------------------------------------

        analysis = (
            analyze_with_gemini(
                email=email,
                hr_name=supplied_hr_name,
                company_text=company_text,
                source_urls=pages,
            )
        )

        # ----------------------------------------------------
        # 7. Recruiter first-name normalization
        # ----------------------------------------------------

        first_name = get_first_name(
            analysis.get("hr_name")
        )

        # If main analysis could not determine the name,
        # make one focused fallback attempt.
        if first_name == "Unknown":

            first_name = (
                infer_first_name_with_gemini(
                    email=email,
                    company_text=company_text,
                )
            )

        # ----------------------------------------------------
        # 8. Build research record
        # ----------------------------------------------------

        record_to_save = {
            "email": email,

            "hr_name": first_name,

            "company_name": analysis.get(
                "company_name",
                "",
            ),

            "website": company.get(
                "website",
                f"https://{domain}",
            ),

            "scrape_status": "SUCCESS",

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

        # ----------------------------------------------------
        # 9. Persist research
        # ----------------------------------------------------

        save_research(
            record_to_save
        )

        print(
            "[Database] Research saved: "
            f"{email}"
        )

        # ----------------------------------------------------
        # 10. Return new research
        # ----------------------------------------------------

        return {
            **record_to_save,

            "company": record_to_save[
                "company_name"
            ],

            "research_status": "READY",

            "research_error": None,

            "email_status": "NOT_SENT",

            "sent_at": None,

            "gmail_message_id": None,

            "processing_status": "NEW",

            "from_cache": False,
        }

    except ValueError as exc:

        print(
            f"[Validation Error] {exc}"
        )

        if email:
            mark_research_failed(
                email,
                str(exc),
            )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        print(
            f"[File Error] {exc}"
        )

        if email:
            mark_research_failed(
                email,
                str(exc),
            )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print(
            f"[Server Error] {exc}"
        )

        if email:
            mark_research_failed(
                email,
                str(exc),
            )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# Sending state endpoints
# ============================================================

@app.post("/send/claim")
def send_claim(
    request: SendRequest,
) -> dict[str, Any]:

    email = (
        request.email
        .strip()
        .lower()
    )

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Email address cannot be empty.",
        )

    can_send = claim_send(
        email
    )

    record = get_contact(
        email
    )

    if record is None:

        return {
            "email": email,
            "can_send": False,
            "email_status": "NOT_FOUND",
        }

    return {
        "email": email,
        "can_send": can_send,
        "email_status": record.get(
            "email_status",
            "NOT_SENT",
        ),
        "research_status": record.get(
            "research_status",
            "",
        ),
        "company_name": record.get(
            "company_name",
            "",
        ),
        "email_subject": record.get(
            "email_subject",
            "",
        ),
        "email_body": record.get(
            "email_body",
            "",
        ),
    }


@app.post("/send/mark-sent")
def send_mark_sent(
    request: MarkSentRequest,
) -> dict[str, Any]:

    email = (
        request.email
        .strip()
        .lower()
    )

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Email address cannot be empty.",
        )

    mark_sent(
        email,
        request.gmail_message_id,
    )

    return {
        "email": email,
        "email_status": "SENT",
        "gmail_message_id": (
            request.gmail_message_id
        ),
    }


@app.post("/send/mark-failed")
def send_mark_failed(
    request: SendRequest,
) -> dict[str, Any]:

    email = (
        request.email
        .strip()
        .lower()
    )

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Email address cannot be empty.",
        )

    mark_send_failed(
        email
    )

    return {
        "email": email,
        "email_status": "NOT_SENT",
    }