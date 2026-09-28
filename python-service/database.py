from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "hr_research.db"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        DB_FILE,
        timeout=30,
    )

    conn.row_factory = sqlite3.Row

    # Better reliability for concurrent reads/writes.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    return conn


def init_db() -> None:
    with get_connection() as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hr_contacts (
                email TEXT PRIMARY KEY,

                hr_name TEXT NOT NULL DEFAULT 'Unknown',

                company_name TEXT,
                website TEXT,

                scrape_status TEXT,

                company_summary TEXT,
                industry TEXT,
                ai_ml_relevance TEXT,

                relevant_technologies TEXT NOT NULL DEFAULT '[]',
                potential_roles TEXT NOT NULL DEFAULT '[]',

                fit_score INTEGER NOT NULL DEFAULT 0,
                fit_reason TEXT,

                email_subject TEXT,
                email_body TEXT,

                source_urls TEXT NOT NULL DEFAULT '[]',

                research_status TEXT NOT NULL DEFAULT 'PROCESSING',
                research_error TEXT,

                email_status TEXT NOT NULL DEFAULT 'NOT_SENT',
                sent_at TEXT,
                gmail_message_id TEXT,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_hr_contacts_research_status
            ON hr_contacts(research_status)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_hr_contacts_email_status
            ON hr_contacts(email_status)
            """
        )

        conn.commit()


def begin_research(
    email: str,
    stale_after_minutes: int = 30,
) -> str:
    """
    Claim an email for research.

    Returns:

        NEW
            New email or retryable failed/stale record.

        ALREADY_PROCESSED
            Research already exists. Do not regenerate.

        IN_PROGRESS
            Another/previous process is currently working on it.
    """

    email = normalize_email(email)
    current_time = now_utc()

    with get_connection() as conn:

        row = conn.execute(
            """
            SELECT *
            FROM hr_contacts
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

        # ----------------------------------------------------
        # Completely new email
        # ----------------------------------------------------

        if row is None:

            conn.execute(
                """
                INSERT INTO hr_contacts (
                    email,
                    research_status,
                    email_status,
                    created_at,
                    updated_at
                )
                VALUES (?, 'PROCESSING', 'NOT_SENT', ?, ?)
                """,
                (
                    email,
                    current_time,
                    current_time,
                ),
            )

            conn.commit()

            return "NEW"

        # ----------------------------------------------------
        # Research already completed
        # ----------------------------------------------------

        if row["research_status"] == "READY":

            return "ALREADY_PROCESSED"

        # ----------------------------------------------------
        # Research currently in progress
        # ----------------------------------------------------

        if row["research_status"] == "PROCESSING":

            try:
                previous_time = datetime.fromisoformat(
                    row["updated_at"]
                )

                cutoff = datetime.now(
                    timezone.utc
                ) - timedelta(
                    minutes=stale_after_minutes
                )

                if previous_time > cutoff:
                    return "IN_PROGRESS"

            except (TypeError, ValueError):
                pass

        # ----------------------------------------------------
        # FAILED or stale PROCESSING
        # Allow retry
        # ----------------------------------------------------

        conn.execute(
            """
            UPDATE hr_contacts
            SET
                research_status = 'PROCESSING',
                research_error = NULL,
                updated_at = ?
            WHERE email = ?
            """,
            (
                current_time,
                email,
            ),
        )

        conn.commit()

        return "NEW"


def get_contact(email: str) -> dict[str, Any] | None:
    email = normalize_email(email)

    with get_connection() as conn:

        row = conn.execute(
            """
            SELECT *
            FROM hr_contacts
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

    if row is None:
        return None

    result = dict(row)

    try:
        result["relevant_technologies"] = json.loads(
            result["relevant_technologies"]
        )
    except (TypeError, json.JSONDecodeError):
        result["relevant_technologies"] = []

    try:
        result["potential_roles"] = json.loads(
            result["potential_roles"]
        )
    except (TypeError, json.JSONDecodeError):
        result["potential_roles"] = []

    try:
        result["source_urls"] = json.loads(
            result["source_urls"]
        )
    except (TypeError, json.JSONDecodeError):
        result["source_urls"] = []

    return result


def save_research(record: dict[str, Any]) -> None:
    """
    Save completed company research.

    Existing email_status is preserved.
    This prevents a previously SENT contact from being reset.
    """

    email = normalize_email(record["email"])
    current_time = now_utc()

    relevant_technologies = json.dumps(
        record.get(
            "relevant_technologies",
            [],
        ),
        ensure_ascii=False,
    )

    potential_roles = json.dumps(
        record.get(
            "potential_roles",
            [],
        ),
        ensure_ascii=False,
    )

    source_urls = json.dumps(
        record.get(
            "source_urls",
            [],
        ),
        ensure_ascii=False,
    )

    with get_connection() as conn:

        conn.execute(
            """
            INSERT INTO hr_contacts (
                email,
                hr_name,
                company_name,
                website,
                scrape_status,
                company_summary,
                industry,
                ai_ml_relevance,
                relevant_technologies,
                potential_roles,
                fit_score,
                fit_reason,
                email_subject,
                email_body,
                source_urls,
                research_status,
                research_error,
                created_at,
                updated_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                'READY', NULL, ?, ?
            )

            ON CONFLICT(email)
            DO UPDATE SET
                hr_name = excluded.hr_name,
                company_name = excluded.company_name,
                website = excluded.website,
                scrape_status = excluded.scrape_status,
                company_summary = excluded.company_summary,
                industry = excluded.industry,
                ai_ml_relevance = excluded.ai_ml_relevance,
                relevant_technologies = excluded.relevant_technologies,
                potential_roles = excluded.potential_roles,
                fit_score = excluded.fit_score,
                fit_reason = excluded.fit_reason,
                email_subject = excluded.email_subject,
                email_body = excluded.email_body,
                source_urls = excluded.source_urls,
                research_status = 'READY',
                research_error = NULL,
                updated_at = excluded.updated_at
            """,
            (
                email,
                record.get("hr_name", "Unknown"),
                record.get("company_name", ""),
                record.get("website", ""),
                record.get("scrape_status", ""),
                record.get("company_summary", ""),
                record.get("industry", ""),
                record.get("ai_ml_relevance", ""),
                relevant_technologies,
                potential_roles,
                int(record.get("fit_score", 0)),
                record.get("fit_reason", ""),
                record.get("email_subject", ""),
                record.get("email_body", ""),
                source_urls,
                current_time,
                current_time,
            ),
        )

        conn.commit()


def mark_research_failed(
    email: str,
    error: str,
) -> None:

    email = normalize_email(email)

    with get_connection() as conn:

        conn.execute(
            """
            UPDATE hr_contacts
            SET
                research_status = 'FAILED',
                research_error = ?,
                updated_at = ?
            WHERE email = ?
            """,
            (
                error[:4000],
                now_utc(),
                email,
            ),
        )

        conn.commit()


def claim_send(
    email: str,
    stale_after_minutes: int = 60,
) -> bool:
    """
    Atomically reserve a contact for email sending.

    Returns False when:
        - already SENT
        - currently SENDING
        - research is not READY
    """

    email = normalize_email(email)

    with get_connection() as conn:

        row = conn.execute(
            """
            SELECT *
            FROM hr_contacts
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

        if row is None:
            return False

        if row["research_status"] != "READY":
            return False

        if row["email_status"] == "SENT":
            return False

        if row["email_status"] == "SENDING":

            # Allow recovery from a stale sending state.
            try:
                updated = datetime.fromisoformat(
                    row["updated_at"]
                )

                cutoff = datetime.now(
                    timezone.utc
                ) - timedelta(
                    minutes=stale_after_minutes
                )

                if updated > cutoff:
                    return False

            except (TypeError, ValueError):
                return False

        conn.execute(
            """
            UPDATE hr_contacts
            SET
                email_status = 'SENDING',
                updated_at = ?
            WHERE email = ?
            """,
            (
                now_utc(),
                email,
            ),
        )

        conn.commit()

        return True


def mark_sent(
    email: str,
    gmail_message_id: str | None = None,
) -> None:

    email = normalize_email(email)

    with get_connection() as conn:

        conn.execute(
            """
            UPDATE hr_contacts
            SET
                email_status = 'SENT',
                sent_at = ?,
                gmail_message_id = ?,
                updated_at = ?
            WHERE email = ?
            """,
            (
                now_utc(),
                gmail_message_id,
                now_utc(),
                email,
            ),
        )

        conn.commit()


def mark_send_failed(email: str) -> None:

    email = normalize_email(email)

    with get_connection() as conn:

        conn.execute(
            """
            UPDATE hr_contacts
            SET
                email_status = 'NOT_SENT',
                updated_at = ?
            WHERE email = ?
            """,
            (
                now_utc(),
                email,
            ),
        )

        conn.commit()


# Initialize database when the module is imported.
init_db()