"""Session lifetime rules, kept free of Streamlit so they can be tested: an idle timeout and an absolute maximum."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

IDLE_MINUTES = float(os.environ.get("BIZIQ_SESSION_IDLE_MINUTES", "30"))
MAX_HOURS = float(os.environ.get("BIZIQ_SESSION_MAX_HOURS", "12"))


def now() -> datetime:
    return datetime.now(timezone.utc)


def expired(login_at: datetime, last_seen: datetime, at: datetime | None = None,
            idle_minutes: float | None = None, max_hours: float | None = None) -> str | None:
    """None if the session is still valid, else a short reason to show the user."""
    at = at or now()
    idle = IDLE_MINUTES if idle_minutes is None else idle_minutes
    cap = MAX_HOURS if max_hours is None else max_hours
    if idle and at - last_seen > timedelta(minutes=idle):
        return f"You were signed out after {idle:g} minutes of inactivity."
    if cap and at - login_at > timedelta(hours=cap):
        return f"For safety, sessions last at most {cap:g} hours. Please sign in again."
    return None
