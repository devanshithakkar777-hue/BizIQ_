"""Transactional email for account flows (verification, invites, password reset).

Uses the same SMTP settings as the rest of BizIQ (BIZIQ_SMTP_HOST / _USER / _PASSWORD / BIZIQ_EMAIL_FROM) and the same safe default: with
BIZIQ_SEND_EMAIL unset nothing leaves the machine, the message is written to output/outbox/ instead and `delivered` is False.
"""
from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

import pandas as pd

from . import automation


@dataclass
class Sent:
    delivered: bool          # True only when an SMTP server accepted the message
    note: str


def base_url() -> str:
    return os.environ.get("BIZIQ_BASE_URL", "http://localhost:8501").rstrip("/")


def can_deliver() -> bool:
    cfg = automation.EmailConfig.from_env()
    return bool(cfg.enabled and cfg.host and cfg.sender)


def send(to: str, subject: str, body: str) -> Sent:
    cfg = automation.EmailConfig.from_env()
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, cfg.sender or "biziq@localhost", to
    msg.set_content(body)
    if can_deliver():
        with smtplib.SMTP(cfg.host, cfg.port, timeout=30) as s:
            s.starttls()
            if cfg.user:
                s.login(cfg.user, cfg.password)
            s.send_message(msg)
        return Sent(True, "sent")
    automation.OUTBOX.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in subject)[:40]
    path = automation.OUTBOX / f"{pd.Timestamp.now():%Y%m%d_%H%M%S_%f}_{safe}.eml"
    path.write_bytes(bytes(msg))
    return Sent(False, f"dry run: saved to {path.name}")
