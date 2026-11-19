"""Approval workflow: request -> review -> approve / reject, with a saved record.

Used for anything that leaves the business or commits money (purchase orders, outbound emails, price changes,
payment reminders). A request cannot be approved by the person who made it unless the owner has switched on
self-approval for a one-person organisation. Every step is written to the audit log.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from . import accounts, audit, db, platform_db
from .accounts import AccountError, User

KINDS = ("purchase_order", "email", "price_change", "payment_reminder", "report_share", "other")


def _ddl(c) -> None:
    c.execute("""CREATE TABLE IF NOT EXISTS approvals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, org_id TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL,
        payload TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'pending', requested_by TEXT NOT NULL,
        requested_at TEXT NOT NULL, decided_by TEXT, decided_at TEXT, note TEXT NOT NULL DEFAULT '')""")


def _conn():
    c = platform_db.connect()
    db.ensure(c, "approvals", _ddl)
    return c


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def request(user: User, kind: str, title: str, payload: dict | None = None) -> int:
    accounts.require(user, "request")
    if kind not in KINDS:
        raise AccountError(f"Unknown request type. Choose from: {', '.join(KINDS)}.")
    if not title.strip():
        raise AccountError("Give the request a short title.")
    with _conn() as c:
        cur = c.execute("INSERT INTO approvals (org_id, kind, title, payload, requested_by, requested_at) "
                        "VALUES (?,?,?,?,?,?) RETURNING id", (user.org_id, kind, title.strip(),
                                                 json.dumps(payload or {}, default=str), user.email, _now()))
        rid = cur.fetchone()[0]
    audit.log(user.org_id, user.email, "approval.request", f"#{rid}", f"{kind}: {title.strip()}")
    return rid


def decide(user: User, approval_id: int, approve: bool, note: str = "") -> None:
    accounts.require(user, "approve")
    with _conn() as c:
        r = c.execute("SELECT org_id, status, requested_by, title FROM approvals WHERE id=?", (approval_id,)).fetchone()
        if not r or r[0] != user.org_id:
            raise AccountError("Request not found.")
        if r[1] != "pending":
            raise AccountError(f"This request is already {r[1]}.")
        if r[2] == user.email and not accounts.self_approval_allowed(user.org_id):
            raise AccountError("You cannot approve your own request. Ask another approver "
                               "(an owner can allow self-approval for one-person teams).")
        c.execute("UPDATE approvals SET status=?, decided_by=?, decided_at=?, note=? WHERE id=?",
                  ("approved" if approve else "rejected", user.email, _now(), note.strip(), approval_id))
    audit.log(user.org_id, user.email, "approval.approve" if approve else "approval.reject",
              f"#{approval_id}", note.strip() or r[3])


def list_requests(user: User, status: str | None = None) -> pd.DataFrame:
    accounts.require(user, "view")
    q = ("SELECT id, kind, title, status, requested_by, requested_at, decided_by, decided_at, note "
         "FROM approvals WHERE org_id=?")
    args: list = [user.org_id]
    if status:
        q += " AND status=?"
        args.append(status)
    q += " ORDER BY id DESC"
    with _conn() as c:
        return db.read_df(c, q, args)


def get_payload(user: User, approval_id: int) -> dict:
    with _conn() as c:
        r = c.execute("SELECT payload FROM approvals WHERE id=? AND org_id=?", (approval_id, user.org_id)).fetchone()
    return json.loads(r[0]) if r else {}
