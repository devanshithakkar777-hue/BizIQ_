"""Append-only, tamper-evident audit log.

Every important action (login, upload, approval, export, user change) is written to `audit_log` in the
platform database. Rows cannot be updated or deleted (SQLite triggers refuse), and each row stores a hash
that covers the previous row, so any edit made behind the application's back breaks the chain and is detected
by verify_chain().
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from . import db, platform_db

GENESIS = "0" * 64


def _ddl(c) -> None:
    c.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, org_id TEXT NOT NULL, user_email TEXT NOT NULL,
        action TEXT NOT NULL, target TEXT NOT NULL DEFAULT '', detail TEXT NOT NULL DEFAULT '',
        prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL)""")
    if isinstance(c, db.PGConn):
        c.execute("""CREATE OR REPLACE FUNCTION audit_append_only() RETURNS trigger AS $$
                     BEGIN RAISE EXCEPTION 'audit_log is append-only'; END; $$ LANGUAGE plpgsql""")
        c.execute("DROP TRIGGER IF EXISTS audit_no_update ON audit_log")
        c.execute("DROP TRIGGER IF EXISTS audit_no_delete ON audit_log")
        c.execute("DROP TRIGGER IF EXISTS audit_no_truncate ON audit_log")
        c.execute("CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_log FOR EACH ROW EXECUTE FUNCTION audit_append_only()")
        c.execute("CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_log FOR EACH ROW EXECUTE FUNCTION audit_append_only()")
        c.execute("CREATE TRIGGER audit_no_truncate BEFORE TRUNCATE ON audit_log FOR EACH STATEMENT EXECUTE FUNCTION audit_append_only()")
        return
    c.execute("""CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
                 BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END""")
    c.execute("""CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
                 BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END""")


def _conn():
    c = platform_db.connect()
    db.ensure(c, "audit", _ddl)
    return c


def _hash(prev: str, ts: str, org_id: str, user: str, action: str, target: str, detail: str) -> str:
    payload = json.dumps([prev, ts, org_id, user, action, target, detail], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def log(org_id: str, user_email: str, action: str, target: str = "", detail: str = "") -> None:
    """Record one event. `detail` should never contain passwords or personal data you do not need."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with _conn() as c:
        if isinstance(c, db.PGConn):                      # serialise writers per organisation so the hash chain cannot fork
            c.execute("SELECT pg_advisory_xact_lock(hashtext(?))", (org_id,))
        row = c.execute("SELECT row_hash FROM audit_log WHERE org_id=? ORDER BY id DESC LIMIT 1",
                        (org_id,)).fetchone()
        prev = row[0] if row else GENESIS
        h = _hash(prev, ts, org_id, user_email, action, target, detail)
        c.execute("INSERT INTO audit_log (ts, org_id, user_email, action, target, detail, prev_hash, row_hash) "
                  "VALUES (?,?,?,?,?,?,?,?)", (ts, org_id, user_email, action, target, detail, prev, h))


def read(org_id: str, limit: int = 500, action: str | None = None) -> pd.DataFrame:
    q = "SELECT ts, user_email, action, target, detail FROM audit_log WHERE org_id=?"
    args: list = [org_id]
    if action:
        q += " AND action=?"
        args.append(action)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        return db.read_df(c, q, args)


def verify_chain(org_id: str) -> tuple[bool, str]:
    """Recompute every hash for one organisation. Returns (ok, message)."""
    with _conn() as c:
        rows = c.execute("SELECT id, ts, user_email, action, target, detail, prev_hash, row_hash "
                         "FROM audit_log WHERE org_id=? ORDER BY id", (org_id,)).fetchall()
    prev = GENESIS
    for rid, ts, user, action, target, detail, p, h in rows:
        if p != prev or _hash(prev, ts, org_id, user, action, target, detail) != h:
            return False, f"Audit log was altered at entry {rid}."
        prev = h
    return True, f"Audit log intact ({len(rows)} entries)."
