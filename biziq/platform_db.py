"""Platform database: organisations, users, approvals and the audit log.

This is separate from each organisation's business-data database, so business data of one organisation can
never be reached through another's account. Path can be overridden for tests / deployment with BIZIQ_PLATFORM_DB.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from . import db

OUT = Path(os.environ.get("BIZIQ_DATA_DIR") or Path(__file__).resolve().parent.parent / "output")
PLATFORM_DB = Path(os.environ.get("BIZIQ_PLATFORM_DB", OUT / "platform.db"))
_DEFAULT_PLATFORM = PLATFORM_DB


class _Conn(sqlite3.Connection):
    """`with connect() as c:` commits (or rolls back) and then closes."""
    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


def connect():
    if db.IS_PG:
        return db.pg_connect(db.schema_for(PLATFORM_DB, _DEFAULT_PLATFORM))
    PLATFORM_DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(PLATFORM_DB, timeout=30, factory=_Conn)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c
