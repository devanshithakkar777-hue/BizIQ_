"""Where connection settings and credentials live.

Settings that are not secret (shop address, region, organisation id) are stored as plain JSON. Secrets (tokens,
client secrets) are stored only when BIZIQ_SECRET_KEY is set (a Fernet key), and are encrypted with it. Without a key
nothing secret is written to disk: set the BIZIQ_<PROVIDER>_<FIELD> environment variables instead, e.g.
BIZIQ_SHOPIFY_TOKEN. Generate a key with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .. import crypto, db, platform_db


def _ddl(c) -> None:
    c.execute("""CREATE TABLE IF NOT EXISTS connections (
        org_id TEXT NOT NULL, provider TEXT NOT NULL, settings TEXT NOT NULL DEFAULT '{}', secrets BLOB,
        updated_at TEXT, last_sync_at TEXT, last_status TEXT NOT NULL DEFAULT '', PRIMARY KEY (org_id, provider))""")


def _conn():
    c = platform_db.connect()
    db.ensure(c, "connections", _ddl)
    return c


def _fernet():
    key = os.environ.get("BIZIQ_SECRET_KEY", "").strip()
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet
        return Fernet(key.encode())
    except Exception:
        return None


def secrets_available() -> bool:
    return crypto.available()


def env_name(provider: str, field: str) -> str:
    return f"BIZIQ_{provider}_{field}".upper()


def save(org_id: str, provider: str, settings: dict, secrets: dict | None = None) -> None:
    """Save settings; save secrets only if encryption is available. Blank secret values keep the stored ones."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        row = c.execute("SELECT secrets FROM connections WHERE org_id=? AND provider=?", (org_id, provider)).fetchone()
        current = _decrypt(row[0], org_id, provider) if row and row[0] else {}
        blob = row[0] if row else None
        if secrets:
            if not crypto.available():
                raise ValueError("Secrets cannot be stored: no BIZIQ_MASTER_KEY is set. Use environment variables instead.")
            merged = {**current, **{k: v for k, v in secrets.items() if v}}
            blob = crypto.encrypt(f"secrets:{org_id}", json.dumps(merged).encode(), provider)      # envelope, bound to org + provider
        c.execute("INSERT INTO connections (org_id, provider, settings, secrets, updated_at) VALUES (?,?,?,?,?) "
                  "ON CONFLICT(org_id, provider) DO UPDATE SET settings=excluded.settings, secrets=excluded.secrets, "
                  "updated_at=excluded.updated_at", (org_id, provider, json.dumps(settings), blob, now))


def _decrypt(blob, org_id: str = "", provider: str = "") -> dict:
    if not blob:
        return {}
    if crypto.is_encrypted(blob):                       # current format: raises CryptoError if tampered or the key is wrong
        return json.loads(crypto.decrypt(f"secrets:{org_id}", bytes(blob), provider).decode())
    f = _fernet()                                       # older installs: a single Fernet key; re-saved in the new format on next save
    if f is None:
        return {}
    try:
        return json.loads(f.decrypt(bytes(blob)).decode())
    except Exception:
        return {}


def load(org_id: str, provider: str, fields: list[str] = ()) -> dict:
    """Settings + secrets for one connection. An environment variable BIZIQ_<PROVIDER>_<FIELD> overrides what is stored."""
    with _conn() as c:
        row = c.execute("SELECT settings, secrets FROM connections WHERE org_id=? AND provider=?", (org_id, provider)).fetchone()
    out = json.loads(row[0]) if row else {}
    out.update(_decrypt(row[1], org_id, provider) if row else {})
    for fld in list(fields) + list(out):
        v = os.environ.get(env_name(provider, fld))
        if v:
            out[fld] = v
    return out


def configured(org_id: str) -> list[str]:
    with _conn() as c:
        return [r[0] for r in c.execute("SELECT provider FROM connections WHERE org_id=?", (org_id,)).fetchall()]


def status(org_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT provider, updated_at, last_sync_at, last_status FROM connections WHERE org_id=? ORDER BY provider",
                         (org_id,)).fetchall()
    return [dict(provider=r[0], saved=r[1], last_sync=r[2], last_status=r[3]) for r in rows]


def record_sync(org_id: str, provider: str, message: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("INSERT INTO connections (org_id, provider, last_sync_at, last_status) VALUES (?,?,?,?) "
                  "ON CONFLICT(org_id, provider) DO UPDATE SET last_sync_at=excluded.last_sync_at, last_status=excluded.last_status",
                  (org_id, provider, now, message[:500]))
