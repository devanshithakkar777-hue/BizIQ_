"""Envelope encryption for data at rest.

    master key (you hold it)  ->  wraps  ->  one data key (DEK) per scope  ->  encrypts  ->  your data

* Algorithm: AES-256-GCM (authenticated: any change to the stored bytes is detected, never silently decrypted).
* A *scope* is "data:<organisation schema>" for business datasets or "secrets:<org id>" for saved connection secrets.
  Each scope has its own random DEK, so compromising or rotating one organisation never touches another.
* Every ciphertext is bound to where it belongs (additional authenticated data = scope | dataset | chunk), so a blob
  copied into another organisation's table, or another dataset, fails to decrypt.
* DEKs are stored wrapped (encrypted by the master key) in the platform database table `data_keys`. Without the master
  key, a database dump or a stolen file contains nothing readable.

Master key: BIZIQ_MASTER_KEY (urlsafe base64, 32 bytes) or BIZIQ_MASTER_KEY_FILE. For deployments that already set
BIZIQ_SECRET_KEY a master key is derived from it, so upgrading needs no new setting. During master-key rotation, put the
old key(s) in BIZIQ_MASTER_KEY_PREVIOUS (comma separated); see scripts/rotate_keys.py.

Ciphertext format:  b"BZ1" | dek id (4 bytes) | nonce (12 bytes) | AES-GCM ciphertext+tag
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets as _secrets
import threading
from datetime import datetime, timezone

from . import db, platform_db

MAGIC = b"BZ1"
_cache: dict[tuple[str, int], bytes] = {}
_lock = threading.Lock()


class CryptoError(Exception):
    pass


# ------------------------------------------------------------------------------------------------ master key
def generate_master_key() -> str:
    return base64.urlsafe_b64encode(_secrets.token_bytes(32)).decode()


def _decode(k: str) -> bytes:
    try:
        raw = base64.urlsafe_b64decode(k.strip() + "=" * (-len(k.strip()) % 4))
    except Exception as e:
        raise CryptoError("BIZIQ_MASTER_KEY is not valid base64.") from e
    if len(raw) != 32:
        raise CryptoError("BIZIQ_MASTER_KEY must decode to exactly 32 bytes. Generate one with: python -c \"from biziq import crypto; print(crypto.generate_master_key())\"")
    return raw


def _from_secret_key(s: str) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"biziq-master-v1").derive(s.strip().encode())


def master_keys() -> list[bytes]:
    """[current, *previous]. Empty when no key is configured."""
    keys: list[bytes] = []
    cur = os.environ.get("BIZIQ_MASTER_KEY", "").strip()
    f = os.environ.get("BIZIQ_MASTER_KEY_FILE", "").strip()
    if not cur and f and os.path.exists(f):
        cur = open(f, encoding="utf-8").read().strip()
    if cur:
        keys.append(_decode(cur))
    elif os.environ.get("BIZIQ_SECRET_KEY", "").strip():
        keys.append(_from_secret_key(os.environ["BIZIQ_SECRET_KEY"]))
    for p in os.environ.get("BIZIQ_MASTER_KEY_PREVIOUS", "").split(","):
        if p.strip():
            keys.append(_decode(p))
    return keys


def available() -> bool:
    try:
        return bool(master_keys())
    except CryptoError:
        return False


def _kid(master: bytes) -> str:
    return hashlib.sha256(b"kid" + master).hexdigest()[:8]


def _current_master() -> bytes:
    ks = master_keys()
    if not ks:
        raise CryptoError("No encryption key is configured. Set BIZIQ_MASTER_KEY (see docs/SECURITY.md).")
    return ks[0]


def _aes(key: bytes):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key)


# ------------------------------------------------------------------------------------------------ data keys
def _ddl(c) -> None:
    c.execute("""CREATE TABLE IF NOT EXISTS data_keys (
        scope TEXT NOT NULL, kid INTEGER NOT NULL, wrapped BLOB NOT NULL, master_kid TEXT NOT NULL,
        created_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1, PRIMARY KEY (scope, kid))""")


def _conn():
    c = platform_db.connect()
    db.ensure(c, "crypto", _ddl)
    return c


def _wrap(master: bytes, dek: bytes, scope: str, kid: int) -> bytes:
    nonce = _secrets.token_bytes(12)
    return nonce + _aes(master).encrypt(nonce, dek, f"dek|{scope}|{kid}".encode())


def _unwrap(master: bytes, wrapped: bytes, scope: str, kid: int) -> bytes:
    return _aes(master).decrypt(bytes(wrapped[:12]), bytes(wrapped[12:]), f"dek|{scope}|{kid}".encode())


def _unwrap_any(wrapped: bytes, scope: str, kid: int) -> bytes:
    last = None
    for m in master_keys():
        try:
            return _unwrap(m, wrapped, scope, kid)
        except Exception as e:           # wrong master key for this row: try the next (previous) one
            last = e
    raise CryptoError("Cannot unlock the data key: the master key is wrong or missing.") from last


def _active_key(scope: str) -> tuple[int, bytes]:
    master = _current_master()
    with _conn() as c:
        row = c.execute("SELECT kid, wrapped FROM data_keys WHERE scope=? AND active=1 ORDER BY kid DESC LIMIT 1", (scope,)).fetchone()
        if row is None:
            kid, dek = 1, _secrets.token_bytes(32)
            c.execute("INSERT INTO data_keys (scope, kid, wrapped, master_kid, created_at, active) VALUES (?,?,?,?,?,1) ON CONFLICT DO NOTHING",
                      (scope, kid, _wrap(master, dek, scope, kid), _kid(master), datetime.now(timezone.utc).isoformat()))
            row = c.execute("SELECT kid, wrapped FROM data_keys WHERE scope=? AND active=1 ORDER BY kid DESC LIMIT 1", (scope,)).fetchone()
    kid, wrapped = int(row[0]), bytes(row[1])
    return kid, _dek(scope, kid, wrapped)


def _dek(scope: str, kid: int, wrapped: bytes | None = None) -> bytes:
    key = (scope, kid)
    with _lock:
        hit = _cache.get(key)
    if hit:
        return hit
    if wrapped is None:
        with _conn() as c:
            r = c.execute("SELECT wrapped FROM data_keys WHERE scope=? AND kid=?", (scope, kid)).fetchone()
        if r is None:
            raise CryptoError("The data key for this record no longer exists.")
        wrapped = bytes(r[0])
    dek = _unwrap_any(wrapped, scope, kid)
    with _lock:
        _cache[key] = dek
    return dek


def clear_cache() -> None:
    with _lock:
        _cache.clear()


# ------------------------------------------------------------------------------------------------ encrypt / decrypt
def encrypt(scope: str, plaintext: bytes, aad: str) -> bytes:
    kid, dek = _active_key(scope)
    nonce = _secrets.token_bytes(12)
    ct = _aes(dek).encrypt(nonce, plaintext, f"{scope}|{aad}".encode())
    return MAGIC + kid.to_bytes(4, "big") + nonce + ct


def decrypt(scope: str, blob: bytes, aad: str) -> bytes:
    blob = bytes(blob)
    if blob[:3] != MAGIC or len(blob) < 3 + 4 + 12 + 16:
        raise CryptoError("This value is not in BizIQ's encrypted format.")
    kid = int.from_bytes(blob[3:7], "big")
    try:
        return _aes(_dek(scope, kid)).decrypt(blob[7:19], blob[19:], f"{scope}|{aad}".encode())
    except CryptoError:
        raise
    except Exception as e:
        raise CryptoError("Decryption failed: the data was changed, moved, or encrypted under a different key.") from e


def is_encrypted(blob) -> bool:
    return blob is not None and bytes(blob[:3]) == MAGIC


# ------------------------------------------------------------------------------------------------ rotation
def rewrap_all() -> int:
    """After setting a new BIZIQ_MASTER_KEY (and the old one in BIZIQ_MASTER_KEY_PREVIOUS): re-wrap every DEK under the new
    master key. Cheap, and no business data is touched. Returns the number of keys re-wrapped."""
    master = _current_master()
    n = 0
    with _conn() as c:
        rows = c.execute("SELECT scope, kid, wrapped FROM data_keys").fetchall()
        for scope, kid, wrapped in rows:
            dek = _unwrap_any(bytes(wrapped), scope, int(kid))
            c.execute("UPDATE data_keys SET wrapped=?, master_kid=? WHERE scope=? AND kid=?", (_wrap(master, dek, scope, int(kid)), _kid(master), scope, kid))
            n += 1
    clear_cache()
    return n


def new_key_version(scope: str) -> int:
    """Create a fresh DEK for the scope and make it the active one (older versions stay readable until data is re-encrypted)."""
    master = _current_master()
    with _conn() as c:
        row = c.execute("SELECT COALESCE(MAX(kid), 0) FROM data_keys WHERE scope=?", (scope,)).fetchone()
        kid = int(row[0]) + 1
        c.execute("UPDATE data_keys SET active=0 WHERE scope=?", (scope,))
        c.execute("INSERT INTO data_keys (scope, kid, wrapped, master_kid, created_at, active) VALUES (?,?,?,?,?,1)",
                  (scope, kid, _wrap(master, _secrets.token_bytes(32), scope, kid), _kid(master), datetime.now(timezone.utc).isoformat()))
    return kid


def retire_old_versions(scope: str) -> int:
    """Delete inactive DEK versions once all data has been re-encrypted under the active one."""
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM data_keys WHERE scope=? AND active=0", (scope,)).fetchone()[0]
        c.execute("DELETE FROM data_keys WHERE scope=? AND active=0", (scope,))
    clear_cache()
    return int(n)


def key_status() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT scope, kid, master_kid, created_at, active FROM data_keys ORDER BY scope, kid").fetchall()
    cur = _kid(master_keys()[0]) if master_keys() else None
    return [dict(scope=r[0], version=int(r[1]), wrapped_by_current_master=(r[2] == cur), created=r[3], active=bool(r[4])) for r in rows]


# ------------------------------------------------------------------------------------------------ small helpers used elsewhere
def constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())
