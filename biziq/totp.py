"""Time-based one-time passwords (RFC 6238) and one-time recovery codes. Standard library only.

Works with Google Authenticator, Microsoft Authenticator, Authy, 1Password and any other TOTP app (SHA-1, 6 digits, 30 s).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote

STEP = 30
DIGITS = 6


def generate_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def _hotp(secret: str, counter: int, digits: int = DIGITS) -> str:
    key = base64.b32decode(secret.upper() + "=" * (-len(secret) % 8))
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    num = (struct.unpack(">I", mac[off:off + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(num).zfill(digits)


def code_at(secret: str, at: float | None = None, digits: int = DIGITS) -> str:
    return _hotp(secret, int((time.time() if at is None else at) // STEP), digits)


def verify(secret: str, code: str, last_step: int = 0, window: int = 1, at: float | None = None) -> int | None:
    """Return the time step that matched, or None. Accepts +-`window` steps for clock drift, and refuses any step at or
    before `last_step` so a code that was already used (or captured and replayed) cannot be used again."""
    code = "".join(ch for ch in (code or "") if ch.isdigit())
    if len(code) != DIGITS:
        return None
    now = int((time.time() if at is None else at) // STEP)
    hit = None
    for step in range(now - window, now + window + 1):          # no early exit: constant work whatever the input
        if hmac.compare_digest(_hotp(secret, step), code) and step > last_step:
            hit = step
    return hit


def provisioning_uri(secret: str, account: str, issuer: str = "BizIQ") -> str:
    return f"otpauth://totp/{quote(issuer)}:{quote(account)}?secret={secret}&issuer={quote(issuer)}&algorithm=SHA1&digits={DIGITS}&period={STEP}"


_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"        # no look-alikes (0/o, 1/l/i)


def new_recovery_codes(n: int = 8) -> list[str]:
    out = []
    for _ in range(n):
        raw = "".join(secrets.choice(_ALPHABET) for _ in range(12))
        out.append("-".join(raw[i:i + 4] for i in range(0, 12, 4)))
    return out


def normalise_code(code: str) -> str:
    return "".join(ch for ch in (code or "").lower() if ch.isalnum())


def hash_recovery_code(code: str) -> str:
    """Recovery codes are 60 bits of randomness, so a plain SHA-256 is sufficient (no password-style stretching needed)."""
    return hashlib.sha256(b"biziq-recovery|" + normalise_code(code).encode()).hexdigest()
