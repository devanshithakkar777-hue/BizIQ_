"""Organisations, users, roles and per-organisation data separation.

* Passwords are hashed with scrypt (salted); plain passwords are never stored or logged.
* Five roles with explicit permissions (see PERMISSIONS). Modules about people (salary, hiring, performance)
  need the `view_people` permission.
* Each organisation has its own business-data database folder; activate_org() points the current Streamlit
  session (a ContextVar, so sessions never share it) at that folder.
* Repeated failed logins lock the account for a short time.
"""
from __future__ import annotations

import hashlib
import json
import hmac
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import audit, crypto, data_layer, db, mailer, platform_db, profile as profiles, totp

ROLES = ("owner", "admin", "analyst", "hr", "viewer")
PERMISSIONS: dict[str, set[str]] = {
    "owner":   {"view", "upload", "run", "request", "approve", "manage_users", "view_people", "view_audit", "export"},
    "admin":   {"view", "upload", "run", "request", "approve", "manage_users", "view_people", "view_audit", "export"},
    "analyst": {"view", "upload", "run", "request", "export"},
    "hr":      {"view", "request", "view_people"},
    "viewer":  {"view"},
}
MAX_FAILED = 5
LOCK_MINUTES = 15
MIN_PASSWORD = int(os.environ.get("BIZIQ_MIN_PASSWORD", "10"))


class TwoFactorRequired(Exception):
    """Password was right and the account has two-factor on: ask for the 6-digit code (or a recovery code), then call authenticate again."""


class AccountError(ValueError):
    """Raised with a message that is safe to show to the user."""


@dataclass(frozen=True)
class User:
    id: int
    org_id: str
    org_name: str
    email: str
    name: str
    role: str
    org_slug: str = ""


# ------------------------------------------------------------------ storage
def _ddl(c) -> None:
    c.execute("""CREATE TABLE IF NOT EXISTS orgs (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, allow_self_approval INTEGER NOT NULL DEFAULT 0)""")
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, org_id TEXT NOT NULL REFERENCES orgs(id), email TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL, role TEXT NOT NULL, pw_hash TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
        failed_count INTEGER NOT NULL DEFAULT 0, locked_until TEXT, created_at TEXT NOT NULL)""")
    for table, coldef in (("users", "totp_secret BLOB"), ("users", "totp_enabled INTEGER NOT NULL DEFAULT 0"),
                          ("users", "totp_last_step INTEGER NOT NULL DEFAULT 0"), ("users", "recovery_hashes TEXT NOT NULL DEFAULT '[]'"),
                          ("orgs", "require_2fa INTEGER NOT NULL DEFAULT 0"), ("orgs", "slug TEXT"), ("orgs", "profile TEXT"),
                          ("users", "email_verified INTEGER NOT NULL DEFAULT 1"), ("users", "last_login TEXT")):
        db.add_column_if_missing(c, table, coldef)
    if c.execute("SELECT 1 FROM orgs WHERE slug IS NULL LIMIT 1").fetchone():      # legacy organisations: their id becomes their login name
        c.execute("UPDATE orgs SET slug = id WHERE slug IS NULL")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS orgs_slug_unique ON orgs (slug)")
    c.execute("""CREATE TABLE IF NOT EXISTS auth_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, token_hash TEXT NOT NULL, email TEXT NOT NULL, org_id TEXT,
        name TEXT, role TEXT, expires_at TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)""")
    c.execute("CREATE TABLE IF NOT EXISTS rate_events (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, key TEXT NOT NULL, at TEXT NOT NULL)")


def _conn():
    c = platform_db.connect()
    db.ensure(c, "accounts", _ddl)
    return c


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        h = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex), n=2 ** 14, r=8, p=1, dklen=32)
        return hmac.compare_digest(h.hex(), hash_hex)
    except Exception:
        return False


COMMON_PASSWORDS = {
    "password", "password1", "password123", "passw0rd1", "12345678", "123456789", "1234567890", "qwerty123", "qwertyuiop", "iloveyou1",
    "admin123", "welcome1", "welcome123", "letmein123", "abc12345", "abcd1234", "changeme1", "changeme123", "india1234", "india@123",
    "pass@123", "password@123", "test1234", "biziq123", "biziq1234", "qwerty@123", "admin@123", "welcome@123", "123456abc", "11111111aa"}


def check_password_strength(password: str, email: str = "") -> None:
    if len(password) < MIN_PASSWORD:
        raise AccountError(f"Password must be at least {MIN_PASSWORD} characters.")
    if len(password) > 256:
        raise AccountError("Password is too long (256 characters at most).")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise AccountError("Password must contain at least one letter and one number.")
    low = password.lower()
    if low in COMMON_PASSWORDS or len(set(low)) <= 3:
        raise AccountError("That password is too common or too repetitive. Choose something less guessable.")
    local = (email or "").split("@")[0].lower()
    if len(local) >= 4 and local in low:
        raise AccountError("Password must not contain your email name.")


def _clean_email(email: str) -> str:
    e = (email or "").strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", e):
        raise AccountError("Enter a valid email address.")
    return e


# ------------------------------------------------------------------ permissions
def can(user: User | None, perm: str) -> bool:
    return bool(user) and perm in PERMISSIONS.get(user.role, set())


def can_view_module(user: User | None, spec) -> bool:
    """Sensitive (people) modules need view_people; everything else needs view."""
    if not can(user, "view"):
        return False
    return can(user, "view_people") if getattr(spec, "sensitive", False) else True


def require(user: User, perm: str) -> None:
    if not can(user, perm):
        raise AccountError("Your role does not allow this action.")


# ------------------------------------------------------------------ organisations and users
def list_org_ids() -> list[str]:
    with _conn() as c:
        return [r[0] for r in c.execute("SELECT id FROM orgs ORDER BY created_at").fetchall()]


def any_org_exists() -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM orgs LIMIT 1").fetchone() is not None


RESERVED_SLUGS = {"admin", "api", "app", "login", "signup", "www", "static", "biziq", "help", "support", "billing", "root", "demo", "test", "status"}


def clean_slug(raw: str) -> str:
    """A business's own login name: 3-30 letters, digits and hyphens. It appears in its login link, so keep it plain."""
    slug = re.sub(r"[^a-z0-9]+", "-", (raw or "").lower()).strip("-")
    if not 3 <= len(slug) <= 30:
        raise AccountError("The business login name must be 3 to 30 letters or digits (hyphens allowed).")
    if slug in RESERVED_SLUGS:
        raise AccountError("That login name is reserved. Choose another.")
    return slug


def create_org(org_name: str, owner_email: str, owner_name: str, password: str, business_type: str = "other",
               slug: str | None = None, email_verified: bool = True) -> User:
    org_name = (org_name or "").strip()
    if len(org_name) < 2:
        raise AccountError("Enter the business name.")
    email = _clean_email(owner_email)
    check_password_strength(password, email)
    if business_type not in profiles.BUSINESS_TYPES:
        raise AccountError("Choose your type of business.")
    explicit = bool(slug and slug.strip())
    base = clean_slug(slug) if explicit else (re.sub(r"[^a-z0-9]+", "-", org_name.lower()).strip("-")[:24] or "business")
    if not explicit and (len(base) < 3 or base in RESERVED_SLUGS):
        base = f"{base}-co"
    with _conn() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise AccountError("An account with this email already exists.")
        taken = c.execute("SELECT 1 FROM orgs WHERE slug=?", (base,)).fetchone()
        if taken and explicit:
            raise AccountError("That business login name is taken. Choose another.")
        slug_ = base if not taken else f"{base}-{secrets.token_hex(2)}"
        org_id = re.sub(r"[^a-z0-9]+", "-", org_name.lower()).strip("-")[:24] + "-" + secrets.token_hex(3)
        prof = profiles.Profile(business_type, profiles.defaults_for(business_type))
        c.execute("INSERT INTO orgs (id, name, created_at, slug, profile) VALUES (?,?,?,?,?)", (org_id, org_name, _now().isoformat(), slug_, prof.to_json()))
        c.execute("INSERT INTO users (org_id, email, name, role, pw_hash, created_at, email_verified) VALUES (?,?,?,?,?,?,?)",
                  (org_id, email, owner_name.strip() or email, "owner", hash_password(password), _now().isoformat(), 1 if email_verified else 0))
    if not db.IS_PG:
        org_dir(org_id).mkdir(parents=True, exist_ok=True)
    audit.log(org_id, email, "org.create", org_id, f"{org_name} ({business_type})")
    return _user_by_email(email, include_unverified=True)


def _user_by_email(email: str, include_unverified: bool = False) -> User | None:
    with _conn() as c:
        r = c.execute("SELECT u.id, u.org_id, o.name, u.email, u.name, u.role, o.slug FROM users u "
                      "JOIN orgs o ON o.id=u.org_id WHERE u.email=? AND u.active=1" + ("" if include_unverified else " AND u.email_verified=1"), (email,)).fetchone()
    return User(*r) if r else None


def add_user(actor: User, email: str, name: str, role: str, password: str) -> User:
    require(actor, "manage_users")
    email = _clean_email(email)
    if role not in ROLES or role == "owner":
        raise AccountError("Choose a role: admin, analyst, hr or viewer.")
    if role == "admin" and actor.role != "owner":
        raise AccountError("Only an owner can create admins.")
    check_password_strength(password, email)
    with _conn() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise AccountError("An account with this email already exists.")
        c.execute("INSERT INTO users (org_id, email, name, role, pw_hash, created_at) VALUES (?,?,?,?,?,?)",
                  (actor.org_id, email, name.strip() or email, role, hash_password(password), _now().isoformat()))
    audit.log(actor.org_id, actor.email, "user.add", email, f"role={role}")
    return _user_by_email(email)


def list_users(actor: User) -> list[dict]:
    require(actor, "manage_users")
    with _conn() as c:
        rows = c.execute("SELECT email, name, role, active, created_at FROM users WHERE org_id=? ORDER BY id",
                         (actor.org_id,)).fetchall()
    return [dict(email=e, name=n, role=r, active=bool(a), created_at=ca) for e, n, r, a, ca in rows]


def set_role(actor: User, email: str, role: str) -> None:
    require(actor, "manage_users")
    email = _clean_email(email)
    if role not in ROLES:
        raise AccountError("Unknown role.")
    with _conn() as c:
        r = c.execute("SELECT role FROM users WHERE email=? AND org_id=?", (email, actor.org_id)).fetchone()
        if not r:
            raise AccountError("No such user in your organisation.")
        owners = c.execute("SELECT COUNT(*) FROM users WHERE org_id=? AND role='owner' AND active=1",
                           (actor.org_id,)).fetchone()[0]
        if r[0] == "owner" and role != "owner" and owners <= 1:
            raise AccountError("An organisation must keep at least one owner.")
        if (r[0] == "owner" or role in ("owner", "admin")) and actor.role != "owner":
            raise AccountError("Only an owner can change owner or admin roles.")
        c.execute("UPDATE users SET role=? WHERE email=? AND org_id=?", (role, email, actor.org_id))
    audit.log(actor.org_id, actor.email, "user.role", email, f"role={role}")


def deactivate_user(actor: User, email: str) -> None:
    require(actor, "manage_users")
    email = _clean_email(email)
    if email == actor.email:
        raise AccountError("You cannot deactivate your own account.")
    with _conn() as c:
        r = c.execute("SELECT role FROM users WHERE email=? AND org_id=?", (email, actor.org_id)).fetchone()
        if not r:
            raise AccountError("No such user in your organisation.")
        if r[0] == "owner" and actor.role != "owner":
            raise AccountError("Only an owner can deactivate an owner.")
        c.execute("UPDATE users SET active=0 WHERE email=? AND org_id=?", (email, actor.org_id))
    audit.log(actor.org_id, actor.email, "user.deactivate", email)


def change_password(user: User, old: str, new: str) -> None:
    check_password_strength(new, user.email)
    with _conn() as c:
        r = c.execute("SELECT pw_hash FROM users WHERE id=?", (user.id,)).fetchone()
        if not r or not verify_password(old, r[0]):
            raise AccountError("Current password is incorrect.")
        c.execute("UPDATE users SET pw_hash=? WHERE id=?", (hash_password(new), user.id))
    audit.log(user.org_id, user.email, "user.password_change", user.email)


def set_self_approval(actor: User, allowed: bool) -> None:
    if actor.role != "owner":
        raise AccountError("Only an owner can change this setting.")
    with _conn() as c:
        c.execute("UPDATE orgs SET allow_self_approval=? WHERE id=?", (int(allowed), actor.org_id))
    audit.log(actor.org_id, actor.email, "org.self_approval", str(allowed))


def self_approval_allowed(org_id: str) -> bool:
    with _conn() as c:
        r = c.execute("SELECT allow_self_approval FROM orgs WHERE id=?", (org_id,)).fetchone()
    return bool(r and r[0])


# ------------------------------------------------------------------ login
def _register_failure(c, uid: int, failed: int) -> None:
    failed += 1
    lock = (_now() + timedelta(minutes=LOCK_MINUTES)).isoformat() if failed >= MAX_FAILED else None
    c.execute("UPDATE users SET failed_count=?, locked_until=? WHERE id=?", (0 if lock else failed, lock, uid))


def authenticate(email: str, password: str, otp: str | None = None, org_slug: str | None = None) -> User | None:
    """Return the user on success. Raises AccountError if the account is locked. Never says which part was wrong.
    If the account has two-factor on and no `otp` is given, raises TwoFactorRequired once the password is right.
    A wrong code counts toward the same lockout as a wrong password.
    `org_slug` (a business's own login page): a user of another business gets the same "incorrect" answer, so the page reveals nothing about who belongs where."""
    try:
        e = _clean_email(email)
    except AccountError:
        return None
    event, recovery_left = None, None
    with _conn() as c:
        r = c.execute("SELECT u.id, u.org_id, u.pw_hash, u.failed_count, u.locked_until, u.totp_enabled, u.totp_secret, u.totp_last_step, u.recovery_hashes, "
                      "u.email_verified, o.slug FROM users u JOIN orgs o ON o.id=u.org_id WHERE u.email=? AND u.active=1", (e,)).fetchone()
        if not r:
            verify_password(password, "scrypt$00$00")     # keep timing similar for unknown emails
            return None
        uid, org_id, pw_hash, failed, locked_until, t_on, t_secret, t_last, rec, verified, slug = r
        if locked_until and datetime.fromisoformat(locked_until) > _now():
            raise AccountError(f"Too many failed attempts. Try again in {LOCK_MINUTES} minutes.")
        if not verify_password(password, pw_hash):
            _register_failure(c, uid, failed)
            event = "login.fail"
        elif org_slug and (slug or "") != org_slug:
            return None
        elif not verified:
            raise AccountError("Please verify your email first: use the link or code we sent you.")
        elif t_on:
            if otp is None or not str(otp).strip():
                raise TwoFactorRequired()
            step = totp.verify(crypto.decrypt(f"secrets:{org_id}", bytes(t_secret), f"totp|{e}").decode(), otp, int(t_last or 0))
            used = None
            if step is not None:
                c.execute("UPDATE users SET totp_last_step=?, failed_count=0, locked_until=NULL WHERE id=?", (step, uid))
                event = "login.ok"
            else:                                         # not a current code: maybe a one-time recovery code
                h, codes = totp.hash_recovery_code(otp), json.loads(rec or "[]")
                if h in codes:
                    codes.remove(h)
                    c.execute("UPDATE users SET recovery_hashes=?, failed_count=0, locked_until=NULL WHERE id=?", (json.dumps(codes), uid))
                    event, used = "login.ok", len(codes)
                else:
                    _register_failure(c, uid, failed)
                    event = "login.2fa_fail"
            recovery_left = used
        else:
            c.execute("UPDATE users SET failed_count=0, locked_until=NULL WHERE id=?", (uid,))
            event = "login.ok"
    if recovery_left is not None:
        audit.log(org_id, e, "2fa.recovery_used", e, f"{recovery_left} recovery code(s) left")
    audit.log(org_id, e, event, e)
    if event == "login.2fa_fail":
        raise AccountError("That code is not valid or has already been used.")
    if event == "login.ok":
        with _conn() as c:
            c.execute("UPDATE users SET last_login=? WHERE id=?", (_now().isoformat(), uid))
        return _user_by_email(e)
    return None


# ------------------------------------------------------------------ two-factor
def two_factor_enabled(user: User) -> bool:
    with _conn() as c:
        r = c.execute("SELECT totp_enabled FROM users WHERE id=?", (user.id,)).fetchone()
    return bool(r and r[0])


def two_factor_required(org_id: str) -> bool:
    with _conn() as c:
        r = c.execute("SELECT require_2fa FROM orgs WHERE id=?", (org_id,)).fetchone()
    return bool(r and r[0])


def needs_2fa_enrollment(user: User) -> bool:
    """True when the organisation requires two-factor and this user has not set it up yet."""
    return two_factor_required(user.org_id) and not two_factor_enabled(user)


def set_require_2fa(actor: User, required: bool) -> None:
    if actor.role not in ("owner", "admin"):
        raise AccountError("Only an owner or admin can change this setting.")
    if required and not two_factor_enabled(actor):
        raise AccountError("Set up two-factor for your own account first, so you cannot lock yourself out.")
    with _conn() as c:
        c.execute("UPDATE orgs SET require_2fa=? WHERE id=?", (int(required), actor.org_id))
    audit.log(actor.org_id, actor.email, "org.require_2fa", str(required))


def begin_2fa_enrollment(user: User) -> tuple[str, str]:
    """Create a pending secret. Returns (secret, otpauth URI for a QR code). Nothing is enforced until confirm_2fa_enrollment."""
    if not crypto.available():
        raise AccountError("Two-factor needs an encryption key on the server (BIZIQ_MASTER_KEY) to store the secret safely.")
    if two_factor_enabled(user):
        raise AccountError("Two-factor is already on. Turn it off first to set it up again.")
    secret = totp.generate_secret()
    blob = crypto.encrypt(f"secrets:{user.org_id}", secret.encode(), f"totp|{user.email}")
    with _conn() as c:
        c.execute("UPDATE users SET totp_secret=?, totp_enabled=0, totp_last_step=0 WHERE id=?", (blob, user.id))
    return secret, totp.provisioning_uri(secret, user.email)


def confirm_2fa_enrollment(user: User, code: str) -> list[str]:
    """Check a code from the app; if right, turn two-factor on and return ten one-time recovery codes (shown once, stored hashed)."""
    with _conn() as c:
        r = c.execute("SELECT totp_secret, totp_enabled FROM users WHERE id=?", (user.id,)).fetchone()
        if not r or not r[0] or r[1]:
            raise AccountError("Start setup first.")
        secret = crypto.decrypt(f"secrets:{user.org_id}", bytes(r[0]), f"totp|{user.email}").decode()
        step = totp.verify(secret, code)
        if step is None:
            raise AccountError("That code is not right. Check the phone's clock is automatic, then try the newest code.")
        codes = totp.new_recovery_codes(10)
        c.execute("UPDATE users SET totp_enabled=1, totp_last_step=?, recovery_hashes=? WHERE id=?",
                  (step, json.dumps([totp.hash_recovery_code(x) for x in codes]), user.id))
    audit.log(user.org_id, user.email, "2fa.enabled", user.email)
    return codes


def disable_2fa(actor: User, target_email: str | None = None, password: str | None = None) -> None:
    """Turn two-factor off. Your own account: needs your password. Someone else's (lost phone): owner/admin only."""
    own = target_email is None or _clean_email(target_email) == actor.email
    email = actor.email if own else _clean_email(target_email)
    with _conn() as c:
        if own:
            r = c.execute("SELECT pw_hash FROM users WHERE id=?", (actor.id,)).fetchone()
            if not r or not verify_password(password or "", r[0]):
                raise AccountError("Enter your current password to turn two-factor off.")
        else:
            require(actor, "manage_users")
            t = c.execute("SELECT role FROM users WHERE email=? AND org_id=?", (email, actor.org_id)).fetchone()
            if not t:
                raise AccountError("No such user in your organisation.")
            if t[0] == "owner" and actor.role != "owner":
                raise AccountError("Only an owner can reset another owner.")
        if two_factor_required(actor.org_id):
            raise AccountError("Your organisation requires two-factor. Turn the requirement off first.")
        c.execute("UPDATE users SET totp_enabled=0, totp_secret=NULL, totp_last_step=0, recovery_hashes='[]' WHERE email=? AND org_id=?",
                  (email, actor.org_id))
    audit.log(actor.org_id, actor.email, "2fa.disabled" if own else "2fa.reset", email)


def regenerate_recovery_codes(user: User, password: str) -> list[str]:
    with _conn() as c:
        r = c.execute("SELECT pw_hash, totp_enabled FROM users WHERE id=?", (user.id,)).fetchone()
        if not r or not verify_password(password or "", r[0]):
            raise AccountError("Enter your current password.")
        if not r[1]:
            raise AccountError("Two-factor is not on.")
        codes = totp.new_recovery_codes(10)
        c.execute("UPDATE users SET recovery_hashes=? WHERE id=?", (json.dumps([totp.hash_recovery_code(x) for x in codes]), user.id))
    audit.log(user.org_id, user.email, "2fa.recovery_regenerated", user.email)
    return codes


# ------------------------------------------------------------------ single sign-on
def sso_user(email: str, email_verified: bool) -> User | None:
    """Map a verified identity from the company's identity provider to an existing, active BizIQ user.
    Never creates accounts: an owner or admin must add the person first. Unverified emails are refused."""
    if not email_verified:
        return None
    try:
        e = _clean_email(email)
    except AccountError:
        return None
    u = _user_by_email(e)
    if u:
        audit.log(u.org_id, e, "login.sso", e)
    return u


# ------------------------------------------------------------------ per-organisation data
def org_dir(org_id: str) -> Path:
    safe = re.sub(r"[^a-z0-9-]", "", org_id)
    return platform_db.OUT / "orgs" / safe


def activate_org(org_id: str) -> None:
    """Point this session's data layer at the organisation's own database (a ContextVar: per session)."""
    d = org_dir(org_id)
    if not db.IS_PG:
        d.mkdir(parents=True, exist_ok=True)
    data_layer.use_db(d / "biziq.db")          # on PostgreSQL this path selects the schema org_<id>


# ------------------------------------------------------------------ business profile (what the business does -> which modules apply)
def org_by_slug(slug: str) -> dict | None:
    """Public facts for a business's own login page: its name and look. Nothing about its people or data."""
    try:
        slug = re.sub(r"[^a-z0-9-]", "", (slug or "").lower())[:40]
    except TypeError:
        return None
    if not slug:
        return None
    with _conn() as c:
        r = c.execute("SELECT id, name, slug, profile FROM orgs WHERE slug=?", (slug,)).fetchone()
    if not r:
        return None
    pf = profiles.Profile.from_json(r[3])
    return dict(id=r[0], name=r[1], slug=r[2], accent=pf.accent, tagline=pf.tagline, business_type=pf.business_type)


def get_profile(org_id: str) -> "profiles.Profile":
    with _conn() as c:
        r = c.execute("SELECT profile FROM orgs WHERE id=?", (org_id,)).fetchone()
    return profiles.Profile.from_json(r[0] if r else None)


def set_profile(actor: User, business_type: str, caps, accent: str | None = None, tagline: str | None = None) -> "profiles.Profile":
    require(actor, "manage_users")
    if business_type not in profiles.BUSINESS_TYPES:
        raise AccountError("Choose your type of business.")
    cur = get_profile(actor.org_id)
    new = profiles.Profile(business_type, frozenset(c for c in caps if c in profiles.CAPABILITIES), cur.overrides,
                           profiles.clean_accent(accent) if accent else cur.accent, (tagline if tagline is not None else cur.tagline)[:120], True)
    with _conn() as c:
        c.execute("UPDATE orgs SET profile=? WHERE id=?", (new.to_json(), actor.org_id))
    audit.log(actor.org_id, actor.email, "profile.update", actor.org_id, f"{business_type}; {len(new.caps)} capabilities")
    return new


def set_module_override(actor: User, module: str, enabled: bool) -> None:
    """Owner/admin: show a module that does not fit the business profile ("enable anyway"), or hide it again."""
    require(actor, "manage_users")
    if module not in profiles.MODULE_CAPS:
        raise AccountError("Unknown module.")
    cur = get_profile(actor.org_id)
    ov = set(cur.overrides)
    (ov.add if enabled else ov.discard)(module)
    new = profiles.Profile(cur.business_type, cur.caps, frozenset(ov), cur.accent, cur.tagline, cur.confirmed)
    with _conn() as c:
        c.execute("UPDATE orgs SET profile=? WHERE id=?", (new.to_json(), actor.org_id))
    audit.log(actor.org_id, actor.email, "module.enable" if enabled else "module.disable", module)


# ------------------------------------------------------------------ rate limits, tokens
def rate_limit(kind: str, key: str, max_events: int, window_minutes: int) -> None:
    """Allow at most `max_events` of `kind` for `key` per window; raises AccountError beyond that (and records the attempt otherwise)."""
    since = (_now() - timedelta(minutes=window_minutes)).isoformat()
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM rate_events WHERE kind=? AND key=? AND at>?", (kind, key, since)).fetchone()[0]
        if n >= max_events:
            raise AccountError("Too many attempts. Please wait a while and try again.")
        c.execute("INSERT INTO rate_events (kind, key, at) VALUES (?,?,?)", (kind, key, _now().isoformat()))
        if secrets.randbelow(50) == 0:
            c.execute("DELETE FROM rate_events WHERE at<?", ((_now() - timedelta(days=2)).isoformat(),))


def _hash_token(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def _issue_token(kind: str, email: str, org_id: str | None, hours: float, name: str = "", role: str = "") -> str:
    raw = secrets.token_urlsafe(24)
    with _conn() as c:
        c.execute("UPDATE auth_tokens SET used=1 WHERE kind=? AND email=? AND used=0", (kind, email))        # a new link replaces older ones
        c.execute("INSERT INTO auth_tokens (kind, token_hash, email, org_id, name, role, expires_at, created_at) VALUES (?,?,?,?,?,?,?,?)",
                  (kind, _hash_token(raw), email, org_id, name, role, (_now() + timedelta(hours=hours)).isoformat(), _now().isoformat()))
    return raw


def _consume_token(kind: str, token: str) -> dict | None:
    h = _hash_token((token or "").strip())
    with _conn() as c:
        r = c.execute("SELECT id, email, org_id, name, role, expires_at, used FROM auth_tokens WHERE kind=? AND token_hash=?", (kind, h)).fetchone()
        if not r or r[6] or datetime.fromisoformat(r[5]) < _now():
            return None
        c.execute("UPDATE auth_tokens SET used=1 WHERE id=?", (r[0],))
    return dict(email=r[1], org_id=r[2], name=r[3], role=r[4])


def _peek_token(kind: str, token: str) -> dict | None:
    h = _hash_token((token or "").strip())
    with _conn() as c:
        r = c.execute("SELECT email, org_id, name, role, expires_at, used FROM auth_tokens WHERE kind=? AND token_hash=?", (kind, h)).fetchone()
    if not r or r[5] or datetime.fromisoformat(r[4]) < _now():
        return None
    return dict(email=r[0], org_id=r[1], name=r[2], role=r[3])


def _link(slug: str, **q) -> str:
    from urllib.parse import urlencode
    return f"{mailer.base_url()}/?{urlencode({'org': slug, **q})}"


def require_email_verification() -> bool:
    return os.environ.get("BIZIQ_REQUIRE_EMAIL_VERIFICATION", "0") == "1"


# ------------------------------------------------------------------ sign-up
@dataclass
class SignupResult:
    user: User | None          # set when the new owner can be signed in straight away
    message: str               # always safe to show: it never reveals whether an email already has an account
    needs_verification: bool = False


def signup(org_name: str, owner_name: str, email: str, password: str, business_type: str, slug: str | None = None) -> SignupResult:
    """Public sign-up. With BIZIQ_REQUIRE_EMAIL_VERIFICATION=1 the account stays locked until the emailed link is used, and an
    existing email gets the same on-screen answer as a new one (the real owner is emailed instead), so the form cannot be used to
    discover who has an account."""
    cap = int(os.environ.get("BIZIQ_SIGNUP_MAX_PER_HOUR", "30"))
    rate_limit("signup_all", "all", cap, 60)
    e = _clean_email(email)
    rate_limit("signup_email", e, 5, 60)
    if not require_email_verification():
        return SignupResult(create_org(org_name, e, owner_name, password, business_type, slug), "Account created.")
    if not mailer.can_deliver():
        raise AccountError("This server requires email verification but cannot send email yet. The administrator needs to configure SMTP.")
    generic = "Check your email: we sent a message with the next step. It can take a minute."
    with _conn() as c:
        exists = c.execute("SELECT 1 FROM users WHERE email=?", (e,)).fetchone() is not None
    if exists:
        mailer.send(e, "Someone tried to create a BizIQ account with this email",
                    "Someone just tried to sign up to BizIQ with this email address, but it already has an account.\n\n"
                    "If it was you, log in or use 'Forgot password'. If it was not you, you can ignore this message.")
        return SignupResult(None, generic, True)
    u = create_org(org_name, e, owner_name, password, business_type, slug, email_verified=False)
    tok = _issue_token("verify", e, u.org_id, 24)
    mailer.send(e, "Confirm your BizIQ account", f"Welcome to BizIQ, {u.name}.\n\nConfirm your email to finish setting up {u.org_name}:\n{_link(u.org_slug, verify=tok)}\n\n"
                "The link works for 24 hours. If you did not sign up, ignore this message.")
    return SignupResult(None, generic, True)


def verify_email(token: str) -> User | None:
    info = _consume_token("verify", token)
    if not info:
        return None
    with _conn() as c:
        c.execute("UPDATE users SET email_verified=1 WHERE email=?", (info["email"],))
    audit.log(info["org_id"] or "", info["email"], "email.verified", info["email"])
    return _user_by_email(info["email"])


# ------------------------------------------------------------------ password reset
def request_password_reset(email: str) -> str:
    """Always the same answer. If the account exists and email is configured, a one-hour link is sent."""
    msg = "If that email has an account, we have sent a link to reset the password."
    try:
        e = _clean_email(email)
    except AccountError:
        return msg
    rate_limit("reset_all", "all", int(os.environ.get("BIZIQ_RESET_MAX_PER_HOUR", "60")), 60)
    rate_limit("reset_email", e, 3, 60)
    u = _user_by_email(e, include_unverified=True)
    if u:
        tok = _issue_token("reset", e, u.org_id, 1)
        mailer.send(e, "Reset your BizIQ password", f"Use this link within an hour to choose a new password:\n{_link(u.org_slug, reset=tok)}\n\n"
                    "If you did not ask for this, ignore this message: your password has not changed.")
        audit.log(u.org_id, e, "password.reset_requested", e)
    return msg


def reset_password(token: str, new_password: str) -> User:
    info = _peek_token("reset", token)
    if not info:
        raise AccountError("This reset link is not valid or has expired. Ask for a new one.")
    check_password_strength(new_password, info["email"])
    _consume_token("reset", token)
    with _conn() as c:
        c.execute("UPDATE users SET pw_hash=?, failed_count=0, locked_until=NULL, email_verified=1 WHERE email=?", (hash_password(new_password), info["email"]))
    audit.log(info["org_id"] or "", info["email"], "password.reset", info["email"])
    return _user_by_email(info["email"])


# ------------------------------------------------------------------ invitations (the invited person chooses their own password)
def invite_user(actor: User, email: str, name: str, role: str) -> str:
    """Owner/admin invites a colleague. Returns the invite link: it is emailed when SMTP is on, and shown to the inviter either way."""
    require(actor, "manage_users")
    e = _clean_email(email)
    if role not in ROLES or role == "owner":
        raise AccountError("Choose a role: admin, analyst, hr or viewer.")
    if role == "admin" and actor.role != "owner":
        raise AccountError("Only an owner can invite admins.")
    rate_limit("invite", actor.org_id, 50, 24 * 60)
    with _conn() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (e,)).fetchone():
            raise AccountError("An account with this email already exists.")
    tok = _issue_token("invite", e, actor.org_id, 24 * 7, name.strip(), role)
    link = _link(actor.org_slug, invite=tok)
    mailer.send(e, f"{actor.name} invited you to {actor.org_name} on BizIQ", f"You have been invited to {actor.org_name} as {role}.\n\nChoose your password here (valid for 7 days):\n{link}")
    audit.log(actor.org_id, actor.email, "user.invite", e, f"role={role}")
    return link


def invite_info(token: str) -> dict | None:
    info = _peek_token("invite", token)
    if not info:
        return None
    with _conn() as c:
        r = c.execute("SELECT name, slug FROM orgs WHERE id=?", (info["org_id"],)).fetchone()
    return dict(email=info["email"], role=info["role"], name=info["name"], org_name=r[0] if r else "", slug=r[1] if r else "")


def accept_invite(token: str, password: str, name: str | None = None) -> User:
    info = _peek_token("invite", token)
    if not info:
        raise AccountError("This invitation is not valid or has expired. Ask for a new one.")
    check_password_strength(password, info["email"])
    _consume_token("invite", token)
    with _conn() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (info["email"],)).fetchone():
            raise AccountError("An account with this email already exists.")
        c.execute("INSERT INTO users (org_id, email, name, role, pw_hash, created_at, email_verified) VALUES (?,?,?,?,?,?,1)",
                  (info["org_id"], info["email"], (name or info["name"] or info["email"]).strip(), info["role"], hash_password(password), _now().isoformat()))
    audit.log(info["org_id"], info["email"], "user.accept_invite", info["email"], f"role={info['role']}")
    return _user_by_email(info["email"])
