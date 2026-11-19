# BizIQ security: what exists, how to run it, what it does not do

## What is implemented (and tested)
| Area | Behaviour | Tests |
|---|---|---|
| Passwords | scrypt hashes, min length 10 (`BIZIQ_MIN_PASSWORD`), common-password denylist, no repeats, may not contain the email name, lockout after repeated failures | `tests/test_security.py`, `tests/test_platform.py` |
| Two-factor (TOTP) | RFC 6238 (checked against the RFC test vectors), replay refused, ±1 step drift, 10 one-time recovery codes (stored hashed), secret stored encrypted, wrong codes count toward lockout, org owners can require it for everyone, owner can reset a lost phone (audited) | yes |
| Single sign-on | OpenID Connect via Streamlit's built-in `st.login`. A verified email is mapped to an **existing active** user only; never auto-creates accounts | mapping tested, **not tested against a real identity provider** |
| Sessions | idle timeout (30 min) and absolute timeout (12 h), configurable | yes |
| Encryption at rest | AES-256-GCM envelope encryption: master key -> per-organisation data keys -> 8 MB chunks. Ciphertext is bound to organisation + dataset + chunk position, so moving or truncating blobs fails loudly. Applies to datasets (`BIZIQ_ENCRYPT_DATA=1`), integration secrets (Shopify/Tally/Zoho tokens) and 2FA secrets | yes, both backends |
| Sign-up and recovery | per-business sign-in slug, identical failure for a wrong business or wrong password, email verification (`BIZIQ_REQUIRE_EMAIL_VERIFICATION=1`, needs SMTP), hashed single-use expiring tokens (a new link voids the old one), generic responses that do not reveal whether an email exists, rate limits (sign-up 5/h per email and `BIZIQ_SIGNUP_MAX_PER_HOUR` overall, reset 3/h per email) | `tests/test_access.py` |
| Audit log | hash-chained, append-only (database triggers on Postgres), serialised per org | yes |
| Key rotation | master key (re-wrap, no data rewrite), per-org data key (re-encrypt + retire), encrypt/decrypt existing data: `scripts/rotate_keys.py` | yes |

## Key management (read this before turning encryption on)
* `BIZIQ_MASTER_KEY` = 32 random bytes, urlsafe base64. Generate: `python -c "from biziq import crypto; print(crypto.generate_master_key())"`.
  Alternatives: `BIZIQ_MASTER_KEY_FILE` (a mounted secret), or, for old installs, keys are derived from `BIZIQ_SECRET_KEY`.
* **Back the master key up somewhere that is not the database or its backups** (password manager, cloud secret manager). Lose it and the encrypted data is gone for everyone, including you. Database backups alone are useless without it, which is the point.
* Rotate master key: new key in `BIZIQ_MASTER_KEY`, old in `BIZIQ_MASTER_KEY_PREVIOUS`, run `python scripts/rotate_keys.py master`, then remove the old one.
* Rotate one customer's data key: `python scripts/rotate_keys.py data --org <id>`.
* Turn encryption on for existing data: set the key, then `python scripts/rotate_keys.py encrypt`. Roll back: `decrypt`. `status` shows what is encrypted.
* Moving SQLite -> Postgres keeps ciphertext untouched (`scripts/migrate_to_postgres.py` also copies `data_keys`). Same master key needed on the new server.

## Deployment checklist
1. Postgres: use a dedicated role that owns only the BizIQ database, not a superuser. Require TLS: add `?sslmode=require` (or `verify-full` with a CA) to `BIZIQ_DATABASE_URL`. Do not expose port 5432 publicly.
2. Put BizIQ behind a TLS reverse proxy (Caddy or nginx). Streamlit itself does not terminate HTTPS. Send `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`.
3. Set `BIZIQ_AUTH=1`. Create the owner, enrol 2FA, then tick "require two-factor" for the organisation.
4. SSO: add an `[auth]` block to `.streamlit/secrets.toml` (redirect_uri, cookie_secret, client_id, client_secret, server_metadata_url), then `BIZIQ_SSO=1`. Two-factor for SSO users is the identity provider's job; BizIQ does not ask SSO users for a second code.
5. Files in `output/` (exports, backups, generated reports) are **not** encrypted by BizIQ. Use disk or volume encryption on the host, and keep `backups/` off the same machine.
6. Keep `BIZIQ_SEND_EMAIL` / `BIZIQ_SEND_WHATSAPP` off until you have reviewed the first dry-run logs.

## BillBook data
* Customer phone numbers are encrypted with the organisation data key **only when `BIZIQ_ENCRYPT_DATA=1`**. The other BillBook fields (item names, amounts, customer names, bill notes) are stored in plain columns of that business's own database or schema, like the rest of its tables, and are protected by database access and disk encryption, not field encryption.
* The data export (zip of CSVs) is generated on demand, requires the `export` permission and is audited; the file you download is not encrypted.

## Reports that leave the building
* The consultant pack (PDF and Excel) needs the `export` permission and every preparation is audit-logged (which sections, whether people sections were included, whether names were hidden). The downloaded files are not encrypted. Hiding names is a best-effort text replacement and can miss a name that is spelled differently in free text.

## Honest limits
* Unlike logins and sign-up, uploads and billing writes have no rate limit.
* Security headers (HSTS, `X-Frame-Options`) are a reverse-proxy job; no automated check of a deployed site has been run.
* No third-party penetration test or security review has been done. Treat this as a solid baseline, not a certification (no SOC 2 / ISO 27001 claim).
* Decrypted data lives in server memory while it is being analysed; encryption at rest does not protect against someone with live access to the running process.
* The master key is held in the app's environment. A real HSM or cloud KMS (AWS KMS, GCP KMS, Vault) would be stronger; `crypto.py` is the single place to swap that in.
* SQL access: all queries are parameterised, but identifiers (table names) come from a fixed list. No automated fuzzing for injection has been run beyond the tests.
* Rate limiting exists for logins only, not for uploads or API-style calls.
