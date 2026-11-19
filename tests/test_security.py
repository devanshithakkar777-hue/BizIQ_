"""Encryption at rest, two-factor, SSO mapping, sessions and password policy."""
import io
import json
import sqlite3
from datetime import timedelta

import pandas as pd
import pytest

from biziq import accounts, audit, crypto, data_layer as d, db, platform_db, session, totp

PW = "Passw0rdOK"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_db, "PLATFORM_DB", tmp_path / "platform.db")
    monkeypatch.setattr(platform_db, "OUT", tmp_path)
    monkeypatch.setenv("BIZIQ_MASTER_KEY", crypto.generate_master_key())
    monkeypatch.delenv("BIZIQ_SECRET_KEY", raising=False)
    monkeypatch.delenv("BIZIQ_MASTER_KEY_PREVIOUS", raising=False)
    monkeypatch.delenv("BIZIQ_ENCRYPT_DATA", raising=False)
    crypto.clear_cache()
    yield tmp_path
    d.use_db(None)
    crypto.clear_cache()


@pytest.fixture()
def owner(env):
    u = accounts.create_org("Acme", "owner@acme.test", "Olive", PW)
    accounts.activate_org(u.org_id)
    return u


def inv(i, **kw):
    return dict(invoice_id=i, party=kw.get("party", "P"), type="receivable", invoice_date="2026-01-01", due_date="2026-02-01", amount=5, paid_date=None)


# ------------------------------------------------------------------ envelope encryption
class TestCrypto:
    def test_roundtrip_and_ciphertext_is_not_plaintext(self, env):
        blob = crypto.encrypt("data:org_a", b"top secret ledger", "sales|0/1")
        assert b"top secret" not in blob and crypto.decrypt("data:org_a", blob, "sales|0/1") == b"top secret ledger"

    def test_same_plaintext_encrypts_differently_each_time(self, env):
        assert crypto.encrypt("s", b"x", "a") != crypto.encrypt("s", b"x", "a")

    @pytest.mark.parametrize("where", [10, 25, -1])
    def test_any_flipped_byte_is_detected(self, env, where):
        blob = bytearray(crypto.encrypt("s", b"payload payload", "a"))
        blob[where] ^= 1
        with pytest.raises(crypto.CryptoError):
            crypto.decrypt("s", bytes(blob), "a")

    def test_blob_cannot_be_moved_to_another_org_or_dataset(self, env):
        blob = crypto.encrypt("data:org_a", b"secret", "sales|0/1")
        with pytest.raises(crypto.CryptoError):
            crypto.decrypt("data:org_b", blob, "sales|0/1")           # another organisation
        with pytest.raises(crypto.CryptoError):
            crypto.decrypt("data:org_a", blob, "invoices|0/1")        # another dataset
        with pytest.raises(crypto.CryptoError):
            crypto.decrypt("data:org_a", blob, "sales|0/2")           # chunk count changed (truncation)

    def test_wrong_master_key_cannot_unlock(self, env, monkeypatch):
        blob = crypto.encrypt("s", b"x", "a")
        monkeypatch.setenv("BIZIQ_MASTER_KEY", crypto.generate_master_key())
        crypto.clear_cache()
        with pytest.raises(crypto.CryptoError):
            crypto.decrypt("s", blob, "a")

    def test_no_key_means_unavailable_and_refuses(self, env, monkeypatch):
        monkeypatch.delenv("BIZIQ_MASTER_KEY")
        assert not crypto.available()
        with pytest.raises(crypto.CryptoError, match="No encryption key"):
            crypto.encrypt("s", b"x", "a")

    def test_bad_master_key_is_rejected_clearly(self, env, monkeypatch):
        monkeypatch.setenv("BIZIQ_MASTER_KEY", "tooshort")
        with pytest.raises(crypto.CryptoError, match="32 bytes|base64"):
            crypto.master_keys()

    def test_existing_secret_key_deployments_get_a_master_key_automatically(self, env, monkeypatch):
        monkeypatch.delenv("BIZIQ_MASTER_KEY")
        monkeypatch.setenv("BIZIQ_SECRET_KEY", "old-fernet-style-key-value")
        crypto.clear_cache()
        assert crypto.decrypt("s", crypto.encrypt("s", b"hello", "a"), "a") == b"hello"

    def test_master_key_rotation_rewraps_without_touching_data(self, env, monkeypatch):
        blob = crypto.encrypt("data:o", b"data", "a")
        old = __import__("os").environ["BIZIQ_MASTER_KEY"]
        new = crypto.generate_master_key()
        monkeypatch.setenv("BIZIQ_MASTER_KEY", new)
        monkeypatch.setenv("BIZIQ_MASTER_KEY_PREVIOUS", old)
        crypto.clear_cache()
        assert crypto.rewrap_all() >= 1
        monkeypatch.delenv("BIZIQ_MASTER_KEY_PREVIOUS")                 # the old key is no longer needed anywhere
        crypto.clear_cache()
        assert crypto.decrypt("data:o", blob, "a") == b"data"
        assert all(k["wrapped_by_current_master"] for k in crypto.key_status())

    def test_new_key_version_keeps_old_data_readable_then_retire(self, env):
        old = crypto.encrypt("data:o", b"v1", "a")
        v = crypto.new_key_version("data:o")
        new = crypto.encrypt("data:o", b"v2", "a")
        assert v == 2 and int.from_bytes(new[3:7], "big") == 2 and int.from_bytes(old[3:7], "big") == 1
        assert crypto.decrypt("data:o", old, "a") == b"v1"
        assert crypto.retire_old_versions("data:o") == 1
        crypto.clear_cache()
        with pytest.raises(crypto.CryptoError):
            crypto.decrypt("data:o", old, "a")                          # retired key: the v1 blob is now unreadable by design


# ------------------------------------------------------------------ encrypted datasets
MARK = "UNIQUE-MARKER-9f3a"


class TestEncryptedData:
    def _raw_text(self):
        with d._conn() as c:
            cols = db.columns(c, "invoices")
            rows = c.execute('SELECT * FROM "invoices"').fetchall()
        return cols, repr(rows)

    def test_database_holds_no_readable_data_when_on(self, owner, monkeypatch):
        d.ingest(pd.DataFrame([inv("I1", party=MARK), inv("I2")]), "invoices")                  # control: plaintext
        cols, text = self._raw_text()
        assert MARK in text and "party" in cols
        monkeypatch.setenv("BIZIQ_ENCRYPT_DATA", "1")
        d.ingest(pd.DataFrame([inv("I1", party=MARK), inv("I2")]), "invoices")
        cols, text = self._raw_text()
        assert cols == ["chunk", "blob"] and MARK not in text
        assert sorted(d.load_dataset("invoices")["invoice_id"]) == ["I1", "I2"]                   # still works
        assert d.load_dataset("invoices").party.iloc[0] == MARK

    def test_sqlite_file_bytes_contain_nothing_readable(self, owner, monkeypatch):
        if db.IS_PG:
            pytest.skip("file-level check is SQLite")
        monkeypatch.setenv("BIZIQ_ENCRYPT_DATA", "1")
        d.ingest(pd.DataFrame([inv("I1", party=MARK)]), "invoices")
        raw = d.current_db_path().read_bytes() + (d.current_db_path().with_suffix(".db-wal").read_bytes() if d.current_db_path().with_suffix(".db-wal").exists() else b"")
        assert MARK.encode() not in raw

    def test_merge_and_undo_work_on_encrypted_tables(self, owner, monkeypatch):
        monkeypatch.setenv("BIZIQ_ENCRYPT_DATA", "1")
        d.ingest(pd.DataFrame([inv("I1"), inv("I2")]), "invoices")
        r = d.ingest(pd.DataFrame([inv("I1"), inv("I3")]), "invoices", mode="merge")
        assert (r.save.added, r.save.unchanged) == (1, 1) and len(d.load_dataset("invoices")) == 3
        d.restore_previous("invoices")
        assert sorted(d.load_dataset("invoices")["invoice_id"]) == ["I1", "I2"]
        d.restore_previous("invoices")
        assert len(d.load_dataset("invoices")) == 3

    def test_plaintext_and_encrypted_tables_coexist(self, owner, monkeypatch):
        d.ingest(pd.DataFrame([inv("PLAIN")]), "invoices")
        monkeypatch.setenv("BIZIQ_ENCRYPT_DATA", "1")
        d.save_dataset(pd.DataFrame([dict(order_id="1", date=pd.Timestamp("2026-01-01"), customer_id="c", product="p", region="r", quantity=1.0, unit_price=2.0)]), "sales")
        monkeypatch.delenv("BIZIQ_ENCRYPT_DATA")
        assert d.load_dataset("invoices").invoice_id.iloc[0] == "PLAIN" and len(d.load_dataset("sales")) == 1

    def test_turning_encryption_on_without_a_key_refuses_to_store_plaintext(self, owner, monkeypatch):
        monkeypatch.setenv("BIZIQ_ENCRYPT_DATA", "1")
        monkeypatch.delenv("BIZIQ_MASTER_KEY")
        with pytest.raises(ValueError, match="cannot be encrypted"):
            d.ingest(pd.DataFrame([inv("I1")]), "invoices")

    def test_tampered_blob_is_an_error_not_empty_data(self, owner, monkeypatch):
        monkeypatch.setenv("BIZIQ_ENCRYPT_DATA", "1")
        d.ingest(pd.DataFrame([inv("I1")]), "invoices")
        with d._conn() as c:
            b = bytearray(bytes(c.execute("SELECT blob FROM invoices").fetchone()[0]))
            b[40] ^= 1
            c.execute("UPDATE invoices SET blob=?", (bytes(b),))
        with pytest.raises(crypto.CryptoError):
            d.load_dataset("invoices")

    def test_one_organisations_blob_cannot_be_planted_in_another(self, env, monkeypatch):
        monkeypatch.setenv("BIZIQ_ENCRYPT_DATA", "1")
        a = accounts.create_org("Alpha", "a@alpha.test", "Ann", PW)
        b = accounts.create_org("Beta", "b@beta.test", "Bob", PW)
        accounts.activate_org(a.org_id)
        d.ingest(pd.DataFrame([inv("A-ONLY")]), "invoices")
        with d._conn() as c:
            stolen = bytes(c.execute("SELECT blob FROM invoices").fetchone()[0])
        accounts.activate_org(b.org_id)
        d.ingest(pd.DataFrame([inv("B1")]), "invoices")
        with d._conn() as c:
            c.execute("UPDATE invoices SET blob=?", (stolen,))
        with pytest.raises(crypto.CryptoError):
            d.load_dataset("invoices")

    def test_data_key_rotation_via_reencrypt(self, owner, monkeypatch):
        monkeypatch.setenv("BIZIQ_ENCRYPT_DATA", "1")
        d.ingest(pd.DataFrame([inv("I1"), inv("I2")]), "invoices")
        scope = d.current_scope()
        crypto.new_key_version(scope)
        df = d.load_dataset("invoices")
        d.save_dataset(df, "invoices")                                   # re-write = encrypted under the new key
        with d._conn() as c:
            kid = int.from_bytes(bytes(c.execute("SELECT blob FROM invoices").fetchone()[0])[3:7], "big")
        assert kid == 2 and crypto.retire_old_versions(scope) == 1
        crypto.clear_cache()
        assert sorted(d.load_dataset("invoices")["invoice_id"]) == ["I1", "I2"]

    def test_large_dataset_is_chunked(self, owner, monkeypatch):
        monkeypatch.setenv("BIZIQ_ENCRYPT_DATA", "1")
        monkeypatch.setattr(d, "CHUNK_BYTES", 2000)
        import numpy as np
        big = pd.DataFrame({"invoice_id": [f"I{i}" for i in range(3000)], "party": "P", "type": "receivable", "invoice_date": pd.Timestamp("2026-01-01"),
                            "due_date": pd.Timestamp("2026-02-01"), "amount": np.arange(3000, dtype=float), "paid_date": pd.NaT})
        d.save_dataset(big, "invoices")
        with d._conn() as c:
            assert c.execute("SELECT COUNT(*) FROM invoices").fetchone()[0] > 3
        assert d.load_dataset("invoices").amount.sum() == big.amount.sum()


class TestSecretsAtRest:
    def test_connection_secrets_are_envelope_encrypted_and_bound_to_org_and_provider(self, env):
        from biziq.live import store
        store.save("o1", "shopify", {"shop": "s"}, {"token": "SHPAT-123"})
        raw = bytes(platform_db.connect().execute("SELECT secrets FROM connections").fetchone()[0])
        assert b"SHPAT" not in raw and crypto.is_encrypted(raw)
        assert store.load("o1", "shopify", ["token"])["token"] == "SHPAT-123"
        with platform_db.connect() as c:                                   # move o1's blob to org o2: must not decrypt there
            c.execute("INSERT INTO connections (org_id, provider, settings, secrets) VALUES ('o2','shopify','{}',?)", (raw,))
        with pytest.raises(crypto.CryptoError):
            store.load("o2", "shopify", ["token"])


# ------------------------------------------------------------------ TOTP
class TestTotp:
    SECRET = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"                          # ASCII "12345678901234567890" (RFC 6238 test key)

    @pytest.mark.parametrize("t,expected", [(59, "94287082"), (1111111109, "07081804"), (1111111111, "14050471"), (1234567890, "89005924"), (2000000000, "69279037")])
    def test_rfc6238_vectors(self, t, expected):
        assert totp.code_at(self.SECRET, t, 8) == expected

    def test_accepts_current_and_adjacent_steps_only(self):
        t = 1_700_000_000
        assert totp.verify(self.SECRET, totp.code_at(self.SECRET, t), at=t) is not None
        assert totp.verify(self.SECRET, totp.code_at(self.SECRET, t - 30), at=t) is not None           # one step of clock drift
        assert totp.verify(self.SECRET, totp.code_at(self.SECRET, t - 90), at=t) is None

    def test_a_used_code_cannot_be_replayed(self):
        t = 1_700_000_000
        code = totp.code_at(self.SECRET, t)
        step = totp.verify(self.SECRET, code, at=t)
        assert totp.verify(self.SECRET, code, last_step=step, at=t) is None
        assert totp.verify(self.SECRET, totp.code_at(self.SECRET, t + 30), last_step=step, at=t + 30) is not None

    @pytest.mark.parametrize("bad", ["", "12345", "1234567", "abcdef", None])
    def test_malformed_codes_rejected(self, bad):
        assert totp.verify(self.SECRET, bad) is None

    def test_provisioning_uri_and_recovery_codes(self):
        uri = totp.provisioning_uri("ABC", "a@b.test")
        assert uri.startswith("otpauth://totp/BizIQ:a%40b.test?secret=ABC") and "period=30" in uri
        codes = totp.new_recovery_codes(10)
        assert len(set(codes)) == 10 and all(len(c) == 14 for c in codes)
        assert totp.hash_recovery_code(codes[0]) == totp.hash_recovery_code(codes[0].upper().replace("-", " "))


# ------------------------------------------------------------------ two-factor login
def now_code(secret, shift=0):
    return totp.code_at(secret, __import__("time").time() + shift * 30)


class TestTwoFactorLogin:
    def _enroll(self, u):
        secret, uri = accounts.begin_2fa_enrollment(u)
        codes = accounts.confirm_2fa_enrollment(u, now_code(secret))
        return secret, codes

    def test_enrol_then_login_needs_the_code(self, owner):
        secret, codes = self._enroll(owner)
        assert accounts.two_factor_enabled(owner) and len(codes) == 10
        with pytest.raises(accounts.TwoFactorRequired):
            accounts.authenticate("owner@acme.test", PW)
        assert accounts.authenticate("owner@acme.test", PW, now_code(secret, 1)).email == "owner@acme.test"

    def test_secret_is_not_stored_in_plaintext(self, owner):
        secret, _ = self._enroll(owner)
        raw = platform_db.connect().execute("SELECT totp_secret FROM users WHERE id=?", (owner.id,)).fetchone()[0]
        assert secret.encode() not in bytes(raw) and crypto.is_encrypted(bytes(raw))

    def test_wrong_password_never_reveals_two_factor(self, owner):
        self._enroll(owner)
        assert accounts.authenticate("owner@acme.test", "Wrong-pass9") is None        # not TwoFactorRequired: no hint that the password was right

    def test_same_code_cannot_log_in_twice(self, owner):
        secret, _ = self._enroll(owner)
        c = now_code(secret, 1)
        assert accounts.authenticate("owner@acme.test", PW, c)
        with pytest.raises(accounts.AccountError, match="not valid or has already been used"):
            accounts.authenticate("owner@acme.test", PW, c)

    def test_wrong_codes_lock_the_account_like_wrong_passwords(self, owner):
        self._enroll(owner)
        for _ in range(accounts.MAX_FAILED):
            with pytest.raises(accounts.AccountError):
                accounts.authenticate("owner@acme.test", PW, "000000")
        with pytest.raises(accounts.AccountError, match="Too many failed"):
            accounts.authenticate("owner@acme.test", PW, "000000")

    def test_recovery_code_works_exactly_once(self, owner):
        secret, codes = self._enroll(owner)
        assert accounts.authenticate("owner@acme.test", PW, codes[0])
        with pytest.raises(accounts.AccountError):
            accounts.authenticate("owner@acme.test", PW, codes[0])
        assert accounts.authenticate("owner@acme.test", PW, codes[1])
        assert any(r == "2fa.recovery_used" for r in audit.read(owner.org_id)["action"])

    def test_wrong_confirmation_code_does_not_enable(self, owner):
        accounts.begin_2fa_enrollment(owner)
        with pytest.raises(accounts.AccountError):
            accounts.confirm_2fa_enrollment(owner, "000000")
        assert not accounts.two_factor_enabled(owner)
        assert accounts.authenticate("owner@acme.test", PW)                            # login unaffected

    def test_enrolment_needs_the_server_key(self, owner, monkeypatch):
        monkeypatch.delenv("BIZIQ_MASTER_KEY")
        with pytest.raises(accounts.AccountError, match="encryption key"):
            accounts.begin_2fa_enrollment(owner)

    def test_turning_off_needs_the_password_and_admin_can_reset_a_lost_phone(self, owner):
        self._enroll(owner)
        with pytest.raises(accounts.AccountError):
            accounts.disable_2fa(owner, password="nope")
        accounts.disable_2fa(owner, password=PW)
        assert not accounts.two_factor_enabled(owner)
        a = accounts.add_user(owner, "an@acme.test", "Analyst", "analyst", "An4lystPass!")
        s2, _ = accounts.begin_2fa_enrollment(a)
        accounts.confirm_2fa_enrollment(a, now_code(s2))
        accounts.disable_2fa(owner, "an@acme.test")                                    # owner resets: lost phone
        assert not accounts.two_factor_enabled(a)
        assert {"2fa.disabled", "2fa.reset"} <= set(audit.read(owner.org_id)["action"])

    def test_org_policy_forces_enrolment_and_cannot_be_bypassed(self, owner):
        with pytest.raises(accounts.AccountError, match="your own account first"):
            accounts.set_require_2fa(owner, True)
        self._enroll(owner)
        accounts.set_require_2fa(owner, True)
        a = accounts.add_user(owner, "an@acme.test", "Analyst", "analyst", "An4lystPass!")
        assert accounts.needs_2fa_enrollment(a)
        with pytest.raises(accounts.AccountError, match="requires two-factor"):
            accounts.disable_2fa(owner, password=PW)
        assert not accounts.needs_2fa_enrollment(owner)

    def test_analyst_cannot_change_policy_or_reset_others(self, owner):
        a = accounts.add_user(owner, "an@acme.test", "Analyst", "analyst", "An4lystPass!")
        with pytest.raises(accounts.AccountError):
            accounts.set_require_2fa(a, True)
        with pytest.raises(accounts.AccountError):
            accounts.disable_2fa(a, "owner@acme.test")


class TestSso:
    def test_only_verified_existing_users_get_in(self, owner):
        assert accounts.sso_user("owner@acme.test", True).role == "owner"
        assert accounts.sso_user("owner@acme.test", False) is None          # unverified email at the identity provider
        assert accounts.sso_user("stranger@acme.test", True) is None        # never auto-creates accounts
        assert accounts.sso_user("not-an-email", True) is None
        assert "login.sso" in set(audit.read(owner.org_id)["action"])

    def test_deactivated_user_cannot_use_sso(self, owner):
        a = accounts.add_user(owner, "an@acme.test", "Analyst", "analyst", "An4lystPass!")
        accounts.deactivate_user(owner, "an@acme.test")
        assert accounts.sso_user("an@acme.test", True) is None


# ------------------------------------------------------------------ password policy and sessions
class TestPasswordsAndSessions:
    @pytest.mark.parametrize("pw", ["short1", "allletterslong", "1234567890", "Password123", "aaaaaaaa11", "Passw0rd"])
    def test_weak_passwords_rejected(self, pw):
        with pytest.raises(accounts.AccountError):
            accounts.check_password_strength(pw)

    def test_password_containing_email_name_rejected(self):
        with pytest.raises(accounts.AccountError, match="email name"):
            accounts.check_password_strength("maria.santos2026", "maria.santos@x.test")

    def test_good_password_accepted(self):
        accounts.check_password_strength("Tr1cky-Horse-Staple")

    def test_session_idle_and_absolute_limits(self):
        t0 = session.now()
        assert session.expired(t0, t0, t0 + timedelta(minutes=29), 30, 12) is None
        assert "inactivity" in session.expired(t0, t0, t0 + timedelta(minutes=31), 30, 12)
        assert "at most 12" in session.expired(t0, t0 + timedelta(hours=11, minutes=59), t0 + timedelta(hours=12, minutes=1), 30, 12)
        assert session.expired(t0, t0, t0 + timedelta(days=9), 0, 0) is None            # 0 disables a limit


# ------------------------------------------------------------------ the real login screen
class TestLoginScreen:
    def test_two_factor_login_through_the_app(self, owner, monkeypatch):
        from streamlit.testing.v1 import AppTest
        import pathlib
        monkeypatch.setenv("BIZIQ_AUTH", "1")
        secret, uri = accounts.begin_2fa_enrollment(owner)
        accounts.confirm_2fa_enrollment(owner, now_code(secret))
        root = pathlib.Path(__file__).resolve().parent.parent
        at = AppTest.from_file(str(root / "app.py"), default_timeout=120)
        at.query_params["org"] = owner.org_slug
        at.run()
        assert not at.exception
        at.text_input[0].set_value("owner@acme.test"); at.text_input[1].set_value(PW)
        next(b for b in at.button if b.label == "Log in").click().run()
        assert any("authenticator" in w.value for w in at.warning) and "user" not in at.session_state       # password ok, code missing: not in
        at.text_input[0].set_value("owner@acme.test"); at.text_input[1].set_value(PW); at.text_input[2].set_value("000000")
        next(b for b in at.button if b.label == "Log in").click().run()
        assert at.error and "user" not in at.session_state                                                    # bad code: not in
        at.text_input[0].set_value("owner@acme.test"); at.text_input[1].set_value(PW); at.text_input[2].set_value(now_code(secret, 1))
        next(b for b in at.button if b.label == "Log in").click().run()
        assert not at.exception and at.session_state["user"].email == "owner@acme.test"

    def test_org_requiring_two_factor_shows_only_the_setup_page(self, owner, monkeypatch):
        from streamlit.testing.v1 import AppTest
        import pathlib
        monkeypatch.setenv("BIZIQ_AUTH", "1")
        secret, _ = accounts.begin_2fa_enrollment(owner)
        accounts.confirm_2fa_enrollment(owner, now_code(secret))
        accounts.set_require_2fa(owner, True)
        a = accounts.add_user(owner, "an@acme.test", "Analyst", "analyst", "An4lystPass!")
        root = pathlib.Path(__file__).resolve().parent.parent
        at = AppTest.from_file(str(root / "app.py"), default_timeout=120)
        at.query_params["org"] = owner.org_slug
        at.run()
        at.text_input[0].set_value("an@acme.test"); at.text_input[1].set_value("An4lystPass!")
        next(b for b in at.button if b.label == "Log in").click().run()
        assert at.session_state["must_enroll"] is True and not at.exception
        assert any("requires two-factor" in w.value for w in at.warning)
        assert not at.sidebar.radio                                         # no navigation until enrolled


# ------------------------------------------------------------------ the key-operations script
class TestRotateKeysScript:
    def _run(self, *args):
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location("rk", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "rotate_keys.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.main(list(args))

    def test_encrypt_existing_then_rotate_data_key_then_decrypt(self, owner, capsys, monkeypatch):
        d.ingest(pd.DataFrame([inv("I1", party=MARK)]), "invoices")
        d.ingest(pd.DataFrame([inv("I2", party=MARK)]), "invoices")                       # makes an undo copy too
        assert self._run("encrypt") == 0
        with d._conn() as c:
            assert d._is_encrypted_table(c, "invoices") and d._is_encrypted_table(c, "_prev_invoices")
        assert d.load_dataset("invoices").invoice_id.iloc[0] == "I2"
        d.restore_previous("invoices")                                                    # undo still works after conversion
        assert d.load_dataset("invoices").invoice_id.iloc[0] == "I1"
        assert self._run("data", "--org", owner.org_id) == 0
        assert d.load_dataset("invoices").party.iloc[0] == MARK
        assert self._run("status") == 0 and "datasets encrypted" in capsys.readouterr().out
        assert self._run("decrypt") == 0
        with d._conn() as c:
            assert not d._is_encrypted_table(c, "invoices")
        assert d.load_dataset("invoices").party.iloc[0] == MARK

    def test_refuses_without_a_key(self, env, monkeypatch, capsys):
        monkeypatch.delenv("BIZIQ_MASTER_KEY")
        assert self._run("status") == 2
