"""Sign-up, per-business sign-in, verification, reset, invites, profile personalisation and data persistence."""
import pandas as pd
import pytest

from biziq import accounts, crypto, data_layer as d, mailer, platform_db, profile as pf, registry
from biziq.accounts import AccountError

PW = "Passw0rdOK"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_db, "PLATFORM_DB", tmp_path / "platform.db")
    monkeypatch.setattr(platform_db, "OUT", tmp_path)
    monkeypatch.setenv("BIZIQ_MASTER_KEY", crypto.generate_master_key())
    monkeypatch.delenv("BIZIQ_REQUIRE_EMAIL_VERIFICATION", raising=False)
    monkeypatch.setattr("biziq.automation.OUTBOX", tmp_path / "outbox")
    crypto.clear_cache()
    yield tmp_path
    d.use_db(None)
    crypto.clear_cache()


def last_mail(env):
    files = sorted((env / "outbox").glob("*.eml"), key=lambda p: p.stat().st_mtime)
    if not files:
        return ""
    import email
    from email import policy
    m = email.message_from_bytes(files[-1].read_bytes(), policy=policy.default)
    return f"{m['Subject']}\n{m.get_body(preferencelist=('plain',)).get_content()}"


def token_from(text, key):
    import re
    return re.search(rf"{key}=([\w\-]+)", text).group(1)


@pytest.fixture()
def smtp(env, monkeypatch):
    """A fake SMTP server: messages go through the real send path and land in the outbox folder."""
    import smtplib

    class Fake:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def send_message(self, msg):
            (env / "outbox").mkdir(exist_ok=True)
            import time
            (env / "outbox" / f"{time.time_ns()}.eml").write_bytes(bytes(msg))
    monkeypatch.setattr(smtplib, "SMTP", Fake)
    for k, v in dict(BIZIQ_SEND_EMAIL="1", BIZIQ_SMTP_HOST="smtp.test", BIZIQ_EMAIL_FROM="biziq@test.test", BIZIQ_EMAIL_TO="x@test.test").items():
        monkeypatch.setenv(k, v)


class TestSignup:
    def test_signup_gives_a_business_login_slug_and_profile(self, env):
        r = accounts.signup("Acme Traders", "Asha", "asha@acme.test", PW, "retail")
        assert r.user and r.user.org_slug.startswith("acme-traders")
        org = accounts.org_by_slug(r.user.org_slug)
        assert org["name"] == "Acme Traders" and org["business_type"] == "retail"
        assert accounts.get_profile(r.user.org_id).confirmed is False      # onboarding still to do

    def test_slug_rules(self, env):
        for bad in ("ab", "admin", "x" * 40):
            with pytest.raises(AccountError):
                accounts.signup("Co", "N", f"{bad}@x.test", PW, "retail", slug=bad)
        accounts.signup("Co One", "N", "one@x.test", PW, "retail", slug="shared-name")
        with pytest.raises(AccountError, match="taken"):
            accounts.signup("Co Two", "N", "two@x.test", PW, "retail", slug="shared-name")

    def test_unknown_business_type_and_weak_password(self, env):
        with pytest.raises(AccountError):
            accounts.signup("Co", "N", "a@x.test", PW, "spaceship")
        with pytest.raises(AccountError):
            accounts.signup("Co", "N", "a@x.test", "short", "retail")

    def test_signup_rate_limited_per_email(self, env):
        for _ in range(5):
            try:
                accounts.signup("Co", "N", "same@x.test", "weak", "retail")
            except AccountError:
                pass
        with pytest.raises(AccountError, match="Too many"):
            accounts.signup("Co", "N", "same@x.test", PW, "retail")

    def test_verification_flow_and_anti_enumeration(self, env, smtp, monkeypatch):
        monkeypatch.setenv("BIZIQ_REQUIRE_EMAIL_VERIFICATION", "1")
        r1 = accounts.signup("Acme", "Asha", "asha@acme.test", PW, "retail")
        assert r1.user is None and r1.needs_verification
        mail = last_mail(env)
        assert "Confirm your BizIQ account" in mail
        with pytest.raises(AccountError, match="verify"):
            accounts.authenticate("asha@acme.test", PW)                  # locked until confirmed
        # same email again: identical on-screen answer, the real owner is told by email instead
        r2 = accounts.signup("Other", "Someone", "asha@acme.test", PW, "retail")
        assert r2.message == r1.message and r2.user is None
        assert "already has an account" in last_mail(env)
        u = accounts.verify_email(token_from(mail, "verify"))
        assert u and accounts.authenticate("asha@acme.test", PW).email == "asha@acme.test"
        assert accounts.verify_email(token_from(mail, "verify")) is None   # single use

    def test_verification_needs_working_email(self, env, monkeypatch):
        monkeypatch.setenv("BIZIQ_REQUIRE_EMAIL_VERIFICATION", "1")
        with pytest.raises(AccountError, match="SMTP"):
            accounts.signup("Acme", "Asha", "asha@acme.test", PW, "retail")


class TestLoginPerBusiness:
    def test_login_only_works_on_own_business_page(self, env):
        a = accounts.signup("Alpha", "A", "a@x.test", PW, "retail").user
        b = accounts.signup("Beta", "B", "b@x.test", PW, "food").user
        assert accounts.authenticate("a@x.test", PW, org_slug=a.org_slug).org_id == a.org_id
        assert accounts.authenticate("a@x.test", PW, org_slug=b.org_slug) is None      # same failure as a wrong password
        assert accounts.authenticate("a@x.test", PW).org_id == a.org_id               # central login still works
        assert accounts.org_by_slug("nope-nope") is None
        assert accounts.org_by_slug("'; drop table orgs;--") is None

    def test_public_page_data_has_no_people_or_data(self, env):
        u = accounts.signup("Alpha", "A", "a@x.test", PW, "retail").user
        assert set(accounts.org_by_slug(u.org_slug)) == {"id", "name", "slug", "accent", "tagline", "business_type"}

    def test_data_is_still_there_next_login(self, env):
        u = accounts.signup("Alpha", "A", "a@x.test", PW, "retail").user
        accounts.activate_org(u.org_id)
        d.save_dataset(pd.DataFrame([dict(invoice_id="1", party="P", type="receivable", invoice_date="2026-01-01", due_date="2026-02-01", amount=5, paid_date=None)]), "invoices")
        d.use_db(None)                                                                 # "logout"
        again = accounts.authenticate("a@x.test", PW, org_slug=u.org_slug)
        accounts.activate_org(again.org_id)
        assert len(d.load_all()["invoices"]) == 1                                      # no re-upload
        other = accounts.signup("Beta", "B", "b@x.test", PW, "food").user
        accounts.activate_org(other.org_id)
        assert "invoices" not in d.load_all()                                          # and never another business's


class TestResetAndInvite:
    def test_reset_flow_is_single_use_and_generic(self, env):
        accounts.signup("Alpha", "A", "a@x.test", PW, "retail")
        m1 = accounts.request_password_reset("a@x.test")
        mail = last_mail(env)
        m2 = accounts.request_password_reset("ghost@x.test")
        assert m1 == m2                                                                # no account discovery
        tok = token_from(mail, "reset")
        with pytest.raises(AccountError):
            accounts.reset_password(tok, "weak")
        accounts.reset_password(tok, "N3wPassword1")
        assert accounts.authenticate("a@x.test", "N3wPassword1")
        with pytest.raises(AccountError):
            accounts.reset_password(tok, "An0therPass1")                               # used

    def test_newer_reset_link_replaces_older(self, env):
        accounts.signup("Alpha", "A", "a@x.test", PW, "retail")
        accounts.request_password_reset("a@x.test"); t1 = token_from(last_mail(env), "reset")
        accounts.request_password_reset("a@x.test"); t2 = token_from(last_mail(env), "reset")
        with pytest.raises(AccountError):
            accounts.reset_password(t1, "N3wPassword1")
        accounts.reset_password(t2, "N3wPassword1")

    def test_invite_accept_and_permissions(self, env):
        o = accounts.signup("Alpha", "A", "a@x.test", PW, "retail").user
        link = accounts.invite_user(o, "n@x.test", "Nia", "analyst")
        tok = token_from(link, "invite")
        assert accounts.invite_info(tok)["org_name"] == "Alpha"
        n = accounts.accept_invite(tok, "Nia1Password", "Nia")
        assert n.role == "analyst" and n.org_id == o.org_id and n.org_slug == o.org_slug
        with pytest.raises(AccountError):
            accounts.accept_invite(tok, "Nia1Password")
        with pytest.raises(AccountError):
            accounts.invite_user(n, "z@x.test", "Z", "viewer")                         # analysts cannot invite
        with pytest.raises(AccountError):
            accounts.invite_user(o, "x@x.test", "X", "owner")

    def test_tokens_are_stored_hashed(self, env):
        accounts.signup("Alpha", "A", "a@x.test", PW, "retail")
        accounts.request_password_reset("a@x.test")
        tok = token_from(last_mail(env), "reset")
        with accounts._conn() as c:
            stored = [r[0] for r in c.execute("SELECT token_hash FROM auth_tokens").fetchall()]
        assert stored and tok not in stored


class TestProfile:
    def test_every_registered_module_is_classified(self, env):
        names = {s.name for s in registry.all_specs()} if hasattr(registry, "all_specs") else set(registry.REGISTRY)
        assert names <= set(pf.MODULE_CAPS), f"unclassified: {names - set(pf.MODULE_CAPS)}"

    def test_retail_does_not_get_recipe_but_food_does(self, env):
        retail = pf.Profile("retail", pf.defaults_for("retail"))
        food = pf.Profile("food", pf.defaults_for("food"))
        assert not pf.applies("RecipeIQ", retail) and pf.applies("RecipeIQ", food)
        assert "Not applicable for your business model" in pf.why_not("RecipeIQ", retail)
        assert pf.applies("SalesIQ", retail) and pf.applies("SalesIQ", food)           # universal modules always apply

    def test_other_shows_everything(self, env):
        p = pf.Profile("other", pf.defaults_for("other"))
        assert all(pf.applies(n, p) for n in pf.MODULE_CAPS)

    def test_owner_can_override_and_viewer_cannot(self, env):
        o = accounts.signup("Alpha", "A", "a@x.test", PW, "retail").user
        v = accounts.add_user(o, "v@x.test", "V", "viewer", "Vi3werPass1")
        accounts.set_module_override(o, "RecipeIQ", True)
        assert pf.applies("RecipeIQ", accounts.get_profile(o.org_id))
        with pytest.raises(AccountError):
            accounts.set_module_override(v, "RecipeIQ", False)
        accounts.set_module_override(o, "RecipeIQ", False)
        assert not pf.applies("RecipeIQ", accounts.get_profile(o.org_id))
        with pytest.raises(AccountError):
            accounts.set_module_override(o, "NotAModule", True)

    def test_set_profile_validates_and_confirms(self, env):
        o = accounts.signup("Alpha", "A", "a@x.test", PW, "retail").user
        p = accounts.set_profile(o, "food", ["food", "bogus"], "#12ab34", "Fresh daily")
        assert p.confirmed and p.caps == frozenset({"food"}) and p.accent == "#12ab34"
        p = accounts.set_profile(o, "food", ["food"], "red;}</style><script>")
        assert p.accent != "red;}</style><script>"                                     # accent goes into CSS: must be sanitised
        with pytest.raises(AccountError):
            accounts.set_profile(o, "spaceship", [])

    def test_profiles_are_per_business(self, env):
        a = accounts.signup("Alpha", "A", "a@x.test", PW, "retail").user
        b = accounts.signup("Beta", "B", "b@x.test", PW, "food").user
        accounts.set_module_override(a, "RecipeIQ", True)
        assert not pf.applies("RecipeIQ", pf.Profile("food", frozenset()))            # sanity
        assert "RecipeIQ" not in accounts.get_profile(b.org_id).overrides
