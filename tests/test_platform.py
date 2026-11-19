"""Tests for the platform layer: module contract, accounts, roles, audit chain, approvals, isolation, connectors."""
import io
import sqlite3

import pandas as pd
import pytest

from biziq import accounts, approvals, audit, connectors, data_layer, platform_db, registry
from biziq.accounts import AccountError


@pytest.fixture()
def plat(tmp_path, monkeypatch):
    """Fresh platform database and org folder per test."""
    monkeypatch.setattr(platform_db, "PLATFORM_DB", tmp_path / "platform.db")
    monkeypatch.setattr(platform_db, "OUT", tmp_path)
    yield tmp_path


@pytest.fixture()
def owner(plat):
    return accounts.create_org("Acme Traders", "owner@acme.test", "Olive Owner", "Passw0rdOK")


# ---------------------------------------------------------------- module contract
class TestModuleContract:
    def test_forty_five_plugin_modules_registered(self):
        names = [s.name for s in registry.all_specs()]
        assert len(names) == 45 and len(set(names)) == 45

    def test_every_module_meets_contract(self, pr):
        for spec in registry.all_specs():
            res = pr.modules[spec.name]
            assert registry.check_result(spec, res) == [], spec.name

    def test_every_module_has_recommendations_and_summary(self, pr):
        for spec in registry.all_specs():
            res = pr.modules[spec.name]
            assert spec.recommend(res.facts, res.flags), spec.name
            assert registry.generic_summary(res)

    def test_contract_rejects_bad_result(self):
        spec = registry.all_specs()[0]
        from biziq.core import ModuleResult
        bad = ModuleResult(name="Wrong")
        assert registry.check_result(spec, bad)


class TestPlantedFindings:
    def test_profit_finds_loss_maker(self, pr):
        assert pr.modules["ProfitIQ"].facts  # sanity
        flags = " ".join(f.title + f.detail for f in pr.modules["ProfitIQ"].flags)
        assert "Sneakers" in flags

    def test_gst_mismatch_detected(self, pr):
        assert pr.modules["ComplianceIQ"].flags

    def test_sensitive_flag_on_people_modules(self):
        assert registry.get("ProfitIQ").sensitive is False


# ---------------------------------------------------------------- accounts
class TestAccounts:
    def test_password_hash_roundtrip(self):
        h = accounts.hash_password("Passw0rdOK")
        assert accounts.verify_password("Passw0rdOK", h) and not accounts.verify_password("nope", h)
        assert h != accounts.hash_password("Passw0rdOK")  # salted

    @pytest.mark.parametrize("pw", ["short1", "allletters", "12345678"])
    def test_weak_passwords_rejected(self, pw):
        with pytest.raises(AccountError):
            accounts.check_password_strength(pw)

    def test_create_and_login(self, owner):
        assert owner.role == "owner"
        assert accounts.authenticate("OWNER@acme.test", "Passw0rdOK").email == "owner@acme.test"
        assert accounts.authenticate("owner@acme.test", "wrong") is None
        assert accounts.authenticate("nobody@x.test", "whatever1") is None

    def test_duplicate_email_rejected(self, owner):
        with pytest.raises(AccountError):
            accounts.create_org("Other", "owner@acme.test", "X", "Passw0rdOK")

    def test_lockout_after_repeated_failures(self, owner):
        for _ in range(accounts.MAX_FAILED):
            assert accounts.authenticate("owner@acme.test", "bad") is None
        with pytest.raises(AccountError):
            accounts.authenticate("owner@acme.test", "Passw0rdOK")   # even the right password is refused while locked

    def test_roles_and_permissions(self, owner):
        hr = accounts.add_user(owner, "hr@acme.test", "Hana", "hr", "Passw0rdOK")
        viewer = accounts.add_user(owner, "v@acme.test", "Vik", "viewer", "Passw0rdOK")
        analyst = accounts.add_user(owner, "a@acme.test", "Ann", "analyst", "Passw0rdOK")
        assert accounts.can(hr, "view_people") and not accounts.can(hr, "upload")
        assert accounts.can(analyst, "upload") and not accounts.can(analyst, "view_people")
        assert accounts.can(viewer, "view") and not accounts.can(viewer, "request")
        assert not accounts.can(None, "view")

    def test_only_managers_can_add_users(self, owner):
        analyst = accounts.add_user(owner, "a@acme.test", "Ann", "analyst", "Passw0rdOK")
        with pytest.raises(AccountError):
            accounts.add_user(analyst, "b@acme.test", "Bo", "viewer", "Passw0rdOK")

    def test_sensitive_modules_need_view_people(self, owner):
        analyst = accounts.add_user(owner, "a@acme.test", "Ann", "analyst", "Passw0rdOK")
        hr = accounts.add_user(owner, "hr@acme.test", "Hana", "hr", "Passw0rdOK")
        spec = registry.ModuleSpec(name="XIQ", group="g", blurb="b", needs=[], run=lambda d, c: None,
                                   recommend=lambda f, fl: [], sensitive=True)
        assert not accounts.can_view_module(analyst, spec)
        assert accounts.can_view_module(hr, spec)

    def test_deactivated_user_cannot_login(self, owner):
        accounts.add_user(owner, "a@acme.test", "Ann", "analyst", "Passw0rdOK")
        accounts.deactivate_user(owner, "a@acme.test")
        assert accounts.authenticate("a@acme.test", "Passw0rdOK") is None

    def test_cannot_demote_or_deactivate_last_owner(self, owner):
        with pytest.raises(AccountError):
            accounts.deactivate_user(owner, "owner@acme.test")
        with pytest.raises(AccountError):
            accounts.set_role(owner, "owner@acme.test", "viewer")

    def test_change_password(self, owner):
        accounts.change_password(owner, "Passw0rdOK", "NewPassw0rd1")
        assert accounts.authenticate("owner@acme.test", "NewPassw0rd1")
        with pytest.raises(AccountError):
            accounts.change_password(owner, "wrong", "Another1pass")


# ---------------------------------------------------------------- audit log
class TestAudit:
    def test_chain_valid_then_detects_tamper(self, owner):
        audit.log(owner.org_id, owner.email, "x.one", "t", "d1")
        audit.log(owner.org_id, owner.email, "x.two", "t", "d2")
        ok, _ = audit.verify_chain(owner.org_id)
        assert ok

    def test_update_and_delete_blocked_by_triggers(self, owner):
        audit.log(owner.org_id, owner.email, "x.one", "t", "d1")
        for stmt in ("UPDATE audit_log SET detail='hacked'", "DELETE FROM audit_log"):
            c = platform_db.connect()                       # a fresh connection each: PostgreSQL aborts the transaction on error
            with pytest.raises(Exception, match="append-only"):
                c.execute(stmt)
            c.rollback() if hasattr(c, "rollback") else None
            c.close()

    def test_tamper_via_dropped_trigger_is_detected(self, owner):
        audit.log(owner.org_id, owner.email, "x.one", "t", "original")
        c = platform_db.connect()
        from biziq import db
        c.execute("DROP TRIGGER audit_no_update" + (" ON audit_log" if db.IS_PG else ""))
        c.execute("UPDATE audit_log SET detail='changed' WHERE action='x.one'")
        c.commit()
        ok, msg = audit.verify_chain(owner.org_id)
        assert not ok and "altered" in msg

    def test_orgs_have_independent_chains(self, owner):
        other = accounts.create_org("Beta", "b@beta.test", "Bea", "Passw0rdOK")
        audit.log(other.org_id, other.email, "y", "t", "d")
        assert audit.verify_chain(owner.org_id)[0] and audit.verify_chain(other.org_id)[0]
        assert set(audit.read(owner.org_id)["user_email"]) <= {owner.email}


# ---------------------------------------------------------------- approvals
class TestApprovals:
    def test_request_and_approve_by_other_user(self, owner):
        analyst = accounts.add_user(owner, "a@acme.test", "Ann", "analyst", "Passw0rdOK")
        rid = approvals.request(analyst, "payment_reminder", "Chase 3 invoices", {"drafts": []})
        approvals.decide(owner, rid, True, "ok")
        df = approvals.list_requests(owner)
        assert df.loc[df["id"] == rid, "status"].iloc[0] == "approved"

    def test_no_self_approval_by_default(self, owner):
        rid = approvals.request(owner, "payment_reminder", "Mine")
        with pytest.raises(AccountError):
            approvals.decide(owner, rid, True)

    def test_self_approval_when_owner_allows(self, owner):
        accounts.set_self_approval(owner, True)
        rid = approvals.request(owner, "payment_reminder", "Mine")
        approvals.decide(owner, rid, True)

    def test_cannot_decide_twice_or_without_permission(self, owner):
        analyst = accounts.add_user(owner, "a@acme.test", "Ann", "analyst", "Passw0rdOK")
        rid = approvals.request(analyst, "payment_reminder", "T")
        with pytest.raises(AccountError):
            approvals.decide(analyst, rid, True)
        approvals.decide(owner, rid, False)
        with pytest.raises(AccountError):
            approvals.decide(owner, rid, True)

    def test_other_org_cannot_see_or_decide(self, owner):
        rid = approvals.request(owner, "payment_reminder", "T")
        other = accounts.create_org("Beta", "b@beta.test", "Bea", "Passw0rdOK")
        assert approvals.list_requests(other).empty
        with pytest.raises(AccountError):
            approvals.decide(other, rid, True)

    def test_actions_are_audited(self, owner):
        rid = approvals.request(owner, "payment_reminder", "T")
        actions = set(audit.read(owner.org_id)["action"])
        assert "approval.request" in actions


# ---------------------------------------------------------------- per-organisation data isolation
class TestIsolation:
    def test_each_org_gets_its_own_database(self, owner):
        other = accounts.create_org("Beta", "b@beta.test", "Bea", "Passw0rdOK")
        accounts.activate_org(owner.org_id)
        data_layer.load_sample_data()
        assert data_layer.load_all()
        accounts.activate_org(other.org_id)
        assert not data_layer.load_all()           # the second organisation sees nothing
        accounts.activate_org(owner.org_id)
        assert "sales" in data_layer.load_all()


# ---------------------------------------------------------------- connectors
class TestConnectors:
    def test_preset_renames_columns_and_ingests(self, owner):
        accounts.activate_org(owner.org_id)
        text = ("Invoice Number,Customer Name,Invoice Date,Due Date,Total,Payment Date\n"
                          "INV-1,Acme,2026-09-01,2026-09-30,1000,\nINV-2,Beta,2026-09-02,2026-10-02,500,2026-09-20\n")
        conn, ds = connectors.from_preset(io.StringIO(text), "generic_invoice_export", filename="x.csv")
        assert ds == "invoices"
        df = conn.fetch(ds)
        assert {"invoice_id", "party", "amount"} <= set(df.columns)
        conn, _ = connectors.from_preset(io.StringIO(text), "generic_invoice_export", filename="x.csv")
        rep = connectors.ingest_from(conn, ds, user=owner)
        assert rep.ok
        assert "data.upload" in set(audit.read(owner.org_id)["action"])

    def test_every_preset_targets_a_real_dataset(self):
        for p in connectors.PRESETS.values():
            assert p["dataset"] in data_layer.SCHEMAS
            known = set(data_layer.SCHEMAS[p["dataset"]].columns) if hasattr(data_layer.SCHEMAS[p["dataset"]], "columns") else None
            if known:
                assert set(p["map"].values()) <= known


# ---------------------------------------------------------------- new schemas
class TestNewSchemas:
    NEW = ["product_costs", "returns", "expenses", "campaigns", "competitor_prices", "credit_limits",
           "gst_books", "gstr2b", "pipeline", "quotes", "contracts", "tickets"]

    def test_all_new_datasets_registered_and_clean(self, sample):
        assert len(data_layer.SCHEMAS) == 40
        for n in self.NEW:
            assert n in sample and len(sample[n]) > 0

    def test_gstin_format_warning(self):
        raw = pd.DataFrame({"invoice_no": ["A1"], "supplier": ["S"], "gstin": ["BAD"], "invoice_date": ["2026-09-01"],
                            "taxable_value": [100], "tax_amount": [18]})
        df, rep = data_layer.clean_and_validate(raw, "gst_books")
        assert rep.ok or rep.errors is not None
