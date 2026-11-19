"""The screens a shop owner actually uses, driven through the real app: sign-in, billing, personalisation, evidence, export."""
import pathlib

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from biziq import accounts, billbook as bb, crypto, data_layer as d, platform_db

PW = "Passw0rdOK"
ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture()
def shop(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_db, "PLATFORM_DB", tmp_path / "platform.db")
    monkeypatch.setattr(platform_db, "OUT", tmp_path)
    monkeypatch.setenv("BIZIQ_MASTER_KEY", crypto.generate_master_key())
    monkeypatch.setenv("BIZIQ_AUTH", "1")
    monkeypatch.delenv("BIZIQ_REQUIRE_EMAIL_VERIFICATION", raising=False)
    monkeypatch.setattr("biziq.automation.OUTBOX", tmp_path / "outbox")
    crypto.clear_cache()
    u = accounts.create_org("Sharma Kirana", "owner@sharma.test", "Asha", PW, business_type="retail")
    accounts.activate_org(u.org_id)
    yield u
    d.use_db(None)
    crypto.clear_cache()


def login(u) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.query_params["org"] = u.org_slug
    at.run()
    at.text_input[0].set_value(u.email)
    at.text_input[1].set_value(PW)
    next(b for b in at.button if b.label == "Log in").click().run()
    assert not at.exception, at.exception
    if any(b.label == "Save and continue" for b in at.button):          # first sign-in: confirm what the business does
        next(b for b in at.button if b.label == "Save and continue").click().run()
        assert not at.exception, at.exception
    return at


def go(at, area, page):
    at.sidebar.selectbox[0].select(area).run()
    next(r for r in at.sidebar.radio if page in r.options).set_value(page).run()
    assert not at.exception, at.exception


class TestShopFlow:
    def test_retail_shop_lands_on_billing_with_no_data_loaded(self, shop):
        at = login(shop)
        assert at.sidebar.selectbox[0].value == "Billing"
        assert any("New bill" in o for r in at.sidebar.radio for o in r.options)

    def test_every_billing_page_renders_empty_and_filled(self, shop):
        at = login(shop)
        for page in ("New bill", "Day book", "Customers and dues", "Items and stock"):
            go(at, "Billing", page)
        bb.upsert_item("Rice 1kg", 100, cost_price=60, stock=50, reorder_level=10)
        bb.create_bill([dict(item="Rice 1kg", qty=2, rate=100)], customer="Ravi", paid=50, mode="cash")
        for page in ("New bill", "Day book", "Customers and dues", "Items and stock"):
            go(at, "Billing", page)
        go(at, "Billing", "Customers and dues")
        assert any("Ravi" in str(df.value) for df in at.dataframe)

    def test_hindi_toggle_switches_labels(self, shop):
        at = login(shop)
        go(at, "Billing", "Day book")
        at.session_state["lang_choice"] = "हिन्दी"
        at.run()
        assert not at.exception
        page_text = " ".join(m.value for m in at.markdown) + " ".join(str(getattr(w, "label", "")) for w in at.button)
        import unicodedata as ud
        assert ud.normalize("NFC", "रोज़ का हिसाब") in ud.normalize("NFC", page_text), page_text[-600:]

    def test_non_applicable_module_says_so_for_this_business(self, shop):
        at = login(shop)
        go(at, "Billing", "Day book")
        areas = at.sidebar.selectbox[0].options
        hit = None
        for a in areas:
            at.sidebar.selectbox[0].select(a).run()
            for r in at.sidebar.radio:
                if "RecipeIQ" in " ".join(r.options):
                    hit = next(o for o in r.options if "RecipeIQ" in o)
                    r.set_value(hit).run()
                    break
            if hit:
                break
        assert hit and "n/a" in hit
        assert any("Not applicable for your business model" in m.value for m in at.markdown)

    def test_bills_reach_the_analysis_after_sync(self, shop):
        bb.upsert_item("Soap", 40, cost_price=25, stock=20)
        for i in range(3):
            bb.create_bill([dict(item="Soap", qty=1, rate=40)], customer=f"C{i}")
        st = bb.sync_to_datasets()
        assert d.load_dataset("sales") is not None and len(d.load_dataset("sales")) == 3
        assert st


class TestExportAndEvidence:
    def test_export_zip_contains_datasets_billbook_and_manifest(self, shop):
        import io, zipfile
        from biziq import exporter
        bb.upsert_item("Soap", 40, cost_price=25, stock=20)
        bb.create_bill([dict(item="Soap", qty=1, rate=40)], customer="Ravi")
        z = zipfile.ZipFile(io.BytesIO(exporter.build_zip("Sharma Kirana")))
        names = z.namelist()
        assert any("README" in n for n in names) and any("manifest" in n.lower() for n in names)
        assert any("bb_" in n or "bill" in n.lower() for n in names)

    def test_export_blocks_spreadsheet_formula_injection(self, shop):
        import io, zipfile
        from biziq import exporter
        bb.upsert_item("=HYPERLINK(\"http://x\")", 10, stock=1)
        z = zipfile.ZipFile(io.BytesIO(exporter.build_zip("Sharma Kirana")))
        blob = b"".join(z.read(n) for n in z.namelist() if n.endswith(".csv")).decode("utf-8", "ignore")
        assert "\n=HYPERLINK" not in blob and ",=HYPERLINK" not in blob and ",\"=HYPERLINK" not in blob

    def test_evidence_levels_follow_the_data(self, shop):
        from biziq import evidence
        thin = pd.DataFrame(dict(invoice_id=["I1"], party=["P"], type=["receivable"], invoice_date=["2026-01-01"], due_date=["2026-02-01"], amount=[5], paid_date=[None]))
        ev = evidence.assess("ReceivablesIQ", {"invoices": thin}, None)
        assert ev is not None and ev.level in ("Low", "Medium")
        assert evidence.assess("ComplianceCalendarIQ", {}, None).level == "Rule-based"


class TestI18n:
    def test_every_label_exists_in_both_languages(self):
        from biziq import i18n
        for key, v in i18n.T.items():
            assert v.get("en") and v.get("hi"), key


class TestCounterBilling:
    def test_make_a_credit_bill_through_the_screen_then_see_the_due(self, shop):
        bb.upsert_item("Rice 1kg", 100, cost_price=60, stock=50)
        bb.upsert_customer("Ravi", "9876543210")                 # AppTest cannot carry a typed-in new option across reruns, so use a saved customer
        at = login(shop)
        go(at, "Billing", "New bill")
        at.selectbox(key="bb_customer").set_value("Ravi").run()
        at.selectbox(key="bb_item").set_value("Rice 1kg").run()
        next(b for b in at.button if b.label == "Add to bill").click().run()
        assert not at.exception, at.exception
        assert at.session_state["bb_cart"] == [dict(item="Rice 1kg", qty=1.0, rate=100.0)]
        at.session_state["bb_mode"] = "credit"
        at.run()
        next(b for b in at.button if b.key == "bb_save").click().run()
        assert not at.exception, at.exception
        assert any("B00001" in s.value for s in at.success)
        assert bb.credit_ledger() is not None
        assert float(bb.dues_table()["due"].sum()) == 100.0
        assert bb.list_items().set_index("name").loc["Rice 1kg", "stock"] == 49

    def test_credit_without_a_customer_is_refused_on_screen(self, shop):
        bb.upsert_item("Soap", 40, cost_price=25, stock=5)
        at = login(shop)
        go(at, "Billing", "New bill")
        at.selectbox(key="bb_item").set_value("Soap").run()
        next(b for b in at.button if b.label == "Add to bill").click().run()
        at.session_state["bb_mode"] = "credit"
        at.run()
        next(b for b in at.button if b.key == "bb_save").click().run()
        assert at.error and bb.dues_table().empty and bb.list_items().set_index("name").loc["Soap", "stock"] == 5


class TestConsultantPackPage:
    def test_page_renders_for_the_owner_with_people_unticked(self, shop):
        from biziq import pipeline
        d.load_sample_data()
        at = login(shop)
        go(at, "Overview", "Consultant pack")
        for c in at.checkbox:                                          # people sections, when offered at all, start unticked
            if "People sections" in c.label:
                assert not c.value
        assert any("Hide customer" in c.label for c in at.checkbox) and not next(c for c in at.checkbox if "Hide customer" in c.label).value
        assert any(b.label == "Prepare the pack" for b in at.button)

    def test_preparing_the_pack_is_audited_and_offers_both_files(self, shop):
        from biziq import audit
        d.load_sample_data()
        at = login(shop)
        go(at, "Overview", "Consultant pack")
        next(b for b in at.button if b.label == "Prepare the pack").click().run()
        assert not at.exception, at.exception
        assert at.session_state["pack_files"][0].startswith(b"%PDF")
        assert "report.consultant_pack" in set(audit.read(shop.org_id, 20)["action"])
