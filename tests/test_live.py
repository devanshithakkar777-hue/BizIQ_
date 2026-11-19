"""Live integrations tested against SIMULATED vendor responses (no network, no real accounts)."""
import io
import json

import pandas as pd
import pytest
from cryptography.fernet import Fernet

from biziq import accounts, approvals, audit, data_layer, platform_db
from biziq.live import store, sync
from biziq.live.http import Http, HttpError
from biziq.live.shopify import ShopifyConnector
from biziq.live.tally import TallyConnector, TallyXmlFileConnector, parse_vouchers
from biziq.live.whatsapp import WhatsAppSender, dispatch_approved_reminders, load_contacts, normalise_phone, save_contacts, send_log
from biziq.live.zoho import ZohoBooksConnector


class Fake:
    """Transport that returns queued (status, body) pairs per URL fragment and records every call."""
    def __init__(self, routes):
        self.routes, self.calls = {k: list(v) for k, v in routes.items()}, []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append(dict(method=method, url=url, headers=headers, body=body))
        for frag, queue in self.routes.items():
            if frag in url:
                status, payload = queue.pop(0) if len(queue) > 1 else queue[0]
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                return status, {}, raw
        return 404, {}, b"no route"


def http(routes):
    f = Fake(routes)
    return Http(transport=f, sleep=lambda s: None), f


@pytest.fixture()
def plat(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_db, "PLATFORM_DB", tmp_path / "platform.db")
    monkeypatch.setattr(platform_db, "OUT", tmp_path)
    monkeypatch.delenv("BIZIQ_SECRET_KEY", raising=False)
    return tmp_path


@pytest.fixture()
def owner(plat):
    u = accounts.create_org("Acme", "o@acme.test", "Olive", "Passw0rdOK")
    accounts.activate_org(u.org_id)
    return u


# ------------------------------------------------------------------ http client
class TestHttp:
    def test_retries_then_succeeds(self):
        h, f = http({"x.test": [(503, {}), (429, {}), (200, {"ok": 1})]})
        assert h.request("GET", "https://x.test/a").json() == {"ok": 1} and len(f.calls) == 3

    def test_gives_up_after_retries(self):
        h, _ = http({"x.test": [(500, b"boom")]})
        with pytest.raises(HttpError):
            h.request("GET", "https://x.test/a")

    def test_redirect_is_refused_not_followed(self):
        h, _ = http({"x.test": [(302, b"")]})
        with pytest.raises(HttpError, match="redirect"):
            h.request("GET", "https://x.test/a")

    def test_only_http_schemes(self):
        h, _ = http({})
        with pytest.raises(HttpError):
            h.request("GET", "file:///etc/passwd")

    def test_error_text_never_contains_token(self):
        h, _ = http({"x.test": [(401, {"error": "bad"})]})
        with pytest.raises(HttpError) as e:
            h.request("GET", "https://x.test/a", headers={"Authorization": "Bearer SECRET123"})
        assert "SECRET123" not in str(e.value) and "Credentials" in str(e.value)


# ------------------------------------------------------------------ shopify
def order(name, cancelled=False, items=1, more=False):
    return {"name": name, "createdAt": "2026-10-05T10:00:00Z", "cancelledAt": "2026-10-06" if cancelled else None,
            "customer": {"id": "gid://shopify/Customer/777"}, "shippingAddress": {"provinceCode": "MH", "province": "Maharashtra"},
            "lineItems": {"pageInfo": {"hasNextPage": more}, "nodes": [
                {"title": f"Tee {i}", "quantity": 2, "discountedUnitPriceSet": {"shopMoney": {"amount": "499.50"}},
                 "product": {"productType": "Apparel"}} for i in range(items)]}}


def page(nodes, nxt=None):
    return {"data": {"orders": {"pageInfo": {"hasNextPage": bool(nxt), "endCursor": nxt}, "nodes": nodes}}}


class TestShopify:
    @pytest.mark.parametrize("bad", ["evil.com", "x.myshopify.com.evil.com", "", "http://evil.com/x.myshopify.com", "a b.myshopify.com"])
    def test_rejects_non_shopify_hosts(self, bad):
        with pytest.raises(ValueError):
            ShopifyConnector(bad, "tok")

    def test_fetches_pages_skips_cancelled_and_validates(self):
        h, f = http({"graphql.json": [(200, page([order("#1001"), order("#1002", cancelled=True)], "c1")),
                                      (200, page([order("#1003", items=2, more=True)]))]})
        c = ShopifyConnector("My-Store.myshopify.com", "shpat_TOKEN", http=h)
        df = c.fetch("sales")
        assert list(df["order_id"]) == ["#1001", "#1003", "#1003"] and len(f.calls) == 2
        assert f.calls[0]["headers"]["X-Shopify-Access-Token"] == "shpat_TOKEN"
        assert "my-store.myshopify.com/admin/api/" in f.calls[0]["url"]
        assert json.loads(f.calls[1]["body"])["variables"]["after"] == "c1"
        assert any("cancelled" in n for n in c.notes) and any("100 line items" in n for n in c.notes)
        clean, rep = data_layer.clean_and_validate(df, "sales")
        assert rep.ok and clean["customer_id"].iloc[0] == "777" and clean["region"].iloc[0] == "Maharashtra"

    def test_retries_when_throttled(self):
        throttled = {"errors": [{"message": "Throttled", "extensions": {"code": "THROTTLED"}}]}
        h, f = http({"graphql.json": [(200, throttled), (200, page([order("#1")]))]})
        c = ShopifyConnector("s.myshopify.com", "t", http=h, sleep=lambda s: None)
        assert len(c.fetch("sales")) == 1 and len(f.calls) == 2

    def test_graphql_error_is_reported(self):
        h, _ = http({"graphql.json": [(200, {"errors": [{"message": "Access denied for orders field"}]})]})
        with pytest.raises(HttpError, match="Access denied"):
            ShopifyConnector("s.myshopify.com", "t", http=h).fetch("sales")


# ------------------------------------------------------------------ zoho
def zoho_http(invoices, bills):
    return http({"oauth/v2/token": [(200, {"access_token": "AT1", "expires_in": 3600})],
                 "/books/v3/invoices": invoices, "/books/v3/bills": bills,
                 "/books/v3/organizations": [(200, {"code": 0, "organizations": [{"organization_id": "999", "name": "Acme Pvt"}]})]})


def zc(h, **k):
    return ZohoBooksConnector("in", "999", "cid", "csec", "rtok", http=h, **k)


class TestZoho:
    def test_rejects_unknown_data_centre(self):
        with pytest.raises(ValueError):
            ZohoBooksConnector("evil.com/", "1", "a", "b", "c")

    def test_maps_invoices_and_bills_and_handles_paid_and_partial(self):
        inv = {"code": 0, "page_context": {"has_more_page": False}, "invoices": [
            {"invoice_number": "INV-1", "customer_name": "Acme", "date": "2026-09-01", "due_date": "2026-09-30", "total": 1000, "balance": 1000, "status": "overdue"},
            {"invoice_number": "INV-2", "customer_name": "Beta", "date": "2026-09-02", "due_date": "2026-10-02", "total": 500, "balance": 0, "status": "paid", "last_payment_date": "2026-09-20"},
            {"invoice_number": "INV-3", "customer_name": "Gamma", "date": "2026-09-03", "due_date": "2026-10-03", "total": 800, "balance": 300, "status": "partially_paid"},
            {"invoice_number": "INV-4", "customer_name": "Delta", "date": "2026-09-04", "due_date": "2026-10-04", "total": 700, "balance": 0, "status": "paid"},
            {"invoice_number": "INV-5", "customer_name": "Draft", "date": "2026-09-05", "due_date": "2026-10-05", "total": 1, "balance": 1, "status": "draft"}]}
        bills = {"code": 0, "page_context": {"has_more_page": False}, "bills": [
            {"bill_number": "B-1", "vendor_name": "Supplier", "date": "2026-09-10", "due_date": "2026-10-10", "total": 2000, "balance": 2000, "status": "open"}]}
        h, f = zoho_http([(200, inv)], [(200, bills)])
        c = zc(h)
        df = c.fetch("invoices")
        assert set(df["invoice_id"]) == {"INV-1", "INV-2", "INV-3", "B-1"}          # draft and undated-paid left out
        assert df.set_index("invoice_id").loc["INV-3", "amount"] == 300              # outstanding only
        assert df.set_index("invoice_id").loc["B-1", "type"] == "payable"
        assert any("no payment date" in n for n in c.notes) and any("part-paid" in n for n in c.notes)
        api = [x for x in f.calls if "/books/v3/invoices" in x["url"]][0]
        assert api["headers"]["Authorization"] == "Zoho-oauthtoken AT1" and "organization_id=999" in api["url"]
        assert "zohoapis.in" in api["url"]
        assert data_layer.clean_and_validate(df, "invoices")[1].ok

    def test_pagination_and_token_cached(self):
        p1 = {"code": 0, "page_context": {"has_more_page": True}, "invoices": [{"invoice_number": "A", "customer_name": "x", "date": "2026-09-01", "due_date": "2026-09-30", "total": 10, "balance": 10, "status": "sent"}]}
        p2 = {"code": 0, "page_context": {"has_more_page": False}, "invoices": [{"invoice_number": "B", "customer_name": "x", "date": "2026-09-01", "due_date": "2026-09-30", "total": 10, "balance": 10, "status": "sent"}]}
        h, f = zoho_http([(200, p1), (200, p2)], [(200, {"code": 0, "bills": [], "page_context": {"has_more_page": False}})])
        zc(h).fetch("invoices")
        assert sum("oauth/v2/token" in c["url"] for c in f.calls) == 1
        assert [c["url"] for c in f.calls if "/invoices" in c["url"]][1].count("&page=2&") == 1

    def test_zoho_api_error_and_bad_token(self):
        h, _ = http({"oauth/v2/token": [(200, {"error": "invalid_code"})]})
        with pytest.raises(HttpError, match="access token"):
            zc(h).fetch("invoices")
        h, _ = zoho_http([(200, {"code": 57, "message": "You are not authorized"})], [(200, {})])
        with pytest.raises(HttpError, match="57"):
            zc(h).fetch("invoices")

    def test_test_checks_organisation(self):
        h, _ = zoho_http([(200, {})], [(200, {})])
        assert "Acme Pvt" in zc(h).test()
        h, _ = zoho_http([(200, {})], [(200, {})])
        with pytest.raises(HttpError):
            ZohoBooksConnector("in", "123", "a", "b", "c", http=h).test()


# ------------------------------------------------------------------ tally
def voucher(vtype, num, date, party, party_amt, items=None, allocs="", extra=""):
    inv = "".join(f"<ALLINVENTORYENTRIES.LIST><STOCKITEMNAME>{n}</STOCKITEMNAME><RATE>{r}/Nos</RATE><BILLEDQTY>{q} Nos</BILLEDQTY>"
                  f"<AMOUNT>{q * r}</AMOUNT></ALLINVENTORYENTRIES.LIST>" for n, q, r in (items or []))
    return (f'<TALLYMESSAGE><VOUCHER VCHTYPE="{vtype}"><DATE>{date}</DATE><VOUCHERTYPENAME>{vtype}</VOUCHERTYPENAME>'
            f"<VOUCHERNUMBER>{num}</VOUCHERNUMBER><PARTYLEDGERNAME>{party}</PARTYLEDGERNAME>{extra}{inv}"
            f"<ALLLEDGERENTRIES.LIST><LEDGERNAME>{party}</LEDGERNAME><AMOUNT>{party_amt}</AMOUNT>{allocs}</ALLLEDGERENTRIES.LIST>"
            "</VOUCHER></TALLYMESSAGE>")


NEW = "<BILLALLOCATIONS.LIST><NAME>{n}</NAME><BILLTYPE>New Ref</BILLTYPE><BILLCREDITPERIOD>{d} Days</BILLCREDITPERIOD><AMOUNT>{a}</AMOUNT></BILLALLOCATIONS.LIST>"
AGST = "<BILLALLOCATIONS.LIST><NAME>{n}</NAME><BILLTYPE>Agst Ref</BILLTYPE><AMOUNT>{a}</AMOUNT></BILLALLOCATIONS.LIST>"


def tally_xml(*vs, extra=""):
    return f"<ENVELOPE><BODY><IMPORTDATA><REQUESTDATA>{''.join(vs)}{extra}</REQUESTDATA></IMPORTDATA></BODY></ENVELOPE>"


SAMPLE = tally_xml(
    voucher("Sales", "S-1", "20260901", "Acme Traders", "-11800", [("Widget", 10, 1000)], NEW.format(n="S-1", d=15, a="-11800"), "<PLACEOFSUPPLY>Maharashtra</PLACEOFSUPPLY>"),
    voucher("Sales", "S-2", "20260905", "Beta Ltd", "-5900", [("Gadget", 5, 1000)], NEW.format(n="S-2", d=30, a="-5900")),
    voucher("Sales", "S-3", "20260906", "Gamma Co", "-2000", [("Cable", 4, 500)]),
    voucher("Sales", "S-X", "20260907", "Ghost", "-100", [("Void", 1, 100)], extra="<CANCELLED>Yes</CANCELLED>"),
    voucher("Receipt", "R-1", "20260920", "Acme Traders", "11800", None, AGST.format(n="S-1", a="11800")),
    voucher("Receipt", "R-2", "20260921", "Beta Ltd", "2000", None, AGST.format(n="S-2", a="2000")),
    voucher("Purchase", "P-1", "20260910", "Supplier One", "-4000", [("Raw", 8, 500)], NEW.format(n="P-1", d=45, a="-4000")),
    voucher("Sales Order", "SO-1", "20260901", "Nobody", "-1", [("X", 1, 1)]),
)


class TestTally:
    def test_parse_skips_cancelled(self):
        assert [v["number"] for v in parse_vouchers(SAMPLE)] == ["S-1", "S-2", "S-3", "R-1", "R-2", "P-1", "SO-1"]

    def test_sales_dataset(self):
        c = TallyXmlFileConnector(io.BytesIO(SAMPLE.encode()), "x.xml")
        df = c.fetch("sales")
        assert set(df["order_id"]) == {"S-1", "S-2", "S-3"}                        # orders and cancelled excluded
        r = df.set_index("order_id")
        assert r.loc["S-1", "quantity"] == 10 and r.loc["S-1", "unit_price"] == 1000 and r.loc["S-1", "region"] == "Maharashtra"
        assert data_layer.clean_and_validate(df, "sales")[1].ok

    def test_invoices_settlement_partial_and_default_credit(self):
        c = TallyXmlFileConnector(io.BytesIO(SAMPLE.encode()), "x.xml")
        df = c.fetch("invoices").set_index("invoice_id")
        assert df.loc["S-1", "paid_date"] == pd.Timestamp("2026-09-20").date() and df.loc["S-1", "amount"] == 11800   # settled
        assert df.loc["S-2", "amount"] == 3900 and pd.isna(df.loc["S-2", "paid_date"])                                # part paid
        assert (df.loc["S-1", "due_date"] - df.loc["S-1", "invoice_date"]).days == 15
        assert (df.loc["S-3", "due_date"] - df.loc["S-3", "invoice_date"]).days == 30                                 # default
        assert df.loc["P-1", "type"] == "payable" and df.loc["S-3", "type"] == "receivable"
        assert any("assumed" in n for n in c.notes) and any("part-paid" in n for n in c.notes)
        assert data_layer.clean_and_validate(df.reset_index(), "invoices")[1].ok

    def test_forbidden_characters_and_utf16(self):
        dirty = SAMPLE.replace("Widget", "Wid&#4;get\x03")
        assert parse_vouchers(dirty)[0]["items"][0]["item"] == "Widget"
        c = TallyXmlFileConnector(io.BytesIO(SAMPLE.encode("utf-16")), "u16.xml")
        assert len(c.fetch("sales")) == 3

    def test_doctype_and_empty_rejected(self):
        with pytest.raises(ValueError, match="DOCTYPE"):
            parse_vouchers('<!DOCTYPE x [<!ENTITY a "b">]><ENVELOPE/>')
        with pytest.raises(ValueError, match="No vouchers"):
            TallyXmlFileConnector(io.BytesIO(b"<ENVELOPE/>"), "e.xml").fetch("sales")

    def test_live_request_shape_and_error(self):
        h, f = http({"localhost:9000": [(200, SAMPLE.encode("utf-16"))]})
        c = TallyConnector("http://localhost:9000/", company="A & B <Co>", from_date="2026-04-01", to_date="2026-10-31", http=h)
        assert len(c.fetch("sales")) == 3
        body = f.calls[0]["body"].decode()
        assert "<ID>Day Book</ID>" in body and "<SVFROMDATE>20260401</SVFROMDATE>" in body and "A &amp; B &lt;Co&gt;" in body
        h, _ = http({"localhost:9000": [(200, b"<ENVELOPE><LINEERROR>Could not set SVCurrentCompany</LINEERROR></ENVELOPE>")]})
        with pytest.raises(HttpError, match="Could not set"):
            TallyConnector(http=h).fetch("sales")

    def test_test_recognises_tally(self):
        h, _ = http({"localhost:9000": [(200, b"TallyPrime Server is Running")]})
        assert "reachable" in TallyConnector(http=h).test()
        h, _ = http({"localhost:9000": [(200, b"hello")]})
        with pytest.raises(HttpError):
            TallyConnector(http=h).test()


# ------------------------------------------------------------------ credential store
class TestStore:
    def test_no_key_means_no_secrets_on_disk(self, plat):
        store.save("o1", "shopify", {"shop": "s.myshopify.com"})
        with pytest.raises(ValueError, match="BIZIQ_MASTER_KEY"):
            store.save("o1", "shopify", {"shop": "s.myshopify.com"}, {"token": "SECRET"})

    def test_encrypted_at_rest_and_blank_keeps_old(self, plat, monkeypatch):
        monkeypatch.setenv("BIZIQ_SECRET_KEY", Fernet.generate_key().decode())
        store.save("o1", "shopify", {"shop": "s.myshopify.com"}, {"token": "SECRET-TOKEN"})
        raw = platform_db.connect().execute("SELECT settings, secrets FROM connections").fetchone()
        assert b"SECRET-TOKEN" not in bytes(raw[1]) and "SECRET-TOKEN" not in raw[0]
        store.save("o1", "shopify", {"shop": "s2.myshopify.com"}, {"token": ""})                 # blank = keep
        got = store.load("o1", "shopify", ["token"])
        assert got["token"] == "SECRET-TOKEN" and got["shop"] == "s2.myshopify.com"

    def test_env_variables_work_without_a_key(self, plat, monkeypatch):
        monkeypatch.setenv("BIZIQ_SHOPIFY_TOKEN", "FROM-ENV")
        assert store.load("o1", "shopify", ["token"])["token"] == "FROM-ENV"

    def test_wrong_key_cannot_read(self, plat, monkeypatch):
        monkeypatch.setenv("BIZIQ_SECRET_KEY", Fernet.generate_key().decode())
        store.save("o1", "zoho", {}, {"client_secret": "S"})
        monkeypatch.setenv("BIZIQ_SECRET_KEY", Fernet.generate_key().decode())
        from biziq import crypto
        crypto.clear_cache()
        with pytest.raises(crypto.CryptoError):                       # a wrong key is an error, never a silently empty config
            store.load("o1", "zoho", ["client_secret"])


# ------------------------------------------------------------------ sync end to end
class TestSync:
    def test_shopify_sync_saves_audits_and_records_status(self, owner, monkeypatch):
        monkeypatch.setenv("BIZIQ_SECRET_KEY", Fernet.generate_key().decode())
        store.save(owner.org_id, "shopify", {"shop": "s.myshopify.com"}, {"token": "T"})
        h, _ = http({"graphql.json": [(200, page([order("#1"), order("#2")]))]})
        lines = sync.sync(owner.org_id, owner.email, "shopify", http=h)
        assert lines[0].startswith("sales: saved 2 of 2")
        assert len(data_layer.load_dataset("sales")) == 2
        assert "sync.ok" in set(audit.read(owner.org_id)["action"])
        assert store.status(owner.org_id)[0]["last_status"].startswith("sales: saved")

    def test_empty_pull_never_wipes_existing_data(self, owner, monkeypatch):
        monkeypatch.setenv("BIZIQ_SHOPIFY_SHOP", "s.myshopify.com")
        monkeypatch.setenv("BIZIQ_SHOPIFY_TOKEN", "T")
        data_layer.load_sample_data()
        before = len(data_layer.load_dataset("sales"))
        h, _ = http({"graphql.json": [(200, page([]))]})
        out = sync.sync(owner.org_id, owner.email, "shopify", http=h)
        assert "left as it is" in out[0] and len(data_layer.load_dataset("sales")) == before

    def test_bad_setup_and_failure_are_reported_not_raised(self, owner):
        assert "setup problem" in sync.sync(owner.org_id, owner.email, "shopify")[0]
        assert "fail" in {a.split(".")[-1] for a in audit.read(owner.org_id)["action"]}


# ------------------------------------------------------------------ whatsapp
def wa(h=None):
    return WhatsAppSender("123456789012345", "WATOKEN", "payment_reminder", "en", http=h)


class TestWhatsApp:
    @pytest.mark.parametrize("raw,exp", [("98765 43210", "919876543210"), ("+91 98765-43210", "919876543210"),
                                         ("09876543210", "919876543210"), ("12345", None), ("", None)])
    def test_phone_normalisation(self, raw, exp):
        assert normalise_phone(raw) == exp

    def test_payload_shape_and_sanitised_params(self):
        p = wa().payload("919876543210", ["Acme\nTraders", "Rs 5,000"])
        assert p["type"] == "template" and p["template"]["name"] == "payment_reminder"
        assert p["template"]["components"][0]["parameters"][0] == {"type": "text", "text": "Acme Traders"}

    def test_dry_run_is_default_and_sends_nothing(self, monkeypatch):
        monkeypatch.delenv("BIZIQ_SEND_WHATSAPP", raising=False)
        h, f = http({})
        assert wa(h).send("919876543210", ["a", "b"]) == "dry-run" and not f.calls

    def test_live_send_when_enabled(self, monkeypatch):
        monkeypatch.setenv("BIZIQ_SEND_WHATSAPP", "1")
        h, f = http({"/messages": [(200, {"messages": [{"id": "wamid.X"}]})]})
        assert wa(h).send("919876543210", ["a", "b"]) == "wamid.X"
        assert f.calls[0]["headers"]["Authorization"] == "Bearer WATOKEN" and "/123456789012345/messages" in f.calls[0]["url"]

    def test_rejects_bad_phone_number_id(self):
        with pytest.raises(ValueError):
            WhatsAppSender("not-digits", "t", "tpl")

    def test_dispatch_requires_approval_optin_and_never_double_sends(self, owner, monkeypatch):
        monkeypatch.setenv("BIZIQ_SEND_WHATSAPP", "1")
        analyst = accounts.add_user(owner, "a@acme.test", "Ann", "analyst", "Passw0rdOK")
        drafts = [{"party": "Acme", "overdue_amount": 5000}, {"party": "Beta", "overdue_amount": 900}, {"party": "Gamma", "overdue_amount": 10}]
        rid = approvals.request(analyst, "payment_reminder", "Reminders", {"drafts": drafts})
        save_contacts(pd.DataFrame([{"party": "Acme", "phone": "9876543210", "whatsapp_opt_in": True},
                                    {"party": "Beta", "phone": "9876500000", "whatsapp_opt_in": False}]))
        h, f = http({"/messages": [(200, {"messages": [{"id": "wamid.1"}]})]})
        s = wa(h)
        assert dispatch_approved_reminders(owner, s) == {"sent": 0, "dry-run": 0, "skipped": 0, "failed": 0}   # not approved yet
        approvals.decide(owner, rid, True)
        with pytest.raises(accounts.AccountError):
            dispatch_approved_reminders(analyst, s)                                                     # needs approve permission
        assert dispatch_approved_reminders(owner, s) == {"sent": 1, "dry-run": 0, "skipped": 2, "failed": 0}
        assert dispatch_approved_reminders(owner, s)["sent"] == 0 and len(f.calls) == 1                  # no resend
        body = json.loads(f.calls[0]["body"])
        assert body["to"] == "919876543210" and body["template"]["components"][0]["parameters"][1]["text"] == "Rs 5,000"
        assert set(send_log()["status"]) == {"sent", "skipped"}
        assert "whatsapp.sent" in set(audit.read(owner.org_id)["action"])

    def test_failure_is_logged_and_retried_never(self, owner, monkeypatch):
        monkeypatch.setenv("BIZIQ_SEND_WHATSAPP", "1")
        rid = approvals.request(owner, "payment_reminder", "R", {"drafts": [{"party": "Acme", "overdue_amount": 1}]})
        accounts.set_self_approval(owner, True)
        approvals.decide(owner, rid, True)
        save_contacts(pd.DataFrame([{"party": "Acme", "phone": "9876543210", "whatsapp_opt_in": True}]))
        h, _ = http({"/messages": [(400, {"error": "template not approved"})]})
        assert dispatch_approved_reminders(owner, wa(h))["failed"] == 1
        assert load_contacts().shape[0] == 1
