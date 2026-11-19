"""Provider catalogue and the one function that pulls data: fetch -> clean and validate -> store -> audit -> record status."""
from __future__ import annotations

from .. import audit, data_layer
from . import store
from .http import Http
from .shopify import DEFAULT_VERSION as SHOPIFY_VERSION, ShopifyConnector
from .tally import TallyConnector
from .whatsapp import DEFAULT_VERSION as WA_VERSION, WhatsAppSender
from .zoho import ZohoBooksConnector

# fields: (key, label, secret?, default)
PROVIDERS = {
    "shopify": dict(label="Shopify (orders)", datasets=["sales"], fields=[
        ("shop", "Store address (your-store.myshopify.com)", False, ""), ("token", "Admin API access token", True, ""),
        ("api_version", "API version", False, SHOPIFY_VERSION), ("since_days", "Days of orders to read", False, "365")],
        build=lambda s, http=None: ShopifyConnector(s.get("shop", ""), s.get("token", ""), s.get("api_version") or SHOPIFY_VERSION,
                                                    int(s.get("since_days") or 365), http=http)),
    "zoho": dict(label="Zoho Books (invoices and bills)", datasets=["invoices"], fields=[
        ("dc", "Data centre (in, com, eu, com.au, jp)", False, "in"), ("organization_id", "Organization id", False, ""),
        ("client_id", "Client id", False, ""), ("client_secret", "Client secret", True, ""),
        ("refresh_token", "Refresh token", True, "")],
        build=lambda s, http=None: ZohoBooksConnector(s.get("dc", ""), s.get("organization_id", ""), s.get("client_id", ""),
                                                      s.get("client_secret", ""), s.get("refresh_token", ""), http=http)),
    "tally": dict(label="Tally (vouchers: sales and invoices)", datasets=["sales", "invoices"], fields=[
        ("url", "Tally address", False, "http://localhost:9000"), ("company", "Company name (blank = the open one)", False, ""),
        ("from_date", "From date (YYYY-MM-DD, blank = this financial year)", False, ""),
        ("to_date", "To date (blank = today)", False, ""), ("default_credit_days", "Credit days when a bill has none", False, "30")],
        build=lambda s, http=None: TallyConnector(s.get("url") or "http://localhost:9000", s.get("company", ""),
                                                  s.get("from_date", ""), s.get("to_date", ""),
                                                  int(s.get("default_credit_days") or 30), http=http)),
}
WHATSAPP_FIELDS = [("phone_number_id", "Phone number id", False, ""), ("token", "Access token", True, ""),
                   ("template", "Approved template name", False, ""), ("language", "Template language code", False, "en"),
                   ("api_version", "Graph API version", False, WA_VERSION)]


def fields_of(provider: str) -> list:
    return WHATSAPP_FIELDS if provider == "whatsapp" else PROVIDERS[provider]["fields"]


def secret_fields(provider: str) -> list[str]:
    return [k for k, _, sec, _ in fields_of(provider) if sec]


def all_field_keys(provider: str) -> list[str]:
    return [k for k, _, _, _ in fields_of(provider)]


def build(org_id: str, provider: str, http: Http | None = None):
    return PROVIDERS[provider]["build"](store.load(org_id, provider, all_field_keys(provider)), http)


def build_whatsapp(org_id: str, http: Http | None = None) -> WhatsAppSender:
    s = store.load(org_id, "whatsapp", all_field_keys("whatsapp"))
    return WhatsAppSender(s.get("phone_number_id", ""), s.get("token", ""), s.get("template", ""), s.get("language") or "en",
                          s.get("api_version") or WA_VERSION, http=http)


def sync(org_id: str, actor_email: str, provider: str, http: Http | None = None) -> list[str]:
    """Pull every dataset the provider offers. Returns human-readable result lines. Never overwrites data with an empty pull."""
    lines = []
    try:
        conn = build(org_id, provider, http)
    except Exception as e:
        msg = f"{provider}: setup problem - {e}"
        store.record_sync(org_id, provider, msg)
        audit.log(org_id, actor_email, "sync.fail", provider, str(e)[:200])
        return [msg]
    for ds in PROVIDERS[provider]["datasets"]:
        try:
            df = conn.fetch(ds)
            if df is None or df.empty:
                lines.append(f"{ds}: nothing returned, so existing data was left as it is.")
                continue
            rep = data_layer.ingest(df, ds, source=conn.name)
            if not rep.ok:
                lines.append(f"{ds}: not saved - " + "; ".join(rep.errors)[:300])
                continue
            lines.append(f"{ds}: saved {rep.rows_out} of {rep.rows_in} rows.")
            audit.log(org_id, actor_email, "sync.ok", f"{provider}/{ds}", f"rows in {rep.rows_in}, kept {rep.rows_out}")
        except Exception as e:
            lines.append(f"{ds}: failed - {e}")
            audit.log(org_id, actor_email, "sync.fail", f"{provider}/{ds}", str(e)[:200])
    lines += [f"note: {n}" for n in getattr(conn, "notes", [])]
    store.record_sync(org_id, provider, " | ".join(lines))
    return lines


def sync_all(org_id: str, actor_email: str = "scheduler") -> dict:
    """Used by the scheduler: sync every configured data provider for one organisation."""
    return {p: sync(org_id, actor_email, p) for p in store.configured(org_id) if p in PROVIDERS}
