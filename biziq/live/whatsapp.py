"""WhatsApp Cloud API: send approved payment reminders as TEMPLATE messages.

Nothing is sent unless BIZIQ_SEND_WHATSAPP=1 (otherwise a dry run is recorded). Business-started messages must use a
template that Meta has approved, and only to people who opted in. Your template body needs two variables:
{{1}} = customer name, {{2}} = amount overdue, e.g. "Hello {{1}}, a payment of {{2}} is overdue. Please let us know when we can expect it."
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import pandas as pd

from .. import audit, data_layer
from .http import Http, HttpError

DEFAULT_VERSION = "v23.0"


def normalise_phone(raw: str, default_country: str = "91") -> str | None:
    d = re.sub(r"\D", "", raw or "")
    if len(d) == 10:
        d = default_country + d
    elif len(d) == 11 and d.startswith("0"):
        d = default_country + d[1:]
    return d if 11 <= len(d) <= 15 else None


def _param(s: str) -> str:
    return re.sub(r"[\n\t]+| {4,}", " ", str(s)).strip()[:1000]      # Meta rejects newlines, tabs and long space runs


def live_sending_enabled() -> bool:
    return os.environ.get("BIZIQ_SEND_WHATSAPP", "0") == "1"


class WhatsAppSender:
    def __init__(self, phone_number_id: str, token: str, template: str, language: str = "en",
                 api_version: str = DEFAULT_VERSION, http: Http | None = None):
        if not re.fullmatch(r"\d{5,20}", str(phone_number_id or "")):
            raise ValueError("Phone number id should be digits only (find it in Meta's WhatsApp > API setup).")
        if not token or not template:
            raise ValueError("WhatsApp needs an access token and an approved template name.")
        self.pid, self.token, self.template, self.lang, self.version = str(phone_number_id), token, template, language, api_version
        self.http = http or Http()

    def payload(self, to: str, params: list[str]) -> dict:
        return {"messaging_product": "whatsapp", "to": to, "type": "template",
                "template": {"name": self.template, "language": {"code": self.lang},
                             "components": [{"type": "body", "parameters": [{"type": "text", "text": _param(p)} for p in params]}]}}

    def test(self) -> str:
        r = self.http.request("GET", f"https://graph.facebook.com/{self.version}/{self.pid}",
                              headers={"Authorization": f"Bearer {self.token}"}, params={"fields": "display_phone_number,quality_rating"}).json()
        return f"Connected to WhatsApp number {r.get('display_phone_number', self.pid)} (quality: {r.get('quality_rating', 'unknown')})."

    def send(self, to: str, params: list[str], dry_run: bool | None = None) -> str:
        dry = (not live_sending_enabled()) if dry_run is None else dry_run
        if dry:
            return "dry-run"
        r = self.http.request("POST", f"https://graph.facebook.com/{self.version}/{self.pid}/messages",
                              headers={"Authorization": f"Bearer {self.token}"}, json_body=self.payload(to, params)).json()
        msgs = r.get("messages") or []
        if not msgs:
            raise HttpError("WhatsApp accepted the request but returned no message id.")
        return msgs[0].get("id", "sent")


# ------------------------------------------------------------------ contacts and send log (per organisation database)
def _init(c):
    c.execute("CREATE TABLE IF NOT EXISTS contacts (party TEXT PRIMARY KEY, phone TEXT, whatsapp_opt_in INTEGER NOT NULL DEFAULT 0, "
              "updated_at TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS dispatch_log (approval_id INTEGER, party TEXT, status TEXT, detail TEXT, ts TEXT, "
              "PRIMARY KEY (approval_id, party))")


def load_contacts() -> pd.DataFrame:
    with data_layer._conn() as c:
        _init(c)
        return data_layer.db.read_df(c, "SELECT party, phone, whatsapp_opt_in FROM contacts ORDER BY party")


def save_contacts(df: pd.DataFrame) -> int:
    now, n = datetime.now(timezone.utc).isoformat(), 0
    with data_layer._conn() as c:
        _init(c)
        for _, r in df.iterrows():
            party = str(r.get("party") or "").strip()
            if not party:
                continue
            c.execute("INSERT INTO contacts VALUES (?,?,?,?) ON CONFLICT(party) DO UPDATE SET phone=excluded.phone, "
                      "whatsapp_opt_in=excluded.whatsapp_opt_in, updated_at=excluded.updated_at",
                      (party, normalise_phone(str(r.get("phone") or "")) or "", int(bool(r.get("whatsapp_opt_in"))), now))
            n += 1
    return n


def _already(approval_id: int, party: str) -> bool:
    with data_layer._conn() as c:
        _init(c)
        return c.execute("SELECT 1 FROM dispatch_log WHERE approval_id=? AND party=?", (approval_id, party)).fetchone() is not None


def _log(approval_id: int, party: str, status: str, detail: str) -> None:
    with data_layer._conn() as c:
        _init(c)
        c.execute(data_layer.db.upsert_sql("dispatch_log", ["approval_id", "party", "status", "detail", "ts"], ["approval_id", "party"]),
                  (approval_id, party, status, detail[:300], datetime.now(timezone.utc).isoformat()))


def send_log() -> pd.DataFrame:
    with data_layer._conn() as c:
        _init(c)
        return data_layer.db.read_df(c, "SELECT * FROM dispatch_log ORDER BY ts DESC")


def dispatch_approved_reminders(user, sender: WhatsAppSender, dry_run: bool | None = None) -> dict:
    """Send every APPROVED payment-reminder draft that has not been sent. One message per approval and customer, ever."""
    from .. import accounts, approvals
    accounts.require(user, "approve")
    counts = {"sent": 0, "dry-run": 0, "skipped": 0, "failed": 0}
    contacts = load_contacts().set_index("party")
    for _, req in approvals.list_requests(user, "approved").iterrows():
        if req["kind"] != "payment_reminder":
            continue
        for d in approvals.get_payload(user, int(req["id"])).get("drafts", []):
            party = str(d.get("party", ""))
            if not party or _already(int(req["id"]), party):
                continue
            row = contacts.loc[party] if party in contacts.index else None
            if row is None or not row["phone"] or not int(row["whatsapp_opt_in"]):
                _log(int(req["id"]), party, "skipped", "no phone number or no WhatsApp opt-in")
                counts["skipped"] += 1
                continue
            try:
                amount = f"Rs {float(d.get('overdue_amount', 0)):,.0f}"
                result = sender.send(row["phone"], [party, amount], dry_run)
            except Exception as e:
                _log(int(req["id"]), party, "failed", str(e))
                counts["failed"] += 1
                audit.log(user.org_id, user.email, "whatsapp.fail", f"#{req['id']}", f"{party}: {str(e)[:150]}")
                continue
            status = "dry-run" if result == "dry-run" else "sent"
            _log(int(req["id"]), party, status, result)
            counts[status] += 1
            audit.log(user.org_id, user.email, f"whatsapp.{status}", f"#{req['id']}", f"{party}: {result}")
    return counts
