"""Zoho Books invoices and bills -> BizIQ `invoices` (receivable and payable).

Needs: data-centre code (in, com, eu, com.au, jp ...), organization id, and an OAuth client with a long-lived refresh token
(scopes ZohoBooks.invoices.READ and ZohoBooks.bills.READ). Zoho signs you in on a regional domain; the wrong one gives 401.
Fully paid documents that carry no payment date are left out rather than guessed, and part-paid ones are loaded at the
amount still outstanding.
"""
from __future__ import annotations

import time

import pandas as pd

from ..connectors import Connector
from .http import Http, HttpError

DATA_CENTRES = {"com", "eu", "in", "com.au", "jp", "ca", "sa"}


class ZohoBooksConnector(Connector):
    datasets = ["invoices"]

    def __init__(self, dc: str, organization_id: str, client_id: str, client_secret: str, refresh_token: str,
                 http: Http | None = None, clock=time.time):
        dc = (dc or "").strip().lower().lstrip(".")
        if dc not in DATA_CENTRES:                       # credentials are sent to this domain: only known Zoho ones
            raise ValueError(f"Data centre must be one of: {', '.join(sorted(DATA_CENTRES))}.")
        if not all([organization_id, client_id, client_secret, refresh_token]):
            raise ValueError("Zoho needs organization id, client id, client secret and refresh token.")
        self.dc, self.org = dc, str(organization_id).strip()
        self.client_id, self.client_secret, self.refresh_token = client_id, client_secret, refresh_token
        self.http, self.clock, self.notes = http or Http(), clock, []
        self._token, self._expires = None, 0.0
        self.name = f"live:zoho:{self.org}"

    def _access_token(self) -> str:
        if self._token and self.clock() < self._expires - 60:
            return self._token
        r = self.http.request("POST", f"https://accounts.zoho.{self.dc}/oauth/v2/token",
                              form={"refresh_token": self.refresh_token, "client_id": self.client_id,
                                    "client_secret": self.client_secret, "grant_type": "refresh_token"}).json()
        if "access_token" not in r:
            raise HttpError(f"Zoho did not give an access token ({r.get('error', 'unknown error')}). "
                            "Check the refresh token and data centre.")
        self._token, self._expires = r["access_token"], self.clock() + float(r.get("expires_in", 3600))
        return self._token

    def _get(self, path: str, params: dict) -> dict:
        r = self.http.request("GET", f"https://www.zohoapis.{self.dc}/books/v3/{path}",
                              headers={"Authorization": f"Zoho-oauthtoken {self._access_token()}"},
                              params={"organization_id": self.org, **params}).json()
        if r.get("code", 0) != 0:
            raise HttpError(f"Zoho Books error {r.get('code')}: {str(r.get('message', ''))[:200]}")
        return r

    def _paged(self, path: str, key: str, max_pages: int = 100):
        page = 1
        while page <= max_pages:
            r = self._get(path, {"page": page, "per_page": 200})
            yield from r.get(key, [])
            if not (r.get("page_context") or {}).get("has_more_page"):
                return
            page += 1
        self.notes.append(f"Stopped after {max_pages} pages of {key}.")

    def test(self) -> str:
        r = self._get("organizations", {})
        names = [o.get("name") for o in r.get("organizations", []) if str(o.get("organization_id")) == self.org]
        if not names:
            raise HttpError("Signed in, but that organization id was not found in this account.")
        return f"Connected to Zoho Books organisation: {names[0]}."

    def fetch(self, dataset: str = "invoices") -> pd.DataFrame:
        if dataset != "invoices":
            raise ValueError("Zoho Books provides the invoices dataset (receivables and payables).")
        rows, skipped_paid, partial = [], 0, 0
        for kind, path, key, num, who in (("receivable", "invoices", "invoices", "invoice_number", "customer_name"),
                                          ("payable", "bills", "bills", "bill_number", "vendor_name")):
            for d in self._paged(path, key):
                if str(d.get("status", "")).lower() in ("draft", "void"):
                    continue
                total, bal = float(d.get("total") or 0), float(d.get("balance") or 0)
                fully_paid = bal <= 0.005 or str(d.get("status", "")).lower() == "paid"
                paid = d.get("last_payment_date") or None
                if fully_paid and not paid:
                    skipped_paid += 1
                    continue
                if not fully_paid and bal < total - 0.005:
                    partial += 1
                rows.append(dict(invoice_id=d.get(num) or d.get(f"{key[:-1]}_id"), party=d.get(who), type=kind,
                                 invoice_date=d.get("date"), due_date=d.get("due_date") or d.get("date"),
                                 amount=total if fully_paid else bal, paid_date=paid if fully_paid else None))
        if skipped_paid:
            self.notes.append(f"{skipped_paid} fully paid documents had no payment date and were left out.")
        if partial:
            self.notes.append(f"{partial} part-paid documents were loaded at the amount still outstanding.")
        return pd.DataFrame(rows)
