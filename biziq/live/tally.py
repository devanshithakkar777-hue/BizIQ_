"""Tally (Prime / ERP 9) and Tally-compatible XML -> BizIQ `sales` and `invoices`.

Two ways in, one parser:
  * TallyConnector       - live: posts a Day Book export request to Tally's HTTP server (default http://localhost:9000).
                           Tally must be running with a company loaded and the HTTP server switched on. BizIQ must be able
                           to reach that address, so a cloud copy of BizIQ cannot see a Tally on your office PC (run BizIQ
                           on the same network, or use a tunnel).
  * TallyXmlFileConnector- reads a Tally XML file, including the Tally-compatible XML that myBillBook emails you.

Receivable/payable status is worked out from Receipt/Payment vouchers that settle a bill by reference ("Agst Ref").
Bills with no credit period use the default (30 days). Credit/debit notes are ignored. This is best-effort: check the result
against Tally's own Outstandings report for your company before trusting it.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import pandas as pd

from ..connectors import Connector
from .http import Http, HttpError

_BAD_REF = re.compile(r"&#(?:x[0-9a-fA-F]+|\d+);")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    """Tally sometimes writes characters XML forbids (control characters, numeric references to them)."""
    def keep(m):
        s = m.group(0)[2:-1]
        n = int(s[1:], 16) if s[:1] in "xX" else int(s)
        return m.group(0) if n in (9, 10, 13) or n >= 32 else ""
    return _CTRL.sub("", _BAD_REF.sub(keep, text))


def _txt(el, tag: str) -> str:
    x = el.find(tag)
    return (x.text or "").strip() if x is not None else ""


def _num(s: str) -> float:
    m = re.search(r"-?\d[\d,]*\.?\d*", s or "")
    return float(m.group(0).replace(",", "")) if m else 0.0


def _date(s: str):
    try:
        return datetime.strptime(s.strip()[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _entries(v, *tags):
    for t in tags:
        yield from v.findall(t)


def parse_vouchers(text: str) -> list[dict]:
    if re.search(r"<!\s*(DOCTYPE|ENTITY)", text, re.I):
        raise ValueError("This XML contains a DOCTYPE or ENTITY declaration, which BizIQ does not accept.")
    try:
        root = ET.fromstring(_clean(text))
    except ET.ParseError as e:
        raise ValueError(f"Could not read the XML: {e}") from e
    out = []
    for v in root.iter("VOUCHER"):
        if _txt(v, "CANCELLED").lower() == "yes" or _txt(v, "ISOPTIONAL").lower() == "yes":
            continue
        d = _date(_txt(v, "DATE"))
        if d is None:
            continue
        party = _txt(v, "PARTYLEDGERNAME")
        led = []
        for e in _entries(v, "ALLLEDGERENTRIES.LIST", "LEDGERENTRIES.LIST"):
            allocs = [dict(name=_txt(b, "NAME"), kind=_txt(b, "BILLTYPE"), amount=abs(_num(_txt(b, "AMOUNT"))),
                           credit=int(_num(_txt(b, "BILLCREDITPERIOD"))) if _txt(b, "BILLCREDITPERIOD") else None)
                      for b in e.findall("BILLALLOCATIONS.LIST")]
            led.append(dict(name=_txt(e, "LEDGERNAME"), amount=abs(_num(_txt(e, "AMOUNT"))), allocs=allocs))
        inv = [dict(item=_txt(i, "STOCKITEMNAME"), qty=_num(_txt(i, "BILLEDQTY") or _txt(i, "ACTUALQTY")),
                    rate=_num(_txt(i, "RATE")), amount=abs(_num(_txt(i, "AMOUNT"))))
               for i in _entries(v, "ALLINVENTORYENTRIES.LIST", "INVENTORYENTRIES.LIST")]
        out.append(dict(date=d, type=(_txt(v, "VOUCHERTYPENAME") or v.get("VCHTYPE") or "").strip(),
                        number=_txt(v, "VOUCHERNUMBER"), party=party, ledgers=led, items=inv,
                        region=_txt(v, "PLACEOFSUPPLY") or _txt(v, "STATENAME") or _txt(v, "BUYERSTATENAME") or "Unknown"))
    return out


def _kind(vtype: str) -> str:
    t = vtype.lower()
    if "order" in t or "note" in t:
        return "other"
    for k in ("sales", "purchase", "receipt", "payment"):
        if k in t:
            return k
    return "other"


def vouchers_to_sales(vs: list[dict], notes: list[str]) -> pd.DataFrame:
    rows = []
    for v in vs:
        if _kind(v["type"]) != "sales":
            continue
        cust = v["party"] or "Unknown"
        if v["items"]:
            for i in v["items"]:
                qty = i["qty"] or 1.0
                rows.append(dict(order_id=v["number"], date=v["date"], customer_id=cust, product=i["item"] or "Item",
                                 region=v["region"], quantity=qty, unit_price=i["rate"] or (i["amount"] / qty)))
        else:                                              # service or accounts-only invoice: one line at the party total
            total = next((l["amount"] for l in v["ledgers"] if l["name"] == v["party"]), 0.0)
            if total:
                rows.append(dict(order_id=v["number"], date=v["date"], customer_id=cust, product="Sales (no stock items)",
                                 region=v["region"], quantity=1, unit_price=total))
    if rows and any(r["product"] == "Sales (no stock items)" for r in rows):
        notes.append("Some sales vouchers have no stock items and were loaded as one line at the invoice total.")
    return pd.DataFrame(rows)


def vouchers_to_invoices(vs: list[dict], notes: list[str], default_credit_days: int = 30) -> pd.DataFrame:
    paid: dict[tuple, list] = {}
    for v in vs:
        if _kind(v["type"]) in ("receipt", "payment"):
            for l in v["ledgers"]:
                for a in l["allocs"]:
                    if a["kind"].lower().startswith("agst"):
                        paid.setdefault((l["name"], a["name"]), []).append((v["date"], a["amount"]))
    rows, defaulted, partial = [], 0, 0
    for v in vs:
        k = _kind(v["type"])
        if k not in ("sales", "purchase") or not v["party"]:
            continue
        pl = next((l for l in v["ledgers"] if l["name"] == v["party"]), None)
        if not pl or not pl["amount"]:
            continue
        new = next((a for a in pl["allocs"] if a["kind"].lower().startswith("new")), None)
        ref = (new or {}).get("name") or v["number"]
        days = (new or {}).get("credit")
        if days is None:
            days, defaulted = default_credit_days, defaulted + 1
        pays = paid.get((v["party"], ref), [])
        settled = sum(a for _, a in pays)
        outstanding = pl["amount"] - settled
        full = outstanding <= 0.5
        if not full and settled > 0:
            partial += 1
        rows.append(dict(invoice_id=v["number"], party=v["party"], type="receivable" if k == "sales" else "payable",
                         invoice_date=v["date"], due_date=v["date"] + timedelta(days=int(days)),
                         amount=pl["amount"] if full else outstanding, paid_date=max(d for d, _ in pays) if full else None))
    if defaulted:
        notes.append(f"{defaulted} bills had no credit period in Tally; {default_credit_days} days was assumed.")
    if partial:
        notes.append(f"{partial} part-paid bills were loaded at the amount still outstanding.")
    return pd.DataFrame(rows)


class _TallyBase(Connector):
    datasets = ["sales", "invoices"]

    def __init__(self, default_credit_days: int = 30):
        self.default_credit_days, self.notes, self._vs = int(default_credit_days), [], None

    def _xml(self) -> str:
        raise NotImplementedError

    def vouchers(self) -> list[dict]:
        if self._vs is None:
            self._vs = parse_vouchers(self._xml())
            if not self._vs:
                raise ValueError("No vouchers were found. Check the date range, the company, and that this is a voucher export.")
        return self._vs

    def fetch(self, dataset: str) -> pd.DataFrame:
        if dataset == "sales":
            return vouchers_to_sales(self.vouchers(), self.notes)
        if dataset == "invoices":
            return vouchers_to_invoices(self.vouchers(), self.notes, self.default_credit_days)
        raise ValueError("Tally provides the sales and invoices datasets.")


class TallyXmlFileConnector(_TallyBase):
    def __init__(self, file_or_path, filename: str | None = None, default_credit_days: int = 30):
        super().__init__(default_credit_days)
        self.file_or_path, self.name = file_or_path, f"file:tally-xml:{filename or 'upload'}"

    def _xml(self) -> str:
        f = self.file_or_path
        if hasattr(f, "read"):
            raw = f.read()
            raw = raw.encode() if isinstance(raw, str) else raw
        else:
            raw = open(f, "rb").read()
        return decode(raw)


class TallyConnector(_TallyBase):
    def __init__(self, url: str = "http://localhost:9000", company: str = "", from_date: str = "", to_date: str = "",
                 default_credit_days: int = 30, http: Http | None = None):
        super().__init__(default_credit_days)
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("Tally address must start with http:// or https://")
        today = date.today()
        fy = today.year if today.month >= 4 else today.year - 1
        self.url, self.company = url.strip().rstrip("/"), (company or "").strip()
        self.from_date = (from_date or f"{fy}0401").replace("-", "")
        self.to_date = (to_date or today.strftime("%Y%m%d")).replace("-", "")
        self.http, self.name = http or Http(timeout=120, retries=1), f"live:tally:{self.url}"

    def _request_xml(self) -> str:
        co = f"<SVCURRENTCOMPANY>{_esc(self.company)}</SVCURRENTCOMPANY>" if self.company else ""
        return ("<ENVELOPE><HEADER><VERSION>1</VERSION><TALLYREQUEST>Export</TALLYREQUEST><TYPE>Data</TYPE>"
                "<ID>Day Book</ID></HEADER><BODY><DESC><STATICVARIABLES><SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>"
                f"<SVFROMDATE>{self.from_date}</SVFROMDATE><SVTODATE>{self.to_date}</SVTODATE>{co}"
                "</STATICVARIABLES></DESC></BODY></ENVELOPE>")

    def test(self) -> str:
        r = self.http.request("GET", self.url + "/", headers={"Accept": "*/*"})
        if "tally" not in r.text.lower():
            raise HttpError("The address answered, but it does not look like Tally's HTTP server.")
        return "Tally's HTTP server is reachable. Load a company in Tally, then run a sync."

    def _xml(self) -> str:
        r = self.http.request("POST", self.url, headers={"Content-Type": "text/xml; charset=utf-8", "Accept": "*/*"},
                              data=self._request_xml().encode("utf-8"))
        text = decode(r.body)
        if "<LINEERROR>" in text:
            msg = re.search(r"<LINEERROR>(.*?)</LINEERROR>", text, re.S)
            raise HttpError("Tally said: " + (msg.group(1).strip()[:200] if msg else "an error"))
        return text


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
