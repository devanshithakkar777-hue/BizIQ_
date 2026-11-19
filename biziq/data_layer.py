"""Data layer: bring data in and make it trustworthy.

Upload (CSV / Excel) -> clean -> validate -> store (SQLite).
Validation is strict and produces clear, human-readable messages.
"""
from __future__ import annotations

import io
import os
import re
import sqlite3
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import crypto, db
from . import ingest as ingest_mod
from . import mapping as mapping_mod
from . import platform_db

DB_PATH = platform_db.OUT / "biziq.db"
DB_PATH_DEFAULT = DB_PATH
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"

# Per-session database override. Each Streamlit session runs in its own thread/context, so setting this at the
# top of every script run keeps one organisation's data away from another's. None -> the default DB_PATH.
_ACTIVE_DB: ContextVar[Path | None] = ContextVar("biziq_active_db", default=None)


def use_db(path: Path | None) -> None:
    _ACTIVE_DB.set(Path(path) if path else None)


def current_db_path() -> Path:
    return _ACTIVE_DB.get() or DB_PATH


@dataclass
class Schema:
    required: dict[str, str]                 # column -> kind: num | date | text
    optional: dict[str, str] = field(default_factory=dict)
    nonneg: list[str] = field(default_factory=list)
    description: str = ""
    required_for: str = ""


SCHEMAS: dict[str, Schema] = {
    "financials": Schema(
        required={c: "num" for c in [
            "revenue", "cogs", "operating_expenses", "interest_expense", "tax", "net_income",
            "cash", "receivables", "inventory", "total_assets", "current_liabilities",
            "total_liabilities", "equity", "retained_earnings", "operating_cash_flow",
            "long_term_debt", "short_term_debt", "payables"]} | {"month": "date"},
        optional={c: "num" for c in ["depreciation", "other_current_assets", "fixed_assets"]},
        nonneg=["revenue", "cogs", "cash", "inventory", "receivables", "payables"],
        description="Monthly P&L, balance sheet and cash flow (one row per month).",
        required_for="WealthIQ, CashFlowIQ, TaxIQ, InvestIQ, RiskIQ"),
    "sales": Schema(
        required={"order_id": "text", "date": "date", "customer_id": "text", "product": "text",
                  "region": "text", "quantity": "num", "unit_price": "num"},
        optional={"category": "text", "revenue": "num", "discount_pct": "num"},
        nonneg=["quantity", "unit_price"],
        description="Transaction-level sales lines (discount_pct = discount already given off list price; the price is what was charged).",
        required_for="SalesIQ, ClientIQ, SupplyChainIQ, RiskIQ"),
    "inventory": Schema(
        required={"product": "text", "stock_on_hand": "num", "lead_time_days": "num"},
        optional={"avg_daily_demand": "num"}, nonneg=["stock_on_hand", "lead_time_days"],
        description="Current stock snapshot per product.",
        required_for="SupplyChainIQ, ProcureIQ"),
    "supplier_deliveries": Schema(
        required={"supplier": "text", "product": "text", "order_date": "date",
                  "promised_date": "date", "delivered_date": "date", "qty_ordered": "num",
                  "qty_received": "num", "unit_price": "num"},
        optional={"defective_units": "num", "delivery_id": "text"},
        nonneg=["qty_ordered", "qty_received", "unit_price"],
        description="Supplier purchase and delivery records.",
        required_for="SupplyChainIQ, ProcureIQ, RiskIQ"),
    "invoices": Schema(
        required={"invoice_id": "text", "party": "text", "type": "text",
                  "invoice_date": "date", "due_date": "date", "amount": "num"},
        optional={"paid_date": "date"}, nonneg=["amount"],
        description="Receivable and payable invoices (type = receivable | payable).",
        required_for="CashFlowIQ, RiskIQ"),
    "hr": Schema(
        required={"employee_id": "text", "department": "text", "monthly_salary": "num",
                  "join_date": "date"},
        optional={"exit_date": "date", "performance_rating": "num", "gender": "text", "level": "text", "manager": "text",
                  "overtime_hours_month": "num", "satisfaction_score": "num"},
        nonneg=["monthly_salary"], description="Employee records.", required_for="PeopleIQ"),
    "social": Schema(
        required={"date": "date", "platform": "text", "post_type": "text", "impressions": "num"},
        optional={c: "num" for c in ["likes", "comments", "shares", "clicks"]} | {"theme": "text"},
        nonneg=["impressions"], description="Social media post performance.",
        required_for="BrandIQ"),
    "reviews": Schema(
        required={"date": "date", "rating": "num", "review_text": "text"},
        optional={"product": "text"}, description="Customer reviews.", required_for="BrandIQ"),
    "portfolio": Schema(
        required={"instrument": "text", "asset_class": "text", "units": "num",
                  "buy_price": "num", "current_price": "num"},
        nonneg=["units", "buy_price", "current_price"],
        description="Holdings of surplus funds.", required_for="InvestIQ"),

    # ---- datasets for the platform modules (phase 2 and 3) ----
    "product_costs": Schema(
        required={"product": "text", "unit_cost": "num"},
        optional={"channel_fee_pct": "num", "shipping_cost_per_order": "num", "packaging_cost": "num"},
        nonneg=["unit_cost", "channel_fee_pct", "shipping_cost_per_order", "packaging_cost"],
        description="Landed cost per product, plus optional marketplace fee % and per-order shipping.",
        required_for="ProfitIQ, PricingIQ"),
    "returns": Schema(
        required={"return_id": "text", "order_id": "text", "date": "date", "product": "text", "refund_amount": "num"},
        optional={"reason": "text", "channel": "text", "is_cod": "num", "quantity": "num"},
        nonneg=["refund_amount"], description="Returned and refunded orders.",
        required_for="ReturnsIQ, ProfitIQ"),
    "expenses": Schema(
        required={"expense_id": "text", "date": "date", "category": "text", "vendor": "text", "amount": "num"},
        optional={"department": "text", "payment_mode": "text", "approved_by": "text", "receipt_attached": "num"},
        nonneg=["amount"], description="Operating expense transactions.", required_for="ExpenseIQ"),
    "campaigns": Schema(
        required={"date": "date", "channel": "text", "spend": "num", "impressions": "num", "clicks": "num",
                  "conversions": "num"},
        optional={"campaign": "text", "revenue": "num"},
        nonneg=["spend", "impressions", "clicks", "conversions"], description="Marketing campaign performance by day.",
        required_for="CampaignIQ"),
    "competitor_prices": Schema(
        required={"date": "date", "product": "text", "competitor": "text", "price": "num"}, nonneg=["price"],
        description="Observed competitor prices.", required_for="PricingIQ"),
    "credit_limits": Schema(
        required={"customer": "text", "credit_limit": "num"}, optional={"payment_terms_days": "num"},
        nonneg=["credit_limit"], description="Credit limit and terms agreed with each customer.",
        required_for="CreditIQ"),
    "gst_books": Schema(
        required={"invoice_no": "text", "party_gstin": "text", "invoice_date": "date", "taxable_value": "num",
                  "gst_amount": "num"},
        optional={"party": "text"}, nonneg=["taxable_value"],
        description="Purchase invoices as recorded in your books (for input tax credit).", required_for="ComplianceIQ"),
    "gstr2b": Schema(
        required={"invoice_no": "text", "party_gstin": "text", "invoice_date": "date", "taxable_value": "num",
                  "gst_amount": "num"},
        optional={"party": "text"}, nonneg=["taxable_value"],
        description="Supplier-filed invoices as shown in GSTR-2B.", required_for="ComplianceIQ"),
    "pipeline": Schema(
        required={"deal_id": "text", "account": "text", "stage": "text", "amount": "num",
                  "created_date": "date", "last_activity_date": "date"},
        optional={"expected_close": "date", "owner": "text", "source": "text"}, nonneg=["amount"],
        description="Open and closed sales deals.", required_for="PipelineIQ"),
    "quotes": Schema(
        required={"quote_id": "text", "customer": "text", "sent_date": "date", "amount": "num", "status": "text"},
        optional={"cost": "num", "lost_reason": "text", "decided_date": "date"}, nonneg=["amount"],
        description="Quotes and bids sent (status = won | lost | open).", required_for="QuoteIQ"),
    "contracts": Schema(
        required={"contract_id": "text", "party": "text", "end_date": "date", "value": "num"},
        optional={"start_date": "date", "renewal_notice_days": "num", "auto_renew": "num",
                  "escalation_pct": "num", "type": "text"}, nonneg=["value"],
        description="Customer and supplier contracts.", required_for="ContractIQ"),
    "tickets": Schema(
        required={"ticket_id": "text", "opened_date": "date", "category": "text", "priority": "text"},
        optional={"closed_date": "date", "csat": "num", "channel": "text"},
        description="Customer support tickets.", required_for="SupportIQ"),
    "marketplace_orders": Schema(
        required={"order_id": "text", "date": "date", "channel": "text", "gross_amount": "num", "commission": "num"},
        optional={"product": "text", "shipping_fee": "num", "other_fees": "num", "settlement_amount": "num", "settled_date": "date", "returned": "num"},
        nonneg=["gross_amount", "commission"], description="Orders sold on marketplaces (Amazon, Flipkart, Meesho...) with the fees charged and what was paid out.",
        required_for="MarketplaceIQ"),
    "whatsapp_messages": Schema(
        required={"date": "date", "campaign": "text", "sent": "num", "delivered": "num"},
        optional={"read": "num", "replied": "num", "opted_out": "num"}, nonneg=["sent", "delivered"],
        description="WhatsApp campaign results per day.", required_for="WhatsAppIQ"),
    "store_daily": Schema(
        required={"store": "text", "date": "date", "sales": "num", "transactions": "num"},
        optional={"footfall": "num", "staff_hours": "num", "area_sqft": "num"}, nonneg=["sales", "transactions"],
        description="Daily sales per store or outlet.", required_for="StoreIQ"),
    "distributors": Schema(
        required={"distributor": "text", "month": "date", "primary_sales": "num", "secondary_sales": "num"},
        optional={"closing_stock": "num", "outstanding": "num", "credit_days": "num"}, nonneg=["primary_sales", "secondary_sales"],
        description="Per distributor and month: what you sold in (primary) and what they sold on to retailers (secondary).",
        required_for="DistributorIQ"),
    "shipments": Schema(
        required={"shipment_id": "text", "carrier": "text", "ship_date": "date", "promised_date": "date", "status": "text"},
        optional={"delivered_date": "date", "cost": "num", "zone": "text", "cod_amount": "num", "weight_kg": "num"}, nonneg=["cost", "cod_amount"],
        description="Outbound shipments (status = delivered | rto | lost | in_transit).", required_for="LogisticsIQ"),
    "production": Schema(
        required={"line": "text", "date": "date", "planned_units": "num", "produced_units": "num"},
        optional={"defective_units": "num", "downtime_min": "num", "planned_minutes": "num", "product": "text"},
        nonneg=["planned_units", "produced_units", "defective_units", "downtime_min"],
        description="Daily production per line against plan.", required_for="ProductionIQ"),
    "maintenance": Schema(
        required={"asset": "text", "date": "date", "event_type": "text"},
        optional={"downtime_hours": "num", "cost": "num", "cause": "text"}, nonneg=["downtime_hours", "cost"],
        description="Machine events (event_type = breakdown | preventive).", required_for="MaintenanceIQ"),
    "tenders": Schema(
        required={"tender_id": "text", "title": "text", "deadline": "date", "value": "num", "status": "text"},
        optional={"our_bid": "num", "winning_bid": "num", "emd_amount": "num", "submitted_date": "date"}, nonneg=["value"],
        description="Tenders tracked (status = open | submitted | won | lost | no_bid).", required_for="TenderIQ"),
    "esg": Schema(
        required={"month": "date", "electricity_kwh": "num"},
        optional={c: "num" for c in ["fuel_litres", "water_kl", "waste_kg", "recycled_kg", "revenue"]},
        nonneg=["electricity_kwh", "fuel_litres", "water_kl", "waste_kg", "recycled_kg"],
        description="Monthly energy, water and waste.", required_for="ESGIQ"),
    "candidates": Schema(
        required={"candidate_id": "text", "role": "text", "applied_date": "date", "stage": "text"},
        optional={"source": "text", "department": "text", "offer_date": "date", "hired_date": "date", "offer_accepted": "num", "last_stage_date": "date"},
        description="Hiring pipeline, one row per candidate (stage = applied | screen | interview | offer | hired | rejected | withdrawn).",
        required_for="HiringIQ"),
}


@dataclass
class ValidationReport:
    dataset: str
    rows_in: int = 0
    rows_out: int = 0
    errors: list[str] = field(default_factory=list)      # block use of the dataset
    warnings: list[str] = field(default_factory=list)    # data cleaned or suspicious
    cleaning: list[str] = field(default_factory=list)    # what was fixed automatically
    rejects: pd.DataFrame = field(default_factory=pd.DataFrame)   # rows set aside, with source_row and reason
    rows_rejected: int = 0
    suggestions: list = field(default_factory=list)      # weaker column matches to confirm
    available_columns: list = field(default_factory=list)
    date_order: str | None = None
    save: object = None                                  # SaveResult once stored

    @property
    def ok(self) -> bool:
        return not self.errors


SCHEMAS.update({
    "budgets": Schema(required={"month": "date", "category": "text", "budget": "num"}, nonneg=["budget"],
                      description="Planned spend per category per month.", required_for="BudgetIQ (with expenses)"),
    "bank_statement": Schema(required={"date": "date", "description": "text", "amount": "num"}, optional={"reference": "text"},
                             description="Bank statement lines (credits positive, debits negative).", required_for="ReconIQ"),
    "ledger_entries": Schema(required={"date": "date", "description": "text", "amount": "num"}, optional={"reference": "text"},
                             description="Cash/bank book entries from your accounts (receipts positive, payments negative).", required_for="ReconIQ"),
    "projects": Schema(required={"project_id": "text", "client": "text", "budget": "num", "start_date": "date"},
                       optional={"project": "text", "due_date": "date", "completed_date": "date", "status": "text", "billed": "num", "cost_to_date": "num",
                                 "hours_budget": "num", "hours_spent": "num", "percent_complete": "num"},
                       nonneg=["budget", "billed", "cost_to_date", "hours_budget", "hours_spent"],
                       description="One row per client project: budget, billing, cost and progress.", required_for="ProjectIQ"),
    "subscriptions": Schema(required={"subscription_id": "text", "customer": "text", "plan": "text", "start_date": "date", "mrr": "num"},
                            optional={"end_date": "date"}, nonneg=["mrr"],
                            description="One row per subscription with its monthly recurring revenue; end_date empty while active.", required_for="SubscriptionIQ"),
    "appointments": Schema(required={"appointment_id": "text", "date": "date", "customer": "text", "service": "text", "status": "text"},
                           optional={"staff": "text", "price": "num", "duration_min": "num", "start_time": "text"}, nonneg=["price", "duration_min"],
                           description="Bookings with status completed, no_show or cancelled.", required_for="AppointmentIQ"),
    "recipes": Schema(required={"dish": "text", "ingredient": "text", "qty_per_portion": "num", "unit_cost": "num"},
                      optional={"unit": "text", "selling_price": "num", "portions_sold": "num"}, nonneg=["qty_per_portion", "unit_cost", "selling_price", "portions_sold"],
                      description="One row per ingredient per dish; unit_cost is the cost of one unit of the ingredient.", required_for="RecipeIQ"),
    "assets": Schema(required={"asset_id": "text", "category": "text", "purchase_date": "date", "cost": "num"},
                     optional={"asset_name": "text", "useful_life_years": "num", "salvage_value": "num", "status": "text"}, nonneg=["cost", "useful_life_years", "salvage_value"],
                     description="Fixed-asset register.", required_for="AssetIQ"),
    "survey_responses": Schema(required={"response_id": "text", "date": "date", "score": "num"},
                               optional={"segment": "text", "channel": "text", "comment": "text", "csat": "num"}, nonneg=["score"],
                               description="Customer survey answers: score is 0-10 (NPS style); csat 1-5 optional.", required_for="SurveyIQ"),
})

# Line-level data where two identical rows can be two real sales: keep them, warn, never delete.
KEEP_DUPLICATES = {"sales", "returns", "expenses", "campaigns", "competitor_prices", "social", "reviews", "supplier_deliveries", "tickets", "whatsapp_messages", "production", "maintenance", "bank_statement", "ledger_entries", "appointments"}
# Natural keys: a re-uploaded row with the same key UPDATES the stored one (e.g. an invoice that has since been paid).
KEYS = {"invoices": ["invoice_id", "type"], "financials": ["month"], "pipeline": ["deal_id"], "quotes": ["quote_id"],
        "contracts": ["contract_id"], "tickets": ["ticket_id"], "returns": ["return_id"], "expenses": ["expense_id"],
        "hr": ["employee_id"], "product_costs": ["product"], "credit_limits": ["customer"], "inventory": ["product"],
        "portfolio": ["instrument"], "marketplace_orders": ["order_id"], "shipments": ["shipment_id"], "tenders": ["tender_id"],
        "candidates": ["candidate_id"], "store_daily": ["store", "date"], "distributors": ["distributor", "month"], "esg": ["month"], "gst_books": ["invoice_no", "party_gstin"], "gstr2b": ["invoice_no", "party_gstin"],
        "budgets": ["month", "category"], "projects": ["project_id"], "subscriptions": ["subscription_id"], "appointments": ["appointment_id"],
        "recipes": ["dish", "ingredient"], "assets": ["asset_id"], "survey_responses": ["response_id"]}
NAME_COLUMNS = ("customer_id", "party", "customer", "vendor", "supplier", "account", "product", "competitor", "instrument", "store", "distributor", "carrier", "asset", "line", "campaign", "client", "dish", "staff")
MAX_REJECT_ROWS = 100_000


def read_upload(file_or_path, filename: str | None = None, dataset: str | None = None, sheet: str | None = None) -> pd.DataFrame:
    """Read CSV / TSV / Excel from a path or file-like object. Pass `dataset` so title rows above the headings can be found.
    What was handled on the way (title rows, totals dropped, encoding, sheet picked) is in df.attrs['notes']."""
    targets = None
    if dataset in SCHEMAS:
        targets = list(SCHEMAS[dataset].required) + list(SCHEMAS[dataset].optional)
    df, notes = ingest_mod.read_table(file_or_path, filename, dataset, targets, sheet)
    df.attrs["notes"] = notes
    return df


def _norm_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def _idfix(v):
    return str(int(v)) if isinstance(v, float) and v == v and v.is_integer() else v


def clean_and_validate(df: pd.DataFrame, dataset: str, column_map: dict | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    """Map headings, clean values, validate. Never silently drops data: every dropped row is listed in report.rejects with a reason.
    `column_map` ({your heading: BizIQ column}) is a manual mapping and wins over automatic matching."""
    schema = SCHEMAS[dataset]
    targets = list(schema.required) + list(schema.optional)
    header_row = int(df.attrs.get("header_row", 0))
    notes = list(df.attrs.get("notes", []))
    rows = df.attrs.get("source_rows")
    rows = rows.arr if rows is not None else None
    src = df.reset_index(drop=True)
    df = src.copy()
    rep = ValidationReport(dataset=dataset, rows_in=len(src))
    rep.cleaning.extend(notes)
    rep.available_columns = [str(c) for c in src.columns]
    dropped: list[tuple[np.ndarray, str]] = []

    def drop(mask, reason: str) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            dropped.append((df.index.values[mask], reason))

    def apply_drop(mask, reason: str) -> None:
        nonlocal df
        mask = pd.Series(np.asarray(mask, dtype=bool), index=df.index)
        drop(mask.values, reason)
        df = df[~mask]

    # ---- column mapping: manual first, then well-known aliases; weaker matches are only suggested
    if column_map:
        valid = {s_: t for s_, t in column_map.items() if s_ in df.columns and t in targets}
        df = df.rename(columns=valid)
        rep.cleaning.extend(f"Used your mapping: '{a}' -> '{b}'" for a, b in valid.items())
    auto, weak = mapping_mod.auto_map(list(df.columns), dataset, targets)
    auto = {a: b for a, b in auto.items() if b not in df.columns}
    if auto:
        df = df.rename(columns=auto)
        rep.cleaning.extend(f"Matched column '{a}' to '{b}'" for a, b in auto.items())
    rep.suggestions = [m for m in weak if m["target"] not in [_norm_col(c) for c in df.columns]]
    df.columns = [_norm_col(c) for c in df.columns]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in schema.required if c not in df.columns]
    if missing:
        hint = "".join(f" '{m['source']}' looks like '{m['target']}' (please confirm)." for m in rep.suggestions if m["target"] in missing)
        rep.errors.append(f"Missing required column(s): {', '.join(missing)}.{hint} "
                          f"Your file has: {', '.join(rep.available_columns[:14])}{' ...' if len(rep.available_columns) > 14 else ''}. "
                          "Use the column-matching step to say which of your columns is which.")
        rep.rows_out = 0
        return df, rep

    keep = [c for c in targets if c in df.columns]
    extra = [c for c in df.columns if c not in keep]
    if extra:
        rep.cleaning.append(f"Ignored {len(extra)} unrecognised column(s): {', '.join(extra[:6])}")
    df = df[keep]

    # ---- duplicates
    if dataset in KEEP_DUPLICATES:
        n = int(df.duplicated().sum())
        if n:
            rep.warnings.append(f"{n} row(s) are identical to another row. They were kept (two identical lines can be two real sales). "
                                "If you loaded the same file twice, they are double counted: use 'add new rows only' next time.")
    else:
        apply_drop(df.duplicated(keep="first").values, "exact duplicate of an earlier row")
        if len(src) - len(df) > 0:
            rep.cleaning.append(f"Removed {int(sum(len(i) for i, r in dropped if 'duplicate' in r))} duplicate row(s)")

    # ---- values
    for col, kind in {**schema.required, **schema.optional}.items():
        if col not in df.columns:
            continue
        if kind == "text":
            ser = df[col]
            if ser.dtype == object and ser.map(type).eq(float).any():
                ser = ser.map(_idfix)
            ser = ser.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
            df[col] = ser.mask(ser == "")
        elif kind == "num":
            conv, info = ingest_mod.parse_numbers(df[col])
            if info["bad"]:
                rep.warnings.append(f"Column '{col}': {info['bad']} value(s) are not numbers (e.g. {', '.join(repr(x) for x in info['examples'])}) and were left blank")
            if info["drcr"]:
                rep.warnings.append(f"Column '{col}': {info['drcr']} value(s) had a Dr/Cr suffix. It was removed; the sign was not changed, so check these.")
            df[col] = conv
        elif kind == "date":
            conv, info = ingest_mod.parse_dates(df[col])
            if info["bad"]:
                rep.warnings.append(f"Column '{col}': {info['bad']} value(s) are not valid dates (e.g. {', '.join(repr(x) for x in info['examples'])}) and were left blank")
            if info["order"]:
                rep.date_order = info["order"]
                word = "day/month/year" if info["order"] == "dmy" else "month/day/year"
                if info["conflict"]:
                    rep.warnings.append(f"Column '{col}' mixes day-first and month-first dates. Read the majority way ({word}); check the file.")
                elif info["ambiguous"]:
                    rep.warnings.append(f"Column '{col}': {info['ambiguous']} date(s) like 03/04/2026 could be read two ways. Read as {word}. "
                                        "If that is wrong, set BIZIQ_DATE_ORDER=mdy.")
                else:
                    rep.cleaning.append(f"Column '{col}': dates read as {word} (values above 12 confirm it)")
            if info["implausible"]:
                rep.warnings.append(f"Column '{col}': {info['implausible']} date(s) fall before 1990 or more than 2 years ahead. Check for typos or a wrong format.")
            df[col] = conv

    # ---- required fields must be populated
    for col in schema.required:
        n_null = int(df[col].isna().sum())
        if n_null:
            share = n_null / max(len(df), 1)
            if share > 0.20:
                rep.errors.append(f"Column '{col}' is {share:.0%} blank ({n_null} rows). Fix the source file and re-upload.")
            else:
                rep.warnings.append(f"Set aside {n_null} row(s) with blank '{col}' (see the rejected-rows file)")
    if not rep.errors:
        for col in schema.required:
            apply_drop(df[col].isna().values, f"blank required field: {col}")

    for col in schema.nonneg:
        if col in df.columns:
            neg = (df[col] < 0).values
            if neg.any():
                extra_note = " These look like returns or credit notes: load them in the Returns dataset instead." if dataset == "sales" else ""
                rep.warnings.append(f"Column '{col}': {int(neg.sum())} negative value(s) set aside.{extra_note}")
                apply_drop(neg, f"negative {col}")

    if dataset == "financials" and not rep.errors:
        if df["month"].duplicated().any():
            rep.errors.append("Duplicate months found in financials; each month must appear once.")
        df = df.sort_values("month")
        if len(df) < 6:
            rep.warnings.append("Fewer than 6 months of financials; trend analysis will be limited.")
        gap = (df["total_assets"] - df["total_liabilities"] - df["equity"]).abs()
        loose = int((gap > 0.02 * df["total_assets"].abs().clip(lower=1)).sum())
        if loose:
            rep.warnings.append(f"Balance sheet does not balance (assets != liabilities + equity) in {loose} month(s); check the source.")
    if dataset == "sales" and not rep.errors:
        if "revenue" not in df.columns or df["revenue"].isna().all():
            df["revenue"] = df["quantity"] * df["unit_price"]
            rep.cleaning.append("Derived revenue = quantity x unit_price")
        else:
            off = ((df["revenue"] - df["quantity"] * df["unit_price"]).abs() > 0.02 * (df["quantity"] * df["unit_price"]).abs().clip(lower=1)).mean()
            if off > 0.30:
                rep.warnings.append(f"For {off:.0%} of rows, revenue differs from quantity x unit price by more than 2% (discounts, tax, or a wrong column?).")
        if "category" not in df.columns:
            df["category"] = "Uncategorised"
            rep.cleaning.append("No category column; all products grouped as 'Uncategorised'")
    if dataset == "invoices" and not rep.errors:
        df["type"] = df["type"].str.lower()
        df["type"] = df["type"].replace({"sales": "receivable", "sale": "receivable", "customer": "receivable", "debtor": "receivable", "ar": "receivable",
                                         "purchase": "payable", "bill": "payable", "vendor": "payable", "supplier": "payable", "creditor": "payable", "ap": "payable"})
        bad_type = ~df["type"].isin(["receivable", "payable"])
        if bad_type.any():
            rep.warnings.append(f"{int(bad_type.sum())} invoice(s) with a type other than receivable/payable were set aside")
            apply_drop(bad_type.values, "type is not receivable/payable")
    if dataset == "supplier_deliveries" and "defective_units" not in df.columns and not rep.errors:
        df["defective_units"] = 0
        rep.cleaning.append("No defective_units column; assumed 0")
    if dataset == "quotes" and not rep.errors:
        df["status"] = df["status"].str.lower()
        bad = ~df["status"].isin(["won", "lost", "open"])
        if bad.any():
            rep.warnings.append(f"{int(bad.sum())} quote(s) with status other than won/lost/open were set aside")
            apply_drop(bad.values, "status is not won/lost/open")
    if dataset in ("pipeline", "tickets") and not rep.errors:
        col = "stage" if dataset == "pipeline" else "priority"
        df[col] = df[col].str.lower()
    if dataset in ("gst_books", "gstr2b") and not rep.errors:
        df["party_gstin"] = df["party_gstin"].str.upper().str.replace(r"\s", "", regex=True)
        df["invoice_no"] = df["invoice_no"].str.upper().str.replace(r"\s", "", regex=True)
        bad = ~df["party_gstin"].str.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]", na=False)
        if bad.any():
            rep.warnings.append(f"{int(bad.sum())} row(s) have a GSTIN that does not look valid (15 characters); check them before relying on the reconciliation.")
    if dataset == "campaigns" and not rep.errors and "revenue" not in df.columns:
        df["revenue"] = float("nan")
        rep.cleaning.append("No revenue column; return on ad spend will be unavailable")

    # ---- same key twice: the later row wins (it is usually the corrected one)
    keys = KEYS.get(dataset)
    if keys and dataset != "financials" and not rep.errors and all(k in df.columns for k in keys):
        older = df.duplicated(keys, keep="last").values
        if older.any():
            rep.warnings.append(f"{int(older.sum())} row(s) repeat an existing {' + '.join(keys)}; the later row was kept.")
            apply_drop(older, f"older row for the same {' + '.join(keys)}")

    # ---- names that differ only by capitals or spacing split one customer into two
    for col in NAME_COLUMNS:
        if col in df.columns:
            u = pd.Series(df[col].dropna().unique(), dtype="string")
            if len(u):
                key = u.str.lower().str.replace(r"[\s\.,]+", " ", regex=True).str.strip()
                grp = u.groupby(key).agg(list)
                multi = grp[grp.map(len) > 1]
                if len(multi):
                    ex = "; ".join(" / ".join(repr(x) for x in v[:3]) for v in multi.head(2))
                    rep.warnings.append(f"Column '{col}': {len(multi)} name(s) are written more than one way (e.g. {ex}). They are counted as separate "
                                        "until you make them match in the source.")

    rep.rows_out = len(df)
    if dropped:
        parts = []
        for idx, reason in dropped:
            r = src.iloc[idx].copy()
            r.insert(0, "reason", reason)
            r.insert(0, "source_row", rows[np.asarray(idx)] if rows is not None else np.asarray(idx) + header_row + 2)
            parts.append(r)
        allr = pd.concat(parts, ignore_index=True)
        rep.rows_rejected = len(allr)
        rep.rejects = allr.head(MAX_REJECT_ROWS)
    if rep.rows_out == 0 and not rep.errors:
        rep.errors.append("No valid rows remain after cleaning.")
    return df.reset_index(drop=True), rep


# ---------------------------------------------------------------- storage
class _Conn(sqlite3.Connection):
    """`with _conn() as c:` commits (or rolls back) AND closes, so connections never pile up."""
    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


def _conn():
    if db.IS_PG:
        return db.pg_connect(db.schema_for(current_db_path(), default_local=DB_PATH_DEFAULT))
    p = current_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p, timeout=30, factory=_Conn)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA busy_timeout=30000")
    return c


@dataclass
class SaveResult:
    mode: str
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    total: int = 0

    def summary(self) -> str:
        if self.mode == "replace":
            return f"Replaced the dataset with {self.total} rows."
        return f"Added {self.added}, updated {self.updated}, already present {self.unchanged}. Now {self.total} rows."


def _check(dataset: str) -> str:
    if dataset not in SCHEMAS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return dataset


def _row_hash(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.Series(pd.util.hash_pandas_object(df[cols].astype(str), index=False).values, index=df.index)


def _merge(old: pd.DataFrame, new: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, SaveResult]:
    cols = list(new.columns)
    old = old.reindex(columns=cols)
    keys = KEYS.get(dataset)
    res = SaveResult("merge")
    if keys:
        old_h, new_h = _row_hash(old, cols), _row_hash(new, cols)
        o = old[keys].copy(); o["_h"] = old_h.values
        n = new[keys].copy(); n["_h"] = new_h.values
        m = n.merge(o, on=keys, how="left", suffixes=("", "_old"), indicator=True)
        res.added = int((m["_merge"] == "left_only").sum())
        res.unchanged = int(((m["_merge"] == "both") & (m["_h"] == m["_h_old"])).sum())
        res.updated = len(m) - res.added - res.unchanged
        gone = old.set_index(keys).index.isin(new.set_index(keys).index)
        out = pd.concat([old[~gone], new], ignore_index=True)
    else:                                           # no natural key: add rows not already stored (counting repeats properly)
        old_h, new_h = _row_hash(old, cols), _row_hash(new, cols)
        have = old_h.value_counts()
        occ = new_h.groupby(new_h).cumcount()
        fresh = occ >= new_h.map(have).fillna(0).astype(int)
        res.added, res.unchanged = int(fresh.sum()), int((~fresh).sum())
        out = pd.concat([old, new[fresh.values]], ignore_index=True)
    res.total = len(out)
    return out, res


def encrypt_on() -> bool:
    return os.environ.get("BIZIQ_ENCRYPT_DATA", "").strip().lower() in ("1", "true", "yes", "on")


def current_scope() -> str:
    """Encryption scope of the organisation whose data this session is using."""
    return "data:" + db.schema_for(current_db_path(), default_local=DB_PATH_DEFAULT)


def _logical(table: str) -> str:
    for pre in ("_prev_", "_swap_"):
        if table.startswith(pre):
            return table[len(pre):]
    return table


CHUNK_BYTES = 8_000_000


def _write_encrypted(c, df: pd.DataFrame, table: str) -> None:
    if not crypto.available():
        raise ValueError("BIZIQ_ENCRYPT_DATA is on but no BIZIQ_MASTER_KEY is set, so the data cannot be encrypted. Set the key (see docs/SECURITY.md).")
    buf = io.BytesIO()
    df = df.copy(deep=False)
    df.attrs = {}                                        # attrs hold non-JSON objects (RowNumbers); parquet would try to json.dumps them
    try:
        df.to_parquet(buf, index=False, compression="zstd")
    except Exception:                                    # mixed-type text columns: store them as text
        buf = io.BytesIO()
        x = df.copy()
        for col in x.columns:
            if x[col].dtype == object:
                x[col] = x[col].astype("string")
        x.to_parquet(buf, index=False, compression="zstd")
    raw, scope, name = buf.getvalue(), current_scope(), _logical(table)
    parts = [raw[i:i + CHUNK_BYTES] for i in range(0, len(raw), CHUNK_BYTES)] or [b""]
    c.execute(f'DROP TABLE IF EXISTS "{table}"')
    c.execute(f'CREATE TABLE "{table}" (chunk INTEGER PRIMARY KEY, blob BLOB NOT NULL)')
    for i, part in enumerate(parts):
        c.execute(f'INSERT INTO "{table}" (chunk, blob) VALUES (?,?)', (i, crypto.encrypt(scope, part, f"{name}|{i}/{len(parts)}")))


def _read_encrypted(c, table: str) -> pd.DataFrame:
    rows = c.execute(f'SELECT chunk, blob FROM "{table}" ORDER BY chunk').fetchall()
    scope, name, n = current_scope(), _logical(table), len(rows)
    raw = b"".join(crypto.decrypt(scope, blob, f"{name}|{i}/{n}") for i, (_, blob) in enumerate(rows))
    return pd.read_parquet(io.BytesIO(raw))


def _is_encrypted_table(c, table: str) -> bool:
    return db.columns(c, table) == ["chunk", "blob"]


def convert_storage(to_encrypted: bool) -> dict:
    """Re-store every dataset table (current and undo copies) of the active organisation, encrypted or plaintext.
    Idempotent, and each table is converted inside one transaction. Returns {table: 'converted' | 'already'}."""
    out = {}
    with _conn() as c:
        names = [n for n in SCHEMAS] + [f"_prev_{n}" for n in SCHEMAS]
        for t in names:
            if not db.table_exists(c, t):
                continue
            enc = _is_encrypted_table(c, t)
            if enc == to_encrypted:
                out[t] = "already"
                continue
            df = _read_table(c, t)
            (_write_encrypted if to_encrypted else db.write_df)(c, df, t)
            out[t] = "converted"
    return out


def reencrypt_all() -> int:
    """Rewrite every encrypted table under the scope's active data key (after crypto.new_key_version)."""
    n = 0
    with _conn() as c:
        for t in [n_ for n_ in SCHEMAS] + [f"_prev_{n_}" for n_ in SCHEMAS]:
            if db.table_exists(c, t) and _is_encrypted_table(c, t):
                _write_encrypted(c, _read_table(c, t), t)
                n += 1
    return n


def _write_table(c, df: pd.DataFrame, table: str) -> None:
    (_write_encrypted if encrypt_on() else db.write_df)(c, df, table)


def _read_table(c, table: str) -> pd.DataFrame:
    return _read_encrypted(c, table) if _is_encrypted_table(c, table) else db.read_df(c, f'SELECT * FROM "{table}"')


_META_COLS = ["dataset", "source", "rows", "loaded_at"]
_META_UPSERT = "ON CONFLICT(dataset) DO UPDATE SET source=excluded.source, rows=excluded.rows, loaded_at=excluded.loaded_at"


def _init_meta(c) -> None:
    c.execute("CREATE TABLE IF NOT EXISTS _meta (dataset TEXT PRIMARY KEY, source TEXT, rows INTEGER, loaded_at TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS _meta_prev (dataset TEXT PRIMARY KEY, source TEXT, rows INTEGER, loaded_at TEXT)")


def save_dataset(df: pd.DataFrame, dataset: str, source: str = "upload", mode: str = "replace") -> SaveResult:
    """mode='replace' swaps the whole dataset. mode='merge' adds new rows and updates rows with the same key (see KEYS).
    The previous version is always kept so a load can be undone once (restore_previous)."""
    _check(dataset)
    result = SaveResult("replace", total=len(df))
    if mode == "merge":
        old = load_dataset(dataset)
        if old is not None and len(old):
            df, result = _merge(old, df, dataset)
        else:
            result = SaveResult("merge", added=len(df), total=len(df))
    with _conn() as c:
        db.ensure(c, "meta", _init_meta)
        exists = db.table_exists(c, dataset)
        if exists:
            c.execute(f"DROP TABLE IF EXISTS _prev_{dataset}")
            c.execute(f"CREATE TABLE _prev_{dataset} AS SELECT * FROM {dataset}")
            c.execute("INSERT INTO _meta_prev SELECT * FROM _meta WHERE dataset=? " + _META_UPSERT, (dataset,))
        _write_table(c, df, dataset)
        c.execute(db.upsert_sql("_meta", _META_COLS, ["dataset"]), (dataset, source, len(df), pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")))
    return result


def has_previous(dataset: str) -> bool:
    _check(dataset)
    try:
        with _conn() as c:
            return db.table_exists(c, f"_prev_{dataset}")
    except Exception:
        return False


def restore_previous(dataset: str) -> None:
    """Undo the last load of a dataset (swap the current and previous versions; undo again to redo)."""
    _check(dataset)
    if not has_previous(dataset):
        raise ValueError("There is no earlier version to go back to.")
    with _conn() as c:
        c.execute(f"DROP TABLE IF EXISTS _swap_{dataset}")
        c.execute(f"ALTER TABLE {dataset} RENAME TO _swap_{dataset}")
        c.execute(f"ALTER TABLE _prev_{dataset} RENAME TO {dataset}")
        c.execute(f"ALTER TABLE _swap_{dataset} RENAME TO _prev_{dataset}")
        cur = c.execute("SELECT source, rows, loaded_at FROM _meta WHERE dataset=?", (dataset,)).fetchone()
        prev = c.execute("SELECT source, rows, loaded_at FROM _meta_prev WHERE dataset=?", (dataset,)).fetchone()
        if prev:
            c.execute(db.upsert_sql("_meta", _META_COLS, ["dataset"]), (dataset, *prev))
        if cur:
            c.execute(db.upsert_sql("_meta_prev", _META_COLS, ["dataset"]), (dataset, *cur))


def load_dataset(dataset: str) -> pd.DataFrame | None:
    schema = SCHEMAS[_check(dataset)]
    try:
        with _conn() as c:
            df = _read_table(c, dataset)
    except crypto.CryptoError:
        raise                       # tampered, moved or wrongly-keyed data must be loud, never "no data"
    except Exception:
        return None
    for col, kind in {**schema.required, **schema.optional}.items():
        if col in df.columns and kind == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def dataset_meta() -> pd.DataFrame:
    try:
        with _conn() as c:
            return db.read_df(c, "SELECT * FROM _meta ORDER BY dataset")
    except Exception:
        return pd.DataFrame(columns=["dataset", "source", "rows", "loaded_at"])


def load_all() -> dict[str, pd.DataFrame]:
    out = {}
    for name in SCHEMAS:
        df = load_dataset(name)
        if df is not None and len(df):
            out[name] = df
    return out


def ingest(df: pd.DataFrame, dataset: str, source: str = "upload", mode: str = "replace", column_map: dict | None = None) -> ValidationReport:
    cleaned, rep = clean_and_validate(df, dataset, column_map)
    if rep.ok:
        rep.save = save_dataset(cleaned, dataset, source, mode)
    return rep


STALE_DAYS = 45
_DATED = {"sales": "date", "invoices": "invoice_date", "returns": "date", "expenses": "date", "campaigns": "date", "tickets": "opened_date",
          "social": "date", "reviews": "date", "competitor_prices": "date", "financials": "month", "supplier_deliveries": "order_date",
          "pipeline": "last_activity_date", "quotes": "sent_date"}


def data_health(today: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per dataset: size, date range, how old the newest record is, where it came from, and a plain-English status."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    meta = dataset_meta().set_index("dataset") if len(dataset_meta()) else pd.DataFrame()
    rows = []
    for name in SCHEMAS:
        df = load_dataset(name)
        if df is None or df.empty:
            rows.append(dict(dataset=name, rows=0, first=None, last=None, days_since_latest=None, source="", loaded_at="", status="not loaded"))
            continue
        col = _DATED.get(name)
        first = last = age = None
        status = "ok"
        if col and col in df.columns and df[col].notna().any():
            first, last = df[col].min(), df[col].max()
            age = int((today - last).days)
            if age > STALE_DAYS:
                status = f"stale: newest record is {age} days old"
            elif last > today + pd.Timedelta(days=2):
                status = "check: dates run into the future"
        rows.append(dict(dataset=name, rows=len(df), first=first, last=last, days_since_latest=age,
                         source=str(meta["source"].get(name, "")) if len(meta) else "", loaded_at=str(meta["loaded_at"].get(name, "")) if len(meta) else "",
                         status=status))
    return pd.DataFrame(rows)


def load_sample_data() -> dict[str, ValidationReport]:
    """Load the bundled SYNTHETIC datasets into the database."""
    reports = {}
    for name in SCHEMAS:
        path = SAMPLE_DIR / f"{name}.csv"
        if path.exists():
            reports[name] = ingest(read_upload(path, dataset=name), name, source="synthetic sample")
    return reports
