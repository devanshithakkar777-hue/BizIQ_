"""How sure should you be about this page? A plain assessment of the data behind each module: how much there is, how recent, how complete.

It does not judge whether a module's logic is right (the tests do that); it says whether the data fed to it is thick enough, fresh enough and complete enough
to lean on. The rules are simple on purpose so anyone can see why a page says 'Low'."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data_layer, registry

BUILTIN_READS = {
    "WealthIQ": ["financials"], "SalesIQ": ["sales"], "ClientIQ": ["sales"], "SupplyChainIQ": ["sales", "inventory"], "CashFlowIQ": ["financials", "invoices"],
    "TaxIQ": ["financials"], "RiskIQ": ["financials"], "InvestIQ": ["financials", "portfolio"], "PeopleIQ": ["hr"], "BrandIQ": ["social", "reviews"], "ProcureIQ": ["supplier_deliveries"],
}
# data that is a snapshot of how things stand, not a history: judged by row count only
SNAPSHOT = {"inventory", "product_costs", "assets", "recipes", "projects", "credit_limits", "hr", "portfolio", "budgets", "distributors", "esg", "contracts", "pipeline", "quotes", "tenders", "candidates"}
MONTHLY = {"financials"}
STALE_MEDIUM, STALE_LOW = 45, 120
RANK = {"High": 0, "Medium": 1, "Low": 2}


@dataclass
class Evidence:
    level: str                       # High | Medium | Low | Rule-based
    reasons: list[str] = field(default_factory=list)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)


def reads(name: str) -> list[str]:
    spec = registry.get(name)
    return list(spec.needs) if spec is not None else list(BUILTIN_READS.get(name, []))


def _date_col(ds: str) -> str | None:
    sch = data_layer.SCHEMAS.get(ds)
    if sch is None:
        return None
    return next((c for c, k in {**sch.required, **sch.optional}.items() if k == "date"), None)


def _one(ds: str, df: pd.DataFrame, today: pd.Timestamp) -> tuple[dict, str, list[str]]:
    n = len(df)
    sch = data_layer.SCHEMAS.get(ds)
    level, why = "High", []

    def worse(l, msg):
        nonlocal level
        if RANK[l] > RANK[level]:
            level = l
        why.append(f"{ds}: {msg}")
    dc = _date_col(ds)
    first = last = None
    if dc and dc in df.columns:
        d = pd.to_datetime(df[dc], errors="coerce").dropna()
        if len(d):
            first, last = d.min(), d.max()
    if ds in MONTHLY:
        if n < 6:
            worse("Low", f"only {n} month(s) of figures")
        elif n < 12:
            worse("Medium", f"{n} months of figures (a full year is better)")
    elif ds in SNAPSHOT:
        if n < 3:
            worse("Low", f"only {n} row(s)")
    else:
        if n < 20:
            worse("Low", f"only {n} row(s)")
        elif n < 100:
            worse("Medium", f"{n} rows is a small sample")
    if first is not None and ds not in SNAPSHOT and ds not in MONTHLY:
        span = (last - first).days
        if span < 60:
            worse("Medium", f"covers only {span} day(s)")
    age = None
    if last is not None and ds not in SNAPSHOT:
        age = int((today - last).days)
        if age > STALE_LOW:
            worse("Low", f"the newest record is {age} days old")
        elif age > STALE_MEDIUM:
            worse("Medium", f"the newest record is {age} days old")
    miss = None
    if sch is not None and n:
        cols = [c for c in sch.required if c in df.columns]
        if cols:
            miss = float(df[cols].isna().to_numpy().mean() * 100)
            if miss > 40:
                worse("Low", f"{miss:.0f}% of required values are blank")
            elif miss > 15:
                worse("Medium", f"{miss:.0f}% of required values are blank")
    row = dict(dataset=ds, rows=n, first_record=None if first is None else first.strftime("%Y-%m-%d"), latest_record=None if last is None else last.strftime("%Y-%m-%d"),
               latest_age_days=age, blank_values_pct=None if miss is None else round(miss, 1))
    return row, level, why


def assess(name: str, data: dict[str, pd.DataFrame], today: pd.Timestamp | None = None) -> Evidence | None:
    today = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize()
    if name == "ComplianceCalendarIQ":
        return Evidence("Rule-based", ["The dates come from the statutory calendar, not from your data. Confirm each one with your CA or on the government portal."])
    needs = reads(name)
    if name == "AnomalyIQ":
        needs = [d for d in ("sales", "expenses") if d in data]
    if not needs:
        return None
    rows, levels, reasons = [], [], []
    for ds in needs:
        if ds not in data:
            rows.append(dict(dataset=ds, rows=0))
            levels.append("Low")
            reasons.append(f"{ds}: not loaded")
            continue
        row, lvl, why = _one(ds, data[ds], today)
        rows.append(row)
        levels.append(lvl)
        reasons += why
    level = max(levels, key=lambda l: RANK[l])
    if not reasons:
        r0 = rows[0]
        reasons = [f"Enough to lean on: {sum(r['rows'] for r in rows):,} rows across {len(rows)} dataset(s)" + (f", the newest record {r0['latest_age_days']} days old." if r0.get("latest_age_days") is not None else ".")]
    return Evidence(level, reasons, pd.DataFrame(rows))
