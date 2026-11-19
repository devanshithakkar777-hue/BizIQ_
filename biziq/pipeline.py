"""Pipeline: run every module for which data is available, in dependency order.

Each module runs in isolation: a failure in one never stops the others; it becomes a
red flag on that module so the problem is visible rather than hidden.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import (brandiq, cashflowiq, clientiq, investiq, peopleiq, procureiq, registry, riskiq, salesiq,
               supplychainiq, taxiq, wealthiq)
from .core import RED, Flag, ModuleResult

MODULE_ORDER = ["WealthIQ", "SalesIQ", "ClientIQ", "SupplyChainIQ", "CashFlowIQ", "TaxIQ", "RiskIQ",
                "InvestIQ", "PeopleIQ", "BrandIQ", "ProcureIQ"]


def pyify(obj):
    """Convert numpy / pandas scalars to plain Python so facts serialise cleanly."""
    if isinstance(obj, dict):
        return {k: pyify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [pyify(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


@dataclass
class PipelineResult:
    modules: dict[str, ModuleResult] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    failed: dict[str, Flag] = field(default_factory=dict)
    as_of: pd.Timestamp | None = None
    total_seconds: float = 0.0

    def all_flags(self) -> list[Flag]:
        order = {"red": 0, "amber": 1, "green": 2}
        flags = [f for m in self.modules.values() for f in m.flags] + list(self.failed.values())
        return sorted(flags, key=lambda f: order[f.level])

    def facts(self) -> dict[str, dict]:
        return {k: v.facts for k, v in self.modules.items()}


def infer_as_of(data: dict[str, pd.DataFrame]) -> pd.Timestamp:
    cands = []
    if "sales" in data:
        cands.append(data["sales"]["date"].max())
    if "financials" in data:
        cands.append(data["financials"]["month"].max() + pd.offsets.MonthEnd(0))
    if "invoices" in data:
        cands.append(data["invoices"]["invoice_date"].max())
    cands = [c for c in cands if pd.notna(c)]
    return pd.Timestamp(max(cands)).normalize() if cands else pd.Timestamp.today().normalize()


def run_all(data: dict[str, pd.DataFrame], as_of: pd.Timestamp | None = None,
            risk_appetite: str = "Balanced", tax_cfg: taxiq.TaxConfig | None = None,
            today: pd.Timestamp | None = None, settings: dict | None = None) -> PipelineResult:
    t_all = time.perf_counter()
    out = PipelineResult()
    data = {k: v for k, v in data.items() if v is not None and len(v)}      # an empty table is the same as a missing one
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else infer_as_of(data)
    out.as_of = as_of
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    fin, sales = data.get("financials"), data.get("sales")
    inv_t, deliv = data.get("inventory"), data.get("supplier_deliveries")
    invoices = data.get("invoices")

    def step(name, needs, fn):
        missing = [n for n in needs if n not in data]
        if missing:
            out.skipped[name] = "needs " + ", ".join(missing)
            return None
        t0 = time.perf_counter()
        try:
            res = fn()
        except Exception as e:  # keep going; surface the failure, never feed half-built results downstream
            if isinstance(e, ValueError) and str(e).startswith("needs "):      # inputs absent: a skip, not a failure
                out.skipped[name] = str(e)
                out.timings[name] = time.perf_counter() - t0
                return None
            out.failed[name] = Flag(RED, f"{name} could not run: {type(e).__name__}: {e}",
                                    traceback.format_exc(limit=2), name)
            out.skipped[name] = f"failed: {type(e).__name__}: {e}"
            out.timings[name] = time.perf_counter() - t0
            return None
        res.facts = pyify(res.facts)
        res.metrics = pyify(res.metrics)
        out.modules[name] = res
        out.timings[name] = time.perf_counter() - t0
        return res

    wealth = step("WealthIQ", ["financials"], lambda: wealthiq.run(fin))
    client = step("ClientIQ", ["sales"],
                  lambda: clientiq.run(sales, gross_margin=max(0.15, (wealth.facts.get("gross_margin_pct", 32) or 32) / 100)
                                       if wealth else 0.32))
    segs = client.tables["customers"].set_index("customer_id") if client and "customers" in client.tables else None
    step("SalesIQ", ["sales"], lambda: salesiq.run(sales, segs))
    sc = step("SupplyChainIQ", ["sales", "inventory"], lambda: supplychainiq.run(sales, inv_t, deliv, fin))
    step("CashFlowIQ", ["financials"], lambda: cashflowiq.run(fin, invoices, as_of))
    tax = step("TaxIQ", ["financials"], lambda: taxiq.run(fin, as_of, tax_cfg, today=today))
    step("RiskIQ", [], lambda: riskiq.run(fin, sales, invoices, deliv, as_of))
    step("InvestIQ", ["financials"], lambda: investiq.run(
        fin, data.get("portfolio"), risk_appetite,
        float(tax.metrics.get("income_tax_estimate", 0)) if tax else 0.0))
    step("PeopleIQ", ["hr"], lambda: peopleiq.run(data["hr"], fin, as_of))
    step("BrandIQ", [], lambda: brandiq.run(data.get("social"), data.get("reviews"), as_of)
         if ("social" in data or "reviews" in data) else (_ for _ in ()).throw(ValueError("needs social or reviews")))
    step("ProcureIQ", ["supplier_deliveries"], lambda: procureiq.run(
        deliv, as_of, sc.tables.get("reorder") if sc else None,
        dict(zip(inv_t["product"], inv_t["lead_time_days"])) if inv_t is not None else {}))
    # Platform modules registered through biziq.registry run after the built-ins, each isolated like the rest.
    ctx = registry.Context(as_of=as_of, today=today, modules=out.modules, settings=settings or {})
    specs = registry.all_specs()

    def run_spec(spec):
        res = spec.run(data, ctx)
        problems = registry.check_result(spec, res)
        if problems:
            raise ValueError("module contract broken: " + "; ".join(problems))
        return res

    for spec in specs:
        step(spec.name, spec.needs, lambda spec=spec: run_spec(spec))
    order = MODULE_ORDER + [s.name for s in specs]
    out.modules = {k: out.modules[k] for k in order if k in out.modules}
    out.total_seconds = time.perf_counter() - t_all
    return out
