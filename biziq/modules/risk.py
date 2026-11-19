"""Risk modules: VendorRiskIQ, FraudIQ."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import AMBER, GREEN, RED, ModuleResult, inr, safe_div
from ..registry import Context, ModuleSpec, register
from ._common import clean_facts, pct1, r1


# ============================================================================== VendorRiskIQ
def run_vendor_risk(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("VendorRiskIQ")
    d = data["supplier_deliveries"].copy()
    d["spend"] = d["qty_ordered"] * d["unit_price"]
    d["late_days"] = (d["delivered_date"] - d["promised_date"]).dt.days
    d["late"] = d["late_days"] > 0
    d["short"] = d["qty_received"] < d["qty_ordered"] * 0.98
    d["defect_units"] = pd.to_numeric(d["defective_units"], errors="coerce").fillna(0) if "defective_units" in d else 0
    sup = d.groupby("supplier").agg(orders=("supplier", "size"), spend=("spend", "sum"), late_pct=("late", "mean"), short_pct=("short", "mean"),
                                    defect_units=("defect_units", "sum"), received=("qty_received", "sum"), avg_delay=("late_days", lambda x: float(x[x > 0].mean()) if (x > 0).any() else 0.0),
                                    products=("product", "nunique"))
    tot = float(sup["spend"].sum())
    sup["spend_share_pct"] = 100 * sup["spend"] / tot if tot else np.nan
    sup["defect_pct"] = 100 * sup["defect_units"] / sup["received"].replace(0, np.nan)
    sup["late_pct"] *= 100
    sup["short_pct"] *= 100
    per_prod = d.groupby("product")["supplier"].nunique()
    single = per_prod[per_prod == 1].index
    sp = d.groupby("product")["spend"].sum()
    single_spend = float(sp.reindex(single).sum())
    only = d[d["product"].isin(single)].groupby("supplier").agg(sole_source_products=("product", "nunique"))
    sup["sole_source_products"] = only["sole_source_products"].reindex(sup.index).fillna(0).astype(int)
    # explainable score: each part 0-100, weights below
    score = (np.clip(sup["spend_share_pct"] / 40, 0, 1) * 35 + np.clip(sup["late_pct"] / 40, 0, 1) * 25 + np.clip(sup["defect_pct"].fillna(0) / 5, 0, 1) * 20
             + np.clip(sup["short_pct"] / 20, 0, 1) * 10 + np.clip(sup["sole_source_products"] / 3, 0, 1) * 10)
    sup["risk_score"] = score.fillna(0).round(0)
    sup["rating"] = np.select([sup["risk_score"] >= 55, sup["risk_score"] >= 35], ["high", "medium"], "low")
    res.tables["suppliers"] = sup.round(1).reset_index().sort_values("risk_score", ascending=False)
    res.tables["sole_source_products"] = pd.DataFrame({"product": single, "supplier": [d.loc[d["product"] == p, "supplier"].iloc[0] for p in single],
                                                       "spend": [float(sp[p]) for p in single]}).sort_values("spend", ascending=False)
    share = (sup["spend"] / tot) if tot else sup["spend"] * 0
    hhi = float((share ** 2).sum() * 10000)
    top = sup["spend"].idxmax()
    high = sup[sup["rating"] == "high"]
    res.facts = clean_facts(dict(
        suppliers=len(sup), total_spend=inr(tot), top_supplier=str(top), top_supplier_share_pct=r1(float(sup.loc[top, "spend_share_pct"])), concentration_hhi=int(round(hhi)),
        sole_source_products=len(single), sole_source_spend_pct=pct1(safe_div(single_spend, tot)), high_risk_suppliers=len(high),
        highest_risk_supplier=str(sup["risk_score"].idxmax()), highest_risk_score=int(sup["risk_score"].max()),
        average_late_delivery_pct=r1(float(d["late"].mean() * 100))))
    if sup["spend_share_pct"].max() > 40:
        res.add_flag(RED if sup["spend_share_pct"].max() > 60 else AMBER, f"{top} supplies {sup.loc[top, 'spend_share_pct']:.0f}% of your purchase spend")
    if len(single) and safe_div(single_spend, tot) > 0.25:
        res.add_flag(AMBER, f"{len(single)} product(s) have a single supplier, {safe_div(single_spend, tot):.0%} of spend")
    if len(high):
        res.add_flag(AMBER, f"{len(high)} supplier(s) rate high risk: {', '.join(map(str, high['risk_score'].sort_values(ascending=False).index[:3]))}")
    if not res.flags:
        res.add_flag(GREEN, "Supply base is spread out and reliable")
    res.notes.append("Risk score (0-100) = spend share 35 + late deliveries 25 + defects 20 + short deliveries 10 + sole-source products 10, each capped. The weights are a judgement, shown so you can "
                     "argue with them. It uses YOUR delivery history only: no financial-health, geography, sanctions or news signals. HHI above 2500 means highly concentrated spend. "
                     "ProcureIQ covers price and delivery performance; this one is about dependence.")
    return res


def rec_vendor_risk(f, flags):
    r = []
    if (f.get("top_supplier_share_pct") or 0) > 40:
        r.append(f"Qualify a second source for what {f['top_supplier']} supplies ({f['top_supplier_share_pct']}% of spend). Start with the products in the sole-source table.")
    if f.get("sole_source_products"):
        r.append(f"{f['sole_source_products']} product(s) depend on one supplier ({f.get('sole_source_spend_pct')}% of spend); hold extra safety stock until a backup is approved.")
    if f.get("high_risk_suppliers"):
        r.append(f"Review the {f['high_risk_suppliers']} high-risk supplier(s) with them, starting with {f.get('highest_risk_supplier')} (score {f.get('highest_risk_score')}).")
    return r or ["Supply base looks balanced; re-check each quarter."]


# ============================================================================== FraudIQ
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def run_fraud(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("FraudIQ")
    e = data["expenses"].copy()
    e = e[e["amount"] > 0]
    thr = ctx.settings.get("approval_threshold", 50000)
    ind = []                                                      # (indicator, count, amount, how to read it)
    # 1 amounts just under the approval limit
    near = e[(e["amount"] >= thr * 0.9) & (e["amount"] < thr)]
    above = e[(e["amount"] >= thr) & (e["amount"] < thr * 1.1)]
    res.tables["just_below_limit"] = near.sort_values("amount", ascending=False)[["expense_id", "date", "vendor", "amount"]
                                                                                  + [c for c in ("approved_by", "category") if c in e]].head(100)
    # 2 split payments: same vendor, same day, several payments each under the limit, adding to over the limit
    g = e[e["amount"] < thr].groupby(["vendor", "date"]).agg(payments=("amount", "size"), total=("amount", "sum")).reset_index()
    split = g[(g["payments"] >= 2) & (g["total"] >= thr)].sort_values("total", ascending=False)
    res.tables["possible_split_payments"] = split.head(100)
    # 3 round numbers
    rnd = e[(e["amount"] >= 5000) & (e["amount"] % 1000 == 0)]
    rv = e.groupby("vendor").agg(n=("amount", "size"), round_n=("amount", lambda x: int(((x >= 5000) & (x % 1000 == 0)).sum())))
    rv["round_share_pct"] = 100 * rv["round_n"] / rv["n"]
    rv = rv[(rv["n"] >= 5) & (rv["round_share_pct"] >= 60)].reset_index().sort_values("round_share_pct", ascending=False)
    res.tables["vendors_mostly_round_amounts"] = rv.head(50)
    # 4 weekend payments
    wk = e[pd.to_datetime(e["date"]).dt.dayofweek >= 6]
    wk_share = safe_div(len(wk), len(e))
    # 5 missing approvals above the limit
    unappr = e.iloc[0:0]
    if "approved_by" in e:
        unappr = e[(e["amount"] >= thr) & (e["approved_by"].isna() | (e["approved_by"].astype(str).str.strip() == ""))]
        res.tables["large_without_approver"] = unappr.sort_values("amount", ascending=False)[["expense_id", "date", "vendor", "amount"]].head(100)
    # 6 Benford (only with enough data)
    first = e["amount"][e["amount"] >= 10].astype(str).str.replace(".", "", regex=False).str.lstrip("0").str[0].astype(int)
    benford_mad = None
    if len(first) >= 300:
        obs = first.value_counts(normalize=True).reindex(range(1, 10)).fillna(0).to_numpy()
        benford_mad = float(np.abs(obs - BENFORD).mean())
        res.tables["first_digit_check"] = pd.DataFrame({"digit": range(1, 10), "observed_pct": (obs * 100).round(1), "expected_pct": (BENFORD * 100).round(1)})
    res.metrics = dict(near_limit=len(near), split_groups=len(split))
    res.facts = clean_facts(dict(
        transactions_checked=len(e), approval_threshold=inr(thr), payments_just_below_limit=len(near), payments_just_above_limit=len(above),
        possible_split_payment_groups=len(split), split_payments_value=inr(float(split["total"].sum())) if len(split) else None,
        vendors_with_mostly_round_amounts=len(rv), weekend_payment_share_pct=pct1(wk_share), large_payments_without_approver=len(unappr),
        large_without_approver_value=inr(float(unappr["amount"].sum())) if len(unappr) else None,
        first_digit_deviation=r1(benford_mad, 4) if benford_mad is not None else None,
        first_digit_conformity=("close" if benford_mad < 0.012 else "acceptable" if benford_mad < 0.015 else "nonconforming") if benford_mad is not None else None))
    expected_near = len(e[(e["amount"] >= thr * 0.8) & (e["amount"] < thr * 0.9)])
    if len(near) >= 3 and len(near) > 2 * max(1, expected_near):
        res.add_flag(AMBER, f"{len(near)} payments sit just under the {inr(thr)} approval limit, about twice as many as just below that band")
    if len(split):
        res.add_flag(AMBER, f"{len(split)} vendor-day(s) show several payments that together pass the approval limit")
    if len(unappr):
        res.add_flag(RED if len(unappr) >= 3 else AMBER, f"{len(unappr)} payment(s) over the limit have no approver recorded")
    if benford_mad is not None and benford_mad >= 0.015:
        res.add_flag(AMBER, "First digits of expense amounts deviate from the expected pattern")
    if len(rv):
        res.add_flag(AMBER, f"{len(rv)} vendor(s) are paid mostly in round thousands")
    if not res.flags:
        res.add_flag(GREEN, "No unusual payment patterns found by these checks")
    res.notes.append("These are INDICATORS TO REVIEW, not findings of fraud: each has innocent explanations (fixed fees, retainers, genuine batch payments, vendors who quote in round numbers). "
                     "Duplicate payments are covered by ExpenseIQ. The first-digit (Benford) test needs 300+ transactions and is unreliable on prices that are set, not natural. "
                     f"The approval limit is assumed at {inr(thr)}; set yours in settings. Never act against an individual on this output alone.")
    return res


def rec_fraud(f, flags):
    r = []
    if f.get("large_payments_without_approver"):
        r.append(f"Find the approver for {f['large_payments_without_approver']} large payment(s) worth {f.get('large_without_approver_value')}; if there is none on record, treat it as a control gap and fix the process.")
    if f.get("payments_just_below_limit"):
        r.append(f"Look at the {f['payments_just_below_limit']} payments just under {f.get('approval_threshold')}: confirm each has a real invoice and that nobody is steering around approval.")
    if f.get("possible_split_payment_groups"):
        r.append("Check the split-payment groups against purchase orders; if one invoice was paid in parts, require a single approval for the whole amount.")
    return r or ["No unusual patterns; keep approvals and receipts consistent so these checks stay quiet."]


register(ModuleSpec("VendorRiskIQ", "Risk and control", "Supplier dependence, sole-source products and a transparent risk score.", ["supplier_deliveries"], run_vendor_risk, rec_vendor_risk, audience="B2B", phase="3"))
register(ModuleSpec("FraudIQ", "Risk and control", "Payment patterns that deserve a second look: split payments, limit-hugging, missing approvers.", ["expenses"], run_fraud, rec_fraud, audience="Both", phase="3"))
