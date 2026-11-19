"""RiskIQ - credit risk of customers and suppliers, anomaly / possible-fraud detection,
and concentration risk."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import AMBER, GREEN, RED, ModuleResult, inr, safe_div, score_between


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        mad = s.std() * 0.6745 if s.std() else 1.0
    return 0.6745 * (s - med) / mad


def hhi(shares: pd.Series) -> float:
    s = shares / shares.sum()
    return float((s ** 2).sum())


def customer_credit_risk(inv: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    r = inv[inv["type"] == "receivable"].copy()
    paid = r[r["paid_date"].notna()].copy()
    paid["days_late"] = (paid["paid_date"] - paid["due_date"]).dt.days.clip(lower=0)
    paid["is_late"] = (paid["days_late"] > 0).astype(float)
    open_ = r[r["paid_date"].isna()].copy()
    open_["overdue_days"] = (as_of - open_["due_date"]).dt.days.clip(lower=0)
    open_["overdue_amt"] = open_["amount"].where(open_["overdue_days"] > 0, 0.0)
    g = paid.groupby("party").agg(avg_days_late=("days_late", "mean"), late_share=("is_late", "mean"))
    o = open_.groupby("party").agg(exposure=("amount", "sum"), overdue_amount=("overdue_amt", "sum"),
                                   max_overdue_days=("overdue_days", "max"))
    t = g.join(o, how="outer").fillna(0).reset_index()
    t["overdue_share"] = t["overdue_amount"] / t["exposure"].replace(0, np.nan)
    t["overdue_share"] = t["overdue_share"].fillna(0)
    t["risk_score"] = (0.35 * t["avg_days_late"].apply(lambda v: score_between(v, 0, 45)) +
                       0.25 * t["late_share"].apply(lambda v: score_between(v, 0.1, 0.9)) +
                       0.25 * t["overdue_share"].apply(lambda v: score_between(v, 0, 0.8)) +
                       0.15 * t["max_overdue_days"].apply(lambda v: score_between(v, 0, 90)))
    t["rating"] = pd.cut(t["risk_score"], [-1, 30, 55, 101], labels=["Low", "Medium", "High"])
    return t.sort_values("risk_score", ascending=False)


def supplier_risk(deliv: pd.DataFrame) -> pd.DataFrame:
    d = deliv.copy()
    d["late"] = (d["delivered_date"] > d["promised_date"]).astype(int)
    d["spend"] = d["qty_received"] * d["unit_price"]
    g = d.groupby("supplier")
    t = pd.DataFrame({"spend": g["spend"].sum(), "late_rate": g["late"].mean(),
                      "defect_rate": g["defective_units"].sum() / g["qty_received"].sum().clip(lower=1),
                      "fill_shortfall": 1 - g["qty_received"].sum() / g["qty_ordered"].sum()})
    t["spend_share"] = t["spend"] / t["spend"].sum()
    t["risk_score"] = (0.40 * t["late_rate"].apply(lambda v: score_between(v, 0.05, 0.5)) +
                       0.30 * t["defect_rate"].apply(lambda v: score_between(v, 0.005, 0.05)) +
                       0.15 * t["fill_shortfall"].apply(lambda v: score_between(v, 0, 0.1)) +
                       0.15 * t["spend_share"].apply(lambda v: score_between(v, 0.2, 0.5)))
    t["rating"] = pd.cut(t["risk_score"], [-1, 30, 55, 101], labels=["Low", "Medium", "High"])
    return t.sort_values("risk_score", ascending=False).reset_index()


def single_source(deliv: pd.DataFrame) -> pd.DataFrame:
    d = deliv.copy()
    d["spend"] = d["qty_received"] * d["unit_price"]
    p = d.groupby(["product", "supplier"])["spend"].sum().reset_index()
    top = p.sort_values("spend", ascending=False).drop_duplicates("product")
    tot = p.groupby("product")["spend"].sum()
    top["share"] = top["spend"].values / tot.reindex(top["product"]).values
    return top[top["share"] >= 0.7][["product", "supplier", "share"]].sort_values("share", ascending=False)


def anomalies(fin: pd.DataFrame | None, sales: pd.DataFrame | None, inv: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    if fin is not None and len(fin) >= 8:
        f = fin.sort_values("month").reset_index(drop=True)
        series = {"Operating expenses": f["operating_expenses"], "COGS % of revenue": f["cogs"] / f["revenue"],
                  "Revenue": f["revenue"], "Interest expense": f["interest_expense"]}
        for name, s in series.items():
            # remove trend with a centred rolling median, then test the residual
            resid = s - s.rolling(5, center=True, min_periods=2).median()
            z = robust_z(resid)
            rel = (resid / (s - resid)).abs()
            for i in np.where((z.abs() > 3.5) & (rel >= 0.10))[0]:  # statistically odd AND material (10%+)
                direction = "above" if resid.iloc[i] > 0 else "below"
                rows.append(dict(source="Financials", date=f.loc[i, "month"], item=name,
                                 value=float(s.iloc[i]), expected=float(s.iloc[i] - resid.iloc[i]),
                                 severity="High" if abs(z.iloc[i]) > 5 else "Medium",
                                 note=f"{name} is {abs(resid.iloc[i] / (s.iloc[i] - resid.iloc[i])):.0%} {direction} its local norm"))
    if sales is not None and len(sales):
        daily = sales.set_index("date")["revenue"].resample("D").sum()
        # Baseline = median of days 15-56 back, so a developing slump cannot contaminate it
        base = daily.shift(15).rolling(42, min_periods=21).median()
        flagged = daily[daily < 0.6 * base].tail(60)
        # group consecutive flagged days into a single episode
        if len(flagged):
            ep, start, prev = [], flagged.index[0], flagged.index[0]
            for d in flagged.index[1:]:
                if (d - prev).days > 2:
                    ep.append((start, prev))
                    start = d
                prev = d
            ep.append((start, prev))
            for a, b in ep:
                seg = daily[a:b]
                exp = base[a:b]
                if len(seg) >= 4 and seg.mean() < 0.7 * exp.mean():
                    rows.append(dict(source="Daily sales", date=a, item=f"Sales drop {a:%d %b} to {b:%d %b}",
                                     value=float(seg.mean()), expected=float(exp.mean()),
                                     severity="High" if seg.mean() < 0.6 * exp.mean() else "Medium",
                                     note=f"Average {inr(float(seg.mean()), False)}/day vs a normal "
                                          f"{inr(float(exp.mean()), False)}/day over {len(seg)} days"))
    if inv is not None and len(inv):
        for typ in ("receivable", "payable"):
            x = inv[inv["type"] == typ]
            q1, q3 = x["amount"].quantile([0.25, 0.75])
            hi = q3 + 3 * (q3 - q1)
            for _, r in x[x["amount"] > hi].iterrows():
                rows.append(dict(source="Invoices", date=r["invoice_date"], item=f"{r['invoice_id']} ({r['party']})",
                                 value=float(r["amount"]), expected=float(x["amount"].median()), severity="Medium",
                                 note=f"{typ.title()} amount is unusually large"))
            dup = x[x.duplicated(["party", "amount", "invoice_date"], keep=False)]
            for _, r in dup.iterrows():
                rows.append(dict(source="Invoices", date=r["invoice_date"], item=f"{r['invoice_id']} ({r['party']})",
                                 value=float(r["amount"]), expected=float("nan"), severity="High",
                                 note="Possible duplicate invoice (same party, amount and date)"))
    cols = ["source", "date", "item", "value", "expected", "severity", "note"]
    return pd.DataFrame(rows, columns=cols).sort_values("date", ascending=False).reset_index(drop=True)


def run(fin=None, sales=None, invoices=None, deliveries=None, as_of: pd.Timestamp | None = None) -> ModuleResult:
    res = ModuleResult("RiskIQ")
    as_of = as_of or pd.Timestamp.today().normalize()
    score_parts = []

    if invoices is not None and len(invoices):
        cr = customer_credit_risk(invoices, as_of)
        res.tables["customer_credit_risk"] = cr
        high = cr[cr["rating"] == "High"]
        res.facts["high_risk_customers"] = len(high)
        res.facts["credit_exposure_to_high_risk"] = inr(float(high["exposure"].sum()))
        if len(high):
            res.facts["riskiest_customer"] = high.iloc[0]["party"]
            res.add_flag(RED if high["exposure"].sum() > 0.25 * cr["exposure"].sum() else AMBER,
                         f"{len(high)} customer(s) rated high credit risk",
                         f"{', '.join(high['party'].head(3))}; exposure {inr(float(high['exposure'].sum()))}.")
        score_parts.append(100 - float(cr["risk_score"].mean()))
        top_ar = cr.sort_values("exposure", ascending=False)
        if top_ar["exposure"].sum() > 0:
            top_share = float(top_ar["exposure"].iloc[0] / top_ar["exposure"].sum())
            res.facts["largest_receivable_share_pct"] = round(top_share * 100, 1)
            if top_share > 0.40:
                res.add_flag(AMBER, f"{top_ar.iloc[0]['party']} is {top_share:.0%} of open receivables",
                             "Customer concentration in credit exposure.")

    if deliveries is not None and len(deliveries):
        sr = supplier_risk(deliveries)
        res.tables["supplier_risk"] = sr
        high_s = sr[sr["rating"] == "High"]
        res.facts["high_risk_suppliers"] = len(high_s)
        if len(high_s):
            res.facts["riskiest_supplier"] = high_s.iloc[0]["supplier"]
            res.add_flag(AMBER, f"{len(high_s)} supplier(s) rated high risk", ", ".join(high_s["supplier"]))
        ss = single_source(deliveries)
        res.tables["single_source"] = ss
        res.facts["single_source_products"] = len(ss)
        if len(ss):
            res.add_flag(AMBER, f"{len(ss)} product(s) rely on a single supplier (70%+ of spend)",
                         ", ".join(ss["product"].head(4)))
        res.facts["top_supplier_spend_share_pct"] = round(float(sr["spend_share"].max()) * 100, 1)
        score_parts.append(100 - float(sr["risk_score"].mean()))

    an = anomalies(fin, sales, invoices)
    res.tables["anomalies"] = an
    res.facts["anomalies_detected"] = len(an)
    res.facts["high_severity_anomalies"] = int((an["severity"] == "High").sum()) if len(an) else 0
    for _, r in an.head(5).iterrows():
        res.add_flag(RED if r["severity"] == "High" else AMBER, f"Anomaly: {r['item']}", r["note"])
    if len(an):
        res.facts["latest_anomaly"] = f"{an.iloc[0]['item']}: {an.iloc[0]['note']}"
    score_parts.append(100 - min(len(an) * 8, 50))

    conc = {}
    if sales is not None and len(sales):
        cust = sales.groupby("customer_id")["revenue"].sum()
        prod = sales.groupby("product")["revenue"].sum()
        conc = {"Customers (HHI)": hhi(cust), "Products (HHI)": hhi(prod),
                "Top 10% of customers' revenue share": float(cust.sort_values(ascending=False)
                                                              .head(max(1, len(cust) // 10)).sum() / cust.sum()),
                "Top product's revenue share": float(prod.max() / prod.sum())}
        res.facts["top_product_revenue_share_pct"] = round(conc["Top product's revenue share"] * 100, 1)
        res.facts["top10pct_customers_revenue_share_pct"] = round(conc["Top 10% of customers' revenue share"] * 100, 1)
        if conc["Products (HHI)"] > 0.25:
            res.add_flag(AMBER, "High product concentration", f"HHI {conc['Products (HHI)']:.2f}.")
    res.tables["concentration"] = pd.DataFrame([{"measure": k, "value": v} for k, v in conc.items()])

    res.score = float(np.mean(score_parts)) if score_parts else None
    res.metrics = dict(risk_health_score=res.score, anomalies=len(an))
    res.facts["risk_health_score"] = None if res.score is None else round(res.score, 1)
    if not res.flags:
        res.add_flag(GREEN, "No material risks detected")
    res.notes.append("Anomalies use a robust (median / MAD) z-score after removing local trend; "
                     "flagged items are prompts to investigate, not proof of fraud. "
                     "Credit risk scores combine payment lateness, overdue share and ageing.")
    return res
