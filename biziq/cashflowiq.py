"""CashFlowIQ - cash forecasting, shortfall warnings, receivables / payables ageing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import AMBER, GREEN, RED, ModuleResult, inr, pct, safe_div
from .supplychainiq import holt_damped

BUCKETS = ["Not due", "1-30 days", "31-60 days", "61-90 days", "90+ days"]


def _bucket(days_overdue: int) -> str:
    if days_overdue <= 0:
        return BUCKETS[0]
    if days_overdue <= 30:
        return BUCKETS[1]
    if days_overdue <= 60:
        return BUCKETS[2]
    if days_overdue <= 90:
        return BUCKETS[3]
    return BUCKETS[4]


def open_invoices(inv: pd.DataFrame, typ: str, as_of: pd.Timestamp) -> pd.DataFrame:
    u = inv[(inv["type"] == typ) & (inv["paid_date"].isna() | (inv["paid_date"] > as_of))].copy()
    u["days_overdue"] = (as_of - u["due_date"]).dt.days
    u["bucket"] = u["days_overdue"].apply(_bucket)
    return u


def party_payment_behaviour(inv: pd.DataFrame, typ: str) -> pd.DataFrame:
    p = inv[(inv["type"] == typ) & inv["paid_date"].notna()].copy()
    if p.empty:
        return pd.DataFrame(columns=["party", "avg_days_to_pay", "avg_days_late", "late_share", "paid_invoices"])
    p["days_to_pay"] = (p["paid_date"] - p["invoice_date"]).dt.days
    p["days_late"] = (p["paid_date"] - p["due_date"]).dt.days
    p["late_clipped"] = p["days_late"].clip(lower=0)
    p["is_late"] = (p["days_late"] > 0).astype(float)
    g = p.groupby("party")
    return pd.DataFrame({
        "avg_days_to_pay": g["days_to_pay"].mean(),
        "avg_days_late": g["late_clipped"].mean(),
        "late_share": g["is_late"].mean(),
        "paid_invoices": g.size()}).reset_index()


def weekly_view(inv: pd.DataFrame, as_of: pd.Timestamp, weeks: int = 8) -> pd.DataFrame:
    beh = party_payment_behaviour(inv, "receivable").set_index("party")
    glob = beh["avg_days_to_pay"].mean() if len(beh) else 45
    ar = open_invoices(inv, "receivable", as_of)
    ap = open_invoices(inv, "payable", as_of)

    if len(ar):
        d = ar["party"].map(beh["avg_days_to_pay"]).fillna(glob)
        e = ar["invoice_date"] + pd.to_timedelta(d.astype(float), unit="D")
        late = e <= as_of          # already later than usual: assume a delay that grows with age
        extra = np.select([ar["days_overdue"] <= 30, ar["days_overdue"] <= 90], [7, 21], default=45)
        ar["expected_date"] = e.where(~late, as_of + pd.to_timedelta(extra, unit="D"))
    else:
        ar["expected_date"] = pd.NaT
    rows = []
    for w in range(weeks):
        start = as_of + pd.Timedelta(days=7 * w + 1)
        end = start + pd.Timedelta(days=6)
        inflow = ar[(ar["expected_date"] >= start) & (ar["expected_date"] <= end)]["amount"].sum() if len(ar) else 0
        due_ap = ap[(ap["due_date"] <= end) & ((ap["due_date"] >= start) if w else True)]["amount"].sum() if len(ap) else 0
        rows.append(dict(week=w + 1, week_start=start, expected_collections=float(inflow),
                         scheduled_payments=float(due_ap)))
    out = pd.DataFrame(rows)
    out["net"] = out["expected_collections"] - out["scheduled_payments"]
    return out


def run(fin: pd.DataFrame, invoices: pd.DataFrame | None, as_of: pd.Timestamp, horizon: int = 6) -> ModuleResult:
    res = ModuleResult("CashFlowIQ")
    fin = fin.sort_values("month").reset_index(drop=True)
    cash0 = float(fin["cash"].iloc[-1])
    ocf = fin["operating_cash_flow"].values
    ocf_fc, sd = holt_damped(ocf[-18:], horizon)
    # keep the forecast anchored: a damped smoothed OCF can swing, so cap at the observed range
    lo, hi = np.percentile(ocf[-18:], 5), np.percentile(ocf[-18:], 95)
    ocf_fc = np.clip(ocf_fc, lo, hi)
    capex = 0.0
    if "fixed_assets" in fin.columns and "depreciation" in fin.columns and len(fin) > 3:
        capex = float(max(0, (fin["fixed_assets"].diff() + fin["depreciation"]).tail(12).mean()))
    repay = float(max(0, -fin["long_term_debt"].diff().tail(12).mean())) if len(fin) > 3 else 0.0
    monthly_cost = float((fin["operating_expenses"] + fin["interest_expense"]).tail(3).mean())
    buffer = 1.5 * monthly_cost

    months = [fin["month"].iloc[-1] + pd.DateOffset(months=i + 1) for i in range(horizon)]
    scen = {"Base": 1.0, "Downside": 0.65, "Upside": 1.1}
    proj = pd.DataFrame({"month": months})
    for name, k in scen.items():
        c, series = cash0, []
        for i in range(horizon):
            flow = ocf_fc[i] * (k if ocf_fc[i] > 0 else (2 - k))  # downside also worsens negative flow
            c = c + flow - capex - repay
            series.append(c)
        proj[name] = series
    proj["buffer"] = buffer
    res.tables["forecast"] = proj
    res.tables["history"] = fin[["month", "cash", "operating_cash_flow"]]

    base_min, down_min = float(proj["Base"].min()), float(proj["Downside"].min())
    first_short = next((m for m, v in zip(proj["month"], proj["Downside"]) if v < buffer), None)
    res.metrics = dict(cash=cash0, buffer=buffer, base_min_cash=base_min, downside_min_cash=down_min,
                       monthly_cash_costs=monthly_cost, forecast_end_cash_base=float(proj["Base"].iloc[-1]))
    res.facts = dict(cash_today=inr(cash0), minimum_cash_buffer=inr(buffer),
                     base_case_cash_in_6_months=inr(float(proj["Base"].iloc[-1])),
                     downside_lowest_cash=inr(down_min), monthly_cash_costs=inr(monthly_cost),
                     assumed_monthly_debt_repayment=inr(repay), assumed_monthly_capex=inr(capex))
    if down_min < 0:
        res.add_flag(RED, "Cash could run out in the downside scenario",
                     f"Lowest projected cash is {inr(down_min)}.")
    elif base_min < buffer:
        res.add_flag(RED, "Base-case cash falls below the minimum buffer",
                     f"Lowest {inr(base_min)} against a buffer of {inr(buffer)}.")
    elif down_min < buffer:
        res.add_flag(AMBER, "Downside scenario takes cash below the minimum buffer",
                     f"Lowest {inr(down_min)} against a buffer of {inr(buffer)}"
                     + (f"; first shortfall in {first_short:%b %Y}." if first_short is not None else "."))

    if invoices is not None and len(invoices):
        ar = open_invoices(invoices, "receivable", as_of)
        ap = open_invoices(invoices, "payable", as_of)
        for nm, df in (("receivables", ar), ("payables", ap)):
            ag = (df.groupby("bucket")["amount"].sum().reindex(BUCKETS).fillna(0).reset_index())
            ag.columns = ["bucket", "amount"]
            ag["share"] = ag["amount"] / ag["amount"].sum() if ag["amount"].sum() else 0
            res.tables[f"{nm}_ageing"] = ag
        beh = party_payment_behaviour(invoices, "receivable")
        by_party = ar.groupby("party").agg(open_amount=("amount", "sum"),
                                           overdue_amount=("amount", lambda s: s[ar.loc[s.index, "days_overdue"] > 0].sum()),
                                           oldest_days_overdue=("days_overdue", "max")).reset_index()
        res.tables["receivables_by_party"] = by_party.merge(beh, on="party", how="left") \
            .sort_values("overdue_amount", ascending=False)
        res.tables["open_receivables"] = ar.sort_values("days_overdue", ascending=False)
        res.tables["open_payables"] = ap.sort_values("due_date")
        wk = weekly_view(invoices, as_of)
        res.tables["weekly_view"] = wk
        ar_total, ar_over = float(ar["amount"].sum()), float(ar[ar["days_overdue"] > 0]["amount"].sum())
        ar_90 = float(ar[ar["days_overdue"] > 90]["amount"].sum())
        ap_total, ap_over = float(ap["amount"].sum()), float(ap[ap["days_overdue"] > 0]["amount"].sum())
        res.metrics.update(receivables_open=ar_total, receivables_overdue=ar_over, receivables_90plus=ar_90,
                           payables_open=ap_total, payables_overdue=ap_over)
        res.facts.update(receivables_open=inr(ar_total), receivables_overdue=inr(ar_over),
                         receivables_overdue_pct=round(safe_div(ar_over, ar_total, 0) * 100, 1),
                         receivables_over_90_days=inr(ar_90), payables_open=inr(ap_total),
                         payables_overdue=inr(ap_over),
                         next_8_weeks_collections=inr(float(wk["expected_collections"].sum())),
                         next_8_weeks_payments=inr(float(wk["scheduled_payments"].sum())))
        if len(res.tables["receivables_by_party"]):
            worst = res.tables["receivables_by_party"].iloc[0]
            res.facts["largest_overdue_party"] = worst["party"]
            res.facts["largest_overdue_amount"] = inr(float(worst["overdue_amount"]))
        share = safe_div(ar_over, ar_total, 0)
        if ar_90 > 0 and ar_90 / max(ar_total, 1) > 0.10:
            res.add_flag(RED, f"{inr(ar_90)} of receivables are over 90 days overdue",
                         "Escalate collection and consider provisioning.")
        elif share > 0.25:
            res.add_flag(AMBER, f"{share:.0%} of open receivables are overdue ({inr(ar_over)})",
                         f"Largest: {res.facts.get('largest_overdue_party', 'n/a')}.")
        if ap_over > 0:
            res.add_flag(AMBER, f"{inr(ap_over)} of payables are overdue",
                         "Late supplier payments can hurt terms and supply.")
        if wk["net"].cumsum().min() < -0.5 * cash0:
            res.add_flag(AMBER, "Scheduled payments exceed expected collections in the next 8 weeks",
                         f"Net {inr(float(wk['net'].sum()))}.")
    if not res.flags:
        res.add_flag(GREEN, "Cash position is comfortable across scenarios")
    res.notes.append("Forecast = damped-trend smoothing of operating cash flow, less assumed capex and "
                     "debt repayment taken from recent history. Downside = 65% of forecast inflow. "
                     "Minimum buffer = 1.5 months of operating expenses plus interest.")
    return res
