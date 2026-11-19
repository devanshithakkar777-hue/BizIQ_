"""PeopleIQ - payroll cost, attrition and attrition risk, hiring needs, productivity.

Group-level only. A rule-based "strain signals" score (overtime, satisfaction, tenure under a year) is worked out
per person internally but is NEVER shown against an individual: it is a guess that cannot know about illness,
family trouble or workload, and a label on a named person can cost them a raise or a project. Only counts per
department are reported, and departments under MIN_GROUP people are not reported at all.
Performance ratings are deliberately not an input.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .core import AMBER, GREEN, RED, ModuleResult, inr, safe_div

FTE_HOURS = 160  # working hours in an FTE-month
MIN_GROUP = 5    # smaller groups are not reported: a figure for one to four people is a figure about named individuals


def run(hr: pd.DataFrame, fin: pd.DataFrame | None, as_of: pd.Timestamp) -> ModuleResult:
    res = ModuleResult("PeopleIQ")
    h = hr.copy()
    h["exit_date"] = pd.to_datetime(h.get("exit_date"), errors="coerce")
    active = h[h["exit_date"].isna() | (h["exit_date"] > as_of)].copy()
    year_ago = as_of - pd.Timedelta(days=365)
    exits = h[(h["exit_date"] > year_ago) & (h["exit_date"] <= as_of)]
    hc_start = int(((h["join_date"] <= year_ago) & (h["exit_date"].isna() | (h["exit_date"] > year_ago))).sum())
    avg_hc = max((hc_start + len(active)) / 2, 1)
    attrition = len(exits) / avg_hc

    payroll = float(active["monthly_salary"].sum())
    revenue_ttm = float(fin.sort_values("month")["revenue"].tail(12).sum()) if fin is not None and len(fin) else float("nan")
    payroll_pct = safe_div(payroll * 12, revenue_ttm)
    rev_per_emp = safe_div(revenue_ttm, len(active))

    # Rule-based attrition risk for active employees
    a = active
    tenure_y = (as_of - a["join_date"]).dt.days / 365
    ot = a["overtime_hours_month"].fillna(a["overtime_hours_month"].median()) if "overtime_hours_month" in a else 0
    sat = a["satisfaction_score"].fillna(7) if "satisfaction_score" in a else 7
    z = -2.2 + 0.07 * (ot - 12) + 0.55 * (6.5 - sat) + 0.6 * (tenure_y < 1).astype(float)
    a["attrition_risk"] = 1 / (1 + np.exp(-z))
    a["risk_level"] = pd.cut(a["attrition_risk"], [-1, 0.20, 0.40, 2], labels=["Low", "Medium", "High"])
    a["tenure_years"] = tenure_y
    high = a[a["risk_level"] == "High"]                    # used only to count per department; never listed by person

    # Department view
    d = a.groupby("department").agg(headcount=("employee_id", "size"), payroll=("monthly_salary", "sum"),
                                    avg_overtime=("overtime_hours_month", "mean"),
                                    avg_satisfaction=("satisfaction_score", "mean"),
                                    avg_risk=("attrition_risk", "mean"),
                                    people_with_strain_signals=("risk_level", lambda x: int((x == "High").sum())))
    d["exits_12m"] = exits.groupby("department").size().reindex(d.index).fillna(0).astype(int)
    d["attrition_pct"] = d["exits_12m"] / (d["headcount"] + d["exits_12m"] / 2)
    d["overtime_fte_equivalent"] = (a.groupby("department")["overtime_hours_month"].sum() / FTE_HOURS)
    d["payroll_share"] = d["payroll"] / d["payroll"].sum()
    d["suggested_hires"] = np.where(d["avg_overtime"] > 15, np.floor(d["overtime_fte_equivalent"] * 0.6).clip(lower=0), 0).astype(int)
    d["backfill_needed"] = d["exits_12m"]
    small = d["headcount"] < MIN_GROUP
    for col in ("avg_overtime", "avg_satisfaction", "avg_risk", "people_with_strain_signals", "overtime_fte_equivalent", "attrition_pct"):
        d[col] = d[col].astype(float).where(~small)       # not reported for groups under MIN_GROUP
    d["suggested_hires"] = d["suggested_hires"].where(~small, 0)
    d["exits_12m"] = d["exits_12m"].where(~small)
    d["backfill_needed"] = d["backfill_needed"].where(~small)
    res.tables["departments"] = d.reset_index().sort_values("avg_risk", ascending=False, na_position="last")

    # Hiring needs summary
    need = d[d["suggested_hires"] > 0]
    res.tables["hiring_needs"] = need.reset_index()[["department", "headcount", "avg_overtime",
                                                    "overtime_fte_equivalent", "suggested_hires"]]
    tenure_bands = pd.cut(a["tenure_years"], [-0.01, 1, 3, 100], labels=["< 1 yr", "1-3 yrs", "3+ yrs"])
    res.tables["tenure_mix"] = tenure_bands.value_counts().reindex(["< 1 yr", "1-3 yrs", "3+ yrs"]).reset_index()

    res.metrics = dict(headcount=len(active), monthly_payroll=payroll, attrition_rate=attrition,
                       exits_12m=len(exits), payroll_pct_revenue=payroll_pct,
                       revenue_per_employee=rev_per_emp, staff_with_strain_signals=len(high))
    res.facts = dict(headcount=len(active), monthly_payroll=inr(payroll),
                     annual_payroll=inr(payroll * 12), exits_last_12_months=len(exits),
                     attrition_rate_pct=round(attrition * 100, 1),
                     payroll_pct_of_revenue=None if pd.isna(payroll_pct) else round(payroll_pct * 100, 1),
                     revenue_per_employee=inr(rev_per_emp) if not pd.isna(rev_per_emp) else "n/a",
                     staff_with_strain_signals=len(high) if len(active) >= 2 * MIN_GROUP else None,
                     suggested_hires_total=int(d["suggested_hires"].sum()))
    rep = d[~small]
    if len(rep):
        top = rep.sort_values("avg_overtime", ascending=False).iloc[0]
        res.facts["highest_overtime_department"] = rep["avg_overtime"].idxmax()
        res.facts["highest_overtime_hours"] = round(float(top["avg_overtime"]), 1)
        res.facts["highest_attrition_department"] = rep["attrition_pct"].idxmax()
        res.facts["highest_attrition_department_pct"] = round(float(rep["attrition_pct"].max()) * 100, 1)
    if attrition > 0.30:
        res.add_flag(RED, f"Attrition is {attrition:.0%} ({len(exits)} exits in 12 months)", "Well above a typical 15-20% for retail.")
    elif attrition > 0.20:
        res.add_flag(AMBER, f"Attrition is {attrition:.0%} ({len(exits)} exits in 12 months)", "Above a typical 15-20% for retail.")
    if not pd.isna(payroll_pct) and payroll_pct > 0.25:
        res.add_flag(AMBER, f"Payroll is {payroll_pct:.0%} of revenue", "High for a retail business; check productivity.")
    for dep, r in rep.iterrows():
        if r["avg_overtime"] > 20:
            res.add_flag(AMBER, f"{dep}: average overtime {r['avg_overtime']:.0f} hours a month",
                         f"Equivalent to {r['overtime_fte_equivalent']:.1f} FTE; burnout and attrition risk.")
    if not res.flags:
        res.add_flag(GREEN, "People metrics look healthy")
    res.notes.append("Signals, not judgements: this cannot know about illness, family trouble, workload or what a manager knows. "
                     "Attrition = exits in last 12 months / average headcount. 'Strain signals' is a rule-based count (overtime, satisfaction, tenure under a year) "
                     f"shown per department only, and departments under {MIN_GROUP} people are not reported. It is never shown against a named person and performance ratings are not used. "
                     "Suggested hires cover 60% of overtime hours at 160 hours per FTE-month.")
    return res
