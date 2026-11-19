"""TaxIQ - GST and income tax deadlines, liability estimates, compliance gaps, reminders.

IMPORTANT: Rates and due dates below are configurable assumptions for an Indian company
and MUST be verified against current law and CBIC / CBDT notifications (due dates are
extended from time to time). Outputs are estimates, not tax advice.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .core import AMBER, GREEN, RED, ModuleResult, inr, safe_div


@dataclass
class TaxConfig:
    gst_rate: float = 0.18              # blended GST rate on sales (assumption)
    itc_eligible_share: float = 0.90    # share of taxable purchases on which ITC is claimable
    taxable_opex_share: float = 0.50    # share of operating expenses that carry GST
    income_tax_rate: float = 0.2517     # 22% + 10% surcharge + 4% cess (s.115BAA) - verify applicability
    gstr1_day: int = 11
    gstr3b_day: int = 20
    tds_day: int = 7
    pf_esi_day: int = 15


ADVANCE_TAX = [(6, 15, 0.15), (9, 15, 0.45), (12, 15, 0.75), (3, 15, 1.00)]  # month, day, cumulative share


def fy_bounds(as_of: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    y = as_of.year if as_of.month >= 4 else as_of.year - 1
    return pd.Timestamp(y, 4, 1), pd.Timestamp(y + 1, 3, 31)


def deadlines(as_of: pd.Timestamp, cfg: TaxConfig, days: int = 90) -> pd.DataFrame:
    rows = []
    end = as_of + pd.Timedelta(days=days)
    m = pd.Timestamp(as_of.year, as_of.month, 1)
    while m <= end:
        for day, name, cat in ((cfg.tds_day, "TDS deposit (previous month)", "TDS"),
                               (cfg.gstr1_day, "GSTR-1 (monthly filer)", "GST"),
                               (cfg.pf_esi_day, "PF / ESI contribution", "Payroll"),
                               (cfg.gstr3b_day, "GSTR-3B and GST payment", "GST")):
            d = pd.Timestamp(m.year, m.month, day)
            if as_of <= d <= end:
                rows.append(dict(due_date=d, obligation=name, category=cat))
        m += pd.DateOffset(months=1)
    for yr in (as_of.year, as_of.year + 1):
        for mo, dy, share in ADVANCE_TAX:
            d = pd.Timestamp(yr, mo, dy)
            if as_of <= d <= end:
                rows.append(dict(due_date=d, obligation=f"Advance tax instalment ({share:.0%} cumulative)",
                                 category="Income tax"))
        d = pd.Timestamp(yr, 10, 31)
        if as_of <= d <= end:
            rows.append(dict(due_date=d, obligation="Income tax return due (companies, audit cases)",
                             category="Income tax"))
    out = pd.DataFrame(rows).sort_values("due_date").reset_index(drop=True)
    out["days_left"] = (out["due_date"] - as_of).dt.days
    out["urgency"] = pd.cut(out["days_left"], [-1, 7, 21, 10_000], labels=["Due within 7 days", "Due soon", "Upcoming"])
    return out


def run(fin: pd.DataFrame, as_of: pd.Timestamp, cfg: TaxConfig | None = None,
        today: pd.Timestamp | None = None) -> ModuleResult:
    """as_of = end of the data; today = real date used for the deadline calendar."""
    cfg = cfg or TaxConfig()
    today = pd.Timestamp(today).normalize() if today is not None else as_of
    res = ModuleResult("TaxIQ")
    fin = fin.sort_values("month").reset_index(drop=True)
    start, end = fy_bounds(as_of)
    f = fin.copy()
    f["pbt"] = f["net_income"] + f["tax"]
    ytd = f[(f["month"] >= start) & (f["month"] <= as_of)]
    n = len(ytd)
    if n == 0:
        res.add_flag(AMBER, "No financials in the current financial year", "Upload this year's data.")
        return res
    remaining = 12 - n
    run_rate = float(f["pbt"].tail(3).mean())
    pbt_fy = float(ytd["pbt"].sum()) + max(remaining, 0) * run_rate
    est_tax = max(pbt_fy, 0) * cfg.income_tax_rate
    paid_ytd = float(ytd["tax"].sum())

    # Advance tax: required cumulative by each instalment already passed vs tax recorded through that month
    rows = []
    for mo, dy, share in ADVANCE_TAX:
        yr = start.year if mo >= 4 else start.year + 1
        due = pd.Timestamp(yr, mo, dy)
        required = est_tax * share
        paid_by = float(f[(f["month"] >= start) & (f["month"] <= due)]["tax"].sum())
        passed = due <= as_of
        rows.append(dict(instalment_due=due, cumulative_share=share, required_cumulative=required,
                         tax_recorded_by_date=paid_by if passed else float("nan"),
                         shortfall=max(required - paid_by, 0) if passed else float("nan"),
                         status=("Met" if paid_by >= required * 0.98 else "Shortfall") if passed else "Upcoming"))
    adv = pd.DataFrame(rows)
    res.tables["advance_tax"] = adv

    # GST estimate per month (last 12)
    g = f.tail(12).copy()
    g["output_gst"] = g["revenue"] * cfg.gst_rate
    g["itc"] = (g["cogs"] + cfg.taxable_opex_share * g["operating_expenses"]) * cfg.gst_rate * cfg.itc_eligible_share
    g["net_gst_payable"] = g["output_gst"] - g["itc"]
    g["itc_to_output"] = g["itc"] / g["output_gst"]
    res.tables["gst"] = g[["month", "revenue", "output_gst", "itc", "net_gst_payable", "itc_to_output"]]
    last_gst = g.iloc[-1]
    dl = deadlines(today, cfg)
    res.tables["deadlines"] = dl
    soon = dl[dl["days_left"] <= 7]

    next_gst = float(last_gst["net_gst_payable"])
    res.metrics = dict(fy=f"{start.year}-{str(start.year + 1)[2:]}", pbt_fy_estimate=pbt_fy,
                       income_tax_estimate=est_tax, tax_recorded_ytd=paid_ytd,
                       gst_payable_last_month=next_gst, deadlines_within_7_days=len(soon))
    res.facts = dict(financial_year=res.metrics["fy"], months_of_data_in_fy=n,
                     estimated_pbt_full_year=inr(pbt_fy), estimated_income_tax=inr(est_tax),
                     tax_recorded_so_far=inr(paid_ytd), estimated_gst_payable_last_month=inr(next_gst),
                     deadlines_next_7_days=len(soon),
                     next_deadline=(f"{dl.iloc[0]['obligation']} on {dl.iloc[0]['due_date']:%d %b %Y}" if len(dl) else "none"),
                     income_tax_rate_pct=round(cfg.income_tax_rate * 100, 2), gst_rate_pct=round(cfg.gst_rate * 100, 1))
    short = adv[adv["status"] == "Shortfall"]
    if len(short):
        worst = short.iloc[-1]
        res.add_flag(RED if worst["shortfall"] > 0.1 * est_tax else AMBER,
                     f"Advance tax shortfall of {inr(float(worst['shortfall']))} against the "
                     f"{worst['instalment_due']:%d %b} instalment",
                     "Interest under sections 234B/234C may apply; confirm with your CA.")
        res.facts["advance_tax_shortfall"] = inr(float(worst["shortfall"]))
    if last_gst["itc_to_output"] > 0.95:
        res.add_flag(AMBER, f"ITC is {last_gst['itc_to_output']:.0%} of output GST",
                     "Unusually high; check eligibility and reconciliation with GSTR-2B.")
    if pbt_fy < 0:
        res.add_flag(AMBER, "Projected loss for the year", "Check carry-forward of losses with your CA.")
    for _, r in soon.iterrows():
        res.add_flag(AMBER if r["days_left"] > 2 else RED,
                     f"{r['obligation']} due {r['due_date']:%d %b} ({int(r['days_left'])} days)", r["category"])
    if not res.flags:
        res.add_flag(GREEN, "No tax compliance gaps detected")
    res.notes.append("All tax figures are ESTIMATES on stated assumptions (GST "
                     f"{cfg.gst_rate:.0%}, income tax {cfg.income_tax_rate:.2%}). Due dates are "
                     "standard dates; verify against current notifications. Not tax advice.")
    return res
