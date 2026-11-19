"""InvestIQ - surplus-funds analysis, illustrative allocation by risk appetite, portfolio tracking.

Allocation templates and return ranges are ILLUSTRATIVE assumptions for planning, not
investment advice or forecasts. Confirm treasury decisions with a qualified adviser.
"""
from __future__ import annotations

import pandas as pd

from .core import AMBER, GREEN, RED, ModuleResult, inr, safe_div

ALLOCATIONS = {
    "Conservative": {"Liquid": 40, "Fixed Deposit": 40, "Debt": 20, "Equity": 0, "Gold": 0},
    "Balanced": {"Liquid": 25, "Fixed Deposit": 25, "Debt": 25, "Equity": 15, "Gold": 10},
    "Growth": {"Liquid": 15, "Fixed Deposit": 15, "Debt": 20, "Equity": 35, "Gold": 15},
}
# Illustrative annual return assumptions (low, high), % - assumptions only
RETURN_RANGE = {"Liquid": (5.5, 7.0), "Fixed Deposit": (6.5, 7.5), "Debt": (6.5, 8.0),
                "Equity": (8.0, 14.0), "Gold": (4.0, 9.0)}


def run(fin: pd.DataFrame, portfolio: pd.DataFrame | None = None, risk_appetite: str = "Balanced",
        tax_estimate: float = 0.0) -> ModuleResult:
    res = ModuleResult("InvestIQ")
    fin = fin.sort_values("month").reset_index(drop=True)
    cash = float(fin["cash"].iloc[-1])
    monthly_cost = float((fin["operating_expenses"] + fin["interest_expense"]).tail(3).mean())
    buffer = 3 * monthly_cost + tax_estimate * 0.25   # 3 months of fixed costs + one tax instalment
    surplus = max(cash - buffer, 0.0)
    debt = float((fin["long_term_debt"] + fin["short_term_debt"]).iloc[-1])
    cost_of_debt = safe_div(float(fin["interest_expense"].tail(12).sum()), float(
        (fin["long_term_debt"] + fin["short_term_debt"]).tail(12).mean()))

    alloc = ALLOCATIONS[risk_appetite]
    rows = []
    for cls, w in alloc.items():
        lo, hi = RETURN_RANGE[cls]
        rows.append(dict(asset_class=cls, weight_pct=w, amount=surplus * w / 100,
                         assumed_return_low_pct=lo, assumed_return_high_pct=hi))
    plan = pd.DataFrame(rows)
    plan["annual_income_low"] = plan["amount"] * plan["assumed_return_low_pct"] / 100
    plan["annual_income_high"] = plan["amount"] * plan["assumed_return_high_pct"] / 100
    res.tables["allocation_plan"] = plan
    res.tables["allocation_templates"] = pd.DataFrame(ALLOCATIONS).T.reset_index().rename(columns={"index": "risk_appetite"})

    exp_lo = float(plan["annual_income_low"].sum())
    exp_hi = float(plan["annual_income_high"].sum())
    res.metrics = dict(cash=cash, required_buffer=buffer, surplus=surplus, cost_of_debt=cost_of_debt,
                       risk_appetite=risk_appetite, expected_income_low=exp_lo, expected_income_high=exp_hi)
    res.facts = dict(cash=inr(cash), required_cash_buffer=inr(buffer), surplus_funds=inr(surplus),
                     risk_appetite=risk_appetite, cost_of_debt_pct=round(cost_of_debt * 100, 1),
                     illustrative_annual_income_range=f"{inr(exp_lo)} to {inr(exp_hi)}")

    # Debt vs invest comparison: if borrowing costs exceed likely safe yields, prepaying can beat investing
    safe_yield = RETURN_RANGE["Fixed Deposit"][1] / 100
    if surplus > 0 and cost_of_debt > safe_yield + 0.01:
        res.add_flag(AMBER, f"Cost of debt ({cost_of_debt:.1%}) exceeds safe deposit yields (about {safe_yield:.1%})",
                     "Consider prepaying costly debt with part of the surplus before investing.")
        res.facts["prepay_debt_worth_considering"] = True
    if surplus <= 0:
        res.add_flag(RED if cash < monthly_cost else AMBER, "No surplus funds above the required buffer",
                     f"Cash {inr(cash)} versus a buffer of {inr(buffer)}.")

    if portfolio is not None and len(portfolio):
        p = portfolio.copy()
        p["cost_value"] = p["units"] * p["buy_price"]
        p["market_value"] = p["units"] * p["current_price"]
        p["gain"] = p["market_value"] - p["cost_value"]
        p["gain_pct"] = p["gain"] / p["cost_value"]
        p["weight"] = p["market_value"] / p["market_value"].sum()
        res.tables["portfolio"] = p
        cls = p.groupby("asset_class").agg(market_value=("market_value", "sum"), cost_value=("cost_value", "sum")).reset_index()
        cls["actual_weight_pct"] = cls["market_value"] / cls["market_value"].sum() * 100
        target = pd.Series(alloc, name="target_weight_pct")
        cls = cls.merge(target, left_on="asset_class", right_index=True, how="outer").fillna(0)
        cls["drift_pts"] = cls["actual_weight_pct"] - cls["target_weight_pct"]
        res.tables["allocation_vs_target"] = cls
        mv, cv = float(p["market_value"].sum()), float(p["cost_value"].sum())
        max_drift = float(cls["drift_pts"].abs().max())
        liquid = float(p[p["asset_class"].isin(["Liquid", "Debt"])]["market_value"].sum() / mv)
        res.metrics.update(portfolio_value=mv, portfolio_gain=mv - cv, portfolio_return=safe_div(mv - cv, cv),
                           max_drift_pts=max_drift, liquid_share=liquid)
        res.facts.update(portfolio_value=inr(mv), unrealised_gain=inr(mv - cv),
                         unrealised_gain_pct=round(safe_div(mv - cv, cv) * 100, 1),
                         max_allocation_drift_pts=round(max_drift, 1),
                         liquid_and_debt_share_pct=round(liquid * 100, 1),
                         best_holding=p.sort_values("gain_pct").iloc[-1]["instrument"],
                         weakest_holding=p.sort_values("gain_pct").iloc[0]["instrument"])
        if max_drift > 15:
            worst = cls.iloc[cls["drift_pts"].abs().argmax()]
            res.add_flag(AMBER, f"Portfolio has drifted {max_drift:.0f} points from the {risk_appetite} template",
                         f"{worst['asset_class']}: {worst['actual_weight_pct']:.0f}% vs target {worst['target_weight_pct']:.0f}%.")
        if (p["gain"] < 0).any():
            res.add_flag(AMBER, f"{int((p['gain'] < 0).sum())} holding(s) below cost", ", ".join(p[p["gain"] < 0]["instrument"]))
        if liquid < 0.25 and surplus > 0:
            res.add_flag(AMBER, f"Only {liquid:.0%} of the portfolio is in liquid or debt instruments",
                         "Keep enough quick-access funds for working capital.")
    if not res.flags:
        res.add_flag(GREEN, "Surplus funds and portfolio look in order")
    res.notes.append("ILLUSTRATIVE: allocation templates and return ranges are planning assumptions, "
                     "not predictions or personalised investment advice. Buffer = 3 months of operating "
                     "expenses plus interest, plus one quarter's estimated tax instalment.")
    return res
