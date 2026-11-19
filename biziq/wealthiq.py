"""WealthIQ - financial health: ratios, Altman Z-score, 0-100 health score, flags."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import AMBER, GREEN, RED, ModuleResult, inr, pct, safe_div, score_between

# name: (label, bad, good, weight, amber_threshold, red_threshold, higher_is_better)
RATIO_SPEC = {
    "current_ratio":    ("Current ratio", 0.8, 2.0, 1.5, 1.5, 1.0, True),
    "quick_ratio":      ("Quick ratio", 0.5, 1.5, 1.0, 1.0, 0.7, True),
    "cash_ratio":       ("Cash ratio", 0.1, 0.8, 0.5, 0.4, 0.2, True),
    "gross_margin":     ("Gross margin", 0.15, 0.50, 1.5, 0.35, 0.25, True),
    "operating_margin": ("Operating margin", 0.0, 0.15, 1.5, 0.08, 0.03, True),
    "net_margin":       ("Net margin", 0.0, 0.12, 1.0, 0.06, 0.02, True),
    "roa":              ("Return on assets", 0.0, 0.12, 1.0, 0.06, 0.02, True),
    "roe":              ("Return on equity", 0.0, 0.25, 1.0, 0.12, 0.05, True),
    "debt_to_equity":   ("Debt-to-equity", 3.0, 0.5, 1.5, 1.5, 2.5, False),
    "interest_coverage": ("Interest coverage", 1.0, 6.0, 1.5, 3.0, 1.5, True),
    "ccc_days":         ("Cash conversion cycle (days)", 120, 30, 1.0, 70, 100, False),
}


def _ri(x):
    """Round to a whole number, or None when the figure cannot be computed (e.g. no cost of goods yet)."""
    return None if x is None or pd.isna(x) else round(x)


def _ttm(df: pd.DataFrame, col: str, end_idx: int, n: int = 12) -> float:
    lo = max(0, end_idx - n + 1)
    return float(df[col].iloc[lo:end_idx + 1].sum())


def compute_ratios(fin: pd.DataFrame, idx: int | None = None) -> dict:
    """Ratios as of month `idx` (default latest), income items on a trailing-12-month basis."""
    fin = fin.sort_values("month").reset_index(drop=True)
    idx = len(fin) - 1 if idx is None else idx
    r = fin.iloc[idx]
    n = min(12, idx + 1)
    scale = 12 / n  # annualise if less than 12 months of history
    rev = _ttm(fin, "revenue", idx) * scale
    cogs = _ttm(fin, "cogs", idx) * scale
    opex = _ttm(fin, "operating_expenses", idx) * scale
    dep = _ttm(fin, "depreciation", idx) * scale if "depreciation" in fin else 0.0
    interest = _ttm(fin, "interest_expense", idx) * scale
    tax = _ttm(fin, "tax", idx) * scale
    ni = _ttm(fin, "net_income", idx) * scale
    ebit = rev - cogs - opex - dep
    debt = r["long_term_debt"] + r["short_term_debt"]
    ca = r["cash"] + r["receivables"] + r["inventory"] + (r.get("other_current_assets", 0) or 0)
    cl = r["current_liabilities"]
    ta = r["total_assets"]
    out = dict(
        month=r["month"], revenue_ttm=rev, ebit_ttm=ebit, net_income_ttm=ni,
        current_ratio=safe_div(ca, cl), quick_ratio=safe_div(r["cash"] + r["receivables"], cl),
        cash_ratio=safe_div(r["cash"], cl),
        gross_margin=safe_div(rev - cogs, rev), operating_margin=safe_div(ebit, rev),
        net_margin=safe_div(ni, rev), roa=safe_div(ni, ta), roe=safe_div(ni, r["equity"]),
        debt_to_equity=safe_div(debt, r["equity"]), debt_to_assets=safe_div(debt, ta),
        interest_coverage=safe_div(ebit, interest),
        asset_turnover=safe_div(rev, ta),
        dio=safe_div(r["inventory"], cogs) * 365, dso=safe_div(r["receivables"], rev) * 365,
        dpo=safe_div(r["payables"], cogs) * 365,
    )
    out["ccc_days"] = out["dio"] + out["dso"] - out["dpo"]
    # Altman Z'-score (private-firm variant; book value of equity)
    wc = ca - cl
    x1, x2 = safe_div(wc, ta), safe_div(r["retained_earnings"], ta)
    x3, x4 = safe_div(ebit, ta), safe_div(r["equity"], r["total_liabilities"])
    x5 = safe_div(rev, ta)
    out["altman_z"] = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5
    return out


def altman_zone(z: float) -> str:
    if pd.isna(z):
        return "unknown"
    return "safe" if z > 2.9 else ("grey" if z >= 1.23 else "distress")


def health_score(ratios: dict) -> tuple[float, dict]:
    parts, wsum, total = {}, 0.0, 0.0
    for key, (label, bad, good, w, *_rest) in RATIO_SPEC.items():
        s = score_between(ratios.get(key), bad, good)
        if pd.isna(s):
            continue
        parts[key] = s
        total += s * w
        wsum += w
    base = total / wsum if wsum else float("nan")
    z = ratios.get("altman_z")
    z_score = score_between(z, 1.23, 2.9)
    score = base * 0.85 + z_score * 0.15 if not pd.isna(z_score) else base
    return float(score), parts


def _flag_level(key: str, value: float) -> str:
    _, _, _, _, amber, red, higher = RATIO_SPEC[key]
    if pd.isna(value):
        return AMBER
    if higher:
        return RED if value < red else (AMBER if value < amber else GREEN)
    return RED if value > red else (AMBER if value > amber else GREEN)


def _fmt_ratio(key: str, v: float) -> str:
    if pd.isna(v):
        return "n/a"
    if key in ("gross_margin", "operating_margin", "net_margin", "roa", "roe"):
        return pct(v)
    if key == "ccc_days":
        return f"{v:.0f} days"
    return f"{v:.2f}"


def run(fin: pd.DataFrame) -> ModuleResult:
    fin = fin.sort_values("month").reset_index(drop=True)
    res = ModuleResult("WealthIQ")
    cur = compute_ratios(fin)
    score, parts = health_score(cur)
    res.score = score
    z = cur["altman_z"]
    res.metrics = dict(
        health_score=round(score, 1), altman_z=round(z, 2), altman_zone=altman_zone(z),
        revenue_ttm=cur["revenue_ttm"], net_income_ttm=cur["net_income_ttm"],
        cash=float(fin["cash"].iloc[-1]), latest_month=str(cur["month"])[:7],
    )

    # Ratio table with RAG flags
    rows = []
    for key, (label, *_rest) in RATIO_SPEC.items():
        v = cur.get(key)
        lvl = _flag_level(key, v)
        rows.append(dict(ratio=label, value=_fmt_ratio(key, v), status=lvl,
                         score=round(parts.get(key, float("nan")), 0)))
        if lvl != GREEN:
            res.add_flag(lvl, f"{label} is {_fmt_ratio(key, v)}",
                         "Below the healthy range." if RATIO_SPEC[key][6] else "Above the healthy range.")
    extra = [("Debt-to-assets", pct(cur["debt_to_assets"])), ("Asset turnover", f"{cur['asset_turnover']:.2f}x"),
             ("Inventory days (DIO)", f"{cur['dio']:.0f}"), ("Receivable days (DSO)", f"{cur['dso']:.0f}"),
             ("Payable days (DPO)", f"{cur['dpo']:.0f}")]
    rows += [dict(ratio=a, value=b, status="", score=float("nan")) for a, b in extra]
    res.tables["ratios"] = pd.DataFrame(rows)

    # Trend table: ratios over each month with >= 3 months of history
    trend = []
    for i in range(2, len(fin)):
        c = compute_ratios(fin, i)
        s, _ = health_score(c)
        trend.append(dict(month=c["month"], health_score=s, current_ratio=c["current_ratio"],
                          net_margin=c["net_margin"], operating_margin=c["operating_margin"],
                          gross_margin=c["gross_margin"], debt_to_equity=c["debt_to_equity"],
                          ccc_days=c["ccc_days"], altman_z=c["altman_z"]))
    trend = pd.DataFrame(trend)
    res.tables["trend"] = trend

    # Period-over-period comparison
    delta = {}
    if len(trend) > 6:
        prev = trend.iloc[-7]
        last = trend.iloc[-1]
        for k in ("health_score", "gross_margin", "net_margin", "ccc_days", "current_ratio"):
            delta[k] = float(last[k] - prev[k])
        if delta["health_score"] < -5:
            res.add_flag(AMBER, "Financial health is trending down",
                         f"Score moved {delta['health_score']:+.1f} points over six months.")
    zone = altman_zone(z)
    if zone == "distress":
        res.add_flag(RED, f"Altman Z-score {z:.2f} is in the distress zone", "Z' below 1.23.")
    elif zone == "grey":
        res.add_flag(AMBER, f"Altman Z-score {z:.2f} is in the grey zone", "Z' between 1.23 and 2.90.")

    res.facts = dict(
        health_score=round(score, 1), altman_z=round(z, 2), altman_zone=zone,
        current_ratio=round(cur["current_ratio"], 2), quick_ratio=round(cur["quick_ratio"], 2),
        gross_margin_pct=round(cur["gross_margin"] * 100, 1),
        operating_margin_pct=round(cur["operating_margin"] * 100, 1),
        net_margin_pct=round(cur["net_margin"] * 100, 1), roe_pct=round(cur["roe"] * 100, 1),
        debt_to_equity=round(cur["debt_to_equity"], 2),
        interest_coverage=round(cur["interest_coverage"], 2),
        dio_days=_ri(cur["dio"]), dso_days=_ri(cur["dso"]), dpo_days=_ri(cur["dpo"]),
        ccc_days=_ri(cur["ccc_days"]),
        revenue_ttm=inr(cur["revenue_ttm"]), net_income_ttm=inr(cur["net_income_ttm"]),
        health_score_change_6m=round(delta.get("health_score", 0.0), 1),
        gross_margin_change_6m_pts=round(delta.get("gross_margin", 0.0) * 100, 1),
        ccc_change_6m_days=_ri(delta.get("ccc_days", 0.0)),
        red_flags=sum(f.level == RED for f in res.flags),
        amber_flags=sum(f.level == AMBER for f in res.flags),
    )
    res.notes.append("Income items use trailing-12-month totals. Altman Z' (private-firm variant) "
                     "uses book value of equity.")
    return res


def summary_text(res: ModuleResult) -> str:
    """Written summary generated from computed numbers only (template, never invented)."""
    f = res.facts
    band = "strong" if f["health_score"] >= 75 else ("moderate" if f["health_score"] >= 50 else "weak")
    return (f"Financial health is {band} at {f['health_score']}/100 (Altman Z' {f['altman_z']}, "
            f"{f['altman_zone']} zone). Trailing-12-month revenue is {f['revenue_ttm']} with a "
            f"{f['gross_margin_pct']}% gross margin and {f['net_margin_pct']}% net margin. "
            f"The cash conversion cycle is {f['ccc_days']} days "
            f"(inventory {f['dio_days']}, receivables {f['dso_days']}, payables {f['dpo_days']}). "
            f"There are {f['red_flags']} red and {f['amber_flags']} amber flags.")
