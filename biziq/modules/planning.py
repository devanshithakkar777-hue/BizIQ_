"""Planning modules: BudgetIQ, ScenarioIQ, UnitEconomicsIQ, DiscountIQ, AnomalyIQ."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import AMBER, GREEN, RED, ModuleResult, inr, safe_div
from ..registry import Context, ModuleSpec, register
from ._common import argext, clean_facts, pct1, r1


def _month_end_done(m: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    """Months that are complete on `as_of` (a half-finished month would look like a huge underspend)."""
    return (m + pd.offsets.MonthEnd(0)) <= as_of.normalize()


# ============================================================================== BudgetIQ
def run_budget(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("BudgetIQ")
    b = data["budgets"].copy()
    e = data["expenses"].copy()
    b["month"] = b["month"].dt.to_period("M").dt.to_timestamp()
    e["month"] = e["date"].dt.to_period("M").dt.to_timestamp()
    b = b[_month_end_done(b["month"], ctx.as_of)]
    if b.empty:
        raise ValueError("needs budgets for at least one completed month")
    bud = b.groupby(["month", "category"], as_index=False)["budget"].sum()
    act = e.groupby(["month", "category"], as_index=False)["amount"].sum().rename(columns={"amount": "actual"})
    m = bud.merge(act, on=["month", "category"], how="left")
    m["actual"] = m["actual"].fillna(0.0)
    if not (e["month"].isin(m["month"])).any():
        raise ValueError("needs expenses in the budgeted months")
    m["variance"] = m["actual"] - m["budget"]

    def roll(df):
        g = df.groupby("category").agg(budget=("budget", "sum"), actual=("actual", "sum"))
        g["variance"] = g["actual"] - g["budget"]
        g["variance_pct"] = (100 * g["variance"] / g["budget"].replace(0, np.nan)).round(1)
        return g.reset_index().sort_values("variance", ascending=False)

    last3 = sorted(m["month"].unique())[-3:]
    all_cat, rec = roll(m), roll(m[m["month"].isin(last3)])
    res.tables["by_category_all_months"] = all_cat
    res.tables["by_category_last_3_months"] = rec
    res.tables["by_month"] = m.groupby("month").agg(budget=("budget", "sum"), actual=("actual", "sum")).reset_index().assign(month=lambda x: x["month"].dt.strftime("%Y-%m"))
    tb, ta = float(m["budget"].sum()), float(m["actual"].sum())
    over = rec[(rec["variance"] > np.maximum(0.10 * rec["budget"], 10000))]
    under = all_cat[(all_cat["budget"] > 0) & (all_cat["actual"] < 0.75 * all_cat["budget"])]
    worst = rec.iloc[0] if len(rec) else None
    res.facts = clean_facts(dict(
        months_compared=int(m["month"].nunique()), total_budget=inr(tb), total_actual=inr(ta), total_variance=inr(ta - tb), total_variance_pct=pct1(safe_div(ta - tb, tb)),
        categories_over_budget_recent=len(over), worst_category=None if worst is None else str(worst["category"]),
        worst_category_overspend=None if worst is None or worst["variance"] <= 0 else inr(float(worst["variance"])), worst_category_overspend_pct=None if worst is None or worst["variance"] <= 0 else r1(worst["variance_pct"]),
        persistent_underspend_categories=len(under), most_underspent=None if under.empty else str(under.sort_values("variance").iloc[0]["category"])))
    for _, r in over.head(3).iterrows():
        res.add_flag(RED if r["variance_pct"] > 30 else AMBER, f"{r['category']} is {r['variance_pct']:.0f}% over budget over the last 3 months", f"{inr(float(r['variance']))} above plan")
    if len(under):
        res.add_flag(AMBER, f"{len(under)} categor{'y is' if len(under) == 1 else 'ies are'} running below 75% of budget: {', '.join(under['category'].astype(str).head(3))}",
                     "Either the budget is padded or planned work is not happening")
    if tb and ta > tb * 1.1:
        res.add_flag(RED, f"Total spend is {100 * (ta - tb) / tb:.0f}% over the budget")
    if not res.flags:
        res.add_flag(GREEN, "Spending is within budget in every category")
    res.notes.append("Actuals come from the expenses file, so anything not recorded there (payroll, stock purchases, tax) is not in this comparison. Only completed months are compared. "
                     "A category is flagged when it is over by more than 10% of its budget and at least Rs 10,000 across the last 3 months.")
    return res


def rec_budget(f, flags):
    r = []
    if f.get("worst_category_overspend"):
        r.append(f"Review {f['worst_category']}: it is {f['worst_category_overspend']} over plan in the last 3 months. Decide whether to cut spend or to reset the budget on purpose.")
    if f.get("persistent_underspend_categories"):
        r.append(f"Check {f.get('most_underspent')} and the other low-spend categories: free that money or find out what is not getting done.")
    return r or ["Spending is on plan; keep comparing monthly."]


# ============================================================================== ScenarioIQ
def project(base: dict, price_pct: float = 0, volume_pct: float = 0, cogs_pct: float = 0, opex_pct: float = 0) -> dict:
    """What-if on a base year. COGS moves with volume (and with the cost change you give); operating expenses are treated as fixed unless changed."""
    rev = base["revenue"] * (1 + price_pct / 100) * (1 + volume_pct / 100)
    cogs = base["cogs"] * (1 + volume_pct / 100) * (1 + cogs_pct / 100)
    opex = base["opex"] * (1 + opex_pct / 100)
    return dict(revenue=rev, cogs=cogs, opex=opex, operating_profit=rev - cogs - opex, margin_pct=100 * safe_div(rev - cogs - opex, rev, 0.0))


def scenario_base(fin: pd.DataFrame) -> dict:
    f = fin.sort_values("month").tail(12)
    return dict(revenue=float(f["revenue"].sum()), cogs=float(f["cogs"].sum()), opex=float(f["operating_expenses"].sum()), months=int(len(f)),
                cash=float(f["cash"].iloc[-1]) if "cash" in f and len(f) else float("nan"))


DEFAULT_SCENARIOS = {  # illustrative inputs, shown as such; the interactive page lets the owner set their own
    "Price up 5%, volume unchanged": dict(price_pct=5), "Price down 5%, volume unchanged": dict(price_pct=-5), "Volume down 10%": dict(volume_pct=-10),
    "Volume up 10%": dict(volume_pct=10), "Costs of goods up 8%": dict(cogs_pct=8), "Operating expenses up 15%": dict(opex_pct=15),
}


def run_scenario(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ScenarioIQ")
    fin = data["financials"]
    if len(fin) < 3:
        raise ValueError("needs at least 3 months of financials")
    base = scenario_base(fin)
    if base["revenue"] <= 0:
        raise ValueError("needs financials with revenue")
    b = project(base)
    custom = ctx.settings.get("scenarios") or {}
    rows = [dict(scenario="Base (trailing year)", **{k: round(v, 1) for k, v in b.items()}, profit_change=0.0)]
    for name, kw in {**DEFAULT_SCENARIOS, **custom}.items():
        p = project(base, **kw)
        rows.append(dict(scenario=name, **{k: round(v, 1) for k, v in p.items()}, profit_change=round(p["operating_profit"] - b["operating_profit"], 1)))
    res.tables["scenarios"] = pd.DataFrame(rows)
    res.tables["base_year"] = pd.DataFrame([dict(revenue=base["revenue"], cogs=base["cogs"], opex=base["opex"], months=base["months"])])
    contribution = base["revenue"] - base["cogs"]
    cushion_vol = safe_div(b["operating_profit"], contribution)                       # volume fall (costs of goods move, opex does not) that erases operating profit
    cushion_price = safe_div(b["operating_profit"], base["revenue"])                  # price cut, no volume change, that erases it
    res.facts = clean_facts(dict(
        base_months=base["months"], base_revenue=inr(base["revenue"]), base_operating_profit=inr(b["operating_profit"]), base_operating_margin_pct=r1(b["margin_pct"]),
        volume_fall_that_erases_profit_pct=pct1(cushion_vol) if b["operating_profit"] > 0 else None, price_cut_that_erases_profit_pct=pct1(cushion_price) if b["operating_profit"] > 0 else None,
        gross_margin_pct=pct1(safe_div(contribution, base["revenue"])), operating_expenses_share_of_revenue_pct=pct1(safe_div(base["opex"], base["revenue"])),
        profit_if_price_down_5pct=inr(project(base, price_pct=-5)["operating_profit"]), profit_if_volume_down_10pct=inr(project(base, volume_pct=-10)["operating_profit"])))
    if b["operating_profit"] <= 0:
        res.add_flag(RED, "The business is operating at a loss on the trailing year's numbers")
    elif cushion_vol is not None and not pd.isna(cushion_vol) and cushion_vol < 0.15:
        res.add_flag(AMBER, f"A {cushion_vol:.0%} fall in volume would erase operating profit", "High fixed costs leave little cushion")
    if b["operating_profit"] > 0 and cushion_price < 0.10:
        res.add_flag(AMBER, f"A price cut of {cushion_price:.0%} with no extra sales would erase operating profit")
    if not res.flags:
        res.add_flag(GREEN, "Operating profit can absorb a moderate drop in volume or price")
    res.notes.append("A simple what-if on the trailing 12 months, not a forecast. Cost of goods moves with volume; operating expenses are held fixed unless you change them; working capital, tax and "
                     "one-off items are ignored. The scenario inputs are illustrative: use the sliders on the page to test your own. Price changes assume customers do not buy less because of them, "
                     "which is the optimistic case.")
    return res


def rec_scenario(f, flags):
    r = []
    if f.get("volume_fall_that_erases_profit_pct") is not None and f["volume_fall_that_erases_profit_pct"] < 15:
        r.append(f"Your profit is exposed to a volume fall of only {f['volume_fall_that_erases_profit_pct']}%: look at which fixed costs can be made variable or delayed.")
    if f.get("price_cut_that_erases_profit_pct") is not None and f["price_cut_that_erases_profit_pct"] < 10:
        r.append("Discounting is expensive for you: a small price cut needs a large volume gain to pay for itself. Test any promotion on a small group first.")
    return r or ["There is a reasonable cushion; use the sliders before any pricing or hiring decision."]


# ============================================================================== UnitEconomicsIQ
def run_unit_economics(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("UnitEconomicsIQ")
    s = data["sales"].copy()
    c = data["campaigns"].copy()
    s["rev"] = s["quantity"] * s["unit_price"]
    c = c[(c["date"] <= ctx.as_of)]
    start = ctx.as_of - pd.Timedelta(days=180)
    spend_win = float(c.loc[c["date"] > start, "spend"].sum())
    if spend_win <= 0:
        raise ValueError("needs campaign spend in the last 6 months")
    first = s.groupby("customer_id")["date"].min()
    data_start = s["date"].min()
    new = first[(first > start) & (first > data_start + pd.Timedelta(days=90)) & (first <= ctx.as_of)]      # customers genuinely new, not just first seen at the start of the file
    if len(new) < 5:
        raise ValueError("needs at least 5 new customers in the last 6 months")
    cac = spend_win / len(new)
    # gross margin: financials if present, else product costs, else revenue only
    gm, basis = None, "revenue"
    fin = data.get("financials")
    if fin is not None and len(fin) and float(fin["revenue"].sum()) > 0:
        gm, basis = 1 - float(fin["cogs"].sum()) / float(fin["revenue"].sum()), "gross profit (from your financials)"
    elif data.get("product_costs") is not None:
        pc = data["product_costs"].groupby("product")["unit_cost"].last()
        ss = s.assign(cost=s["quantity"] * s["product"].map(pc))
        if ss["cost"].notna().any():
            gm, basis = 1 - float(ss["cost"].sum()) / float(ss.loc[ss["cost"].notna(), "rev"].sum()), "gross profit (from product costs)"
    factor = 1.0 if gm is None else gm
    mature = new[new <= ctx.as_of - pd.Timedelta(days=90)]
    first_date = first.reindex(s["customer_id"]).values
    in90 = s[(s["customer_id"].isin(mature.index)) & (s["date"] <= pd.to_datetime(first_date) + pd.Timedelta(days=90))] if len(mature) else s.iloc[0:0]
    value90 = float(in90["rev"].sum()) * factor / len(mature) if len(mature) else None
    repeat = float((s[s["customer_id"].isin(mature.index)].groupby("customer_id")["order_id"].nunique() > 1).mean()) if len(mature) else None
    ratio = safe_div(value90, cac) if value90 is not None else None
    ch = c[c["date"] > start].groupby("channel").agg(spend=("spend", "sum"), conversions=("conversions", "sum"), reported_revenue=("revenue", "sum")).reset_index()
    ch["cost_per_reported_conversion"] = (ch["spend"] / ch["conversions"].replace(0, np.nan)).round(0)
    ch["reported_roas"] = (ch["reported_revenue"] / ch["spend"].replace(0, np.nan)).round(2)
    res.tables["channels"] = ch.sort_values("spend", ascending=False)
    res.tables["new_customers_by_month"] = new.dt.to_period("M").astype(str).value_counts().sort_index().rename_axis("month").reset_index(name="new_customers")
    best, worst = argext(ch.set_index("channel")["cost_per_reported_conversion"], "min"), argext(ch.set_index("channel")["cost_per_reported_conversion"], "max")
    res.facts = clean_facts(dict(
        ad_spend_6m=inr(spend_win), new_customers_6m=int(len(new)), blended_cac=inr(cac), value_basis=basis, first_90_day_value_per_customer=None if value90 is None else inr(value90),
        recovery_ratio_90d=r1(ratio, 2), repeat_purchase_rate_pct=pct1(repeat), gross_margin_used_pct=None if gm is None else pct1(gm),
        cheapest_channel=None if best is None else str(best), costliest_channel=None if worst is None else str(worst),
        costliest_vs_cheapest_x=None if best is None or worst is None else r1(ch.set_index("channel").loc[worst, "cost_per_reported_conversion"] / ch.set_index("channel").loc[best, "cost_per_reported_conversion"])))
    if ratio is not None and ratio < 1:
        res.add_flag(AMBER, f"New customers return {ratio:.1f}x of what they cost to win within 90 days", "Payback takes longer than 3 months")
    elif ratio is not None and ratio >= 3:
        res.add_flag(GREEN, f"New customers return {ratio:.1f}x their acquisition cost within 90 days")
    if best is not None and worst is not None and best != worst and res.facts.get("costliest_vs_cheapest_x", 0) and res.facts["costliest_vs_cheapest_x"] >= 2:
        res.add_flag(AMBER, f"{worst} costs {res.facts['costliest_vs_cheapest_x']}x more per conversion than {best}")
    if not res.flags:
        res.add_flag(GREEN, "Acquisition cost is recovered in a reasonable time")
    res.notes.append("CAC here is ad spend divided by customers whose first order fell in the last 6 months, so it ignores salaries, tools and discounts given to win customers. "
                     f"Value is measured on {basis} in the first 90 days of customers acquired at least 90 days ago. 'Reported conversions' are what the ad platforms claim and often "
                     "double-count; the blended CAC is the figure to trust. Customers first seen in the first 3 months of the file are not counted as new.")
    return res


def rec_unit_economics(f, flags):
    r = []
    if f.get("recovery_ratio_90d") is not None and f["recovery_ratio_90d"] < 1:
        r.append(f"A new customer costs {f.get('blended_cac')} to win but returns {f.get('first_90_day_value_per_customer')} in 90 days: raise repeat purchases or cut acquisition cost before spending more.")
    if f.get("costliest_vs_cheapest_x") and f["costliest_vs_cheapest_x"] >= 2:
        r.append(f"Move budget from {f['costliest_channel']} towards {f['cheapest_channel']} and watch the blended CAC, not the platform numbers.")
    return r or ["Acquisition economics look healthy; keep tracking blended CAC each month."]


# ============================================================================== DiscountIQ
def run_discount(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("DiscountIQ")
    s = data["sales"].copy()
    if "discount_pct" not in s.columns or pd.to_numeric(s["discount_pct"], errors="coerce").notna().sum() == 0:
        raise ValueError("needs a discount_pct column in sales")
    s["disc"] = pd.to_numeric(s["discount_pct"], errors="coerce").fillna(0).clip(0, 95) / 100
    s["rev"] = s["quantity"] * s["unit_price"]
    s["list_rev"] = s["rev"] / (1 - s["disc"])
    s["discount_cost"] = s["list_rev"] - s["rev"]
    s["band"] = pd.cut(s["disc"] * 100, [-1, 0, 10, 20, 30, 100], labels=["none", "1-10%", "11-20%", "21-30%", "over 30%"])
    bands = s.groupby("band", observed=True).agg(lines=("rev", "size"), avg_qty=("quantity", "mean"), avg_revenue=("rev", "mean"), revenue=("rev", "sum"), discount_cost=("discount_cost", "sum")).reset_index()
    bands["share_of_lines_pct"] = (100 * bands["lines"] / len(s)).round(1)
    margin_known = data.get("product_costs") is not None
    loss_lines, loss_total = None, None
    if margin_known:
        pc = data["product_costs"].groupby("product")["unit_cost"].last()
        s["cost"] = s["quantity"] * s["product"].map(pc)
        known = s["cost"].notna()
        bm = s[known].groupby("band", observed=True).apply(lambda g: 100 * (g["rev"].sum() - g["cost"].sum()) / g["rev"].sum() if g["rev"].sum() else np.nan, include_groups=False)
        bands["gross_margin_pct"] = bands["band"].map(bm).round(1)
        lossy = s[known & (s["rev"] < s["cost"])]
        loss_lines, loss_total = len(lossy), float((lossy["cost"] - lossy["rev"]).sum())
    res.tables["by_discount_band"] = bands
    # quantity difference within each product, so a product mix change does not pose as a discount effect
    d_, n_ = s[s["disc"] > 0], s[s["disc"] == 0]
    diffs, w = [], []
    for p, g in d_.groupby("product"):
        base = n_.loc[n_["product"] == p, "quantity"]
        if len(base) >= 20 and len(g) >= 20 and base.mean() > 0:
            diffs.append(g["quantity"].mean() / base.mean() - 1)
            w.append(len(g))
    qdiff = float(np.average(diffs, weights=w)) if diffs else None
    deep = s[s["disc"] >= 0.30]
    deep_q = None
    dd, dw = [], []
    for p, g in deep.groupby("product"):
        base = n_.loc[n_["product"] == p, "quantity"]
        if len(base) >= 20 and len(g) >= 10 and base.mean() > 0:
            dd.append(g["quantity"].mean() / base.mean() - 1)
            dw.append(len(g))
    deep_q = float(np.average(dd, weights=dw)) if dd else None
    tot_list = float(s["list_rev"].sum())
    res.facts = clean_facts(dict(
        discounted_lines_pct=pct1(len(d_) / len(s)), average_discount_when_given_pct=r1(float(d_["disc"].mean() * 100)) if len(d_) else None, discount_cost=inr(float(s["discount_cost"].sum())),
        discount_cost_pct_of_list_revenue=pct1(safe_div(float(s["discount_cost"].sum()), tot_list)), quantity_difference_discounted_vs_full_price_pct=pct1(qdiff),
        quantity_difference_at_30pct_plus=pct1(deep_q), deep_discount_lines=int(len(deep)), deep_discount_cost=inr(float(deep["discount_cost"].sum())),
        lines_sold_below_cost=loss_lines, loss_on_below_cost_lines=None if loss_total is None else inr(loss_total)))
    if deep_q is not None and len(deep) >= 30 and deep_q < 0.05:
        res.add_flag(AMBER, f"Discounts of 30% or more cost {inr(float(deep['discount_cost'].sum()))} and orders are not any bigger", f"Quantity per line is {deep_q:+.0%} versus full price")
    if qdiff is not None and qdiff < 0.03 and len(d_) / len(s) > 0.15:
        res.add_flag(AMBER, f"{len(d_) / len(s):.0%} of lines are discounted but quantity per line is only {qdiff:+.0%} versus full price")
    if loss_lines:
        res.add_flag(RED, f"{loss_lines} sales lines were sold below cost after discount", f"{inr(loss_total)} lost")
    if tot_list and float(s["discount_cost"].sum()) / tot_list > 0.08:
        res.add_flag(AMBER, f"Discounts give away {float(s['discount_cost'].sum()) / tot_list:.0%} of list-price revenue")
    if not res.flags:
        res.add_flag(GREEN, "Discounting is limited and volumes respond to it")
    res.notes.append("The sales price is taken as the price actually charged, after the discount; the discount cost is the difference from the list price implied by discount_pct. "
                     "The quantity comparison is within each product (at least 20 full-price and 20 discounted lines) and shows an association, not proof the discount caused the order size: "
                     "discounts are often given to customers who were going to buy anyway. Margins use your product costs where available.")
    return res


def rec_discount(f, flags):
    r = []
    if f.get("quantity_difference_at_30pct_plus") is not None and f["quantity_difference_at_30pct_plus"] < 5:
        r.append(f"Stop or cap discounts of 30% or more: they cost {f.get('deep_discount_cost')} and orders did not grow. Run any deep discount as a test against a group that gets none.")
    if f.get("lines_sold_below_cost"):
        r.append(f"Block discounts that take the price below cost: {f['lines_sold_below_cost']} lines lost {f.get('loss_on_below_cost_lines')}.")
    return r or ["Discounting looks controlled; keep a ceiling on discount percentage and track it monthly."]


# ============================================================================== AnomalyIQ
def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if not mad or pd.isna(mad):
        return pd.Series(np.nan, index=x.index)
    return 0.6745 * (x - med) / mad


def run_anomaly(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("AnomalyIQ")
    rows = []
    if "sales" in data:
        s = data["sales"].copy()
        s["rev"] = s["quantity"] * s["unit_price"]
        daily = s.groupby(s["date"].dt.normalize())["rev"].sum()
        if len(daily) >= 56:
            daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0.0)
            for wd, g in daily.groupby(daily.index.weekday):                                   # compare Mondays with Mondays
                z = _robust_z(g)
                med = g.median()
                for dt, v in g[z.abs() >= 3.5].items():
                    rows.append(dict(series="Daily revenue", period=dt.strftime("%Y-%m-%d"), value=float(v), expected=float(med), z=float(z[dt])))
    if "expenses" in data:
        e = data["expenses"].copy()
        e["m"] = e["date"].dt.to_period("M").dt.to_timestamp()
        e = e[_month_end_done(e["m"], ctx.as_of)]
        pv = e.pivot_table(index="m", columns="category", values="amount", aggfunc="sum", fill_value=0.0)
        if len(pv) >= 6:
            for cat in pv.columns:
                z = _robust_z(pv[cat])
                med = pv[cat].median()
                for m, v in pv[cat][z.abs() >= 3.5].items():
                    rows.append(dict(series=f"Monthly spend: {cat}", period=m.strftime("%Y-%m"), value=float(v), expected=float(med), z=float(z[m])))
    if "sales" not in data and "expenses" not in data:
        raise ValueError("needs sales or expenses")
    an = pd.DataFrame(rows, columns=["series", "period", "value", "expected", "z"])
    if len(an):
        an["deviation_pct"] = (100 * (an["value"] - an["expected"]) / an["expected"].replace(0, np.nan)).round(0)
        an["direction"] = np.where(an["value"] > an["expected"], "above normal", "below normal")
        an = an.assign(abs_z=an["z"].abs()).sort_values("abs_z", ascending=False).drop(columns="abs_z").round(2)
        recent_cut = (ctx.as_of - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        recent = an[an["period"] >= recent_cut[:7]] if an["period"].str.len().eq(7).all() else an[an["period"] >= recent_cut]
    else:
        recent = an
    res.tables["unusual_values"] = an.head(30)
    res.facts = clean_facts(dict(
        unusual_values_found=int(len(an)), unusual_values_last_30_days=int(len(recent)),
        biggest_series=None if an.empty else str(an.iloc[0]["series"]), biggest_period=None if an.empty else str(an.iloc[0]["period"]),
        biggest_deviation_pct=None if an.empty or pd.isna(an.iloc[0]["deviation_pct"]) else float(an.iloc[0]["deviation_pct"]),
        series_checked=int(("sales" in data) + (len(pv.columns) if "expenses" in data and len(pv) >= 6 else 0))))
    if len(recent):
        top = recent.iloc[0]
        res.add_flag(AMBER, f"{len(recent)} unusual value(s) in the last 30 days, the biggest: {top['series']} on {top['period']} was {top['direction']}",
                     f"{top['deviation_pct']:+.0f}% versus its usual level" if not pd.isna(top["deviation_pct"]) else "")
    elif len(an):
        res.add_flag(GREEN, f"{len(an)} unusual value(s) in the past, none in the last 30 days")
    else:
        res.add_flag(GREEN, "No values stand out from their normal pattern")
    res.notes.append("A value is 'unusual' when it is far from the usual level for the same weekday (daily revenue) or the same category (monthly spend), measured with a robust score of 3.5 or more. "
                     "Unusual is not wrong: festivals, one-off orders, annual bills and closures all show up. This tells you where to look, not what happened. Needs 8 weeks of sales or 6 months of expenses.")
    return res


def rec_anomaly(f, flags):
    if f.get("unusual_values_last_30_days"):
        return [f"Open {f.get('biggest_series')} around {f.get('biggest_period')} and find the cause; if it is a one-off, note it so the same pattern is not mistaken for a trend."]
    return ["Nothing unusual lately; this check runs automatically on each new upload."]


register(ModuleSpec("BudgetIQ", "Finance and compliance", "Budget versus actual by category, with overspend and padded budgets.", ["budgets", "expenses"], run_budget, rec_budget, audience="Both", phase="4"))
register(ModuleSpec("ScenarioIQ", "Finance and compliance", "What-if on price, volume and costs: how much cushion your profit has.", ["financials"], run_scenario, rec_scenario, audience="Both", phase="4"))
register(ModuleSpec("UnitEconomicsIQ", "Growth and marketing", "What a new customer costs to win and whether they pay it back.", ["sales", "campaigns"], run_unit_economics, rec_unit_economics, audience="Both", phase="4"))
register(ModuleSpec("DiscountIQ", "Growth and marketing", "What discounts cost and whether they actually lift order size.", ["sales"], run_discount, rec_discount, audience="Both", phase="4"))
register(ModuleSpec("AnomalyIQ", "Risk and control", "Revenue days and spend months that are far from normal.", [], run_anomaly, rec_anomaly, audience="Both", phase="4"))
