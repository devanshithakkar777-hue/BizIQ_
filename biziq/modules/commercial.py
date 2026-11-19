"""Commercial modules: PricingIQ, ForecastIQ, ReturnsIQ, CampaignIQ, LoyaltyIQ."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import AMBER, GREEN, RED, ModuleResult, inr, safe_div
from ..registry import Context, ModuleSpec, register
from ._common import clean_facts, pct1, r1

# ============================================================================== PricingIQ
TARGET_MARGIN = 0.20          # assumption: net margin the price should clear after fees and shipping
HIGH_GAP, LOW_GAP = 0.10, -0.08


def _elasticity(weekly: pd.DataFrame) -> float:
    """Log-log slope of weekly units on weekly average price; NaN when prices barely moved."""
    w = weekly[(weekly["units"] > 0) & (weekly["price"] > 0)]
    if len(w) < 12 or w["price"].std() / w["price"].mean() < 0.02:
        return float("nan")
    x, y = np.log(w["price"]), np.log(w["units"])
    return float(np.polyfit(x, y, 1)[0])


def run_pricing(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("PricingIQ")
    sales, costs = data["sales"], data["product_costs"].drop_duplicates("product", keep="last").set_index("product")
    comp = data.get("competitor_prices")
    recent = sales[sales["date"] > ctx.as_of - pd.Timedelta(days=90)]
    price_now = recent.groupby("product").apply(lambda g: g["revenue"].sum() / g["quantity"].sum(), include_groups=False)
    weekly = sales.assign(week=sales["date"].dt.to_period("W").dt.start_time).groupby(["product", "week"]).agg(
        units=("quantity", "sum"), rev=("revenue", "sum")).reset_index()
    weekly["price"] = weekly["rev"] / weekly["units"]
    rows = []
    comp_now = None
    if comp is not None and len(comp):
        last = comp[comp["date"] >= comp["date"].max() - pd.Timedelta(days=14)]
        comp_now = last.groupby("product")["price"].mean()
    for p, price in price_now.items():
        if p not in costs.index:
            continue
        c = costs.loc[p]
        fee = float(c.get("channel_fee_pct", 0) or 0) / 100
        fixed = float(c.get("shipping_cost_per_order", 0) or 0) + float(c.get("packaging_cost", 0) or 0)
        margin_now = (price * (1 - fee) - c["unit_cost"] - fixed) / price
        floor = (c["unit_cost"] + fixed) / (1 - fee - TARGET_MARGIN)
        el = _elasticity(weekly[weekly["product"] == p])
        gap = safe_div(price - comp_now[p], comp_now[p]) if comp_now is not None and p in comp_now.index else float("nan")
        if margin_now < 0.05:
            sug = f"Margin too thin: raise toward {inr(floor, compact=False)} or cut cost"
        elif pd.notna(gap) and gap > HIGH_GAP and margin_now > TARGET_MARGIN + 0.08:
            sug = "Priced above competitors with healthy margin: test a small cut to win volume"
        elif pd.notna(gap) and gap > HIGH_GAP:
            sug = "Priced above competitors but margin is tight: fix cost before cutting price"
        elif pd.notna(gap) and gap < LOW_GAP and margin_now < 0.35:
            sug = "Cheaper than competitors: test a small increase"
        else:
            sug = "Hold price"
        rows.append(dict(product=p, our_price=price, competitor_avg=comp_now.get(p, np.nan) if comp_now is not None else np.nan,
                         gap_vs_competitors=gap, margin_at_current_price=margin_now, price_for_target_margin=floor,
                         elasticity=el, suggestion=sug))
    if not rows:
        raise ValueError("needs a cost in product_costs for the products sold in the last 90 days")
    t = pd.DataFrame(rows).sort_values("margin_at_current_price")
    res.tables["products"] = t
    if comp is not None and len(comp):
        trend = comp.assign(week=comp["date"].dt.to_period("W").dt.start_time).groupby(["week", "product"])["price"].mean().unstack()
        res.tables["competitor_trend"] = trend.reset_index()
    above, below = t[t["gap_vs_competitors"] > HIGH_GAP], t[t["gap_vs_competitors"] < LOW_GAP]
    below_floor = t[t["margin_at_current_price"] < TARGET_MARGIN]
    est = int(t["elasticity"].notna().sum())
    res.metrics = dict(products=len(t), above_market=len(above), below_market=len(below))
    res.facts = clean_facts(dict(
        products_priced=len(t), products_above_competitors=len(above), products_below_competitors=len(below),
        products_below_target_margin=len(below_floor), target_margin_pct=pct1(TARGET_MARGIN),
        most_overpriced_product=str(above.sort_values("gap_vs_competitors").iloc[-1]["product"]) if len(above) else None,
        most_overpriced_gap_pct=pct1(above["gap_vs_competitors"].max()) if len(above) else None,
        most_underpriced_product=str(below.sort_values("gap_vs_competitors").iloc[0]["product"]) if len(below) else None,
        most_underpriced_gap_pct=pct1(below["gap_vs_competitors"].min()) if len(below) else None,
        lowest_margin_product=str(t.iloc[0]["product"]), lowest_margin_pct=pct1(t.iloc[0]["margin_at_current_price"]),
        price_for_target_margin_lowest=inr(float(t.iloc[0]["price_for_target_margin"]), compact=False),
        products_with_elasticity_estimate=est))
    for _, r in above.iterrows():
        res.add_flag(AMBER, f"{r['product']} is {r['gap_vs_competitors']:.0%} above competitor prices")
    for _, r in t[t["margin_at_current_price"] < 0.05].iterrows():
        res.add_flag(RED, f"{r['product']} earns only {r['margin_at_current_price']:.1%} at its current price",
                     f"About {inr(r['price_for_target_margin'], compact=False)} would earn the {TARGET_MARGIN:.0%} target.")
    if not res.flags:
        res.add_flag(GREEN, "Prices look competitive and profitable")
    res.notes.append(f"Margin is after marketplace fee, shipping and packaging; the target is {TARGET_MARGIN:.0%} (an assumption you can change). "
                     f"Elasticity is estimated only where prices actually moved week to week ({est} of {len(t)} products here); "
                     "where it is blank there is no evidence about how demand reacts to price, so suggestions rest on competitor gaps and margin. "
                     "Treat every suggestion as a small test, not a rule.")
    return res


def rec_pricing(f, flags):
    r = []
    if f.get("products_below_target_margin"):
        r.append(f"{f['products_below_target_margin']} product(s) earn under the {f['target_margin_pct']}% target margin; {f['lowest_margin_product']} "
                 f"is lowest ({f['lowest_margin_pct']}%) and would need about {f['price_for_target_margin_lowest']} to reach the target.")
    if f.get("most_overpriced_product"):
        r.append(f"{f['most_overpriced_product']} is {f['most_overpriced_gap_pct']}% above competitors; if its margin allows, test a small cut.")
    if f.get("most_underpriced_product"):
        r.append(f"{f['most_underpriced_product']} is {abs(f['most_underpriced_gap_pct'])}% below competitors; test a small increase for a few weeks and watch units.")
    return r or ["Prices are in line with competitors and margins."]


# ============================================================================== ForecastIQ
def _holt(y: np.ndarray, alpha: float, beta: float, phi: float, h: int) -> tuple[np.ndarray, np.ndarray]:
    level, trend = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0
    fitted = [y[0]]
    for t in range(1, len(y)):
        fitted.append(level + phi * trend)
        new_level = alpha * y[t] + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    steps = np.cumsum([phi ** k for k in range(1, h + 1)])
    return np.array(fitted), level + steps * trend


def _best_holt(y: np.ndarray, h: int):
    best = None
    for a in (0.2, 0.4, 0.6, 0.8):
        for b in (0.1, 0.3, 0.5):
            for ph in (0.8, 0.9, 0.98):
                fit, fc = _holt(y, a, b, ph, h)
                sse = float(((y[1:] - fit[1:]) ** 2).sum())
                if best is None or sse < best[0]:
                    best = (sse, fit, fc)
    return best[1], best[2]


def _forecast_series(y: np.ndarray, h: int = 3, holdout: int = 3):
    """Pick the best of naive / moving average / damped Holt by backtest on the last `holdout` points."""
    methods = {"naive (last month repeats)": lambda s, k: np.repeat(s[-1], k),
               "3-month average": lambda s, k: np.repeat(s[-3:].mean(), k),
               "damped trend": lambda s, k: _best_holt(s, k)[1]}
    scores = {}
    if len(y) >= holdout + 5:
        train, test = y[:-holdout], y[-holdout:]
        for name, fn in methods.items():
            pred = fn(train, holdout)
            scores[name] = float(np.mean(np.abs(pred - test) / np.abs(test)))
    pick = min(scores, key=scores.get) if scores else "3-month average"
    fc = methods[pick](y, h)
    resid_sd = float(np.std(y - pd.Series(y).rolling(3, min_periods=1).mean().to_numpy()))
    return pick, fc, resid_sd, scores


def run_forecast(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ForecastIQ")
    s = data["sales"]
    s = s[s["date"] <= ctx.as_of]
    m = s.assign(month=s["date"].dt.to_period("M").dt.to_timestamp()).groupby("month")["revenue"].sum()
    if len(m) and (ctx.as_of + pd.Timedelta(days=1)).day != 1:     # last month is incomplete: drop it
        m = m.iloc[:-1]
    if len(m) < 6:
        raise ValueError("needs at least 6 complete months of sales to forecast")
    y = m.to_numpy(dtype=float)
    pick, fc, sd, scores = _forecast_series(y)
    months_f = pd.date_range(m.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq="MS")
    z = 1.28                      # about an 80% range
    fdf = pd.DataFrame(dict(month=months_f, forecast=fc, low=np.maximum(fc - z * sd * np.sqrt(np.arange(1, 4)), 0),
                            high=fc + z * sd * np.sqrt(np.arange(1, 4))))
    res.tables["revenue_forecast"] = fdf
    res.tables["history"] = m.reset_index()
    # product-level next-3-months units
    q = s.assign(month=s["date"].dt.to_period("M").dt.to_timestamp())
    if (ctx.as_of + pd.Timedelta(days=1)).day != 1:
        q = q[q["month"] < q["month"].max()]
    pu = q.pivot_table(index="month", columns="product", values="quantity", aggfunc="sum").fillna(0.0)
    rows = []
    for p in pu.columns:
        k, f3, _, sc = _forecast_series(pu[p].to_numpy(dtype=float))
        rows.append(dict(product=p, last_3_months_units=float(pu[p].tail(3).sum()), next_3_months_units=float(max(f3.sum(), 0)),
                         method=k, backtest_error=sc.get(k, np.nan)))
    pt = pd.DataFrame(rows)
    pt["change"] = pt["next_3_months_units"] / pt["last_3_months_units"].replace(0, np.nan) - 1
    res.tables["product_forecast"] = pt.sort_values("next_3_months_units", ascending=False)
    err = scores.get(pick, float("nan"))
    last3 = float(y[-3:].sum())
    growth = safe_div(float(fc.sum()) - last3, last3)
    res.metrics = dict(next_3_months=float(fc.sum()), backtest_error=err)
    res.facts = clean_facts(dict(
        next_month_revenue=inr(float(fc[0])), next_month_range=f"{inr(float(fdf['low'].iloc[0]))} to {inr(float(fdf['high'].iloc[0]))}",
        next_3_months_revenue=inr(float(fc.sum())), last_3_months_revenue=inr(last3), expected_change_pct=pct1(growth),
        method=pick, backtest_error_pct=pct1(err), months_of_history=len(y),
        fastest_growing_product=str(pt.sort_values("change").iloc[-1]["product"]) if pt["change"].notna().any() else None,
        fastest_growing_change_pct=pct1(pt["change"].max()) if pt["change"].notna().any() else None,
        fastest_declining_product=str(pt.sort_values("change").iloc[0]["product"]) if pt["change"].notna().any() else None,
        fastest_declining_change_pct=pct1(pt["change"].min()) if pt["change"].notna().any() else None))
    if pd.notna(err) and err > 0.25:
        res.add_flag(AMBER, f"Forecast accuracy is low (typical error {err:.0%} on a three-month back-test)",
                     "Treat the range, not the single number, as the plan.")
    if pd.notna(growth) and growth < -0.10:
        res.add_flag(AMBER, f"Revenue is forecast to fall {abs(growth):.0%} versus the last three months")
    if len(y) < 18:
        res.add_flag(AMBER, f"Only {len(y)} months of history", "Seasonality cannot be measured; festive peaks are not modelled.")
    if not res.flags:
        res.add_flag(GREEN, "Forecast is stable and back-tested")
    res.notes.append(f"Method chosen by back-test on the last 3 known months: {pick}. Candidates: naive, 3-month average, damped trend. "
                     "The range is an approximate 80% band from past volatility. With under two years of history, seasonality is not modelled, "
                     "so festive spikes will be missed. Forecasts are statistical projections, not promises.")
    return res


def rec_forecast(f, flags):
    r = [f"Plan for about {f['next_month_revenue']} next month (likely range {f['next_month_range']}) and {f['next_3_months_revenue']} over three months "
         f"({f['expected_change_pct']}% versus the last three)."]
    if f.get("backtest_error_pct", 0) and f["backtest_error_pct"] > 20:
        r.append(f"Accuracy is modest (typical error {f['backtest_error_pct']}%), so keep buffers in stock and cash.")
    if f.get("fastest_growing_product"):
        r.append(f"{f['fastest_growing_product']} is trending up ({f['fastest_growing_change_pct']}%) and {f['fastest_declining_product']} down "
                 f"({f['fastest_declining_change_pct']}%); adjust reorders accordingly.")
    return r


# ============================================================================== ReturnsIQ
def run_returns(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ReturnsIQ")
    ret, sales = data["returns"].copy(), data["sales"]
    cutoff = ret["date"].max()
    sold = sales[sales["date"] <= cutoff]
    n_sold = sold.groupby("product").size()
    by_prod = ret.groupby("product").agg(returns=("return_id", "size"), refunded=("refund_amount", "sum"))
    by_prod["lines_sold"] = n_sold.reindex(by_prod.index)
    by_prod["return_rate"] = by_prod["returns"] / by_prod["lines_sold"]
    top_reason = ret.groupby(["product", "reason"]).size().reset_index(name="n").sort_values("n", ascending=False).drop_duplicates("product").set_index("product")["reason"]
    by_prod["top_reason"] = top_reason.reindex(by_prod.index)
    res.tables["by_product"] = by_prod.reset_index().sort_values("return_rate", ascending=False)
    reasons = ret.groupby("reason").agg(returns=("return_id", "size"), refunded=("refund_amount", "sum"))
    reasons["share"] = reasons["returns"] / len(ret)
    res.tables["by_reason"] = reasons.sort_values("returns", ascending=False).reset_index()
    if "channel" in ret:
        ch = ret.groupby("channel").agg(returns=("return_id", "size"), refunded=("refund_amount", "sum"))
        ch["share_of_returns"] = ch["returns"] / len(ret)
        res.tables["by_channel"] = ch.sort_values("returns", ascending=False).reset_index()
    if "is_cod" in ret:
        cod = ret.groupby("is_cod").agg(returns=("return_id", "size"), refunded=("refund_amount", "sum")).reset_index()
        cod["type"] = cod["is_cod"].map({1: "Cash on delivery", 0: "Prepaid"})
        res.tables["cod_vs_prepaid"] = cod[["type", "returns", "refunded"]]
    reg = ret.merge(sales[["order_id", "product", "region", "customer_id"]].drop_duplicates(["order_id", "product"]), on=["order_id", "product"], how="left")
    unmatched = int(reg["region"].isna().sum())
    reg["region"] = reg["region"].fillna("Unknown (order not in sales data)")
    rr = reg.groupby("region").size().rename("returns").to_frame()
    rr["lines_sold"] = sold.groupby("region").size().reindex(rr.index)
    rr["return_rate"] = rr["returns"] / rr["lines_sold"]
    res.tables["by_region"] = rr.reset_index().sort_values("return_rate", ascending=False)
    repeat = reg.groupby("customer_id").size()
    repeat = repeat[repeat >= 3].sort_values(ascending=False)
    res.tables["repeat_returners"] = repeat.rename("returns").reset_index()
    overall = safe_div(len(ret), len(sold))
    worst = by_prod.sort_values("return_rate", ascending=False).iloc[0]
    cod_share = pct1(float(ret["is_cod"].mean())) if "is_cod" in ret else None
    top_r = reasons.sort_values("returns", ascending=False).iloc[0]
    fit_share = float(reasons.loc[reasons.index.str.contains("Size", case=False), "returns"].sum() / len(ret))
    res.metrics = dict(return_rate=overall, refunded=float(ret["refund_amount"].sum()))
    res.facts = clean_facts(dict(
        return_rate_pct=pct1(overall), returns=len(ret), refunded_amount=inr(float(ret["refund_amount"].sum())),
        highest_return_product=str(worst.name), highest_return_rate_pct=pct1(worst["return_rate"]),
        highest_return_top_reason=str(worst["top_reason"]),
        top_reason=str(top_r.name), top_reason_share_pct=pct1(top_r["returns"] / len(ret)), size_or_fit_share_pct=pct1(fit_share),
        cod_share_of_returns_pct=cod_share, repeat_returners=len(repeat),
        highest_return_region=str(rr["return_rate"].dropna().idxmax()) if rr["return_rate"].notna().any() else "n/a",
        highest_return_region_rate_pct=pct1(rr["return_rate"].max()) if rr["return_rate"].notna().any() else None,
        returns_not_matched_to_sales=unmatched))
    for p, r in by_prod.iterrows():
        if pd.isna(r["return_rate"]):
            continue
        if r["return_rate"] > 0.15:
            res.add_flag(RED, f"{p} has a {r['return_rate']:.0%} return rate", f"Most common reason: {r['top_reason']}.")
        elif r["return_rate"] > 0.10:
            res.add_flag(AMBER, f"{p} has a {r['return_rate']:.0%} return rate", f"Most common reason: {r['top_reason']}.")
    if overall > 0.10:
        res.add_flag(AMBER, f"Overall return rate is {overall:.1%}")
    if len(repeat):
        res.add_flag(AMBER, f"{len(repeat)} customer(s) have returned 3 or more orders", "Check for abuse or a product mismatch.")
    if not res.flags:
        res.add_flag(GREEN, "Return rates are low")
    res.notes.append("Return rate = returned order lines / order lines sold up to the last return date. Channel and cash-on-delivery views show "
                     "shares of returns only: the sales data has no channel or payment mode, so true return rates by channel or COD cannot be "
                     "computed until orders carry those fields.")
    return res


def rec_returns(f, flags):
    r = [f"Overall return rate is {f['return_rate_pct']}% ({f['refunded_amount']} refunded). {f['highest_return_product']} is highest at "
         f"{f['highest_return_rate_pct']}%, mainly for '{f['highest_return_top_reason']}'."]
    if f.get("size_or_fit_share_pct", 0) and f["size_or_fit_share_pct"] > 25:
        r.append(f"Size or fit drives {f['size_or_fit_share_pct']}% of returns; add a size guide, fit notes and customer photos on the product page.")
    if f.get("cod_share_of_returns_pct", 0) and f["cod_share_of_returns_pct"] > 30:
        r.append(f"{f['cod_share_of_returns_pct']}% of returns are cash-on-delivery; confirm COD orders on WhatsApp before dispatch or add a small prepaid incentive.")
    if f.get("repeat_returners"):
        r.append(f"Review the {f['repeat_returners']} repeat returner(s) before the next order.")
    return r


# ============================================================================== CampaignIQ
def run_campaigns(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("CampaignIQ")
    c = data["campaigns"].copy()
    has_rev = "revenue" in c and c["revenue"].notna().any()
    has_cpc = False
    agg = {"spend": "sum", "impressions": "sum", "clicks": "sum", "conversions": "sum"}
    if has_rev:
        agg["revenue"] = "sum"
    ch = c.groupby("channel").agg(agg)
    ch["ctr"] = ch["clicks"] / ch["impressions"].replace(0, np.nan)
    ch["cpc"] = ch["spend"] / ch["clicks"].replace(0, np.nan)
    ch["conversion_rate"] = ch["conversions"] / ch["clicks"].replace(0, np.nan)
    ch["cost_per_conversion"] = ch["spend"] / ch["conversions"].replace(0, np.nan)
    if has_rev:
        ch["roas"] = ch["revenue"] / ch["spend"].replace(0, np.nan)
    last = c["date"].max()
    recent = c[c["date"] > last - pd.Timedelta(days=30)].groupby("channel")[["spend", "revenue"] if has_rev else ["spend"]].sum()
    prior = c[(c["date"] <= last - pd.Timedelta(days=30)) & (c["date"] > last - pd.Timedelta(days=60))].groupby("channel")[["spend", "revenue"] if has_rev else ["spend"]].sum()
    if has_rev:
        ch["roas_last_30d"] = (recent["revenue"] / recent["spend"]).reindex(ch.index)
        ch["roas_prior_30d"] = (prior["revenue"] / prior["spend"]).reindex(ch.index)
        ch["roas_change"] = ch["roas_last_30d"] / ch["roas_prior_30d"] - 1
    ch["share_of_spend"] = ch["spend"] / ch["spend"].sum()
    res.tables["channels"] = ch.reset_index().sort_values("roas" if has_rev else "spend", ascending=False)
    wk = c.assign(week=c["date"].dt.to_period("W").dt.start_time).groupby("week")[["spend"] + (["revenue"] if has_rev else [])].sum()
    res.tables["weekly"] = wk.reset_index()
    has_cpc = bool(ch["cost_per_conversion"].notna().any())
    total = float(ch["spend"].sum())
    blended = safe_div(float(ch["revenue"].sum()), total) if has_rev else None
    weak = ch[ch["roas"] < 1] if has_rev else ch.iloc[0:0]
    strong = ch[ch["roas"] > 3] if has_rev else ch.iloc[0:0]
    moves = []
    if has_rev:
        for w in weak.index:
            moves.append(dict(action=f"Cut or pause {w}", from_channel=w, spend_to_move=float(ch.loc[w, "spend"] * 0.5 / (len(c["date"].unique()) / 30)),
                              reason=f"Returns {ch.loc[w, 'roas']:.2f} of revenue per rupee spent"))
    res.tables["budget_moves"] = pd.DataFrame(moves)
    res.metrics = dict(total_spend=total, blended_roas=blended, loss_making_channels=len(weak))
    res.facts = clean_facts(dict(
        total_spend=inr(total), blended_roas=r1(blended, 2), period_days=int(c["date"].nunique()),
        best_channel=str(ch["roas"].idxmax()) if has_rev and ch["roas"].notna().any() else None, best_channel_roas=r1(ch["roas"].max(), 2) if has_rev and ch["roas"].notna().any() else None,
        worst_channel=str(ch["roas"].idxmin()) if has_rev and ch["roas"].notna().any() else None, worst_channel_roas=r1(ch["roas"].min(), 2) if has_rev and ch["roas"].notna().any() else None,
        channels_below_break_even=len(weak), spend_share_on_weak_channels_pct=pct1(float(weak["share_of_spend"].sum())) if has_rev else None,
        cheapest_conversion_channel=str(ch["cost_per_conversion"].idxmin()) if has_cpc else None,
        cheapest_cost_per_conversion=inr(float(ch["cost_per_conversion"].min()), compact=False) if has_cpc else None,
        costliest_conversion_channel=str(ch["cost_per_conversion"].idxmax()) if has_cpc else None,
        costliest_cost_per_conversion=inr(float(ch["cost_per_conversion"].max()), compact=False) if has_cpc else None))
    for n, r in weak.iterrows():
        res.add_flag(RED, f"{n} returns only {r['roas']:.2f} of revenue per rupee spent", f"{r['share_of_spend']:.0%} of the ad budget goes to it.")
    if has_rev:
        for n, r in ch.iterrows():
            if pd.notna(r.get("roas_change")) and r["roas_change"] < -0.2 and r["roas"] >= 1:
                res.add_flag(AMBER, f"{n} return on ad spend fell {abs(r['roas_change']):.0%} in the last 30 days")
    if not res.flags:
        res.add_flag(GREEN, "Every channel earns back more than it costs")
    res.notes.append("Return on ad spend (ROAS) = revenue attributed to the channel / its spend. Attribution is whatever your ad platforms report, "
                     "which usually over-credits the last click. ROAS is revenue, not profit: a ROAS under about 3 may still lose money after "
                     "product costs (compare with ProfitIQ margins). Moving budget to strong channels has diminishing returns; shift in small steps.")
    return res


def rec_campaigns(f, flags):
    r = []
    if f.get("channels_below_break_even"):
        r.append(f"{f['worst_channel']} returns {f['worst_channel_roas']} of revenue per rupee; {f['spend_share_on_weak_channels_pct']}% of spend sits in channels "
                 "below break-even. Pause or rework them, then move about half of that budget to the strongest channel in small steps.")
    r.append(f"Best channel: {f['best_channel']} at {f['best_channel_roas']}x; blended return is {f['blended_roas']}x on {f['total_spend']} of spend.")
    r.append(f"Cheapest conversion is {f['cheapest_conversion_channel']} ({f['cheapest_cost_per_conversion']}); costliest is "
             f"{f['costliest_conversion_channel']} ({f['costliest_cost_per_conversion']}).")
    return r


# ============================================================================== LoyaltyIQ
TIER_CUTS = {"Platinum": 0.95, "Gold": 0.80, "Silver": 0.50}      # percentile of customer revenue
ASSUMED_GROSS_MARGIN = 0.35


def run_loyalty(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("LoyaltyIQ")
    s = data["sales"]
    s = s[s["date"] <= ctx.as_of]
    cust = s.groupby("customer_id").agg(orders=("order_id", "nunique"), revenue=("revenue", "sum"),
                                        first=("date", "min"), last=("date", "max"))
    cust["days_since_last"] = (ctx.as_of - cust["last"]).dt.days
    cust["repeat"] = cust["orders"] >= 2
    q = {k: cust["revenue"].quantile(v) for k, v in TIER_CUTS.items()}
    cust["tier"] = np.select([cust["revenue"] >= q["Platinum"], cust["revenue"] >= q["Gold"], cust["revenue"] >= q["Silver"]],
                             ["Platinum", "Gold", "Silver"], "Bronze")
    tiers = cust.groupby("tier").agg(customers=("revenue", "size"), revenue=("revenue", "sum"), avg_orders=("orders", "mean"),
                                     active_90d=("days_since_last", lambda x: float((x <= 90).mean())))
    tiers["share_of_revenue"] = tiers["revenue"] / tiers["revenue"].sum()
    tiers = tiers.reindex(["Platinum", "Gold", "Silver", "Bronze"])
    res.tables["tiers"] = tiers.reset_index()
    # retention by first-purchase month: share of the cohort buying again in the following months
    s2 = s.assign(month=s["date"].dt.to_period("M"))
    first_month = s2.groupby("customer_id")["month"].min().rename("cohort")
    s2 = s2.join(first_month, on="customer_id")
    s2["age"] = (s2["month"] - s2["cohort"]).apply(lambda x: x.n)
    size = s2[s2["age"] == 0].groupby("cohort")["customer_id"].nunique()
    ret = {}
    for k in (1, 2, 3):
        ret[f"month_{k}"] = s2[s2["age"] == k].groupby("cohort")["customer_id"].nunique().reindex(size.index).div(size)
    coh = pd.DataFrame(ret).assign(new_customers=size)
    coh = coh[coh.index <= ctx.as_of.to_period("M") - 3]            # only cohorts old enough to be measured
    res.tables["cohort_retention"] = coh.reset_index().assign(cohort=lambda d: d["cohort"].astype(str))
    # reward programme cost vs break-even uplift, at an assumed gross margin
    gm = ASSUMED_GROSS_MARGIN
    prof = ctx.modules.get("ProfitIQ")
    if prof is not None and prof.metrics.get("net_margin") is not None:
        gm = float(prof.metrics["net_margin"])
    reward = ctx.settings.get("loyalty_reward_rate", 0.05)
    cost = float(cust["revenue"].sum() * reward)
    breakeven = safe_div(reward, max(gm - reward, 1e-6))
    res.tables["reward_scenarios"] = pd.DataFrame([dict(reward_rate=r, annual_cost=float(cust["revenue"].sum() * r),
                                                        break_even_extra_revenue_pct=safe_div(r, max(gm - r, 1e-6)))
                                                   for r in (0.02, 0.03, 0.05, 0.08) if gm > r])
    dormant = cust[(cust["days_since_last"] > 90) & (cust["tier"].isin(["Gold", "Platinum"]))]
    res.tables["valuable_but_dormant"] = dormant.sort_values("revenue", ascending=False).head(25).reset_index()[
        ["customer_id", "tier", "orders", "revenue", "days_since_last"]]
    repeat_rate = float(cust["repeat"].mean())
    m1 = float(coh["month_1"].mean()) if len(coh) else float("nan")
    res.metrics = dict(repeat_rate=repeat_rate, customers=len(cust))
    res.facts = clean_facts(dict(
        customers=len(cust), repeat_customer_rate_pct=pct1(repeat_rate), avg_orders_per_customer=r1(float(cust["orders"].mean()), 1),
        active_last_90_days_pct=pct1(float((cust["days_since_last"] <= 90).mean())),
        platinum_customers=int(tiers.loc["Platinum", "customers"]), platinum_revenue_share_pct=pct1(tiers.loc["Platinum", "share_of_revenue"]),
        gold_revenue_share_pct=pct1(tiers.loc["Gold", "share_of_revenue"]),
        month_1_retention_pct=pct1(m1), valuable_dormant_customers=len(dormant),
        valuable_dormant_revenue=inr(float(dormant["revenue"].sum())),
        assumed_reward_rate_pct=pct1(reward), reward_programme_annual_cost=inr(cost), break_even_extra_revenue_pct=pct1(breakeven),
        margin_used_pct=pct1(gm)))
    if repeat_rate < 0.25:
        res.add_flag(AMBER, f"Only {repeat_rate:.0%} of customers have bought more than once")
    if len(dormant):
        res.add_flag(AMBER, f"{len(dormant)} Gold or Platinum customers have not bought in 90+ days",
                     f"They represent {inr(float(dormant['revenue'].sum()))} of past revenue.")
    if not res.flags:
        res.add_flag(GREEN, "Repeat purchasing is healthy")
    res.notes.append("Tiers are percentile bands of lifetime revenue (Silver from the 50th, Gold 80th, Platinum 95th), a transparent rule. "
                     f"The reward scenario costs {reward:.0%} of revenue and shows the extra revenue needed to break even at a "
                     f"{gm:.0%} margin; it does NOT predict the uplift, which only a live test can measure. Month-N retention counts "
                     "customers from each first-purchase cohort who bought again N months later.")
    return res


def rec_loyalty(f, flags):
    r = [f"{f['repeat_customer_rate_pct']}% of customers have bought more than once, and Platinum customers ({f['platinum_customers']}) "
         f"give {f['platinum_revenue_share_pct']}% of revenue. Protect them first."]
    if f.get("valuable_dormant_customers"):
        r.append(f"{f['valuable_dormant_customers']} high-value customers have gone quiet ({f['valuable_dormant_revenue']} of past revenue); "
                 "send a personal win-back message before any discount.")
    r.append(f"A {f['assumed_reward_rate_pct']}% reward programme would cost about {f['reward_programme_annual_cost']} a year and must lift revenue "
             f"by {f['break_even_extra_revenue_pct']}% to break even at a {f['margin_used_pct']}% margin; pilot it with one tier before rolling it out.")
    return r


# ============================================================================== registration
register(ModuleSpec("PricingIQ", "Commercial", "Price versus competitors and margin, with small tests to run.",
                    ["sales", "product_costs"], run_pricing, rec_pricing, "Both", "2"))
register(ModuleSpec("ForecastIQ", "Commercial", "Back-tested revenue and product forecasts with ranges.",
                    ["sales"], run_forecast, rec_forecast, "Both", "2"))
register(ModuleSpec("ReturnsIQ", "Commercial", "Return rates, reasons, cash-on-delivery risk and repeat returners.",
                    ["returns", "sales"], run_returns, rec_returns, "B2C", "3a"))
register(ModuleSpec("CampaignIQ", "Commercial", "Return on ad spend by channel and where to move the budget.",
                    ["campaigns"], run_campaigns, rec_campaigns, "B2C", "3a"))
register(ModuleSpec("LoyaltyIQ", "Commercial", "Repeat buying, customer tiers, retention cohorts and reward cost.",
                    ["sales"], run_loyalty, rec_loyalty, "B2C", "3a", depends_on=["ProfitIQ"]))
