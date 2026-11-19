"""SalesIQ - sales analytics and campaigns: trends and seasonality, top/bottom analysis,
segment performance, basket analysis, campaign recommendations, weekly digest."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .core import AMBER, GREEN, RED, ModuleResult, inr, pct, safe_div

# Assumed response rates per customer segment - ESTIMATES used only to size campaign uplift.
RESPONSE_RATE = {"Champions": 0.12, "Loyal": 0.10, "Potential": 0.08, "New": 0.10,
                 "Needs Attention": 0.06, "At Risk": 0.06, "Hibernating": 0.03}
CAMPAIGN = {
    "Champions": ("VIP early access", "Early access to new arrivals for your best customers."),
    "Loyal": ("Bundle and upsell", "Bundle their favourite category with a complementary product."),
    "Potential": ("Second-purchase offer", "Personalised offer on a complementary product."),
    "New": ("Welcome series", "Welcome message plus an offer for a second order within 30 days."),
    "Needs Attention": ("Re-engagement reminder", "New arrivals in their favourite category."),
    "At Risk": ("Win-back offer", "Time-limited offer (10%) on their favourite category."),
    "Hibernating": ("Reactivation email", "One low-cost 'we miss you' email; suppress if unopened."),
}


def monthly_trend(sales: pd.DataFrame) -> pd.DataFrame:
    m = sales.set_index("date").resample("MS").agg(revenue=("revenue", "sum"), units=("quantity", "sum"),
                                                   orders=("order_id", "nunique"))
    m["aov"] = m["revenue"] / m["orders"]
    m["mom_growth"] = m["revenue"].pct_change()
    m["seasonality_index"] = m["revenue"] / m["revenue"].mean()
    return m.reset_index()


MAX_BASKET = 60          # orders with more distinct products than this are skipped for pairing (pair count grows with the square)


def basket_pairs(sales: pd.DataFrame, min_support: int = 8) -> pd.DataFrame:
    """Product pairs bought together, with support, confidence and lift."""
    cols = ["product_a", "product_b", "orders_together", "support", "confidence_a_to_b", "confidence_b_to_a", "lift"]
    u = sales[["order_id", "product"]].drop_duplicates()
    n = u["order_id"].nunique()
    if n == 0:
        return pd.DataFrame(columns=cols)
    single = u.groupby("product").size()
    size = u.groupby("order_id")["product"].transform("size")
    multi = u[(size >= 2) & (size <= MAX_BASKET)]
    if multi.empty:
        return pd.DataFrame(columns=cols)
    m = multi.merge(multi, on="order_id")
    m = m[m["product_x"] < m["product_y"]]
    pair = m.groupby(["product_x", "product_y"]).size().rename("orders_together").reset_index()
    pair = pair[pair["orders_together"] >= min_support]
    if pair.empty:
        return pd.DataFrame(columns=cols)
    sa, sb = single.reindex(pair["product_x"]).values, single.reindex(pair["product_y"]).values
    c = pair["orders_together"].values
    out = pd.DataFrame({"product_a": pair["product_x"].values, "product_b": pair["product_y"].values, "orders_together": c,
                        "support": c / n, "confidence_a_to_b": c / sa, "confidence_b_to_a": c / sb,
                        "lift": (c / n) / ((sa / n) * (sb / n))})
    return out.sort_values("lift", ascending=False, kind="stable").reset_index(drop=True)


def campaign_recommendations(segments: pd.DataFrame | None) -> pd.DataFrame:
    if segments is None or segments.empty:
        return pd.DataFrame()
    rows = []
    for seg, g in segments.groupby("segment"):
        n = len(g)
        aov = float(g["aov"].mean())
        rr = RESPONSE_RATE.get(seg, 0.05)
        name, desc = CAMPAIGN.get(seg, ("General offer", "Broad promotion."))
        rows.append(dict(segment=seg, customers=n, campaign=name, description=desc,
                         assumed_response_rate=rr, avg_order_value=aov,
                         projected_uplift_estimate=n * rr * aov))
    out = pd.DataFrame(rows).sort_values("projected_uplift_estimate", ascending=False)
    return out.reset_index(drop=True)


def run(sales: pd.DataFrame, segments: pd.DataFrame | None = None) -> ModuleResult:
    res = ModuleResult("SalesIQ")
    mt = monthly_trend(sales)
    res.tables["monthly"] = mt
    sales = sales.copy()
    sales["dow"] = sales["date"].dt.day_name()
    dow = sales.groupby("dow")["revenue"].sum()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = dow.reindex(order)
    res.tables["day_of_week"] = pd.DataFrame({"day": order, "revenue": dow.values,
                                              "index": dow.values / dow.mean()})

    def agg(col):
        a = sales.groupby(col).agg(revenue=("revenue", "sum"), units=("quantity", "sum"),
                                   orders=("order_id", "nunique")).sort_values("revenue", ascending=False)
        a["share"] = a["revenue"] / a["revenue"].sum()
        return a.reset_index()
    prod, reg, cat = agg("product"), agg("region"), agg("category")
    res.tables.update(products=prod, regions=reg, categories=cat)
    res.tables["top_products"] = prod.head(5)
    res.tables["bottom_products"] = prod.tail(5).sort_values("revenue")
    res.tables["basket_pairs"] = basket_pairs(sales)

    # Recent performance vs baseline (sales-drop detection)
    last = sales["date"].max()
    recent = sales[sales["date"] > last - pd.Timedelta(days=28)]["revenue"].sum() / 28
    base_df = sales[(sales["date"] <= last - pd.Timedelta(days=28)) &
                    (sales["date"] > last - pd.Timedelta(days=28 + 84))]
    baseline = base_df["revenue"].sum() / 84 if len(base_df) else float("nan")
    drop = safe_div(recent - baseline, baseline)
    w = sales.set_index("date")["revenue"].resample("D").sum()
    wk = w.tail(14)
    last7, prev7 = float(wk.tail(7).sum()), float(wk.head(7).sum())
    wow = safe_div(last7 - prev7, prev7)

    total = float(sales["revenue"].sum())
    last_m, prev_m = mt.iloc[-1], mt.iloc[-2] if len(mt) > 1 else mt.iloc[-1]
    res.metrics = dict(total_revenue=total, orders=int(sales["order_id"].nunique()),
                       aov=float(sales.groupby("order_id")["revenue"].sum().mean()),
                       last_month_revenue=float(last_m["revenue"]), last_month_growth=float(last_m["mom_growth"])
                       if pd.notna(last_m["mom_growth"]) else float("nan"),
                       recent_vs_baseline=drop, week_over_week=wow)
    pairs = res.tables["basket_pairs"]
    top_pair = pairs.iloc[0] if len(pairs) else None
    res.facts = dict(
        total_revenue=inr(total), orders=res.metrics["orders"], avg_order_value=inr(res.metrics["aov"], False),
        last_month=str(last_m["date"])[:7], last_month_revenue=inr(float(last_m["revenue"])),
        last_month_growth_pct=None if pd.isna(last_m["mom_growth"]) else round(float(last_m["mom_growth"]) * 100, 1),
        last_28d_vs_prior_12w_pct=None if pd.isna(drop) else round(drop * 100, 1),
        last_7d_vs_prev_7d_pct=None if pd.isna(wow) else round(wow * 100, 1),
        top_product=prod.iloc[0]["product"], top_product_share_pct=round(prod.iloc[0]["share"] * 100, 1),
        bottom_product=prod.iloc[-1]["product"], bottom_product_share_pct=round(prod.iloc[-1]["share"] * 100, 1),
        top_region=reg.iloc[0]["region"], top_region_share_pct=round(reg.iloc[0]["share"] * 100, 1),
        weakest_region=reg.iloc[-1]["region"], weakest_region_share_pct=round(reg.iloc[-1]["share"] * 100, 1),
        top_category=cat.iloc[0]["category"], top_category_share_pct=round(cat.iloc[0]["share"] * 100, 1),
        best_day=dow.idxmax(), peak_month=str(mt.loc[mt["revenue"].idxmax(), "date"])[:7],
        slowest_month=str(mt.loc[mt["revenue"].idxmin(), "date"])[:7],
        top_pair=None if top_pair is None else f"{top_pair['product_a']} + {top_pair['product_b']}",
        top_pair_lift=None if top_pair is None else round(float(top_pair["lift"]), 1),
    )
    if pd.notna(drop) and drop < -0.20:
        res.add_flag(RED, f"Sales are {abs(drop):.0%} below the recent baseline",
                     f"Last 28 days average {inr(recent, False)} a day versus {inr(baseline, False)} before.")
    elif pd.notna(drop) and drop < -0.10:
        res.add_flag(AMBER, f"Sales are {abs(drop):.0%} below the recent baseline",
                     f"Last 28 days average {inr(recent, False)} a day versus {inr(baseline, False)} before.")
    if pd.notna(wow) and wow < -0.25:
        res.add_flag(AMBER, f"Last 7 days revenue fell {abs(wow):.0%} week over week",
                     f"{inr(last7)} versus {inr(prev7)} the week before.")
    if prod.iloc[0]["share"] > 0.30:
        res.add_flag(AMBER, f"{prod.iloc[0]['product']} makes up {prod.iloc[0]['share']:.0%} of revenue",
                     "Heavy reliance on one product.")
    if len(reg) > 1 and reg.iloc[-1]["share"] < 0.10:
        res.add_flag(AMBER, f"{reg.iloc[-1]['region']} region contributes only {reg.iloc[-1]['share']:.0%}",
                     "Underperforming region; review distribution and marketing.")
    if not res.flags:
        res.add_flag(GREEN, "Sales are tracking in line with the recent baseline")

    camp = campaign_recommendations(segments)
    res.tables["campaigns"] = camp
    if len(camp):
        res.facts["best_campaign_segment"] = camp.iloc[0]["segment"]
        res.facts["best_campaign_uplift_estimate"] = inr(float(camp.iloc[0]["projected_uplift_estimate"]))
    res.notes.append("Projected campaign uplift is an ESTIMATE: customers x assumed response rate x "
                     "average order value. Response rates are assumptions, not measured results.")
    return res


def weekly_digest(res: ModuleResult) -> str:
    f = res.facts
    parts = [f"Sales digest: {f['orders']} orders and {f['total_revenue']} revenue over the period. "
             f"{f['last_month']} revenue was {f['last_month_revenue']}"]
    if f.get("last_month_growth_pct") is not None:
        parts[0] += f" ({f['last_month_growth_pct']:+.1f}% month over month)"
    parts[0] += "."
    if f.get("last_7d_vs_prev_7d_pct") is not None:
        parts.append(f"The last 7 days moved {f['last_7d_vs_prev_7d_pct']:+.1f}% against the previous 7.")
    parts.append(f"{f['top_product']} leads at {f['top_product_share_pct']}% of revenue; "
                 f"{f['bottom_product']} trails at {f['bottom_product_share_pct']}%. "
                 f"{f['top_region']} is the strongest region ({f['top_region_share_pct']}%).")
    if f.get("top_pair"):
        parts.append(f"{f['top_pair']} is the strongest bundle (lift {f['top_pair_lift']}).")
    return " ".join(parts)
