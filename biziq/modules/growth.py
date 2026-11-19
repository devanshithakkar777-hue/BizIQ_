"""Customer and channel modules: ChurnIQ, PersonalisationIQ, WhatsAppIQ, MarketplaceIQ, StoreIQ, DistributorIQ."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..core import AMBER, GREEN, RED, ModuleResult, inr, safe_div
from ..registry import Context, ModuleSpec, register
from ._common import argext, clean_facts, pct1, r1


def _rev(s: pd.DataFrame) -> pd.Series:
    return s["revenue"] if "revenue" in s and s["revenue"].notna().any() else s["quantity"] * s["unit_price"]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df else pd.Series(0.0, index=df.index)


# ============================================================================== ChurnIQ
def run_churn(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ChurnIQ")
    s = data["sales"].copy()
    s["revenue_"] = _rev(s)
    as_of = ctx.as_of
    s = s[s["date"] <= as_of]
    cust = s.groupby("customer_id").agg(first=("date", "min"), last=("date", "max"), revenue_12m=("revenue_", lambda x: 0.0))
    last12 = s[s["date"] > as_of - pd.Timedelta(days=365)].groupby("customer_id")["revenue_"].sum()
    cust["revenue_12m"] = last12.reindex(cust.index).fillna(0.0)
    days = s.drop_duplicates(["customer_id", "date"]).sort_values(["customer_id", "date"])
    days["gap"] = days.groupby("customer_id")["date"].diff().dt.days
    cust["orders"] = days.groupby("customer_id").size()
    cust["typical_gap_days"] = days.groupby("customer_id")["gap"].median()
    cust["days_since_last"] = (as_of - cust["last"]).dt.days
    min_orders = ctx.settings.get("churn_min_orders", 4)
    cust["gap_ratio"] = cust["days_since_last"] / cust["typical_gap_days"].clip(lower=7)
    repeat = cust[cust["orders"] >= min_orders].copy()
    repeat["status"] = np.select([(repeat["gap_ratio"] >= 4) & (repeat["days_since_last"] >= 60), (repeat["gap_ratio"] >= 2.5) & (repeat["days_since_last"] >= 45)], ["lapsed", "at risk"], "active")
    res.tables["customers_to_win_back"] = repeat[repeat["status"] != "active"].reset_index().sort_values("revenue_12m", ascending=False)[
        ["customer_id", "status", "orders", "typical_gap_days", "days_since_last", "gap_ratio", "revenue_12m"]].head(200)
    total12 = float(cust["revenue_12m"].sum())
    lapsed, risky = repeat[repeat["status"] == "lapsed"], repeat[repeat["status"] == "at risk"]
    rev_risk = float(lapsed["revenue_12m"].sum() + risky["revenue_12m"].sum())
    one_time = int((cust["orders"] == 1).sum())
    res.metrics = dict(at_risk_customers=len(lapsed) + len(risky))
    top = res.tables["customers_to_win_back"].head(1)
    res.facts = clean_facts(dict(
        customers=len(cust), repeat_customers_analysed=len(repeat), one_time_customers=one_time, one_time_pct=pct1(safe_div(one_time, len(cust))),
        at_risk_customers=len(risky), lapsed_customers=len(lapsed), past_12m_revenue_from_at_risk_and_lapsed=inr(rev_risk),
        share_of_12m_revenue_pct=pct1(safe_div(rev_risk, total12)),
        top_customer_to_win_back=str(top.iloc[0]["customer_id"]) if len(top) else None,
        top_customer_days_since_last=int(top.iloc[0]["days_since_last"]) if len(top) else None,
        top_customer_typical_gap_days=r1(float(top.iloc[0]["typical_gap_days"]), 0) if len(top) else None))
    share = safe_div(rev_risk, total12)
    if pd.notna(share) and share > 0.35:
        res.add_flag(RED, f"{len(lapsed) + len(risky)} repeat customers who bought {inr(rev_risk)} in the last year have gone quiet")
    elif pd.notna(share) and share > 0.10:
        res.add_flag(AMBER, f"{len(lapsed) + len(risky)} repeat customers ({share:.0%} of last year's revenue) are overdue")
    if len(cust) and one_time / len(cust) > 0.6:
        res.add_flag(AMBER, f"{one_time / len(cust):.0%} of customers have bought only once")
    if not res.flags:
        res.add_flag(GREEN, "Repeat customers are buying on their usual rhythm")
    res.notes.append("This compares each customer's silence with THEIR OWN usual gap between orders (at risk = 2.5x and 45+ days, lapsed = 4x and 60+ days, only customers with "
                     f"{min_orders}+ order days). It is a rule, not a trained prediction model: it cannot tell a churned customer from a seasonal buyer. "
                     "Test win-back offers on a small group first. Overlaps LoyaltyIQ on repeat buying, but this one is about who to contact.")
    return res


def rec_churn(f, flags):
    r = []
    if f.get("lapsed_customers") or f.get("at_risk_customers"):
        r.append(f"Contact the {f.get('lapsed_customers', 0)} lapsed and {f.get('at_risk_customers', 0)} at-risk repeat customers, biggest spenders first; they bought {f.get('past_12m_revenue_from_at_risk_and_lapsed')} in the last year.")
        r.append("Open by acknowledging the gap, not by discounting; try one offer on half the list and compare with the other half.")
    if f.get("top_customer_to_win_back"):
        r.append(f"Start with {f['top_customer_to_win_back']}: {f.get('top_customer_days_since_last')} days since the last order against a usual gap of {f.get('top_customer_typical_gap_days')} days.")
    return r or ["Repeat customers are on rhythm; keep watching the one-time buyer share."]


# ============================================================================== PersonalisationIQ
def run_personalisation(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("PersonalisationIQ")
    s = data["sales"][["order_id", "customer_id", "product", "date"]].dropna().copy()
    multi_orders = s.groupby("order_id")["product"].nunique()
    share_multi = safe_div(int((multi_orders > 1).sum()), len(multi_orders))
    basis = "order"
    if not (pd.notna(share_multi) and share_multi >= 0.05):          # single-item orders: fall back to what one customer buys within 60 days
        basis = "customer in 60 days"
        s["key"] = s["customer_id"].astype(str) + "|" + (s["date"] - s["date"].min()).dt.days.floordiv(60).astype(str)
    else:
        s["key"] = s["order_id"]
    baskets = s.groupby("key")["product"].apply(lambda x: sorted(set(x)))
    baskets = baskets[baskets.map(len) > 1]
    n_b = len(s["key"].unique())
    min_n = ctx.settings.get("affinity_min_baskets", 8)
    single = s.groupby("key")["product"].apply(lambda x: set(x))
    item_n = pd.Series([p for b in single for p in b]).value_counts()
    pair_n: dict = {}
    for b in baskets:
        for a, c in combinations(b, 2):
            pair_n[(a, c)] = pair_n.get((a, c), 0) + 1
    rows = []
    for (a, c), n in pair_n.items():
        if n < min_n:
            continue
        lift = (n / n_b) / ((item_n[a] / n_b) * (item_n[c] / n_b))
        rows.append(dict(product_a=a, product_b=c, baskets_together=n, lift=round(lift, 2),
                         chance_b_given_a_pct=round(100 * n / item_n[a], 1), chance_a_given_b_pct=round(100 * n / item_n[c], 1)))
    pairs = pd.DataFrame(rows, columns=["product_a", "product_b", "baskets_together", "lift", "chance_b_given_a_pct", "chance_a_given_b_pct"]).sort_values("lift", ascending=False)
    res.tables["product_pairs"] = pairs.head(30)
    # who bought A but never B, for the three strongest pairs: the offer lists
    owned = s.groupby("customer_id")["product"].apply(set)
    targets = []
    for r in pairs.head(3).itertuples(index=False):
        for who, have, want in ((owned.index[owned.map(lambda x: r.product_a in x and r.product_b not in x)], r.product_a, r.product_b),
                                (owned.index[owned.map(lambda x: r.product_b in x and r.product_a not in x)], r.product_b, r.product_a)):
            targets.append(dict(bought=have, suggest=want, customers=len(who)))
    res.tables["offer_audiences"] = pd.DataFrame(targets, columns=["bought", "suggest", "customers"])
    best = pairs.iloc[0] if len(pairs) else None
    res.metrics = dict(pairs_found=len(pairs))
    res.facts = clean_facts(dict(
        baskets_analysed=int(n_b), multi_item_baskets_pct=pct1(safe_div(len(baskets), n_b)), basket_basis=basis, pairs_found=len(pairs),
        strongest_pair=f"{best['product_a']} + {best['product_b']}" if best is not None else None,
        strongest_pair_lift=float(best["lift"]) if best is not None else None,
        strongest_pair_baskets=int(best["baskets_together"]) if best is not None else None,
        customers_to_target_top_pair=int(res.tables["offer_audiences"].iloc[:2]["customers"].sum()) if len(res.tables["offer_audiences"]) else None))
    if not len(pairs):
        res.add_flag(AMBER, "No product pair shows up often enough to base an offer on")
    elif best["lift"] >= 1.5:
        res.add_flag(GREEN, f"{best['product_a']} and {best['product_b']} are bought together {best['lift']:.1f}x more often than chance")
    else:
        res.add_flag(GREEN, "Products are bought roughly independently; bundles will not be a big lever")
    res.notes.append(f"Basket = {basis}. Lift above 1 means two products appear together more than their individual popularity predicts; it shows association, "
                     f"not that showing one causes the other to sell. Pairs need at least {min_n} baskets to count. Check a bundle with a small test before rolling it out.")
    return res


def rec_personalisation(f, flags):
    if not f.get("strongest_pair"):
        return ["No reliable product pairs yet; revisit after more orders."]
    return [f"Test a bundle or 'bought together' prompt for {f['strongest_pair']} (lift {f.get('strongest_pair_lift')}, seen together in {f.get('strongest_pair_baskets')} baskets).",
            f"About {f.get('customers_to_target_top_pair', 0)} existing customers bought one half of the top pair but not the other; message them with the missing one.",
            "Run it on half the audience and compare repeat purchase before expanding."]


# ============================================================================== WhatsAppIQ
def run_whatsapp(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("WhatsAppIQ")
    m = data["whatsapp_messages"].copy()
    for c in ("sent", "delivered", "read", "replied", "opted_out"):
        m[c] = _num(m, c)
    g = m.groupby("campaign")[["sent", "delivered", "read", "replied", "opted_out"]].sum()
    g["delivery_rate_pct"] = 100 * g["delivered"] / g["sent"].replace(0, np.nan)
    g["read_rate_pct"] = 100 * g["read"] / g["delivered"].replace(0, np.nan)
    g["reply_rate_pct"] = 100 * g["replied"] / g["delivered"].replace(0, np.nan)
    g["opt_out_rate_pct"] = 100 * g["opted_out"] / g["delivered"].replace(0, np.nan)
    res.tables["campaigns"] = g.round(1).reset_index().sort_values("reply_rate_pct", ascending=False)
    tot = g.sum()
    d_rate, o_rate = safe_div(tot["delivered"], tot["sent"]), safe_div(tot["opted_out"], tot["delivered"])
    big = g[g["delivered"] >= 200].dropna(subset=["reply_rate_pct"])
    best = big["reply_rate_pct"].idxmax() if len(big) else None
    worst_o = big["opt_out_rate_pct"].idxmax() if len(big) else None
    recent = m[m["date"] > ctx.as_of - pd.Timedelta(days=30)]
    prior = m[(m["date"] <= ctx.as_of - pd.Timedelta(days=30)) & (m["date"] > ctx.as_of - pd.Timedelta(days=60))]
    o_recent, o_prior = safe_div(recent["opted_out"].sum(), recent["delivered"].sum()), safe_div(prior["opted_out"].sum(), prior["delivered"].sum())
    res.facts = clean_facts(dict(
        messages_sent=int(tot["sent"]), delivery_rate_pct=pct1(d_rate), read_rate_pct=pct1(safe_div(tot["read"], tot["delivered"])),
        reply_rate_pct=pct1(safe_div(tot["replied"], tot["delivered"])), opt_out_rate_pct=pct1(o_rate), campaigns=len(g),
        best_campaign=str(best) if best else None, best_campaign_reply_rate_pct=r1(big.loc[best, "reply_rate_pct"]) if best else None,
        highest_opt_out_campaign=str(worst_o) if worst_o else None, highest_opt_out_rate_pct=r1(big.loc[worst_o, "opt_out_rate_pct"], 2) if worst_o else None,
        opt_out_rate_last_30d_pct=pct1(o_recent), opt_out_rate_prior_30d_pct=pct1(o_prior)))
    if pd.notna(o_rate) and o_rate >= 0.02:
        res.add_flag(RED, f"{o_rate:.1%} of people who received your messages opted out")
    elif pd.notna(o_rate) and o_rate >= 0.01:
        res.add_flag(AMBER, f"Opt-outs are {o_rate:.1%} of delivered messages")
    if pd.notna(d_rate) and d_rate < 0.90:
        res.add_flag(AMBER, f"Only {d_rate:.0%} of messages are delivered; check the number list and opt-in records")
    if pd.notna(o_recent) and pd.notna(o_prior) and o_prior > 0 and o_recent > 1.5 * o_prior:
        res.add_flag(AMBER, "Opt-outs have risen sharply in the last 30 days")
    if not res.flags:
        res.add_flag(GREEN, "Delivery and opt-outs are within normal ranges")
    res.notes.append("Thresholds (delivery below 90%, opt-outs 1% / 2%) are rules of thumb, not WhatsApp's own limits. Meta's quality rating and template status are not "
                     "in this data. Only message people who opted in.")
    return res


def rec_whatsapp(f, flags):
    r = []
    if f.get("best_campaign"):
        r.append(f"'{f['best_campaign']}' gets the most replies ({f.get('best_campaign_reply_rate_pct')}% of delivered); reuse its format and timing.")
    if f.get("highest_opt_out_campaign") and (f.get("highest_opt_out_rate_pct") or 0) >= 1:
        r.append(f"'{f['highest_opt_out_campaign']}' loses {f['highest_opt_out_rate_pct']}% of recipients to opt-outs; send it less often or to a narrower audience.")
    if (f.get("delivery_rate_pct") or 100) < 90:
        r.append("Clean the contact list: low delivery usually means wrong numbers or people who never opted in.")
    return r or ["Keep sending only to opted-in contacts and watch the opt-out rate each month."]


# ============================================================================== MarketplaceIQ
def run_marketplace(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("MarketplaceIQ")
    o = data["marketplace_orders"].copy()
    for c in ("shipping_fee", "other_fees", "returned"):
        o[c] = _num(o, c)
    o["fees"] = o["commission"] + o["shipping_fee"] + o["other_fees"]
    o["expected_payout"] = o["gross_amount"] - o["fees"]
    has_settle = "settlement_amount" in o
    ch = o.groupby("channel").agg(orders=("order_id", "size"), gross=("gross_amount", "sum"), fees=("fees", "sum"), returned=("returned", "sum"))
    ch["fee_pct"] = 100 * ch["fees"] / ch["gross"].replace(0, np.nan)
    ch["return_pct"] = 100 * ch["returned"] / ch["orders"]
    refunded = o[o["returned"] > 0].groupby("channel")["gross_amount"].sum()
    ch["kept_after_fees_and_returns"] = ch["gross"] - ch["fees"] - refunded.reindex(ch.index).fillna(0)
    ch["kept_pct"] = 100 * ch["kept_after_fees_and_returns"] / ch["gross"].replace(0, np.nan)
    res.tables["channels"] = ch.round(1).reset_index().sort_values("kept_pct")
    short_amt, short_n, overdue_amt, overdue_n = 0.0, 0, 0.0, 0
    if has_settle:
        settled = o[o["settlement_amount"].notna()].copy()
        settled["shortfall"] = settled["expected_payout"] - settled["settlement_amount"]
        bad = settled[(settled["shortfall"] > 1) & (settled["shortfall"] > 0.01 * settled["gross_amount"])]
        short_amt, short_n = float(bad["shortfall"].sum()), len(bad)
        res.tables["short_paid_orders"] = bad.sort_values("shortfall", ascending=False)[["order_id", "date", "channel", "gross_amount", "expected_payout", "settlement_amount", "shortfall"]].head(200)
        wait = ctx.settings.get("settlement_wait_days", 15)
        pending = o[o["settlement_amount"].isna() & (o["returned"] == 0) & (o["date"] < ctx.as_of - pd.Timedelta(days=wait))]
        overdue_amt, overdue_n = float(pending["expected_payout"].sum()), len(pending)
        res.tables["unsettled_orders"] = pending.sort_values("expected_payout", ascending=False)[["order_id", "date", "channel", "expected_payout"]].head(200)
    worst = res.tables["channels"].iloc[0] if len(ch) else None
    costliest = argext(ch["fee_pct"])
    res.facts = clean_facts(dict(
        orders=len(o), channels=len(ch), gross_sales=inr(float(o["gross_amount"].sum())), total_fees=inr(float(o["fees"].sum())),
        effective_fee_pct=pct1(safe_div(float(o["fees"].sum()), float(o["gross_amount"].sum()))),
        highest_fee_channel=str(costliest) if costliest else None, highest_fee_pct=r1(float(ch.loc[costliest, "fee_pct"])) if costliest else None,
        lowest_kept_channel=str(worst["channel"]) if worst is not None and pd.notna(worst["kept_pct"]) else None, lowest_kept_pct=r1(float(worst["kept_pct"])) if worst is not None and pd.notna(worst["kept_pct"]) else None,
        short_paid_orders=short_n if has_settle else None, short_paid_amount=inr(short_amt) if has_settle else None,
        unsettled_orders_past_wait=overdue_n if has_settle else None, unsettled_amount=inr(overdue_amt) if has_settle else None))
    gross_total = float(o["gross_amount"].sum())
    if has_settle and short_amt > 0.005 * gross_total:
        res.add_flag(RED, f"Marketplaces paid {inr(short_amt)} less than the fee terms imply on {short_n} orders")
    elif has_settle and short_n:
        res.add_flag(AMBER, f"{short_n} orders were paid out less than expected ({inr(short_amt)})")
    if has_settle and overdue_n and overdue_amt > 0.01 * gross_total:
        res.add_flag(AMBER, f"{inr(overdue_amt)} across {overdue_n} orders is still unpaid past the usual settlement wait")
    if worst is not None and pd.notna(worst["kept_pct"]) and worst["kept_pct"] < 70:
        res.add_flag(AMBER, f"On {worst['channel']} you keep only {worst['kept_pct']:.0f}% of sales after fees and returns")
    if not res.flags:
        res.add_flag(GREEN, "Marketplace payouts match the fees charged")
    res.notes.append("Expected payout = gross - commission - shipping fee - other fees, as recorded in YOUR data; where the real fee rules differ (tiered, promotional or "
                     "TCS/TDS deductions) a gap may be legitimate, so raise disputes only after checking the seller statement. Returns are counted at the order's gross value.")
    return res


def rec_marketplace(f, flags):
    r = []
    if f.get("short_paid_orders"):
        r.append(f"Raise a payment dispute for {f['short_paid_orders']} short-paid orders worth {f.get('short_paid_amount')}; the list is in the short-paid table.")
    if f.get("unsettled_orders_past_wait"):
        r.append(f"Chase {f['unsettled_orders_past_wait']} orders ({f.get('unsettled_amount')}) that were never settled.")
    if f.get("lowest_kept_channel"):
        r.append(f"{f['lowest_kept_channel']} leaves you the least ({f.get('lowest_kept_pct')}% of sales after fees and returns); review pricing or product mix there.")
    return r or ["Payouts reconcile; keep checking each settlement report."]


# ============================================================================== StoreIQ
def run_store(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("StoreIQ")
    d = data["store_daily"].copy()
    d = d[d["date"] <= ctx.as_of]
    has_ff = "footfall" in d and d["footfall"].notna().any()
    agg = dict(sales=("sales", "sum"), transactions=("transactions", "sum"), days=("date", "nunique"))
    if has_ff:
        agg["footfall"] = ("footfall", "sum")
    for c in ("staff_hours", "area_sqft"):
        if c in d and d[c].notna().any():
            agg[c] = (c, "sum") if c == "staff_hours" else (c, "max")
    st = d.groupby("store").agg(**agg)
    st["avg_basket"] = st["sales"] / st["transactions"].replace(0, np.nan)
    st["sales_per_day"] = st["sales"] / st["days"]
    if has_ff:
        st["conversion_pct"] = 100 * st["transactions"] / st["footfall"].replace(0, np.nan)
    if "staff_hours" in st:
        st["sales_per_staff_hour"] = st["sales"] / st["staff_hours"].replace(0, np.nan)
    if "area_sqft" in st:
        st["sales_per_sqft_month"] = st["sales_per_day"] * 30 / st["area_sqft"].replace(0, np.nan)
    recent = d[d["date"] > ctx.as_of - pd.Timedelta(days=28)].groupby("store")["sales"].sum()
    prior = d[(d["date"] <= ctx.as_of - pd.Timedelta(days=28)) & (d["date"] > ctx.as_of - pd.Timedelta(days=56))].groupby("store")["sales"].sum()
    st["change_last_28d_pct"] = (100 * (recent / prior.replace(0, np.nan) - 1)).reindex(st.index)
    key = "sales_per_sqft_month" if "sales_per_sqft_month" in st else "sales_per_day"
    med = st[key].median()
    st["vs_median_pct"] = 100 * (st[key] / med - 1) if med else np.nan
    st["flag"] = np.where(st["vs_median_pct"] < -30, "well below peers", "")
    res.tables["stores"] = st.round(2).reset_index().sort_values(key, ascending=False)
    laggards = st[st["vs_median_pct"] < -30]
    falling = st[st["change_last_28d_pct"] < -15]
    best, worst = argext(st[key]), argext(st[key], "min")
    res.facts = clean_facts(dict(
        stores=len(st), total_sales=inr(float(st["sales"].sum())), comparison_measure=key.replace("_", " "),
        best_store=str(best) if best is not None else None, worst_store=str(worst) if worst is not None else None,
        worst_vs_median_pct=r1(float(st.loc[worst, "vs_median_pct"])) if worst is not None else None,
        stores_well_below_peers=len(laggards), stores_falling_over_15pct=len(falling),
        chain_conversion_pct=pct1(safe_div(float(st["transactions"].sum()), float(st["footfall"].sum()))) if has_ff else None,
        lowest_conversion_store=str(argext(st["conversion_pct"], "min")) if has_ff and st["conversion_pct"].notna().any() else None,
        lowest_conversion_pct=r1(float(st["conversion_pct"].min())) if has_ff and st["conversion_pct"].notna().any() else None))
    if len(laggards):
        res.add_flag(AMBER, f"{len(laggards)} store(s) sell over 30% less than the median store: {', '.join(map(str, laggards.index[:3]))}")
    if len(falling):
        res.add_flag(AMBER if len(falling) < 3 else RED, f"{len(falling)} store(s) fell more than 15% against the previous 28 days: {', '.join(map(str, falling.index[:3]))}")
    if not res.flags:
        res.add_flag(GREEN, "No store is far off its peers or falling sharply")
    res.notes.append(f"Stores are compared on {key.replace('_', ' ')}. Size, location and opening dates differ, so 'below peers' is a prompt to look, not proof of poor management. "
                     "Seasonal or festival timing can move the 28-day comparison.")
    return res


def rec_store(f, flags):
    r = []
    if f.get("stores_well_below_peers"):
        r.append(f"Visit or review {f['worst_store']} first: it is {abs(f.get('worst_vs_median_pct', 0))}% below the median store on {f.get('comparison_measure')}.")
    if f.get("lowest_conversion_store"):
        r.append(f"{f['lowest_conversion_store']} converts only {f.get('lowest_conversion_pct')}% of visitors; check staffing at peak hours and stock of fast sellers.")
    if f.get("stores_falling_over_15pct"):
        r.append(f"{f['stores_falling_over_15pct']} store(s) are falling; compare with local events, stock-outs or staff changes.")
    return r or [f"{f.get('best_store')} is the benchmark; share what it does with the others."]


# ============================================================================== DistributorIQ
def run_distributor(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("DistributorIQ")
    d = data["distributors"].copy()
    d = d[d["month"] <= ctx.as_of + pd.Timedelta(days=31)]
    months = sorted(d["month"].unique())
    window = months[-3:]
    w = d[d["month"].isin(window)]
    agg = dict(primary=("primary_sales", "sum"), secondary=("secondary_sales", "sum"))
    t = w.groupby("distributor").agg(**agg)
    t["sell_in_to_sell_out"] = t["primary"] / t["secondary"].replace(0, np.nan)
    if "closing_stock" in d:
        last = d[d["month"] == months[-1]].groupby("distributor")["closing_stock"].last()
        t["closing_stock"] = last.reindex(t.index)
        t["stock_months_cover"] = t["closing_stock"] / (t["secondary"] / len(window)).replace(0, np.nan)
    if "outstanding" in d:
        lo = d[d["month"] == months[-1]].groupby("distributor")["outstanding"].last()
        t["outstanding"] = lo.reindex(t.index)
        t["days_of_sales_outstanding"] = 30 * t["outstanding"] / (t["primary"] / len(window)).replace(0, np.nan)
    if "credit_days" in d:
        t["credit_days"] = d.groupby("distributor")["credit_days"].last().reindex(t.index)
    t["status"] = np.select([t["sell_in_to_sell_out"] > 1.5, t["sell_in_to_sell_out"] > 1.25], ["stock building up", "watch"], "ok")
    res.tables["distributors"] = t.round(2).reset_index().sort_values("sell_in_to_sell_out", ascending=False)
    stuffed, watch = t[t["sell_in_to_sell_out"] > 1.5], t[(t["sell_in_to_sell_out"] > 1.25) & (t["sell_in_to_sell_out"] <= 1.5)]
    over = t[t["days_of_sales_outstanding"] > t["credit_days"] * 1.5] if {"days_of_sales_outstanding", "credit_days"} <= set(t.columns) else t.iloc[0:0]
    worst = argext(t["sell_in_to_sell_out"])
    res.facts = clean_facts(dict(
        distributors=len(t), months_analysed=len(window), sell_in_to_sell_out_overall=r1(safe_div(float(t["primary"].sum()), float(t["secondary"].sum())), 2),
        distributors_building_stock=len(stuffed), distributors_to_watch=len(watch), distributors_beyond_credit_terms=len(over),
        worst_distributor=str(worst) if worst else None, worst_sell_in_to_sell_out=r1(float(t.loc[worst, "sell_in_to_sell_out"]), 2) if worst else None,
        worst_stock_months_cover=r1(float(t.loc[worst, "stock_months_cover"])) if worst and "stock_months_cover" in t and pd.notna(t.loc[worst, "stock_months_cover"]) else None,
        unsold_stock_pushed_value=inr(float((stuffed["primary"] - stuffed["secondary"]).sum())) if len(stuffed) else None))
    if len(stuffed):
        res.add_flag(RED if len(stuffed) >= 2 else AMBER, f"{len(stuffed)} distributor(s) are being sold much more than they sell on: {', '.join(map(str, stuffed.index[:3]))}")
    if len(over):
        res.add_flag(AMBER, f"{len(over)} distributor(s) are paying well beyond their credit terms")
    if not res.flags:
        res.add_flag(GREEN, "Sell-in tracks sell-out across distributors")
    res.notes.append("Sell-in / sell-out above 1 over three months means stock is accumulating in the channel: it can precede returns, discounting or stalled reorders. "
                     "Thresholds (1.25 watch, 1.5 high) are rules of thumb; seasonal pre-stocking is normal. Primary sales that overstate real demand also inflate your revenue.")
    return res


def rec_distributor(f, flags):
    r = []
    if f.get("distributors_building_stock"):
        r.append(f"Pause pushing extra stock to {f['worst_distributor']} (sell-in is {f.get('worst_sell_in_to_sell_out')}x its sell-out); agree a sell-out target before the next dispatch.")
    if f.get("distributors_beyond_credit_terms"):
        r.append(f"Chase {f['distributors_beyond_credit_terms']} distributor(s) paying beyond terms before shipping more.")
    r.append("Ask for stock and sell-out reports monthly; sell-in alone hides what the channel actually sells.")
    return r


register(ModuleSpec("ChurnIQ", "Customers and channels", "Repeat customers who have gone quiet compared with their own rhythm, and who to win back first.",
                    ["sales"], run_churn, rec_churn, audience="Both", phase="3"))
register(ModuleSpec("PersonalisationIQ", "Customers and channels", "Products bought together, and who to offer the missing half.",
                    ["sales"], run_personalisation, rec_personalisation, audience="B2C", phase="3"))
register(ModuleSpec("WhatsAppIQ", "Customers and channels", "WhatsApp campaign delivery, replies and opt-outs.",
                    ["whatsapp_messages"], run_whatsapp, rec_whatsapp, audience="B2C", phase="3"))
register(ModuleSpec("MarketplaceIQ", "Customers and channels", "Marketplace fees, short payments and unsettled orders by channel.",
                    ["marketplace_orders"], run_marketplace, rec_marketplace, audience="B2C", phase="3"))
register(ModuleSpec("StoreIQ", "Customers and channels", "Store-by-store sales, conversion and who is falling behind.",
                    ["store_daily"], run_store, rec_store, audience="B2C", phase="3"))
register(ModuleSpec("DistributorIQ", "Customers and channels", "Sell-in versus sell-out, channel stock and distributor payments.",
                    ["distributors"], run_distributor, rec_distributor, audience="B2B", phase="3"))
