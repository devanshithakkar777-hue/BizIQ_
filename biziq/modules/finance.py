"""Finance and compliance modules: ProfitIQ, ReceivablesIQ, CreditIQ, ComplianceIQ, ExpenseIQ."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..core import AMBER, GREEN, RED, ModuleResult, inr, safe_div
from ..registry import Context, ModuleSpec, register
from ._common import clean_facts, open_mask, pct1, r1

# ============================================================================== ProfitIQ
RETURN_RECOVERY = 0.70        # assumption: share of a returned item's cost recovered by reselling it


def run_profit(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ProfitIQ")
    costs = data["product_costs"].set_index("product")
    s = data["sales"].merge(costs, left_on="product", right_index=True, how="left")
    unmatched = int(s["unit_cost"].isna().sum())
    if unmatched == len(s):
        raise ValueError("needs a cost in product_costs for the products sold")
    s = s[s["unit_cost"].notna()].copy()
    for col in ("channel_fee_pct", "shipping_cost_per_order", "packaging_cost"):
        s[col] = s[col].fillna(0.0) if col in s else 0.0
    n_lines = s.groupby("order_id")["order_id"].transform("size")
    s["cogs"] = s["quantity"] * s["unit_cost"]
    s["fees"] = s["revenue"] * s["channel_fee_pct"] / 100
    s["ship"] = (s["shipping_cost_per_order"] + s["packaging_cost"]) / n_lines
    s["profit_before_returns"] = s["revenue"] - s["cogs"] - s["fees"] - s["ship"]

    ret = data.get("returns")
    s["refund"] = 0.0
    if ret is not None and len(ret):
        refunds = ret.groupby(["order_id", "product"])["refund_amount"].sum().rename("refund_in")
        s = s.merge(refunds, on=["order_id", "product"], how="left")
        s["refund"] = s["refund_in"].fillna(0.0).clip(upper=s["revenue"])
    share = (s["refund"] / s["revenue"].replace(0, np.nan)).fillna(0.0)
    s["recovered"] = s["cogs"] * RETURN_RECOVERY * share
    s["reverse_ship"] = np.where(s["refund"] > 0, s["shipping_cost_per_order"], 0.0)
    s["net_revenue"] = s["revenue"] - s["refund"]
    s["net_profit"] = s["net_revenue"] - (s["cogs"] - s["recovered"]) - s["fees"] - s["ship"] - s["reverse_ship"]

    def summarise(g):
        return pd.Series(dict(
            net_revenue=g["net_revenue"].sum(), units=g["quantity"].sum(), refunds=g["refund"].sum(),
            cogs=(g["cogs"] - g["recovered"]).sum(), fees=g["fees"].sum(), shipping=(g["ship"] + g["reverse_ship"]).sum(),
            net_profit=g["net_profit"].sum()))

    by_prod = s.groupby("product").apply(summarise, include_groups=False)
    by_prod["net_margin"] = by_prod["net_profit"] / by_prod["net_revenue"].replace(0, np.nan)
    by_prod["profit_per_unit"] = by_prod["net_profit"] / by_prod["units"].replace(0, np.nan)
    by_prod = by_prod.sort_values("net_margin")
    res.tables["by_product"] = by_prod.reset_index()
    by_region = s.groupby("region").apply(summarise, include_groups=False)
    by_region["net_margin"] = by_region["net_profit"] / by_region["net_revenue"].replace(0, np.nan)
    res.tables["by_region"] = by_region.reset_index().sort_values("net_margin")
    s["month"] = s["date"].dt.to_period("M").dt.to_timestamp()
    by_month = s.groupby("month").apply(summarise, include_groups=False)
    by_month["net_margin"] = by_month["net_profit"] / by_month["net_revenue"].replace(0, np.nan)
    res.tables["by_month"] = by_month.reset_index()
    cust = s.groupby("customer_id").agg(orders=("order_id", "nunique"), net_revenue=("net_revenue", "sum"),
                                        net_profit=("net_profit", "sum"))
    cust["net_margin"] = cust["net_profit"] / cust["net_revenue"].replace(0, np.nan)
    res.tables["least_profitable_customers"] = cust[cust["orders"] >= 2].sort_values("net_profit").head(20).reset_index()

    tot_rev, tot_profit = float(s["net_revenue"].sum()), float(s["net_profit"].sum())
    margin = safe_div(tot_profit, tot_rev)
    drag = float(s["profit_before_returns"].sum() - s["net_profit"].sum())
    loss = by_prod[by_prod["net_profit"] < 0]
    thin = by_prod[(by_prod["net_margin"] >= 0) & (by_prod["net_margin"] < 0.10)]
    worst, best = by_prod.iloc[0], by_prod.iloc[-1]
    res.metrics = dict(net_revenue=tot_rev, net_profit=tot_profit, net_margin=margin, profit_lost_to_returns=drag,
                       loss_making_products=len(loss), thin_margin_products=len(thin))
    res.facts = clean_facts(dict(
        net_revenue=inr(tot_rev), net_profit=inr(tot_profit), net_margin_pct=pct1(margin),
        profit_lost_to_returns=inr(drag), profit_lost_to_returns_pct_of_revenue=pct1(safe_div(drag, tot_rev)),
        loss_making_products=len(loss), thin_margin_products=len(thin),
        lowest_margin_product=str(by_prod.index[0]), lowest_margin_pct=pct1(worst["net_margin"]),
        highest_margin_product=str(by_prod.index[-1]), highest_margin_pct=pct1(best["net_margin"]),
        lowest_margin_profit_per_unit=inr(worst["profit_per_unit"], compact=False),
        lowest_margin_product_revenue=inr(worst["net_revenue"]),
        unprofitable_customers=int((cust["net_profit"] < 0).sum()),
        products_without_cost_data=unmatched))
    for name, r in by_prod.iterrows():
        if r["net_profit"] < 0:
            res.add_flag(RED, f"{name} loses money after costs and returns ({r['net_margin']:.1%} margin)",
                         f"Net profit {inr(r['net_profit'])} on {inr(r['net_revenue'])} of revenue.")
        elif r["net_margin"] < 0.10:
            res.add_flag(AMBER, f"{name} has a thin net margin of {r['net_margin']:.1%}",
                         "Review price, supplier cost, channel fees or returns.")
    if safe_div(drag, tot_rev) > 0.04:
        res.add_flag(AMBER, f"Returns cost {inr(drag)} of profit ({safe_div(drag, tot_rev):.1%} of revenue)")
    if unmatched:
        res.add_flag(AMBER, f"{unmatched} sales line(s) have no product cost and were left out",
                     "Add the missing products to product_costs.")
    if not res.flags:
        res.add_flag(GREEN, "All products are profitable after costs and returns")
    res.notes.append(f"Net profit = revenue - refunds - cost of goods (net of {RETURN_RECOVERY:.0%} recovery on returned goods, an "
                     "assumption) - marketplace fees (not refunded) - shipping and packaging - reverse shipping on returns. "
                     "Overheads (rent, salaries, marketing) are not allocated here.")
    return res


def rec_profit(f, flags):
    r = []
    if f.get("loss_making_products"):
        r.append(f"{f['loss_making_products']} product(s) lose money after costs and returns; {f['lowest_margin_product']} is the "
                 f"weakest at {f['lowest_margin_pct']}% net margin ({f['lowest_margin_profit_per_unit']} per unit). "
                 "Raise the price, renegotiate its cost, cut its marketplace fee or fix its return rate before pushing more volume.")
    elif f.get("thin_margin_products"):
        r.append(f"{f['thin_margin_products']} product(s) earn under 10% net margin. Check their price against competitors.")
    if f.get("profit_lost_to_returns_pct_of_revenue", 0) and f["profit_lost_to_returns_pct_of_revenue"] > 3:
        r.append(f"Returns take {f['profit_lost_to_returns']} of profit ({f['profit_lost_to_returns_pct_of_revenue']}% of revenue). "
                 "Use ReturnsIQ to find the products, channels and reasons behind them.")
    if f.get("unprofitable_customers"):
        r.append(f"{f['unprofitable_customers']} customers cost more to serve than they bring in; review discounts and return behaviour for them.")
    r.append(f"Best performer: {f['highest_margin_product']} at {f['highest_margin_pct']}% net margin; give it more shelf and ad budget.")
    return r


# ============================================================================== ReceivablesIQ
def run_receivables(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ReceivablesIQ")
    inv = data["invoices"]
    rec = inv[inv["type"] == "receivable"].copy()
    if rec.empty:
        raise ValueError("needs receivable invoices")
    as_of = ctx.as_of
    paid = rec[rec["paid_date"].notna() & (rec["paid_date"] <= as_of)].copy()
    paid["days_to_pay"] = (paid["paid_date"] - paid["invoice_date"]).dt.days
    paid["days_late"] = (paid["paid_date"] - paid["due_date"]).dt.days
    rec["open"] = open_mask(rec["paid_date"], as_of)
    op = rec[rec["open"]].copy()
    op["days_overdue"] = (as_of - op["due_date"]).dt.days.clip(lower=0)
    overall_dtp = float(paid["days_to_pay"].median()) if len(paid) else 45.0

    beh = paid.groupby("party").agg(paid_invoices=("amount", "size"), avg_days_to_pay=("days_to_pay", "mean"),
                                    avg_days_late=("days_late", "mean"),
                                    late_share=("days_late", lambda x: float((x > 0).mean())))
    parties = pd.DataFrame(index=sorted(rec["party"].unique()))
    parties = parties.join(beh).join(op.groupby("party").agg(open_amount=("amount", "sum"), open_invoices=("amount", "size"),
                                                           oldest_days_overdue=("days_overdue", "max"))).fillna(
        {"open_amount": 0.0, "open_invoices": 0, "oldest_days_overdue": 0, "late_share": 0.0})
    od = op[op["days_overdue"] > 0].groupby("party")["amount"].sum()
    parties["overdue_amount"] = od.reindex(parties.index).fillna(0.0)
    parties["expected_days_to_pay"] = parties["avg_days_to_pay"].fillna(overall_dtp)
    parties["late_risk"] = (0.15 + 0.7 * parties["late_share"]).clip(0, 0.95)
    parties["priority_score"] = parties["overdue_amount"] * (0.5 + parties["late_risk"])

    op = op.merge(parties[["expected_days_to_pay", "late_risk"]], left_on="party", right_index=True, how="left")
    op["expected_pay_date"] = op["invoice_date"] + pd.to_timedelta(op["expected_days_to_pay"].round(), unit="D")
    op["expected_pay_date"] = op["expected_pay_date"].where(op["expected_pay_date"] > as_of, as_of + pd.Timedelta(days=7))
    in30 = op[op["expected_pay_date"] <= as_of + pd.Timedelta(days=30)]
    res.tables["parties"] = parties.reset_index(names="party").sort_values("priority_score", ascending=False)
    res.tables["open_invoices"] = op.sort_values("days_overdue", ascending=False)[
        ["invoice_id", "party", "invoice_date", "due_date", "amount", "days_overdue", "expected_pay_date"]]
    aging = pd.cut(op["days_overdue"], [-1, 0, 30, 60, 90, 10 ** 6], labels=["Not yet due", "1-30", "31-60", "61-90", "90+"])
    res.tables["aging"] = op.groupby(aging, observed=False)["amount"].agg(["sum", "size"]).rename(
        columns={"sum": "amount", "size": "invoices"}).reset_index(names="bucket")

    # reminder drafts: tone follows payment history. Drafts only; nothing is sent.
    drafts = []
    for p, r in parties[parties["overdue_amount"] > 0].sort_values("priority_score", ascending=False).head(8).iterrows():
        firm = r["late_share"] > 0.5 or r["oldest_days_overdue"] > 60
        tone = "firm" if firm else "friendly"
        body = (f"Dear {p} team,\n\nOur records show {inr(r['overdue_amount'], compact=False)} is overdue, the oldest invoice by "
                f"{int(r['oldest_days_overdue'])} days. " + ("Please arrange payment within 3 working days or let us know the "
                                                            "expected date, so we can keep your account in good standing.\n"
                                                            if firm else "A quick check on when we can expect it would be appreciated; "
                                                                         "please ignore this note if payment is already on its way.\n")
                + "\nThank you,\nAccounts")
        drafts.append(dict(party=p, tone=tone, overdue_amount=r["overdue_amount"], draft=body))
    res.tables["reminder_drafts"] = pd.DataFrame(drafts)

    open_total, overdue_total = float(op["amount"].sum()), float(op.loc[op["days_overdue"] > 0, "amount"].sum())
    worst = parties.sort_values("priority_score", ascending=False).iloc[0]
    over90 = float(op.loc[op["days_overdue"] > 90, "amount"].sum())
    res.metrics = dict(open_receivables=open_total, overdue=overdue_total, avg_days_to_pay=overall_dtp)
    res.facts = clean_facts(dict(
        open_receivables=inr(open_total), overdue_amount=inr(overdue_total),
        overdue_pct=pct1(safe_div(overdue_total, open_total)), over_90_days_amount=inr(over90),
        median_days_to_pay=r1(overall_dtp, 0), expected_collections_next_30_days=inr(float(in30["amount"].sum())),
        first_party_to_chase=str(worst.name), first_party_overdue=inr(worst["overdue_amount"]),
        parties_with_overdue=int((parties["overdue_amount"] > 0).sum()), reminder_drafts_ready=len(drafts)))
    if safe_div(overdue_total, open_total) > 0.40:
        res.add_flag(RED, f"{safe_div(overdue_total, open_total):.0%} of receivables are overdue ({inr(overdue_total)})")
    elif safe_div(overdue_total, open_total) > 0.20:
        res.add_flag(AMBER, f"{safe_div(overdue_total, open_total):.0%} of receivables are overdue ({inr(overdue_total)})")
    if over90 > 0:
        res.add_flag(RED, f"{inr(over90)} is more than 90 days overdue", "Escalate beyond routine reminders.")
    for p, r in parties.iterrows():
        if r["late_share"] > 0.6 and r["paid_invoices"] >= 4:
            res.add_flag(AMBER, f"{p} pays late on {r['late_share']:.0%} of invoices",
                         f"Average {r['avg_days_late']:.0f} days after the due date.")
    if not res.flags:
        res.add_flag(GREEN, "Receivables are healthy")
    res.notes.append("Expected payment date = invoice date + that customer's average days to pay (the overall median when there is "
                     "no history). Late-payment risk is a simple rule from each customer's share of late payments, not a trained model. "
                     "Reminder drafts are never sent automatically.")
    return res


def rec_receivables(f, flags):
    r = []
    if f.get("overdue_pct", 0) and f["overdue_pct"] > 20:
        r.append(f"{f['overdue_pct']}% of receivables ({f['overdue_amount']}) are overdue. Start with {f['first_party_to_chase']} "
                 f"({f['first_party_overdue']} overdue); {f['reminder_drafts_ready']} reminder drafts are ready to review.")
    if f.get("over_90_days_amount") and f["over_90_days_amount"] != "Rs 0":
        r.append(f"{f['over_90_days_amount']} is over 90 days overdue; move to a call and a written final notice.")
    r.append(f"Expect about {f['expected_collections_next_30_days']} in the next 30 days based on each customer's payment pattern; "
             "plan payables against that, not against invoice due dates.")
    return r


# ============================================================================== CreditIQ
def run_credit(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("CreditIQ")
    inv, lim = data["invoices"], data["credit_limits"].copy()
    rec = inv[inv["type"] == "receivable"].copy()
    as_of = ctx.as_of
    rec["open"] = open_mask(rec["paid_date"], as_of)
    expo = rec[rec["open"]].groupby("party")["amount"].sum()
    over = rec[rec["open"] & (rec["due_date"] < as_of)].groupby("party")["amount"].sum()
    paid = rec[rec["paid_date"].notna() & (rec["paid_date"] <= as_of)].copy()
    paid["late"] = (paid["paid_date"] - paid["due_date"]).dt.days
    behaviour = paid.groupby("party").agg(avg_days_late=("late", "mean"), late_share=("late", lambda x: float((x > 0).mean())))
    recent = rec[rec["invoice_date"] >= as_of - pd.Timedelta(days=180)].groupby("party")["amount"].sum() / 180
    t = lim.rename(columns={"customer": "party"}).set_index("party")
    t["exposure"] = expo.reindex(t.index).fillna(0.0)
    t["overdue"] = over.reindex(t.index).fillna(0.0)
    t["utilisation"] = t["exposure"] / t["credit_limit"].replace(0, np.nan)
    t = t.join(behaviour)
    terms = t["payment_terms_days"].fillna(30) if "payment_terms_days" in t else pd.Series(30, index=t.index)
    t["suggested_limit"] = (recent.reindex(t.index).fillna(0.0) * terms * 1.25).round(-4)

    def action(r):
        if r["exposure"] > r["credit_limit"]:
            return "Hold new orders until payment brings exposure under the limit"
        if r["late_share"] > 0.6 and r["utilisation"] > 0.6:
            return "Shorten terms or ask for part advance payment"
        if r["utilisation"] > 0.85:
            return "Near the limit: review before the next large order"
        if r["utilisation"] < 0.5 and (r["late_share"] if pd.notna(r["late_share"]) else 0) < 0.25:
            return "Good payer with headroom: can offer a higher limit"
        return "Keep as is"

    t["action"] = t.apply(action, axis=1)
    res.tables["customers"] = t.reset_index().sort_values("utilisation", ascending=False)
    total = float(t["exposure"].sum())
    over_limit = t[t["exposure"] > t["credit_limit"]]
    top = t["exposure"].idxmax() if total > 0 else None
    res.metrics = dict(total_exposure=total, customers_over_limit=len(over_limit))
    res.facts = clean_facts(dict(
        total_credit_exposure=inr(total), customers_over_limit=len(over_limit),
        excess_over_limits=inr(float((over_limit["exposure"] - over_limit["credit_limit"]).sum())),
        customers_near_limit=int(((t["utilisation"] > 0.85) & (t["exposure"] <= t["credit_limit"])).sum()),
        largest_exposure_customer=str(top) if top else None,
        largest_exposure_share_pct=pct1(safe_div(float(t["exposure"].max()), total)) if total else None,
        customers_with_headroom=int(t["action"].str.startswith("Good payer").sum())))
    for p, r in over_limit.iterrows():
        res.add_flag(RED, f"{p} is over its credit limit by {inr(r['exposure'] - r['credit_limit'])}",
                     f"Exposure {inr(r['exposure'])} against a limit of {inr(r['credit_limit'])}.")
    for p, r in t[(t["utilisation"] > 0.85) & (t["exposure"] <= t["credit_limit"])].iterrows():
        res.add_flag(AMBER, f"{p} has used {r['utilisation']:.0%} of its credit limit")
    if total and safe_div(float(t["exposure"].max()), total) > 0.35:
        res.add_flag(AMBER, f"{top} makes up {safe_div(float(t['exposure'].max()), total):.0%} of all credit exposure",
                     "Concentration risk: one late payment would hurt cash flow.")
    if not res.flags:
        res.add_flag(GREEN, "Credit exposure is within limits")
    res.notes.append("Exposure = unpaid receivable invoices. Suggested limit = average daily billing over 180 days x payment terms x 1.25, "
                     "a simple heuristic to start a conversation, not a credit rating. Actions are suggestions for a person to decide.")
    return res


def rec_credit(f, flags):
    r = []
    if f.get("customers_over_limit"):
        r.append(f"{f['customers_over_limit']} customer(s) exceed their limit by {f['excess_over_limits']} in total. Hold new orders until "
                 "they pay down, and confirm any exception in writing.")
    if f.get("largest_exposure_share_pct", 0) and f["largest_exposure_share_pct"] > 35:
        r.append(f"{f['largest_exposure_customer']} is {f['largest_exposure_share_pct']}% of exposure; spread sales or ask for security.")
    if f.get("customers_with_headroom"):
        r.append(f"{f['customers_with_headroom']} reliable customer(s) have headroom; a higher limit could grow sales safely.")
    return r or ["Credit exposure is within limits."]


# ============================================================================== ComplianceIQ
def _norm_inv(x: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())


def _next_due(today: pd.Timestamp, day: int) -> pd.Timestamp:
    cand = pd.Timestamp(year=today.year, month=today.month, day=day)
    return cand if cand >= today else (cand + pd.offsets.MonthBegin(1)).replace(day=day)


def run_compliance(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ComplianceIQ")
    b, g = data["gst_books"].copy(), data["gstr2b"].copy()
    for df in (b, g):
        df["key"] = df["party_gstin"] + "|" + df["invoice_no"].map(_norm_inv)
    m = b.merge(g, on="key", how="outer", suffixes=("_books", "_2b"), indicator=True)
    both = m[m["_merge"] == "both"].copy()
    both["gst_diff"] = (both["gst_amount_books"] - both["gst_amount_2b"]).round(2)
    both["taxable_diff"] = (both["taxable_value_books"] - both["taxable_value_2b"]).round(2)
    matched = both[(both["gst_diff"].abs() <= 1) & (both["taxable_diff"].abs() <= 1)]
    mismatch = both[(both["gst_diff"].abs() > 1) | (both["taxable_diff"].abs() > 1)]
    missing_2b = m[m["_merge"] == "left_only"]
    not_in_books = m[m["_merge"] == "right_only"]

    res.tables["mismatched_amounts"] = mismatch[["invoice_no_books", "party_gstin_books", "gst_amount_books", "gst_amount_2b", "gst_diff"]].rename(
        columns={"invoice_no_books": "invoice_no", "party_gstin_books": "party_gstin"}).sort_values("gst_diff", ascending=False)
    res.tables["in_books_not_in_2b"] = missing_2b[["invoice_no_books", "party_gstin_books", "invoice_date_books", "gst_amount_books"]].rename(
        columns={"invoice_no_books": "invoice_no", "party_gstin_books": "party_gstin", "invoice_date_books": "invoice_date",
                 "gst_amount_books": "gst_amount"}).sort_values("gst_amount", ascending=False)
    res.tables["in_2b_not_in_books"] = not_in_books[["invoice_no_2b", "party_gstin_2b", "invoice_date_2b", "gst_amount_2b"]].rename(
        columns={"invoice_no_2b": "invoice_no", "party_gstin_2b": "party_gstin", "invoice_date_2b": "invoice_date",
                 "gst_amount_2b": "gst_amount"})
    # supplier view
    issues = pd.concat([
        missing_2b.assign(issue="Not filed by supplier (not in 2B)", amount=missing_2b["gst_amount_books"], gstin=missing_2b["party_gstin_books"]),
        mismatch.assign(issue="Amount differs", amount=mismatch["gst_diff"].clip(lower=0), gstin=mismatch["party_gstin_books"])])
    sup = issues.groupby("gstin").agg(problem_invoices=("issue", "size"), itc_at_risk=("amount", "sum")).sort_values("itc_at_risk", ascending=False)
    party_name = b.drop_duplicates("party_gstin").set_index("party_gstin")["party"] if "party" in b else pd.Series(dtype=str)
    sup["supplier"] = party_name.reindex(sup.index).fillna("")
    res.tables["suppliers_to_chase"] = sup.reset_index()

    total_itc = float(b["gst_amount"].sum())
    at_risk = float(missing_2b["gst_amount_books"].sum() + mismatch["gst_diff"].clip(lower=0).sum())
    today = ctx.today
    g1, g3b = _next_due(today, 11), _next_due(today, 20)
    res.tables["deadlines"] = pd.DataFrame([
        dict(filing="GSTR-1 (outward supplies)", due=g1, days_left=(g1 - today).days),
        dict(filing="GSTR-3B (summary return and tax payment)", due=g3b, days_left=(g3b - today).days)])
    res.metrics = dict(itc_in_books=total_itc, itc_at_risk=at_risk, matched_invoices=len(matched))
    res.facts = clean_facts(dict(
        invoices_in_books=len(b), invoices_matched=len(matched), matched_pct=pct1(safe_div(len(matched), len(b))),
        itc_in_books=inr(total_itc), itc_at_risk=inr(at_risk), itc_at_risk_pct=pct1(safe_div(at_risk, total_itc)),
        invoices_missing_in_2b=len(missing_2b), invoices_with_amount_mismatch=len(mismatch),
        invoices_in_2b_not_in_books=len(not_in_books),
        supplier_to_chase_first=str(sup["supplier"].iloc[0] or sup.index[0]) if len(sup) else None,
        days_to_gstr3b=(g3b - today).days, days_to_gstr1=(g1 - today).days))
    share = safe_div(at_risk, total_itc)
    if share > 0.15:
        res.add_flag(RED, f"{inr(at_risk)} of input tax credit ({share:.0%}) is at risk", "Supplier filings do not match your books.")
    elif share > 0.05:
        res.add_flag(AMBER, f"{inr(at_risk)} of input tax credit ({share:.0%}) is at risk")
    if len(not_in_books):
        res.add_flag(AMBER, f"{len(not_in_books)} invoice(s) appear in GSTR-2B but not in your books",
                     "Check whether purchases are unrecorded or the supplier used the wrong GSTIN.")
    if (g3b - today).days <= 7:
        res.add_flag(AMBER, f"GSTR-3B is due in {(g3b - today).days} days")
    if not res.flags:
        res.add_flag(GREEN, "Books and GSTR-2B agree")
    res.notes.append("Invoices are matched on supplier GSTIN and invoice number with punctuation and case ignored; amounts match within Rs 1. "
                     "Due dates use the usual 11th (GSTR-1) and 20th (GSTR-3B) monthly, which the government extends from time to time and "
                     "which differ for quarterly filers: confirm with your CA. This is a reconciliation aid, not tax advice.")
    return res


def rec_compliance(f, flags):
    r = []
    if f.get("itc_at_risk_pct", 0) and f["itc_at_risk_pct"] > 3:
        r.append(f"{f['itc_at_risk']} of input tax credit ({f['itc_at_risk_pct']}%) cannot be claimed until fixed: "
                 f"{f['invoices_missing_in_2b']} invoice(s) are missing from GSTR-2B and {f['invoices_with_amount_mismatch']} differ in amount. "
                 f"Ask {f.get('supplier_to_chase_first', 'the suppliers')} to file or correct them before the return is due.")
    if f.get("invoices_in_2b_not_in_books"):
        r.append(f"{f['invoices_in_2b_not_in_books']} invoice(s) are in GSTR-2B but not in your books; record or dispute them.")
    r.append(f"GSTR-1 is due in {f['days_to_gstr1']} days and GSTR-3B in {f['days_to_gstr3b']} days (verify the dates with your CA).")
    return r


# ============================================================================== ExpenseIQ
APPROVAL_THRESHOLD = 50_000   # spend at or above this should carry an approver


def run_expenses(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ExpenseIQ")
    e = data["expenses"].copy().sort_values("date")
    thr = ctx.settings.get("expense_approval_threshold", APPROVAL_THRESHOLD)
    total = float(e["amount"].sum())

    # possible duplicates: same vendor and amount within 7 days
    dups = []
    for (v, a), g in e.groupby(["vendor", "amount"]):
        if len(g) > 1:
            dates = g["date"].sort_values().tolist()
            ids = g.sort_values("date")["expense_id"].tolist()
            for i in range(1, len(dates)):
                if (dates[i] - dates[i - 1]).days <= 7:
                    dups.append(dict(vendor=v, amount=a, first_date=dates[i - 1], second_date=dates[i], expense_ids=f"{ids[i - 1]}, {ids[i]}"))
    dups = pd.DataFrame(dups, columns=["vendor", "amount", "first_date", "second_date", "expense_ids"])
    # outliers within category (robust z-score on log amount)
    e["log"] = np.log(e["amount"].clip(lower=1))
    med = e.groupby("category")["log"].transform("median")
    mad = e.groupby("category")["log"].transform(lambda x: (x - x.median()).abs().median()).replace(0, np.nan)
    e["robust_z"] = 0.6745 * (e["log"] - med) / mad
    out = e[e["robust_z"] > 3.5].sort_values("amount", ascending=False)
    unapproved = e[(e["amount"] >= thr) & (e["approved_by"].isna() if "approved_by" in e else True)]
    cash_no_receipt = e[(e["payment_mode"].str.lower() == "cash") & (e["receipt_attached"] == 0)] if {"payment_mode", "receipt_attached"} <= set(e.columns) else e.iloc[0:0]
    res.tables["possible_duplicates"] = dups
    res.tables["outliers"] = out[["expense_id", "date", "category", "vendor", "amount"]]
    res.tables["large_without_approval"] = unapproved[["expense_id", "date", "category", "vendor", "amount"]]
    res.tables["cash_without_receipt"] = cash_no_receipt[["expense_id", "date", "category", "vendor", "amount"]]
    by_cat = e.groupby("category")["amount"].agg(["sum", "size"]).rename(columns={"sum": "amount", "size": "transactions"}).sort_values("amount", ascending=False)
    by_cat["share"] = by_cat["amount"] / total
    res.tables["by_category"] = by_cat.reset_index()
    e["month"] = e["date"].dt.to_period("M").dt.to_timestamp()
    bm = e.pivot_table(index="month", columns="category", values="amount", aggfunc="sum").fillna(0.0)
    res.tables["by_month"] = bm.reset_index()
    res.tables["top_vendors"] = e.groupby("vendor")["amount"].agg(["sum", "size"]).rename(columns={"sum": "amount", "size": "transactions"}).sort_values("amount", ascending=False).head(10).reset_index()
    months = sorted(e["month"].unique())
    last3, prev3 = e[e["month"].isin(months[-3:])]["amount"].sum(), e[e["month"].isin(months[-6:-3])]["amount"].sum()
    trend = safe_div(last3 - prev3, prev3)
    dup_amt = float(dups["amount"].sum()) if len(dups) else 0.0
    res.metrics = dict(total_spend=total, possible_duplicates=len(dups), outliers=len(out), large_without_approval=len(unapproved))
    res.facts = clean_facts(dict(
        total_spend=inr(total), transactions=len(e), top_category=str(by_cat.index[0]), top_category_share_pct=pct1(by_cat["share"].iloc[0]),
        possible_duplicate_payments=len(dups), possible_duplicate_value=inr(dup_amt),
        unusually_large_expenses=len(out), largest_outlier=inr(float(out["amount"].iloc[0])) if len(out) else None,
        large_expenses_without_approval=len(unapproved), unapproved_value=inr(float(unapproved["amount"].sum())),
        cash_payments_without_receipt=len(cash_no_receipt), approval_threshold=inr(thr, compact=False),
        spend_change_last_3_vs_prior_3_months_pct=pct1(trend)))
    if len(dups):
        res.add_flag(RED if dup_amt > 100000 else AMBER, f"{len(dups)} possible duplicate payment(s) worth {inr(dup_amt)}",
                     "Same vendor and amount within 7 days; confirm they were not paid twice.")
    if len(unapproved):
        res.add_flag(RED, f"{len(unapproved)} expense(s) of {inr(thr, compact=False)} or more have no approver ({inr(float(unapproved['amount'].sum()))})")
    if len(out):
        res.add_flag(AMBER, f"{len(out)} unusually large expense(s) for their category", f"Largest is {inr(float(out['amount'].iloc[0]))}.")
    if len(cash_no_receipt):
        res.add_flag(AMBER, f"{len(cash_no_receipt)} cash payment(s) have no receipt", "Needed for audit and for claiming input credit.")
    if pd.notna(trend) and trend > 0.2:
        res.add_flag(AMBER, f"Spend is up {trend:.0%} over the last three months versus the three before")
    if not res.flags:
        res.add_flag(GREEN, "No expense irregularities found")
    res.notes.append("Checks are rules, not proof of wrongdoing: duplicates (same vendor and amount within 7 days), outliers (robust z-score above 3.5 "
                     "within category), missing approval above the threshold, and cash with no receipt. Each hit needs a person to confirm.")
    return res


def rec_expenses(f, flags):
    r = []
    if f.get("possible_duplicate_payments"):
        r.append(f"Review {f['possible_duplicate_payments']} possible duplicate payment(s) worth {f['possible_duplicate_value']}; ask the vendor for a credit if any were paid twice.")
    if f.get("large_expenses_without_approval"):
        r.append(f"{f['large_expenses_without_approval']} large expense(s) ({f['unapproved_value']}) lack an approver. Require approval above {f['approval_threshold']} before payment.")
    if f.get("cash_payments_without_receipt"):
        r.append(f"Collect receipts for {f['cash_payments_without_receipt']} cash payment(s), or move vendors to UPI or bank transfer.")
    r.append(f"{f['top_category']} is the biggest cost at {f['top_category_share_pct']}% of spend; start any savings review there.")
    return r


# ============================================================================== registration
register(ModuleSpec("ProfitIQ", "Finance and compliance", "True profit by product, region and customer, after costs, fees and returns.",
                    ["sales", "product_costs"], run_profit, rec_profit, "Both", "2"))
register(ModuleSpec("ReceivablesIQ", "Finance and compliance", "Who owes you, who is likely to pay late, and who to chase first.",
                    ["invoices"], run_receivables, rec_receivables, "Both", "2"))
register(ModuleSpec("CreditIQ", "Finance and compliance", "Credit exposure and limits by customer, with suggested actions.",
                    ["invoices", "credit_limits"], run_credit, rec_credit, "B2B", "2"))
register(ModuleSpec("ComplianceIQ", "Finance and compliance", "GST reconciliation (books versus GSTR-2B) and filing deadlines.",
                    ["gst_books", "gstr2b"], run_compliance, rec_compliance, "Both", "2"))
register(ModuleSpec("ExpenseIQ", "Finance and compliance", "Spend analysis with duplicate, outlier and approval checks.",
                    ["expenses"], run_expenses, rec_expenses, "Both", "2"))
