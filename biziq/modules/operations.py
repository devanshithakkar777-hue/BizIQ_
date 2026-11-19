"""Sales operations and service modules: PipelineIQ, QuoteIQ, ContractIQ, SupportIQ."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import AMBER, GREEN, RED, ModuleResult, inr, safe_div
from ..registry import Context, ModuleSpec, register
from ._common import clean_facts, pct1, r1

# ============================================================================== PipelineIQ
OPEN_STAGES = ["lead", "qualified", "proposal", "negotiation"]
STAGE_PROB = {"lead": 0.10, "qualified": 0.25, "proposal": 0.50, "negotiation": 0.70}   # assumptions, editable via settings
STALL_DAYS = 21


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / b that survives empty groups (whose aggregates come back as text) and zero denominators."""
    return pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0, np.nan)


def run_pipeline(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("PipelineIQ")
    p = data["pipeline"].copy()
    probs = {**STAGE_PROB, **ctx.settings.get("stage_probabilities", {})}
    stall = ctx.settings.get("stall_days", STALL_DAYS)
    as_of = ctx.as_of
    won, lost = p[p["stage"] == "won"], p[p["stage"] == "lost"]
    op = p[p["stage"].isin(OPEN_STAGES)].copy()
    op["days_since_activity"] = (as_of - op["last_activity_date"]).dt.days
    op["age_days"] = (as_of - op["created_date"]).dt.days
    op["probability"] = op["stage"].map(probs)
    op["weighted_value"] = op["amount"] * op["probability"]
    op["stalled"] = op["days_since_activity"] > stall
    if "expected_close" in op:
        op["past_expected_close"] = op["expected_close"] < as_of
    else:
        op["past_expected_close"] = False
    funnel = op.groupby("stage").agg(deals=("deal_id", "size"), value=("amount", "sum"), weighted=("weighted_value", "sum")).reindex(OPEN_STAGES).fillna(0)
    res.tables["funnel"] = funnel.reset_index()
    res.tables["stalled_deals"] = op[op["stalled"] | op["past_expected_close"]].sort_values("amount", ascending=False)[
        ["deal_id", "account", "stage", "amount", "days_since_activity", "past_expected_close"] + (["owner"] if "owner" in op else [])]
    hist = p[p["stage"].isin(["won", "lost"])]
    win_rate = safe_div(len(won), len(hist))
    win_rate_value = safe_div(float(won["amount"].sum()), float(hist["amount"].sum()))
    for col, name in (("source", "by_source"), ("owner", "by_owner")):
        if col in hist and len(hist):
            g = hist.groupby(col).agg(closed=("deal_id", "size"), won=("stage", lambda x: int((x == "won").sum())), won_value=("amount", lambda x: float(x[hist.loc[x.index, "stage"] == "won"].sum())))
            g["win_rate"] = _ratio(g["won"], g["closed"])
            res.tables[name] = g.reset_index().sort_values("win_rate", ascending=False)
    cycle = (won["last_activity_date"] - won["created_date"]).dt.days
    open_val, weighted = float(op["amount"].sum()), float(op["weighted_value"].sum())
    stalled_val = float(op.loc[op["stalled"], "amount"].sum())
    res.metrics = dict(open_value=open_val, weighted_value=weighted, win_rate=win_rate)
    res.facts = clean_facts(dict(
        open_deals=len(op), open_pipeline_value=inr(open_val), weighted_pipeline_value=inr(weighted),
        win_rate_by_count_pct=pct1(win_rate), win_rate_by_value_pct=pct1(win_rate_value),
        average_sales_cycle_days=r1(float(cycle.mean()), 0) if len(cycle) else None,
        stalled_deals=int(op["stalled"].sum()), stalled_value=inr(stalled_val), stall_threshold_days=int(stall),
        stalled_value_pct=pct1(safe_div(stalled_val, open_val)), deals_past_expected_close=int(op["past_expected_close"].sum()),
        largest_stalled_deal=str(op[op["stalled"]].sort_values("amount").iloc[-1]["account"]) if op["stalled"].any() else None,
        largest_stalled_deal_value=inr(float(op[op["stalled"]]["amount"].max())) if op["stalled"].any() else None,
        best_source=str(res.tables["by_source"].iloc[0]["source"]) if "by_source" in res.tables and len(res.tables["by_source"]) else None))
    if safe_div(stalled_val, open_val) > 0.25:
        res.add_flag(RED if safe_div(stalled_val, open_val) > 0.4 else AMBER,
                     f"{int(op['stalled'].sum())} deal(s) worth {inr(stalled_val)} have had no activity for over {stall} days")
    if op["past_expected_close"].sum():
        res.add_flag(AMBER, f"{int(op['past_expected_close'].sum())} open deal(s) are past their expected close date")
    if pd.notna(win_rate) and win_rate < 0.25:
        res.add_flag(AMBER, f"Win rate is {win_rate:.0%} on closed deals")
    if not res.flags:
        res.add_flag(GREEN, "Pipeline is active and moving")
    res.notes.append("Weighted value uses ASSUMED stage probabilities (lead 10%, qualified 25%, proposal 50%, negotiation 70%); replace them with your own "
                     f"closed-deal history once you have enough deals. A deal is stalled after {stall} days without activity. The sales cycle is "
                     "approximated as created date to last activity date for won deals.")
    return res


def rec_pipeline(f, flags):
    r = [f"{f['open_deals']} open deals are worth {f['open_pipeline_value']}; weighted by stage that is about {f['weighted_pipeline_value']}. "
         f"Historic win rate is {f['win_rate_by_count_pct']}% by count."]
    if f.get("stalled_deals"):
        r.append(f"{f['stalled_deals']} deal(s) worth {f['stalled_value']} have gone quiet for over {f['stall_threshold_days']} days"
                 + (f"; start with {f['largest_stalled_deal']} ({f['largest_stalled_deal_value']})." if f.get("largest_stalled_deal") else "."))
    if f.get("deals_past_expected_close"):
        r.append(f"Update or close out the {f['deals_past_expected_close']} deal(s) past their expected close date so the forecast stays honest.")
    return r


# ============================================================================== QuoteIQ
def run_quotes(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("QuoteIQ")
    q = data["quotes"].copy()
    as_of = ctx.as_of
    done = q[q["status"].isin(["won", "lost"])].copy()
    op = q[q["status"] == "open"].copy()
    win_n = safe_div(int((done["status"] == "won").sum()), len(done))
    win_v = safe_div(float(done.loc[done["status"] == "won", "amount"].sum()), float(done["amount"].sum()))
    if "cost" in done and done["cost"].notna().any():
        done["margin"] = (done["amount"] - done["cost"]) / done["amount"]
        bands = pd.cut(done["margin"], [-1, 0.15, 0.22, 0.28, 1], labels=["under 15%", "15-22%", "22-28%", "over 28%"])
        mb = done.groupby(bands, observed=True).agg(quotes=("quote_id", "size"), won=("status", lambda x: int((x == "won").sum())))
        mb["win_rate"] = _ratio(mb["won"], mb["quotes"])
        res.tables["win_rate_by_margin"] = mb.reset_index(names="quoted_margin")
        avg_m_won = float(done.loc[done["status"] == "won", "margin"].mean())
        avg_m_lost = float(done.loc[done["status"] == "lost", "margin"].mean())
    else:
        avg_m_won = avg_m_lost = float("nan")
    try:
        sz = pd.qcut(done["amount"].rank(method="first"), 3, labels=["small", "medium", "large"])
    except ValueError:       # fewer than three closed quotes: one group
        sz = pd.Series("all sizes", index=done.index)
    bs = done.groupby(sz, observed=True).agg(quotes=("quote_id", "size"), won=("status", lambda x: int((x == "won").sum())))
    bs["win_rate"] = _ratio(bs["won"], bs["quotes"])
    res.tables["win_rate_by_size"] = bs.reset_index(names="size")
    if "lost_reason" in done:
        lr = done[done["status"] == "lost"].groupby("lost_reason").size().rename("quotes").sort_values(ascending=False).reset_index()
        lr = lr[lr["lost_reason"].astype(str).str.len() > 0]
        lr["share"] = lr["quotes"] / lr["quotes"].sum()
        res.tables["lost_reasons"] = lr
    cust = done.groupby("customer").agg(quotes=("quote_id", "size"), won=("status", lambda x: int((x == "won").sum())), value=("amount", "sum"))
    cust["win_rate"] = _ratio(cust["won"], cust["quotes"])
    res.tables["by_customer"] = cust.sort_values("value", ascending=False).reset_index()
    op["age_days"] = (as_of - op["sent_date"]).dt.days
    stale = op[op["age_days"] > 14].sort_values("amount", ascending=False)
    res.tables["open_quotes_to_follow_up"] = stale[["quote_id", "customer", "sent_date", "amount", "age_days"]]
    days = (done["decided_date"] - done["sent_date"]).dt.days if "decided_date" in done else pd.Series(dtype=float)
    top_r = res.tables["lost_reasons"].iloc[0] if "lost_reasons" in res.tables and len(res.tables["lost_reasons"]) else None
    res.metrics = dict(win_rate=win_n, open_value=float(op["amount"].sum()))
    res.facts = clean_facts(dict(
        quotes_decided=len(done), win_rate_by_count_pct=pct1(win_n), win_rate_by_value_pct=pct1(win_v),
        average_quoted_margin_on_won_pct=pct1(avg_m_won), average_quoted_margin_on_lost_pct=pct1(avg_m_lost),
        top_loss_reason=str(top_r["lost_reason"]) if top_r is not None else None, top_loss_reason_share_pct=pct1(top_r["share"]) if top_r is not None else None,
        open_quotes=len(op), open_quote_value=inr(float(op["amount"].sum())), open_quotes_older_than_14_days=len(stale),
        stale_quote_value=inr(float(stale["amount"].sum())), average_days_to_decision=r1(float(days.mean()), 0) if len(days) else None))
    if top_r is not None and top_r["lost_reason"] == "Price" and top_r["share"] > 0.4:
        res.add_flag(AMBER, f"Price is the reason for {top_r['share']:.0%} of lost quotes")
    if len(stale):
        res.add_flag(AMBER, f"{len(stale)} open quote(s) worth {inr(float(stale['amount'].sum()))} are older than 14 days without a decision",
                     "Follow up while the buyer still remembers the request.")
    if pd.notna(win_n) and win_n < 0.25:
        res.add_flag(AMBER, f"Only {win_n:.0%} of decided quotes are won")
    if not res.flags:
        res.add_flag(GREEN, "Quote conversion is healthy")
    res.notes.append("Quoted margin = (amount - cost) / amount as recorded on each quote. Win rate by margin band shows whether higher-margin quotes lose more often; "
                     "with small samples treat differences between bands as hints, not proof.")
    return res


def rec_quotes(f, flags):
    r = [f"{f['win_rate_by_count_pct']}% of decided quotes are won ({f['win_rate_by_value_pct']}% by value)."]
    if f.get("top_loss_reason"):
        r.append(f"The most common reason for losing is '{f['top_loss_reason']}' ({f['top_loss_reason_share_pct']}% of losses); address it in the next quotes.")
    if f.get("open_quotes_older_than_14_days"):
        r.append(f"Follow up on {f['open_quotes_older_than_14_days']} quote(s) worth {f['stale_quote_value']} that have waited over 14 days.")
    if f.get("average_quoted_margin_on_won_pct") and f.get("average_quoted_margin_on_lost_pct"):
        r.append(f"Won quotes carried {f['average_quoted_margin_on_won_pct']}% margin versus {f['average_quoted_margin_on_lost_pct']}% on lost ones; "
                 "check the win-rate-by-margin table before cutting prices.")
    return r


# ============================================================================== ContractIQ
def run_contracts(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ContractIQ")
    c = data["contracts"].copy()
    as_of = ctx.as_of
    c["days_to_end"] = (c["end_date"] - as_of).dt.days
    notice = c["renewal_notice_days"].fillna(30) if "renewal_notice_days" in c else pd.Series(30, index=c.index)
    c["notice_deadline"] = c["end_date"] - pd.to_timedelta(notice, unit="D")
    c["days_to_notice_deadline"] = (c["notice_deadline"] - as_of).dt.days
    auto = c["auto_renew"].fillna(0).astype(int) if "auto_renew" in c else pd.Series(0, index=c.index)

    def status(r, a):
        if r["days_to_end"] < 0:
            return "Expired: renew, replace or remove"
        if r["days_to_notice_deadline"] < 0:
            return "Notice date passed: auto-renews unless cancelled" if a else "Notice date passed: renew or let lapse now"
        if r["days_to_notice_deadline"] <= 30:
            return "Decide now: notice deadline within 30 days"
        if r["days_to_end"] <= 90:
            return "Ending within 90 days: start renewal talks"
        return "OK"

    c["status"] = [status(r, a) for (_, r), a in zip(c.iterrows(), auto)]
    esc = c["escalation_pct"].fillna(0) / 100 if "escalation_pct" in c else pd.Series(0.0, index=c.index)
    c["escalation_cost_next_year"] = np.where(c["type"].isin(["supplier", "lease", "service"]) if "type" in c else True, c["value"] * esc, 0.0)
    cols = ["contract_id", "party", "type", "value", "end_date", "days_to_end", "notice_deadline", "days_to_notice_deadline", "status"] if "type" in c else \
        ["contract_id", "party", "value", "end_date", "days_to_end", "notice_deadline", "days_to_notice_deadline", "status"]
    res.tables["contracts"] = c.sort_values("days_to_end")[cols]
    res.tables["escalations"] = c[c["escalation_cost_next_year"] > 0].sort_values("escalation_cost_next_year", ascending=False)[
        ["contract_id", "party", "value", "escalation_pct", "escalation_cost_next_year"]]
    ending90 = c[(c["days_to_end"] >= 0) & (c["days_to_end"] <= 90)]
    expired = c[c["days_to_end"] < 0]
    missed = c[(c["days_to_end"] >= 0) & (c["days_to_notice_deadline"] < 0)]
    urgent = c[(c["days_to_notice_deadline"] >= 0) & (c["days_to_notice_deadline"] <= 30)]
    res.metrics = dict(contracts=len(c), ending_90_days=len(ending90))
    res.facts = clean_facts(dict(
        contracts=len(c), contract_value=inr(float(c["value"].sum())), ending_within_90_days=len(ending90),
        value_ending_within_90_days=inr(float(ending90["value"].sum())), expired_contracts=len(expired),
        notice_deadline_within_30_days=len(urgent), notice_deadline_already_passed=len(missed),
        escalation_cost_next_year=inr(float(c["escalation_cost_next_year"].sum())),
        most_urgent_party=str(urgent.sort_values("days_to_notice_deadline").iloc[0]["party"]) if len(urgent) else None,
        most_urgent_days=int(urgent["days_to_notice_deadline"].min()) if len(urgent) else None))
    for _, r in expired.iterrows():
        res.add_flag(RED, f"Contract with {r['party']} ended {abs(int(r['days_to_end']))} days ago and is still listed")
    for _, r in urgent.iterrows():
        res.add_flag(RED if r["days_to_notice_deadline"] <= 14 else AMBER,
                     f"{r['party']}: notice deadline in {int(r['days_to_notice_deadline'])} days ({r['notice_deadline']:%d %b %Y})",
                     f"Contract worth {inr(r['value'])} ends {r['end_date']:%d %b %Y}.")
    for _, r in missed.iterrows():
        if auto.loc[_]:
            res.add_flag(AMBER, f"{r['party']} will auto-renew: the notice date has passed", f"Ends {r['end_date']:%d %b %Y}.")
    if not res.flags:
        res.add_flag(GREEN, "No contract needs action in the next 30 days")
    res.notes.append("Notice deadline = end date minus the notice period recorded on the contract. This is a tracking aid built from the fields you enter; "
                     "it does not read contract documents, and legal terms (cancellation, penalties) need a lawyer's review.")
    return res


def rec_contracts(f, flags):
    r = []
    if f.get("notice_deadline_within_30_days"):
        r.append(f"{f['notice_deadline_within_30_days']} contract notice deadline(s) fall within 30 days; {f['most_urgent_party']} is first "
                 f"in {f['most_urgent_days']} days. Decide to renew, renegotiate or cancel before it passes.")
    if f.get("notice_deadline_already_passed"):
        r.append(f"{f['notice_deadline_already_passed']} contract(s) are past their notice date; check which will auto-renew and what that costs.")
    if f.get("expired_contracts"):
        r.append(f"{f['expired_contracts']} listed contract(s) have already ended; confirm whether work continues without one.")
    r.append(f"{f['ending_within_90_days']} contract(s) worth {f['value_ending_within_90_days']} end within 90 days, and price escalations add about "
             f"{f['escalation_cost_next_year']} next year.")
    return r


# ============================================================================== SupportIQ
SLA_HOURS = {"urgent": 4, "high": 12, "medium": 36, "low": 72}      # assumption: first-resolution targets, editable


def run_support(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("SupportIQ")
    t = data["tickets"].copy()
    sla = {**SLA_HOURS, **ctx.settings.get("sla_hours", {})}
    as_of = ctx.as_of + pd.Timedelta(days=1)
    t = t[t["opened_date"] <= as_of]
    has_closed = "closed_date" in t
    closed_ok = t["closed_date"].notna() & (t["closed_date"] <= as_of) if has_closed else pd.Series(False, index=t.index)
    t["resolved"] = closed_ok
    t["hours_to_resolve"] = np.where(closed_ok, (t["closed_date"] - t["opened_date"]).dt.total_seconds() / 3600, np.nan)
    t["sla_hours"] = t["priority"].map(sla)
    t["age_hours_open"] = np.where(~closed_ok, (as_of - t["opened_date"]).dt.total_seconds() / 3600, np.nan)
    t["breached"] = np.where(closed_ok, t["hours_to_resolve"] > t["sla_hours"], t["age_hours_open"] > t["sla_hours"])
    by_pr = t.groupby("priority").agg(tickets=("ticket_id", "size"), median_hours=("hours_to_resolve", "median"), sla_hours=("sla_hours", "first"),
                                       breach_rate=("breached", "mean")).reindex(["urgent", "high", "medium", "low"])
    res.tables["by_priority"] = by_pr.reset_index()
    cat = t.groupby("category").agg(tickets=("ticket_id", "size"), median_hours=("hours_to_resolve", "median"), breach_rate=("breached", "mean"),
                                    avg_csat=("csat", "mean")).sort_values("tickets", ascending=False)
    cat["share"] = cat["tickets"] / cat["tickets"].sum()
    res.tables["by_category"] = cat.reset_index()
    wk = t.assign(week=t["opened_date"].dt.to_period("W").dt.start_time).groupby("week").agg(tickets=("ticket_id", "size"), breach_rate=("breached", "mean"))
    res.tables["weekly"] = wk.reset_index()
    if "channel" in t:
        res.tables["by_channel"] = t.groupby("channel").agg(tickets=("ticket_id", "size"), median_hours=("hours_to_resolve", "median"),
                                                          avg_csat=("csat", "mean")).reset_index()
    open_t = t[~closed_ok].sort_values("age_hours_open", ascending=False)
    res.tables["open_tickets"] = open_t[["ticket_id", "opened_date", "category", "priority", "age_hours_open", "sla_hours", "breached"]]
    # reply drafts per category (templates; a person reviews and sends)
    drafts = {
        "Delivery delay": "Hi {name}, sorry your order is late. We have checked with the courier and will update you by {date}. If it has not arrived by then we will send a replacement or refund.",
        "Refund status": "Hi {name}, your refund has been initiated and should reach your account in {days} working days. Reply here if it has not arrived after that.",
        "Size or fit": "Hi {name}, sorry the fit was not right. We can arrange a free exchange for a different size; please reply with the size you would like.",
        "Product quality": "Hi {name}, we are sorry about the quality. Please share a photo and we will arrange a replacement or refund right away.",
        "Billing or payment": "Hi {name}, thanks for flagging this. We are checking the payment against your order and will confirm within one working day.",
        "Order change": "Hi {name}, we can update your order if it has not shipped yet. Please confirm the change you need.",
    }
    res.tables["reply_drafts"] = pd.DataFrame([dict(category=k, draft=v) for k, v in drafts.items() if k in set(t["category"])])
    n, n_open = len(t), int((~closed_ok).sum())
    breach = float(t["breached"].mean())
    worst_csat = cat["avg_csat"].dropna().sort_values().head(1)
    top = cat.iloc[0]
    old_open = int((open_t["age_hours_open"] > 24 * 7).sum())
    res.metrics = dict(tickets=n, open_tickets=n_open, sla_breach_rate=breach)
    res.facts = clean_facts(dict(
        tickets=n, open_tickets=n_open, tickets_open_over_7_days=old_open, sla_breach_pct=pct1(breach),
        median_resolution_hours=r1(float(t["hours_to_resolve"].median()), 1), average_csat=r1(float(t["csat"].mean()), 2),
        top_category=str(top.name), top_category_share_pct=pct1(top["share"]),
        lowest_csat_category=str(worst_csat.index[0]) if len(worst_csat) else None, lowest_csat=r1(float(worst_csat.iloc[0]), 2) if len(worst_csat) else None,
        worst_sla_category=str(cat["breach_rate"].idxmax()), worst_sla_category_breach_pct=pct1(cat["breach_rate"].max())))
    if breach > 0.40:
        res.add_flag(RED, f"{breach:.0%} of tickets breach the response target")
    elif breach > 0.25:
        res.add_flag(AMBER, f"{breach:.0%} of tickets breach the response target")
    if old_open:
        res.add_flag(AMBER, f"{old_open} ticket(s) have been open more than 7 days")
    if t["csat"].notna().any() and float(t["csat"].mean()) < 3.8:
        res.add_flag(AMBER, f"Average customer satisfaction is {float(t['csat'].mean()):.1f} out of 5")
    if not res.flags:
        res.add_flag(GREEN, "Support is meeting its targets")
    res.notes.append("Targets (hours to resolve): urgent 4, high 12, medium 36, low 72: assumptions you should set to your real promises. "
                     "Open tickets count as breached once older than their target. CSAT comes only from tickets where the customer answered. "
                     "Reply drafts are templates for a person to review; nothing is sent automatically.")
    return res


def rec_support(f, flags):
    r = [f"{f['sla_breach_pct']}% of tickets miss their response target; the median ticket takes {f['median_resolution_hours']} hours. "
         f"{f['worst_sla_category']} is worst at {f['worst_sla_category_breach_pct']}% breached."]
    r.append(f"{f['top_category']} is {f['top_category_share_pct']}% of all tickets; fixing its root cause (proactive delivery updates, size guides, "
             "clearer refund messages) removes the most work.")
    if f.get("lowest_csat_category"):
        r.append(f"Satisfaction is lowest for {f['lowest_csat_category']} (average rating {f['lowest_csat']}).")
    if f.get("tickets_open_over_7_days"):
        r.append(f"Clear the {f['tickets_open_over_7_days']} ticket(s) open over a week first.")
    return r


# ============================================================================== registration
register(ModuleSpec("PipelineIQ", "Sales operations", "Open deals, stalled deals, win rates and a stage-weighted pipeline value.",
                    ["pipeline"], run_pipeline, rec_pipeline, "B2B", "3b"))
register(ModuleSpec("QuoteIQ", "Sales operations", "Quote win rate, price versus margin, loss reasons and follow-ups.",
                    ["quotes"], run_quotes, rec_quotes, "B2B", "3b"))
register(ModuleSpec("ContractIQ", "Sales operations", "Contract end dates, notice deadlines, auto-renewals and price escalations.",
                    ["contracts"], run_contracts, rec_contracts, "B2B", "3b"))
register(ModuleSpec("SupportIQ", "Customer service", "Ticket volume, response targets, satisfaction and recurring issues.",
                    ["tickets"], run_support, rec_support, "Both", "3a"))
