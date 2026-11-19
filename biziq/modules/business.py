"""Business-model modules: InventoryIQ, ProjectIQ, SubscriptionIQ, AppointmentIQ, RecipeIQ, SurveyIQ."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import AMBER, GREEN, RED, ModuleResult, inr, safe_div
from ..registry import Context, ModuleSpec, register
from ._common import argext, clean_facts, pct1, r1


# ============================================================================== InventoryIQ
def run_inventory(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("InventoryIQ")
    inv = data["inventory"].copy()
    s = data["sales"].copy()
    s = s[s["date"] <= ctx.as_of]
    if inv.empty or s.empty:
        raise ValueError("needs stock and sales")
    s["rev"] = s["quantity"] * s["unit_price"]
    win = 90
    rec = s[s["date"] > ctx.as_of - pd.Timedelta(days=win)]
    units90 = rec.groupby("product")["quantity"].sum()
    daily = rec.groupby(["product", rec["date"].dt.normalize()])["quantity"].sum()
    # daily variation, counting days with no sales as zero
    idx = pd.date_range(ctx.as_of.normalize() - pd.Timedelta(days=win - 1), ctx.as_of.normalize())
    sd = {p: float(g.droplevel(0).reindex(idx, fill_value=0).std()) for p, g in daily.groupby(level=0)}
    rev365 = s[s["date"] > ctx.as_of - pd.Timedelta(days=365)].groupby("product")["rev"].sum()
    t = inv.drop_duplicates("product", keep="last").set_index("product")[[c for c in ("stock_on_hand", "lead_time_days", "avg_daily_demand") if c in inv.columns]].copy()
    t["demand_per_day"] = (units90.reindex(t.index).fillna(0) / win)
    if "avg_daily_demand" in t:
        t["demand_per_day"] = pd.to_numeric(t["avg_daily_demand"], errors="coerce").fillna(t["demand_per_day"])
    t["safety_stock"] = [1.65 * sd.get(p, 0.0) * np.sqrt(max(float(l), 0)) if pd.notna(l) else 0.0 for p, l in zip(t.index, t["lead_time_days"])]
    t["reorder_point"] = t["demand_per_day"] * t["lead_time_days"] + t["safety_stock"]
    t["days_of_cover"] = (t["stock_on_hand"] / t["demand_per_day"].replace(0, np.nan)).round(0)
    r = rev365.reindex(t.index).fillna(0).sort_values(ascending=False)
    cum = r.cumsum() / r.sum() if r.sum() > 0 else r * 0
    t["abc"] = pd.Series(np.where(cum.shift(fill_value=0) < 0.80, "A", np.where(cum.shift(fill_value=0) < 0.95, "B", "C")), index=r.index).reindex(t.index)
    pc = data.get("product_costs")
    t["stock_value"] = (t["stock_on_hand"] * pc.groupby("product")["unit_cost"].last().reindex(t.index)) if pc is not None else np.nan
    t["status"] = np.select(
        [(t["demand_per_day"] == 0) & (t["stock_on_hand"] > 0), t["days_of_cover"] < t["lead_time_days"], t["stock_on_hand"] < t["reorder_point"], t["days_of_cover"] > 180],
        ["no recent sales", "will run out before resupply", "below reorder point", "overstocked"], "ok")
    t["suggested_order_qty"] = np.where(t["status"].isin(["will run out before resupply", "below reorder point"]),
                                        np.maximum(t["reorder_point"] + 30 * t["demand_per_day"] - t["stock_on_hand"], 0).round(0), 0)
    res.tables["products"] = t.round(1).reset_index().sort_values(["abc", "days_of_cover"])
    short = t[t["status"] == "will run out before resupply"]
    low = t[t["status"].isin(["will run out before resupply", "below reorder point"])]
    dead, over = t[t["status"] == "no recent sales"], t[t["status"] == "overstocked"]
    a_risk = low[low["abc"] == "A"]
    tied = float(pd.concat([dead, over])["stock_value"].sum()) if pc is not None else None
    res.facts = clean_facts(dict(
        products=int(len(t)), a_class_products=int((t["abc"] == "A").sum()), below_reorder_point=int(len(low)), will_run_out_before_resupply=int(len(short)),
        a_class_at_risk=int(len(a_risk)), no_recent_sales=int(len(dead)), overstocked=int(len(over)), cash_tied_in_dead_or_overstock=None if tied is None else inr(tied),
        total_stock_value=None if pc is None else inr(float(t["stock_value"].sum())), most_urgent_product=None if short.empty else str(short["days_of_cover"].idxmin()),
        most_urgent_days_of_cover=None if short.empty else r1(float(short["days_of_cover"].min()), 0)))
    if len(a_risk):
        res.add_flag(RED, f"{len(a_risk)} top-selling (A) product(s) will run short: {', '.join(map(str, a_risk.index[:3]))}", "These carry most of your revenue")
    elif len(short):
        res.add_flag(AMBER, f"{len(short)} product(s) will run out before a new order can arrive: {', '.join(map(str, short.index[:3]))}")
    if len(dead):
        res.add_flag(AMBER, f"{len(dead)} product(s) have stock but no sales in 90 days: {', '.join(map(str, dead.index[:3]))}")
    if len(over):
        res.add_flag(AMBER, f"{len(over)} product(s) hold over 180 days of stock: {', '.join(map(str, over.index[:3]))}")
    if not res.flags:
        res.add_flag(GREEN, "Stock levels cover demand with no obvious dead or surplus stock")
    res.notes.append("Demand is units sold per day over the last 90 days (or the average you gave in the stock file). Reorder point = demand x lead time + safety stock (95% service level from day-to-day variation). "
                     "A/B/C ranks products by the last 12 months of revenue: A is the top 80%. Suggested order = back to reorder point plus 30 days of demand. Seasonality and promotions are not modelled, "
                     "and stock value needs your product costs.")
    return res


def rec_inventory(f, flags):
    r = []
    if f.get("will_run_out_before_resupply"):
        r.append(f"Order now for {f.get('most_urgent_product')}: about {f.get('most_urgent_days_of_cover')} days of cover left and the lead time is longer. See the products table for quantities.")
    if f.get("no_recent_sales") or f.get("overstocked"):
        r.append(f"Clear or stop reordering the {f.get('no_recent_sales', 0)} no-sales and {f.get('overstocked', 0)} overstocked product(s)" + (f"; {f['cash_tied_in_dead_or_overstock']} is tied up in them." if f.get("cash_tied_in_dead_or_overstock") else "."))
    return r or ["Stock is balanced; re-check weekly."]


# ============================================================================== ProjectIQ
def run_project(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ProjectIQ")
    p = data["projects"].copy()
    if p.empty:
        raise ValueError("needs at least one project")
    status = p["status"].astype("string").str.lower().fillna("") if "status" in p else pd.Series("", index=p.index)
    done = status.eq("completed") | (p["completed_date"].notna() if "completed_date" in p else False)
    for c in ("billed", "cost_to_date", "hours_budget", "hours_spent", "percent_complete"):
        p[c] = pd.to_numeric(p[c], errors="coerce") if c in p else np.nan
    p["done"] = done
    p["pc"] = np.where(p["done"], 100.0, p["percent_complete"])
    # forecast the final outcome by progress so far (only once at least 10% done: earlier ratios are noise)
    ok = (~p["done"]) & (p["pc"] >= 10)
    p["forecast_hours"] = np.where(ok, p["hours_spent"] / (p["pc"] / 100), np.nan)
    p["forecast_cost"] = np.where(ok, p["cost_to_date"] / (p["pc"] / 100), np.nan)
    p["hours_overrun_pct"] = (100 * (p["forecast_hours"] / p["hours_budget"].replace(0, np.nan) - 1)).round(0)
    p["cost_overrun_pct"] = (100 * (p["forecast_cost"] / p["budget"].replace(0, np.nan) - 1)).round(0)
    p["margin_pct"] = np.where(p["done"], (100 * (p["billed"] - p["cost_to_date"]) / p["billed"].replace(0, np.nan)), np.nan)
    p["unbilled_work"] = np.where(~p["done"], np.maximum(p["budget"] * p["pc"] / 100 - p["billed"].fillna(0), 0), 0.0)
    act = p[~p["done"]]
    late = act[act["due_date"].notna() & (act["due_date"] < ctx.as_of)] if "due_date" in act else act.iloc[0:0]
    p["days_late"] = np.where(p["project_id"].isin(late["project_id"]), (ctx.as_of - p["due_date"]).dt.days, np.nan) if "due_date" in p else np.nan
    over = act[(act["project_id"].isin(p.loc[(p["cost_overrun_pct"] > 20) | (p["hours_overrun_pct"] > 20), "project_id"]))]
    res.tables["projects"] = p[["project_id", "client", "status", "budget", "billed", "cost_to_date", "percent_complete", "hours_overrun_pct", "cost_overrun_pct", "margin_pct", "unbilled_work", "days_late"]].round(1)
    byc = p.groupby("client").agg(projects=("project_id", "size"), budget=("budget", "sum"), billed=("billed", "sum"), cost=("cost_to_date", "sum")).reset_index()
    byc["margin_pct"] = (100 * (byc["billed"] - byc["cost"]) / byc["billed"].replace(0, np.nan)).round(1)
    res.tables["by_client"] = byc.sort_values("budget", ascending=False)
    cd = p[p["done"] & p["billed"].notna() & (p["billed"] > 0) & p["cost_to_date"].notna()]
    avg_margin = 100 * (cd["billed"].sum() - cd["cost_to_date"].sum()) / cd["billed"].sum() if len(cd) else None
    res.facts = clean_facts(dict(
        projects=int(len(p)), active_projects=int(len(act)), completed_projects=int(p["done"].sum()),
        projects_forecast_to_overrun=int(len(over)), worst_project=None if over.empty else str(p.loc[over.index, "cost_overrun_pct"].idxmax() and p.loc[p.loc[over.index, "cost_overrun_pct"].idxmax(), "project_id"]),
        worst_forecast_cost_overrun_pct=None if over.empty else r1(float(p.loc[over.index, "cost_overrun_pct"].max()), 0),
        late_projects=int(len(late)), unbilled_work_value=inr(float(p["unbilled_work"].sum())), completed_margin_pct=r1(avg_margin),
        active_value=inr(float(act["budget"].sum()))))
    if len(over):
        top = p.loc[over.index].sort_values("cost_overrun_pct", ascending=False).iloc[0]
        res.add_flag(RED if top["cost_overrun_pct"] > 40 else AMBER, f"{len(over)} active project(s) are on track to overrun, worst {top['project_id']} by {top['cost_overrun_pct']:.0f}% on cost", "Based on cost so far against progress")
    if len(late):
        res.add_flag(AMBER, f"{len(late)} active project(s) are past their due date")
    if p["unbilled_work"].sum() > 0.15 * max(float(act["budget"].sum()), 1):
        res.add_flag(AMBER, f"{inr(float(p['unbilled_work'].sum()))} of finished work is not yet billed", "Cash you have earned but not asked for")
    if avg_margin is not None and avg_margin < 20:
        res.add_flag(AMBER, f"Completed projects earned a {avg_margin:.0f}% margin")
    if not res.flags:
        res.add_flag(GREEN, "Projects are on budget, on time and billed up to date")
    res.notes.append("Forecast final cost and hours = spent so far divided by the share complete, only for projects at least 10% done; progress is as reported in your file and early percentages are unreliable. "
                     "Unbilled work = share complete x budget minus billed. Margin on completed projects = billed minus recorded cost, so it includes only the costs you tracked.")
    return res


def rec_project(f, flags):
    r = []
    if f.get("projects_forecast_to_overrun"):
        r.append(f"Re-scope or re-price the {f['projects_forecast_to_overrun']} overrunning project(s) now: worst is {f.get('worst_project')} at {f.get('worst_forecast_cost_overrun_pct')}% over budget on cost. Agree a change order before more hours go in.")
    if f.get("late_projects"):
        r.append(f"Reset dates with the clients on the {f['late_projects']} late project(s).")
    if f.get("unbilled_work_value") and "0" != str(f.get("unbilled_work_value")).strip("Rs 0."):
        r.append(f"Invoice the work already done: {f['unbilled_work_value']} is earned but unbilled.")
    return r or ["Projects look healthy; keep logging hours and progress weekly."]


# ============================================================================== SubscriptionIQ
def _active_at(s: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    return s[(s["start_date"] <= t) & (s["end_date"].isna() | (s["end_date"] > t))]


def run_subscription(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("SubscriptionIQ")
    s = data["subscriptions"].copy()
    if "end_date" not in s.columns:
        s["end_date"] = pd.NaT
    s = s[s["start_date"] <= ctx.as_of]
    s.loc[s["end_date"] > ctx.as_of, "end_date"] = pd.NaT
    if len(s) < 10:
        raise ValueError("needs at least 10 subscriptions")
    ends = pd.date_range(end=(ctx.as_of + pd.offsets.MonthEnd(0)) if ctx.as_of.is_month_end else ctx.as_of - pd.offsets.MonthEnd(1), periods=13, freq="ME")
    rows = []
    for i, t in enumerate(ends):
        act = _active_at(s, t)
        prev = ends[i - 1] if i else t - pd.offsets.MonthEnd(1)
        new = s[(s["start_date"] > prev) & (s["start_date"] <= t)]
        churn = s[(s["end_date"] > prev) & (s["end_date"] <= t)]
        base = _active_at(s, prev)
        rows.append(dict(month=t.strftime("%Y-%m"), active=len(act), mrr=float(act["mrr"].sum()), new_mrr=float(new["mrr"].sum()), churned_mrr=float(churn["mrr"].sum()),
                         logo_churn_pct=round(100 * len(churn) / len(base), 2) if len(base) else np.nan))
    tr = pd.DataFrame(rows)
    res.tables["monthly"] = tr.iloc[1:].reset_index(drop=True)
    cur = tr.iloc[-1]
    last3 = tr.iloc[-3:]
    ch3 = float(last3["logo_churn_pct"].mean())
    mrr_3 = float(tr.iloc[-4]["mrr"])
    growth3 = safe_div(float(cur["mrr"]) - mrr_3, mrr_3)
    # per plan churn over the last 3 months
    cut = ends[-4]
    plan_rows = []
    for plan, g in s.groupby("plan"):
        base = _active_at(g, cut)
        lost = g[(g["end_date"] > cut) & (g["end_date"] <= ends[-1])]
        plan_rows.append(dict(plan=plan, active=len(_active_at(g, ends[-1])), mrr=float(_active_at(g, ends[-1])["mrr"].sum()), lost_3m=len(lost),
                              monthly_logo_churn_pct=round(100 * (len(lost) / len(base)) / 3, 2) if len(base) else np.nan))
    pl = pd.DataFrame(plan_rows).sort_values("mrr", ascending=False)
    res.tables["by_plan"] = pl
    act = _active_at(s, ends[-1])
    top5 = act.groupby("customer")["mrr"].sum().sort_values(ascending=False).head(5)
    worst = pl[pl["active"] >= 10].sort_values("monthly_logo_churn_pct", ascending=False).head(1)
    res.facts = clean_facts(dict(
        current_mrr=inr(float(cur["mrr"])), annualised_run_rate=inr(float(cur["mrr"]) * 12), active_subscriptions=int(cur["active"]), mrr_change_3m_pct=pct1(growth3),
        monthly_logo_churn_pct_3m=r1(ch3, 2), annualised_logo_churn_pct=r1(100 * (1 - (1 - ch3 / 100) ** 12), 1) if not pd.isna(ch3) else None,
        new_mrr_last_3m=inr(float(last3["new_mrr"].sum())), churned_mrr_last_3m=inr(float(last3["churned_mrr"].sum())),
        highest_churn_plan=None if worst.empty else str(worst.iloc[0]["plan"]), highest_churn_plan_pct=None if worst.empty else r1(float(worst.iloc[0]["monthly_logo_churn_pct"]), 2),
        top5_customers_share_pct=pct1(safe_div(float(top5.sum()), float(cur["mrr"])))))
    if not pd.isna(ch3) and ch3 >= 6:
        res.add_flag(RED, f"Monthly churn is {ch3:.1f}% of subscribers", f"About {100 * (1 - (1 - ch3 / 100) ** 12):.0f}% a year")
    elif not pd.isna(ch3) and ch3 >= 3:
        res.add_flag(AMBER, f"Monthly churn is {ch3:.1f}% of subscribers")
    if len(worst) and not pd.isna(ch3) and ch3 > 0 and float(worst.iloc[0]["monthly_logo_churn_pct"]) >= 2 * ch3 and float(worst.iloc[0]["monthly_logo_churn_pct"]) >= 3:
        res.add_flag(AMBER, f"The {worst.iloc[0]['plan']} plan loses {worst.iloc[0]['monthly_logo_churn_pct']:.1f}% of subscribers a month, over twice the average")
    if growth3 is not None and not pd.isna(growth3) and growth3 < 0:
        res.add_flag(RED, f"Recurring revenue fell {abs(growth3):.0%} in 3 months")
    if safe_div(float(top5.sum()), float(cur["mrr"])) > 0.4:
        res.add_flag(AMBER, f"Five customers make up {safe_div(float(top5.sum()), float(cur['mrr'])):.0%} of recurring revenue")
    if not res.flags:
        res.add_flag(GREEN, "Recurring revenue is growing and churn is low")
    res.notes.append("MRR is the sum of the monthly amount for subscriptions active at each month end. A plan change should appear as one subscription ending and a new one starting, otherwise it reads as churn plus new business; "
                     "expansion revenue is not tracked separately. Churn is the share of subscribers active at the start of the month who ended during it. Needs at least 4 months of history to compare.")
    return res


def rec_subscription(f, flags):
    r = []
    if f.get("highest_churn_plan_pct") and f["highest_churn_plan_pct"] >= 3:
        r.append(f"Find out why {f['highest_churn_plan']} subscribers leave ({f['highest_churn_plan_pct']}% a month): ask the last few who cancelled, and look at onboarding in their first month.")
    if f.get("monthly_logo_churn_pct_3m") and f["monthly_logo_churn_pct_3m"] >= 3:
        r.append(f"At {f['monthly_logo_churn_pct_3m']}% monthly churn, retention is worth more than new sales: offer annual plans and a save offer on cancellation.")
    return r or ["Subscriptions look healthy; track churn by plan monthly."]


# ============================================================================== AppointmentIQ
def run_appointment(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("AppointmentIQ")
    a = data["appointments"].copy()
    a = a[a["date"] <= ctx.as_of]
    a["st"] = a["status"].astype("string").str.lower().str.strip().str.replace(r"[\s\-]+", "_", regex=True).replace({"noshow": "no_show", "did_not_attend": "no_show", "dna": "no_show", "canceled": "cancelled", "done": "completed", "attended": "completed"})
    a = a[a["st"].isin(["completed", "no_show", "cancelled"])]
    if len(a) < 50:
        raise ValueError("needs at least 50 appointments with status completed, no_show or cancelled")
    price = pd.to_numeric(a["price"], errors="coerce") if "price" in a else pd.Series(np.nan, index=a.index)
    a["price_"] = price
    a["weekday"] = a["date"].dt.day_name()
    a["hour"] = pd.to_numeric(a["start_time"].astype("string").str.slice(0, 2), errors="coerce") if "start_time" in a else np.nan
    tot = len(a)
    ns_rate = float((a["st"] == "no_show").mean())
    cx_rate = float((a["st"] == "cancelled").mean())
    lost = float(a.loc[a["st"] == "no_show", "price_"].sum()) if price.notna().any() else None
    done = a[a["st"] == "completed"]

    def by(col):
        g = a.groupby(col).agg(appointments=("st", "size"), no_show_pct=("st", lambda x: round(100 * (x == "no_show").mean(), 1)), cancelled_pct=("st", lambda x: round(100 * (x == "cancelled").mean(), 1)))
        return g.reset_index()
    wd = by("weekday")
    wd["order"] = wd["weekday"].map({d: i for i, d in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])})
    res.tables["by_weekday"] = wd.sort_values("order").drop(columns="order")
    res.tables["by_service"] = by("service")
    if a["hour"].notna().any():
        res.tables["by_hour"] = by("hour")
    clients = done.groupby("customer").size()
    repeat = float((clients >= 2).mean()) if len(clients) else None
    revs = done.groupby("service")["price_"].sum().sort_values(ascending=False) if price.notna().any() else pd.Series(dtype=float)
    bad_day = wd[wd["appointments"] >= 30].sort_values("no_show_pct", ascending=False).head(1)
    res.facts = clean_facts(dict(
        appointments=int(tot), completed_pct=pct1(len(done) / tot), no_show_rate_pct=pct1(ns_rate), cancellation_rate_pct=pct1(cx_rate),
        revenue_lost_to_no_shows=None if lost is None else inr(lost), highest_no_show_day=None if bad_day.empty else str(bad_day.iloc[0]["weekday"]),
        highest_no_show_day_pct=None if bad_day.empty else float(bad_day.iloc[0]["no_show_pct"]), repeat_client_pct=pct1(repeat),
        top_service_by_revenue=None if revs.empty else str(revs.index[0])))
    if ns_rate >= 0.15:
        res.add_flag(RED if ns_rate >= 0.25 else AMBER, f"{ns_rate:.0%} of appointments are no-shows", f"About {inr(lost)} of booked revenue lost" if lost is not None else "")
    if len(bad_day) and ns_rate > 0 and float(bad_day.iloc[0]["no_show_pct"]) >= 1.5 * 100 * ns_rate and float(bad_day.iloc[0]["no_show_pct"]) >= 15:
        res.add_flag(AMBER, f"{bad_day.iloc[0]['weekday']} no-shows are {bad_day.iloc[0]['no_show_pct']:.0f}% versus {ns_rate:.0%} overall")
    if cx_rate >= 0.15:
        res.add_flag(AMBER, f"{cx_rate:.0%} of appointments are cancelled")
    if not res.flags:
        res.add_flag(GREEN, "Few no-shows or cancellations")
    res.notes.append("Rates use appointments marked completed, no-show or cancelled; future and other statuses are ignored. Lost revenue = price of no-shows, so it ignores any deposit kept and slots refilled. "
                     "A weekday is flagged only with at least 30 appointments. Results are not broken down by staff member: a rate beside a name says more about who books them and which clients they see than about the person.")
    return res


def rec_appointment(f, flags):
    r = []
    if f.get("no_show_rate_pct") and f["no_show_rate_pct"] >= 15:
        r.append(f"Cut no-shows ({f['no_show_rate_pct']}%): send a reminder 24 hours before, ask for a small deposit on long slots, and double-book carefully on {f.get('highest_no_show_day', 'the worst day')}.")
    if f.get("repeat_client_pct") is not None and f["repeat_client_pct"] < 40:
        r.append("Few clients come back: book the next visit before they leave.")
    return r or ["Attendance is healthy; keep the reminder routine."]


# ============================================================================== RecipeIQ
def run_recipe(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("RecipeIQ")
    r = data["recipes"].copy()
    if "selling_price" not in r.columns or pd.to_numeric(r["selling_price"], errors="coerce").notna().sum() == 0:
        raise ValueError("needs selling_price for the dishes")
    r["line_cost"] = r["qty_per_portion"] * r["unit_cost"]
    price = pd.to_numeric(r["selling_price"], errors="coerce")
    sold = pd.to_numeric(r["portions_sold"], errors="coerce") if "portions_sold" in r else pd.Series(np.nan, index=r.index)
    d = r.assign(price=price, sold=sold).groupby("dish").agg(cost=("line_cost", "sum"), price=("price", "first"), sold=("sold", "first"), ingredients=("ingredient", "nunique"))
    d = d[d["price"] > 0].copy()
    if d.empty:
        raise ValueError("needs selling_price above zero")
    target = float(ctx.settings.get("target_food_cost_pct", 35))
    d["food_cost_pct"] = (100 * d["cost"] / d["price"]).round(1)
    d["margin_per_portion"] = (d["price"] - d["cost"]).round(1)
    have_sales = d["sold"].notna().all() and d["sold"].sum() > 0
    if have_sales:
        d["contribution"] = d["margin_per_portion"] * d["sold"]
        n = len(d)
        avg_margin = float((d["margin_per_portion"] * d["sold"]).sum() / d["sold"].sum())
        pop = d["sold"] >= 0.7 * d["sold"].sum() / n                                    # the standard menu-engineering rule: popular if at least 70% of an equal share
        high = d["margin_per_portion"] >= avg_margin
        d["class"] = np.select([pop & high, pop & ~high, ~pop & high], ["Star", "Plowhorse", "Puzzle"], "Dog")
    res.tables["dishes"] = d.round(1).reset_index().sort_values("food_cost_pct", ascending=False)
    ing = r.assign(sold=sold.fillna(1)).assign(spend=lambda x: x["line_cost"] * x["sold"]).groupby("ingredient")["spend"].sum().sort_values(ascending=False)
    res.tables["ingredient_cost_share"] = (100 * ing / ing.sum()).round(1).rename("share_of_food_cost_pct").reset_index().head(15)
    over = d[d["food_cost_pct"] > target]
    below = d[d["cost"] > d["price"]]
    w_fc = float(d["cost"].mul(d["sold"]).sum() / d["price"].mul(d["sold"]).sum() * 100) if have_sales else float(d["cost"].sum() / d["price"].sum() * 100)
    top_ing = ing.index[0] if len(ing) else None
    cls = d["class"].value_counts() if have_sales else pd.Series(dtype=int)
    ploughs = d[d["class"] == "Plowhorse"] if have_sales else d.iloc[0:0]
    res.facts = clean_facts(dict(
        dishes=int(len(d)), average_food_cost_pct=r1(w_fc), target_food_cost_pct=target, dishes_over_target=int(len(over)), dishes_priced_below_cost=int(len(below)),
        highest_food_cost_dish=str(d["food_cost_pct"].idxmax()), highest_food_cost_pct=float(d["food_cost_pct"].max()),
        stars=int(cls.get("Star", 0)) if have_sales else None, plowhorses=int(cls.get("Plowhorse", 0)) if have_sales else None, puzzles=int(cls.get("Puzzle", 0)) if have_sales else None, dogs=int(cls.get("Dog", 0)) if have_sales else None,
        costliest_ingredient=None if top_ing is None else str(top_ing), costliest_ingredient_share_pct=None if top_ing is None else r1(float(100 * ing.iloc[0] / ing.sum())),
        cost_impact_if_top_ingredient_rises_10pct=None if top_ing is None else inr(float(ing.iloc[0] * 0.10)),
        biggest_plowhorse=None if ploughs.empty else str(ploughs["sold"].idxmax())))
    for dish, row in below.iterrows():
        res.add_flag(RED, f"{dish} costs more to make than it sells for", f"Cost {inr(float(row['cost']))} against price {inr(float(row['price']))}")
    for dish, row in over.sort_values("food_cost_pct", ascending=False).head(3).iterrows():
        if dish not in below.index:
            res.add_flag(AMBER, f"{dish} runs a {row['food_cost_pct']:.0f}% food cost against a {target:.0f}% target")
    if len(ploughs):
        res.add_flag(AMBER, f"{len(ploughs)} popular dish(es) earn below-average margin: {', '.join(map(str, ploughs.sort_values('sold', ascending=False).index[:3]))}", "Small price or portion changes here move the most money")
    if top_ing is not None and ing.iloc[0] / ing.sum() > 0.25:
        res.add_flag(AMBER, f"{top_ing} is {100 * ing.iloc[0] / ing.sum():.0f}% of food cost", "A price rise there hits hard")
    if not res.flags:
        res.add_flag(GREEN, "Dishes are priced above target and no ingredient dominates cost")
    res.notes.append("Dish cost = sum of quantity per portion x ingredient unit cost; it leaves out labour, gas, packaging and wastage, which usually add 5 to 15%. Menu classes follow the standard menu-engineering rule: "
                     "popular = at least 70% of an equal share of portions sold; high margin = at or above the sales-weighted average margin per portion. Needs portions sold for the classes and the ingredient shares.")
    return res


def rec_recipe(f, flags):
    r = []
    if f.get("dishes_priced_below_cost"):
        r.append(f"Reprice or drop the {f['dishes_priced_below_cost']} dish(es) that cost more than they sell for.")
    if f.get("plowhorses"):
        r.append(f"Lift the margin on {f.get('biggest_plowhorse')}, your best seller with below-average margin: a small price rise, a smaller portion, or a cheaper ingredient mix.")
    if f.get("dishes_over_target"):
        r.append(f"{f['dishes_over_target']} dish(es) exceed your {f.get('target_food_cost_pct'):.0f}% food-cost target; start with {f.get('highest_food_cost_dish')} at {f.get('highest_food_cost_pct')}%.")
    if f.get("costliest_ingredient_share_pct") and f["costliest_ingredient_share_pct"] > 25:
        r.append(f"{f['costliest_ingredient']} is {f['costliest_ingredient_share_pct']}% of food cost: line up a second supplier or fix a price.")
    return r or ["Menu economics look sound; refresh ingredient prices monthly."]


# ============================================================================== SurveyIQ
THEMES = {
    "Delivery": ["deliver", "late", "shipping", "courier", "delay"], "Price": ["price", "expensive", "costly", "overpriced", "cheaper"],
    "Support": ["support", "service desk", "reply", "response", "slow to", "call back", "rude"], "Quality": ["quality", "defect", "broken", "damaged", "stale", "taste"],
    "Staff": ["staff", "friendly", "helpful", "behaviour", "behavior"], "App or website": ["app", "website", "crash", "bug", "login", "checkout"],
    "Packaging": ["packag", "leak", "box"], "Refund": ["refund", "return", "exchange"],
}


def _nps(x: pd.Series) -> float:
    return float(100 * ((x >= 9).mean() - (x <= 6).mean()))


def run_survey(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("SurveyIQ")
    s = data["survey_responses"].copy()
    s = s[(s["date"] <= ctx.as_of)]
    s["score"] = pd.to_numeric(s["score"], errors="coerce")
    s = s[s["score"].between(0, 10)]
    if len(s) < 30:
        raise ValueError("needs at least 30 survey responses with a score from 0 to 10")
    nps = _nps(s["score"])
    prom, det = float((s["score"] >= 9).mean()), float((s["score"] <= 6).mean())
    cut = ctx.as_of - pd.Timedelta(days=60)
    r60, prior = s[s["date"] > cut], s[(s["date"] <= cut) & (s["date"] > cut - pd.Timedelta(days=60))]
    nps60 = _nps(r60["score"]) if len(r60) >= 30 else None
    npsp = _nps(prior["score"]) if len(prior) >= 30 else None
    change = None if nps60 is None or npsp is None else nps60 - npsp
    mo = s.assign(month=s["date"].dt.to_period("M").astype(str)).groupby("month").agg(responses=("score", "size"), nps=("score", _nps), average_score=("score", "mean")).round(1).reset_index()
    res.tables["by_month"] = mo
    segs = {}
    for col in ("segment", "channel"):
        if col in s.columns and s[col].notna().any():
            g = s.groupby(col).agg(responses=("score", "size"), nps=("score", _nps)).round(1).reset_index()
            res.tables[f"by_{col}"] = g
            segs[col] = g[g["responses"] >= 30]
    theme_rows = []
    if "comment" in s.columns and s["comment"].notna().any():
        txt = s["comment"].astype("string").str.lower().fillna("")
        dm = s["score"] <= 6
        for th, kws in THEMES.items():
            hit = txt.apply(lambda t: any(k in t for k in kws))
            theme_rows.append(dict(theme=th, mentions=int(hit.sum()), among_detractors=int((hit & dm).sum()), among_promoters=int((hit & (s["score"] >= 9)).sum())))
        th_df = pd.DataFrame(theme_rows).sort_values("among_detractors", ascending=False)
        res.tables["themes_in_comments"] = th_df
        top_th = th_df.iloc[0] if th_df.iloc[0]["among_detractors"] > 0 else None
        det_n = int(dm.sum())
    else:
        top_th, det_n = None, 0
    worst_seg = None
    if "segment" in segs and len(segs["segment"]) >= 2:
        worst_seg = segs["segment"].sort_values("nps").iloc[0]
    res.facts = clean_facts(dict(
        responses=int(len(s)), nps=r1(nps, 0), promoters_pct=pct1(prom), detractors_pct=pct1(det), average_score=r1(float(s["score"].mean()), 2),
        nps_last_60_days=None if nps60 is None else r1(nps60, 0), nps_previous_60_days=None if npsp is None else r1(npsp, 0), nps_change_points=None if change is None else r1(change, 0),
        weakest_segment=None if worst_seg is None else str(worst_seg.iloc[0]), weakest_segment_nps=None if worst_seg is None else r1(float(worst_seg["nps"]), 0),
        top_detractor_theme=None if top_th is None else str(top_th["theme"]), top_detractor_theme_share_pct=None if top_th is None or det_n == 0 else r1(100 * top_th["among_detractors"] / det_n)))
    if change is not None and change <= -8:
        res.add_flag(RED if change <= -15 else AMBER, f"NPS fell {abs(change):.0f} points: {npsp:.0f} to {nps60:.0f} over the last 60 days versus the 60 before")
    if worst_seg is not None and float(worst_seg["nps"]) <= nps - 15:
        res.add_flag(AMBER, f"{worst_seg.iloc[0]} scores {worst_seg['nps']:.0f}, {nps - float(worst_seg['nps']):.0f} points below overall")
    if top_th is not None and det_n >= 20 and top_th["among_detractors"] / det_n >= 0.30:
        res.add_flag(AMBER, f"{top_th['theme']} comes up in {100 * top_th['among_detractors'] / det_n:.0f}% of detractor comments")
    if nps < 0:
        res.add_flag(RED, f"More detractors than promoters (NPS {nps:.0f})")
    if not res.flags:
        res.add_flag(GREEN, f"NPS is {nps:.0f} and steady")
    res.notes.append("NPS = % of scores 9-10 minus % of scores 0-6. Themes come from keyword matching in comments, which is rough: it counts mentions, it does not understand sentiment or sarcasm. "
                     "The two 60-day windows are only compared with at least 30 responses each. Segments are shown only with 30 responses. Responses are from people who chose to answer, so quiet unhappy customers are under-counted.")
    return res


def rec_survey(f, flags):
    r = []
    if f.get("nps_change_points") is not None and f["nps_change_points"] <= -8:
        r.append(f"Satisfaction dropped {abs(f['nps_change_points']):.0f} points recently; ask what changed in the last two months (suppliers, staff, process).")
    if f.get("top_detractor_theme") and (f.get("top_detractor_theme_share_pct") or 0) >= 30:
        r.append(f"Fix {f['top_detractor_theme'].lower()} first: it appears in {f['top_detractor_theme_share_pct']}% of detractor comments.")
    if f.get("weakest_segment") and f.get("nps") is not None and f.get("weakest_segment_nps", 100) <= f["nps"] - 15:
        r.append(f"Look at {f['weakest_segment']} separately: it trails the rest by {f['nps'] - f['weakest_segment_nps']:.0f} points.")
    return r or ["Customers are broadly happy; keep surveying so a change shows up early."]


register(ModuleSpec("InventoryIQ", "Operations and industry", "Reorder points, ABC ranking, dead stock and stock-outs from sales.", ["inventory", "sales"], run_inventory, rec_inventory, audience="Both", phase="4"))
register(ModuleSpec("ProjectIQ", "Operations and industry", "Project profitability, overruns, late work and unbilled effort.", ["projects"], run_project, rec_project, audience="B2B", phase="4"))
register(ModuleSpec("SubscriptionIQ", "Customers and channels", "Recurring revenue, churn and retention by plan.", ["subscriptions"], run_subscription, rec_subscription, audience="Both", phase="4"))
register(ModuleSpec("AppointmentIQ", "Customers and channels", "No-shows, cancellations and repeat clients for booking businesses.", ["appointments"], run_appointment, rec_appointment, audience="B2C", phase="4"))
register(ModuleSpec("RecipeIQ", "Operations and industry", "Dish costs, food-cost percentage and a star / plowhorse / puzzle / dog menu grid.", ["recipes"], run_recipe, rec_recipe, audience="B2C", phase="4"))
register(ModuleSpec("SurveyIQ", "Customers and channels", "NPS trend, weak segments and what unhappy customers mention.", ["survey_responses"], run_survey, rec_survey, audience="Both", phase="4"))
