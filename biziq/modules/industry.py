"""Operations modules: LogisticsIQ, ProductionIQ, MaintenanceIQ, TenderIQ, ESGIQ."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import AMBER, GREEN, RED, ModuleResult, inr, safe_div
from ..registry import Context, ModuleSpec, register
from ._common import clean_facts, pct1, r1


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df else pd.Series(0.0, index=df.index)


# ============================================================================== LogisticsIQ
def run_logistics(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("LogisticsIQ")
    s = data["shipments"].copy()
    s = s[s["ship_date"] <= ctx.as_of]
    s["status"] = s["status"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    s["delivered_flag"] = s["status"] == "delivered"
    s["late_days"] = (s["delivered_date"] - s["promised_date"]).dt.days
    s["on_time"] = s["delivered_flag"] & (s["late_days"] <= 0)
    s["rto"] = s["status"] == "rto"
    s["lost"] = s["status"] == "lost"
    s["overdue_open"] = (s["status"] == "in_transit") & (s["promised_date"] < ctx.as_of)
    s["cod_"] = _n(s, "cod_amount")
    s["cost_"] = _n(s, "cost")

    def roll(by):
        g = s.groupby(by).agg(shipments=("shipment_id", "size"), delivered=("delivered_flag", "sum"), on_time=("on_time", "sum"), rto=("rto", "sum"),
                              lost=("lost", "sum"), cost=("cost_", "mean"), cod_stuck_in_rto=("cod_", lambda x: float(x[s.loc[x.index, "rto"]].sum())))
        g["on_time_pct"] = 100 * g["on_time"] / g["delivered"].replace(0, np.nan)
        g["rto_pct"] = 100 * g["rto"] / g["shipments"]
        g["lost_pct"] = 100 * g["lost"] / g["shipments"]
        g["cost_per_shipment"] = g["cost"].round(1)
        return g.drop(columns="cost").round(1).reset_index()
    car = roll("carrier")
    res.tables["carriers"] = car.sort_values("on_time_pct")
    if "zone" in s and s["zone"].notna().any():
        res.tables["zones"] = roll("zone").sort_values("rto_pct", ascending=False)
        res.tables["carrier_by_zone"] = roll(["carrier", "zone"]).sort_values("rto_pct", ascending=False)
    res.tables["overdue_in_transit"] = s[s["overdue_open"]].sort_values("promised_date")[[c for c in ("shipment_id", "carrier", "ship_date", "promised_date", "cod_amount") if c in s]].head(200)
    done = s[s["delivered_flag"]]
    on_time = safe_div(float(s["on_time"].sum()), len(done))
    rto = safe_div(float(s["rto"].sum()), len(s))
    big = car[car["shipments"] >= 50]
    best = big.loc[big["on_time_pct"].idxmax()] if len(big) else None
    worst = big.loc[big["on_time_pct"].idxmin()] if len(big) else None
    res.facts = clean_facts(dict(
        shipments=len(s), on_time_delivery_pct=pct1(on_time), rto_pct=pct1(rto), lost_pct=pct1(safe_div(float(s["lost"].sum()), len(s))),
        average_cost_per_shipment=r1(float(s["cost_"].mean()), 0), cod_value_in_rto=inr(float(s.loc[s["rto"], "cod_"].sum())),
        in_transit_past_promise=int(s["overdue_open"].sum()),
        best_carrier=str(best["carrier"]) if best is not None else None, best_carrier_on_time_pct=r1(float(best["on_time_pct"])) if best is not None else None,
        worst_carrier=str(worst["carrier"]) if worst is not None else None, worst_carrier_on_time_pct=r1(float(worst["on_time_pct"])) if worst is not None else None,
        worst_carrier_rto_pct=r1(float(worst["rto_pct"])) if worst is not None else None))
    if worst is not None and best is not None and best["on_time_pct"] - worst["on_time_pct"] > 10:
        res.add_flag(RED if worst["on_time_pct"] < 75 else AMBER, f"{worst['carrier']} delivers on time only {worst['on_time_pct']:.0f}% of the time, against {best['on_time_pct']:.0f}% for {best['carrier']}")
    if pd.notna(rto) and rto > 0.12:
        res.add_flag(AMBER if rto < 0.2 else RED, f"{rto:.0%} of shipments come back undelivered (return to origin)")
    if s["overdue_open"].sum():
        res.add_flag(AMBER, f"{int(s['overdue_open'].sum())} shipments are still in transit past their promised date")
    if not res.flags:
        res.add_flag(GREEN, "Carriers are delivering on time with low returns")
    res.notes.append("On time = delivered on or before the promised date. Return to origin (RTO) rate is shown per carrier and zone because remote zones and "
                     "cash-on-delivery orders fail more often whichever carrier is used; compare carriers within the same zone before switching.")
    return res


def rec_logistics(f, flags):
    r = []
    if f.get("worst_carrier") and f.get("best_carrier") and f["worst_carrier"] != f["best_carrier"]:
        r.append(f"Shift volume from {f['worst_carrier']} ({f.get('worst_carrier_on_time_pct')}% on time, {f.get('worst_carrier_rto_pct')}% returned) toward {f['best_carrier']} on the zones where it is comparable; check the carrier-by-zone table first.")
    if (f.get("rto_pct") or 0) > 12:
        r.append("Confirm cash-on-delivery orders by call or WhatsApp before dispatch; RTO is where most delivery cost is lost.")
    if f.get("in_transit_past_promise"):
        r.append(f"Escalate the {f['in_transit_past_promise']} overdue shipments with the carrier; the list is in the overdue table.")
    return r or ["Delivery performance is steady; review monthly."]


# ============================================================================== ProductionIQ
def run_production(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ProductionIQ")
    p = data["production"].copy()
    p = p[p["date"] <= ctx.as_of]
    if pd.to_numeric(p["planned_units"], errors="coerce").fillna(0).sum() <= 0:
        raise ValueError("needs planned units above zero in the production data")
    p["down"] = _n(p, "downtime_min")
    p["defects"] = _n(p, "defective_units")
    p["mins"] = pd.to_numeric(p["planned_minutes"], errors="coerce").fillna(480) if "planned_minutes" in p else 480
    g = p.groupby("line").agg(days=("date", "nunique"), planned=("planned_units", "sum"), produced=("produced_units", "sum"), defects=("defects", "sum"),
                              downtime_min=("down", "sum"), planned_min=("mins", "sum"))
    g["availability"] = (1 - g["downtime_min"] / g["planned_min"]).clip(0, 1)
    g["performance"] = (g["produced"] / g["planned"].replace(0, np.nan)).clip(0, 1.2)
    g["quality"] = (1 - g["defects"] / g["produced"].replace(0, np.nan)).clip(0, 1)
    g["oee_style"] = g["availability"] * g["performance"].clip(upper=1) * g["quality"]
    g["scrap_pct"] = 100 * (1 - g["quality"])
    res.tables["lines"] = (g.assign(availability=g["availability"] * 100, performance=g["performance"] * 100, quality=g["quality"] * 100, oee_style=g["oee_style"] * 100)
                           .round(1).reset_index().sort_values("oee_style"))
    daily = p.groupby(["date", "line"]).agg(produced=("produced_units", "sum"), planned=("planned_units", "sum")).reset_index()
    daily["attainment_pct"] = 100 * daily["produced"] / daily["planned"].replace(0, np.nan)
    res.tables["daily_attainment"] = daily.round({"produced": 1, "planned": 1, "attainment_pct": 1})
    tot = g[["planned", "produced", "defects", "downtime_min", "planned_min"]].sum()
    overall = (1 - tot["downtime_min"] / tot["planned_min"]) * min(1.0, tot["produced"] / tot["planned"]) * (1 - tot["defects"] / tot["produced"])
    worst, best = g["oee_style"].idxmin(), g["oee_style"].idxmax()
    down_line = g["downtime_min"].idxmax()
    res.facts = clean_facts(dict(
        lines=len(g), units_produced=int(tot["produced"]), plan_attainment_pct=pct1(safe_div(tot["produced"], tot["planned"])), oee_style_pct=pct1(overall),
        scrap_pct=pct1(safe_div(tot["defects"], tot["produced"])), downtime_hours=r1(float(tot["downtime_min"]) / 60),
        best_line=str(best), best_line_oee_pct=pct1(g.loc[best, "oee_style"]), worst_line=str(worst), worst_line_oee_pct=pct1(g.loc[worst, "oee_style"]),
        worst_line_scrap_pct=r1(float(g.loc[worst, "scrap_pct"])), most_downtime_line=str(down_line), most_downtime_hours=r1(float(g.loc[down_line, "downtime_min"]) / 60)))
    if g["oee_style"].min() < 0.6:
        res.add_flag(AMBER if g["oee_style"].min() >= 0.5 else RED, f"{worst} runs at {g.loc[worst, 'oee_style']:.0%} effectiveness against {g.loc[best, 'oee_style']:.0%} for {best}")
    if (g["scrap_pct"] > 4).any():
        bad = g[g["scrap_pct"] > 4]
        res.add_flag(AMBER, f"Scrap above 4% on {', '.join(map(str, bad.index))}")
    if not res.flags:
        res.add_flag(GREEN, "Lines are meeting plan with low scrap")
    res.notes.append("This is an OEE-STYLE measure: availability (planned minutes less downtime) x performance (output against PLAN, not ideal cycle time, "
                     "capped at 100%) x quality (good units). True OEE needs ideal cycle times, so treat the absolute level with care and compare lines with each other. "
                     "The usual 85% 'world class' benchmark does not apply to this version.")
    return res


def rec_production(f, flags):
    r = []
    if f.get("worst_line"):
        r.append(f"Start with {f['worst_line']}: effectiveness {f.get('worst_line_oee_pct')}% against {f.get('best_line_oee_pct')}% for {f.get('best_line')}; scrap there is {f.get('worst_line_scrap_pct')}%.")
    if f.get("most_downtime_line"):
        r.append(f"{f['most_downtime_line']} lost {f.get('most_downtime_hours')} hours to downtime; log stop reasons for a week to see the top two causes.")
    return r or ["Production is on plan."]


# ============================================================================== MaintenanceIQ
def run_maintenance(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("MaintenanceIQ")
    m = data["maintenance"].copy()
    m = m[m["date"] <= ctx.as_of]
    m["event_type"] = m["event_type"].astype(str).str.strip().str.lower()
    m["dh"] = _n(m, "downtime_hours")
    m["c"] = _n(m, "cost")
    window_days = max(1, (ctx.as_of - m["date"].min()).days)
    br = m[m["event_type"].str.startswith("break")]
    pv = m[m["event_type"].str.startswith("prev")]
    a = pd.DataFrame(index=sorted(m["asset"].unique()))
    a["breakdowns"] = br.groupby("asset").size()
    a["downtime_hours"] = br.groupby("asset")["dh"].sum()
    a["repair_cost"] = br.groupby("asset")["c"].sum()
    a["preventive_jobs"] = pv.groupby("asset").size()
    a["last_preventive"] = pv.groupby("asset")["date"].max()
    a = a.fillna({"breakdowns": 0, "downtime_hours": 0, "repair_cost": 0, "preventive_jobs": 0})
    a["mtbf_days"] = np.where(a["breakdowns"] > 0, window_days / a["breakdowns"].replace(0, np.nan), np.nan)
    a["mttr_hours"] = a["downtime_hours"] / a["breakdowns"].replace(0, np.nan)
    a["days_since_preventive"] = (ctx.as_of - a["last_preventive"]).dt.days
    interval = ctx.settings.get("preventive_interval_days", 180)
    a["preventive_overdue"] = a["days_since_preventive"].fillna(10_000) > interval
    a["repeat_failure"] = a["breakdowns"] >= 4
    res.tables["assets"] = a.round({c: 1 for c in a.select_dtypes("number").columns}).reset_index(names="asset").sort_values("breakdowns", ascending=False)
    if "cause" in br:
        res.tables["causes"] = br.groupby("cause").agg(breakdowns=("asset", "size"), downtime_hours=("dh", "sum"), cost=("c", "sum")).round(1).reset_index().sort_values("breakdowns", ascending=False)
    rep = a[a["repeat_failure"]]
    od = a[a["preventive_overdue"] & (a["breakdowns"] >= 2)]
    top = a["breakdowns"].idxmax()
    res.facts = clean_facts(dict(
        assets=len(a), breakdowns=int(a["breakdowns"].sum()), downtime_hours=r1(float(a["downtime_hours"].sum())), repair_cost=inr(float(a["repair_cost"].sum())),
        observation_days=int(window_days), repeat_failure_assets=len(rep), worst_asset=str(top), worst_asset_breakdowns=int(a.loc[top, "breakdowns"]),
        worst_asset_downtime_hours=r1(float(a.loc[top, "downtime_hours"])), worst_asset_mtbf_days=r1(float(a.loc[top, "mtbf_days"]), 0) if pd.notna(a.loc[top, "mtbf_days"]) else None,
        assets_overdue_preventive_with_failures=len(od), preventive_interval_days=int(interval),
        worst_asset_days_since_preventive=int(a.loc[top, "days_since_preventive"]) if pd.notna(a.loc[top, "days_since_preventive"]) else None))
    if len(rep):
        res.add_flag(RED if len(rep) > 1 else AMBER, f"{', '.join(map(str, rep.index[:3]))} broke down 4+ times in {window_days} days")
    if len(od):
        res.add_flag(AMBER, f"{len(od)} asset(s) with repeat failures have had no preventive service for over {interval} days")
    if not res.flags:
        res.add_flag(GREEN, "Breakdowns are rare and preventive service is current")
    res.notes.append(f"MTBF = observation window / breakdowns; MTTR = downtime hours per breakdown. The window runs from the first logged event, so unlogged breakdowns understate the problem. "
                     f"Preventive service is assumed due every {interval} days; use the manufacturer's schedule.")
    return res


def rec_maintenance(f, flags):
    r = []
    if f.get("repeat_failure_assets"):
        r.append(f"Do a root-cause review on {f['worst_asset']}: {f.get('worst_asset_breakdowns')} breakdowns, {f.get('worst_asset_downtime_hours')} hours down. Decide repair against replace with the repair cost in the table.")
    if f.get("assets_overdue_preventive_with_failures"):
        r.append(f"Book preventive service now for the {f['assets_overdue_preventive_with_failures']} failing asset(s) that are overdue ({f.get('worst_asset_days_since_preventive')} days for the worst).")
    return r or ["Keep logging every breakdown with its cause; the history is what makes this useful."]


# ============================================================================== TenderIQ
def run_tender(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("TenderIQ")
    t = data["tenders"].copy()
    t["status"] = t["status"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    t["days_left"] = (t["deadline"] - ctx.as_of).dt.days
    openq = t[t["status"] == "open"]
    upcoming = openq[openq["days_left"] >= 0].sort_values("deadline")
    missed = openq[openq["days_left"] < 0]
    res.tables["upcoming_deadlines"] = upcoming[["tender_id", "title", "deadline", "days_left", "value"] + (["emd_amount"] if "emd_amount" in t else [])]
    res.tables["missed_deadlines"] = missed[["tender_id", "title", "deadline", "value"]]
    dec = t[t["status"].isin(["won", "lost"])]
    won = dec[dec["status"] == "won"]
    win_rate = safe_div(len(won), len(dec))
    lost = dec[(dec["status"] == "lost") & dec.get("our_bid", pd.Series(dtype=float)).notna() & dec.get("winning_bid", pd.Series(dtype=float)).notna()] if "winning_bid" in dec and "our_bid" in dec else dec.iloc[0:0]
    gap = (lost["our_bid"] / lost["winning_bid"] - 1) * 100 if len(lost) else pd.Series(dtype=float)
    if len(lost):
        res.tables["lost_bids"] = lost.assign(above_winner_pct=gap.round(1))[["tender_id", "title", "value", "our_bid", "winning_bid", "above_winner_pct"]].sort_values("above_winner_pct", ascending=False)
    emd_locked = float(t.loc[t["status"].isin(["submitted", "open"]), "emd_amount"].fillna(0).sum()) if "emd_amount" in t else 0.0
    no_bid = int((t["status"] == "no_bid").sum())
    soon = upcoming[upcoming["days_left"] <= 7]
    res.facts = clean_facts(dict(
        tenders_tracked=len(t), open_tenders=len(openq), open_value=inr(float(openq["value"].sum())), due_within_7_days=len(soon), due_within_7_days_value=inr(float(soon["value"].sum())),
        next_deadline_title=str(upcoming.iloc[0]["title"]) if len(upcoming) else None, next_deadline_days=int(upcoming.iloc[0]["days_left"]) if len(upcoming) else None,
        missed_open_tenders=len(missed), decided_tenders=len(dec), win_rate_pct=pct1(win_rate), won_value=inr(float(won["value"].sum())),
        median_lost_bid_above_winner_pct=r1(float(gap.median())) if len(gap) else None, emd_locked_up=inr(emd_locked), no_bid_pct=pct1(safe_div(no_bid, len(t)))))
    if len(soon):
        res.add_flag(RED if len(soon) >= 2 else AMBER, f"{len(soon)} tender(s) worth {inr(float(soon['value'].sum()))} close within 7 days")
    if len(missed):
        res.add_flag(AMBER, f"{len(missed)} tender(s) passed their deadline still marked open: update the status")
    if len(gap) >= 3 and gap.median() > 5:
        res.add_flag(AMBER, f"Lost bids were a median {gap.median():.0f}% above the winning price")
    if pd.notna(win_rate) and len(dec) >= 8 and win_rate < 0.25:
        res.add_flag(AMBER, f"Win rate is {win_rate:.0%} across {len(dec)} decided tenders")
    if not res.flags:
        res.add_flag(GREEN, "Tender pipeline is on track")
    res.notes.append("Win rate counts only tenders recorded as won or lost. The price gap uses lost tenders where the winning bid is known; price is rarely the only reason "
                     "a bid loses (eligibility, technical score, relationships).")
    return res


def rec_tender(f, flags):
    r = []
    if f.get("due_within_7_days"):
        r.append(f"Prioritise the {f['due_within_7_days']} tender(s) due within 7 days ({f.get('due_within_7_days_value')}); next is '{f.get('next_deadline_title')}' in {f.get('next_deadline_days')} days.")
    if f.get("median_lost_bid_above_winner_pct") and f["median_lost_bid_above_winner_pct"] > 5:
        r.append(f"Review pricing: lost bids sat a median {f['median_lost_bid_above_winner_pct']}% above the winner. Test a leaner margin on tenders you can service profitably.")
    if f.get("missed_open_tenders"):
        r.append("Close out tenders whose deadline has passed so win rate and EMD figures stay honest.")
    return r or ["Keep deadlines and bid outcomes up to date; the win-rate history is the useful part."]


# ============================================================================== ESGIQ
def run_esg(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ESGIQ")
    e = data["esg"].copy().sort_values("month")
    e = e[e["month"] <= ctx.as_of + pd.Timedelta(days=31)]
    gf = ctx.settings.get("grid_kg_per_kwh", 0.71)          # editable assumption; use your state's published factor
    df_ = ctx.settings.get("diesel_kg_per_litre", 2.68)
    e["co2e_t"] = (_n(e, "electricity_kwh") * gf + _n(e, "fuel_litres") * df_) / 1000
    for c in ("waste_kg", "recycled_kg"):
        e[c] = _n(e, c)
    e["recycling_pct"] = 100 * e["recycled_kg"] / e["waste_kg"].replace(0, np.nan)
    has_rev = "revenue" in e and e["revenue"].notna().any()
    if has_rev:
        e["kwh_per_lakh_revenue"] = e["electricity_kwh"] / (e["revenue"] / 1e5).replace(0, np.nan)
    res.tables["monthly"] = e.round({c: 2 for c in e.select_dtypes("number").columns})
    last6, prior6 = e.tail(6), e.iloc[-12:-6]

    def chg(col):
        a, b = last6[col].sum(), prior6[col].sum()
        return 100 * (a / b - 1) if len(prior6) >= 3 and b else None
    elec_chg, co2_chg = chg("electricity_kwh"), chg("co2e_t")
    inten_chg = None
    if has_rev and len(prior6) >= 3:
        inten_chg = 100 * ((last6["electricity_kwh"].sum() / last6["revenue"].sum()) / (prior6["electricity_kwh"].sum() / prior6["revenue"].sum()) - 1)
    ytd12 = e.tail(12)
    res.facts = clean_facts(dict(
        months_of_data=len(e), electricity_kwh_last_12m=int(ytd12["electricity_kwh"].sum()), estimated_co2e_tonnes_last_12m=r1(float(ytd12["co2e_t"].sum())),
        electricity_change_last_6m_pct=r1(elec_chg), co2e_change_last_6m_pct=r1(co2_chg), electricity_per_revenue_change_pct=r1(inten_chg),
        recycling_rate_pct=r1(float(ytd12["recycled_kg"].sum() / ytd12["waste_kg"].sum() * 100)) if ytd12["waste_kg"].sum() else None,
        water_kl_last_12m=int(_n(ytd12, "water_kl").sum()) if "water_kl" in ytd12 else None, grid_factor_assumed_kg_per_kwh=gf))
    if elec_chg is not None and elec_chg > 10:
        res.add_flag(AMBER, f"Electricity use is up {elec_chg:.0f}% over the last six months against the six before")
    if inten_chg is not None and inten_chg > 8:
        res.add_flag(AMBER, f"Electricity per rupee of revenue is up {inten_chg:.0f}%: usage is rising faster than business")
    rr = res.facts.get("recycling_rate_pct")
    if rr is not None and rr < 30:
        res.add_flag(AMBER, f"Only {rr:.0f}% of waste is recycled")
    if not res.flags:
        res.add_flag(GREEN, "Energy and waste are stable or improving")
    res.notes.append(f"CO2e is an ESTIMATE: electricity x {gf} kg/kWh and diesel x {df_} kg/litre. These are editable assumptions; use your state grid factor and your own fuel records. "
                     "It covers only the inputs supplied (no refrigerants, purchased goods or transport), so it is not a BRSR, GHG Protocol or audited carbon figure.")
    return res


def rec_esg(f, flags):
    r = []
    if (f.get("electricity_change_last_6m_pct") or 0) > 10:
        r.append(f"Electricity rose {f['electricity_change_last_6m_pct']}% in six months: check for new equipment, tariff or meter changes, then run a night-time load survey.")
    if f.get("recycling_rate_pct") is not None and f["recycling_rate_pct"] < 30:
        r.append(f"Recycling is {f['recycling_rate_pct']}%; segregate the two biggest waste streams at source first.")
    r.append(f"Estimated emissions over the last 12 months are {f.get('estimated_co2e_tonnes_last_12m')} tonnes CO2e (assumed factors); keep the monthly readings so a formal report can be built later.")
    return r


register(ModuleSpec("LogisticsIQ", "Operations and industry", "Carrier on-time rates, returns to origin and cash-on-delivery risk.", ["shipments"], run_logistics, rec_logistics, audience="B2C", phase="3"))
register(ModuleSpec("ProductionIQ", "Operations and industry", "Line output against plan, downtime and scrap, in an OEE-style score.", ["production"], run_production, rec_production, audience="B2B", phase="3"))
register(ModuleSpec("MaintenanceIQ", "Operations and industry", "Machine breakdowns, time between failures and overdue servicing.", ["maintenance"], run_maintenance, rec_maintenance, audience="B2B", phase="3"))
register(ModuleSpec("TenderIQ", "Operations and industry", "Tender deadlines, win rate, price gaps and money locked in deposits.", ["tenders"], run_tender, rec_tender, audience="B2B", phase="3"))
register(ModuleSpec("ESGIQ", "Operations and industry", "Energy, water and waste trends with an estimated carbon footprint.", ["esg"], run_esg, rec_esg, audience="Both", phase="3"))
