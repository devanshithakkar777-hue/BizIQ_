"""People modules (sensitive: need view_people): HiringIQ, SalaryIQ, PerformanceReviewIQ.

These describe groups and processes. They never rank, score or recommend action about a named individual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import AMBER, GREEN, RED, ModuleResult, inr, safe_div
from ..registry import Context, ModuleSpec, register
from ._common import clean_facts, pct1, r1

MIN_GROUP = 5          # never report a statistic for fewer people than this


# ============================================================================== HiringIQ
STAGES = ["applied", "screen", "interview", "offer", "hired"]


def run_hiring(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("HiringIQ")
    c = data["candidates"].copy()
    c["stage"] = c["stage"].astype(str).str.strip().str.lower()
    rank = {s: i for i, s in enumerate(STAGES)}
    # furthest stage reached is only known for people still in the pipeline or hired; rejected/withdrawn stay at "applied" level in this simple model
    reached = c["stage"].map(rank)
    funnel = pd.DataFrame({"stage": STAGES, "candidates_at_or_past": [int((reached >= i).sum()) for i in range(len(STAGES))]})
    funnel["pct_of_applicants"] = (100 * funnel["candidates_at_or_past"] / max(1, len(c))).round(1)
    res.tables["funnel"] = funnel
    hired = c[c["stage"] == "hired"].copy()
    if "hired_date" in hired:
        hired["days_to_hire"] = (hired["hired_date"] - hired["applied_date"]).dt.days
    ttf = hired["days_to_hire"].median() if len(hired) and "days_to_hire" in hired else np.nan
    if "source" in c and c["source"].notna().any():
        g = c.groupby("source").agg(applicants=("candidate_id", "size"), hires=("stage", lambda x: int((x == "hired").sum())))
        g["hire_rate_pct"] = (100 * g["hires"] / g["applicants"]).round(1)
        res.tables["sources"] = g.reset_index().sort_values("hire_rate_pct", ascending=False)
    pipeline_ = c[c["stage"].isin(["screen", "interview", "offer"])].copy()
    last = pipeline_["last_stage_date"] if "last_stage_date" in pipeline_ else pipeline_["applied_date"]
    pipeline_["days_idle"] = (ctx.as_of - last.fillna(pipeline_["applied_date"])).dt.days
    stalled_days = ctx.settings.get("hiring_stall_days", 14)
    stalled = pipeline_[pipeline_["days_idle"] > stalled_days]
    res.tables["stalled_by_role"] = stalled.groupby("role").size().rename("candidates_waiting").reset_index().sort_values("candidates_waiting", ascending=False)
    offers = c[c["stage"].isin(["offer", "hired"])]
    acc = safe_div(len(hired), len(offers))
    best_src = res.tables["sources"][res.tables["sources"]["applicants"] >= MIN_GROUP].head(1) if "sources" in res.tables else pd.DataFrame()
    worst_src = res.tables["sources"][res.tables["sources"]["applicants"] >= MIN_GROUP].tail(1) if "sources" in res.tables else pd.DataFrame()
    res.facts = clean_facts(dict(
        candidates=len(c), hires=len(hired), in_pipeline=len(pipeline_), median_days_to_hire=r1(float(ttf), 0) if pd.notna(ttf) else None,
        applicant_to_hire_pct=pct1(safe_div(len(hired), len(c))), offer_to_hire_pct=pct1(acc),
        candidates_waiting_over_threshold=len(stalled), stall_threshold_days=int(stalled_days),
        best_source=str(best_src.iloc[0]["source"]) if len(best_src) else None, best_source_hire_rate_pct=r1(float(best_src.iloc[0]["hire_rate_pct"])) if len(best_src) else None,
        weakest_source=str(worst_src.iloc[0]["source"]) if len(worst_src) else None, weakest_source_hire_rate_pct=r1(float(worst_src.iloc[0]["hire_rate_pct"])) if len(worst_src) else None,
        weakest_source_applicants=int(worst_src.iloc[0]["applicants"]) if len(worst_src) else None))
    if len(pipeline_) and len(stalled) / len(pipeline_) > 0.25:
        res.add_flag(AMBER, f"{len(stalled)} of {len(pipeline_)} candidates in process have waited over {stalled_days} days without movement")
    if pd.notna(ttf) and ttf > 45:
        res.add_flag(AMBER, f"Median time to hire is {ttf:.0f} days; good candidates often accept elsewhere before this")
    if pd.notna(acc) and len(offers) >= 8 and acc < 0.6:
        res.add_flag(AMBER, f"Only {acc:.0%} of offers end in a hire")
    if not res.flags:
        res.add_flag(GREEN, "Hiring is moving at a healthy pace")
    res.notes.append("A process view only: it never ranks or scores candidates, and must not be used to screen people. The funnel treats rejected and withdrawn candidates as stopping at the first "
                     f"stage because the data has no per-stage history, so it understates how far people got. Sources with fewer than {MIN_GROUP} applicants are not compared. "
                     "Candidate data is personal data under the DPDP Act: keep only what you need and delete it when the role closes.")
    return res


def rec_hiring(f, flags):
    r = []
    if f.get("candidates_waiting_over_threshold"):
        r.append(f"Clear the {f['candidates_waiting_over_threshold']} candidate(s) waiting over {f.get('stall_threshold_days')} days: decide or tell them, today. The role table shows where they sit.")
    if f.get("best_source") and f.get("weakest_source") and f["best_source"] != f["weakest_source"]:
        r.append(f"{f['best_source']} converts {f.get('best_source_hire_rate_pct')}% of applicants to hires against {f.get('weakest_source_hire_rate_pct')}% for {f['weakest_source']} ({f.get('weakest_source_applicants')} applicants); move effort toward the better source.")
    return r or ["Keep stage dates up to date so time-to-hire stays meaningful."]


# ============================================================================== SalaryIQ
def _active(h: pd.DataFrame, as_of) -> pd.DataFrame:
    h = h.copy()
    if "exit_date" in h:
        h = h[h["exit_date"].isna() | (h["exit_date"] > as_of)]
    return h[h["join_date"] <= as_of]


def run_salary(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("SalaryIQ")
    h = _active(data["hr"], ctx.as_of)
    h["tenure_years"] = (ctx.as_of - h["join_date"]).dt.days / 365.25
    dept = h.groupby("department")["monthly_salary"].agg(headcount="size", median="median", p25=lambda x: x.quantile(.25), p75=lambda x: x.quantile(.75), total="sum")
    dept = dept[dept["headcount"] >= MIN_GROUP]
    dept["spread_p75_to_p25"] = dept["p75"] / dept["p25"].replace(0, np.nan)
    res.tables["departments"] = dept.round(0).reset_index()
    # compression: people joined in the last 12 months earning at least the median of colleagues with 3+ years
    comp_rows = []
    for dp, g in h.groupby("department"):
        vet = g[g["tenure_years"] >= 3]["monthly_salary"]
        new = g[g["tenure_years"] <= 1]
        if len(vet) >= MIN_GROUP and len(new) >= 2:
            comp_rows.append(dict(department=dp, newer_hires=len(new), newer_at_or_above_veteran_median=int((new["monthly_salary"] >= vet.median()).sum()), veteran_median=float(vet.median())))
    comp = pd.DataFrame(comp_rows, columns=["department", "newer_hires", "newer_at_or_above_veteran_median", "veteran_median"])
    res.tables["pay_compression"] = comp
    corr = h[["tenure_years", "monthly_salary"]].corr().iloc[0, 1] if len(h) >= 10 else np.nan
    # outliers inside a department: far from the department median (robust z)
    out = []
    for dp, g in h.groupby("department"):
        if len(g) >= MIN_GROUP:
            med = g["monthly_salary"].median()
            mad = (g["monthly_salary"] - med).abs().median() or 1
            z = 0.6745 * (g["monthly_salary"] - med) / mad
            n_out = int((z.abs() > 3.5).sum())               # a COUNT per department: nobody is named and no individual salary is listed
            if n_out:
                out.append(dict(department=dp, people_in_department=len(g), salaries_far_from_department_median=n_out))
    res.tables["pay_outliers_by_department"] = pd.DataFrame(out, columns=["department", "people_in_department", "salaries_far_from_department_median"])
    n_outliers = int(sum(r["salaries_far_from_department_median"] for r in out))
    gap = None
    if "gender" in h and h["gender"].notna().any():
        gg = h.assign(gender=h["gender"].astype(str).str.strip().str.lower())
        rows = []
        for dp, g in gg.groupby("department"):
            by = g.groupby("gender")["monthly_salary"].agg(["size", "median"])
            by = by[by["size"] >= MIN_GROUP]
            if {"female", "male"} <= set(by.index):
                rows.append(dict(department=dp, women=int(by.loc["female", "size"]), men=int(by.loc["male", "size"]), women_median=float(by.loc["female", "median"]),
                                 men_median=float(by.loc["male", "median"]), gap_pct=round(100 * (1 - by.loc["female", "median"] / by.loc["male", "median"]), 1)))
        res.tables["gender_pay_gap_by_department"] = pd.DataFrame(rows, columns=["department", "women", "men", "women_median", "men_median", "gap_pct"])
        if rows:
            gap = float(np.average([r["gap_pct"] for r in rows], weights=[r["women"] + r["men"] for r in rows]))
    payroll = float(h["monthly_salary"].sum())
    comp_n = int(comp["newer_at_or_above_veteran_median"].sum()) if len(comp) else 0
    wide = dept[dept["spread_p75_to_p25"] > 1.8]
    res.facts = clean_facts(dict(
        active_employees=len(h), monthly_payroll=inr(payroll), annual_payroll=inr(payroll * 12), departments_reported=len(dept),
        median_monthly_salary=inr(float(h["monthly_salary"].median())), tenure_pay_correlation=r1(float(corr), 2) if pd.notna(corr) else None,
        newer_hires_paid_at_or_above_veteran_median=comp_n, departments_with_wide_pay_spread=len(wide), pay_outliers_for_review=n_outliers,
        average_gender_pay_gap_within_department_pct=r1(gap) if gap is not None else None))
    if comp_n >= 3:
        res.add_flag(AMBER, f"{comp_n} recent hires are paid at or above the median of colleagues with 3+ years in the same department")
    if gap is not None and abs(gap) >= 8:
        res.add_flag(AMBER, f"Women's median pay is {gap:.0f}% lower than men's within departments (departments with 5+ of each)")
    if len(wide):
        res.add_flag(AMBER, f"Pay spread is very wide in {', '.join(map(str, wide.index[:3]))}")
    if n_outliers:
        res.add_flag(AMBER, f"{n_outliers} salary(ies) sit far outside their department's range, in {len(out)} department(s): review the job levels there")
    if not res.flags:
        res.add_flag(GREEN, "Pay looks consistent within departments")
    res.notes.append("Group-level pay structure only (no individual is named or has a salary listed), using monthly salary of current employees. Groups under 5 people are not reported. Department medians do not account for role, level or "
                     "location, so a gap or outlier is a prompt to check job grading, NOT evidence of unfair pay. The gender comparison is a raw within-department median, not a statistical "
                     "equal-pay analysis; get HR or legal advice before acting. Never change an individual's pay from this output alone.")
    return res


def rec_salary(f, flags):
    r = []
    if f.get("newer_hires_paid_at_or_above_veteran_median"):
        r.append(f"{f['newer_hires_paid_at_or_above_veteran_median']} recent hires earn at least what long-serving colleagues in the same department do; review whether tenured staff need a market correction before they notice.")
    if f.get("average_gender_pay_gap_within_department_pct") is not None and abs(f["average_gender_pay_gap_within_department_pct"]) >= 8:
        r.append(f"Check job levels behind the {f['average_gender_pay_gap_within_department_pct']}% median gap; if the roles are comparable, plan a correction in the next review cycle.")
    if f.get("pay_outliers_for_review"):
        r.append(f"Check job grades in the {f['pay_outliers_for_review']} case(s) where a salary sits far from its department's range (see the department counts); the module does not name anyone.")
    return r or ["Pay structure looks consistent; re-run after each review cycle."]


# ============================================================================== PerformanceReviewIQ
def run_review(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("PerformanceReviewIQ")
    h = _active(data["hr"], ctx.as_of)
    rated = h[pd.to_numeric(h.get("performance_rating"), errors="coerce").notna()].copy()
    rated["performance_rating"] = pd.to_numeric(rated["performance_rating"])
    unrated = len(h) - len(rated)
    dist = rated["performance_rating"].round().value_counts().sort_index()
    res.tables["rating_distribution"] = pd.DataFrame({"rating": dist.index, "people": dist.values, "share_pct": (100 * dist.values / max(1, len(rated))).round(1)})
    mean_all, sd_all = rated["performance_rating"].mean(), rated["performance_rating"].std()
    groups = []
    for col in [c for c in ("department", "manager") if c in rated and rated[c].notna().any()]:
        g = rated.groupby(col)["performance_rating"].agg(people="size", average="mean")
        g = g[g["people"] >= MIN_GROUP].copy()
        g["vs_company"] = g["average"] - mean_all
        g["note"] = np.where(g["vs_company"] > 0.6, "rates high", np.where(g["vs_company"] < -0.6, "rates low", ""))
        g.insert(0, "group_type", col)
        groups.append(g.reset_index(names="group"))
    res.tables["rating_by_group"] = pd.concat(groups) if groups else pd.DataFrame(columns=["group_type", "group", "people", "average", "vs_company", "note"])
    top_share = float((rated["performance_rating"].round() == dist.idxmax()).mean()) if len(dist) else np.nan
    retention = None
    if len(rated) >= 10:
        dep_med = h.groupby("department")["monthly_salary"].transform("median").reindex(rated.index)
        strong = rated["performance_rating"] >= rated["performance_rating"].quantile(0.8)
        retention = int((strong & (rated["monthly_salary"] < 0.9 * dep_med)).sum())
    res.tables["rating_vs_pay_by_department"] = (rated.groupby("department").apply(lambda g: pd.Series(dict(people=len(g), rating_pay_correlation=g["performance_rating"].corr(g["monthly_salary"]) if len(g) >= MIN_GROUP else np.nan)), include_groups=False)
                                                  .reset_index())
    lenient = res.tables["rating_by_group"][res.tables["rating_by_group"]["note"] != ""] if len(res.tables["rating_by_group"]) else pd.DataFrame()
    res.facts = clean_facts(dict(
        employees_rated=len(rated), employees_unrated=int(unrated), average_rating=r1(float(mean_all), 2) if pd.notna(mean_all) else None,
        most_common_rating_share_pct=pct1(top_share) if pd.notna(top_share) else None, groups_rating_far_from_company=len(lenient),
        strong_performers_paid_under_90pct_of_department_median=retention))
    if pd.notna(top_share) and top_share > 0.6:
        res.add_flag(AMBER, f"{top_share:.0%} of employees got the same rating; the scale is not separating performance")
    if len(lenient):
        res.add_flag(AMBER, f"{len(lenient)} group(s) rate much higher or lower than the company average: calibrate before ratings drive pay")
    if retention:
        res.add_flag(AMBER, f"{retention} high-rated employee(s) are paid under 90% of their department median")
    if len(h) and unrated / len(h) > 0.15:
        res.add_flag(AMBER, f"{unrated / len(h):.0%} of employees have no rating")
    if not res.flags:
        res.add_flag(GREEN, "Ratings are spread out and consistent across groups")
    res.notes.append(f"A check on the review PROCESS (spread, consistency, link to pay), not on people. Groups under {MIN_GROUP} are not reported. A group that rates high or low may have genuinely stronger or "
                     "weaker performers; use it to start a calibration discussion. Never use this to rank, reward or discipline an individual.")
    return res


def rec_review(f, flags):
    r = []
    if (f.get("most_common_rating_share_pct") or 0) > 60:
        r.append(f"{f['most_common_rating_share_pct']}% of staff share one rating; hold a calibration session with written examples for each rating level.")
    if f.get("groups_rating_far_from_company"):
        r.append(f"Compare managers or departments flagged in the group table in a calibration meeting before ratings feed into increments.")
    if f.get("strong_performers_paid_under_90pct_of_department_median"):
        r.append(f"{f['strong_performers_paid_under_90pct_of_department_median']} high-rated people are paid well below their department median; review them at the next pay round.")
    if f.get("employees_unrated"):
        r.append(f"Complete reviews for the {f['employees_unrated']} employees with no rating so the picture is not skewed.")
    return r or ["Ratings look healthy; keep calibrating each cycle."]


register(ModuleSpec("HiringIQ", "People", "Hiring funnel, time to hire, source quality and stalled candidates (process view only).", ["candidates"], run_hiring, rec_hiring, audience="Both", phase="3", sensitive=True))
register(ModuleSpec("SalaryIQ", "People", "Pay structure by department: spread, compression and group gaps.", ["hr"], run_salary, rec_salary, audience="Both", phase="3", sensitive=True))
register(ModuleSpec("PerformanceReviewIQ", "People", "Whether reviews are consistent and linked to pay, at group level.", ["hr"], run_review, rec_review, audience="Both", phase="3", sensitive=True))
