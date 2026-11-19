"""Books and compliance modules: PayablesIQ, ReconIQ, ComplianceCalendarIQ, AssetIQ."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import AMBER, GREEN, RED, ModuleResult, inr, safe_div
from ..registry import Context, ModuleSpec, register
from ._common import argext, clean_facts, open_mask, pct1, r1


# ============================================================================== PayablesIQ
def run_payables(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("PayablesIQ")
    inv = data["invoices"]
    p = inv[inv["type"].astype(str).str.lower().isin(["payable", "payables", "bill", "purchase"])].copy()
    if p.empty:
        raise ValueError("needs invoices of type payable (bills you owe)")
    as_of = ctx.as_of
    p["open"] = open_mask(p["paid_date"], as_of)
    o = p[p["open"]].copy()
    o["days_overdue"] = (as_of - o["due_date"]).dt.days
    buckets = pd.cut(o["days_overdue"], [-10 ** 6, 0, 30, 60, 90, 10 ** 6], labels=["not yet due", "1-30 days", "31-60 days", "61-90 days", "over 90 days"])
    aging = o.groupby(buckets, observed=False).agg(bills=("amount", "size"), amount=("amount", "sum")).reset_index().rename(columns={"days_overdue": "bucket"})
    res.tables["aging"] = aging
    due_soon = o[(o["due_date"] >= as_of) & (o["due_date"] <= as_of + pd.Timedelta(days=14))]
    due_30 = o[(o["due_date"] <= as_of + pd.Timedelta(days=30))]
    res.tables["open_bills"] = o.sort_values("due_date")[["invoice_id", "party", "due_date", "amount", "days_overdue"]]
    paid = p[~p["open"] & p["paid_date"].notna()].copy()
    paid["days_late"] = (paid["paid_date"] - paid["due_date"]).dt.days
    late_share = float((paid["days_late"] > 0).mean()) if len(paid) else None
    early = float((paid["days_late"] < -7).mean()) if len(paid) else None
    avg_late = float(paid.loc[paid["days_late"] > 0, "days_late"].mean()) if len(paid) and (paid["days_late"] > 0).any() else None
    vend = p.groupby("party").agg(bills=("amount", "size"), total=("amount", "sum"), open_amount=("amount", lambda x: float(x[p.loc[x.index, "open"]].sum())))
    res.tables["by_vendor"] = vend.reset_index().sort_values("open_amount", ascending=False)
    # same vendor, same amount, close dates: a bill entered twice
    q = p.sort_values("invoice_date")
    dup = q[q.duplicated(["party", "amount"], keep=False)]
    pairs = 0
    dup_amt = 0.0
    for _, g in dup.groupby(["party", "amount"]):
        d_ = g["invoice_date"].sort_values().diff().dt.days
        n = int((d_ <= 30).sum())
        pairs += n
        dup_amt += n * float(g["amount"].iloc[0])
    res.tables["possible_duplicate_bills"] = dup[["invoice_id", "party", "invoice_date", "amount"]] if pairs else pd.DataFrame(columns=["invoice_id", "party", "invoice_date", "amount"])
    overdue = o[o["days_overdue"] > 0]
    cash = None
    fin = data.get("financials")
    if fin is not None and len(fin) and "cash" in fin:
        cash = float(fin.sort_values("month")["cash"].iloc[-1])
    res.facts = clean_facts(dict(
        open_payables=inr(float(o["amount"].sum())), open_bills=int(len(o)), overdue_amount=inr(float(overdue["amount"].sum())), overdue_bills=int(len(overdue)),
        due_next_14_days=inr(float(due_soon["amount"].sum())), due_within_30_days_incl_overdue=inr(float(due_30["amount"].sum())),
        paid_late_pct=pct1(late_share), average_days_late_when_late=r1(avg_late), paid_more_than_a_week_early_pct=pct1(early),
        possible_duplicate_bills=int(pairs), possible_duplicate_value=inr(dup_amt) if pairs else None,
        latest_cash=None if cash is None else inr(cash), cash_covers_due_30d_x=None if cash is None else r1(safe_div(cash, float(due_30["amount"].sum())))))
    if cash is not None and float(due_30["amount"].sum()) > cash:
        res.add_flag(RED, f"Bills due within 30 days ({inr(float(due_30['amount'].sum()))}) exceed the cash on your latest balance sheet ({inr(cash)})")
    if len(overdue) and float(overdue["amount"].sum()) > 0:
        worst = overdue.sort_values("days_overdue", ascending=False).iloc[0]
        res.add_flag(RED if int(worst["days_overdue"]) > 60 else AMBER, f"{len(overdue)} bill(s) are overdue, {inr(float(overdue['amount'].sum()))} in total", f"The oldest is {int(worst['days_overdue'])} days late")
    if late_share is not None and late_share > 0.30:
        res.add_flag(AMBER, f"{late_share:.0%} of bills were paid after the due date", "Late payment risks supplier credit and late fees" + (f"; usually {avg_late:.0f} days late" if avg_late else ""))
    if early is not None and early > 0.30:
        res.add_flag(AMBER, f"{early:.0%} of bills were paid more than a week early", "Paying early without a discount costs you working capital")
    if pairs:
        res.add_flag(AMBER, f"{pairs} bill(s) look entered twice (same vendor and amount within 30 days)", f"{inr(dup_amt)} at stake")
    if not res.flags:
        res.add_flag(GREEN, "Bills are being paid on time and nothing looks duplicated")
    res.notes.append("A bill counts as open when it has no payment date on or before the as-of date. Late and early are measured on bills already paid. Possible duplicates are a prompt to check, "
                     "since a vendor can genuinely bill the same amount twice (a fixed monthly fee). Cash comes from the latest month in your financials, which may be older than today.")
    return res


def rec_payables(f, flags):
    r = []
    if f.get("overdue_bills"):
        r.append(f"Clear or renegotiate the {f['overdue_bills']} overdue bill(s) ({f.get('overdue_amount')}), oldest first, and tell the vendors your date before they chase you.")
    if f.get("possible_duplicate_bills"):
        r.append(f"Check the {f['possible_duplicate_bills']} possible double-entered bill(s) before paying them.")
    if f.get("paid_more_than_a_week_early_pct") and f["paid_more_than_a_week_early_pct"] > 30:
        r.append("Schedule payments for the due date unless the vendor gives an early-payment discount.")
    return r or ["Payables look healthy; keep scheduling payment on the due date."]


# ============================================================================== ReconIQ
def _ref(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper().replace("", pd.NA)


def run_recon(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ReconIQ")
    b = data["bank_statement"].reset_index(drop=True).copy()
    l = data["ledger_entries"].reset_index(drop=True).copy()
    if b.empty or l.empty:
        raise ValueError("needs both a bank statement and ledger entries")
    for df in (b, l):
        df["ref"] = _ref(df["reference"]) if "reference" in df else pd.Series(pd.NA, index=df.index, dtype="string")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").round(2)
    # only compare the period both files cover
    lo, hi = max(b["date"].min(), l["date"].min()) - pd.Timedelta(days=5), min(b["date"].max(), l["date"].max()) + pd.Timedelta(days=5)
    l_in = l[(l["date"] >= lo) & (l["date"] <= hi)]
    b_in = b[(b["date"] >= lo) & (b["date"] <= hi)]
    l_unmatched, matched, mismatch, dup_bank = set(l_in.index), [], [], []
    by_ref = {}
    for i, r in l_in.iterrows():
        if pd.notna(r["ref"]):
            by_ref.setdefault(r["ref"], []).append(i)
    b_left = []
    for i, r in b_in.iterrows():
        if pd.notna(r["ref"]) and r["ref"] in by_ref:
            free = [j for j in by_ref[r["ref"]] if j in l_unmatched]
            if not free:
                dup_bank.append(i)                                                    # same reference already matched: the bank shows it again
                continue
            j = free[0]
            l_unmatched.discard(j)
            (matched if abs(l.at[j, "amount"] - r["amount"]) <= 0.01 else mismatch).append((i, j))
        else:
            b_left.append(i)
    still = []
    for i in b_left:                                                                   # no usable reference: amount and date within 5 days
        r = b.loc[i]
        cands = [j for j in l_unmatched if abs(l.at[j, "amount"] - r["amount"]) <= 0.01 and abs((l.at[j, "date"] - r["date"]).days) <= 5]
        if cands:
            j = min(cands, key=lambda j: abs((l.at[j, "date"] - r["date"]).days))
            l_unmatched.discard(j)
            matched.append((i, j))
        else:
            still.append(i)
    bank_only = b.loc[still].drop(columns=["ref"])
    ledger_only = l.loc[sorted(l_unmatched)].drop(columns=["ref"])
    mm = pd.DataFrame([dict(reference=b.at[i, "ref"], date=b.at[i, "date"], bank_amount=b.at[i, "amount"], ledger_amount=l.at[j, "amount"], difference=round(b.at[i, "amount"] - l.at[j, "amount"], 2)) for i, j in mismatch])
    dups = b.loc[dup_bank].drop(columns=["ref"])
    res.tables["bank_only_not_in_books"] = bank_only
    res.tables["ledger_only_not_on_statement"] = ledger_only
    res.tables["amount_mismatches"] = mm
    res.tables["duplicate_bank_lines"] = dups
    n = len(b_in)
    old_uncleared = ledger_only[ledger_only["date"] < ctx.as_of - pd.Timedelta(days=30)] if len(ledger_only) else ledger_only
    res.facts = clean_facts(dict(
        bank_lines_compared=int(n), matched=int(len(matched)), matched_pct=pct1(safe_div(len(matched), n)),
        in_bank_not_in_books=int(len(bank_only)), in_bank_not_in_books_value=inr(float(bank_only["amount"].sum())) if len(bank_only) else None,
        in_books_not_on_statement=int(len(ledger_only)), in_books_not_on_statement_value=inr(float(ledger_only["amount"].sum())) if len(ledger_only) else None,
        amount_mismatches=int(len(mm)), mismatch_net=inr(float(mm["difference"].sum())) if len(mm) else None,
        duplicate_bank_lines=int(len(dups)), duplicate_bank_value=inr(float(dups["amount"].sum())) if len(dups) else None,
        uncleared_over_30_days=int(len(old_uncleared)),
        period_start=lo.strftime("%Y-%m-%d"), period_end=hi.strftime("%Y-%m-%d")))
    if len(dups):
        res.add_flag(RED, f"{len(dups)} bank line(s) repeat a reference already matched: possible double debit or credit", f"{inr(float(dups['amount'].sum()))}")
    if len(mm):
        res.add_flag(AMBER, f"{len(mm)} transaction(s) match by reference but the amounts differ", f"net {inr(float(mm['difference'].sum()))}")
    if len(bank_only):
        res.add_flag(AMBER, f"{len(bank_only)} bank line(s) are not in your books (charges, interest or missed entries)", f"{inr(float(bank_only['amount'].sum()))}")
    if len(old_uncleared):
        res.add_flag(AMBER, f"{len(old_uncleared)} book entries have not reached the bank in over 30 days", "Stale cheques or entries that never happened")
    if not res.flags:
        res.add_flag(GREEN, "Bank statement and books agree for the period compared")
    res.notes.append("Lines are matched first by reference, then by exact amount within 5 days when there is no usable reference. Only the period covered by both files is compared. "
                     "A line is an amount mismatch when the reference matches but the amount differs by more than 1 paisa. This finds differences to investigate; it does not post any entry.")
    return res


def rec_recon(f, flags):
    r = []
    if f.get("duplicate_bank_lines"):
        r.append(f"Ask the bank about the {f['duplicate_bank_lines']} repeated line(s) ({f.get('duplicate_bank_value')}): double debits are recoverable if raised quickly.")
    if f.get("in_bank_not_in_books"):
        r.append(f"Book the {f['in_bank_not_in_books']} bank-only line(s) ({f.get('in_bank_not_in_books_value')}): they are real money movements missing from your accounts.")
    if f.get("amount_mismatches"):
        r.append("Correct the amount differences at the source: find whether the bank or your entry is wrong.")
    if f.get("uncleared_over_30_days"):
        r.append("Follow up entries older than 30 days that never reached the bank.")
    return r or ["Books and bank agree; reconcile every month to keep it that way."]


# ============================================================================== ComplianceCalendarIQ
def _next_monthly(day: int, as_of: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(year=as_of.year, month=as_of.month, day=day)
    return d if d >= as_of.normalize() else d + pd.DateOffset(months=1)


def _next_fixed(md: list[tuple[int, int]], as_of: pd.Timestamp) -> pd.Timestamp:
    cands = [pd.Timestamp(year=y, month=m, day=d) for y in (as_of.year, as_of.year + 1) for m, d in md]
    return min(c for c in cands if c >= as_of.normalize())


def run_compliance_calendar(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("ComplianceCalendarIQ")
    caps = set(ctx.settings.get("profile") or [])
    has_profile = bool(caps)
    gst = bool(ctx.settings.get("gst_registered", True))
    emp = ("employees" in caps) if has_profile else True
    audit = bool(ctx.settings.get("tax_audit", False))
    a = ctx.as_of
    rows = []

    def add(name, due, why, note=""):
        rows.append(dict(obligation=name, due_date=due.strftime("%Y-%m-%d"), days_left=int((due - a.normalize()).days), applies_because=why, note=note))
    if gst:
        add("GSTR-1 (outward supplies)", _next_monthly(11, a), "GST registered", "Monthly filers; quarterly (QRMP) filers use the 13th after the quarter")
        add("GSTR-3B (summary return and tax payment)", _next_monthly(20, a), "GST registered", "Monthly filers; QRMP filers use the 22nd or 24th depending on state")
        add("GSTR-9 (annual return)", _next_fixed([(12, 31)], a), "GST registered", "For the previous financial year; thresholds apply")
    add("TDS / TCS deposit for last month", _next_monthly(7, a), "Deducts tax at source on payments", "Applies if you deduct TDS on salary, rent, professional fees or contractors")
    add("TDS return (quarterly)", _next_fixed([(7, 31), (10, 31), (1, 31), (5, 31)], a), "Deducts tax at source")
    if emp:
        add("PF contribution", _next_monthly(15, a), "Has employees", "Applies once PF registration is required")
        add("ESI contribution", _next_monthly(15, a), "Has employees", "Applies to eligible employees under the wage ceiling")
    add("Advance tax instalment", _next_fixed([(6, 15), (9, 15), (12, 15), (3, 15)], a), "Estimated tax due above Rs 10,000", "Cumulative 15%, 45%, 75%, 100% by these dates")
    add("Income tax return", _next_fixed([(10, 31)] if audit else [(7, 31)], a), "Every taxpayer", "31 Oct if accounts need audit, otherwise 31 Jul, for the previous financial year")
    cal = pd.DataFrame(rows).sort_values("due_date")
    res.tables["upcoming"] = cal
    soon7, soon3 = cal[cal["days_left"] <= 7], cal[cal["days_left"] <= 3]
    nxt = cal.iloc[0]
    res.facts = clean_facts(dict(
        obligations_tracked=int(len(cal)), due_within_3_days=int(len(soon3)), due_within_7_days=int(len(soon7)), due_within_30_days=int((cal["days_left"] <= 30).sum()),
        next_deadline=str(nxt["obligation"]), next_deadline_date=str(nxt["due_date"]), days_to_next_deadline=int(nxt["days_left"]),
        business_profile_used="yes" if has_profile else "no: all common obligations shown"))
    if len(soon3):
        res.add_flag(RED, f"{len(soon3)} filing(s) due within 3 days: {', '.join(soon3['obligation'].head(3))}")
    elif len(soon7):
        res.add_flag(AMBER, f"{len(soon7)} filing(s) due within a week: {', '.join(soon7['obligation'].head(3))}")
    else:
        res.add_flag(GREEN, f"Nothing due in the next week; next is {nxt['obligation']} on {nxt['due_date']}")
    res.notes.append("These are the commonly applicable Indian statutory dates, worked out from the calendar. Extensions, the QRMP scheme, state rules, holidays and notifications change them, and "
                     "BizIQ does not know what you have already filed. Confirm each date on the government portal or with your CA before relying on it. Which items show depends on your business profile.")
    return res


def rec_compliance_calendar(f, flags):
    r = []
    if f.get("due_within_7_days"):
        r.append(f"Prepare the {f['due_within_7_days']} filing(s) due this week now; late filing carries fees and interest.")
    r.append(f"Next deadline: {f.get('next_deadline')} on {f.get('next_deadline_date')}. Confirm the date with your CA.")
    return r


# ============================================================================== AssetIQ
def run_assets(data: dict, ctx: Context) -> ModuleResult:
    res = ModuleResult("AssetIQ")
    a = data["assets"].copy()
    if "useful_life_years" not in a.columns or pd.to_numeric(a["useful_life_years"], errors="coerce").notna().sum() == 0:
        raise ValueError("needs useful_life_years for the assets")
    a["status"] = a["status"].astype("string").str.lower().str.replace(" ", "_") if "status" in a.columns else "in_use"
    a = a[a["status"].fillna("in_use") != "disposed"].copy()
    if a.empty:
        raise ValueError("needs at least one asset that is not disposed")
    a["life"] = pd.to_numeric(a["useful_life_years"], errors="coerce")
    a["salvage"] = pd.to_numeric(a["salvage_value"], errors="coerce").fillna(0.0) if "salvage_value" in a.columns else 0.0
    a["age_years"] = ((ctx.as_of - a["purchase_date"]).dt.days / 365.25).clip(lower=0)
    life_ok = a["life"].notna() & (a["life"] > 0)
    frac = (a["age_years"] / a["life"]).where(life_ok).clip(upper=1)
    a["book_value"] = np.where(life_ok, np.maximum(a["salvage"], a["cost"] - (a["cost"] - a["salvage"]) * frac), np.nan)
    a["fully_depreciated"] = life_ok & (a["age_years"] >= a["life"])
    a["near_end_of_life"] = life_ok & ~a["fully_depreciated"] & (a["age_years"] >= a["life"] - 1)
    cat = a.groupby("category").agg(assets=("cost", "size"), cost=("cost", "sum"), book_value=("book_value", "sum"), average_age_years=("age_years", "mean"),
                                    fully_depreciated=("fully_depreciated", "sum"), near_end_of_life=("near_end_of_life", "sum")).round(1).reset_index().sort_values("cost", ascending=False)
    res.tables["by_category"] = cat
    old = a[a["fully_depreciated"] & (a["status"].fillna("in_use") != "under_repair")]
    res.tables["fully_depreciated_assets"] = old[["asset_id", "category", "purchase_date", "cost", "age_years"]].assign(age_years=lambda x: x["age_years"].round(1)) if len(old) else pd.DataFrame(columns=["asset_id", "category", "purchase_date", "cost", "age_years"])
    res.tables["replace_within_12_months"] = a[a["near_end_of_life"]][["asset_id", "category", "purchase_date", "cost", "age_years"]].assign(age_years=lambda x: x["age_years"].round(1))
    repair = a[a["status"] == "under_repair"]
    repl_cost = float(old["cost"].sum() + a.loc[a["near_end_of_life"], "cost"].sum())
    res.facts = clean_facts(dict(
        assets_on_register=int(len(a)), total_cost=inr(float(a["cost"].sum())), total_book_value=inr(float(a["book_value"].sum())),
        average_age_years=r1(float(a["age_years"].mean())), fully_depreciated_in_use=int(len(old)), fully_depreciated_share_pct=pct1(len(old) / len(a)),
        near_end_of_life_12m=int(a["near_end_of_life"].sum()), under_repair=int(len(repair)),
        replacement_budget_at_original_cost=inr(repl_cost), oldest_category=None if cat.empty else str(cat.sort_values("average_age_years", ascending=False).iloc[0]["category"]),
        assets_without_life=int((~life_ok).sum())))
    if len(old) and len(old) / len(a) > 0.25:
        res.add_flag(AMBER, f"{len(old)} assets ({len(old) / len(a):.0%}) are past their useful life and still in use", "Higher failure and repair risk")
    elif len(old) >= 3:
        res.add_flag(AMBER, f"{len(old)} assets are past their useful life and still in use")
    if a["near_end_of_life"].sum():
        res.add_flag(AMBER, f"{int(a['near_end_of_life'].sum())} more assets reach the end of their useful life within a year", f"Plan about {inr(repl_cost)} at original cost")
    if len(repair) >= 3:
        res.add_flag(AMBER, f"{len(repair)} assets are under repair")
    if not res.flags:
        res.add_flag(GREEN, "Asset base is well within its useful life")
    res.notes.append("Straight-line depreciation over the useful life you gave, down to salvage value: a management view of age and replacement need, not the tax depreciation under the Income Tax Act "
                     "or the Companies Act schedule. Assets without a useful life are left out of book value. Replacement cost at original cost understates today's prices.")
    return res


def rec_assets(f, flags):
    r = []
    if f.get("fully_depreciated_in_use"):
        r.append(f"Inspect the {f['fully_depreciated_in_use']} asset(s) past their useful life and decide repair, replace or retire; unplanned failures cost more than planned replacement.")
    if f.get("near_end_of_life_12m"):
        r.append(f"Budget for replacing {f['near_end_of_life_12m']} asset(s) in the next 12 months (about {f.get('replacement_budget_at_original_cost')} with those already past life, at original cost).")
    return r or ["The asset base is young; review the register once a year."]


register(ModuleSpec("PayablesIQ", "Finance and compliance", "Bills you owe: ageing, what is due soon, late and duplicate payments.", ["invoices"], run_payables, rec_payables, audience="Both", phase="4"))
register(ModuleSpec("ReconIQ", "Finance and compliance", "Bank statement against your books: what is missing, repeated or different.", ["bank_statement", "ledger_entries"], run_recon, rec_recon, audience="Both", phase="4"))
register(ModuleSpec("ComplianceCalendarIQ", "Finance and compliance", "Statutory filing dates that apply to your business and what is coming up.", [], run_compliance_calendar, rec_compliance_calendar, audience="Both", phase="4"))
register(ModuleSpec("AssetIQ", "Operations and industry", "Fixed assets: age, book value and what needs replacing.", ["assets"], run_assets, rec_assets, audience="Both", phase="4"))
