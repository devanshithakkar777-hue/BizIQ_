"""Decision layer: turn module outputs into plain-language recommendations.

Two paths, in this order:
1. Deterministic templates built from computed facts (always available, never invents numbers).
2. Optional LLM rewrite (needs GROQ_API_KEY or ANTHROPIC_API_KEY). The model receives ONLY the computed
   facts and flags - never raw data - and every number in its answer is checked against them.
   If any number cannot be traced to the facts, the LLM text is rejected and the template
   recommendations are used instead.
"""
from __future__ import annotations

import json
import os
import re

from . import registry
from .core import ModuleResult
from .pipeline import PipelineResult, pyify

DEFAULT_MODEL = os.environ.get("BIZIQ_MODEL", "claude-sonnet-5-5")
GROQ_MODEL = os.environ.get("BIZIQ_GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def llm_provider() -> str | None:
    """'groq' or 'anthropic' (whichever key is set; BIZIQ_PROVIDER forces one), else None."""
    forced = os.environ.get("BIZIQ_PROVIDER", "").strip().lower()
    if forced == "groq":
        return "groq" if os.environ.get("GROQ_API_KEY") else None
    if forced == "anthropic":
        return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else None
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def _g(f: dict, k: str, default="n/a"):
    v = f.get(k, default)
    return default if v is None else v


# --------------------------------------------------------------------------- templates
def _wealth(f, flags):
    r = []
    if f.get("ccc_days", 0) > 60:
        r.append(f"The cash conversion cycle is {f['ccc_days']} days (inventory {f['dio_days']}, receivables "
                 f"{f['dso_days']}, payables {f['dpo_days']}). Tighten receivable collection and stock levels, "
                 f"and renegotiate supplier payment terms to release working capital.")
    if f.get("gross_margin_pct", 100) < 35:
        r.append(f"Gross margin is {f['gross_margin_pct']}%"
                 + (f", down {abs(f['gross_margin_change_6m_pts'])} points in six months" if f.get("gross_margin_change_6m_pts", 0) < -0.5 else "")
                 + ". Review supplier prices and discounting on low-margin products.")
    if f.get("current_ratio", 9) < 1.6 or f.get("quick_ratio", 9) < 1.0:
        r.append(f"Liquidity is thin (current ratio {f['current_ratio']}, quick ratio {f['quick_ratio']}). "
                 f"Avoid new fixed commitments until receivables are collected.")
    if f.get("altman_zone") in ("grey", "distress"):
        r.append(f"The Altman Z' score of {f['altman_z']} sits in the {f['altman_zone']} zone; "
                 f"reducing debt and lifting margins are the fastest ways to move it up.")
    if f.get("health_score_change_6m", 0) < -5:
        r.append(f"The health score changed by {f['health_score_change_6m']} points over six months; "
                 f"review the ratios flagged amber or red first.")
    return r or [f"Financial health is {f.get('health_score')}/100 with no pressing issues; keep monitoring monthly."]


def _sales(f, flags):
    r = []
    d = f.get("last_28d_vs_prior_12w_pct")
    if d is not None and d < -10:
        r.append(f"Sales over the last 28 days are {abs(d)}% below the previous 12-week pace. Check for stock-outs, "
                 f"pricing or campaign gaps in that window and run a recovery promotion.")
    r.append(f"{f.get('top_product')} contributes {f.get('top_product_share_pct')}% of revenue while "
             f"{f.get('bottom_product')} contributes {f.get('bottom_product_share_pct')}%; "
             f"review whether the weakest product earns its shelf space.")
    r.append(f"{f.get('weakest_region')} is the weakest region at {f.get('weakest_region_share_pct')}% of revenue; "
             f"test a regional campaign or distribution fix.")
    if f.get("top_pair"):
        r.append(f"Bundle {f['top_pair']}: they are bought together {f['top_pair_lift']} times more often than chance.")
    if f.get("best_campaign_segment"):
        r.append(f"Start with the {f['best_campaign_segment']} campaign; the estimated uplift is "
                 f"{f['best_campaign_uplift_estimate']} (an estimate based on assumed response rates).")
    return r


def _client(f, flags):
    r = []
    if f.get("at_risk_customers", 0):
        r.append(f"{f['at_risk_customers']} customers are likely to lapse, putting {f['annual_revenue_at_stake']} of annual "
                 f"revenue at stake. Send the drafted follow-ups to the highest-value ones first.")
    r.append(f"Retention is {f.get('retention_rate_pct')}%. Reward the {f.get('champions')} Champions and re-engage the "
             f"{f.get('at_risk_segment')} At Risk customers before they lapse.")
    if f.get("top10_pct_revenue_share_pct", 0) > 40:
        r.append(f"The top 10% of customers generate {f['top10_pct_revenue_share_pct']}% of revenue; "
                 f"protect these relationships with personal outreach.")
    return r


def _supply(f, flags):
    r = []
    if f.get("stockout_risk_products"):
        r.append(f"Order now: {f.get('critical_products')} will run out before new stock can arrive. "
                 f"Use the suggested quantities in the reorder table.")
    if f.get("reorder_now_products"):
        r.append(f"{f.get('reorder_products')} reached the reorder point; place orders this week.")
    if f.get("slow_moving_products") or f.get("overstocked_products"):
        r.append(f"{f.get('slow_moving_products')} slow-moving SKU(s) tie up {f.get('slow_moving_value')} and "
                 f"{f.get('overstocked_products')} product(s) are overstocked ({f.get('overstock_excess_value')} excess). "
                 f"Pause reordering and consider a clearance offer.")
    if f.get("weakest_supplier"):
        r.append(f"{f['weakest_supplier']} scores {f['weakest_supplier_score']}/100 with {f['weakest_supplier_on_time_pct']}% "
                 f"on-time delivery; add buffer to its lead times or shift volume to {f.get('best_supplier')}.")
    return r or ["Inventory levels look healthy."]


def _cash(f, flags):
    r = []
    short = [x for x in flags if x.level in ("red", "amber") and ("buffer" in x.title.lower() or "run out" in x.title.lower())]
    if short:
        r.append(f"Cash is {f.get('cash_today')} against a minimum buffer of {f.get('minimum_cash_buffer')} and the downside "
                 f"scenario falls to {f.get('downside_lowest_cash')}. Arrange a credit line or defer discretionary spend now, "
                 f"before the shortfall arrives.")
    else:
        r.append(f"Cash of {f.get('cash_today')} stays above the {f.get('minimum_cash_buffer')} buffer in every scenario "
                 f"(downside low point {f.get('downside_lowest_cash')}); no emergency funding is needed, so focus on collections.")
    if f.get("receivables_overdue_pct", 0) > 25:
        r.append(f"{f['receivables_overdue_pct']}% of open receivables ({f['receivables_overdue']}) are overdue. "
                 f"Start with {f.get('largest_overdue_party')} ({f.get('largest_overdue_amount')}); "
                 f"consider early-payment discounts or stricter terms.")
    if f.get("payables_overdue") not in (None, "n/a", "Rs 0"):
        r.append(f"Payables of {f['payables_overdue']} are overdue; clear them to protect supplier terms.")
    return r


def _tax(f, flags):
    r = [f"Estimated income tax for {f.get('financial_year')} is {f.get('estimated_income_tax')} on projected profit of "
         f"{f.get('estimated_pbt_full_year')}; keep this in the cash plan.",
         f"Estimated GST payable for the last month is {f.get('estimated_gst_payable_last_month')}; "
         f"{f.get('deadlines_next_7_days')} deadline(s) fall in the next 7 days (next: {f.get('next_deadline')})."]
    if f.get("advance_tax_shortfall"):
        r.append(f"Advance tax is short by {f['advance_tax_shortfall']}; pay the gap to limit interest, and confirm with your CA.")
    r.append("These are estimates on stated assumptions; verify rates and due dates with your CA.")
    return r


def _risk(f, flags):
    r = []
    if f.get("high_risk_customers"):
        r.append(f"{f['high_risk_customers']} customer(s) are rated high credit risk with {f['credit_exposure_to_high_risk']} "
                 f"outstanding; start with {f.get('riskiest_customer')} and tighten credit limits.")
    if f.get("anomalies_detected"):
        r.append(f"{f['anomalies_detected']} anomaly(ies) detected. Latest: {f.get('latest_anomaly')}. Investigate the cause.")
    if f.get("single_source_products"):
        r.append(f"{f['single_source_products']} product(s) rely on a single supplier; qualify a second source.")
    if f.get("high_risk_suppliers"):
        r.append(f"{f.get('riskiest_supplier')} is rated high risk; review its contract and delivery performance.")
    return r or ["No material risks detected."]


def _invest(f, flags):
    r = [f"Surplus funds above the {f.get('required_cash_buffer')} buffer are {f.get('surplus_funds')}. Under the "
         f"{f.get('risk_appetite')} template the illustrative annual income is {f.get('illustrative_annual_income_range')}."]
    if f.get("prepay_debt_worth_considering"):
        r.append(f"The cost of debt ({f.get('cost_of_debt_pct')}%) is higher than safe deposit yields; consider prepaying debt before investing.")
    if f.get("max_allocation_drift_pts", 0) > 15:
        r.append(f"The portfolio has drifted {f['max_allocation_drift_pts']} points from its template; rebalance.")
    r.append("Illustrative only, not investment advice; confirm with a qualified adviser.")
    return r


def _people(f, flags):
    r = []
    if f.get("staff_with_strain_signals"):
        r.append(f"{f['staff_with_strain_signals']} people across the business show several strain signals (overtime, low satisfaction, new joiners); "
                 "look at workload and support by department rather than at individuals.")
    if f.get("highest_overtime_hours", 0) > 15:
        r.append(f"{f.get('highest_overtime_department')} averages {f['highest_overtime_hours']} overtime hours a month; "
                 f"{f.get('suggested_hires_total')} hire(s) would relieve it.")
    if f.get("payroll_pct_of_revenue") is not None:
        r.append(f"Payroll is {f['payroll_pct_of_revenue']}% of revenue and revenue per employee is {f.get('revenue_per_employee')}.")
    return r or ["People metrics look healthy."]


def _brand(f, flags):
    r = []
    if f.get("lowest_rated_product") and f.get("lowest_rated_product_rating", 5) < 3.5:
        r.append(f"{f['lowest_rated_product']} averages {f['lowest_rated_product_rating']} stars; fix the root cause "
                 f"(top complaint overall: {f.get('top_complaint')}) before promoting it.")
    if f.get("best_post_type"):
        r.append(f"{f['best_post_type']} posts earn the best engagement ({f['best_post_type_engagement_pct']}%) and "
                 f"{f.get('best_theme')} is the strongest theme; schedule more of both.")
    r.append(f"Average rating is {f.get('avg_rating')} with {f.get('negative_share_pct')}% negative reviews; "
             f"reply to negative reviews within a day.")
    return r


def _procure(f, flags):
    r = []
    if f.get("draft_purchase_orders"):
        r.append(f"{f['draft_purchase_orders']} draft purchase order(s) worth {f['draft_po_value']} are ready for your approval.")
    if f.get("potential_annual_saving") and f.get("potential_saving_pct", 0) > 0:
        r.append(f"Shifting volume to the best-ranked suppliers could save up to {f['potential_annual_saving']} a year "
                 f"({f['potential_saving_pct']}% of spend); largest opportunity: {f.get('biggest_saving_product')}.")
    if f.get("products_with_wide_price_spread"):
        r.append(f"{f['products_with_wide_price_spread']} product(s) show a price spread above 10%; use quotes to negotiate.")
    return r or ["Procurement looks in order."]


TEMPLATES = {"WealthIQ": _wealth, "SalesIQ": _sales, "ClientIQ": _client, "SupplyChainIQ": _supply,
             "CashFlowIQ": _cash, "TaxIQ": _tax, "RiskIQ": _risk, "InvestIQ": _invest,
             "PeopleIQ": _people, "BrandIQ": _brand, "ProcureIQ": _procure}


def template_recommendations(name: str, res: ModuleResult) -> list[str]:
    fn = TEMPLATES.get(name)
    if fn is None:
        spec = registry.get(name)
        fn = spec.recommend if spec else None
    if fn is None:
        return []
    try:
        return fn(res.facts, res.flags)
    except Exception as e:  # never let a template bug break the dashboard
        return [f"Recommendations unavailable for {name} ({type(e).__name__})."]


# --------------------------------------------------------------------------- number guard
_NUM = re.compile(r"(?<![A-Za-z])-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        s = m.replace(",", "").rstrip(".")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def number_guard(text: str, allowed_source: str) -> tuple[bool, list[float]]:
    """True if every number in `text` appears in `allowed_source` (within rounding)."""
    allowed = set(_numbers(allowed_source))
    allowed |= {abs(x) for x in allowed}
    cleaned = re.sub(r"(?m)^\s*\d+[\.\)]\s+", "", text)           # drop list numbering
    cleaned = re.sub(r"\b(19|20)\d{2}\b", "", cleaned)            # years
    cleaned = re.sub(r"\bGSTR-?\s?\d[A-Za-z]?\b", "", cleaned)       # return form names (GSTR-1, GSTR-3B) are identifiers, not figures
    cleaned = re.sub(r"/\s*100\b", "", cleaned)                   # "score 45/100": 100 is a scale, not a claim
    bad = []
    for n in _numbers(cleaned):
        n = abs(n)
        if not any(abs(n - a) <= max(0.051, 0.005 * abs(a)) for a in allowed):
            bad.append(n)
    return (not bad), bad


def _allowed_source(facts: dict, flags: list) -> str:
    return json.dumps(pyify(facts), default=str) + " " + " ".join(f"{f.title} {f.detail}" for f in flags)


# --------------------------------------------------------------------------- LLM path
def llm_available() -> bool:
    prov = llm_provider()
    if prov is None:
        return False
    if prov == "groq":
        return True          # plain HTTPS, no extra package needed
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


SYSTEM = ("You are a careful business analyst writing for a small-business owner. Use ONLY the numbers in the "
          "JSON facts and flags provided. Do not calculate new figures, percentages or totals, and do not "
          "invent facts. Write 3 to 5 short, concrete recommendations in plain language as a numbered list. "
          "Label estimates as estimates.")


def _call_groq(prompt: str, max_tokens: int) -> str:
    import urllib.request
    body = json.dumps({"model": GROQ_MODEL, "max_tokens": max_tokens, "temperature": 0.2,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(GROQ_URL, data=body, headers={
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
        "Content-Type": "application/json", "User-Agent": "biziq/1.0"})
    with urllib.request.urlopen(req, timeout=45) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"].strip()


def _call_llm(prompt: str, max_tokens: int = 600) -> str:
    if llm_provider() == "groq":
        return _call_groq(prompt, max_tokens)
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(model=DEFAULT_MODEL, max_tokens=max_tokens, system=SYSTEM,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def llm_recommendations(name: str, res: ModuleResult) -> tuple[list[str] | None, str]:
    """Returns (recommendations or None, status)."""
    if not llm_available():
        return None, "LLM not configured"
    prompt = (f"Module: {name}\nFacts (computed by code):\n{json.dumps(pyify(res.facts), indent=1, default=str)}\n"
              f"Flags:\n" + "\n".join(f"- [{f.level}] {f.title}. {f.detail}" for f in res.flags[:12]) +
              "\n\nWrite the recommendations.")
    try:
        text = _call_llm(prompt)
    except Exception as e:
        return None, f"LLM call failed ({type(e).__name__})"
    ok, bad = number_guard(text, _allowed_source(res.facts, res.flags))
    if not ok:
        return None, f"LLM output rejected: numbers not found in facts {bad[:4]}"
    lines = [re.sub(r"^\s*(\d+[\.\)]|[-*])\s*", "", ln).strip() for ln in text.splitlines() if ln.strip()]
    return [ln for ln in lines if len(ln) > 15], "LLM (numbers verified against facts)"


def polish_email(draft: str, facts: dict) -> tuple[str, str]:
    """Optionally have the LLM warm up a drafted email. Numbers must come from `facts`."""
    if not llm_available():
        return draft, "template"
    prompt = ("Rewrite this customer follow-up email so it sounds warm and natural. Keep every {placeholder} and the "
              "DRAFT notice unchanged. Use only these facts and numbers: " + json.dumps(pyify(facts)) +
              "\n\nEMAIL:\n" + draft)
    try:
        text = _call_llm(prompt, 500)
    except Exception:
        return draft, "template (LLM call failed)"
    ok, _ = number_guard(text, json.dumps(pyify(facts)) + " " + draft)
    return (text, "LLM (numbers verified)") if ok and "{first_name}" in text else (draft, "template (LLM output rejected)")


def get_recommendations(pr: PipelineResult, use_llm: bool = False) -> dict[str, dict]:
    """{module: {"items": [...], "source": str}}"""
    out = {}
    for name, res in pr.modules.items():
        items, source = template_recommendations(name, res), "template"
        if use_llm:
            llm, status = llm_recommendations(name, res)
            if llm:
                items, source = llm, status
            else:
                source = f"template ({status})"
        out[name] = {"items": items, "source": source}
    return out
