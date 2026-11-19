"""ReportIQ - executive summary, board-ready narrative, and plain-English Q&A over the data.

Answers are assembled from computed facts (never from raw data or model memory). If an
LLM is configured it may phrase answers, but only from the facts and under the same
number guard as the decision layer.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from . import decision, registry, salesiq, wealthiq
from .core import ModuleResult, inr
from .pipeline import PipelineResult, pyify


def _g(f, k, d="n/a"):
    v = f.get(k, d)
    return d if v is None else v


def module_summary(name: str, res: ModuleResult) -> str:
    """One short paragraph per module, built only from computed facts."""
    f = res.facts
    try:
        if name == "WealthIQ":
            return wealthiq.summary_text(res)
        if name == "SalesIQ":
            return salesiq.weekly_digest(res)
        if name == "ClientIQ":
            return (f"{f['customers']} customers analysed. {f['at_risk_customers']} are at high risk of churn, with "
                    f"{f['annual_revenue_at_stake']} of annual revenue attached. Retention is {f['retention_rate_pct']}% and "
                    f"the top 10% of customers generate {f['top10_pct_revenue_share_pct']}% of revenue.")
        if name == "SupplyChainIQ":
            s = (f"{f['products_tracked']} products tracked with inventory worth {f['inventory_value']}. "
                 f"{f['stockout_risk_products']} are at critical stock-out risk and {f['reorder_now_products']} have reached "
                 f"their reorder point; {f['overstocked_products']} are overstocked ({f['overstock_excess_value']} excess).")
            if f.get("weakest_supplier"):
                s += f" Weakest supplier: {f['weakest_supplier']} ({f['weakest_supplier_score']}/100)."
            return s
        if name == "CashFlowIQ":
            s = (f"Cash is {f['cash_today']} against a {f['minimum_cash_buffer']} minimum buffer. The base case reaches "
                 f"{f['base_case_cash_in_6_months']} in six months and the downside case bottoms at {f['downside_lowest_cash']}.")
            if f.get("receivables_open"):
                s += (f" Open receivables are {f['receivables_open']}, of which {f['receivables_overdue_pct']}% are overdue; "
                      f"open payables are {f['payables_open']}.")
            return s
        if name == "TaxIQ":
            return (f"For {f['financial_year']}, projected profit before tax is {f['estimated_pbt_full_year']} and estimated "
                    f"income tax is {f['estimated_income_tax']}. Estimated GST payable last month was "
                    f"{f['estimated_gst_payable_last_month']}. Next deadline: {f['next_deadline']}. Estimates only.")
        if name == "RiskIQ":
            return (f"{f.get('high_risk_customers', 0)} customer(s) and {f.get('high_risk_suppliers', 0)} supplier(s) are rated "
                    f"high risk. {f['anomalies_detected']} anomaly(ies) were detected ({f['high_severity_anomalies']} high severity). "
                    f"{f.get('single_source_products', 0)} product(s) depend on a single supplier.")
        if name == "InvestIQ":
            return (f"Surplus funds are {f['surplus_funds']} after a {f['required_cash_buffer']} buffer. Cost of debt is "
                    f"{f['cost_of_debt_pct']}%. Under the {f['risk_appetite']} template the illustrative annual income is "
                    f"{f['illustrative_annual_income_range']}."
                    + (f" The tracked portfolio is worth {f['portfolio_value']} ({f['unrealised_gain_pct']}% above cost)." if f.get("portfolio_value") else ""))
        if name == "PeopleIQ":
            return (f"Headcount is {f['headcount']} with monthly payroll of {f['monthly_payroll']} "
                    f"({_g(f, 'payroll_pct_of_revenue')}% of revenue). Attrition over 12 months is {f['attrition_rate_pct']}% "
                    + (f"and {f['staff_with_strain_signals']} people show several strain signals (a group count, not a judgement of anyone)." if f.get("staff_with_strain_signals") is not None else "(Strain signals are not reported for a team this small.)"))
        if name == "BrandIQ":
            parts = []
            if f.get("engagement_rate_pct") is not None:
                parts.append(f"Social engagement rate is {f['engagement_rate_pct']}%; {f['best_post_type']} posts perform best.")
            if f.get("avg_rating") is not None:
                parts.append(f"Average review rating is {f['avg_rating']} with {f['negative_share_pct']}% negative; "
                             f"top complaint: {f['top_complaint']}.")
            return " ".join(parts)
        if name == "ProcureIQ":
            return (f"Annual supplier spend is {f['annual_spend']}; shifting volume to the best-ranked suppliers could save up to "
                    f"{f['potential_annual_saving']}. {f['draft_purchase_orders']} draft purchase order(s) worth "
                    f"{f['draft_po_value']} await approval.")
    except KeyError as e:
        return f"{name}: summary unavailable (missing {e})."
    return registry.generic_summary(res)


def executive_summary(pr: PipelineResult) -> dict:
    flags = pr.all_flags()
    reds = [f for f in flags if f.level == "red"]
    ambers = [f for f in flags if f.level == "amber"]
    wealth = pr.modules.get("WealthIQ")
    kpis = []
    if wealth:
        kpis.append(("Financial health", f"{wealth.facts['health_score']}/100"))
        kpis.append(("Revenue (12m)", wealth.facts["revenue_ttm"]))
        kpis.append(("Net margin", f"{wealth.facts['net_margin_pct']}%"))
    cf = pr.modules.get("CashFlowIQ")
    if cf:
        kpis.append(("Cash", cf.facts["cash_today"]))
    cl = pr.modules.get("ClientIQ")
    if cl:
        kpis.append(("Revenue at churn risk", cl.facts["annual_revenue_at_stake"]))
    sc = pr.modules.get("SupplyChainIQ")
    if sc:
        kpis.append(("Stock-out risks", str(sc.facts["stockout_risk_products"])))
    headline = (f"{len(reds)} red and {len(ambers)} amber flag(s) across {len(pr.modules)} modules."
                if flags else "No flags raised.")
    return dict(kpis=kpis, headline=headline, red=reds, amber=ambers, top_issues=(reds + ambers)[:8])


def board_narrative(pr: PipelineResult) -> str:
    ex = executive_summary(pr)
    lines = [f"Management summary as at {pr.as_of:%d %B %Y}. {ex['headline']}"]
    for name, res in pr.modules.items():
        lines.append(f"{name}: {module_summary(name, res)}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- Q&A
INTENTS = [
    (("summary", "overview", "how is the business", "how are we doing", "overall", "status"), "overview"),
    (("cash", "liquidity", "runway", "receivable", "payable", "collect"), "CashFlowIQ"),
    (("profit", "margin", "health", "ratio", "z-score", "altman", "financial", "debt"), "WealthIQ"),
    (("churn", "retention", "customer", "client", "clv", "lifetime"), "ClientIQ"),
    (("sales", "revenue", "top product", "best product", "best-selling", "region", "campaign", "bundle", "basket"), "SalesIQ"),
    (("stock", "inventory", "reorder", "overstock", "forecast", "demand", "products need"), "SupplyChainIQ"),
    (("purchase order", "procure", "supplier", "vendor"), "ProcureIQ"),
    (("tax", "gst", "deadline", "advance tax", "filing"), "TaxIQ"),
    (("risk", "anomal", "fraud", "concentration", "credit"), "RiskIQ"),
    (("invest", "surplus", "portfolio", "allocation"), "InvestIQ"),
    (("employee", "payroll", "attrition", "hiring", "staff", "headcount", "overtime"), "PeopleIQ"),
    (("review", "sentiment", "social", "brand", "rating", "instagram"), "BrandIQ"),
]


def route(question: str) -> str | None:
    q = question.lower()
    best, hits = None, 0
    for keys, target in INTENTS:
        h = sum(k in q for k in keys)
        if h > hits:
            best, hits = target, h
    return best


def ask(question: str, pr: PipelineResult, use_llm: bool = False) -> dict:
    """Return {"answer": str, "module": str | None, "table": DataFrame | None, "source": str}."""
    target = route(question)
    if target is None:
        topics = ", ".join(pr.modules)
        return dict(answer=f"I can answer questions about: {topics}. Try 'How is cash?' or 'Which products are at risk of stock-out?'",
                    module=None, table=None, source="routing")
    if target == "overview":
        return dict(answer=board_narrative(pr), module=None, table=None, source="facts")
    if target not in pr.modules:
        return dict(answer=f"{target} has no results yet: " + pr.skipped.get(target, "load its data first."),
                    module=target, table=None, source="facts")
    res = pr.modules[target]
    base = module_summary(target, res)
    q = question.lower()
    table = None
    # pick the most relevant table
    prefs = [("reorder", "reorder"), ("stock", "reorder"), ("at risk", "at_risk"), ("churn", "at_risk"),
             ("deadline", "deadlines"), ("anomal", "anomalies"), ("purchase order", "draft_purchase_orders"),
             ("supplier", "suppliers"), ("receivable", "receivables_by_party"), ("top", "top_products"),
             ("region", "regions"), ("ratio", "ratios"), ("hiring", "hiring_needs"), ("employee", "departments"),
             ("review", "by_product"), ("portfolio", "portfolio")]
    for kw, tname in prefs:
        if kw in q and tname in res.tables and len(res.tables[tname]):
            table = res.tables[tname].head(10)
            break
    answer, source = base, "facts"
    if res.flags:
        top = [f for f in res.flags if f.level != "green"][:3]
        if top:
            answer += " Key flags: " + "; ".join(f.title for f in top) + "."
    if use_llm and decision.llm_available():
        prompt = (f"Question: {question}\nFacts for {target}:\n{json.dumps(pyify(res.facts), indent=1, default=str)}\n"
                  "Answer the question in 2-4 sentences using only these facts.")
        try:
            text = decision._call_llm(prompt, 400)
            ok, bad = decision.number_guard(text, decision._allowed_source(res.facts, res.flags))
            if ok:
                answer, source = text, "LLM (numbers verified against facts)"
            else:
                source = f"facts (LLM answer rejected: unverifiable numbers {bad[:3]})"
        except Exception as e:
            source = f"facts (LLM unavailable: {type(e).__name__})"
    return dict(answer=answer, module=target, table=table, source=source)
