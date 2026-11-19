"""BizIQ - AI-powered business intelligence and automation platform.

Run:  streamlit run app.py
Set BIZIQ_AUTH=0 for a single-user local demo with no login (all data in one local database).
"""
import dataclasses
import os
import warnings

import streamlit as st

warnings.filterwarnings("ignore")

from biziq import accounts, data_layer, decision, logconfig, pipeline, profile as profiles, registry, taxiq, theme, ui, ui_access, ui_billing, ui_platform  # noqa: E402

logconfig.setup()

st.set_page_config(page_title="BizIQ", page_icon="📊", layout="wide")
theme.apply()

AUTH_ON = os.environ.get("BIZIQ_AUTH", "1") != "0"
SENSITIVE_BUILTIN = {"PeopleIQ"}          # people data: needs the view_people permission

user = ui_access.login_gate() if AUTH_ON else ui_platform.LOCAL_USER
if AUTH_ON:
    accounts.activate_org(user.org_id)     # per-session: this run only sees this organisation's data
    PROFILE = accounts.get_profile(user.org_id)
    theme.brand(PROFILE.accent)
    if not PROFILE.confirmed and accounts.can(user, "manage_users"):
        ui_access.onboarding(user)         # a new business describes itself once; the right modules then switch on
        st.stop()
else:
    PROFILE = None                         # local demo: chosen in the sidebar below

# page name -> function. Built-in pages first, then one page per registered platform module.
BUILTIN = {
    "Executive summary": ui.page_exec, "Data": ui.page_data,
    "WealthIQ": ui.page_wealth, "SupplyChainIQ": ui.page_supply, "SalesIQ": ui.page_sales, "ClientIQ": ui.page_client,
    "CashFlowIQ": ui.page_cash, "TaxIQ": ui.page_tax, "RiskIQ": ui.page_risk, "InvestIQ": ui.page_invest,
    "PeopleIQ": ui.page_people, "BrandIQ": ui.page_brand, "ProcureIQ": ui.page_procure, "ReportIQ": ui.page_report,
    "Automation": ui.page_automation,
}
AREAS = {
    "Overview": ["Executive summary", "Data", "Consultant pack", "Business profile"],
    "Core modules": ["WealthIQ", "SalesIQ", "ClientIQ", "SupplyChainIQ", "CashFlowIQ", "TaxIQ", "RiskIQ", "InvestIQ",
                     "PeopleIQ", "BrandIQ", "ProcureIQ", "ReportIQ"],
}
PAGES = dict(BUILTIN)
for spec in registry.all_specs():
    PAGES[spec.name] = ui_platform.make_plugin_page(spec.name)
    AREAS.setdefault(spec.group, []).append(spec.name)
AREAS["Billing"] = list(ui_billing.PAGES_BILLING)      # not first: shops are sent here by default, everyone else keeps landing on Overview
AREAS["Automation and control"] = ["Automation", "Connections", "Approvals", "Audit log", "Team and access", "Security"]
PAGES["Business profile"] = ui_access.page_profile
PAGES["Consultant pack"] = ui_access.page_pack
PAGES.update({"New bill": ui_billing.page_new_bill, "Day book": ui_billing.page_day_book, "Customers and dues": ui_billing.page_dues, "Items and stock": ui_billing.page_items})
PAGES.update({"Approvals": ui_platform.page_approvals, "Audit log": ui_platform.page_audit, "Team and access": ui_platform.page_team,
               "Connections": ui_platform.page_connections, "Security": ui_platform.page_security})


def allowed(name: str) -> bool:
    if name in SENSITIVE_BUILTIN:
        return accounts.can(user, "view_people")
    spec = registry.get(name)
    if spec is not None:
        return accounts.can_view_module(user, spec)
    if name == "Connections":
        return accounts.can(user, "upload")
    if name == "Audit log":
        return accounts.can(user, "view_audit")
    if name == "Consultant pack":
        return accounts.can(user, "export")
    if name == "Security":
        return AUTH_ON                               # accounts, two-factor and policy only exist with login on
    if name == "Team and access":
        return True                                  # everyone can change their own password there
    return accounts.can(user, "view")


ss = st.session_state
ss.setdefault("tax_cfg", taxiq.TaxConfig())
ss.setdefault("risk_appetite", "Balanced")
ss.setdefault("use_llm", False)


def refresh():
    """Recompute everything from this organisation's database."""
    data = data_layer.load_all()
    ss["data"] = data
    ss["pr"] = pipeline.run_all(data, risk_appetite=ss["risk_appetite"], tax_cfg=ss["tax_cfg"],
                                settings={"profile": sorted(PROFILE.caps) if PROFILE else []}) if data else None
    ss["recs"] = decision.get_recommendations(ss["pr"], use_llm=ss["use_llm"]) if ss["pr"] else {}


def visible(pr):
    """Hide people modules from users who may not see them, and modules that do not apply to this business, from the summary and reports."""
    if pr is None:
        return pr
    na = {n for n in pr.modules if not profiles.applies(n, PROFILE)}
    if na:
        pr = dataclasses.replace(pr, modules={k: v for k, v in pr.modules.items() if k not in na})
    if accounts.can(user, "view_people"):
        return pr
    hidden = {n for n in pr.modules if (n in SENSITIVE_BUILTIN) or (registry.get(n) and registry.get(n).sensitive)}
    return dataclasses.replace(pr, modules={k: v for k, v in pr.modules.items() if k not in hidden},
                               skipped={k: v for k, v in pr.skipped.items() if k not in hidden},
                               failed={k: v for k, v in pr.failed.items() if k not in hidden})


if AUTH_ON and st.session_state.get("must_enroll"):         # org requires two-factor: nothing else until it is set up
    ui_platform.page_security(None, None, {"user": user})
    st.stop()

with st.sidebar:
    st.markdown('<div style="font-size:1.5rem;font-weight:780;letter-spacing:-.02em;background:linear-gradient(90deg,#fff,#22d3ee 60%,#a78bfa);-webkit-background-clip:text;background-clip:text;color:transparent">◆ BizIQ</div><div class="hint">Business intelligence &amp; automation</div>', unsafe_allow_html=True)
    ui_platform.sidebar_account(user, AUTH_ON)
    if not AUTH_ON:
        _types = {v["label"]: k for k, v in profiles.BUSINESS_TYPES.items()}
        _lab = st.selectbox("Demo as business type", list(_types), index=len(_types) - 1, help="Shows how BizIQ personalises itself: modules that do not fit are marked n/a.")
        PROFILE = profiles.Profile(_types[_lab], profiles.defaults_for(_types[_lab]), frozenset(ss.get("demo_overrides", [])))
    st.divider()
    areas = {a: [p for p in pages if allowed(p)] for a, pages in AREAS.items()}
    areas = {a: p for a, p in areas.items() if p}
    _first = "Billing" if AUTH_ON and PROFILE is not None and PROFILE.business_type in ("retail", "food", "appointments", "distributor") and "Billing" in areas else None
    area = st.selectbox("Area", list(areas), index=list(areas).index(_first) if _first else 0)
    page = st.radio("Navigate", areas[area], label_visibility="collapsed",
                    format_func=lambda n: n + ("   · n/a" if not profiles.applies(n, PROFILE) else ""))
    st.divider()
    new_risk = st.selectbox("Risk appetite (InvestIQ)", ["Conservative", "Balanced", "Growth"],
                            index=["Conservative", "Balanced", "Growth"].index(ss["risk_appetite"]))
    llm_ok = decision.llm_available()
    new_llm = st.toggle("LLM-written recommendations", value=ss["use_llm"] and llm_ok, disabled=not llm_ok,
                        help="Needs GROQ_API_KEY or ANTHROPIC_API_KEY. The model only sees computed figures, and every number it writes "
                             "is checked against them.")
    if not llm_ok:
        st.caption("LLM off: using verified templates. Set GROQ_API_KEY (or ANTHROPIC_API_KEY) to enable.")
    if new_risk != ss["risk_appetite"] or new_llm != ss["use_llm"]:
        ss["risk_appetite"], ss["use_llm"] = new_risk, new_llm
        ss.pop("pr", None)
    if st.button("Re-run analysis", disabled=not accounts.can(user, "run")):
        ss.pop("pr", None)

if ss.get("bb_dirty") and page not in ui_billing.PAGES_BILLING:
    ui_billing.sync_now()                  # bills entered since the last visit to an analysis page: bring the IQs up to date
if "pr" not in ss:
    refresh()

meta = data_layer.dataset_meta()
ctx = dict(profile=PROFILE, auth_on=AUTH_ON, refresh=refresh, tax_cfg=ss["tax_cfg"], use_llm=ss["use_llm"], user=user,
           synthetic=bool(len(meta) and meta["source"].str.contains("synthetic").any()))

pr_view = visible(ss.get("pr"))
if pr_view is None:
    if page in ("Approvals", "Audit log", "Team and access", "Connections", "Business profile") or page in ui_billing.PAGES_BILLING:
        PAGES[page](None, {}, ctx)
    elif not profiles.applies(page, PROFILE):
        ui_access.not_applicable(page, ctx)
    elif page != "Data":
        st.title("Welcome to BizIQ")
        st.info("No data loaded yet. Shops can start with **New bill** in the Billing area: every bill you enter feeds the analysis. Otherwise open **Data** to load the sample data or upload your own files.")
        st.stop()
    else:
        ui.page_data(None, {}, ctx)
elif not profiles.applies(page, PROFILE):
    ui_access.not_applicable(page, ctx)
else:
    PAGES[page](pr_view, ss["recs"], ctx)
    if page in pr_view.modules:
        ui_access.evidence_panel(page, ss.get("data", {}))
