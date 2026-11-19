"""Streamlit pages for the platform layer: login, generic module pages, approvals, audit log and team."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import os

from . import accounts, approvals, audit, crypto, data_layer, db, registry, session
from .accounts import AccountError, User
from . import theme
from .ui import chart, flags_block, header, notes, recs_block, table

LOCAL_USER = User(id=0, org_id="local", org_name="Local demo", email="local@biziq.local", name="Local user", role="owner")


# ------------------------------------------------------------------ login
def _start_session(u: User) -> None:
    st.session_state.clear()                       # a fresh session id of state on every login, nothing carried over
    t = session.now()
    st.session_state.update(user=u, login_at=t, last_seen=t, must_enroll=accounts.needs_2fa_enrollment(u))


def _sso_enabled() -> bool:
    return os.environ.get("BIZIQ_SSO", "").strip().lower() in ("1", "true", "yes", "on")


def login_gate() -> User:
    """Show login / sign-up until a user is signed in. Returns the signed-in user."""
    u0 = st.session_state.get("user")
    if u0:
        why = session.expired(st.session_state.get("login_at", session.now()), st.session_state.get("last_seen", session.now()))
        if why is None:
            st.session_state["last_seen"] = session.now()
            return u0
        audit.log(u0.org_id, u0.email, "session.expired", u0.email)
        st.session_state.clear()
        st.session_state["flash_login"] = why
    st.title("BizIQ")
    st.caption("Business intelligence and automation for small businesses")
    if st.session_state.get("flash_login"):
        st.info(st.session_state["flash_login"])
    first = not accounts.any_org_exists()
    tabs = st.tabs(["Create your business account"] if first else ["Log in", "Create a business account"])
    if not first:
        with tabs[0]:
            if _sso_enabled():
                if not getattr(st.user, "is_logged_in", False):
                    if st.button("Sign in with your company account", type="primary"):
                        st.login()
                    st.caption("or use your email and password below")
                else:
                    u = accounts.sso_user(st.user.get("email", ""), bool(st.user.get("email_verified", False)))
                    if u:
                        _start_session(u)
                        st.rerun()
                    st.error("Your company account signed you in, but BizIQ has no active user for that email. Ask an owner or admin to add you.")
                    if st.button("Sign out of company account"):
                        st.logout()
            with st.form("login"):
                email = st.text_input("Email")
                pw = st.text_input("Password", type="password")
                otp = st.text_input("6-digit code from your authenticator app (only if you turned two-factor on)", max_chars=14, autocomplete="one-time-code")
                go_ = st.form_submit_button("Log in", type="primary")
            if go_:
                u = None
                try:
                    u = accounts.authenticate(email, pw, otp or None)
                except accounts.TwoFactorRequired:
                    st.warning("Enter the 6-digit code from your authenticator app (or a recovery code), then log in again.")
                except AccountError as e:
                    st.error(str(e))
                else:
                    if u is None:
                        st.error("Email or password is incorrect.")
                if u:
                    _start_session(u)
                    st.rerun()
    with tabs[-1]:
        st.caption("You become the owner and can add your team afterwards. Each business's data is kept separate.")
        with st.form("signup"):
            org = st.text_input("Business name")
            name = st.text_input("Your name")
            email = st.text_input("Email", key="su_email")
            pw = st.text_input(f"Password ({accounts.MIN_PASSWORD}+ characters, with a letter and a number)", type="password", key="su_pw")
            ok = st.form_submit_button("Create account", type="primary")
        if ok:
            try:
                _start_session(accounts.create_org(org, email, name, pw))
                st.rerun()
            except AccountError as e:
                st.error(str(e))
    st.stop()


def sidebar_account(user: User, auth_on: bool) -> None:
    st.markdown(f"**{user.name}**  \n{user.org_name} · {user.role}")
    if auth_on and st.button("Log out"):
        audit.log(user.org_id, user.email, "logout", user.email)
        st.session_state.clear()
        if _sso_enabled() and getattr(st.user, "is_logged_in", False):
            st.logout()
        st.rerun()


# ------------------------------------------------------------------ generic module page
# page-specific charts: module -> (table, kind, x, y, title)
CHARTS = {
    "ProfitIQ": ("by_product", "bar", "product", "net_margin", "Net margin by product"),
    "ReceivablesIQ": ("aging", "bar", "bucket", "amount", "Receivables by days overdue"),
    "CreditIQ": ("customers", "bar", "party", "utilisation", "Share of credit limit used"),
    "ExpenseIQ": ("by_category", "bar", "category", "amount", "Spend by category"),
    "PricingIQ": ("products", "bar", "product", "margin_at_current_price", "Net margin at current price"),
    "ReturnsIQ": ("by_product", "bar", "product", "return_rate", "Return rate by product"),
    "CampaignIQ": ("channels", "bar", "channel", "roas", "Return on ad spend by channel"),
    "LoyaltyIQ": ("tiers", "bar", "tier", "revenue", "Revenue by customer tier"),
    "PipelineIQ": ("funnel", "bar", "stage", "value", "Open pipeline value by stage"),
    "QuoteIQ": ("win_rate_by_margin", "bar", "quoted_margin", "win_rate", "Win rate by quoted margin"),
    "SupportIQ": ("by_category", "bar", "category", "tickets", "Tickets by category"),
    "ContractIQ": ("contracts", "bar", "party", "days_to_notice_deadline", "Days until notice deadline"),
    "ComplianceIQ": ("suppliers_to_chase", "bar", "supplier", "itc_at_risk", "Input tax credit at risk by supplier"),
}
PCT_COLS = ("margin", "rate", "share", "utilisation", "late_risk", "gap", "ctr", "change", "conversion", "retention", "month_")


def _pct_cols(df: pd.DataFrame) -> tuple:
    return tuple(c for c in df.columns if df[c].dtype.kind == "f" and any(k in c for k in PCT_COLS))


CHARTS.update({
    "BudgetIQ": ("by_category_last_3_months", "bar", "category", "variance", "Over (+) or under (-) budget, last 3 months"),
    "SubscriptionIQ": ("monthly", "bar", "month", "mrr", "Monthly recurring revenue"),
    "SurveyIQ": ("by_month", "bar", "month", "nps", "NPS by month"),
    "AppointmentIQ": ("by_weekday", "bar", "weekday", "no_show_pct", "No-show % by weekday"),
    "RecipeIQ": ("dishes", "bar", "dish", "food_cost_pct", "Food cost % by dish"),
    "ProjectIQ": ("projects", "bar", "project_id", "cost_overrun_pct", "Forecast cost overrun % (active projects)"),
    "AssetIQ": ("by_category", "bar", "category", "book_value", "Book value by category"),
    "DiscountIQ": ("by_discount_band", "bar", "band", "discount_cost", "Discount given away by band"),
    "UnitEconomicsIQ": ("channels", "bar", "channel", "cost_per_reported_conversion", "Cost per reported conversion"),
    "PayablesIQ": ("aging", "bar", "bucket", "amount", "Bills owed by how overdue"),
})


def page_scenario_sliders(res) -> None:
    """Interactive what-if: the owner moves the sliders and sees operating profit change straight away."""
    from .modules.planning import project
    b = res.tables.get("base_year")
    if b is None or b.empty:
        return
    base = dict(revenue=float(b["revenue"].iloc[0]), cogs=float(b["cogs"].iloc[0]), opex=float(b["opex"].iloc[0]))
    st.markdown("##### Try your own what-if")
    c1, c2, c3, c4 = st.columns(4)
    price = c1.slider("Selling price change %", -30, 30, 0, key="sc_price")
    vol = c2.slider("Units sold change %", -50, 50, 0, key="sc_vol")
    cogs = c3.slider("Cost of goods change %", -30, 30, 0, key="sc_cogs")
    opex = c4.slider("Operating expenses change %", -30, 50, 0, key="sc_opex")
    now, then = project(base), project(base, price, vol, cogs, opex)
    from .core import inr
    theme.kpis([("Operating profit now", inr(now["operating_profit"]), f"{now['margin_pct']:.1f}% margin"),
                ("Operating profit with changes", inr(then["operating_profit"]), f"{then['margin_pct']:.1f}% margin"),
                ("Difference", inr(then["operating_profit"] - now["operating_profit"]))])
    fig = go.Figure(go.Bar(x=["Now", "With your changes"], y=[now["operating_profit"], then["operating_profit"]], marker_color=[theme.MUTED, theme.ACCENT]))
    fig.update_layout(title="Operating profit, trailing 12 months")
    chart(fig, 280)
    st.caption("Cost of goods moves with units sold; operating expenses stay fixed unless you change them. Customers may buy less after a price rise, which this does not model.")


def _chart_for(name: str, res) -> None:
    spec = CHARTS.get(name)
    if not spec:
        return
    tbl, kind, x, y, title = spec
    t = res.tables.get(tbl)
    if t is None or t.empty or x not in t or y not in t:
        return
    d = t.copy()
    if name == "ForecastIQ":
        return
    fig = px.bar(d, x=x, y=y, title=title)
    chart(fig, 300)


def explorer(name: str, res) -> None:
    """Interactive view of every table a module returns: choose chart type, axes and how many rows; the table sits underneath (searchable, sortable)."""
    names = [k for k, t in res.tables.items() if isinstance(t, pd.DataFrame) and len(t)]
    if not names:
        return
    tabs = st.tabs([n.replace("_", " ").capitalize() for n in names])
    for tab, n in zip(tabs, names):
        with tab:
            t = res.tables[n].reset_index(drop=True)
            num = [c for c in t.columns if t[c].dtype.kind in "fi" and t[c].nunique() > 1]
            dim = [c for c in t.columns if c not in num and t[c].dtype.kind in "OMmSUb" or t[c].dtype.kind in "M"]
            key = f"{name}_{n}"
            if num and dim and len(t) > 1:
                c1, c2, c3, c4 = st.columns([1.1, 1.3, 1.3, 1])
                kind = c1.selectbox("Chart", ["Bar", "Line", "Scatter", "Histogram"], key=key + "_k")
                x = c2.selectbox("Category / time" if kind != "Histogram" else "Group by", dim, key=key + "_x")
                y = c3.selectbox("Measure", num, key=key + "_y")
                top = c4.slider("Rows shown", 3, max(4, min(60, len(t))), min(15, max(4, len(t))), key=key + "_n") if len(t) > 6 else len(t)
                d = t.copy()
                if kind == "Bar":
                    d = d.sort_values(y, ascending=False).head(top)
                    fig = px.bar(d, x=x, y=y, color=y, color_continuous_scale=[[0, "#164e63"], [1, theme.ACCENT]])
                    fig.update_layout(coloraxis_showscale=False)
                elif kind == "Line":
                    fig = px.line(d.sort_values(x) if d[x].dtype.kind in "Mmifu" else d, x=x, y=y, markers=True)
                elif kind == "Scatter":
                    other = [c for c in num if c != y] or [y]
                    x2 = c2.selectbox("X measure", other, key=key + "_x2") if False else other[0]
                    fig = px.scatter(d, x=x2, y=y, hover_name=x, size=None)
                else:
                    fig = px.histogram(d, x=y, nbins=20)
                chart(fig, 330)
            table(t, pct=_pct_cols(t), height=min(420, 38 + 35 * len(t)))


def page_forecast_chart(res) -> None:
    hist, fc = res.tables["history"], res.tables["revenue_forecast"]
    fig = go.Figure()
    fig.add_scatter(x=hist["month"], y=hist["revenue"], name="Actual", mode="lines+markers")
    fig.add_scatter(x=fc["month"], y=fc["forecast"], name="Forecast", mode="lines+markers", line=dict(dash="dash"))
    fig.add_scatter(x=list(fc["month"]) + list(fc["month"])[::-1], y=list(fc["high"]) + list(fc["low"])[::-1], fill="toself",
                    name="Likely range", line=dict(width=0), opacity=0.2)
    fig.update_layout(title="Monthly revenue: actual and forecast")
    chart(fig, 340)


def make_plugin_page(name: str):
    spec = registry.get(name)

    def page(pr, recs, ctx):
        res = header(name, pr, recs, spec.blurb, body=False)
        if res is None:
            return
        user = ctx.get("user")
        if spec.group == "People":
            st.info("Signals, not judgements. This checks the process and the groups, not any person. It cannot know about illness, family trouble or what a manager knows. Do not use it to decide about an individual.")
        theme.kpis([(k, v) for k, v in list(res.facts.items())[:8]])
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("##### Findings")
            theme.feed(res.flags, limit=8, show_module=False)
        with right:
            st.markdown("##### What to do")
            recs_block(name, recs)
        if name == "ForecastIQ":
            page_forecast_chart(res)
        elif name == "ScenarioIQ":
            page_scenario_sliders(res)
        else:
            _chart_for(name, res)
        explorer(name, res)
        if name == "ReceivablesIQ" and accounts.can(user, "request") and "reminder_drafts" in res.tables and len(res.tables["reminder_drafts"]):
            with st.container(border=True):
                st.markdown("**Send for approval**")
                st.caption("Reminders are never sent automatically. Submit them so another approver can review before anything goes out.")
                if st.button("Submit reminder drafts for approval"):
                    d = res.tables["reminder_drafts"]
                    approvals.request(user, "payment_reminder", f"{len(d)} payment reminder drafts",
                                      {"drafts": d.to_dict("records")})
                    st.success("Submitted. An approver can review it on the Approvals page.")
        notes(res)

    return page


# ------------------------------------------------------------------ approvals
def page_approvals(pr, recs, ctx):
    user: User = ctx["user"]
    st.title("Approvals")
    st.caption("Anything that leaves the business or commits money goes through request, review and approve, with a saved record.")
    if accounts.can(user, "request"):
        with st.expander("New request"):
            with st.form("new_req"):
                kind = st.selectbox("Type", approvals.KINDS)
                title = st.text_input("Short title")
                detail = st.text_area("Details (what will happen if approved)")
                sent = st.form_submit_button("Submit for approval")
            if sent:
                try:
                    approvals.request(user, kind, title, {"details": detail})
                    st.success("Submitted.")
                    st.rerun()
                except AccountError as e:
                    st.error(str(e))
    pending = approvals.list_requests(user, "pending")
    st.subheader(f"Waiting for review ({len(pending)})")
    for _, r in pending.iterrows():
        with st.container(border=True):
            st.markdown(f"**#{r['id']} · {r['title']}**  \n{r['kind']} · requested by {r['requested_by']} on {r['requested_at'][:10]}")
            payload = approvals.get_payload(user, int(r["id"]))
            if payload.get("details"):
                st.write(payload["details"])
            if payload.get("drafts"):
                st.dataframe(pd.DataFrame(payload["drafts"])[["party", "tone", "overdue_amount"]], hide_index=True, width="stretch")
            if accounts.can(user, "approve"):
                note = st.text_input("Note (optional)", key=f"note_{r['id']}")
                a, b = st.columns(2)
                for col, ok, label in ((a, True, "Approve"), (b, False, "Reject")):
                    if col.button(label, key=f"{label}_{r['id']}"):
                        try:
                            approvals.decide(user, int(r["id"]), ok, note)
                            st.rerun()
                        except AccountError as e:
                            st.error(str(e))
    st.subheader("History")
    hist = approvals.list_requests(user)
    table(hist[hist["status"] != "pending"])


# ------------------------------------------------------------------ audit log
def page_audit(pr, recs, ctx):
    user: User = ctx["user"]
    st.title("Audit log")
    st.caption("Who did what, and when. Entries cannot be edited or deleted, and each one is chained to the one before it.")
    if not accounts.can(user, "view_audit"):
        st.info("Your role cannot view the audit log.")
        return
    ok, msg = audit.verify_chain(user.org_id)
    (st.success if ok else st.error)(msg)
    df = audit.read(user.org_id, limit=1000)
    table(df, height=480)
    st.download_button("Download audit log (CSV)", df.to_csv(index=False), file_name="biziq_audit_log.csv", mime="text/csv")


# ------------------------------------------------------------------ team
def page_team(pr, recs, ctx):
    user: User = ctx["user"]
    st.title("Team and access")
    st.caption("Roles decide what each person can see and do. Salary and people data needs the HR, admin or owner role.")
    st.markdown("| Role | Can |\n|---|---|\n"
                "| owner, admin | everything, including users, approvals, audit log and people data |\n"
                "| analyst | view, upload data, run analysis, export, submit requests |\n"
                "| hr | view, people modules, submit requests |\n"
                "| viewer | view non-sensitive modules only |")
    if accounts.can(user, "manage_users"):
        users = accounts.list_users(user)
        table(pd.DataFrame(users))
        with st.expander("Add a team member"):
            with st.form("add_user"):
                nm, em = st.text_input("Name"), st.text_input("Email")
                role = st.selectbox("Role", [r for r in accounts.ROLES if r != "owner"])
                pw = st.text_input("Temporary password", type="password")
                ok = st.form_submit_button("Add")
            if ok:
                try:
                    accounts.add_user(user, em, nm, role, pw)
                    st.success("Added. Ask them to change the password after first login.")
                    st.rerun()
                except AccountError as e:
                    st.error(str(e))
        with st.expander("Change a role or deactivate"):
            em = st.selectbox("Person", [u["email"] for u in users if u["active"]])
            c1, c2 = st.columns(2)
            new_role = c1.selectbox("New role", accounts.ROLES)
            if c1.button("Change role"):
                try:
                    accounts.set_role(user, em, new_role)
                    st.rerun()
                except AccountError as e:
                    st.error(str(e))
            if c2.button("Deactivate account"):
                try:
                    accounts.deactivate_user(user, em)
                    st.rerun()
                except AccountError as e:
                    st.error(str(e))
    if user.role == "owner":
        allowed = accounts.self_approval_allowed(user.org_id)
        new = st.toggle("Allow people to approve their own requests (for one-person teams)", value=allowed)
        if new != allowed:
            accounts.set_self_approval(user, new)
            st.rerun()
    with st.expander("Change my password"):
        with st.form("pw"):
            old, new = st.text_input("Current password", type="password"), st.text_input("New password", type="password")
            ok = st.form_submit_button("Change")
        if ok:
            try:
                accounts.change_password(user, old, new)
                st.success("Password changed.")
            except AccountError as e:
                st.error(str(e))


# ------------------------------------------------------------------ live connections
def page_connections(pr, recs, ctx):
    from . import connectors, data_layer
    from .live import store, sync, whatsapp
    from .live.tally import TallyXmlFileConnector

    user: User = ctx["user"]
    org = user.org_id
    can_edit = accounts.can(user, "upload")
    st.title("Connections")
    st.caption("Pull data straight from your tools instead of uploading files. Built from the vendors' public documentation and tested "
               "on simulated responses only: do your first sync on a test store or company and check the numbers.")
    if not store.secrets_available():
        st.warning("Tokens cannot be saved inside BizIQ until an encryption key is set (BIZIQ_SECRET_KEY). Until then, set them as "
                   "environment variables on the server, for example BIZIQ_SHOPIFY_TOKEN. Other settings can still be saved here.")

    tabs = st.tabs(["Data sources", "Import a file (Tally / myBillBook XML)", "WhatsApp reminders", "Status"])

    def settings_form(provider: str, fields: list, key: str):
        cur = store.load(org, provider, [f[0] for f in fields])
        vals, secrets = {}, {}
        with st.form(key):
            for k, label, secret, default in fields:
                if secret:
                    have = bool(cur.get(k))
                    secrets[k] = st.text_input(label + (" (saved; leave blank to keep)" if have else ""), type="password",
                                               disabled=not (store.secrets_available() and can_edit), key=f"{key}_{k}")
                else:
                    vals[k] = st.text_input(label, value=str(cur.get(k, default)), disabled=not can_edit, key=f"{key}_{k}")
            saved = st.form_submit_button("Save settings", disabled=not can_edit)
        if saved:
            try:
                store.save(org, provider, vals, {k: v for k, v in secrets.items() if v})
                audit.log(org, user.email, "connection.save", provider, "settings updated" + (" (secrets changed)" if any(secrets.values()) else ""))
                st.success("Saved.")
            except Exception as e:
                st.error(str(e))

    with tabs[0]:
        provider = st.selectbox("Source", list(sync.PROVIDERS), format_func=lambda p: sync.PROVIDERS[p]["label"])
        if provider == "tally":
            st.info("Tally must be running with a company loaded and its HTTP server on. BizIQ has to be able to reach that address, "
                    "so a cloud copy of BizIQ cannot see Tally on your office PC.")
        settings_form(provider, sync.PROVIDERS[provider]["fields"], f"set_{provider}")
        a, b = st.columns(2)
        if a.button("Test connection"):
            try:
                conn = sync.build(org, provider)
                st.success(conn.test() if hasattr(conn, "test") else "Settings look complete.")
            except Exception as e:
                st.error(str(e))
        if b.button("Sync now", type="primary", disabled=not can_edit):
            with st.spinner("Pulling data..."):
                lines = sync.sync(org, user.email, provider)
            for ln in lines:
                (st.warning if any(w in ln for w in ("not saved", "failed", "problem", "left as it is")) else st.success)(ln)
            ctx["refresh"]()
        st.caption("A scheduled sync runs daily at 06:30 when scripts/run_scheduler.py is running. A sync replaces that dataset, "
                   "and never replaces it with an empty result.")

    with tabs[1]:
        st.write("Upload a Tally XML export, or the Tally-compatible XML that myBillBook emails you (Settings > Data export to Tally). "
                 "Both sales and receivables/payables are read from it.")
        days = st.number_input("Credit days to assume when a bill has none", 0, 365, 30)
        up = st.file_uploader("XML file", type=["xml"], disabled=not can_edit)
        if up is not None and st.button("Import", type="primary", disabled=not can_edit):
            try:
                raw = up.getvalue()
                for ds in ("sales", "invoices"):
                    conn = TallyXmlFileConnector(__import__("io").BytesIO(raw), up.name, int(days))
                    rep = connectors.ingest_from(conn, ds, user)
                    (st.success if rep.ok else st.error)(f"{ds}: " + (f"saved {rep.rows_out} of {rep.rows_in} rows." if rep.ok else "; ".join(rep.errors)[:300]))
                    for n in conn.notes:
                        st.caption("Note: " + n)
                ctx["refresh"]()
            except Exception as e:
                st.error(str(e))

    with tabs[2]:
        live = whatsapp.live_sending_enabled()
        (st.error if live else st.info)("LIVE: approved reminders will really be sent." if live else
                                        "DRY RUN: nothing is sent. Set BIZIQ_SEND_WHATSAPP=1 on the server to send for real.")
        st.caption("Meta only lets a business start a chat with an approved template, and only to people who opted in. Your template "
                   "needs two variables: {{1}} customer name, {{2}} amount overdue.")
        settings_form("whatsapp", sync.WHATSAPP_FIELDS, "set_whatsapp")
        if st.button("Test WhatsApp connection"):
            try:
                st.success(sync.build_whatsapp(org).test())
            except Exception as e:
                st.error(str(e))
        st.subheader("Contacts")
        inv = data_layer.load_dataset("invoices")
        known = whatsapp.load_contacts()
        parties = sorted(set(inv[inv["type"] == "receivable"]["party"]) if inv is not None and len(inv) else set())
        base = pd.concat([known, pd.DataFrame({"party": [p for p in parties if p not in set(known["party"])]})], ignore_index=True)
        base["whatsapp_opt_in"] = base["whatsapp_opt_in"].fillna(0).astype(bool)
        base["phone"] = base["phone"].fillna("")
        edited = st.data_editor(base, hide_index=True, width="stretch", disabled=not can_edit,
                                column_config={"whatsapp_opt_in": st.column_config.CheckboxColumn("Opted in to WhatsApp")})
        st.caption("Tick opt-in only for customers who agreed to get WhatsApp messages from you. Numbers without a country code are treated as Indian.")
        if st.button("Save contacts", disabled=not can_edit):
            st.success(f"Saved {whatsapp.save_contacts(edited)} contacts.")
        st.subheader("Send")
        if accounts.can(user, "approve"):
            if st.button("Send approved payment reminders"):
                try:
                    st.write(whatsapp.dispatch_approved_reminders(user, sync.build_whatsapp(org)))
                except Exception as e:
                    st.error(str(e))
        else:
            st.info("Only an owner or admin can send. Reminders must first be approved on the Approvals page.")
        table(whatsapp.send_log())

    with tabs[3]:
        rows = store.status(org)
        table(pd.DataFrame(rows) if rows else pd.DataFrame(columns=["provider", "saved", "last_sync", "last_status"]))


# ------------------------------------------------------------------ security
def _qr_svg(uri: str) -> str | None:
    try:
        import io
        import segno
        buf = io.BytesIO()
        segno.make(uri, error="m").save(buf, kind="svg", scale=5, border=2)
        return buf.getvalue().decode()
    except Exception:
        return None


def page_security(pr, recs, ctx):
    user: User = ctx["user"]
    st.title("Security")
    forced = st.session_state.get("must_enroll")
    if forced:
        st.warning("Your organisation requires two-factor sign-in. Set it up below to continue.")
    st.subheader("Two-factor sign-in")
    if accounts.two_factor_enabled(user):
        st.success("Two-factor is ON for your account.")
        with st.expander("Get new recovery codes"):
            with st.form("regen"):
                pw = st.text_input("Your password", type="password")
                go_ = st.form_submit_button("Replace my recovery codes")
            if go_:
                try:
                    st.session_state["codes"] = accounts.regenerate_recovery_codes(user, pw)
                except AccountError as e:
                    st.error(str(e))
        if not accounts.two_factor_required(user.org_id):
            with st.expander("Turn two-factor off"):
                with st.form("off"):
                    pw = st.text_input("Your password", type="password", key="off_pw")
                    go_ = st.form_submit_button("Turn off")
                if go_:
                    try:
                        accounts.disable_2fa(user, password=pw)
                        st.rerun()
                    except AccountError as e:
                        st.error(str(e))
    else:
        if not crypto.available():
            st.error("The server has no encryption key (BIZIQ_MASTER_KEY), so two-factor secrets cannot be stored. Ask whoever runs BizIQ to set it.")
        elif "totp_pending" not in st.session_state:
            if st.button("Set up two-factor", type="primary"):
                try:
                    st.session_state["totp_pending"] = accounts.begin_2fa_enrollment(user)
                    st.rerun()
                except AccountError as e:
                    st.error(str(e))
        if "totp_pending" in st.session_state:
            secret, uri = st.session_state["totp_pending"]
            st.markdown("1. Install an authenticator app (Google Authenticator, Microsoft Authenticator, Authy, 1Password).  \n"
                        "2. Scan this code, or type the key by hand.  \n3. Enter the 6-digit code it shows.")
            svg = _qr_svg(uri)
            if svg:
                st.markdown(f'<div style="max-width:220px">{svg}</div>', unsafe_allow_html=True)
            st.code(secret, language=None)
            with st.form("confirm2fa"):
                code = st.text_input("6-digit code", max_chars=8)
                go_ = st.form_submit_button("Turn on two-factor", type="primary")
            if go_:
                try:
                    st.session_state["codes"] = accounts.confirm_2fa_enrollment(user, code)
                    st.session_state.pop("totp_pending", None)
                    st.session_state["must_enroll"] = False
                    st.rerun()
                except AccountError as e:
                    st.error(str(e))
    if st.session_state.get("codes"):
        st.warning("Save these recovery codes somewhere safe now. Each works once if you lose your phone. They will not be shown again.")
        st.code("\n".join(st.session_state["codes"]), language=None)
        if st.button("I have saved them"):
            st.session_state.pop("codes", None)
            st.rerun()

    if user.role in ("owner", "admin"):
        st.subheader("Organisation policy")
        req = accounts.two_factor_required(user.org_id)
        new = st.toggle("Require two-factor for everyone", value=req,
                        help="People without it are asked to set it up at their next login. Does not apply to company single sign-on, where your identity provider enforces it.")
        if new != req:
            try:
                accounts.set_require_2fa(user, new)
                st.rerun()
            except AccountError as e:
                st.error(str(e))
        st.subheader("Encryption and storage")
        enc = data_layer.encrypt_on()
        st.markdown(f"* Database: **{db.backend_name()}**  \n"
                    f"* Master key configured: **{'yes' if crypto.available() else 'NO'}**  \n"
                    f"* Business data encrypted at rest (AES-256-GCM): **{'on' if enc else 'off'}** "
                    f"{'' if enc else '(set BIZIQ_ENCRYPT_DATA=1, then run scripts/rotate_keys.py encrypt)'}  \n"
                    f"* Saved connection secrets and two-factor secrets: **{'encrypted' if crypto.available() else 'cannot be stored without a key'}**  \n"
                    f"* Sessions end after **{session.IDLE_MINUTES:g} min** idle or **{session.MAX_HOURS:g} h** in total  \n"
                    f"* Single sign-on: **{'on' if _sso_enabled() else 'off'}**")
        if crypto.available() and user.role == "owner":
            with st.expander("Data key status"):
                st.dataframe(pd.DataFrame([k for k in crypto.key_status() if k["scope"] in (f"secrets:{user.org_id}", data_layer.current_scope())]),
                             hide_index=True)
