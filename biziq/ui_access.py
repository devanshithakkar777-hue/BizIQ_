"""Sign-up, sign-in and business-profile screens.

BizIQ is one central platform. Each business gets its own login page at  <BizIQ address>/?org=<its login name>  that shows the business's
own name and colour, and only lets that business's people in. Data is stored per business, so it is still there at the next login.
"""
from __future__ import annotations

import re

import streamlit as st

from . import accounts, audit, consultant_pack, data_layer, evidence, exporter, mailer, profile as profiles, session, theme
from .accounts import AccountError, User
from .ui_platform import _sso_enabled, _start_session


def _clear(*keys):
    for k in keys:
        if k in st.query_params:
            del st.query_params[k]


def _centered():
    return st.columns([1, 1.6, 1])[1]


def _brand_hero(title: str, sub: str):
    st.markdown(f'<div style="text-align:center;margin:1.5rem 0 1rem"><div class="eyebrow" style="color:var(--acc);letter-spacing:.2em;font-size:.75rem;font-weight:650">◆ BIZIQ</div>'
                f'<h1 style="font-size:2.3rem;margin:.2rem 0">{theme.esc(title)}</h1><div class="hint">{theme.esc(sub)}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ the gate
def login_gate() -> User:
    """Show the right sign-in / sign-up screen until someone is signed in. Returns the signed-in user."""
    u0 = st.session_state.get("user")
    if u0:
        why = session.expired(st.session_state.get("login_at", session.now()), st.session_state.get("last_seen", session.now()))
        if why is None:
            st.session_state["last_seen"] = session.now()
            return u0
        audit.log(u0.org_id, u0.email, "session.expired", u0.email)
        st.session_state.clear()
        st.session_state["flash_login"] = why
    qp = st.query_params
    org = accounts.org_by_slug(qp.get("org", "")) if qp.get("org") else None
    if org:
        theme.brand(org["accent"])
    with _centered():
        if st.session_state.get("flash_login"):
            st.info(st.session_state["flash_login"])
        if qp.get("verify"):
            _verify_screen(qp.get("verify"), org)
        elif qp.get("reset"):
            _reset_screen(qp.get("reset"), org)
        elif qp.get("invite"):
            _invite_screen(qp.get("invite"))
        elif qp.get("org") and not org:
            _brand_hero("Business not found", "Check the login link you were given.")
            if st.button("Go to BizIQ home"):
                _clear("org")
                st.rerun()
        elif org:
            _business_login(org)
        else:
            _central()
    st.stop()


def _login_form(org: dict | None):
    if _sso_enabled() and not getattr(st.user, "is_logged_in", False):
        if st.button("Sign in with your company account", type="primary"):
            st.login()
        st.caption("or use your email and password")
    elif _sso_enabled():
        u = accounts.sso_user(st.user.get("email", ""), bool(st.user.get("email_verified", False)))
        if u and (not org or u.org_slug == org["slug"]):
            _start_session(u)
            st.rerun()
        st.error("Your company account signed you in, but BizIQ has no active user for that email. Ask an owner or admin to invite you.")
        if st.button("Sign out of company account"):
            st.logout()
    with st.form("login"):
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        otp = st.text_input("Authenticator code (only if you turned two-factor on)", max_chars=14, autocomplete="one-time-code")
        go_ = st.form_submit_button("Log in", type="primary")
    if go_:
        u = None
        try:
            u = accounts.authenticate(email, pw, otp or None, org_slug=org["slug"] if org else None)
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
    with st.expander("Forgot your password?"):
        with st.form("forgot"):
            fe = st.text_input("Your email", key="forgot_email")
            sent = st.form_submit_button("Email me a reset link")
        if sent:
            try:
                st.success(accounts.request_password_reset(fe))
                if not mailer.can_deliver():
                    st.caption("Email sending is not set up on this server, so no email was sent. Ask your BizIQ administrator.")
            except AccountError as e:
                st.error(str(e))


def _business_login(org: dict):
    _brand_hero(f"Sign in to {org['name']}", org["tagline"] or "Your business dashboard")
    _login_form(org)
    st.markdown('<div class="hint" style="text-align:center;margin-top:1rem">New here? Ask your administrator to invite you.</div>', unsafe_allow_html=True)
    if st.button("Not your business? Go to BizIQ home", type="tertiary"):
        _clear("org")
        st.rerun()


def _central():
    _brand_hero("BizIQ", "Business intelligence and automation, set up around your business")
    first = not accounts.any_org_exists()
    tabs = st.tabs(["Create your business account"] if first else ["Log in", "Create a business account"])
    if not first:
        with tabs[0]:
            with st.form("find"):
                slug = st.text_input("Your business login name (optional)", placeholder="e.g. acme-traders", help="Opens your business's own login page.")
                go_ = st.form_submit_button("Open my business login page")
            if go_ and slug.strip():
                st.query_params["org"] = re.sub(r"[^a-z0-9-]", "", slug.strip().lower())
                st.rerun()
            st.divider()
            _login_form(None)
    with tabs[-1]:
        st.caption("You become the owner and can invite your team. Each business's data is kept separate, and stays saved for your next login.")
        types = {v["label"]: k for k, v in profiles.BUSINESS_TYPES.items()}
        with st.form("signup"):
            org = st.text_input("Business name")
            label = st.selectbox("What kind of business is it?", list(types), help="BizIQ shows only the modules that fit. You can fine-tune this after signing up.")
            slug = st.text_input("Business login name (optional)", placeholder="leave blank to create one from the name",
                                 help="Your team will sign in at ...?org=this-name")
            name = st.text_input("Your name")
            email = st.text_input("Email", key="su_email")
            pw = st.text_input(f"Password ({accounts.MIN_PASSWORD}+ characters, with a letter and a number)", type="password", key="su_pw")
            ok = st.form_submit_button("Create account", type="primary")
        if ok:
            try:
                res = accounts.signup(org, name, email, pw, types[label], slug or None)
            except AccountError as e:
                st.error(str(e))
            else:
                if res.user:
                    st.query_params["org"] = res.user.org_slug
                    _start_session(res.user)
                    st.rerun()
                st.success(res.message)


def _verify_screen(token: str, org):
    _brand_hero("Confirming your email", "")
    u = accounts.verify_email(token)
    _clear("verify")
    if u:
        st.success("Email confirmed. You can log in now.")
    else:
        st.error("This confirmation link is not valid or has expired. Log in to request a new one, or sign up again.")
    if st.button("Continue to log in", type="primary"):
        st.rerun()


def _reset_screen(token: str, org):
    _brand_hero("Choose a new password", "")
    with st.form("reset"):
        p1 = st.text_input("New password", type="password")
        p2 = st.text_input("Repeat it", type="password")
        ok = st.form_submit_button("Save password", type="primary")
    if ok:
        if p1 != p2:
            st.error("The two passwords do not match.")
        else:
            try:
                accounts.reset_password(token, p1)
                _clear("reset")
                st.success("Password changed. You can log in now.")
                st.rerun()
            except AccountError as e:
                st.error(str(e))


def _invite_screen(token: str):
    info = accounts.invite_info(token)
    if not info:
        _brand_hero("Invitation not valid", "It may have expired or already been used. Ask your administrator for a new one.")
        return
    _brand_hero(f"Join {info['org_name']}", f"You were invited as {info['role']} ({info['email']}).")
    with st.form("accept"):
        nm = st.text_input("Your name", value=info["name"])
        p1 = st.text_input(f"Choose a password ({accounts.MIN_PASSWORD}+ characters, with a letter and a number)", type="password")
        ok = st.form_submit_button("Create my account", type="primary")
    if ok:
        try:
            u = accounts.accept_invite(token, p1, nm)
        except AccountError as e:
            st.error(str(e))
        else:
            _clear("invite")
            st.query_params["org"] = u.org_slug
            _start_session(u)
            st.rerun()


# ------------------------------------------------------------------ after login: describe the business
def _cap_defaults(business_type: str):
    caps = profiles.defaults_for(business_type)
    for k in profiles.CAPABILITIES:
        st.session_state[f"cap_{k}"] = k in caps


def _profile_editor(prefix: str, current: profiles.Profile):
    types = {v["label"]: k for k, v in profiles.BUSINESS_TYPES.items()}
    labels = list(types)
    cur_label = profiles.BUSINESS_TYPES[current.business_type]["label"]
    if f"cap_{next(iter(profiles.CAPABILITIES))}" not in st.session_state:
        for k in profiles.CAPABILITIES:
            st.session_state[f"cap_{k}"] = k in current.caps
    label = st.selectbox("Type of business", labels, index=labels.index(cur_label), key=f"{prefix}_type",
                         on_change=lambda: _cap_defaults(types[st.session_state[f"{prefix}_type"]]))
    st.markdown("**What does your business do?** Switch off what does not apply; each module follows these.")
    cols = st.columns(2)
    for i, (k, (lab, hlp)) in enumerate(profiles.CAPABILITIES.items()):
        cols[i % 2].checkbox(lab, key=f"cap_{k}", help=hlp)
    return types[label], [k for k in profiles.CAPABILITIES if st.session_state.get(f"cap_{k}")]


def onboarding(user: User):
    """First login for a new business: confirm what it does so the right modules switch on."""
    cur = accounts.get_profile(user.org_id)
    theme.hero("Tell us about your business", "BizIQ is one platform for every kind of business. This shows you the modules that fit yours and hides the rest.", eyebrow="Welcome")
    with st.container(border=True):
        bt, caps = _profile_editor("ob", cur)
        accent = st.color_picker("Your accent colour (optional)", cur.accent)
        tag = st.text_input("A line shown on your login page (optional)", cur.tagline, max_chars=120)
        if st.button("Save and continue", type="primary"):
            accounts.set_profile(user, bt, caps, accent, tag)
            for k in profiles.CAPABILITIES:
                st.session_state.pop(f"cap_{k}", None)
            st.rerun()
    st.caption("You can change all of this later under Business profile.")


def page_profile(pr, recs, ctx):
    user: User = ctx["user"]
    prof: profiles.Profile = ctx["profile"]
    theme.hero("Business profile", "What your business does decides which modules apply. Data you upload stays saved for every login.", eyebrow="Your business")
    if not ctx.get("auth_on", True):
        st.info("Local demo mode: pick a business type in the sidebar to see how BizIQ personalises itself. Profiles are saved per business when login is on.")
    org = accounts.org_by_slug(user.org_slug) if user.org_slug else None
    if org:
        link = f"{mailer.base_url()}/?org={org['slug']}"
        st.markdown(f"Your team's own login page: `{link}`")
    if accounts.can(user, "manage_users"):
        with st.container(border=True):
            bt, caps = _profile_editor("pf", prof)
            c1, c2 = st.columns(2)
            accent = c1.color_picker("Accent colour", prof.accent)
            tag = c2.text_input("Login page line", prof.tagline, max_chars=120)
            if st.button("Save profile", type="primary"):
                accounts.set_profile(user, bt, caps, accent, tag)
                for k in profiles.CAPABILITIES:
                    st.session_state.pop(f"cap_{k}", None)
                st.success("Saved.")
                st.rerun()
    else:
        st.write(f"Business type: **{profiles.BUSINESS_TYPES[prof.business_type]['label']}**. Only an owner or admin can change it.")
    export_panel(user)
    names = sorted(profiles.MODULE_CAPS)
    on = [n for n in names if profiles.applies(n, prof)]
    off = [n for n in names if not profiles.applies(n, prof)]
    st.subheader(f"{len(on)} modules apply, {len(off)} do not")
    with st.expander("Not applicable for your business model"):
        st.write(", ".join(off) or "None.")
    if prof.overrides:
        st.caption("Enabled anyway: " + ", ".join(sorted(prof.overrides)))
    if accounts.can(user, "manage_users") and ctx.get("auth_on", True):
        st.subheader("Invite your team")
        st.caption("They choose their own password from the link. Nobody else ever sees it.")
        with st.form("invite"):
            c1, c2, c3 = st.columns(3)
            nm, em = c1.text_input("Name"), c2.text_input("Email")
            role = c3.selectbox("Role", [r for r in accounts.ROLES if r != "owner"])
            go_ = st.form_submit_button("Send invitation")
        if go_:
            try:
                link = accounts.invite_user(user, em, nm, role)
                st.success("Invitation created." + (" It was also emailed." if mailer.can_deliver() else " Email sending is off, so send them this link yourself:"))
                if not mailer.can_deliver():
                    st.code(link)
            except AccountError as e:
                st.error(str(e))


def not_applicable(name: str, ctx):
    user, prof = ctx["user"], ctx["profile"]
    theme.hero(name, "", eyebrow="Module")
    with st.container(border=True):
        st.markdown("### Not applicable for your business model")
        st.write(profiles.why_not(name, prof).replace("Not applicable for your business model. ", ""))
        st.caption(f"Your business type: {profiles.BUSINESS_TYPES[prof.business_type]['label']}.")
        if accounts.can(user, "manage_users") and ctx.get("auth_on", True):
            c1, c2 = st.columns(2)
            if c1.button(f"Enable {name} anyway", key=f"enable_{name}"):
                accounts.set_module_override(user, name, True)
                st.rerun()
            c2.caption("If your business does this too, add it under Business profile instead; that switches on every related module.")
        elif not ctx.get("auth_on", True):
            st.caption("Change the demo business type in the sidebar to see this module.")


# ------------------------------------------------------------------ trust: how sure are we about this page?
LEVEL_NOTE = {"High": "The data behind this page is thick, recent and complete enough to lean on.",
              "Medium": "Usable, but read it as a guide: the data is thin, old or has gaps.",
              "Low": "Treat this as a rough hint only until more or better data is loaded.",
              "Rule-based": "Calendar rules, not your data."}


def evidence_panel(name: str, data: dict, today=None) -> None:
    ev = evidence.assess(name, data or {}, today)
    if ev is None:
        return
    icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴", "Rule-based": "📅"}.get(ev.level, "")
    with st.expander(f"{icon} How sure are we about this page? {ev.level}"):
        st.write(LEVEL_NOTE.get(ev.level, ""))
        for r in ev.reasons:
            st.markdown(f"- {r}")
        if not ev.table.empty:
            st.dataframe(ev.table, hide_index=True, width="stretch")
            st.caption("Every number on this page is computed from exactly these datasets. Download them from Business profile and recompute anything you doubt.")


def export_panel(user: User) -> None:
    """One zip with everything the business has put into BizIQ. No lock-in."""
    if not accounts.can(user, "export"):
        return
    st.subheader("Take your data with you")
    st.caption("Everything you have put into BizIQ (datasets and your bill book) as plain CSV files you can open in Excel. Nothing is locked in.")
    if st.button("Prepare my export", key="export_prepare"):
        st.session_state["export_zip"] = exporter.build_zip(user.org_name)
        audit.log(user.org_id, user.email, "data.export", user.org_id, "full export")
    if st.session_state.get("export_zip"):
        st.download_button("Download my data (zip)", st.session_state["export_zip"], file_name=f"{(user.org_slug or 'biziq')}-data.zip", mime="application/zip", key="export_dl")


def page_pack(pr, recs, ctx):
    """Consultant pack: pick what goes in, then download a PDF (to read) and an Excel file (to recompute)."""
    user = ctx["user"]
    theme.hero("Consultant pack", "A report you can hand to an accountant, banker or consultant, with how sure we are about each part.", eyebrow=user.org_name)
    if not accounts.can(user, "export"):
        st.info("Your role cannot export reports. Ask the business owner.")
        return
    secs = consultant_pack.sections(pr)
    st.markdown("**What goes in**")
    plain = [x["name"] for x in secs if not x["people"]]
    ppl = [x["name"] for x in secs if x["people"]]
    picked = st.multiselect("Sections", plain, default=plain, key="pack_sections")
    if ppl:
        include_people = st.checkbox("Also include people sections (" + ", ".join(ppl) + ") at group level only", value=False, key="pack_people",
                                     help="Tables that list individual employees or candidates are always left out.")
        if include_people:
            picked += ppl
    hide = st.checkbox("Hide customer, vendor, supplier and staff names (best effort; check before sending)", value=False, key="pack_hide")
    st.caption("The pack is not an audit. It states how sure we are about each section and what we did not check. Once downloaded the file is not encrypted: you are responsible for who you send it to.")
    if not picked:
        st.warning("Choose at least one section.")
        return
    if st.button("Prepare the pack", type="primary", key="pack_go"):
        data = st.session_state.get("data", {}) or {}
        pdf = consultant_pack.build_pdf(pr, recs, picked, data, user.org_name, hide)
        xlsx = consultant_pack.build_xlsx(pr, recs, picked, data, user.org_name, hide)
        st.session_state["pack_files"] = (pdf.read_bytes(), xlsx, pdf.name)
        audit.log(user.org_id, user.email, "report.consultant_pack", user.org_id,
                  f"{len(picked)} sections; people={'yes' if any(x in ppl for x in picked) else 'no'}; names hidden={'yes' if hide else 'no'}")
    if st.session_state.get("pack_files"):
        pdf_b, xl_b, nm = st.session_state["pack_files"]
        c1, c2 = st.columns(2)
        c1.download_button("Download PDF", pdf_b, file_name=nm, mime="application/pdf", key="pack_dl_pdf")
        c2.download_button("Download Excel", xl_b, file_name=nm.replace(".pdf", ".xlsx"), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="pack_dl_xlsx")
