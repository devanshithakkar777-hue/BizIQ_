"""Streamlit pages for BizIQ. app.py handles state and navigation; each page is a function here."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import theme, ingest, accounts, audit, automation, connectors, data_layer, decision, reportiq, reports
from .core import inr
from .pipeline import PipelineResult

ICON = {"red": "🔴", "amber": "🟠", "green": "🟢"}
SEQ = ["#1d4ed8", "#0ea5e9", "#16a34a", "#d97706", "#dc2626", "#9333ea", "#64748b"]


# ------------------------------------------------------------------ helpers
def table(df: pd.DataFrame | None, pct=(), dec=2, height=None, hide=()):
    if df is None or len(df) == 0:
        st.caption("No rows.")
        return
    d = df.drop(columns=[c for c in hide if c in df.columns]).copy()
    cfg = {}
    for c in d.columns:
        if c in pct:
            d[c] = d[c] * 100
            cfg[c] = st.column_config.NumberColumn(format="%.1f%%")
        elif d[c].dtype.kind == "f":
            d[c] = d[c].round(dec)
    kw = {"height": height} if height else {}
    st.dataframe(d, width="stretch", hide_index=True, column_config=cfg, **kw)


def chart(fig: go.Figure, h: int = 340):
    st.plotly_chart(theme.style(fig, h), width="stretch", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def flags_block(res, limit=12):
    theme.feed(res.flags, limit, show_module=False)


def recs_block(name: str, recs: dict):
    r = recs.get(name)
    if not r or not r["items"]:
        return
    with st.container(border=True):
        for item in r["items"]:
            st.markdown(f"- {item}")
        st.caption(f"Source: {r['source']}")


def header(name: str, pr: PipelineResult, recs: dict, blurb: str, body: bool = True):
    theme.hero(name, blurb, eyebrow="Module", level=pr.modules[name].worst_level() if name in pr.modules else None)
    if name in pr.failed:
        st.error(f"{name} could not run: {pr.skipped.get(name, '')}. "
                 "Check that all packages are installed: pip install -r requirements.txt")
        return None
    if name not in pr.modules:
        st.info(f"{name} has no results yet. {pr.skipped.get(name, 'Load the required data on the Data page.')}")
        return None
    res = pr.modules[name]
    if body:
        st.info(reportiq.module_summary(name, res))
        flags_block(res)
        recs_block(name, recs)
    return res


def notes(res):
    for n in res.notes:
        st.caption(f"ℹ️ {n}")


def need(res, key):
    return res is not None and key in res.tables and len(res.tables[key]) > 0


# ------------------------------------------------------------------ executive summary
def page_exec(pr, recs, ctx):
    ex = reportiq.executive_summary(pr)
    score = theme.health_score(pr)
    theme.hero("Command centre", f"Data as at {pr.as_of:%d %B %Y} · {len(pr.modules)} modules · analysed in {pr.total_seconds:.1f}s", eyebrow="Executive summary")
    if ctx.get("synthetic"):
        st.warning("You are viewing SYNTHETIC sample data. Never present it as real client data.")
    theme.kpis([(k, v) for k, v in ex["kpis"]])
    left, right = st.columns([1, 2])
    with left:
        with st.container(border=True):
            st.plotly_chart(theme.gauge(score), width="stretch", config={"displayModeBar": False})
            st.caption("Average of each module: 100 if clear, 65 if its worst finding is amber, 25 if red. A simple rule, not a prediction.")
    with right:
        with st.container(border=True):
            st.markdown("**Module map** · size = findings, colour = worst status")
            st.plotly_chart(theme.module_map(pr), width="stretch", config={"displayModeBar": False})
    st.subheader("Needs attention")
    st.write(ex["headline"])
    all_flags = sorted([f for m in pr.modules.values() for f in m.flags], key=lambda f: ("red", "amber", "green").index(f.level))
    pick = st.segmented_control("Show", ["Act now", "Watch", "Healthy", "All"], default="Act now", key="exec_filter")
    want = {"Act now": "red", "Watch": "amber", "Healthy": "green"}.get(pick)
    shown = [f for f in all_flags if want is None or f.level == want]
    theme.feed(shown, limit=15)
    if not shown:
        st.caption("Nothing in this category.")
    st.subheader("Modules")
    names = list(pr.modules)
    for i in range(0, len(names), 3):
        cols = st.columns(3)
        for c, n in zip(cols, names[i:i + 3]):
            res = pr.modules[n]
            with c.container(border=True):
                st.markdown(f"**{theme.esc(n)}** &nbsp; {theme.pill(res.worst_level())}", unsafe_allow_html=True)
                s_ = reportiq.module_summary(n, res)
                st.caption(s_[:230] + ("…" if len(s_) > 230 else ""))
    if pr.skipped:
        st.caption("Not run: " + "; ".join(f"{k} ({v})" for k, v in pr.skipped.items()))
    st.subheader("Reports")
    c1, c2 = st.columns(2)
    for col, kind, label in ((c1, "weekly", "Weekly report (PDF)"), (c2, "board", "Monthly board report (PDF)")):
        with col:
            if st.button(f"Build {label}", key=f"build_{kind}"):
                p = reports.build_report(pr, kind=kind, recs=recs,
                                         data_note="Figures are based on SYNTHETIC sample data." if ctx.get("synthetic") else "")
                st.session_state[f"pdf_{kind}"] = p.read_bytes()
                st.session_state[f"pdfname_{kind}"] = p.name
            if f"pdf_{kind}" in st.session_state:
                st.download_button(f"Download {st.session_state[f'pdfname_{kind}']}", st.session_state[f"pdf_{kind}"],
                                   file_name=st.session_state[f"pdfname_{kind}"], mime="application/pdf", key=f"dl_{kind}")


# ------------------------------------------------------------------ data
def page_data(pr, recs, ctx):
    st.title("Data layer")
    st.caption("Upload CSV or Excel, clean, validate, store. Strict validation with clear messages.")
    meta = data_layer.dataset_meta()
    rows = []
    for name, sch in data_layer.SCHEMAS.items():
        m = meta[meta["dataset"] == name]
        rows.append(dict(dataset=name, status="loaded" if len(m) else "missing", rows=int(m["rows"].iloc[0]) if len(m) else 0,
                         source=m["source"].iloc[0] if len(m) else "", loaded_at=m["loaded_at"].iloc[0] if len(m) else "",
                         used_by=sch.required_for))
    table(pd.DataFrame(rows))
    user = ctx.get("user")
    can_upload = user is None or accounts.can(user, "upload")
    if not can_upload:
        st.info("Your role can view data but not change it. Ask an admin or analyst to upload files.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load synthetic sample data", type="primary", disabled=not can_upload):
            reps = data_layer.load_sample_data()
            bad = {k: v.errors for k, v in reps.items() if not v.ok}
            if user is not None:
                audit.log(user.org_id, user.email, "data.load_sample", "all", f"{len(reps)} datasets, failures={len(bad)}")
            ctx["refresh"]()
            st.success("Sample data loaded (synthetic, made-up numbers)." if not bad else f"Some datasets failed: {bad}")
            st.rerun()
    with c2:
        st.caption("The bundled sample data is generated by scripts/generate_sample_data.py and is clearly synthetic.")
    st.divider()
    st.subheader("Upload a dataset")
    ds = st.selectbox("Dataset", list(data_layer.SCHEMAS), help="Pick which table this file contains")
    sch = data_layer.SCHEMAS[ds]
    st.caption(sch.description)
    req, opt = list(sch.required), list(sch.optional)
    st.markdown(f"**Required columns:** `{', '.join(req)}`" + (f"  \n**Optional:** `{', '.join(opt)}`" if opt else ""))
    sample = data_layer.SAMPLE_DIR / f"{ds}.csv"
    if sample.exists():
        st.download_button("Download a template (first 5 sample rows)", pd.read_csv(sample).head(5).to_csv(index=False),
                           file_name=f"{ds}_template.csv", mime="text/csv")
    presets = [k for k, v in connectors.PRESETS.items() if v["dataset"] == ds]
    preset = st.selectbox("Column preset for a common export (optional)", ["None"] + presets,
                          help="Renames typical export headings to BizIQ's. Headings vary between accounts, so check the preview.") if presets else "None"
    if preset != "None" and connectors.PRESETS[preset].get("note"):
        st.warning(connectors.PRESETS[preset]["note"])
    up = st.file_uploader("CSV or Excel file", type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls"], key=f"up_{ds}", disabled=not can_upload)
    if up is not None:
        _upload_flow(ds, up, preset, user, can_upload, ctx)
    st.divider()
    _data_health_panel(user, can_upload, ctx)


def safe_csv(df: pd.DataFrame) -> str:
    """CSV text that is safe to open in Excel: cells starting with = + - @ are prefixed so they cannot run as formulas."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r")
                                and not re.fullmatch(r"[-+]?[\d.,]+", v) else v)
    return out.to_csv(index=False)


def _upload_flow(ds, up, preset, user, can_upload, ctx):
    sheet = None
    if up.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        try:
            sheets = ingest.list_sheets(up)
            up.seek(0)
        except Exception as e:
            st.error(f"Could not open the workbook: {e}")
            return
        if len(sheets) > 1:
            sheet = st.selectbox("Sheet", sheets, help="BizIQ picked the sheet that looks most like this dataset; change it if wrong.",
                                 index=0, key=f"sheet_{ds}")
    try:
        raw = data_layer.read_upload(up, up.name, dataset=ds, sheet=sheet)
        if preset != "None":
            pm = {k.strip().lower(): v for k, v in connectors.PRESETS[preset]["map"].items()}
            raw = raw.rename(columns={c: pm[str(c).strip().lower()] for c in raw.columns if str(c).strip().lower() in pm})
            for col, val in connectors.PRESETS[preset].get("defaults", {}).items():
                if col not in raw.columns:
                    raw[col] = val
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    # Step 1: matching columns. The user can override any suggestion.
    manual = st.session_state.get(f"map_{ds}_{up.name}", {})
    cleaned, rep = data_layer.clean_and_validate(raw, ds, column_map=manual or None)
    if rep.suggestions or not rep.ok:
        with st.expander("Match columns", expanded=bool(rep.suggestions)):
            st.caption("Pick which of your columns holds each BizIQ field. Leave as '(auto)' to keep BizIQ's choice.")
            missing = [c for c in data_layer.SCHEMAS[ds].required if c not in cleaned.columns]
            chosen = dict(manual)
            for tgt in missing:
                guess = next((m["source"] for m in rep.suggestions if m["target"] == tgt), None)
                opts = ["(auto)"] + [c for c in rep.available_columns]
                pick = st.selectbox(f"{tgt}", opts, index=(opts.index(guess) if guess in opts else 0), key=f"pick_{ds}_{tgt}_{up.name}")
                if pick != "(auto)":
                    chosen = {k: v for k, v in chosen.items() if v != tgt}
                    chosen[pick] = tgt
            if chosen != manual and st.button("Apply column choices", key=f"apply_{ds}"):
                st.session_state[f"map_{ds}_{up.name}"] = chosen
                st.rerun()
    st.markdown(f"**Rows read:** {rep.rows_in:,} · **kept:** {rep.rows_out:,} · **set aside:** {rep.rows_rejected:,}")
    for e in rep.errors:
        st.error(e)
    for w in rep.warnings:
        st.warning(w)
    for c in rep.cleaning:
        st.info(c)
    if rep.rows_rejected:
        with st.expander(f"{rep.rows_rejected:,} rows were set aside. See why"):
            table(rep.rejects.head(200))
            st.download_button("Download all set-aside rows (CSV)", safe_csv(rep.rejects), file_name=f"{ds}_rejected_rows.csv", mime="text/csv")
    if not rep.ok:
        return
    table(cleaned.head(10))

    # Step 2: how to store it
    mode_label = "Replace everything" 
    if data_layer.dataset_meta().pipe(lambda m: (m["dataset"] == ds).any()):
        mode_label = st.radio("This dataset already has data. What should this file do?",
                              ["Add new rows and update existing ones", "Replace everything"], key=f"mode_{ds}",
                              help="Adding is safe for monthly exports: rows already stored are not duplicated, and records that changed (e.g. an invoice now paid) are updated.")
    mode = "merge" if mode_label.startswith("Add") else "replace"
    if st.button(f"Save {ds} to BizIQ", type="primary", disabled=not can_upload):
        result = data_layer.save_dataset(cleaned, ds, source=f"upload: {up.name}", mode=mode)
        if user is not None:
            audit.log(user.org_id, user.email, "data.upload", ds,
                      f"file {up.name}; mode {mode}; read {rep.rows_in}, kept {rep.rows_out}, set aside {rep.rows_rejected}; {result.summary()}")
        ctx["refresh"]()
        msg = result.summary()
        if ds == "financials":
            info = automation.on_new_financials_job(data_layer.load_dataset("financials"))
            msg += f" New-month summary written ({info['file']})."
        st.session_state["flash"] = msg + " All modules refreshed."
        st.rerun()


def _data_health_panel(user, can_upload, ctx):
    st.subheader("Data health")
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)
    h = data_layer.data_health()
    table(h)
    undo = [n for n in data_layer.SCHEMAS if data_layer.has_previous(n)]
    if undo and can_upload:
        pick = st.selectbox("Undo the last load of", undo, key="undo_pick",
                            help="Brings back the version before the most recent load. Pressing it again re-applies the load.")
        if st.button("Undo last load"):
            data_layer.restore_previous(pick)
            if user is not None:
                audit.log(user.org_id, user.email, "data.undo", pick, "restored previous version")
            ctx["refresh"]()
            st.session_state["flash"] = f"Restored the earlier version of {pick}."
            st.rerun()


# ------------------------------------------------------------------ core modules
def page_wealth(pr, recs, ctx):
    res = header("WealthIQ", pr, recs, "Financial health: ratios, Altman Z-score, 0-100 health score, trends.")
    if res is None:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Health score", f"{res.metrics['health_score']}/100")
    c2.metric("Altman Z'", res.metrics["altman_z"], res.metrics["altman_zone"], delta_color="off")
    c3.metric("Revenue (12m)", inr(res.metrics["revenue_ttm"]))
    c4.metric("Net income (12m)", inr(res.metrics["net_income_ttm"]))
    t1, t2, t3 = st.tabs(["Score and trend", "Ratios", "Trend table"])
    with t1:
        g = go.Figure(go.Indicator(mode="gauge+number", value=res.metrics["health_score"],
                                   gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#1d4ed8"),
                                              steps=[dict(range=[0, 50], color="#fee2e2"), dict(range=[50, 75], color="#fef3c7"),
                                                     dict(range=[75, 100], color="#dcfce7")])))
        chart(g, 250)
        tr = res.tables["trend"]
        chart(px.line(tr, x="month", y="health_score", title="Health score over time", markers=True))
        chart(px.line(tr, x="month", y=["gross_margin", "operating_margin", "net_margin"], title="Margins"))
    with t2:
        r = res.tables["ratios"].copy()
        r["status"] = r["status"].map(lambda s: ICON.get(s, ""))
        table(r, dec=0)
    with t3:
        table(res.tables["trend"].tail(18))
    notes(res)


def page_supply(pr, recs, ctx):
    res = header("SupplyChainIQ", pr, recs, "Inventory turnover, stock-outs, overstock, ABC, demand forecast, supplier scoring, reorder points.")
    if res is None:
        return
    m = res.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inventory value", inr(m["inventory_value"]))
    c2.metric("Critical / reorder", f"{m['stockout_risk']} / {m['reorder_now']}")
    c3.metric("Overstocked", m["overstocked"])
    c4.metric("Slow-moving value", inr(m["slow_moving_value"]))
    t1, t2, t3, t4, t5 = st.tabs(["Reorder list", "Stock status", "ABC", "Demand forecast", "Suppliers"])
    inv = res.tables["inventory"]
    with t1:
        table(res.tables["reorder"], height=220)
        st.caption("Suggested quantity covers lead time plus 14 days plus safety stock. Purchase orders are drafted in ProcureIQ.")
    with t2:
        s = inv[["product", "abc_class", "stock_on_hand", "avg_daily_demand", "days_of_cover", "lead_time_days", "reorder_point",
                 "safety_stock", "turnover", "stock_value", "status"]]
        table(s)
        fig = px.bar(inv, x="product", y="days_of_cover", color="status", title="Days of cover by product")
        chart(fig)
    with t3:
        a = res.tables["abc"]
        fig = px.bar(a, x="product", y="revenue", color="abc_class", title="Revenue by product (ABC class)")
        chart(fig)
        table(res.tables["abc_summary"], pct=("share",))
    with t4:
        prod = st.selectbox("Product", sorted(inv["product"]), key="fc_prod")
        h = res.tables["history_units"][["date", prod]].rename(columns={prod: "units"})
        h["kind"] = "history"
        f = res.tables["forecast"]
        f = f[f["product"] == prod]
        fig = go.Figure()
        fig.add_scatter(x=h["date"], y=h["units"], name="History", mode="lines+markers")
        fig.add_scatter(x=f["month"], y=f["forecast_units"], name="Forecast", mode="lines+markers", line=dict(dash="dash"))
        fig.add_scatter(x=list(f["month"]) + list(f["month"][::-1]), y=list(f["upper"]) + list(f["lower"][::-1]),
                        fill="toself", name="80% range", line=dict(width=0), opacity=0.2)
        fig.update_layout(title=f"{prod}: monthly units")
        chart(fig)
    with t5:
        if "suppliers" in res.tables:
            sp = res.tables["suppliers"]
            table(sp, pct=("on_time_rate", "fill_rate", "defect_rate"))
            chart(px.bar(sp, x="supplier", y="score", color="grade", title="Supplier performance score (0-100)"))
        else:
            st.info("Load supplier_deliveries to score suppliers.")
    notes(res)


def page_sales(pr, recs, ctx):
    res = header("SalesIQ", pr, recs, "Trends and seasonality, top and bottom analysis, segment performance, basket analysis, campaigns.")
    if res is None:
        return
    m = res.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", inr(m["total_revenue"]))
    c2.metric("Orders", f"{m['orders']:,}")
    c3.metric("Avg order value", inr(m["aov"], False))
    c4.metric("Last 28d vs baseline", "n/a" if pd.isna(m["recent_vs_baseline"]) else f"{m['recent_vs_baseline']:+.0%}")
    t1, t2, t3, t4, t5 = st.tabs(["Trend", "Products and regions", "Basket analysis", "Campaigns", "Weekly digest"])
    with t1:
        mt = res.tables["monthly"]
        chart(px.bar(mt, x="date", y="revenue", title="Monthly revenue"))
        chart(px.line(mt, x="date", y="seasonality_index", markers=True, title="Seasonality index (1.0 = average month)"))
        chart(px.bar(res.tables["day_of_week"], x="day", y="index", title="Day-of-week index"), 260)
    with t2:
        a, b = st.columns(2)
        with a:
            st.markdown("**Top 5 products**")
            table(res.tables["top_products"], pct=("share",))
        with b:
            st.markdown("**Bottom 5 products**")
            table(res.tables["bottom_products"], pct=("share",))
        chart(px.bar(res.tables["regions"], x="region", y="revenue", title="Revenue by region"), 280)
        chart(px.pie(res.tables["categories"], names="category", values="revenue", title="Category (segment) mix"), 300)
    with t3:
        table(res.tables["basket_pairs"], pct=("support", "confidence_a_to_b", "confidence_b_to_a"))
        st.caption("Lift above 1 means the pair is bought together more often than chance.")
    with t4:
        if need(res, "campaigns"):
            table(res.tables["campaigns"], pct=("assumed_response_rate",))
            st.warning("Projected uplift is an ESTIMATE: customers × assumed response rate × average order value.")
        else:
            st.info("Campaign sizing needs ClientIQ segments.")
    with t5:
        from . import salesiq
        st.write(salesiq.weekly_digest(res))
    notes(res)


def page_client(pr, recs, ctx):
    res = header("ClientIQ", pr, recs, "RFM segmentation, churn model, customer lifetime value, next-best-action, drafted follow-ups.")
    if res is None:
        return
    m = res.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{m['customers']:,}")
    c2.metric("At risk", f"{m['at_risk_customers']:,}")
    c3.metric("Annual revenue at stake", inr(m["annual_revenue_at_stake"]))
    c4.metric("Retention", f"{m['retention_rate']:.0%}")
    t1, t2, t3, t4 = st.tabs(["Segments", "At-risk clients", "Drafted follow-ups", "Customer lookup"])
    with t1:
        sg = res.tables["segments"]
        chart(px.bar(sg, x="segment", y="customers", color="segment", title="Customers by segment"), 300)
        chart(px.bar(sg, x="segment", y="revenue", color="segment", title="Revenue by segment"), 300)
        table(sg, pct=("share_of_revenue", "avg_churn_prob"))
    with t2:
        table(res.tables["at_risk"], pct=("churn_prob",), height=420)
    with t3:
        st.warning("Drafts only. Review, personalise and send them yourself; BizIQ never sends these automatically.")
        for _, r in res.tables["drafted_emails"].head(8).iterrows():
            with st.expander(f"{r['customer_id']} · {r['segment']} · churn {r['churn_prob']:.0%} · {inr(r['annual_revenue_at_stake'])} at stake"):
                st.text(r["draft"])
        if decision.llm_available():
            st.caption("An LLM can polish drafts (numbers verified) when enabled in the sidebar.")
        st.download_button("Download all drafts (CSV)", res.tables["drafted_emails"].to_csv(index=False), "followup_drafts.csv")
    with t4:
        cid = st.text_input("Customer ID", value=str(res.tables["customers"]["customer_id"].iloc[0]))
        row = res.tables["customers"][res.tables["customers"]["customer_id"] == cid]
        if len(row):
            r = row.iloc[0]
            st.write(f"**{cid}** · segment **{r['segment']}** · churn probability **{r['churn_prob']:.0%}** · "
                     f"CLV **{inr(r['clv'])}** · last order {int(r['recency_days'])} days ago")
            st.info(r["next_best_action"])
        else:
            st.caption("Not found.")
    notes(res)


# ------------------------------------------------------------------ extended modules
def page_cash(pr, recs, ctx):
    res = header("CashFlowIQ", pr, recs, "Cash forecast, shortfall warnings, receivables and payables ageing.")
    if res is None:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash", inr(res.metrics["cash"]))
    c2.metric("Minimum buffer", inr(res.metrics["buffer"]))
    c3.metric("Downside low point", inr(res.metrics["downside_min_cash"]))
    c4.metric("Overdue receivables", inr(res.metrics.get("receivables_overdue", 0)))
    t1, t2, t3 = st.tabs(["Forecast", "Ageing", "8-week view"])
    with t1:
        f = res.tables["forecast"]
        fig = go.Figure()
        h = res.tables["history"].tail(12)
        fig.add_scatter(x=h["month"], y=h["cash"], name="Actual", mode="lines")
        for s in ("Base", "Downside", "Upside"):
            fig.add_scatter(x=f["month"], y=f[s], name=s, mode="lines+markers", line=dict(dash="dash" if s != "Base" else "solid"))
        fig.add_hline(y=res.metrics["buffer"], line_dash="dot", annotation_text="Minimum buffer")
        fig.update_layout(title="Cash balance: history and 6-month forecast")
        chart(fig)
    with t2:
        if need(res, "receivables_ageing"):
            a, b = st.columns(2)
            a.plotly_chart(px.bar(res.tables["receivables_ageing"], x="bucket", y="amount", title="Receivables ageing"), width="stretch")
            b.plotly_chart(px.bar(res.tables["payables_ageing"], x="bucket", y="amount", title="Payables ageing"), width="stretch")
            st.markdown("**Receivables by customer**")
            table(res.tables["receivables_by_party"], pct=("late_share",))
            with st.expander("Open invoices"):
                table(res.tables["open_receivables"][["invoice_id", "party", "invoice_date", "due_date", "amount", "days_overdue", "bucket"]])
        else:
            st.info("Load invoices for ageing analysis.")
    with t3:
        if need(res, "weekly_view"):
            w = res.tables["weekly_view"]
            fig = go.Figure()
            fig.add_bar(x=w["week_start"], y=w["expected_collections"], name="Expected collections")
            fig.add_bar(x=w["week_start"], y=-w["scheduled_payments"], name="Scheduled payments")
            fig.update_layout(barmode="relative", title="Weekly cash movements (next 8 weeks)")
            chart(fig)
            table(w)
    notes(res)


def page_tax(pr, recs, ctx):
    res = header("TaxIQ", pr, recs, "GST and income tax deadlines, liability estimates, compliance gaps, reminders.")
    st.warning("TaxIQ shows ESTIMATES on stated assumptions, not tax advice. Verify rates and due dates with your CA.")
    with st.expander("Tax assumptions"):
        cfg = ctx["tax_cfg"]
        a, b, c = st.columns(3)
        gst = a.number_input("GST rate %", 0.0, 28.0, float(cfg.gst_rate * 100), 0.5)
        it = b.number_input("Income tax rate %", 0.0, 45.0, float(cfg.income_tax_rate * 100), 0.01)
        itc = c.number_input("ITC eligible share %", 0.0, 100.0, float(cfg.itc_eligible_share * 100), 1.0)
        if st.button("Apply assumptions"):
            cfg.gst_rate, cfg.income_tax_rate, cfg.itc_eligible_share = gst / 100, it / 100, itc / 100
            ctx["refresh"]()
            st.rerun()
    if res is None:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Projected PBT (FY)", inr(res.metrics["pbt_fy_estimate"]))
    c2.metric("Estimated income tax", inr(res.metrics["income_tax_estimate"]))
    c3.metric("GST payable (last month, est.)", inr(res.metrics["gst_payable_last_month"]))
    t1, t2, t3 = st.tabs(["Deadlines", "Advance tax", "GST"])
    with t1:
        table(res.tables["deadlines"].assign(due_date=lambda d: d["due_date"].dt.strftime("%d %b %Y")).astype({"urgency": str}))
    with t2:
        table(res.tables["advance_tax"], pct=("cumulative_share",))
    with t3:
        g = res.tables["gst"]
        chart(px.line(g, x="month", y=["output_gst", "itc", "net_gst_payable"], title="Estimated GST (monthly)"))
        table(g, pct=("itc_to_output",))
    notes(res)


def page_risk(pr, recs, ctx):
    res = header("RiskIQ", pr, recs, "Credit risk, anomaly and possible-fraud detection, concentration risk.")
    if res is None:
        return
    t1, t2, t3, t4 = st.tabs(["Anomalies", "Customer credit risk", "Supplier risk", "Concentration"])
    with t1:
        table(res.tables["anomalies"], height=300)
        st.caption("Flagged items are prompts to investigate, not proof of fraud.")
    with t2:
        if need(res, "customer_credit_risk"):
            cr = res.tables["customer_credit_risk"].copy()
            cr["rating"] = cr["rating"].astype(str)
            table(cr, pct=("late_share", "overdue_share"))
            chart(px.bar(cr, x="party", y="risk_score", color="rating", title="Credit risk score (0-100, higher = riskier)"), 300)
    with t3:
        if need(res, "supplier_risk"):
            sr = res.tables["supplier_risk"].copy()
            sr["rating"] = sr["rating"].astype(str)
            table(sr, pct=("late_rate", "defect_rate", "fill_shortfall", "spend_share"))
            st.markdown("**Single-source products (70%+ of spend with one supplier)**")
            table(res.tables["single_source"], pct=("share",))
    with t4:
        table(res.tables["concentration"], dec=3)
    notes(res)


def page_invest(pr, recs, ctx):
    res = header("InvestIQ", pr, recs, "Surplus funds, illustrative allocation by risk appetite, portfolio tracking.")
    st.warning("Illustrative planning only: not investment advice or a forecast. Confirm with a qualified adviser.")
    if res is None:
        return
    st.caption(f"Risk appetite: **{res.metrics['risk_appetite']}** (change it in the sidebar)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash", inr(res.metrics["cash"]))
    c2.metric("Required buffer", inr(res.metrics["required_buffer"]))
    c3.metric("Surplus", inr(res.metrics["surplus"]))
    c4.metric("Cost of debt", f"{res.metrics['cost_of_debt']:.1%}")
    t1, t2 = st.tabs(["Allocation plan", "Portfolio tracking"])
    with t1:
        plan = res.tables["allocation_plan"]
        a, b = st.columns(2)
        a.plotly_chart(px.pie(plan[plan["weight_pct"] > 0], names="asset_class", values="amount", title="Surplus allocation"), width="stretch")
        with b:
            table(plan)
        st.caption("Return ranges are assumptions for planning, not predictions.")
    with t2:
        if need(res, "portfolio"):
            table(res.tables["portfolio"], pct=("gain_pct", "weight"))
            d = res.tables["allocation_vs_target"]
            fig = go.Figure()
            fig.add_bar(x=d["asset_class"], y=d["actual_weight_pct"], name="Actual %")
            fig.add_bar(x=d["asset_class"], y=d["target_weight_pct"], name="Target %")
            fig.update_layout(barmode="group", title="Actual vs target allocation")
            chart(fig, 300)
        else:
            st.info("Load a portfolio file to track holdings.")
    notes(res)


def page_people(pr, recs, ctx):
    res = header("PeopleIQ", pr, recs, "Payroll cost, attrition risk, hiring needs, productivity.")
    if res is None:
        return
    m = res.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headcount", m["headcount"])
    c2.metric("Monthly payroll", inr(m["monthly_payroll"]))
    c3.metric("Attrition (12m)", f"{m['attrition_rate']:.0%}")
    c4.metric("Revenue / employee", inr(m["revenue_per_employee"]))
    st.info("Signals, not judgements. This cannot know about illness, family trouble or workload. Figures are by department only; teams under 5 people are not shown, and nobody is scored by name.")
    t1, t3 = st.tabs(["Departments", "Hiring needs"])
    with t1:
        d = res.tables["departments"]
        table(d, pct=("attrition_pct", "avg_risk", "payroll_share"))
        chart(px.bar(d.dropna(subset=["avg_overtime"]), x="department", y="avg_overtime", title="Average overtime hours per month"), 280)
    with t3:
        table(res.tables["hiring_needs"])
    notes(res)


def page_brand(pr, recs, ctx):
    res = header("BrandIQ", pr, recs, "Social media performance, customer sentiment, content themes.")
    if res is None:
        return
    t1, t2, t3, t4 = st.tabs(["Social", "Reviews and sentiment", "Products", "Content ideas"])
    with t1:
        if need(res, "by_platform"):
            chart(px.bar(res.tables["by_platform"], x="platform", y="engagement_rate", title="Engagement rate by platform"), 280)
            chart(px.bar(res.tables["by_post_type"], x="post_type", y="engagement_rate", title="Engagement rate by post type"), 280)
            if need(res, "by_theme"):
                chart(px.bar(res.tables["by_theme"], x="theme", y="engagement_rate", title="Engagement rate by theme"), 280)
        else:
            st.info("Load social data.")
    with t2:
        if need(res, "sentiment_distribution"):
            a, b = st.columns(2)
            a.plotly_chart(px.pie(res.tables["sentiment_distribution"], names="sentiment", values="reviews",
                                  title="Review sentiment", color="sentiment",
                                  color_discrete_map={"Positive": "#16a34a", "Neutral": "#94a3b8", "Negative": "#dc2626"}), width="stretch")
            b.plotly_chart(px.bar(res.tables["issue_themes"], x="issue", y="mentions", title="What unhappy customers mention"), width="stretch")
        else:
            st.info("Load reviews.")
    with t3:
        if need(res, "by_product"):
            table(res.tables["by_product"], pct=("negative_share",))
    with t4:
        table(res.tables["content_ideas"])
    notes(res)


def page_procure(pr, recs, ctx):
    res = header("ProcureIQ", pr, recs, "Compare suppliers on price, quality and delivery; draft purchase orders.")
    if res is None:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Annual spend", inr(res.metrics["annual_spend"]))
    c2.metric("Potential saving (est.)", inr(res.metrics["potential_saving"]))
    c3.metric("Draft POs", f"{res.metrics['draft_pos']} · {inr(res.metrics['po_value'])}")
    t1, t2, t3 = st.tabs(["Recommended suppliers", "Draft purchase orders", "Full comparison"])
    with t1:
        table(res.tables["recommended_suppliers"], pct=("on_time_rate", "defect_rate"))
        table(res.tables["savings_opportunities"])
    with t2:
        st.warning("Drafts for human approval. BizIQ never issues or sends purchase orders.")
        docs = res.tables["purchase_order_documents"]
        for _, r in docs.iterrows():
            with st.expander(r["supplier"], expanded=True):
                st.code(r["document"], language=None)
                st.download_button("Download draft", r["document"], file_name=f"PO_DRAFT_{r['supplier'].replace(' ', '_')}.txt", key=f"po_{r['supplier']}")
        if docs.empty:
            st.info("No reorders needed right now.")
    with t3:
        table(res.tables["comparison"], pct=("on_time_rate", "defect_rate", "price_vs_best_pct"))
    notes(res)


# ------------------------------------------------------------------ ReportIQ
def page_report(pr, recs, ctx):
    st.title("ReportIQ")
    st.caption("Board-ready reports and plain-English answers about your data. Answers come from computed figures only.")
    t1, t2 = st.tabs(["Ask a question", "Management summary"])
    with t1:
        st.session_state.setdefault("chat", [])
        for who, msg, tbl in st.session_state["chat"]:
            with st.chat_message(who):
                st.write(msg)
                if tbl is not None:
                    table(tbl)
        q = st.chat_input("e.g. How is cash? Which products need reordering? Which customers are at risk?")
        if q:
            a = reportiq.ask(q, pr, use_llm=ctx["use_llm"])
            st.session_state["chat"] += [("user", q, None), ("assistant", a["answer"] + f"\n\n_Source: {a['source']}_", a["table"])]
            st.rerun()
        st.caption("Try: cash · profit margin · churn · stock · tax deadlines · anomalies · surplus funds · attrition · reviews · suppliers")
    with t2:
        st.markdown(reportiq.board_narrative(pr).replace("\n\n", "\n\n---\n\n"))


# ------------------------------------------------------------------ Automation
def page_automation(pr, recs, ctx):
    st.title("Automation")
    st.caption("Rules and triggers that act on the insights: alerts, scheduled reports, drafted emails, reorder suggestions.")
    cfg = automation.EmailConfig.from_env()
    if cfg.can_send():
        st.error(f"LIVE email is enabled: messages will be sent from {cfg.sender} to {', '.join(cfg.recipients)}.")
    else:
        st.success("Email is in DRY-RUN mode: messages are saved to output/outbox/ and nothing is sent. "
                   "Set BIZIQ_SEND_EMAIL=1 plus SMTP settings to go live.")
    t1, t2, t3, t4, t5 = st.tabs(["Run jobs", "Alerts", "Outbox and drafts", "Run log", "Measure automation"])
    with t1:
        cols = st.columns(3)
        jobs = [("Weekly report + digest", lambda: automation.weekly_report_job(use_llm=ctx["use_llm"])),
                ("Check for new alerts", automation.alert_job),
                ("Draft client follow-ups", automation.followup_drafts_job),
                ("Draft purchase orders", automation.purchase_order_job),
                ("Monthly financial summary", automation.on_new_financials_job)]
        for i, (label, fn) in enumerate(jobs):
            with cols[i % 3]:
                if st.button(label, key=f"job_{i}"):
                    with st.spinner("Running..."):
                        try:
                            st.json(fn())
                        except Exception as e:
                            st.error(f"{type(e).__name__}: {e}")
        st.markdown("#### Schedule it")
        st.code("# Python scheduler (weekly report Mondays 08:00, alert check daily 09:00)\n"
                "python scripts/run_scheduler.py\n\n"
                "# Or cron\n"
                "0 8 * * 1  cd /path/to/biziq && python -c \"from biziq import automation; automation.weekly_report_job()\"\n"
                "0 9 * * *  cd /path/to/biziq && python -c \"from biziq import automation; automation.alert_job()\"", language="bash")
        st.caption("n8n or Zapier can also call these jobs through a webhook or a scheduled command.")
    with t2:
        alerts = automation.build_alerts(pr)
        st.markdown(f"**{len(alerts)} active alert(s) from the current data**")
        for a in alerts[:25]:
            st.markdown(f"{ICON[a.level]} **{a.module}** — {a.title}" + (f"  \n<span style='color:#6b7280'>{a.detail} {a.action}</span>" if a.detail or a.action else ""),
                        unsafe_allow_html=True)
        with st.expander("Alert history (de-duplicated, stored in the database)"):
            table(automation.alert_history())
    with t3:
        ob = sorted(automation.OUTBOX.glob("*.eml"), reverse=True) if automation.OUTBOX.exists() else []
        st.markdown(f"**Outbox (dry-run emails): {len(ob)}**")
        if ob:
            sel = st.selectbox("Message", [p.name for p in ob[:20]])
            from email import message_from_bytes, policy
            msg = message_from_bytes((automation.OUTBOX / sel).read_bytes(), policy=policy.default)
            st.write(f"**To:** {msg['To']} · **Subject:** {msg['Subject']}")
            body = msg.get_body(preferencelist=("plain",))
            st.text(body.get_content() if body else "")
        dr = sorted(automation.DRAFTS.glob("*"), reverse=True) if automation.DRAFTS.exists() else []
        st.markdown(f"**Drafts awaiting human review: {len(dr)}**")
        for p in dr[:10]:
            st.caption(p.name)
    with t4:
        table(automation.run_log())
    with t5:
        st.markdown("Only **measured** figures should go on a CV. List recurring tasks, time each one done manually and with "
                    "BizIQ, then report either the share of workflows automated or the reduction in time.")
        tasks = automation.load_tasks()
        edited = st.data_editor(tasks, width="stretch", hide_index=True, num_rows="dynamic", key="tasks_editor",
                                column_config={"automation_level": st.column_config.SelectboxColumn(
                                    options=["Fully automated", "Assisted (human review)", "Manual"]),
                                    "manual_minutes": st.column_config.NumberColumn("manual_minutes", min_value=0.0),
                                    "biziq_minutes": st.column_config.NumberColumn("biziq_minutes", min_value=0.0)})
        if st.button("Save measurements"):
            automation.save_tasks(edited)
            st.success("Saved.")
        s = automation.measurement_summary(edited)
        a, b, c = st.columns(3)
        a.metric("Workflows fully automated", f"{s['fully_automated']} of {s['tasks']}")
        b.metric("Fully or partly automated", f"{s['assisted'] + s['fully_automated']} of {s['tasks']}")
        c.metric("Time reduction (measured)", "not measured yet" if s["time_reduction"] is None else f"{s['time_reduction']:.0%}")
        if s["time_reduction"] is not None:
            st.success(f"Measured over {s['measured_tasks']} task(s): {s['manual_minutes']:.0f} min manually vs "
                       f"{s['biziq_minutes']:.0f} min with BizIQ.")
        else:
            st.info("Enter manual and BizIQ minutes for each task to compute a time reduction. "
                    f"For reference, the full BizIQ analysis of all modules ran in {pr.total_seconds:.1f} seconds on this dataset.")
        st.caption("Keep this table (and timings) so the working can be shown in an interview.")
