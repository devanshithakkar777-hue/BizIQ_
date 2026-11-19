"""The billing screens: New bill, Day book, Customers and dues, Items and stock. Built for use at a counter: big inputs, one action per step, English or Hindi."""
from __future__ import annotations

import secrets

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import accounts, billbook as bb, data_layer, i18n, theme
from .ui import chart

PAGES_BILLING = ["New bill", "Day book", "Customers and dues", "Items and stock"]
MODE_KEYS = ["cash", "upi", "card", "bank", "credit"]


def lang() -> str:
    return st.session_state.get("lang", "en")


def L(key: str) -> str:
    return i18n.t(key, lang())


def _toggle() -> None:
    st.session_state.setdefault("lang_choice", "हिन्दी" if lang() == "hi" else "English")      # the widget is the single source of truth
    pick = st.segmented_control(i18n.t("language", lang()), ["English", "हिन्दी"], key="lang_choice", label_visibility="collapsed")
    new = "hi" if pick == "हिन्दी" else "en"
    if new != lang():
        st.session_state["lang"] = new
        st.rerun()


def _can_write(user) -> bool:
    return accounts.can(user, "upload")


def _dirty() -> None:
    st.session_state["bb_dirty"] = True


def _sync_notice() -> None:
    if st.session_state.get("bb_dirty"):
        c1, c2 = st.columns([3, 1])
        c1.caption("New entries are not in the analysis pages yet. They update when you open another page, or now:")
        if c2.button(L("update_analysis"), key="bb_sync_now"):
            sync_now()
            st.rerun()


def sync_now() -> dict:
    status = bb.sync_to_datasets()
    st.session_state["bb_dirty"] = False
    st.session_state.pop("pr", None)
    return status


def _money(x) -> str:
    return bb.fmt(x)


# ------------------------------------------------------------------ New bill
def page_new_bill(pr, recs, ctx):
    user, shop = ctx["user"], ctx["user"].org_name
    _toggle()
    theme.hero(L("new_bill"), L("new_bill_sub"), eyebrow=shop)
    if not _can_write(user):
        st.info(L("read_only"))
        return
    ss = st.session_state
    ss.setdefault("bb_cart", [])
    ss.setdefault("bb_token", secrets.token_hex(8))
    last = ss.get("bb_last")
    if last:
        res, lg = last
        st.success(f"{L('saved')}: {res.bill_no}  ·  {L('total')} {_money(res.total)}" + (f"  ·  {_money(res.due)} {L('goes_on_credit')}" if res.due > 0.005 else ""))
        for w in res.warnings:
            st.warning(w)
        text = bb.receipt_text(res, shop, lg)
        st.code(text, language=None)
        phone = bb.customer_phone(res.customer) if res.customer else ""
        st.link_button(L("send_whatsapp"), bb.whatsapp_link(phone, text))
        if st.button(L("next_bill"), type="primary", key="bb_next"):
            ss["bb_last"] = None
            st.rerun()
        _sync_notice()
        return
    items = bb.list_items()
    names = list(items["name"])
    known = bb.customer_names()
    customer = st.selectbox(L("customer"), known, index=None, accept_new_options=True, placeholder="", key="bb_customer") or ""
    phone = ""
    if customer and customer.strip().lower() not in {k.lower() for k in known}:
        phone = st.text_input(L("phone"), key="bb_phone", max_chars=20)
    st.markdown("---")
    c1, c2, c3 = st.columns([3, 1, 1])
    item = c1.selectbox(L("item"), names, index=None, accept_new_options=True, placeholder="", key="bb_item") or ""
    row = items[items["name"].str.lower() == item.strip().lower()] if item else items.iloc[0:0]
    default_rate = float(row["sale_price"].iloc[0]) if len(row) else 0.0
    qty = c2.number_input(L("qty"), min_value=0.0, value=1.0, step=1.0, key=f"bb_qty_{item}")
    rate = c3.number_input(L("rate"), min_value=0.0, value=default_rate, step=1.0, key=f"bb_rate_{item}")
    if st.button(L("add_to_bill"), key="bb_add", disabled=not item.strip()):
        if qty <= 0:
            st.error(L("qty") + " > 0")
        else:
            ss["bb_cart"].append(dict(item=item.strip(), qty=float(qty), rate=float(rate)))
            for k in ("bb_item",):
                ss.pop(k, None)
            st.rerun()
    cart = ss["bb_cart"]
    if not cart:
        st.caption(L("empty_cart"))
        _sync_notice()
        return
    df = pd.DataFrame(cart)
    df[L("amount")] = (df["qty"] * df["rate"]).round(2)
    st.markdown(f"**{L('on_this_bill')}**")
    st.dataframe(df.rename(columns={"item": L("item"), "qty": L("qty"), "rate": L("rate")}), hide_index=True, width="stretch")
    b1, b2 = st.columns(2)
    if b1.button(L("remove_last"), key="bb_rm"):
        cart.pop()
        st.rerun()
    if b2.button(L("clear"), key="bb_clear"):
        ss["bb_cart"] = []
        st.rerun()
    subtotal = round(float(sum(c["qty"] * c["rate"] for c in cart)), 2)
    disc = st.number_input(L("discount"), min_value=0.0, max_value=max(subtotal, 0.0), value=0.0, step=1.0, key=f"bb_disc_{subtotal}")
    total = round(subtotal - disc, 2)
    st.markdown(f"### {L('total')}: {_money(total)}")
    mode = st.segmented_control(L("how_paid"), MODE_KEYS, default="cash", format_func=lambda k: L(k), key="bb_mode") or "cash"
    paid = 0.0 if mode == "credit" else st.number_input(L("paid_now"), min_value=0.0, max_value=max(total, 0.0), value=total, step=1.0, key=f"bb_paid_{total}_{mode}")
    due = round(total - paid, 2)
    if due > 0.005:
        st.info(f"{_money(due)} {L('goes_on_credit')}" + ("" if customer else "  ·  " + L("customer")))
    note = st.text_input(L("note"), key="bb_note", max_chars=200)
    if st.button(L("save_bill"), type="primary", key="bb_save"):
        try:
            res = bb.create_bill(cart, customer=customer, paid=paid, mode=mode, discount=disc, note=note, by=user.email, phone=phone, token=ss["bb_token"])
        except bb.BillError as e:
            st.error(str(e))
        else:
            ss["bb_last"] = (res, lang())
            ss["bb_cart"] = []
            ss["bb_token"] = secrets.token_hex(8)
            for k in ("bb_customer", "bb_phone", "bb_note", "bb_item"):
                ss.pop(k, None)
            _dirty()
            st.rerun()
    _sync_notice()


# ------------------------------------------------------------------ Day book
def page_day_book(pr, recs, ctx):
    user, shop = ctx["user"], ctx["user"].org_name
    _toggle()
    theme.hero(L("day_book"), L("day_book_sub"), eyebrow=shop)
    day = st.date_input(L("date"), value=pd.Timestamp.today().date(), max_value=pd.Timestamp.today().date(), key="bb_day")
    s = bb.day_summary(day)
    theme.kpis([(L("sales"), _money(s["sales"]), f"{s['bills']} {L('bills_today')}"), (L("collected"), _money(s["collected_total"]),
                f"{_money(s['dues_collected'])} " + ("from old dues" if lang() == "en" else "पुराने बकाया से")),
                (L("on_credit"), _money(s["on_credit"])), (L("expenses"), _money(s["expenses"])), (L("gross_profit"), _money(s["gross_profit"]),
                f"{s['margin_pct']}%" if s["margin_pct"] is not None else ""), (L("net_profit"), _money(s["net_profit"]))])
    if s["sales_without_cost"] > 0:
        st.warning(f"{_money(s['sales_without_cost'])} of today's sales are on items with no cost price, so their profit is not counted. Add cost prices under Items and stock."
                   if lang() == "en" else f"आज की {_money(s['sales_without_cost'])} की बिक्री उन सामानों की है जिनका खरीद भाव नहीं है, इसलिए उनका मुनाफ़ा नहीं जुड़ा।")
    st.caption("Profit is an estimate: sales minus what the items cost you, minus the expenses you entered. Salaries, rent and tax are only included if you entered them." if lang() == "en"
               else "मुनाफ़ा अनुमान है: बिक्री में से सामान की लागत और आपके दर्ज किए खर्चे घटाकर। वेतन, किराया और टैक्स तभी जुड़ते हैं जब आपने दर्ज किए हों।")
    if s["collected_by_mode"]:
        st.markdown(" · ".join(f"**{L(m)}** {_money(v)}" for m, v in sorted(s["collected_by_mode"].items())))
    st.link_button(L("share_summary"), bb.whatsapp_link("", bb.summary_text(s, shop, lang())))
    left, right = st.columns([1.3, 1])
    with left:
        if s["bills_table"].empty:
            st.caption(L("no_bills"))
        else:
            st.dataframe(s["bills_table"].rename(columns={"bill_no": "No", "customer": L("customer_col"), "total": L("total"), "paid": L("paid_now"), "mode": "Mode" if lang() == "en" else "तरीका"}), hide_index=True, width="stretch")
    with right:
        if not s["top_items"].empty:
            st.dataframe(s["top_items"].rename(columns={"item": L("item"), "qty": L("qty"), "sales": L("sales")}), hide_index=True, width="stretch")
    if _can_write(user):
        with st.expander(L("add_expense")):
            with st.form("bb_exp"):
                c1, c2, c3 = st.columns(3)
                cat = c1.selectbox(L("expense_type"), bb.EXPENSE_CATEGORIES, accept_new_options=True)
                amt = c2.number_input(L("amount"), min_value=0.0, step=10.0)
                md = c3.segmented_control(L("how_paid"), MODE_KEYS[:4], default="cash", format_func=L) or "cash"
                nt = st.text_input(L("note"), max_chars=200)
                ok = st.form_submit_button(L("save"))
            if ok:
                try:
                    bb.add_expense(cat, amt, md, day, nt, by=user.email)
                    _dirty()
                    st.rerun()
                except bb.BillError as e:
                    st.error(str(e))
        if not s["expenses_table"].empty:
            st.dataframe(s["expenses_table"].drop(columns="id"), hide_index=True, width="stretch")
    if accounts.can(user, "manage_users") and not s["bills_table"].empty:
        with st.expander(L("cancel_bill")):
            bn = st.selectbox("No", list(s["bills_table"]["bill_no"]), key="bb_cancel_no")
            st.caption("The stock goes back and the bill drops out of every total. The record stays, marked cancelled." if lang() == "en" else "स्टॉक वापस जुड़ जाएगा और बिल सभी हिसाब से हट जाएगा। रिकॉर्ड 'रद्द' लिखकर रहेगा।")
            if st.button(L("cancel_bill"), key="bb_cancel_go"):
                try:
                    bb.cancel_bill(bn, by=user.email, org_id=user.org_id)
                    _dirty()
                    st.rerun()
                except bb.BillError as e:
                    st.error(str(e))
    ser = bb.daily_series(14, day)
    if ser["sales"].sum() > 0:
        fig = go.Figure()
        fig.add_bar(x=ser["date"], y=ser["sales"], name=L("sales"), marker_color=theme.ACCENT)
        fig.add_scatter(x=ser["date"], y=ser["net_profit"], name=L("net_profit"), mode="lines+markers", line=dict(color=theme.ACCENT2))
        fig.update_layout(title="14 days" if lang() == "en" else "14 दिन")
        chart(fig, 300)
    _sync_notice()


# ------------------------------------------------------------------ Customers and dues
def page_dues(pr, recs, ctx):
    user, shop = ctx["user"], ctx["user"].org_name
    _toggle()
    theme.hero(L("dues"), L("dues_sub"), eyebrow=shop)
    t = bb.dues_table()
    if t.empty:
        st.success(L("nobody_owes"))
    else:
        theme.kpis([(L("total_due"), _money(float(t["due"].sum())), f"{len(t)} " + ("customers" if lang() == "en" else "ग्राहक")),
                    ("Oldest" if lang() == "en" else "सबसे पुराना", f"{int(t['days_since_oldest'].max())} " + ("days" if lang() == "en" else "दिन"))])
        view = t.assign(due=t["due"].map(_money)).rename(columns={"customer": L("customer_col"), "due": L("owes"), "oldest_bill_date": L("since"), "open_bills": "Bills" if lang() == "en" else "बिल"})
        st.dataframe(view.drop(columns=["days_since_oldest", "last_payment_date"]), hide_index=True, width="stretch")
        st.markdown(f"**{L('remind')}**")
        for r in t.head(15).itertuples():
            phone = bb.customer_phone(r.customer)
            c1, c2 = st.columns([3, 1])
            c1.write(f"{r.customer}: {_money(r.due)}" + ("" if bb.clean_phone(phone) else ("  (no phone saved)" if lang() == "en" else "  (फ़ोन नहीं है)")))
            c2.link_button("WhatsApp", bb.whatsapp_link(phone, bb.reminder_text(r.customer, r.due, shop, lang())), key=f"wa_{r.Index}")
        if _can_write(user):
            with st.expander(L("record_payment"), expanded=True):
                who = st.selectbox(L("customer_col"), list(t["customer"]), key="bb_pay_who")
                owed = float(t[t["customer"] == who]["due"].iloc[0])
                with st.form("bb_pay"):
                    amt = st.number_input(L("amount"), min_value=0.0, max_value=owed, value=owed, step=10.0, key=f"bb_pay_amt_{who}")
                    md = st.segmented_control(L("how_paid"), MODE_KEYS[:4], default="cash", format_func=L) or "cash"
                    ok = st.form_submit_button(L("save"))
                if ok:
                    try:
                        left = bb.record_payment(who, amt, md, by=user.email)
                        _dirty()
                        st.success(f"{_money(left)} " + ("still pending" if lang() == "en" else "अभी बाकी"))
                        st.rerun()
                    except bb.BillError as e:
                        st.error(str(e))
    names = bb.customer_names()
    if names:
        with st.expander("Customer statement" if lang() == "en" else "ग्राहक का खाता"):
            who = st.selectbox(L("customer_col"), names, key="bb_stmt_who")
            h = bb.customer_history(who)
            st.dataframe(h, hide_index=True, width="stretch") if len(h) else st.caption(L("no_bills"))
    _sync_notice()


# ------------------------------------------------------------------ Items and stock
def page_items(pr, recs, ctx):
    user, shop = ctx["user"], ctx["user"].org_name
    _toggle()
    theme.hero(L("items_stock"), L("items_sub"), eyebrow=shop)
    items = bb.list_items()
    low = bb.low_stock()
    if len(low):
        st.warning(f"{L('low_stock')}: " + ", ".join(f"{r.name} ({r.stock:g})" for r in low.head(8).itertuples()))
    if items.empty:
        st.info("No items yet. Add your first item below, or load a sheet." if lang() == "en" else "अभी कोई सामान नहीं। नीचे पहला सामान जोड़ें, या शीट से भरें।")
    else:
        st.dataframe(items.drop(columns=["active"]).rename(columns={"name": L("item"), "unit": L("unit"), "sale_price": L("sale_price"), "cost_price": L("cost_price"), "stock": L("stock"),
                                                                     "reorder_level": L("reorder"), "lead_days": "Lead days" if lang() == "en" else "आने में दिन"}), hide_index=True, width="stretch")
    if not _can_write(user):
        st.info(L("read_only"))
        return
    with st.expander(L("add_item"), expanded=items.empty):
        names = list(items["name"])
        pick = st.selectbox(L("item"), names, index=None, accept_new_options=True, key="bb_it_pick") or ""
        cur = items[items["name"].str.lower() == pick.strip().lower()] if pick else items.iloc[0:0]
        g = lambda col, d: (None if len(cur) == 0 or pd.isna(cur[col].iloc[0]) else float(cur[col].iloc[0])) if col != "unit" else (cur["unit"].iloc[0] if len(cur) else d)
        with st.form("bb_item_form"):
            c1, c2, c3 = st.columns(3)
            sp = c1.number_input(L("sale_price"), min_value=0.0, value=g("sale_price", 0.0) or 0.0, step=1.0, key=f"bb_sp_{pick}")
            cp = c2.number_input(L("cost_price"), min_value=0.0, value=g("cost_price", 0.0) or 0.0, step=1.0, key=f"bb_cp_{pick}")
            un = c3.text_input(L("unit"), value=g("unit", "pcs"), max_chars=12, key=f"bb_un_{pick}")
            c4, c5 = st.columns(2)
            stock = c4.number_input(L("stock"), value=g("stock", 0.0) or 0.0, step=1.0, key=f"bb_st_{pick}")
            ro = c5.number_input(L("reorder"), min_value=0.0, value=g("reorder_level", 0.0) or 0.0, step=1.0, key=f"bb_ro_{pick}")
            ok = st.form_submit_button(L("save"))
        if ok:
            try:
                bb.upsert_item(pick, sp, cp or None, un, stock, ro)
                _dirty()
                st.rerun()
            except bb.BillError as e:
                st.error(str(e))
    if len(items):
        with st.expander(L("stock_in")):
            with st.form("bb_stock_form"):
                c1, c2, c3 = st.columns(3)
                it = c1.selectbox(L("item"), list(items["name"]))
                q = c2.number_input(L("qty"), min_value=0.0, value=1.0, step=1.0)
                cost = c3.number_input(L("cost_price"), min_value=0.0, value=0.0, step=1.0)
                sup = st.text_input(L("supplier"), max_chars=80)
                ok = st.form_submit_button(L("save"))
            if ok:
                try:
                    now = bb.add_stock(it, q, cost or None, sup)
                    _dirty()
                    st.success(f"{L('stock')}: {now:g}")
                    st.rerun()
                except bb.BillError as e:
                    st.error(str(e))
    with st.expander(L("import_sheet")):
        st.caption("A sheet with columns for the item name and selling price (and optionally cost, stock, unit)." if lang() == "en" else "ऐसी शीट जिसमें सामान का नाम और बेचने का भाव हो (खरीद भाव, स्टॉक, इकाई वैकल्पिक)।")
        up = st.file_uploader("CSV / Excel", type=["csv", "xlsx"], key="bb_import")
        if up is not None and st.button(L("save"), key="bb_import_go"):
            try:
                res = bb.import_items(data_layer.read_upload(up, up.name))
                st.success(f"{res['added']} added, {res['updated']} updated")
                for e in res["errors"][:10]:
                    st.warning(e)
                _dirty()
            except (bb.BillError, ValueError) as e:
                st.error(str(e))
    _sync_notice()
