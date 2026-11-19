"""ProcureIQ - compare suppliers on price, quality and delivery; draft purchase orders."""
from __future__ import annotations

import math

import pandas as pd

from .core import AMBER, GREEN, RED, ModuleResult, inr, safe_div, score_between

W_PRICE, W_DELIVERY, W_QUALITY = 0.40, 0.35, 0.25


def compare(deliv: pd.DataFrame, as_of: pd.Timestamp, months: int = 12) -> pd.DataFrame:
    d = deliv[deliv["order_date"] > as_of - pd.DateOffset(months=months)].copy()
    d["late"] = (d["delivered_date"] > d["promised_date"]).astype(int)
    d["delay_days"] = (d["delivered_date"] - d["promised_date"]).dt.days.clip(lower=0)
    g = d.groupby(["product", "supplier"])
    t = pd.DataFrame({"avg_unit_price": g["unit_price"].mean(), "orders": g.size(),
                      "on_time_rate": 1 - g["late"].mean(), "avg_delay_days": g["delay_days"].mean(),
                      "defect_rate": g["defective_units"].sum() / g["qty_received"].sum().clip(lower=1),
                      "volume": g["qty_received"].sum()}).reset_index()
    best_price = t.groupby("product")["avg_unit_price"].transform("min")
    t["price_vs_best_pct"] = t["avg_unit_price"] / best_price - 1
    t["price_score"] = t["price_vs_best_pct"].apply(lambda v: score_between(v, 0.25, 0.0))
    t["delivery_score"] = t["on_time_rate"].apply(lambda v: score_between(v, 0.5, 0.98))
    t["quality_score"] = t["defect_rate"].apply(lambda v: score_between(v, 0.06, 0.005))
    t["composite"] = W_PRICE * t["price_score"] + W_DELIVERY * t["delivery_score"] + W_QUALITY * t["quality_score"]
    t["rank"] = t.groupby("product")["composite"].rank(ascending=False, method="first", na_option="bottom").astype(int)
    return t.sort_values(["product", "rank"])


def draft_purchase_orders(reorder: pd.DataFrame, comp: pd.DataFrame, as_of: pd.Timestamp,
                          lead_times: dict[str, int]) -> tuple[pd.DataFrame, dict[str, str]]:
    best = comp[comp["rank"] == 1].set_index("product")
    lines = []
    for _, r in reorder.iterrows():
        if pd.isna(r["suggested_order_qty"]) or r["suggested_order_qty"] <= 0 or r["product"] not in best.index:
            continue
        b = best.loc[r["product"]]
        lead = int(lead_times.get(r["product"], 14))
        lines.append(dict(supplier=b["supplier"], product=r["product"], quantity=int(r["suggested_order_qty"]),
                          unit_price=float(b["avg_unit_price"]),
                          line_total=float(b["avg_unit_price"]) * int(r["suggested_order_qty"]),
                          expected_delivery=(as_of + pd.Timedelta(days=lead + int(math.ceil(b["avg_delay_days"])) if pd.notna(b["avg_delay_days"]) else lead)).date(),
                          reason=r["status"]))
    po = pd.DataFrame(lines)
    docs = {}
    if len(po):
        for sup, g in po.groupby("supplier"):
            body = [f"DRAFT PURCHASE ORDER - for human approval", f"Date: {as_of:%d %b %Y}",
                    f"Supplier: {sup}", "", f"{'Product':<20}{'Qty':>8}{'Unit price':>14}{'Total':>16}"]
            for _, r in g.iterrows():
                body.append(f"{r['product']:<20}{r['quantity']:>8}{r['unit_price']:>14,.2f}{r['line_total']:>16,.0f}")
            body += ["", f"Order total (estimate): Rs {g['line_total'].sum():,.0f}",
                     "Prices are recent averages; confirm current quotes and terms before issuing.",
                     "Approved by: ________________   Date: ____________"]
            docs[sup] = "\n".join(body)
    return po, docs


def run(deliveries: pd.DataFrame, as_of: pd.Timestamp, reorder: pd.DataFrame | None = None,
        lead_times: dict[str, int] | None = None) -> ModuleResult:
    res = ModuleResult("ProcureIQ")
    comp = compare(deliveries, as_of)
    res.tables["comparison"] = comp
    best = comp[comp["rank"] == 1][["product", "supplier", "avg_unit_price", "on_time_rate", "defect_rate", "composite"]] \
        .rename(columns={"supplier": "recommended_supplier"})
    res.tables["recommended_suppliers"] = best

    # Savings if each product was bought from its best-ranked supplier
    spend = deliveries[deliveries["order_date"] > as_of - pd.DateOffset(months=12)].copy()
    spend["spend"] = spend["qty_received"] * spend["unit_price"]
    cur = spend.groupby(["product", "supplier"]).agg(spend=("spend", "sum"), qty=("qty_received", "sum")).reset_index() \
        .merge(comp[["product", "supplier", "avg_unit_price", "rank"]], on=["product", "supplier"], how="left")
    bestp = best.set_index("product")["avg_unit_price"]
    cur["best_price"] = cur["product"].map(bestp)
    cur["potential_saving"] = ((cur["avg_unit_price"] - cur["best_price"]) * cur["qty"]).clip(lower=0)
    # Only count savings from suppliers that are not materially worse on delivery
    sav_by_product = cur.groupby("product")["potential_saving"].sum().sort_values(ascending=False).reset_index()
    res.tables["savings_opportunities"] = sav_by_product
    total_sav = float(sav_by_product["potential_saving"].sum())
    total_spend = float(cur["spend"].sum())

    spread = comp.groupby("product")["avg_unit_price"].agg(["min", "max", "count"])
    spread["spread_pct"] = spread["max"] / spread["min"] - 1
    wide = spread[(spread["count"] > 1) & (spread["spread_pct"] > 0.10)]
    res.tables["price_spread"] = spread.reset_index()

    po, docs = (pd.DataFrame(), {})
    if reorder is not None and len(reorder):
        po, docs = draft_purchase_orders(reorder, comp, as_of, lead_times or {})
    res.tables["draft_purchase_orders"] = po
    res.tables["purchase_order_documents"] = pd.DataFrame({"supplier": list(docs), "document": list(docs.values())})

    res.metrics = dict(annual_spend=total_spend, potential_saving=total_sav, draft_pos=len(docs),
                       po_value=float(po["line_total"].sum()) if len(po) else 0.0)
    res.facts = dict(annual_spend=inr(total_spend), potential_annual_saving=inr(total_sav),
                     potential_saving_pct=round(safe_div(total_sav, total_spend, 0) * 100, 1),
                     products_with_wide_price_spread=len(wide), draft_purchase_orders=len(docs),
                     draft_po_value=inr(res.metrics["po_value"]))
    if len(sav_by_product) and sav_by_product.iloc[0]["potential_saving"] > 0:
        res.facts["biggest_saving_product"] = sav_by_product.iloc[0]["product"]
        res.facts["biggest_saving_amount"] = inr(float(sav_by_product.iloc[0]["potential_saving"]))
    if total_spend and total_sav / total_spend > 0.03:
        res.add_flag(AMBER, f"Up to {inr(total_sav)} a year could be saved by shifting volume to the best-ranked suppliers",
                     "Check delivery and quality scores before switching.")
    if len(wide):
        res.add_flag(AMBER, f"{len(wide)} product(s) show a price spread above 10% between suppliers", ", ".join(wide.index[:4]))
    if len(docs):
        res.add_flag(AMBER, f"{len(docs)} draft purchase order(s) ready for approval",
                     f"Estimated value {inr(res.metrics['po_value'])}.")
    if not res.flags:
        res.add_flag(GREEN, "Procurement looks in order")
    res.notes.append(f"Composite = {W_PRICE:.0%} price + {W_DELIVERY:.0%} on-time delivery + {W_QUALITY:.0%} quality "
                     "over the last 12 months. Savings are estimates assuming the best-ranked supplier can absorb the volume. "
                     "Purchase orders are drafts for human approval and are never sent automatically.")
    return res
