"""SupplyChainIQ - inventory turnover, stock-out / overstock detection, ABC analysis,
demand forecasting, supplier performance scoring and reorder-point calculation."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .core import AMBER, GREEN, RED, ModuleResult, inr, safe_div, score_between

Z_SERVICE = 1.65         # 95% service level
REVIEW_DAYS = 14         # order cover beyond lead time
SLOW_TURNOVER = 3.0      # annual turns below this = slow moving
DEFAULT_COST_RATIO = 0.60


def holt_damped(y: np.ndarray, h: int = 3, phi: float = 0.9):
    """Damped-trend exponential smoothing with a small grid search for alpha/beta.
    Returns (forecast array, residual std)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        m = float(np.mean(y)) if len(y) else 0.0
        return np.full(h, m), float(np.std(y)) if len(y) > 1 else 0.0
    best = None
    for a in (0.2, 0.4, 0.6, 0.8):
        for b in (0.05, 0.15, 0.3):
            level, trend = y[0], y[1] - y[0]
            sse = 0.0
            for t in range(1, len(y)):
                pred = level + phi * trend
                sse += (y[t] - pred) ** 2
                new_level = a * y[t] + (1 - a) * (level + phi * trend)
                trend = b * (new_level - level) + (1 - b) * phi * trend
                level = new_level
            if best is None or sse < best[0]:
                best = (sse, level, trend)
    sse, level, trend = best
    fc = np.array([level + sum(phi ** k for k in range(1, i + 2)) * trend for i in range(h)])
    return np.maximum(fc, 0), math.sqrt(sse / max(len(y) - 1, 1))


def demand_stats(sales: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    end = sales["date"].max()
    s = sales[sales["date"] > end - pd.Timedelta(days=window)]
    daily = (s.groupby(["product", "date"])["quantity"].sum().unstack(fill_value=0)
             .reindex(columns=pd.date_range(end - pd.Timedelta(days=window - 1), end), fill_value=0))
    return pd.DataFrame({"avg_daily_demand": daily.mean(axis=1), "demand_std": daily.std(axis=1)})


def abc_analysis(sales: pd.DataFrame) -> pd.DataFrame:
    p = sales.groupby("product")["revenue"].sum().sort_values(ascending=False).to_frame("revenue")
    p["share"] = p["revenue"] / p["revenue"].sum()
    p["cum_share"] = p["share"].cumsum()
    p["abc_class"] = np.where(p["cum_share"] - p["share"] < 0.80, "A",
                              np.where(p["cum_share"] - p["share"] < 0.95, "B", "C"))
    return p.reset_index()


def supplier_scores(deliv: pd.DataFrame) -> pd.DataFrame:
    d = deliv.copy()
    d["delay_days"] = (d["delivered_date"] - d["promised_date"]).dt.days
    d["on_time"] = d["delay_days"] <= 0
    g = d.groupby("supplier")
    out = pd.DataFrame({
        "deliveries": g.size(), "on_time_rate": g["on_time"].mean(),
        "avg_delay_days": g["delay_days"].apply(lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
        "fill_rate": g["qty_received"].sum() / g["qty_ordered"].sum(),
        "defect_rate": g["defective_units"].sum() / g["qty_received"].sum().clip(lower=1),
    })
    # price index: how this supplier's price compares with the product average
    avg_price = d.groupby("product")["unit_price"].transform("mean")
    d["price_idx"] = d["unit_price"] / avg_price
    out["price_index"] = d.groupby("supplier")["price_idx"].mean()
    out["score"] = (0.40 * out["on_time_rate"].apply(lambda v: score_between(v, 0.5, 0.98)) +
                    0.20 * out["fill_rate"].apply(lambda v: score_between(v, 0.85, 1.0)) +
                    0.25 * out["defect_rate"].apply(lambda v: score_between(v, 0.06, 0.005)) +
                    0.15 * out["price_index"].apply(lambda v: score_between(v, 1.15, 0.90)))
    out["grade"] = pd.cut(out["score"], [-1, 50, 70, 101], labels=["Poor", "Fair", "Good"])
    return out.sort_values("score", ascending=False).reset_index()


def unit_costs(sales: pd.DataFrame, deliv: pd.DataFrame | None) -> pd.Series:
    if deliv is not None and len(deliv):
        c = deliv.groupby("product")["unit_price"].mean()
    else:
        c = pd.Series(dtype=float)
    fallback = sales.groupby("product")["unit_price"].mean() * DEFAULT_COST_RATIO
    return c.combine_first(fallback)


def run(sales: pd.DataFrame, inventory: pd.DataFrame, deliveries: pd.DataFrame | None = None,
        fin: pd.DataFrame | None = None) -> ModuleResult:
    res = ModuleResult("SupplyChainIQ")
    stats = demand_stats(sales)
    inv = inventory.set_index("product").join(stats, how="left", rsuffix="_sales")
    if "avg_daily_demand" in inv.columns and "avg_daily_demand_sales" in inv.columns:
        inv["avg_daily_demand"] = inv["avg_daily_demand_sales"].fillna(inv["avg_daily_demand"])
        inv = inv.drop(columns=["avg_daily_demand_sales"])
    inv["avg_daily_demand"] = inv["avg_daily_demand"].fillna(0.0)
    inv["demand_std"] = inv["demand_std"].fillna(inv["avg_daily_demand"] * 0.5)
    costs = unit_costs(sales, deliveries)
    inv["unit_cost"] = costs.reindex(inv.index).fillna(0.0)
    inv["stock_value"] = inv["stock_on_hand"] * inv["unit_cost"]

    d, lt = inv["avg_daily_demand"], inv["lead_time_days"]
    inv["days_of_cover"] = np.where(d > 0, inv["stock_on_hand"] / d.replace(0, np.nan), np.inf)
    inv["safety_stock"] = (Z_SERVICE * inv["demand_std"] * np.sqrt(lt)).round(0)
    inv["reorder_point"] = (d * lt + inv["safety_stock"]).round(0)
    inv["suggested_order_qty"] = np.ceil(np.maximum(0, d * (lt + REVIEW_DAYS) + inv["safety_stock"]
                                                    - inv["stock_on_hand"]))
    annual_units = sales[sales["date"] > sales["date"].max() - pd.Timedelta(days=365)] \
        .groupby("product")["quantity"].sum()
    inv["annual_units_sold"] = annual_units.reindex(inv.index).fillna(0)
    inv["turnover"] = inv["annual_units_sold"] / inv["stock_on_hand"].replace(0, np.nan)
    over_cut = np.maximum(90, 4 * lt)

    def status(r):
        cover, ltd = r["days_of_cover"], r["lead_time_days"]
        if r["stock_on_hand"] <= 0 or cover < ltd * 0.5:
            return "Stock-out risk (critical)"
        if r["stock_on_hand"] <= r["reorder_point"]:
            return "Reorder now"
        if cover > max(90, 4 * ltd):
            return "Overstock"
        return "OK"
    inv["status"] = inv.apply(status, axis=1)
    inv["slow_moving"] = inv["turnover"] < SLOW_TURNOVER
    inv["overstock_value"] = np.where(
        inv["status"] == "Overstock",
        (inv["stock_on_hand"] - d * over_cut).clip(lower=0) * inv["unit_cost"], 0.0)
    abc = abc_analysis(sales).set_index("product")
    inv = inv.join(abc[["abc_class"]])
    inv = inv.reset_index()
    res.tables["inventory"] = inv
    res.tables["reorder"] = inv[inv["status"].isin(["Stock-out risk (critical)", "Reorder now"])][
        ["product", "abc_class", "stock_on_hand", "days_of_cover", "lead_time_days", "reorder_point",
         "suggested_order_qty", "status"]].sort_values("days_of_cover")
    res.tables["abc"] = abc.reset_index()
    res.tables["abc_summary"] = abc.groupby("abc_class").agg(
        products=("revenue", "size"), revenue=("revenue", "sum"), share=("share", "sum")).reset_index()

    # Forecast: next 3 months of units per product + total
    monthly = (sales.set_index("date").groupby("product")["quantity"].resample("MS").sum()
               .unstack(0).fillna(0))
    rows = []
    for prod_name in monthly.columns:
        fc, sd = holt_damped(monthly[prod_name].values, 3)
        for i, v in enumerate(fc, 1):
            rows.append(dict(product=prod_name, month=monthly.index[-1] + pd.DateOffset(months=i),
                             forecast_units=float(v), lower=max(0.0, float(v) - 1.28 * sd),
                             upper=float(v) + 1.28 * sd))
    res.tables["forecast"] = pd.DataFrame(rows)
    res.tables["history_units"] = monthly.reset_index()

    if deliveries is not None and len(deliveries):
        res.tables["suppliers"] = supplier_scores(deliveries)

    # Working capital metrics
    inv_value = float(inv["stock_value"].sum())
    turnover_fin = float("nan")
    if fin is not None and len(fin):
        f = fin.sort_values("month")
        turnover_fin = safe_div(f["cogs"].tail(12).sum(), f["inventory"].tail(12).mean())
    slow = inv[inv["slow_moving"]]
    over = inv[inv["status"] == "Overstock"]
    crit = inv[inv["status"] == "Stock-out risk (critical)"]
    reorder = inv[inv["status"] == "Reorder now"]
    res.metrics = dict(inventory_value=inv_value, turnover_financials=turnover_fin,
                       products=len(inv), stockout_risk=len(crit), reorder_now=len(reorder),
                       overstocked=len(over), slow_moving=len(slow),
                       slow_moving_value=float(slow["stock_value"].sum()),
                       overstock_excess_value=float(inv["overstock_value"].sum()))
    res.facts = dict(
        products_tracked=len(inv), inventory_value=inr(inv_value),
        inventory_turnover_per_year=None if pd.isna(turnover_fin) else round(turnover_fin, 1),
        stockout_risk_products=len(crit), reorder_now_products=len(reorder),
        overstocked_products=len(over), slow_moving_products=len(slow),
        slow_moving_value=inr(float(slow["stock_value"].sum())),
        overstock_excess_value=inr(float(inv["overstock_value"].sum())),
        a_class_products=int((inv["abc_class"] == "A").sum()),
    )
    if len(crit):
        names = ", ".join(crit["product"].head(4))
        res.add_flag(RED, f"{len(crit)} product(s) at critical stock-out risk", names)
        res.facts["critical_products"] = names
    if len(reorder):
        names = ", ".join(reorder["product"].head(4))
        res.add_flag(AMBER, f"{len(reorder)} product(s) have reached their reorder point", names)
        res.facts["reorder_products"] = names
    if len(slow):
        res.add_flag(AMBER, f"{len(slow)} slow-moving SKU(s) tie up {inr(float(slow['stock_value'].sum()))}",
                     ", ".join(slow["product"].head(4)))
    if len(over):
        res.add_flag(AMBER, f"{len(over)} overstocked product(s); excess stock worth "
                     f"{inr(float(inv['overstock_value'].sum()))}", ", ".join(over["product"].head(4)))
    if "suppliers" in res.tables:
        sc = res.tables["suppliers"]
        poor = sc[sc["score"] < 50]
        res.facts["best_supplier"] = sc.iloc[0]["supplier"]
        res.facts["best_supplier_score"] = round(float(sc.iloc[0]["score"]), 0)
        res.facts["weakest_supplier"] = sc.iloc[-1]["supplier"]
        res.facts["weakest_supplier_score"] = round(float(sc.iloc[-1]["score"]), 0)
        res.facts["weakest_supplier_on_time_pct"] = round(float(sc.iloc[-1]["on_time_rate"]) * 100, 0)
        if len(poor):
            res.add_flag(RED if len(poor) > 1 else AMBER,
                         f"{len(poor)} supplier(s) score below 50",
                         ", ".join(poor["supplier"]))
    if not res.flags:
        res.add_flag(GREEN, "Inventory levels look healthy")
    res.notes.append(f"Reorder point = average daily demand x lead time + safety stock "
                     f"({Z_SERVICE} x demand std x sqrt(lead time), ~95% service level). Suggested order "
                     f"covers lead time plus {REVIEW_DAYS} days. Demand uses the last 90 days. "
                     f"Forecast uses damped-trend exponential smoothing on monthly units.")
    return res
