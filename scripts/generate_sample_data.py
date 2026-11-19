"""Generate SYNTHETIC sample data for BizIQ.

Everything produced here is made up. It is labelled as synthetic in the README
and in the app. Never present it as real client data.

Usage:  python scripts/generate_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "sample_data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)


def financials() -> pd.DataFrame:
    """24 months of P&L, balance sheet and cash flow for a fictional retailer.

    Margins erode, receivable and inventory days stretch and cash is derived from
    cash flow, so the platform has real, internally consistent problems to find.
    """
    months = pd.period_range("2024-11", periods=24, freq="M")
    rows = []
    prev = None
    for i, m in enumerate(months):
        season = 1 + 0.15 * np.sin(2 * np.pi * (m.month - 3) / 12)
        revenue = 4_200_000 * (1 + 0.012 * i) * season * rng.normal(1, 0.03)
        cogs = revenue * (0.58 + 0.006 * i) * rng.normal(1, 0.01)
        opex = 850_000 * (1 + 0.008 * i) * rng.normal(1, 0.02)
        if i == 21:  # planted anomaly: unusual expense spike for RiskIQ to catch
            opex *= 1.45
        depreciation = 60_000
        interest = 70_000 + 3_000 * i
        ebit = revenue - cogs - opex - depreciation
        tax = max(ebit - interest, 0) * 0.25
        net_income = ebit - interest - tax

        receivables = revenue * (0.60 + 0.025 * i)
        inventory = cogs * (1.00 + 0.03 * i)
        payables = cogs * 0.70
        other_ca = 150_000
        short_debt = 6_000_000 + 120_000 * i
        long_debt = 16_000_000 - 100_000 * i
        if prev is None:
            cash = 2_400_000
            op_cf = net_income + depreciation
        else:
            op_cf = (net_income + depreciation - (receivables - prev["receivables"])
                     - (inventory - prev["inventory"]) + (payables - prev["payables"]))
            capex = 90_000
            debt_flow = (short_debt - prev["short_term_debt"]) + (long_debt - prev["long_term_debt"])
            cash = prev["cash"] + op_cf - capex + debt_flow
        fixed_assets = 28_000_000 - depreciation * i + 90_000 * i
        current_assets = cash + receivables + inventory + other_ca
        total_assets = current_assets + fixed_assets
        current_liabilities = payables + short_debt + 250_000
        total_liabilities = current_liabilities + long_debt
        equity = total_assets - total_liabilities
        retained_earnings = 3_500_000 + sum(r["net_income"] for r in rows) + net_income
        row = dict(
            month=str(m), revenue=revenue, cogs=cogs, operating_expenses=opex,
            depreciation=depreciation, interest_expense=interest, tax=tax,
            net_income=net_income, cash=cash, receivables=receivables,
            inventory=inventory, other_current_assets=other_ca,
            fixed_assets=fixed_assets, total_assets=total_assets,
            payables=payables, short_term_debt=short_debt,
            current_liabilities=current_liabilities, long_term_debt=long_debt,
            total_liabilities=total_liabilities, retained_earnings=retained_earnings,
            equity=equity, operating_cash_flow=op_cf,
        )
        rows.append(row)
        prev = row
    return pd.DataFrame(rows).round(0)


def sales(n_orders: int = 24000) -> pd.DataFrame:
    """Transaction-level sales for the last 12 months."""
    products = {
        "Classic Tee": ("Apparel", 499), "Denim Jacket": ("Apparel", 2499),
        "Running Shoes": ("Footwear", 3299), "Sandals": ("Footwear", 899),
        "Backpack": ("Accessories", 1599), "Sunglasses": ("Accessories", 1299),
        "Hoodie": ("Apparel", 1799), "Wrist Watch": ("Accessories", 4499),
        "Sneakers": ("Footwear", 2799), "Cap": ("Accessories", 349),
    }
    names = list(products)
    # Popularity skew so top/bottom analysis is interesting
    p = np.array([0.20, 0.06, 0.09, 0.10, 0.08, 0.07, 0.14, 0.03, 0.08, 0.15])
    p = p / p.sum()
    regions = ["North", "South", "East", "West"]
    rp = [0.30, 0.25, 0.15, 0.30]
    n_customers = 2200
    cust = [f"C{idx:04d}" for idx in range(1, n_customers + 1)]
    cust_w = rng.pareto(2.0, n_customers) + 1
    cust_w = cust_w / cust_w.sum()

    end = pd.Timestamp("2026-10-31")
    start = end - pd.Timedelta(days=364)
    days = pd.date_range(start, end)
    day_w = np.array([1 + 0.25 * np.sin(2 * np.pi * d.dayofyear / 365) +
                      (0.25 if d.dayofweek >= 5 else 0) for d in days])
    day_w = day_w / day_w.sum()
    # Planted anomaly: a 10-day sales slump in the most recent weeks of data
    slump = (days >= end - pd.Timedelta(days=16)) & (days <= end - pd.Timedelta(days=7))
    day_w = np.where(slump, day_w * 0.45, day_w)
    day_w = day_w / day_w.sum()

    # Some customers join late, ~28% lapse part-way through the year (churn)
    lapse = {c: start + pd.Timedelta(days=int(rng.integers(60, 330)))
             if rng.random() < 0.28 else None for c in cust}
    first = {c: start + pd.Timedelta(days=int(rng.integers(0, 200)))
             if rng.random() < 0.35 else start for c in cust}
    pairs = [("Sneakers", "Cap"), ("Denim Jacket", "Classic Tee"),
             ("Backpack", "Sunglasses"), ("Hoodie", "Cap")]
    cust_arr = np.array(cust)
    cust_region = {c: rng.choice(regions, p=rp) for c in cust}
    recs = []
    n_orders = n_orders // 2  # orders have 1-3 lines, so ~n_orders lines in total
    for oid in range(1, n_orders + 1):
        d = days[rng.choice(len(days), p=day_w)]
        c = None
        for _ in range(60):
            cand = cust_arr[rng.choice(n_customers, p=cust_w)]
            if first[cand] <= d and (lapse[cand] is None or d <= lapse[cand]):
                c = cand
                break
        if c is None:
            c = cust_arr[0]
        n_lines = int(rng.choice([1, 2, 3], p=[0.55, 0.33, 0.12]))
        if n_lines >= 2 and rng.random() < 0.45:
            basket = list(pairs[rng.integers(0, len(pairs))])
        else:
            basket = []
        while len(basket) < n_lines:
            pick = str(rng.choice(names, p=p))
            if pick not in basket:
                basket.append(pick)
        for pname in basket[:max(n_lines, len(basket))]:
            recs.append(dict(order_id=f"O{oid:06d}", date=d.strftime("%Y-%m-%d"),
                             customer_id=c, product=pname,
                             category=products[pname][0], region=cust_region[c],
                             quantity=int(rng.integers(1, 4)),
                             unit_price=products[pname][1]))
    df = pd.DataFrame(recs)
    df["revenue"] = df["quantity"] * df["unit_price"]
    return df.sort_values("date").reset_index(drop=True)


def inventory(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Stock snapshot sized from actual demand: a few products critically low, one
    at its reorder point, two overstocked, the rest healthy."""
    last = pd.to_datetime(sales_df["date"]).max()
    recent = sales_df[pd.to_datetime(sales_df["date"]) > last - pd.Timedelta(days=90)]
    demand = recent.groupby("product")["quantity"].sum() / 90
    # product: (days of cover, supplier lead time in days)
    plan = {"Classic Tee": (30, 14), "Denim Jacket": (38, 21), "Running Shoes": (6, 21),
            "Sandals": (130, 14), "Backpack": (12, 14), "Sunglasses": (45, 21),
            "Hoodie": (5, 14), "Wrist Watch": (170, 30), "Sneakers": (40, 21), "Cap": (28, 7)}
    rows = [(p, int(round(demand[p] * cover)), round(float(demand[p]), 2), lead)
            for p, (cover, lead) in plan.items()]
    return pd.DataFrame(rows, columns=["product", "stock_on_hand", "avg_daily_demand",
                                       "lead_time_days"])


if __name__ == "__main__":
    financials().to_csv(OUT / "financials.csv", index=False)
    sales_df = sales()
    sales_df.to_csv(OUT / "sales.csv", index=False)
    inventory(sales_df).to_csv(OUT / "inventory.csv", index=False)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import generate_extra_data as ex
    ex.supplier_deliveries().to_csv(OUT / "supplier_deliveries.csv", index=False)
    ex.invoices().to_csv(OUT / "invoices.csv", index=False)
    ex.hr().to_csv(OUT / "hr.csv", index=False)
    ex.social().to_csv(OUT / "social.csv", index=False)
    ex.reviews().to_csv(OUT / "reviews.csv", index=False)
    ex.portfolio().to_csv(OUT / "portfolio.csv", index=False)
    print("Wrote synthetic sample data to", OUT)
