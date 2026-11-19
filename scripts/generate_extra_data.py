"""SYNTHETIC data for the extended modules (suppliers, invoices, HR, social,
reviews, portfolio). Everything here is made up and labelled as such."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
AS_OF = pd.Timestamp("2026-10-31")
PRODUCTS = ["Classic Tee", "Denim Jacket", "Running Shoes", "Sandals", "Backpack",
            "Sunglasses", "Hoodie", "Wrist Watch", "Sneakers", "Cap"]
BASE_COST = {"Classic Tee": 260, "Denim Jacket": 1350, "Running Shoes": 1750,
             "Sandals": 470, "Backpack": 820, "Sunglasses": 640, "Hoodie": 950,
             "Wrist Watch": 2300, "Sneakers": 1500, "Cap": 170}


def supplier_deliveries() -> pd.DataFrame:
    # name: (on-time prob, avg delay when late, defect rate, price multiplier)
    sup = {"Aarav Textiles": (0.92, 2, 0.010, 1.00),
           "Bharat Garments": (0.70, 6, 0.030, 0.93),
           "Coastal Footwear": (0.85, 3, 0.015, 1.02),
           "Delta Accessories": (0.60, 8, 0.045, 0.90),
           "Evergreen Supply Co": (0.95, 1, 0.008, 1.08)}
    supplies = {"Aarav Textiles": ["Classic Tee", "Hoodie", "Denim Jacket"],
                "Bharat Garments": ["Classic Tee", "Hoodie", "Denim Jacket", "Cap"],
                "Coastal Footwear": ["Running Shoes", "Sandals", "Sneakers"],
                "Delta Accessories": ["Backpack", "Sunglasses", "Wrist Watch", "Cap", "Sandals"],
                "Evergreen Supply Co": ["Backpack", "Sunglasses", "Wrist Watch", "Running Shoes",
                                        "Sneakers", "Cap"]}
    rows, k = [], 1
    for s, (ontime, delay, defect, pm) in sup.items():
        for _ in range(80):
            prod = str(rng.choice(supplies[s]))
            od = AS_OF - pd.Timedelta(days=int(rng.integers(10, 360)))
            lead = int(rng.integers(7, 22))
            promised = od + pd.Timedelta(days=lead)
            late = rng.random() > ontime
            dd = promised + pd.Timedelta(days=int(max(1, rng.normal(delay, 1.5)))) if late else \
                promised - pd.Timedelta(days=int(rng.integers(0, 3)))
            qo = int(rng.integers(30, 200))
            qr = int(qo * (1 if rng.random() > 0.12 else rng.uniform(0.8, 0.97)))
            rows.append(dict(delivery_id=f"D{k:05d}", supplier=s, product=prod,
                             order_date=od.strftime("%Y-%m-%d"),
                             promised_date=promised.strftime("%Y-%m-%d"),
                             delivered_date=dd.strftime("%Y-%m-%d"),
                             qty_ordered=qo, qty_received=qr,
                             defective_units=int(rng.binomial(qr, defect)),
                             unit_price=round(BASE_COST[prod] * pm * rng.normal(1, 0.03), 2)))
            k += 1
    return pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)


def invoices() -> pd.DataFrame:
    # (party, type, typical days to pay, share paid)
    recv = {"Metro Retail Pvt Ltd": 38, "Sunrise Boutiques": 62, "Urban Mart": 33,
            "Kapoor & Sons": 95, "Lakeview Stores": 41, "Zenith Outfitters": 72}
    pay = {"Aarav Textiles": 30, "Bharat Garments": 25, "Coastal Footwear": 35,
           "Delta Accessories": 20, "Evergreen Supply Co": 45}
    rows, k = [], 1
    for typ, parties, n in (("receivable", recv, 22), ("payable", pay, 14)):
        for party, avg_days in parties.items():
            for _ in range(n):
                inv = AS_OF - pd.Timedelta(days=int(rng.integers(0, 300)))
                terms = 30 if typ == "receivable" else int(rng.choice([30, 45]))
                due = inv + pd.Timedelta(days=terms)
                pay_days = max(3, int(rng.normal(avg_days, 8)))
                paid = inv + pd.Timedelta(days=pay_days)
                is_paid = paid <= AS_OF
                rows.append(dict(invoice_id=f"INV{k:05d}", party=party, type=typ,
                                 invoice_date=inv.strftime("%Y-%m-%d"),
                                 due_date=due.strftime("%Y-%m-%d"),
                                 amount=int(rng.integers(40_000, 420_000)),
                                 paid_date=paid.strftime("%Y-%m-%d") if is_paid else ""))
                k += 1
    return pd.DataFrame(rows)


def hr() -> pd.DataFrame:
    depts = {"Sales": (14, 38_000), "Store Ops": (18, 26_000), "Warehouse": (10, 24_000),
             "Finance": (4, 55_000), "Marketing": (5, 48_000), "IT": (4, 70_000),
             "HR": (2, 50_000)}
    rows, k = [], 1
    for d, (n, base) in depts.items():
        for _ in range(n):
            join = AS_OF - pd.Timedelta(days=int(rng.integers(30, 2200)))
            perf = int(np.clip(rng.normal(3.4, 0.9), 1, 5))
            ot = float(np.clip(rng.normal(18 if d in ("Store Ops", "Warehouse") else 6, 6), 0, 60))
            sat = float(np.clip(rng.normal(7.2 - ot / 20, 1.3), 1, 10))
            left = rng.random() < (0.15 + max(0, ot - 20) / 150 + max(0, 6 - sat) / 30)
            exit_d = ""
            if left:
                ed = AS_OF - pd.Timedelta(days=int(rng.integers(5, 330)))
                if ed > join:
                    exit_d = ed.strftime("%Y-%m-%d")
            rows.append(dict(employee_id=f"E{k:04d}", department=d,
                             monthly_salary=int(base * 0.55 * rng.normal(1, 0.18)),
                             join_date=join.strftime("%Y-%m-%d"), exit_date=exit_d,
                             performance_rating=perf,
                             overtime_hours_month=round(ot, 1),
                             satisfaction_score=round(sat, 1)))
            k += 1
    return pd.DataFrame(rows)


def social() -> pd.DataFrame:
    plats = {"Instagram": 1.0, "Facebook": 0.55, "LinkedIn": 0.25, "X": 0.35}
    types = {"Reel": 1.8, "Carousel": 1.3, "Image": 1.0, "Story": 0.7, "Text": 0.4}
    themes = ["new arrivals", "festive offer", "behind the scenes", "customer story",
              "styling tips", "sustainability", "sale", "giveaway"]
    rows = []
    for _ in range(180):
        pl = str(rng.choice(list(plats)))
        ty = str(rng.choice(list(types)))
        th = str(rng.choice(themes))
        imp = int(rng.lognormal(8.2, 0.5) * plats[pl] * types[ty])
        eng = imp * rng.uniform(0.02, 0.07) * types[ty] ** 0.5
        rows.append(dict(date=(AS_OF - pd.Timedelta(days=int(rng.integers(0, 180)))).strftime("%Y-%m-%d"),
                         platform=pl, post_type=ty, theme=th, impressions=imp,
                         likes=int(eng * 0.7), comments=int(eng * 0.1),
                         shares=int(eng * 0.08), clicks=int(imp * rng.uniform(0.005, 0.03))))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def reviews() -> pd.DataFrame:
    pos = ["Great quality and fits perfectly", "Love it, very comfortable", "Excellent value for money",
           "Fast delivery and lovely packaging", "Looks stylish, would recommend",
           "Amazing product, my favourite purchase", "Superb material, durable"]
    neg = ["Poor quality, stitching came apart", "Delivery was late and support unhelpful",
           "Too expensive for what it is", "Size was wrong and return was a hassle",
           "Disappointed, colour faded after one wash", "Terrible packaging, item damaged"]
    neu = ["It is okay, nothing special", "Average product, as described"]
    rows = []
    for _ in range(220):
        prod = str(rng.choice(PRODUCTS))
        bad = prod in ("Sandals", "Backpack") and rng.random() < 0.35
        r = rng.random()
        if bad or r < 0.17:
            txt, rating = str(rng.choice(neg)), int(rng.choice([1, 2, 2]))
        elif r < 0.27:
            txt, rating = str(rng.choice(neu)), 3
        else:
            txt, rating = str(rng.choice(pos)), int(rng.choice([4, 5, 5]))
        rows.append(dict(date=(AS_OF - pd.Timedelta(days=int(rng.integers(0, 180)))).strftime("%Y-%m-%d"),
                         product=prod, rating=rating, review_text=txt))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def portfolio() -> pd.DataFrame:
    rows = [("Liquid Fund A", "Liquid", 1200, 1000, 1042), ("Bank FD 1Y", "Fixed Deposit", 1, 500000, 531000),
            ("Short Duration Debt Fund", "Debt", 8000, 52, 55.1), ("Nifty 50 Index Fund", "Equity", 2500, 150, 171.4),
            ("Gold ETF", "Gold", 3000, 58, 66.2)]
    return pd.DataFrame(rows, columns=["instrument", "asset_class", "units", "buy_price", "current_price"])
