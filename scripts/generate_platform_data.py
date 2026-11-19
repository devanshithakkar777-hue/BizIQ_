"""SYNTHETIC data for the platform modules (profit, returns, expenses, campaigns, pricing, credit, GST,
pipeline, quotes, contracts, support tickets). Everything is made up, for the same fictional apparel retailer
as the other sample data, and deliberately contains problems for the modules to find (a loss-making product,
duplicate payments, GST mismatches, stalled deals, ...). Run:  python scripts/generate_platform_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_data"
rng = np.random.default_rng(11)
AS_OF = pd.Timestamp("2026-10-31")

sales = pd.read_csv(OUT / "sales.csv", parse_dates=["date"])
invoices = pd.read_csv(OUT / "invoices.csv", parse_dates=["invoice_date", "due_date", "paid_date"])


def d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}: {len(df)} rows")


# ----------------------------------------------------------------------------- product costs
COST = {"Classic Tee": 260, "Denim Jacket": 1350, "Running Shoes": 1750, "Sandals": 470, "Backpack": 820,
        "Sunglasses": 640, "Hoodie": 950, "Wrist Watch": 2300, "Sneakers": 2250, "Cap": 170}
FEE = {"Classic Tee": 8, "Denim Jacket": 8, "Running Shoes": 9, "Sandals": 18, "Backpack": 8, "Sunglasses": 10,
       "Hoodie": 8, "Wrist Watch": 7, "Sneakers": 12, "Cap": 15}
save(pd.DataFrame([dict(product=p, unit_cost=c, channel_fee_pct=FEE[p], shipping_cost_per_order=75,
                        packaging_cost=15) for p, c in COST.items()]), "product_costs")

# ----------------------------------------------------------------------------- returns
RET_RATE = {"Sneakers": 0.17, "Sandals": 0.15, "Running Shoes": 0.12, "Denim Jacket": 0.10, "Hoodie": 0.08,
            "Classic Tee": 0.05, "Cap": 0.04, "Backpack": 0.06, "Sunglasses": 0.05, "Wrist Watch": 0.04}
REASONS = {"Sneakers": ["Size or fit", "Size or fit", "Quality issue", "Not as described"],
           "Sandals": ["Size or fit", "Quality issue", "Not as described"],
           "default": ["Size or fit", "Changed mind", "Quality issue", "Damaged in transit", "Not as described"]}
CHANNELS = ["Website", "Amazon", "Flipkart", "Meesho", "WhatsApp"]
CH_P = [0.30, 0.25, 0.20, 0.15, 0.10]
rows = []
for i, r in enumerate(sales[sales["date"] <= AS_OF - pd.Timedelta(days=5)].itertuples(index=False)):
    ch = str(rng.choice(CHANNELS, p=CH_P))
    cod = int(rng.random() < (0.45 if ch in ("Meesho", "WhatsApp") else 0.22))
    p = RET_RATE[r.product] * (1.5 if cod else 1.0) * (1.3 if ch == "Meesho" else 1.0)
    if rng.random() < p:
        reason = str(rng.choice(REASONS.get(r.product, REASONS["default"])))
        if cod and rng.random() < 0.5:
            reason = "Refused at delivery"
        when = min(r.date + pd.Timedelta(days=int(rng.integers(3, 16))), AS_OF)
        rows.append(dict(return_id=f"R{len(rows) + 1:05d}", order_id=r.order_id, date=d(when), product=r.product,
                         refund_amount=float(r.revenue), reason=reason, channel=ch, is_cod=cod, quantity=int(r.quantity)))
save(pd.DataFrame(rows), "returns")

# ----------------------------------------------------------------------------- expenses
CATS = {"Marketing": ["Meta Platforms", "Google India", "Creator Collective"],
        "Software": ["Zoho Corp", "Cloud Hosting Co", "Shopify India"],
        "Utilities": ["City Power Board", "Fibernet Broadband"],
        "Logistics": ["SwiftShip Couriers", "BlueRoute Cargo"],
        "Office": ["OfficeMart", "CleanCo Services"],
        "Professional fees": ["Sharma & Associates CA", "LexPoint Legal"],
        "Travel": ["TravelDesk", "CityCabs"],
        "Repairs": ["FixIt Services"]}
BASE = {"Marketing": 55000, "Software": 9000, "Utilities": 18000, "Logistics": 32000, "Office": 7000,
        "Professional fees": 25000, "Travel": 6500, "Repairs": 8000}
rows = []
for m in pd.period_range("2025-11", "2026-10", freq="M"):
    for cat, vendors in CATS.items():
        for _ in range(int(rng.integers(2, 6))):
            v = str(rng.choice(vendors))
            amt = float(round(BASE[cat] * rng.lognormal(0, 0.35) / 100) * 100)
            day = int(rng.integers(1, m.days_in_month + 1))
            mode = str(rng.choice(["UPI", "Bank transfer", "Card", "Cash"], p=[0.30, 0.45, 0.15, 0.10]))
            rows.append(dict(date=d(m.to_timestamp() + pd.Timedelta(days=day - 1)), category=cat, vendor=v,
                             amount=amt, department=str(rng.choice(["Operations", "Marketing", "Admin"])),
                             payment_mode=mode, approved_by="Founder" if amt > 20000 else str(rng.choice(["Founder", "Manager"])),
                             receipt_attached=int(rng.random() > (0.35 if mode == "Cash" else 0.04))))
ex = pd.DataFrame(rows)
# planted problems: duplicate payments, one very large outlier, unapproved large spends
dups = ex.sample(6, random_state=3).copy()
dups["date"] = (pd.to_datetime(dups["date"]) + pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d")
ex = pd.concat([ex, dups], ignore_index=True)
ex.loc[ex.sample(1, random_state=8).index[0], ["category", "vendor", "amount"]] = ["Marketing", "Creator Collective", 410000.0]
big = ex.sample(5, random_state=5).index
ex.loc[big, "amount"] = ex.loc[big, "amount"].clip(lower=60000)
ex.loc[big, "approved_by"] = np.nan
ex = ex.sort_values("date").reset_index(drop=True)
ex.insert(0, "expense_id", [f"E{i + 1:05d}" for i in range(len(ex))])
save(ex, "expenses")

# ----------------------------------------------------------------------------- campaigns
ROAS = {"Google Ads": 3.2, "Meta Ads": 1.7, "Influencer": 0.8, "WhatsApp Broadcast": 6.0, "Email": 8.5}
SPEND = {"Google Ads": 2600, "Meta Ads": 3400, "Influencer": 2200, "WhatsApp Broadcast": 450, "Email": 150}
CPC = {"Google Ads": 14, "Meta Ads": 9, "Influencer": 22, "WhatsApp Broadcast": 2.5, "Email": 1.2}
CVR = {"Google Ads": 0.021, "Meta Ads": 0.014, "Influencer": 0.008, "WhatsApp Broadcast": 0.05, "Email": 0.04}
rows = []
for day in pd.date_range(AS_OF - pd.Timedelta(days=119), AS_OF):
    for ch in ROAS:
        spend = float(round(SPEND[ch] * rng.lognormal(0, 0.25), 0))
        clicks = int(spend / CPC[ch] * rng.normal(1, 0.1))
        conv = int(rng.binomial(max(clicks, 1), CVR[ch]))
        rows.append(dict(date=d(day), channel=ch, campaign=f"{ch} always-on", spend=spend, impressions=int(clicks * rng.uniform(25, 60)),
                         clicks=clicks, conversions=conv, revenue=float(round(spend * ROAS[ch] * rng.lognormal(0, 0.2), 0))))
save(pd.DataFrame(rows), "campaigns")

# ----------------------------------------------------------------------------- competitor prices
OUR = sales.groupby("product")["unit_price"].mean().round(0)
BIAS = {"Sneakers": 1.0, "Cap": 1.0, "Classic Tee": 1.18, "Sunglasses": 0.88, "Wrist Watch": 0.92}
rows = []
for wk in pd.date_range(AS_OF - pd.Timedelta(weeks=11), AS_OF, freq="7D"):
    for p, ours in OUR.items():
        for comp, off in (("StyleHub", 0.95), ("UrbanCart", 1.02), ("TrendBazaar", 0.90)):
            # BIAS < 1 means competitors are cheaper than us; > 1 means they are dearer
            rows.append(dict(date=d(wk), product=p, competitor=comp,
                             price=float(round(ours * BIAS.get(p, 1.0) * off * rng.normal(1, 0.015)))))
save(pd.DataFrame(rows), "competitor_prices")

# ----------------------------------------------------------------------------- credit limits
rec = invoices[invoices["type"] == "receivable"].copy()
open_amt = rec[rec["paid_date"].isna() | (rec["paid_date"] > AS_OF)].groupby("party")["amount"].sum()
limits = {}
for party in rec["party"].unique():
    o = float(open_amt.get(party, 0.0))
    limits[party] = float(max(150000, round(o * (0.8 if party in ("Urban Mart", "Kapoor & Sons") else 1.6), -4)))
save(pd.DataFrame([dict(customer=k, credit_limit=v, payment_terms_days=int(rng.choice([30, 45]))) for k, v in limits.items()]),
     "credit_limits")

# ----------------------------------------------------------------------------- GST books vs GSTR-2B
STATES = ["27", "24", "29", "07"]
SUPPLIERS = {}
for i, name in enumerate(["Aarav Textiles", "Bharat Garments", "Coastal Footwear", "Delta Accessories",
                          "Evergreen Supply Co", "SwiftShip Couriers", "Zoho Corp", "OfficeMart"]):
    SUPPLIERS[name] = f"{STATES[i % 4]}{''.join(rng.choice(list('ABCDEFGHJKLMNPRSTUVWXY'), 5))}{int(rng.integers(1000, 9999))}{rng.choice(list('ABCDEFGH'))}1Z{rng.choice(list('ABCDEFGH'))}"
rows = []
for i in range(120):
    sup = str(rng.choice(list(SUPPLIERS)))
    taxable = float(round(rng.lognormal(10.5, 0.7), 0))
    rate = 0.18 if rng.random() < 0.8 else 0.12
    dt = AS_OF - pd.Timedelta(days=int(rng.integers(1, 180)))
    rows.append(dict(invoice_no=f"{sup[:3].upper()}/{dt:%y%m}/{i + 1:03d}", party_gstin=SUPPLIERS[sup],
                     invoice_date=d(dt), taxable_value=taxable, gst_amount=round(taxable * rate, 2), party=sup))
books = pd.DataFrame(rows)
b2 = books.copy()
miss = b2.sample(6, random_state=1).index                      # supplier has not filed: in books, missing in 2B
amt = b2.drop(miss).sample(6, random_state=2).index           # amount differs
b2.loc[amt, "gst_amount"] = (b2.loc[amt, "gst_amount"] * rng.uniform(0.6, 0.9, len(amt))).round(2)
fmt = b2.drop(miss).sample(8, random_state=4).index           # same invoice written differently
b2.loc[fmt, "invoice_no"] = b2.loc[fmt, "invoice_no"].str.replace("/", "-")
b2 = b2.drop(miss)
extra = pd.DataFrame([dict(invoice_no=f"UNK/{2610 + k}/{k:03d}", party_gstin=SUPPLIERS["Delta Accessories"],
                           invoice_date=d(AS_OF - pd.Timedelta(days=20 + k)), taxable_value=float(30000 + 4000 * k),
                           gst_amount=float(round((30000 + 4000 * k) * 0.18, 2)), party="Delta Accessories") for k in range(4)])
b2 = pd.concat([b2, extra], ignore_index=True)
save(books.drop(columns=["party"]).assign(party=books["party"]), "gst_books")
save(b2, "gstr2b")

# ----------------------------------------------------------------------------- pipeline
STAGES = ["lead", "qualified", "proposal", "negotiation", "won", "lost"]
ACC = ["Metro Retail Pvt Ltd", "Sunrise Boutiques", "Urban Mart", "Kapoor & Sons", "Lakeview Stores",
       "Zenith Outfitters", "Nova Fashions", "Heritage Wear", "Pinnacle Outlets", "Bluebird Mall Kiosks",
       "Grand Bazaar", "Orchid Lifestyle", "Summit Sports Co", "Lotus Garments", "Crescent Retail"]
rows = []
for i in range(64):
    stage = str(rng.choice(STAGES, p=[0.14, 0.16, 0.18, 0.14, 0.2, 0.18]))
    created = AS_OF - pd.Timedelta(days=int(rng.integers(5, 200)))
    if stage in ("won", "lost"):
        last = created + pd.Timedelta(days=int(rng.integers(10, 90)))
        last = min(last, AS_OF)
    else:
        last = AS_OF - pd.Timedelta(days=int(rng.choice([2, 5, 9, 14, 22, 35, 48, 70], p=[.2, .2, .15, .15, .1, .1, .06, .04])))
        last = max(last, created)
    rows.append(dict(deal_id=f"DL{i + 1:03d}", account=str(rng.choice(ACC)), stage=stage,
                     amount=float(round(rng.lognormal(12.4, 0.6), -3)), created_date=d(created), last_activity_date=d(last),
                     expected_close=d(AS_OF + pd.Timedelta(days=int(rng.integers(-20, 90)))),
                     owner=str(rng.choice(["Priya", "Rohan", "Meera"])),
                     source=str(rng.choice(["Referral", "Website", "Trade show", "Outbound"], p=[.35, .25, .15, .25]))))
save(pd.DataFrame(rows), "pipeline")

# ----------------------------------------------------------------------------- quotes
rows = []
for i in range(90):
    sent = AS_OF - pd.Timedelta(days=int(rng.integers(3, 300)))
    amount = float(round(rng.lognormal(12.0, 0.7), -3))
    status = str(rng.choice(["won", "lost", "open"], p=[0.32, 0.5, 0.18])) if (AS_OF - sent).days > 30 else \
        str(rng.choice(["won", "lost", "open"], p=[0.15, 0.2, 0.65]))
    margin = float(rng.uniform(0.12, 0.35))
    rows.append(dict(quote_id=f"Q{i + 1:04d}", customer=str(rng.choice(ACC)), sent_date=d(sent), amount=amount, status=status,
                     cost=float(round(amount * (1 - margin), -2)),
                     lost_reason=str(rng.choice(["Price", "Price", "Delivery time", "Chose competitor", "No response"])) if status == "lost" else "",
                     decided_date=d(min(sent + pd.Timedelta(days=int(rng.integers(5, 40))), AS_OF)) if status != "open" else ""))
save(pd.DataFrame(rows), "quotes")

# ----------------------------------------------------------------------------- contracts
rows = []
spec = [("Metro Retail Pvt Ltd", "customer", 18, 60, 1, 5.0, 2400000), ("Sunrise Boutiques", "customer", 42, 30, 0, 0.0, 900000),
        ("Urban Mart", "customer", 70, 60, 1, 3.0, 1500000), ("Kapoor & Sons", "customer", 150, 30, 0, 0.0, 700000),
        ("Aarav Textiles", "supplier", 9, 30, 1, 6.0, 3200000), ("Coastal Footwear", "supplier", 55, 60, 0, 4.0, 2100000),
        ("Evergreen Supply Co", "supplier", 220, 90, 1, 5.0, 2800000), ("Prime Properties", "lease", 35, 90, 0, 8.0, 1800000),
        ("Fibernet Broadband", "service", 28, 30, 1, 0.0, 120000), ("Zoho Corp", "service", 300, 30, 1, 0.0, 180000),
        ("Delta Accessories", "supplier", -12, 30, 1, 3.0, 950000), ("SwiftShip Couriers", "service", 85, 45, 0, 7.0, 1100000)]
for i, (party, typ, days_left, notice, auto, esc, val) in enumerate(spec):
    end = AS_OF + pd.Timedelta(days=days_left)
    rows.append(dict(contract_id=f"CT{i + 1:03d}", party=party, end_date=d(end), value=float(val), start_date=d(end - pd.Timedelta(days=365)),
                     renewal_notice_days=notice, auto_renew=auto, escalation_pct=esc, type=typ))
save(pd.DataFrame(rows), "contracts")

# ----------------------------------------------------------------------------- support tickets
CAT = {"Delivery delay": 0.28, "Product quality": 0.16, "Billing or payment": 0.12, "Refund status": 0.17,
       "Size or fit": 0.17, "Order change": 0.10}
SLA_H = {"urgent": 4, "high": 12, "medium": 36, "low": 72}
rows = []
for i in range(420):
    opened = AS_OF - pd.Timedelta(hours=int(rng.integers(6, 24 * 180)))
    cat = str(rng.choice(list(CAT), p=list(CAT.values())))
    pr = str(rng.choice(["urgent", "high", "medium", "low"], p=[0.07, 0.2, 0.5, 0.23]))
    hours = float(rng.lognormal(np.log(SLA_H[pr] * (1.4 if cat in ("Delivery delay", "Refund status") else 0.9)), 0.7))
    closed = opened + pd.Timedelta(hours=hours)
    is_open = closed > AS_OF or rng.random() < 0.04
    csat = np.nan if is_open or rng.random() > 0.6 else float(np.clip(round(5 - hours / (SLA_H[pr] * 1.5) + rng.normal(0, 0.8)), 1, 5))
    rows.append(dict(ticket_id=f"T{i + 1:05d}", opened_date=opened.strftime("%Y-%m-%d %H:%M"), category=cat, priority=pr,
                     closed_date="" if is_open else closed.strftime("%Y-%m-%d %H:%M"), csat=csat,
                     channel=str(rng.choice(["WhatsApp", "Email", "Phone", "Website"], p=[0.45, 0.25, 0.15, 0.15]))))
save(pd.DataFrame(rows), "tickets")
