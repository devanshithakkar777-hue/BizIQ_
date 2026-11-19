"""SYNTHETIC data for the channel, operations and people modules. Everything is made up and deliberately contains problems for
the modules to find (short-paid marketplace orders, a distributor being stuffed with stock, a failing machine, a slow carrier...).
Run:  python scripts/generate_industry_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "sample_data"
rng = np.random.default_rng(23)
AS_OF = pd.Timestamp("2026-10-31")


def d(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}: {len(df)} rows")


# ---------------------------------------------------------------- marketplace orders
rows = []
COMM = {"Amazon": 0.12, "Flipkart": 0.10, "Meesho": 0.06, "Myntra": 0.20}
for i in range(2600):
    ch = str(rng.choice(list(COMM), p=[0.38, 0.28, 0.22, 0.12]))
    day = AS_OF - pd.Timedelta(days=int(rng.integers(0, 150)))
    gross = float(round(rng.choice([499, 799, 1199, 1599, 2299, 3499]) * rng.uniform(0.9, 1.05)))
    comm = round(gross * COMM[ch], 2)
    ship = float(rng.choice([45, 60, 75]))
    other = round(gross * 0.02, 2)
    expected = gross - comm - ship - other
    ret = int(rng.random() < (0.22 if ch == "Myntra" else 0.09))
    settle, sdate = expected, day + pd.Timedelta(days=int(rng.integers(7, 12)))
    if ch == "Meesho" and rng.random() < 0.12:
        settle = expected - gross * 0.05                      # planted: short payment
    if ret:
        settle = 0.0
    if (AS_OF - day).days > 20 and not ret and rng.random() < 0.04:
        settle, sdate = None, None                            # planted: never settled
    if (AS_OF - day).days <= 10:
        settle, sdate = None, None                            # too recent to be settled: must NOT be flagged
    rows.append(dict(order_id=f"M{i + 1:05d}", date=d(day), channel=ch, product=str(rng.choice(["Classic Tee", "Hoodie", "Sneakers", "Cap", "Backpack"])),
                     gross_amount=gross, commission=comm, shipping_fee=ship, other_fees=other,
                     settlement_amount=None if settle is None else round(settle, 2), settled_date=None if sdate is None else d(sdate), returned=ret))
save(pd.DataFrame(rows), "marketplace_orders")

# ---------------------------------------------------------------- WhatsApp campaigns
rows = []
camps = {"Welcome offer": (0.96, 0.82, 0.07, 0.004), "New arrivals": (0.95, 0.74, 0.05, 0.006), "Flash sale": (0.94, 0.70, 0.09, 0.032),
         "Cart reminder": (0.97, 0.88, 0.14, 0.003), "Festival greetings": (0.93, 0.66, 0.03, 0.009)}
for name, (dr, rr, rp, oo) in camps.items():
    for k in range(int(rng.integers(5, 9))):
        day = AS_OF - pd.Timedelta(days=int(rng.integers(1, 85)))
        sent = int(rng.integers(900, 3000)); dl = int(sent * dr); rd = int(dl * rr)
        rows.append(dict(date=d(day), campaign=name, sent=sent, delivered=dl, read=rd, replied=int(dl * rp), opted_out=int(dl * oo * rng.uniform(.8, 1.2))))
save(pd.DataFrame(rows), "whatsapp_messages")

# ---------------------------------------------------------------- stores
rows = []
STORES = {"Pune Camp": (1.0, 4200, 1800), "Mumbai Andheri": (1.25, 5200, 2400), "Thane": (1.0, 3600, 1500), "Nashik": (0.52, 3400, 1500),
          "Nagpur": (0.9, 3000, 1400), "Bengaluru Indiranagar": (1.15, 4800, 2000)}
for st_, (mult, area, base) in STORES.items():
    for k in range(120):
        day = AS_OF - pd.Timedelta(days=k)
        f = mult * (1.2 if day.weekday() >= 5 else 1.0)
        if st_ == "Thane" and k < 28:
            f *= 0.72                                          # planted: recent fall
        foot = int(rng.normal(base * mult * 0.12 * (1.3 if day.weekday() >= 5 else 1), 20))
        conv = 0.14 if st_ != "Nagpur" else 0.07               # planted: poor conversion
        tx = max(1, int(foot * conv * rng.uniform(.9, 1.1)))
        rows.append(dict(store=st_, date=d(day), sales=round(tx * 1150 * f * rng.uniform(.9, 1.1) / mult), transactions=tx, footfall=foot,
                         staff_hours=float(rng.choice([36, 40, 48])), area_sqft=area))
save(pd.DataFrame(rows), "store_daily")

# ---------------------------------------------------------------- distributors
rows = []
for dist, ratio, out, cd in [("Sharma Traders", 1.02, 18, 30), ("Deccan Distributors", 1.08, 25, 30), ("Konkan Agencies", 1.85, 40, 30),
                             ("Vidarbha Supplies", 0.98, 20, 45), ("Marathwada Co", 1.10, 95, 30), ("Khandesh Sales", 1.05, 22, 30)]:
    stock = 300000
    for m in range(6, 0, -1):
        mon = (AS_OF - pd.DateOffset(months=m - 1)).replace(day=1)
        sec = float(rng.integers(380000, 520000))
        pri = sec * (ratio if m <= 3 else 1.0) * rng.uniform(.97, 1.03)
        stock = max(50000, stock + pri - sec)
        rows.append(dict(distributor=dist, month=d(mon), primary_sales=round(pri), secondary_sales=round(sec), closing_stock=round(stock),
                         outstanding=round(pri * out / 30), credit_days=cd))
save(pd.DataFrame(rows), "distributors")

# ---------------------------------------------------------------- shipments
rows = []
CAR = {"FastTrack": (0.93, 0.04, 62), "BudgetShip": (0.71, 0.18, 44), "MetroCourier": (0.86, 0.09, 55)}
for i in range(2400):
    car = str(rng.choice(list(CAR), p=[.4, .35, .25]))
    ontime, rto, cost = CAR[car]
    ship = AS_OF - pd.Timedelta(days=int(rng.integers(0, 120)))
    zone = str(rng.choice(["Metro", "Tier-2", "Remote"], p=[.5, .35, .15]))
    promised = ship + pd.Timedelta(days=int({"Metro": 3, "Tier-2": 5, "Remote": 8}[zone]))
    cod = float(rng.choice([0, 0, 899, 1499, 2199]))
    status = "delivered"
    deliv = promised + pd.Timedelta(days=0 if rng.random() < ontime else int(rng.integers(1, 6))) - pd.Timedelta(days=int(rng.integers(0, 2)))
    r = rng.random()
    if r < rto * (1.6 if zone == "Remote" else 1):
        status, deliv = "rto", None
    elif r > 0.985:
        status, deliv = "lost", None
    if deliv is not None and deliv > AS_OF:
        status, deliv = "in_transit", None
    rows.append(dict(shipment_id=f"S{i + 1:05d}", carrier=car, ship_date=d(ship), promised_date=d(promised), delivered_date=None if deliv is None else d(deliv),
                     status=status, cost=round(cost * rng.uniform(.9, 1.2), 1), zone=zone, cod_amount=cod))
save(pd.DataFrame(rows), "shipments")

# ---------------------------------------------------------------- production
rows = []
for line, (eff, down, defect) in {"Line A": (0.95, 25, 0.015), "Line B": (0.92, 35, 0.02), "Line C": (0.81, 95, 0.065)}.items():
    for k in range(90):
        day = AS_OF - pd.Timedelta(days=k)
        if day.weekday() == 6:
            continue
        plan = 1000
        prod = int(plan * eff * rng.uniform(.95, 1.04))
        rows.append(dict(line=line, date=d(day), planned_units=plan, produced_units=prod, defective_units=int(prod * defect * rng.uniform(.6, 1.4)),
                         downtime_min=int(down * rng.uniform(.5, 1.6)), planned_minutes=480))
save(pd.DataFrame(rows), "production")

# ---------------------------------------------------------------- maintenance
rows = []
for asset, nb, last_prev in [("Press-1", 1, 30), ("Press-2", 7, 260), ("Cutter-1", 2, 60), ("Packer-1", 3, 45), ("Boiler", 1, 90)]:
    for k in range(nb):
        rows.append(dict(asset=asset, date=d(AS_OF - pd.Timedelta(days=int(rng.integers(5, 300)))), event_type="breakdown",
                         downtime_hours=float(round(rng.uniform(2, 14 if asset == "Press-2" else 6), 1)), cost=float(round(rng.uniform(3000, 25000))), cause=str(rng.choice(["bearing", "belt", "electrical", "hydraulic"]))))
    rows.append(dict(asset=asset, date=d(AS_OF - pd.Timedelta(days=last_prev)), event_type="preventive", downtime_hours=1.5, cost=2500.0, cause="scheduled service"))
save(pd.DataFrame(rows), "maintenance")

# ---------------------------------------------------------------- tenders
rows = []
for i in range(28):
    st_ = ["won"] * 6 + ["lost"] * 11 + ["no_bid"] * 3 + ["submitted"] * 3 + ["open"] * 5
    s = st_[i]
    value = float(rng.integers(8, 90) * 100000)
    if s == "open":
        dl = AS_OF + pd.Timedelta(days=[3, 6, 12, 25, 40][i - 23])
    elif s == "submitted":
        dl = AS_OF - pd.Timedelta(days=int(rng.integers(3, 20)))
    else:
        dl = AS_OF - pd.Timedelta(days=int(rng.integers(20, 300)))
    bid = value * rng.uniform(.9, 1.02) if s in ("won", "lost", "submitted") else None
    win = (bid if s == "won" else bid * rng.uniform(.84, .95)) if s in ("won", "lost") else None
    rows.append(dict(tender_id=f"T{i + 1:03d}", title=f"{rng.choice(['Uniform supply', 'Packaging', 'Garments', 'Footwear', 'Merchandise'])} {2026 - (i % 2)}-{i + 1}", deadline=d(dl),
                     value=value, status=s, our_bid=None if bid is None else round(bid), winning_bid=None if win is None else round(win),
                     emd_amount=round(value * 0.02) if s in ("submitted", "open") else None))
save(pd.DataFrame(rows), "tenders")

# ---------------------------------------------------------------- ESG
rows = []
for m in range(24, 0, -1):
    mon = (AS_OF - pd.DateOffset(months=m - 1)).replace(day=1)
    drift = 1.0 + (0.13 if m <= 6 else 0.0)                    # planted: electricity up 13% in the last six months
    rev = float(rng.integers(1500, 2100) * 1000 * (1 + (24 - m) * 0.01))
    rows.append(dict(month=d(mon), electricity_kwh=round(rng.uniform(41000, 45000) * drift), fuel_litres=round(rng.uniform(2200, 2800)),
                     water_kl=round(rng.uniform(900, 1100)), waste_kg=round(rng.uniform(3800, 4400)), recycled_kg=round(rng.uniform(1100, 1500)), revenue=rev))
save(pd.DataFrame(rows), "esg")

# ---------------------------------------------------------------- candidates
rows = []
SRC = {"Job portal": (70, [.55, .25, .10, .04]), "Referral": (25, [.20, .40, .50, .70]), "Campus": (30, [.35, .30, .25, .45]), "LinkedIn": (35, [.45, .30, .20, .30])}
n = 0
for src, (cnt, _) in SRC.items():
    conv = {"Job portal": 0.03, "Referral": 0.30, "Campus": 0.12, "LinkedIn": 0.10}[src]
    for _ in range(cnt):
        n += 1
        applied = AS_OF - pd.Timedelta(days=int(rng.integers(5, 150)))
        r = rng.random()
        age = (AS_OF - applied).days
        if r < conv:
            stage, hired = "hired", applied + pd.Timedelta(days=int(rng.integers(21, 50)))
            if hired > AS_OF:
                stage, hired = "offer", None
        elif r < conv + 0.10:
            stage, hired = "offer", None
        elif r < conv + 0.25:
            stage, hired = "interview", None
        elif r < conv + 0.45:
            stage, hired = "screen", None
        elif r < conv + 0.50:
            stage, hired = "withdrawn", None
        else:
            stage, hired = "rejected", None
        last = AS_OF - pd.Timedelta(days=int(rng.integers(0, min(age, 12) + 1)))
        if stage in ("screen", "interview") and n % 4 == 0:
            last = AS_OF - pd.Timedelta(days=int(rng.integers(20, 45)))      # planted: stalled candidates
        rows.append(dict(candidate_id=f"C{n:04d}", role=str(rng.choice(["Store Manager", "Sales Associate", "Designer", "Accountant", "Supply Planner"])),
                         applied_date=d(applied), stage=stage, source=src, department=str(rng.choice(["Sales", "Operations", "Finance", "Design"])),
                         hired_date=None if hired is None else d(hired), last_stage_date=d(last), offer_accepted=1 if stage == "hired" else None))
save(pd.DataFrame(rows), "candidates")
