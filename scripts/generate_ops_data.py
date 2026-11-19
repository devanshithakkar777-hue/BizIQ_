"""SYNTHETIC data for BudgetIQ, ReconIQ, ProjectIQ, SubscriptionIQ, AppointmentIQ, RecipeIQ, AssetIQ, SurveyIQ and DiscountIQ.
Everything is made up and contains problems on purpose (a marketing overspend, unmatched bank lines, an overrunning project, a churny plan,
Monday no-shows, a popular dish that loses money, old assets still in use, a falling NPS, discounts that do not lift volume).
Run:  python scripts/generate_ops_data.py      (existing sales.csv gets a discount_pct column; nothing else is changed)
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "sample_data"
rng = np.random.default_rng(41)
AS_OF = pd.Timestamp("2026-10-31")


def d(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}: {len(df)} rows")


# ---------------------------------------------------------------- budgets (vs the expenses file)
exp = pd.read_csv(OUT / "expenses.csv", parse_dates=["date"])
exp["m"] = exp["date"].dt.to_period("M").dt.to_timestamp()
monthly = exp.groupby(["m", "category"])["amount"].sum().unstack(fill_value=0)
rows = []
for m in pd.date_range("2025-11-01", "2026-10-01", freq="MS"):
    for cat in monthly.columns:
        base = float(monthly[cat].mean())
        b = base * rng.uniform(0.95, 1.1)
        if cat == "Marketing" and m >= pd.Timestamp("2026-08-01"):
            b = base * 0.62                                              # planted: budget far below what is actually spent
        if cat == "Software":
            b = base * 1.45                                              # planted: persistent underspend
        rows.append(dict(month=d(m), category=cat, budget=round(b, -2)))
save(pd.DataFrame(rows), "budgets")

# ---------------------------------------------------------------- bank statement vs ledger
n = 240
dates = [AS_OF - pd.Timedelta(days=int(x)) for x in rng.integers(1, 90, n)]
amt = np.round(rng.choice([-1, 1], n, p=[0.55, 0.45]) * rng.uniform(1500, 160000, n), 2)
desc = [f"{'Payment to' if a < 0 else 'Receipt from'} {rng.choice(['Metro Retail', 'Konkan Foods', 'City Power', 'Meta', 'Fibernet', 'Apex Traders', 'Delta Logistics'])}" for a in amt]
led = pd.DataFrame(dict(date=[d(x) for x in dates], description=desc, amount=amt, reference=[f"TXN{i:04d}" for i in range(n)]))
bank = led.copy()
bank["date"] = [d(pd.Timestamp(x) + pd.Timedelta(days=int(rng.integers(0, 4)))) for x in bank["date"]]      # banks post a few days later
uncleared = rng.choice(n, 6, replace=False)                                                              # planted: in ledger, not yet in bank
bank = bank.drop(index=uncleared)
mism = bank.sample(3, random_state=5).index
bank.loc[mism, "amount"] = (bank.loc[mism, "amount"] * 0.99).round(2)                                    # planted: bank amount differs from ledger
extra = pd.DataFrame(dict(date=[d(AS_OF - pd.Timedelta(days=int(x))) for x in (12, 40, 71, 33, 8)], description=["Bank charges", "Bank charges", "Interest debit", "Bank charges", "SMS charges"],
                          amount=[-590.0, -590.0, -4210.0, -590.0, -118.0], reference=[f"BNK{i}" for i in range(5)]))   # planted: bank items never booked
dup = bank.sample(2, random_state=9)                                                                     # planted: same line twice in the bank file
bank = pd.concat([bank, extra, dup], ignore_index=True).sort_values("date")
save(bank, "bank_statement")
save(led.sort_values("date"), "ledger_entries")

# ---------------------------------------------------------------- projects
clients = ["Apex Traders", "Metro Retail", "Delta Logistics", "Konkan Foods", "Sunrise Clinic", "Orbit Media"]
rows = []
for i in range(14):
    start = AS_OF - pd.Timedelta(days=int(rng.integers(30, 400)))
    dur = int(rng.integers(45, 200))
    due = start + pd.Timedelta(days=dur)
    budget = float(rng.choice([180000, 260000, 400000, 650000, 900000]))
    hb = budget / 1500
    done = due < AS_OF - pd.Timedelta(days=10) and rng.random() < 0.8
    pc = 100.0 if done else float(rng.integers(15, 90))
    spent = hb * (pc / 100) * rng.uniform(0.85, 1.1)
    cost = budget * 0.55 * (pc / 100) * rng.uniform(0.9, 1.1)
    billed = budget * (pc / 100) * rng.uniform(0.8, 1.0)
    rows.append(dict(project_id=f"PRJ{i + 1:03d}", client=str(rng.choice(clients)), project=f"Project {i + 1}", start_date=d(start), due_date=d(due),
                     completed_date=d(due + pd.Timedelta(days=int(rng.integers(-5, 20)))) if done else "", status="completed" if done else "active",
                     budget=budget, billed=round(billed), cost_to_date=round(cost), hours_budget=round(hb), hours_spent=round(spent), percent_complete=pc))
p = pd.DataFrame(rows)
for pid in ("PRJ002", "PRJ007"):                                                                        # planted: burning hours far faster than progress
    i = p.index[p.project_id == pid][0]
    p.loc[i, ["status", "completed_date", "percent_complete"]] = ["active", "", 35.0]
    p.loc[i, "hours_spent"] = round(p.loc[i, "hours_budget"] * 0.78)
    p.loc[i, "cost_to_date"] = round(p.loc[i, "budget"] * 0.62)
i = p.index[p.project_id == "PRJ011"][0]                                                                 # planted: overdue, mostly unbilled
p.loc[i, ["status", "completed_date", "percent_complete", "due_date"]] = ["active", "", 80.0, d(AS_OF - pd.Timedelta(days=46))]
p.loc[i, "billed"] = round(p.loc[i, "budget"] * 0.15)
save(p, "projects")

# ---------------------------------------------------------------- subscriptions
plans = {"Starter": 999, "Pro": 2999, "Business": 7999}
rows = []
for i in range(210):
    plan = str(rng.choice(list(plans), p=[0.5, 0.35, 0.15]))
    start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 640)))
    life = int(rng.exponential(520 if plan != "Starter" else 300))
    end = start + pd.Timedelta(days=life)
    if plan == "Starter" and rng.random() < 0.07:
        end = AS_OF - pd.Timedelta(days=int(rng.integers(5, 60)))                                       # planted: a Starter churn wave
        end = max(end, start + pd.Timedelta(days=30))
    rows.append(dict(subscription_id=f"SUB{i + 1:04d}", customer=f"Cust {i + 1:03d}", plan=plan, start_date=d(start), end_date=d(end) if end <= AS_OF else "", mrr=plans[plan]))
save(pd.DataFrame(rows), "subscriptions")

# ---------------------------------------------------------------- appointments
staff = ["Rohit", "Meera", "Anil"]
services = {"Consultation": (800, 30), "Treatment": (2200, 60), "Follow-up": (500, 20)}
rows = []
cust = [f"Client {i:03d}" for i in range(220)]
for i in range(1800):
    day = AS_OF - pd.Timedelta(days=int(rng.integers(0, 150)))
    if day.weekday() == 6:
        day -= pd.Timedelta(days=1)
    svc = str(rng.choice(list(services), p=[0.4, 0.3, 0.3]))
    noshow_p = 0.26 if day.weekday() == 0 else 0.08                                                      # planted: Monday no-shows
    st = "no_show" if rng.random() < noshow_p else ("cancelled" if rng.random() < 0.07 else "completed")
    rows.append(dict(appointment_id=f"APT{i + 1:05d}", date=d(day), start_time=f"{int(rng.integers(9, 19)):02d}:00", customer=str(rng.choice(cust)), service=svc,
                     staff=str(rng.choice(staff, p=[0.25, 0.45, 0.30])), status=st, price=services[svc][0], duration_min=services[svc][1]))
save(pd.DataFrame(rows), "appointments")

# ---------------------------------------------------------------- recipes
menu = {  # dish: (price, portions sold last month, [(ingredient, qty, unit, unit cost)])
    "Butter Chicken": (420, 640, [("Chicken", 0.25, "kg", 330), ("Butter", 0.03, "kg", 520), ("Cream", 0.05, "l", 280), ("Tomato", 0.12, "kg", 40), ("Spices", 1, "pack", 9)]),
    "Paneer Tikka": (320, 410, [("Paneer", 0.18, "kg", 380), ("Curd", 0.05, "kg", 70), ("Capsicum", 0.06, "kg", 60), ("Spices", 1, "pack", 8)]),
    "Dal Makhani": (260, 520, [("Black dal", 0.08, "kg", 150), ("Butter", 0.02, "kg", 520), ("Cream", 0.03, "l", 280), ("Spices", 1, "pack", 6)]),
    "Veg Biryani": (290, 380, [("Basmati rice", 0.12, "kg", 130), ("Mixed veg", 0.15, "kg", 55), ("Spices", 1, "pack", 10), ("Ghee", 0.015, "kg", 640)]),
    "Chicken Biryani": (250, 590, [("Basmati rice", 0.12, "kg", 130), ("Chicken", 0.30, "kg", 330), ("Spices", 1, "pack", 12), ("Ghee", 0.02, "kg", 640)]),
    "Lamb Rogan Josh": (520, 70, [("Lamb", 0.25, "kg", 820), ("Curd", 0.06, "kg", 70), ("Spices", 1, "pack", 14)]),
    "Prawn Masala": (680, 55, [("Prawns", 0.22, "kg", 640), ("Coconut milk", 0.08, "l", 190), ("Spices", 1, "pack", 12), ("Tomato", 0.1, "kg", 40)]),
    "Garlic Naan": (70, 1500, [("Flour", 0.1, "kg", 38), ("Garlic", 0.01, "kg", 160), ("Butter", 0.01, "kg", 520)]),
    "Masala Chai": (60, 1300, [("Milk", 0.15, "l", 62), ("Tea leaves", 0.005, "kg", 480), ("Sugar", 0.01, "kg", 45)]),
    "Gulab Jamun": (110, 330, [("Khoya", 0.06, "kg", 420), ("Sugar", 0.05, "kg", 45), ("Oil", 0.03, "l", 140)]),
    "Dessert Platter": (240, 60, [("Khoya", 0.12, "kg", 420), ("Dry fruits", 0.04, "kg", 900), ("Sugar", 0.06, "kg", 45)]),
    "Mushroom Soup": (180, 210, [("Mushroom", 0.1, "kg", 190), ("Cream", 0.04, "l", 280), ("Butter", 0.01, "kg", 520)]),
}
rows = []
for dish, (price, sold, ings) in menu.items():
    for ing, q, u, c in ings:
        rows.append(dict(dish=dish, ingredient=ing, qty_per_portion=q, unit=u, unit_cost=c, selling_price=price, portions_sold=sold))
save(pd.DataFrame(rows), "recipes")

# ---------------------------------------------------------------- assets
cats = {"Laptop": (60000, 4), "Delivery vehicle": (650000, 8), "Machinery": (1200000, 10), "Furniture": (45000, 8), "AC unit": (55000, 8), "Server": (300000, 5)}
rows = []
for i in range(42):
    cat = str(rng.choice(list(cats)))
    cost, life = cats[cat]
    pdate = AS_OF - pd.Timedelta(days=int(rng.integers(100, 365 * 8)))
    status = "in_use" if rng.random() > 0.1 else str(rng.choice(["under_repair", "disposed"]))
    rows.append(dict(asset_id=f"AST{i + 1:03d}", asset_name=f"{cat} {i + 1}", category=cat, purchase_date=d(pdate), cost=round(cost * rng.uniform(0.8, 1.2), -2),
                     useful_life_years=life, salvage_value=round(cost * 0.05, -2), status=status))
a = pd.DataFrame(rows)
old = a.index[(pd.to_datetime(a.purchase_date) < AS_OF - pd.Timedelta(days=365 * 5)) & (a.category == "Laptop")]
a.loc[old, "status"] = "in_use"                                                                          # planted: laptops long past their useful life, still in use
save(a, "assets")

# ---------------------------------------------------------------- survey responses
themes = {"delivery": "Delivery was late again", "price": "Too expensive compared to others", "support": "Support was slow to reply", "quality": "Great quality, will buy again",
          "staff": "Friendly staff", "app": "The app keeps crashing"}
rows = []
for i in range(640):
    day = AS_OF - pd.Timedelta(days=int(rng.integers(0, 180)))
    seg = str(rng.choice(["Mumbai", "Pune", "Online"], p=[0.4, 0.25, 0.35]))
    recent = day > AS_OF - pd.Timedelta(days=60)
    mean = 8.4 - (0.9 if recent else 0) - (1.0 if seg == "Pune" else 0)                                  # planted: score fall in recent 60 days, Pune weaker
    score = int(np.clip(round(rng.normal(mean, 2.0)), 0, 10))
    if score >= 9:
        key = str(rng.choice(["quality", "staff"]))
    elif score <= 6:
        key = str(rng.choice(["delivery", "support", "price", "app"], p=[0.45 if recent else 0.2, 0.25, 0.15, 0.15 if recent else 0.4]))
    else:
        key = str(rng.choice(list(themes)))
    rows.append(dict(response_id=f"R{i + 1:04d}", date=d(day), score=score, segment=seg, channel=str(rng.choice(["Email", "WhatsApp", "In store"])), comment=themes[key],
                     csat=int(np.clip(round(score / 2), 1, 5))))
save(pd.DataFrame(rows), "survey_responses")

# ---------------------------------------------------------------- discount_pct on the existing sales file (price charged is net of it)
s = pd.read_csv(OUT / "sales.csv")
if "discount_pct" not in s.columns:
    r2 = np.random.default_rng(77)
    disc = np.where(r2.random(len(s)) < 0.30, r2.choice([5, 10, 15, 20, 30, 40], len(s), p=[0.2, 0.3, 0.2, 0.15, 0.1, 0.05]), 0)
    s["discount_pct"] = disc                                                                             # planted: quantity does not rise with the discount
    s.to_csv(OUT / "sales.csv", index=False)
    print("sales: discount_pct added")
