"""BillBook: the digital version of the paper bill book.

A shopkeeper types an item, a quantity, a rate and (optionally) a customer name. BizIQ then keeps the stock count, who still owes money, and
the day's sales, collections, expenses and profit, and feeds the same entries to every IQ module so they work without anyone building a spreadsheet.

Everything lives in the business's own database (so each business only ever sees its own bills). Customer phone numbers are encrypted when
BIZIQ_ENCRYPT_DATA is on. Amounts are kept to the paisa; what a customer owes is always worked out from bills and payments, never typed in,
so it cannot drift out of step with them.
"""
from __future__ import annotations

import base64
import math
import re
import secrets
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from urllib.parse import quote

import pandas as pd

from . import audit, crypto, data_layer, db

MODES = {"cash": "Cash", "upi": "UPI", "card": "Card", "bank": "Bank transfer", "credit": "Credit"}
MAX_AMOUNT = 1e9
MAX_LINES = 200
EPS = 0.005


class BillError(ValueError):
    """Something the person typed that cannot be saved; the message says what to fix, in plain words."""


# ------------------------------------------------------------------ storage
def _ddl(c) -> None:
    c.execute("""CREATE TABLE IF NOT EXISTS bb_items (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,
        unit TEXT NOT NULL DEFAULT 'pcs', sale_price REAL NOT NULL DEFAULT 0, cost_price REAL, stock REAL NOT NULL DEFAULT 0,
        reorder_level REAL NOT NULL DEFAULT 0, lead_days REAL NOT NULL DEFAULT 3, active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS bb_customers (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,
        phone TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS bb_bills (id INTEGER PRIMARY KEY AUTOINCREMENT, bill_no TEXT NOT NULL UNIQUE, bill_date TEXT NOT NULL,
        customer TEXT NOT NULL DEFAULT '', mode TEXT NOT NULL, subtotal REAL NOT NULL, discount REAL NOT NULL DEFAULT 0, total REAL NOT NULL,
        paid REAL NOT NULL, note TEXT NOT NULL DEFAULT '', cancelled INTEGER NOT NULL DEFAULT 0, token TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL, created_by TEXT NOT NULL DEFAULT '')""")
    c.execute("""CREATE TABLE IF NOT EXISTS bb_bill_lines (id INTEGER PRIMARY KEY AUTOINCREMENT, bill_no TEXT NOT NULL, item TEXT NOT NULL,
        qty REAL NOT NULL, rate REAL NOT NULL, cost REAL, unit TEXT NOT NULL DEFAULT 'pcs')""")
    c.execute("""CREATE TABLE IF NOT EXISTS bb_payments (id INTEGER PRIMARY KEY AUTOINCREMENT, pay_date TEXT NOT NULL, customer TEXT NOT NULL,
        amount REAL NOT NULL, mode TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, created_by TEXT NOT NULL DEFAULT '')""")
    c.execute("""CREATE TABLE IF NOT EXISTS bb_stock_moves (id INTEGER PRIMARY KEY AUTOINCREMENT, move_date TEXT NOT NULL, item TEXT NOT NULL, qty REAL NOT NULL,
        cost_price REAL, supplier TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS bb_expenses (id INTEGER PRIMARY KEY AUTOINCREMENT, exp_date TEXT NOT NULL, category TEXT NOT NULL, amount REAL NOT NULL,
        mode TEXT NOT NULL DEFAULT 'cash', note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, created_by TEXT NOT NULL DEFAULT '')""")
    c.execute("CREATE INDEX IF NOT EXISTS bb_bills_date ON bb_bills (bill_date)")
    c.execute("CREATE INDEX IF NOT EXISTS bb_bills_customer ON bb_bills (customer)")
    c.execute("CREATE INDEX IF NOT EXISTS bb_lines_bill ON bb_bill_lines (bill_no)")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS bb_bills_token ON bb_bills (token) WHERE token <> ''")


def _conn():
    c = data_layer._conn()
    db.ensure(c, "billbook", _ddl)
    return c


# ------------------------------------------------------------------ cleaning what people type
def _text(x, field_: str, maxlen: int = 80, required: bool = True) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(x if x is not None else ""))
    s = re.sub(r"\s+", " ", s).strip()
    if required and not s:
        raise BillError(f"Enter the {field_}.")
    if len(s) > maxlen:
        raise BillError(f"The {field_} is too long (at most {maxlen} characters).")
    return s


def _key(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _num(x, field_: str, minimum: float = 0.0, strict: bool = False) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise BillError(f"The {field_} must be a number.") from None
    if math.isnan(v) or math.isinf(v):
        raise BillError(f"The {field_} must be a real number.")
    if v > MAX_AMOUNT:
        raise BillError(f"The {field_} is too large.")
    if (v <= minimum) if strict else (v < minimum):
        raise BillError(f"The {field_} must be {'more than' if strict else 'at least'} {minimum:g}.")
    return round(v, 4)


def _day(d=None) -> str:
    if d is None:
        return date.today().isoformat()
    try:
        v = pd.Timestamp(d).date()
    except (ValueError, TypeError):
        raise BillError("That date is not valid.") from None
    if v > date.today() + timedelta(days=1):
        raise BillError("The date is in the future.")
    return v.isoformat()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _money(x) -> float:
    return round(float(x) + 0.0, 2)


def fmt(x: float) -> str:
    """Rs 1,23,456 style (lakh grouping), whole rupees unless there are paise."""
    x = round(float(x), 2)
    neg, x = x < 0, abs(x)
    whole, frac = divmod(round(x * 100), 100)
    s = str(int(whole))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = f"{head},{tail}"
    return f"{'-' if neg else ''}Rs {s}" + (f".{int(frac):02d}" if frac else "")


# phone numbers are personal data: sealed with the business's data key when encryption is on
def _seal(phone: str) -> str:
    if not phone or not data_layer.encrypt_on():
        return phone
    blob = crypto.encrypt(data_layer.current_scope(), phone.encode(), "bb_customers.phone")
    return "enc:" + base64.b64encode(blob).decode()


def _open(stored: str) -> str:
    if stored and stored.startswith("enc:"):
        return crypto.decrypt(data_layer.current_scope(), base64.b64decode(stored[4:]), "bb_customers.phone").decode()
    return stored or ""


def clean_phone(p) -> str | None:
    """A WhatsApp-ready Indian number (country code 91 + 10 digits), or None when it is not usable."""
    digits = re.sub(r"\D", "", str(p or ""))
    if len(digits) == 10 and digits[0] in "6789":
        return "91" + digits
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return digits
    if len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        return "91" + digits[1:]
    return None


# ------------------------------------------------------------------ items and stock
def upsert_item(name: str, sale_price, cost_price=None, unit: str = "pcs", stock=None, reorder_level=None, lead_days=None, by: str = "") -> str:
    name = _text(name, "item name", 80)
    price = _num(sale_price, "selling price")
    cost = None if cost_price in (None, "") or (isinstance(cost_price, float) and math.isnan(cost_price)) else _num(cost_price, "cost price")
    unit = _text(unit or "pcs", "unit", 12)
    with _conn() as c:
        row = c.execute("SELECT id, stock FROM bb_items WHERE name_key=?", (_key(name),)).fetchone()
        if row is None:
            st = 0.0 if stock is None else _num(stock, "opening stock", minimum=-MAX_AMOUNT)
            c.execute("INSERT INTO bb_items (name, name_key, unit, sale_price, cost_price, stock, reorder_level, lead_days, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                      (name, _key(name), unit, price, cost, st, 0.0 if reorder_level is None else _num(reorder_level, "reorder level"),
                       3.0 if lead_days is None else _num(lead_days, "lead time"), _now()))
            if st:
                c.execute("INSERT INTO bb_stock_moves (move_date, item, qty, cost_price, kind, note, created_at) VALUES (?,?,?,?,?,?,?)", (_day(), name, st, cost, "opening", "opening stock", _now()))
            return "added"
        sets, vals = ["name=?", "unit=?", "sale_price=?", "active=1"], [name, unit, price]
        if cost is not None:
            sets.append("cost_price=?"); vals.append(cost)
        if reorder_level is not None:
            sets.append("reorder_level=?"); vals.append(_num(reorder_level, "reorder level"))
        if lead_days is not None:
            sets.append("lead_days=?"); vals.append(_num(lead_days, "lead time"))
        if stock is not None:
            new = _num(stock, "stock", minimum=-MAX_AMOUNT)
            diff = new - float(row[1])
            if abs(diff) > 1e-9:
                c.execute("INSERT INTO bb_stock_moves (move_date, item, qty, cost_price, kind, note, created_at) VALUES (?,?,?,?,?,?,?)", (_day(), name, diff, None, "adjust", "stock count corrected", _now()))
            sets.append("stock=?"); vals.append(new)
        c.execute(f"UPDATE bb_items SET {', '.join(sets)} WHERE id=?", (*vals, row[0]))
        return "updated"


def list_items(include_inactive: bool = False) -> pd.DataFrame:
    with _conn() as c:
        df = db.read_df(c, "SELECT name, unit, sale_price, cost_price, stock, reorder_level, lead_days, active FROM bb_items" + ("" if include_inactive else " WHERE active=1") + " ORDER BY name")
    return df


def deactivate_item(name: str) -> None:
    with _conn() as c:
        c.execute("UPDATE bb_items SET active=0 WHERE name_key=?", (_key(name),))


def add_stock(item: str, qty, cost_price=None, supplier: str = "", move_date=None, kind: str = "purchase", note: str = "") -> float:
    """Record stock coming in (purchase) or a correction (adjust: negative allowed, for breakage or a recount). Returns the stock now."""
    item = _text(item, "item name", 80)
    if kind not in ("purchase", "adjust"):
        raise BillError("Unknown stock entry type.")
    q = _num(qty, "quantity", minimum=-MAX_AMOUNT if kind == "adjust" else 0, strict=kind == "purchase")
    if kind == "adjust" and q == 0:
        raise BillError("The quantity cannot be zero.")
    cost = None if cost_price in (None, "") else _num(cost_price, "cost price")
    with _conn() as c:
        row = c.execute("SELECT id, name, stock FROM bb_items WHERE name_key=?", (_key(item),)).fetchone()
        if row is None:
            raise BillError(f"'{item}' is not in your item list. Add the item first.")
        c.execute("INSERT INTO bb_stock_moves (move_date, item, qty, cost_price, supplier, kind, note, created_at) VALUES (?,?,?,?,?,?,?,?)",
                  (_day(move_date), row[1], q, cost, _text(supplier, "supplier", 80, False), kind, _text(note, "note", 200, False), _now()))
        c.execute("UPDATE bb_items SET stock=stock+?" + (", cost_price=?" if cost else "") + " WHERE id=?", (q, *([cost] if cost else []), row[0]))
        return round(float(row[2]) + q, 4)


def low_stock() -> pd.DataFrame:
    it = list_items()
    if it.empty:
        return it
    out = it[(it["stock"] <= 0) | ((it["reorder_level"] > 0) & (it["stock"] <= it["reorder_level"]))].copy()
    out["status"] = out["stock"].apply(lambda s: "out of stock" if s <= 0 else "low")
    return out.sort_values("stock")


def import_items(df: pd.DataFrame) -> dict:
    """Bulk load items from a sheet with columns like name, price, cost, stock, unit. Rows that cannot be read are listed, not guessed."""
    cols = {re.sub(r"[^a-z]", "", str(c).lower()): c for c in df.columns}
    pick = lambda *names: next((cols[n] for n in names if n in cols), None)
    cn, cp, cc, cs, cu = pick("name", "item", "itemname", "product"), pick("price", "saleprice", "sellingprice", "rate", "mrp"), pick("cost", "costprice", "purchaseprice"), pick("stock", "qty", "quantity", "openingstock"), pick("unit", "uom")
    if cn is None or cp is None:
        raise BillError("The sheet needs a column for the item name and one for the selling price.")
    out = dict(added=0, updated=0, errors=[])
    for i, r in df.head(5000).iterrows():
        try:
            res = upsert_item(r[cn], r[cp], r[cc] if cc else None, r[cu] if cu and pd.notna(r[cu]) else "pcs", r[cs] if cs and pd.notna(r[cs]) else None)
            out["added" if res == "added" else "updated"] += 1
        except BillError as e:
            if len(out["errors"]) < 100:
                out["errors"].append(f"Row {i + 2}: {e}")
    return out


# ------------------------------------------------------------------ customers
def upsert_customer(name: str, phone: str = "") -> None:
    name = _text(name, "customer name", 80)
    phone = re.sub(r"[^\d+ ]", "", str(phone or "")).strip()[:20]
    with _conn() as c:
        row = c.execute("SELECT id, phone FROM bb_customers WHERE name_key=?", (_key(name),)).fetchone()
        if row is None:
            c.execute("INSERT INTO bb_customers (name, name_key, phone, created_at) VALUES (?,?,?,?)", (name, _key(name), _seal(phone), _now()))
        elif phone:
            c.execute("UPDATE bb_customers SET phone=?, name=? WHERE id=?", (_seal(phone), name, row[0]))


def customer_names() -> list[str]:
    with _conn() as c:
        return [r[0] for r in c.execute("SELECT name FROM bb_customers ORDER BY name").fetchall()]


def customer_phone(name: str) -> str:
    with _conn() as c:
        r = c.execute("SELECT phone FROM bb_customers WHERE name_key=?", (_key(name),)).fetchone()
    return _open(r[0]) if r else ""


# ------------------------------------------------------------------ billing
@dataclass
class BillResult:
    bill_no: str
    total: float
    paid: float
    due: float
    customer: str
    lines: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    duplicate: bool = False


def create_bill(lines: list[dict], customer: str = "", paid=None, mode: str = "cash", bill_date=None, discount=0.0, note: str = "", by: str = "",
                phone: str = "", token: str = "") -> BillResult:
    """Save one bill. `lines` = [{item, qty, rate}]. Stock goes down, and whatever is not paid now becomes the customer's due.

    `token` makes saving safe to repeat: the same token twice returns the first bill, so a double tap never bills twice."""
    if not lines:
        raise BillError("Add at least one item to the bill.")
    if len(lines) > MAX_LINES:
        raise BillError(f"A bill can have at most {MAX_LINES} items.")
    mode = (mode or "cash").lower()
    if mode not in MODES:
        raise BillError("Choose how the customer paid: cash, UPI, card, bank transfer or credit.")
    cust = _text(customer, "customer name", 80, False)
    clean = []
    for i, ln in enumerate(lines, 1):
        nm = _text(ln.get("item"), f"item name on line {i}", 80)
        q = _num(ln.get("qty"), f"quantity on line {i}", strict=True)
        r = _num(ln.get("rate"), f"rate on line {i}")
        clean.append((nm, q, r))
    subtotal = _money(sum(round(q * r, 2) for _, q, r in clean))
    disc = _num(discount or 0, "discount")
    if disc > subtotal + EPS:
        raise BillError("The discount is more than the bill.")
    total = _money(subtotal - disc)
    paid_v = total if paid is None else _num(paid, "amount paid")
    if mode == "credit":
        paid_v = 0.0
    if paid_v > total + EPS:
        raise BillError(f"The amount paid ({fmt(paid_v)}) is more than the bill ({fmt(total)}).")
    paid_v = min(_money(paid_v), total)
    due = _money(total - paid_v)
    if due > EPS and not cust:
        raise BillError("Enter the customer's name to put the balance on credit. An unnamed customer cannot owe money.")
    if mode != "credit" and paid_v <= EPS and total > EPS:
        mode = "credit"
        if not cust:
            raise BillError("Enter the customer's name to put the bill on credit.")
    token = _text(token, "token", 64, False)
    result_lines, warnings = [], []
    for attempt in range(4):
        try:
            with _conn() as c:
                if token:
                    ex = c.execute("SELECT bill_no, total, paid, customer FROM bb_bills WHERE token=?", (token,)).fetchone()
                    if ex:
                        return BillResult(ex[0], float(ex[1]), float(ex[2]), _money(ex[1] - ex[2]), ex[3], [], ["This bill was already saved."], True)
                n = c.execute("SELECT COALESCE(MAX(CAST(SUBSTR(bill_no, 2) AS INTEGER)), 0) + 1 FROM bb_bills").fetchone()[0]
                bill_no = f"B{int(n):05d}"
                result_lines, warnings = [], []
                c.execute("INSERT INTO bb_bills (bill_no, bill_date, customer, mode, subtotal, discount, total, paid, note, token, created_at, created_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                          (bill_no, _day(bill_date), cust, mode, subtotal, disc, total, paid_v, _text(note, "note", 200, False), token, _now(), by[:120]))
                for nm, q, r in clean:
                    it = c.execute("SELECT id, name, unit, cost_price, stock FROM bb_items WHERE name_key=?", (_key(nm),)).fetchone()
                    if it is None:
                        c.execute("INSERT INTO bb_items (name, name_key, unit, sale_price, stock, created_at) VALUES (?,?,?,?,?,?)", (nm, _key(nm), "pcs", r, 0.0, _now()))
                        it = c.execute("SELECT id, name, unit, cost_price, stock FROM bb_items WHERE name_key=?", (_key(nm),)).fetchone()
                        warnings.append(f"'{nm}' was not in your item list, so it was added. Enter its cost price under Items and stock to see profit.")
                    iid, iname, unit, cost, stock = it
                    c.execute("INSERT INTO bb_bill_lines (bill_no, item, qty, rate, cost, unit) VALUES (?,?,?,?,?,?)", (bill_no, iname, q, r, cost, unit))
                    new_stock = round(float(stock) - q, 4)
                    c.execute("UPDATE bb_items SET stock=? WHERE id=?", (new_stock, iid))
                    result_lines.append(dict(item=iname, qty=q, rate=r, amount=_money(q * r), unit=unit))
                    if new_stock < 0:
                        warnings.append(f"{iname}: stock is now {new_stock:g}. If you received more stock, add it under Items and stock.")
                    if cost is None:
                        warnings.append(f"{iname}: no cost price, so profit on it is unknown.")
            break
        except Exception as e:                                   # two people saved a bill at once and drew the same number: try the next one
            if "UNIQUE" in str(e).upper() or "unique" in str(e):
                if attempt == 3:
                    raise BillError("Could not save the bill because others are saving at the same time. Try again.") from e
                continue
            raise
    if cust:
        upsert_customer(cust, phone)
    return BillResult(bill_no, total, paid_v, due, cust, result_lines, list(dict.fromkeys(warnings)))


def get_bill(bill_no: str) -> dict | None:
    with _conn() as c:
        b = db.read_df(c, "SELECT * FROM bb_bills WHERE bill_no=?", (bill_no,))
        if b.empty:
            return None
        ln = db.read_df(c, "SELECT item, qty, rate, unit FROM bb_bill_lines WHERE bill_no=?", (bill_no,))
    return dict(bill=b.iloc[0].to_dict(), lines=ln)


def cancel_bill(bill_no: str, by: str = "", org_id: str = "") -> None:
    """Void a bill: its stock goes back and it drops out of every total. The row is kept (marked cancelled) so the history stays honest."""
    with _conn() as c:
        b = c.execute("SELECT id, cancelled, total, customer FROM bb_bills WHERE bill_no=?", (bill_no,)).fetchone()
        if b is None:
            raise BillError("No such bill.")
        if b[1]:
            raise BillError("This bill is already cancelled.")
        for item, qty in c.execute("SELECT item, qty FROM bb_bill_lines WHERE bill_no=?", (bill_no,)).fetchall():
            c.execute("UPDATE bb_items SET stock=stock+? WHERE name_key=?", (qty, _key(item)))
        c.execute("UPDATE bb_bills SET cancelled=1 WHERE id=?", (b[0],))
    audit.log(org_id or "local", by or "unknown", "bill.cancel", bill_no, f"total {b[2]}")


# ------------------------------------------------------------------ what customers owe (always worked out from bills and payments)
def credit_ledger() -> pd.DataFrame:
    """One row per credit bill: what was left unpaid at the counter, how much of it later payments cleared (oldest bill first), and the date it was cleared."""
    with _conn() as c:
        bills = db.read_df(c, "SELECT bill_no, bill_date, customer, total, paid, id FROM bb_bills WHERE cancelled=0 AND customer<>'' ORDER BY bill_date, id")
        pays = db.read_df(c, "SELECT customer, pay_date, amount, id FROM bb_payments ORDER BY pay_date, id")
    rows = []
    if bills.empty:
        return pd.DataFrame(columns=["bill_no", "customer", "bill_date", "credit", "applied", "outstanding", "cleared_date"])
    bills["credit"] = (bills["total"] - bills["paid"]).round(2)
    bills = bills[bills["credit"] > EPS]
    bills["ck"] = bills["customer"].map(_key)
    pays["ck"] = pays["customer"].map(_key) if len(pays) else pd.Series(dtype=str)
    for ck, g in bills.groupby("ck", sort=False):
        p = pays[pays["ck"] == ck] if len(pays) else pays
        events = [(r.pay_date, float(r.amount)) for r in p.itertuples()]
        cum_pay, cum_idx, running = [], 0.0, 0.0
        for dte, amt in events:
            running += amt
            cum_pay.append((running, dte))
        total_paid = running
        cum_due = 0.0
        for r in g.itertuples():
            before = cum_due
            cum_due += float(r.credit)
            applied = max(0.0, min(float(r.credit), total_paid - before))
            out = round(float(r.credit) - applied, 2)
            cleared = None
            if out <= EPS:
                cleared = next((d for tot, d in cum_pay if tot + EPS >= cum_due), None)
                out = 0.0
            rows.append(dict(bill_no=r.bill_no, customer=r.customer, bill_date=r.bill_date, credit=float(r.credit), applied=round(applied, 2), outstanding=out, cleared_date=cleared))
    return pd.DataFrame(rows)


def dues_table() -> pd.DataFrame:
    led = credit_ledger()
    cols = ["customer", "open_bills", "due", "oldest_bill_date", "days_since_oldest", "last_payment_date"]
    if led.empty:
        return pd.DataFrame(columns=cols)
    op = led[led["outstanding"] > EPS].copy()
    if op.empty:
        return pd.DataFrame(columns=cols)
    g = op.groupby(op["customer"].map(_key)).agg(customer=("customer", "first"), open_bills=("bill_no", "size"), due=("outstanding", "sum"), oldest_bill_date=("bill_date", "min"))
    g["days_since_oldest"] = (pd.Timestamp(date.today()) - pd.to_datetime(g["oldest_bill_date"])).dt.days
    with _conn() as c:
        p = db.read_df(c, "SELECT customer, MAX(pay_date) AS last_payment_date FROM bb_payments GROUP BY customer")
    if len(p):
        g["last_payment_date"] = g.index.map(dict(zip(p["customer"].map(_key), p["last_payment_date"])))
    else:
        g["last_payment_date"] = None
    g["due"] = g["due"].round(2)
    return g[cols].sort_values("due", ascending=False).reset_index(drop=True)


def customer_due(name: str) -> float:
    t = dues_table()
    m = t[t["customer"].map(_key) == _key(name)]
    return float(m["due"].iloc[0]) if len(m) else 0.0


def record_payment(customer: str, amount, mode: str = "cash", pay_date=None, note: str = "", by: str = "") -> float:
    """A customer pays towards what they owe. Returns what they still owe. Cannot take more than they owe."""
    cust = _text(customer, "customer name", 80)
    amt = _money(_num(amount, "amount", strict=True))
    mode = (mode or "cash").lower()
    if mode not in MODES or mode == "credit":
        raise BillError("Choose how they paid: cash, UPI, card or bank transfer.")
    owed = customer_due(cust)
    if owed <= EPS:
        raise BillError(f"{cust} does not owe anything.")
    if amt > owed + EPS:
        raise BillError(f"{cust} owes only {fmt(owed)}. The payment is {fmt(amt)}.")
    with _conn() as c:
        name = c.execute("SELECT customer FROM bb_bills WHERE customer<>'' AND cancelled=0 ORDER BY id DESC").fetchall()
        canon = next((r[0] for r in name if _key(r[0]) == _key(cust)), cust)
        c.execute("INSERT INTO bb_payments (pay_date, customer, amount, mode, note, created_at, created_by) VALUES (?,?,?,?,?,?,?)",
                  (_day(pay_date), canon, amt, mode, _text(note, "note", 200, False), _now(), by[:120]))
    return _money(owed - amt)


def customer_history(name: str) -> pd.DataFrame:
    """Bills and payments for one customer, oldest first, with a running balance."""
    with _conn() as c:
        cb = db.read_df(c, "SELECT bill_date AS date, bill_no AS ref, total, paid, customer FROM bb_bills WHERE cancelled=0 AND customer<>''")
        cp = db.read_df(c, "SELECT pay_date AS date, 'payment' AS ref, amount, customer FROM bb_payments")
    k = _key(name)
    cb = cb[cb["customer"].map(_key) == k]
    cp = cp[cp["customer"].map(_key) == k]
    rows = [dict(date=r.date, entry=f"Bill {r.ref}", bill_total=r.total, paid_at_counter=r.paid, owed_change=_money(r.total - r.paid)) for r in cb.itertuples()]
    rows += [dict(date=r.date, entry="Payment received", bill_total=None, paid_at_counter=None, owed_change=-_money(r.amount)) for r in cp.itertuples()]
    out = pd.DataFrame(rows, columns=["date", "entry", "bill_total", "paid_at_counter", "owed_change"])
    if out.empty:
        return out
    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    out["balance"] = out["owed_change"].cumsum().round(2)
    return out


# ------------------------------------------------------------------ expenses
EXPENSE_CATEGORIES = ["Rent", "Salary", "Electricity", "Transport", "Packaging", "Repairs", "Marketing", "Phone and internet", "Tea and snacks", "Other"]


def add_expense(category: str, amount, mode: str = "cash", exp_date=None, note: str = "", by: str = "") -> None:
    cat = _text(category, "expense type", 60)
    amt = _money(_num(amount, "amount", strict=True))
    mode = (mode or "cash").lower()
    if mode not in MODES or mode == "credit":
        raise BillError("Choose how it was paid: cash, UPI, card or bank transfer.")
    with _conn() as c:
        c.execute("INSERT INTO bb_expenses (exp_date, category, amount, mode, note, created_at, created_by) VALUES (?,?,?,?,?,?,?)",
                  (_day(exp_date), cat, amt, mode, _text(note, "note", 200, False), _now(), by[:120]))


def delete_expense(expense_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM bb_expenses WHERE id=?", (int(expense_id),))


# ------------------------------------------------------------------ the day book
def _frames(start: str, end: str) -> dict:
    with _conn() as c:
        bills = db.read_df(c, "SELECT * FROM bb_bills WHERE cancelled=0 AND bill_date>=? AND bill_date<=?", (start, end))
        lines = db.read_df(c, "SELECT l.bill_no, l.item, l.qty, l.rate, l.cost, b.bill_date, b.subtotal, b.total FROM bb_bill_lines l JOIN bb_bills b ON b.bill_no=l.bill_no "
                              "WHERE b.cancelled=0 AND b.bill_date>=? AND b.bill_date<=?", (start, end))
        pays = db.read_df(c, "SELECT * FROM bb_payments WHERE pay_date>=? AND pay_date<=?", (start, end))
        exps = db.read_df(c, "SELECT * FROM bb_expenses WHERE exp_date>=? AND exp_date<=?", (start, end))
        buys = db.read_df(c, "SELECT * FROM bb_stock_moves WHERE kind='purchase' AND move_date>=? AND move_date<=?", (start, end))
    if len(lines):
        f = (lines["total"] / lines["subtotal"].replace(0, float("nan"))).fillna(1.0)         # a bill discount is shared across its lines
        lines["revenue"] = lines["qty"] * lines["rate"] * f
        lines["cogs"] = lines["qty"] * lines["cost"]
    return dict(bills=bills, lines=lines, pays=pays, exps=exps, buys=buys)


def _day_numbers(fr: dict) -> dict:
    b, l, p, e, buys = fr["bills"], fr["lines"], fr["pays"], fr["exps"], fr["buys"]
    sales = float(b["total"].sum()) if len(b) else 0.0
    counter = b[b["mode"] != "credit"].groupby("mode")["paid"].sum().to_dict() if len(b) else {}
    later = p.groupby("mode")["amount"].sum().to_dict() if len(p) else {}
    by_mode = {m: round(counter.get(m, 0.0) + later.get(m, 0.0), 2) for m in set(counter) | set(later)}
    known = l[l["cost"].notna()] if len(l) else l
    unknown = l[l["cost"].isna()] if len(l) else l
    gross = float(known["revenue"].sum() - known["cogs"].sum()) if len(known) else 0.0
    exp_total = float(e["amount"].sum()) if len(e) else 0.0
    return dict(bills=int(len(b)), sales=round(sales, 2), collected_at_counter=round(float(b["paid"].sum()) if len(b) else 0.0, 2), dues_collected=round(float(p["amount"].sum()) if len(p) else 0.0, 2),
                collected_by_mode=by_mode, on_credit=round(float((b["total"] - b["paid"]).sum()) if len(b) else 0.0, 2), expenses=round(exp_total, 2),
                gross_profit=round(gross, 2), sales_without_cost=round(float(unknown["revenue"].sum()) if len(unknown) else 0.0, 2),
                net_profit=round(gross - exp_total, 2), stock_bought=round(float((buys["qty"] * buys["cost_price"].fillna(0)).sum()) if len(buys) else 0.0, 2))


def day_summary(day=None) -> dict:
    d = _day(day)
    fr = _frames(d, d)
    out = _day_numbers(fr)
    out["date"] = d
    out["collected_total"] = round(out["collected_at_counter"] + out["dues_collected"], 2)
    known_rev = float(fr["lines"].loc[fr["lines"]["cost"].notna(), "revenue"].sum()) if len(fr["lines"]) else 0.0
    out["margin_pct"] = round(100 * out["gross_profit"] / known_rev, 1) if known_rev > 0 else None
    out["bills_table"] = fr["bills"][["bill_no", "customer", "total", "paid", "mode"]].assign(customer=lambda x: x["customer"].replace("", "Walk-in")) if len(fr["bills"]) else pd.DataFrame(columns=["bill_no", "customer", "total", "paid", "mode"])
    out["expenses_table"] = fr["exps"][["id", "category", "amount", "mode", "note"]] if len(fr["exps"]) else pd.DataFrame(columns=["id", "category", "amount", "mode", "note"])
    out["top_items"] = (fr["lines"].groupby("item").agg(qty=("qty", "sum"), sales=("revenue", "sum")).round(2).sort_values("sales", ascending=False).head(8).reset_index()
                        if len(fr["lines"]) else pd.DataFrame(columns=["item", "qty", "sales"]))
    return out


def daily_series(days: int = 14, end=None) -> pd.DataFrame:
    e = pd.Timestamp(_day(end))
    s = e - pd.Timedelta(days=days - 1)
    fr = _frames(s.date().isoformat(), e.date().isoformat())
    rows = []
    for d in pd.date_range(s, e):
        k = d.date().isoformat()
        sub = dict(bills=fr["bills"][fr["bills"]["bill_date"] == k] if len(fr["bills"]) else fr["bills"], lines=fr["lines"][fr["lines"]["bill_date"] == k] if len(fr["lines"]) else fr["lines"],
                   pays=fr["pays"][fr["pays"]["pay_date"] == k] if len(fr["pays"]) else fr["pays"], exps=fr["exps"][fr["exps"]["exp_date"] == k] if len(fr["exps"]) else fr["exps"],
                   buys=fr["buys"][fr["buys"]["move_date"] == k] if len(fr["buys"]) else fr["buys"])
        n = _day_numbers(sub)
        rows.append(dict(date=k, sales=n["sales"], gross_profit=n["gross_profit"], expenses=n["expenses"], net_profit=n["net_profit"], bills=n["bills"]))
    return pd.DataFrame(rows)


def expense_total_month(day=None) -> float:
    d = pd.Timestamp(_day(day))
    fr = _frames(d.replace(day=1).date().isoformat(), d.date().isoformat())
    return float(fr["exps"]["amount"].sum()) if len(fr["exps"]) else 0.0


# ------------------------------------------------------------------ feeding the IQ modules
def build_datasets(credit_days: int = 15) -> dict[str, pd.DataFrame]:
    with _conn() as c:
        bills = db.read_df(c, "SELECT bill_no, bill_date, customer, subtotal, total FROM bb_bills WHERE cancelled=0")
        lines = db.read_df(c, "SELECT bill_no, item, qty, rate FROM bb_bill_lines")
        items = db.read_df(c, "SELECT name, stock, cost_price, lead_days FROM bb_items WHERE active=1")
        exps = db.read_df(c, "SELECT id, exp_date, category, amount, mode, note FROM bb_expenses")
    out: dict[str, pd.DataFrame] = {}
    if len(bills) and len(lines):
        m = lines.merge(bills, on="bill_no")
        f = (m["total"] / m["subtotal"].replace(0, float("nan"))).fillna(1.0)
        # customers who gave no name become one-off customers, so they do not pile up as a single giant "customer"
        cust = [c if c else f"Walk-in {b}" for c, b in zip(m["customer"], m["bill_no"])]
        out["sales"] = pd.DataFrame(dict(order_id=m["bill_no"], date=m["bill_date"], customer_id=cust, product=m["item"], region="Counter", quantity=m["qty"],
                                         unit_price=(m["rate"] * f).round(2), discount_pct=((1 - f) * 100).round(2)))
    led = credit_ledger()
    if len(led):
        out["invoices"] = pd.DataFrame(dict(invoice_id=led["bill_no"], party=led["customer"], type="receivable", invoice_date=led["bill_date"],
                                            due_date=(pd.to_datetime(led["bill_date"]) + pd.Timedelta(days=credit_days)).dt.strftime("%Y-%m-%d"),
                                            amount=led["credit"], paid_date=led["cleared_date"]))
    if len(items):
        out["inventory"] = pd.DataFrame(dict(product=items["name"], stock_on_hand=items["stock"].clip(lower=0), lead_time_days=items["lead_days"]))
        pc = items[items["cost_price"].notna() & (items["cost_price"] > 0)]
        if len(pc):
            out["product_costs"] = pd.DataFrame(dict(product=pc["name"], unit_cost=pc["cost_price"], channel_fee_pct=0.0, shipping_cost_per_order=0.0, packaging_cost=0.0))
    if len(exps):
        out["expenses"] = pd.DataFrame(dict(expense_id=exps["id"].map(lambda i: f"BBE{i}"), date=exps["exp_date"], category=exps["category"], vendor=exps["note"].replace("", "-"),
                                            amount=exps["amount"], department="Operations", payment_mode=exps["mode"].map(lambda m: MODES.get(m, m)), receipt_attached=0))
    return out


def sync_to_datasets(credit_days: int = 15) -> dict[str, str]:
    """Copy what was entered in the bill book into the datasets the IQ modules read. A dataset the business uploaded from another source is never overwritten."""
    built = build_datasets(credit_days)
    meta = data_layer.dataset_meta()
    source = dict(zip(meta["dataset"], meta["source"])) if len(meta) else {}
    status = {}
    for ds in ("sales", "invoices", "inventory", "product_costs", "expenses"):
        df = built.get(ds)
        if df is None or df.empty:
            status[ds] = "nothing entered yet"
        elif ds in source and source[ds] != "billbook":
            status[ds] = f"kept your uploaded {ds} data (not overwritten)"
        else:
            rep = data_layer.ingest(df, ds, source="billbook", mode="replace")
            status[ds] = f"synced {rep.rows_out} rows" if rep.ok else "could not sync: " + "; ".join(rep.errors[:2])
    return status


def export_tables() -> dict[str, pd.DataFrame]:
    """Every bill-book table as plain frames (phone numbers decrypted: this is the owner's own data, for export)."""
    out = {}
    with _conn() as c:
        for name, sql in (("items", "SELECT name, unit, sale_price, cost_price, stock, reorder_level, lead_days, active FROM bb_items ORDER BY name"),
                          ("customers", "SELECT name, phone FROM bb_customers ORDER BY name"),
                          ("bills", "SELECT bill_no, bill_date, customer, mode, subtotal, discount, total, paid, note, cancelled, created_by FROM bb_bills ORDER BY id"),
                          ("bill_lines", "SELECT bill_no, item, qty, rate, cost, unit FROM bb_bill_lines ORDER BY id"),
                          ("payments", "SELECT pay_date, customer, amount, mode, note, created_by FROM bb_payments ORDER BY id"),
                          ("stock_moves", "SELECT move_date, item, qty, cost_price, supplier, kind, note FROM bb_stock_moves ORDER BY id"),
                          ("expenses", "SELECT exp_date, category, amount, mode, note, created_by FROM bb_expenses ORDER BY id")):
            df = db.read_df(c, sql)
            if name == "customers" and len(df):
                df["phone"] = df["phone"].map(_open)
            if len(df):
                out[name] = df
    return out


# ------------------------------------------------------------------ sharing (links only: nothing is sent from here)
def whatsapp_link(phone, text: str) -> str:
    p = clean_phone(phone)
    return f"https://wa.me/{p}?text={quote(text)}" if p else f"https://wa.me/?text={quote(text)}"


def reminder_text(customer: str, amount: float, shop: str, lang: str = "en") -> str:
    if lang == "hi":
        return f"नमस्ते {customer} जी, {shop} में आपका {fmt(amount)} बाकी है। कृपया सुविधा अनुसार भुगतान कर दें। धन्यवाद!"
    return f"Hello {customer}, this is a gentle reminder from {shop}: {fmt(amount)} is pending on your account. Please pay when convenient. Thank you!"


def receipt_text(res: BillResult, shop: str, lang: str = "en") -> str:
    rows = "\n".join(f"{ln['item']} x {ln['qty']:g} @ {ln['rate']:g} = {fmt(ln['amount'])}" for ln in res.lines)
    if lang == "hi":
        tail = f"कुल: {fmt(res.total)}\nदिया: {fmt(res.paid)}" + (f"\nबाकी: {fmt(res.due)}" if res.due > EPS else "")
        return f"{shop}\nबिल {res.bill_no}\n{rows}\n{tail}\nधन्यवाद!"
    tail = f"Total: {fmt(res.total)}\nPaid: {fmt(res.paid)}" + (f"\nBalance due: {fmt(res.due)}" if res.due > EPS else "")
    return f"{shop}\nBill {res.bill_no}\n{rows}\n{tail}\nThank you!"


def summary_text(s: dict, shop: str, lang: str = "en") -> str:
    if lang == "hi":
        return (f"{shop} - {s['date']} का हिसाब\nबिक्री: {fmt(s['sales'])} ({s['bills']} बिल)\nवसूली: {fmt(s['collected_total'])}\nउधार दिया: {fmt(s['on_credit'])}\n"
                f"खर्चा: {fmt(s['expenses'])}\nमुनाफ़ा (अनुमान): {fmt(s['net_profit'])}")
    return (f"{shop} - {s['date']}\nSales: {fmt(s['sales'])} ({s['bills']} bills)\nCollected: {fmt(s['collected_total'])}\nGiven on credit: {fmt(s['on_credit'])}\n"
            f"Expenses: {fmt(s['expenses'])}\nProfit (estimate): {fmt(s['net_profit'])}")
