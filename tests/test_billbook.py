"""BillBook: bills, stock, dues, day book, and feeding the IQ modules. Amounts are worked out by hand."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from biziq import billbook as bb, crypto, data_layer as d, db, pipeline

TODAY = date.today().isoformat()


def ago(n):
    return (date.today() - timedelta(days=n)).isoformat()


@pytest.fixture()
def shop(tmp_path, monkeypatch):
    monkeypatch.delenv("BIZIQ_ENCRYPT_DATA", raising=False)
    d.use_db(tmp_path / "shop" / "biziq.db")
    bb.upsert_item("Rice 1kg", 100, cost_price=60, unit="kg", stock=50, reorder_level=10)
    bb.upsert_item("Soap", 40, cost_price=25, stock=20, reorder_level=5)
    yield tmp_path
    d.use_db(None)


def line(item, qty, rate):
    return dict(item=item, qty=qty, rate=rate)


class TestBilling:
    def test_bill_reduces_stock_and_numbers_add_up(self, shop):
        r = bb.create_bill([line("Rice 1kg", 3, 100), line("Soap", 2, 40)], customer="Ravi", paid=300, mode="upi")
        assert r.bill_no == "B00001" and r.total == 380 and r.paid == 300 and r.due == 80
        it = bb.list_items().set_index("name")
        assert it.loc["Rice 1kg", "stock"] == 47 and it.loc["Soap", "stock"] == 18
        assert bb.create_bill([line("Soap", 1, 40)]).bill_no == "B00002"

    def test_item_names_are_matched_loosely_and_unknown_items_are_added(self, shop):
        r = bb.create_bill([line("  rice   1KG ", 1, 100), line("Sugar", 2, 45)])
        assert bb.list_items().set_index("name").loc["Rice 1kg", "stock"] == 49
        assert any("Sugar" in w and "added" in w for w in r.warnings)

    def test_selling_more_than_stock_warns_but_saves(self, shop):
        r = bb.create_bill([line("Soap", 25, 40)])
        assert any("stock is now -5" in w for w in r.warnings) and bb.list_items().set_index("name").loc["Soap", "stock"] == -5

    @pytest.mark.parametrize("kw,msg", [
        (dict(lines=[]), "at least one item"),
        (dict(lines=[line("Soap", 0, 40)]), "quantity"),
        (dict(lines=[line("Soap", -1, 40)]), "quantity"),
        (dict(lines=[line("Soap", float("nan"), 40)]), "real number"),
        (dict(lines=[line("Soap", 1, -5)]), "rate"),
        (dict(lines=[line("Soap", 1, 1e12)]), "too large"),
        (dict(lines=[line("", 1, 40)]), "item name"),
        (dict(lines=[line("Soap", 1, 40)], discount=100), "discount is more"),
        (dict(lines=[line("Soap", 1, 40)], paid=50), "more than the bill"),
        (dict(lines=[line("Soap", 1, 40)], paid=10), "customer's name"),
        (dict(lines=[line("Soap", 1, 40)], mode="credit"), "customer's name"),
        (dict(lines=[line("Soap", 1, 40)], mode="barter"), "how the customer paid"),
    ])
    def test_bad_input_is_refused_with_a_reason_and_nothing_is_saved(self, shop, kw, msg):
        before = bb.list_items().set_index("name")["stock"].to_dict()
        with pytest.raises(bb.BillError, match=msg):
            bb.create_bill(**kw)
        assert bb.list_items().set_index("name")["stock"].to_dict() == before

    def test_credit_sale_with_name_works_and_walk_in_cannot_owe(self, shop):
        r = bb.create_bill([line("Soap", 2, 40)], customer="Meena", mode="credit")
        assert r.paid == 0 and r.due == 80

    def test_discount_is_shared_across_lines(self, shop):
        r = bb.create_bill([line("Rice 1kg", 1, 100), line("Soap", 1, 100)], discount=20)
        assert r.total == 180
        s = bb.day_summary()
        assert s["sales"] == 180 and s["gross_profit"] == 180 - 60 - 25

    def test_saving_twice_with_the_same_token_bills_once(self, shop):
        a = bb.create_bill([line("Soap", 1, 40)], token="cart-1")
        b = bb.create_bill([line("Soap", 1, 40)], token="cart-1")
        assert a.bill_no == b.bill_no and b.duplicate
        assert bb.list_items().set_index("name").loc["Soap", "stock"] == 19 and bb.day_summary()["bills"] == 1

    def test_sql_looking_text_is_stored_as_text(self, shop):
        bb.upsert_item("x'); DROP TABLE bb_items;--", 10)
        bb.create_bill([line("x'); DROP TABLE bb_items;--", 1, 10)], customer="Robert'); DROP TABLE bb_bills;--", mode="credit")
        assert len(bb.list_items()) == 3 and bb.dues_table().iloc[0]["due"] == 10

    def test_cancel_restores_stock_and_drops_out_of_totals(self, shop):
        r = bb.create_bill([line("Rice 1kg", 4, 100)], customer="Ravi", mode="credit")
        bb.cancel_bill(r.bill_no, by="owner@x", org_id="o")
        assert bb.list_items().set_index("name").loc["Rice 1kg", "stock"] == 50
        assert bb.day_summary()["sales"] == 0 and bb.dues_table().empty
        with pytest.raises(bb.BillError, match="already cancelled"):
            bb.cancel_bill(r.bill_no)
        with pytest.raises(bb.BillError, match="No such bill"):
            bb.cancel_bill("B99999")


class TestDues:
    def _two_credit_bills(self):
        bb.create_bill([line("Rice 1kg", 10, 100)], customer="Ravi", paid=0, mode="credit", bill_date=ago(20))     # owes 1000
        bb.create_bill([line("Rice 1kg", 5, 100)], customer="ravi ", paid=0, mode="credit", bill_date=ago(10))     # owes 500, same person typed differently

    def test_balance_is_the_sum_and_name_case_does_not_split_customers(self, shop):
        self._two_credit_bills()
        t = bb.dues_table()
        assert len(t) == 1 and t.iloc[0]["due"] == 1500 and t.iloc[0]["open_bills"] == 2 and t.iloc[0]["days_since_oldest"] == 20

    def test_payments_clear_the_oldest_bill_first(self, shop):
        self._two_credit_bills()
        left = bb.record_payment("Ravi", 1200, "upi", pay_date=ago(5))
        assert left == 300
        led = bb.credit_ledger().set_index("bill_no")
        assert led.loc["B00001", "outstanding"] == 0 and led.loc["B00001", "cleared_date"] == ago(5)
        assert led.loc["B00002", "outstanding"] == 300 and pd.isna(led.loc["B00002", "cleared_date"])
        bb.record_payment("Ravi", 300, "cash")
        assert bb.dues_table().empty and bb.credit_ledger().set_index("bill_no").loc["B00002", "cleared_date"] == TODAY

    def test_cannot_take_more_than_owed_or_from_a_stranger(self, shop):
        self._two_credit_bills()
        with pytest.raises(bb.BillError, match="owes only Rs 1,500"):
            bb.record_payment("Ravi", 1501)
        with pytest.raises(bb.BillError, match="does not owe"):
            bb.record_payment("Nobody", 10)
        with pytest.raises(bb.BillError):
            bb.record_payment("Ravi", 0)
        with pytest.raises(bb.BillError):
            bb.record_payment("Ravi", 10, "credit")

    def test_part_paid_at_counter_leaves_only_the_balance(self, shop):
        bb.create_bill([line("Rice 1kg", 2, 100)], customer="Sita", paid=150, mode="cash")
        assert bb.dues_table().iloc[0]["due"] == 50

    def test_statement_runs_down_to_the_balance(self, shop):
        self._two_credit_bills()
        bb.record_payment("Ravi", 400)
        h = bb.customer_history("Ravi")
        assert list(h["owed_change"]) == [1000, 500, -400] and h["balance"].iloc[-1] == 1100


class TestDayBook:
    def test_sales_profit_expenses_by_hand(self, shop):
        bb.create_bill([line("Rice 1kg", 3, 100)], paid=300, mode="cash")                      # sales 300, cost 180
        bb.create_bill([line("Soap", 2, 40)], customer="Ravi", paid=30, mode="upi")            # sales 80, cost 50, 30 paid, 50 on credit
        bb.add_expense("Electricity", 70)
        bb.create_bill([line("Soap", 1, 40)], customer="Ravi", paid=0, mode="credit", bill_date=ago(3))
        bb.record_payment("Ravi", 20, "cash")                                                  # an older due collected today
        s = bb.day_summary()
        assert s["sales"] == 380 and s["bills"] == 2 and s["on_credit"] == 50
        assert s["gross_profit"] == 380 - 230 and s["expenses"] == 70 and s["net_profit"] == 80
        assert s["collected_at_counter"] == 330 and s["dues_collected"] == 20 and s["collected_total"] == 350
        assert s["collected_by_mode"] == {"cash": 320.0, "upi": 30.0}

    def test_items_without_cost_are_not_counted_as_profit(self, shop):
        bb.upsert_item("Mystery", 100)
        bb.create_bill([line("Mystery", 2, 100)])
        s = bb.day_summary()
        assert s["sales"] == 200 and s["gross_profit"] == 0 and s["sales_without_cost"] == 200

    def test_other_days_do_not_leak_in(self, shop):
        bb.create_bill([line("Soap", 1, 40)], bill_date=ago(2))
        assert bb.day_summary()["sales"] == 0 and bb.day_summary(ago(2))["sales"] == 40
        ser = bb.daily_series(5)
        assert len(ser) == 5 and ser["sales"].sum() == 40

    def test_future_dates_refused(self, shop):
        with pytest.raises(bb.BillError, match="future"):
            bb.create_bill([line("Soap", 1, 40)], bill_date=date.today() + timedelta(days=30))

    def test_stock_purchase_and_low_stock(self, shop):
        assert bb.add_stock("Soap", 10, cost_price=27, supplier="Agro") == 30
        assert bb.list_items().set_index("name").loc["Soap", "cost_price"] == 27
        bb.create_bill([line("Soap", 26, 40)])
        assert list(bb.low_stock()["name"]) == ["Soap"] and bb.low_stock().iloc[0]["status"] == "low"
        assert bb.add_stock("Soap", -2, kind="adjust", note="broken") == 2
        with pytest.raises(bb.BillError, match="not in your item list"):
            bb.add_stock("Ghost", 5)
        with pytest.raises(bb.BillError):
            bb.add_stock("Soap", 0)


class TestFeedingTheIQs:
    def _fill(self):
        for i in range(30):
            bb.create_bill([line("Rice 1kg", 2, 100), line("Soap", 1, 40)], customer=f"C{i % 6}", paid=100 if i % 3 else 0, mode="cash" if i % 3 else "credit", bill_date=ago(60 - i))
        bb.add_expense("Rent", 5000, exp_date=ago(20))

    def test_sync_builds_the_datasets_and_the_pipeline_reads_them(self, shop):
        self._fill()
        st = bb.sync_to_datasets()
        assert all(v.startswith("synced") for v in st.values()), st
        data = d.load_all()
        assert {"sales", "invoices", "inventory", "product_costs", "expenses"} <= set(data)
        assert len(data["sales"]) == 60 and set(data["sales"]["region"]) == {"Counter"}
        assert (data["invoices"]["type"] == "receivable").all() and len(data["invoices"]) == 30          # every filler bill leaves a balance (paid 100 or 0 of 240)
        pr = pipeline.run_all(data)
        assert not pr.failed, {k: v.title for k, v in pr.failed.items()}

    def test_unnamed_customers_do_not_become_one_giant_customer(self, shop):
        for _ in range(5):
            bb.create_bill([line("Soap", 1, 40)])
        assert bb.build_datasets()["sales"]["customer_id"].nunique() == 5

    def test_uploaded_data_is_never_overwritten(self, shop):
        up = pd.DataFrame([dict(order_id="U1", date="2026-01-01", customer_id="x", product="P", region="R", quantity=1, unit_price=1)])
        d.ingest(up, "sales", source="upload")
        bb.create_bill([line("Soap", 1, 40)])
        st = bb.sync_to_datasets()
        assert "kept your uploaded" in st["sales"] and list(d.load_dataset("sales")["order_id"]) == ["U1"]

    def test_resync_replaces_its_own_rows_not_duplicates_them(self, shop):
        bb.create_bill([line("Soap", 1, 40)])
        bb.sync_to_datasets()
        bb.create_bill([line("Soap", 1, 40)])
        bb.sync_to_datasets()
        assert len(d.load_dataset("sales")) == 2

    def test_nothing_entered_is_said_plainly(self, shop):
        st = bb.sync_to_datasets()
        assert st["sales"] == "nothing entered yet" and st["inventory"].startswith("synced")


class TestSharingAndFormatting:
    @pytest.mark.parametrize("raw,out", [("98765 43210", "919876543210"), ("+91 98765-43210", "919876543210"), ("09876543210", "919876543210"),
                                          ("12345", None), ("5876543210", None), ("", None), (None, None)])
    def test_phone_cleaning(self, raw, out):
        assert bb.clean_phone(raw) == out

    def test_whatsapp_link_is_encoded_and_falls_back_without_a_number(self):
        assert bb.whatsapp_link("9876543210", "Hi & bye").startswith("https://wa.me/919876543210?text=Hi%20%26%20bye")
        assert bb.whatsapp_link("nope", "x") == "https://wa.me/?text=x"

    def test_messages_in_both_languages(self, shop):
        assert "Rs 1,500" in bb.reminder_text("Ravi", 1500, "Sharma Store") and "बाकी" in bb.reminder_text("Ravi", 1500, "Sharma Store", "hi")
        r = bb.create_bill([line("Soap", 2, 40)], customer="Ravi", paid=30, mode="cash")
        assert "Balance due: Rs 50" in bb.receipt_text(r, "Sharma Store") and "बाकी: Rs 50" in bb.receipt_text(r, "Sharma Store", "hi")
        assert "Profit (estimate)" in bb.summary_text(bb.day_summary(), "Sharma Store") and "मुनाफ़ा" in bb.summary_text(bb.day_summary(), "Sharma Store", "hi")

    @pytest.mark.parametrize("x,s", [(0, "Rs 0"), (1000, "Rs 1,000"), (123456, "Rs 1,23,456"), (12345678, "Rs 1,23,45,678"), (12.5, "Rs 12.50"), (-2500, "-Rs 2,500")])
    def test_rupee_format(self, x, s):
        assert bb.fmt(x) == s

    def test_import_items_from_a_sheet(self, shop):
        df = pd.DataFrame({"Item Name": ["Tea", "Salt", "", "Oil"], "Selling Price": [120, 20, 5, "abc"], "Cost": [90, 12, 1, 80], "Qty": [10, 40, 1, 2]})
        r = bb.import_items(df)
        assert r["added"] == 2 and len(r["errors"]) == 2
        with pytest.raises(bb.BillError):
            bb.import_items(pd.DataFrame({"foo": [1]}))


class TestPhoneProtection:
    def test_phone_is_encrypted_at_rest_when_encryption_is_on(self, shop, monkeypatch):
        from biziq import platform_db
        monkeypatch.setattr(platform_db, "PLATFORM_DB", shop / "platform.db")
        monkeypatch.setenv("BIZIQ_MASTER_KEY", crypto.generate_master_key())
        monkeypatch.setenv("BIZIQ_ENCRYPT_DATA", "1")
        crypto.clear_cache()
        bb.upsert_customer("Ravi", "98765 43210")
        with bb._conn() as c:
            raw = c.execute("SELECT phone FROM bb_customers").fetchone()[0]
        assert raw.startswith("enc:") and "98765" not in raw
        assert bb.customer_phone("ravi") == "98765 43210"
        crypto.clear_cache()
