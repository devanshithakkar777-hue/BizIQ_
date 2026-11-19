"""The sixteen channel, operations, risk and people modules: planted findings, edge cases and the people-data privacy rules."""
import numpy as np
import pandas as pd
import pytest

from biziq import accounts, registry
from biziq.registry import Context

AS_OF = pd.Timestamp("2026-10-31")
NEW = ["ChurnIQ", "PersonalisationIQ", "WhatsAppIQ", "MarketplaceIQ", "StoreIQ", "DistributorIQ", "LogisticsIQ", "ProductionIQ", "MaintenanceIQ",
       "TenderIQ", "ESGIQ", "VendorRiskIQ", "FraudIQ", "HiringIQ", "SalaryIQ", "PerformanceReviewIQ"]


def run(name, data, **settings):
    spec = registry.get(name)
    res = spec.run(data, Context(as_of=AS_OF, today=AS_OF + pd.Timedelta(days=1), settings=settings))
    assert registry.check_result(spec, res) == [], name
    assert spec.recommend(res.facts, res.flags)
    return res


def titles(res):
    return " | ".join(f.title for f in res.flags)


class TestOnSampleData:
    def test_all_sixteen_run_and_meet_contract(self, pr):
        for n in NEW:
            assert n in pr.modules, (n, pr.skipped.get(n), pr.failed.get(n))
            assert registry.check_result(registry.get(n), pr.modules[n]) == []
        assert not pr.failed

    @pytest.mark.parametrize("name,needle", [
        ("MarketplaceIQ", "paid"), ("DistributorIQ", "Konkan"), ("StoreIQ", "Thane"), ("StoreIQ", "Nashik"), ("LogisticsIQ", "BudgetShip"),
        ("ProductionIQ", "Line C"), ("MaintenanceIQ", "Press-2"), ("TenderIQ", "close within 7 days"), ("ESGIQ", "Electricity"),
        ("FraudIQ", "no approver"), ("WhatsAppIQ", "Opt-outs")])
    def test_planted_problem_found(self, pr, name, needle):
        assert needle in titles(pr.modules[name]), titles(pr.modules[name])

    def test_people_modules_are_sensitive_and_hidden_from_viewers(self):
        for n in ("HiringIQ", "SalaryIQ", "PerformanceReviewIQ"):
            assert registry.get(n).sensitive
        viewer = accounts.User(id=1, org_id="o", org_name="O", email="v@x.test", name="V", role="viewer")
        analyst = accounts.User(id=2, org_id="o", org_name="O", email="a@x.test", name="A", role="analyst")
        hr = accounts.User(id=3, org_id="o", org_name="O", email="h@x.test", name="H", role="hr")
        for n in ("HiringIQ", "SalaryIQ", "PerformanceReviewIQ"):
            assert not accounts.can_view_module(viewer, registry.get(n)) and accounts.can_view_module(hr, registry.get(n))
        assert accounts.can_view_module(viewer, registry.get("ChurnIQ"))

    def test_people_modules_never_put_individuals_in_facts(self, pr):
        for n in ("HiringIQ", "SalaryIQ", "PerformanceReviewIQ"):
            blob = str(pr.modules[n].facts)
            assert "E0" not in blob and "C0" not in blob          # no employee or candidate ids reach the decision layer


def sales_frame(rows):
    return pd.DataFrame(rows, columns=["order_id", "date", "customer_id", "product", "region", "quantity", "unit_price"])


class TestChurn:
    def test_customer_who_stopped_is_flagged_and_steady_one_is_not(self):
        rows = []
        for i in range(8):                                     # buys every 20 days up to day 160, then silence for 200 days
            rows.append((f"A{i}", AS_OF - pd.Timedelta(days=360 - 20 * i), "gone", "p", "r", 1, 100.0))
        for i in range(18):                                    # steady: every 20 days to today
            rows.append((f"B{i}", AS_OF - pd.Timedelta(days=340 - 20 * i), "steady", "p", "r", 1, 100.0))
        r = run("ChurnIQ", {"sales": sales_frame(rows)})
        t = r.tables["customers_to_win_back"]
        assert list(t["customer_id"]) == ["gone"] and t.iloc[0]["status"] == "lapsed"

    def test_one_time_buyers_are_not_called_churned(self):
        rows = [(f"O{i}", AS_OF - pd.Timedelta(days=300), f"c{i}", "p", "r", 1, 50.0) for i in range(5)]
        r = run("ChurnIQ", {"sales": sales_frame(rows)})
        assert len(r.tables["customers_to_win_back"]) == 0 and r.facts["one_time_customers"] == 5


class TestPersonalisation:
    def test_pair_that_always_travels_together_has_high_lift(self):
        rows = []
        for i in range(40):
            rows += [(f"O{i}", AS_OF - pd.Timedelta(days=i), f"c{i}", "Shoes", "r", 1, 1.0), (f"O{i}", AS_OF - pd.Timedelta(days=i), f"c{i}", "Socks", "r", 1, 1.0)]
        for i in range(40, 120):
            rows.append((f"O{i}", AS_OF - pd.Timedelta(days=i % 90), f"c{i}", "Cap" if i % 2 else "Belt", "r", 1, 1.0))
        r = run("PersonalisationIQ", {"sales": sales_frame(rows)})
        assert r.facts["strongest_pair"] == "Shoes + Socks" and r.facts["strongest_pair_lift"] > 2

    def test_rare_pairs_below_minimum_are_ignored(self):
        rows = [("O1", AS_OF, "c1", "A", "r", 1, 1.0), ("O1", AS_OF, "c1", "B", "r", 1, 1.0)] + [(f"O{i}", AS_OF, f"c{i}", "A", "r", 1, 1.0) for i in range(2, 30)]
        assert run("PersonalisationIQ", {"sales": sales_frame(rows)}).facts["pairs_found"] == 0


class TestMarketplace:
    def _orders(self, settle_factor=1.0, settled=True, days_ago=40):
        d = AS_OF - pd.Timedelta(days=days_ago)
        rows = []
        for i in range(30):
            gross, comm, ship, other = 1000.0, 100.0, 50.0, 20.0
            rows.append(dict(order_id=f"O{i}", date=d, channel="X", gross_amount=gross, commission=comm, shipping_fee=ship, other_fees=other,
                             settlement_amount=(gross - comm - ship - other) * settle_factor if settled else np.nan, settled_date=d + pd.Timedelta(days=9) if settled else pd.NaT, returned=0))
        return pd.DataFrame(rows)

    def test_exact_payouts_raise_nothing(self):
        r = run("MarketplaceIQ", {"marketplace_orders": self._orders()})
        assert r.facts["short_paid_orders"] == 0 and r.worst_level() == "green"

    def test_short_payment_found(self):
        r = run("MarketplaceIQ", {"marketplace_orders": self._orders(settle_factor=0.9)})
        assert r.facts["short_paid_orders"] == 30 and r.worst_level() == "red"

    def test_recent_unsettled_orders_are_not_overdue_but_old_ones_are(self):
        assert run("MarketplaceIQ", {"marketplace_orders": self._orders(settled=False, days_ago=5)}).facts["unsettled_orders_past_wait"] == 0
        assert run("MarketplaceIQ", {"marketplace_orders": self._orders(settled=False, days_ago=40)}).facts["unsettled_orders_past_wait"] == 30

    def test_works_without_settlement_columns(self):
        o = self._orders().drop(columns=["settlement_amount", "settled_date"])
        assert "short_paid_orders" not in run("MarketplaceIQ", {"marketplace_orders": o}).facts


class TestOperations:
    def test_logistics_compares_carriers_and_flags_overdue(self):
        rows = []
        for i in range(120):
            car = "Good" if i % 2 else "Bad"
            ship = AS_OF - pd.Timedelta(days=30)
            late = 0 if car == "Good" or i % 5 == 0 else 3
            rows.append(dict(shipment_id=f"S{i}", carrier=car, ship_date=ship, promised_date=ship + pd.Timedelta(days=4),
                             delivered_date=ship + pd.Timedelta(days=4 + late), status="delivered", cost=50.0))
        rows.append(dict(shipment_id="X", carrier="Good", ship_date=AS_OF - pd.Timedelta(days=20), promised_date=AS_OF - pd.Timedelta(days=15), delivered_date=pd.NaT, status="in_transit", cost=50.0))
        r = run("LogisticsIQ", {"shipments": pd.DataFrame(rows)})
        assert r.facts["worst_carrier"] == "Bad" and r.facts["in_transit_past_promise"] == 1

    def test_production_division_by_zero_safe(self):
        p = pd.DataFrame([dict(line="A", date=AS_OF - pd.Timedelta(days=i), planned_units=0, produced_units=0, defective_units=0, downtime_min=0) for i in range(5)])
        with pytest.raises(ValueError, match="^needs "):               # the pipeline reports this as "skipped", not as a failure
            run("ProductionIQ", {"production": p})

    def test_maintenance_repeat_failure_and_overdue_service(self):
        rows = [dict(asset="Old", date=AS_OF - pd.Timedelta(days=10 * i + 5), event_type="breakdown", downtime_hours=4.0, cost=1000.0) for i in range(5)]
        rows += [dict(asset="Old", date=AS_OF - pd.Timedelta(days=300), event_type="preventive", downtime_hours=1.0, cost=100.0),
                 dict(asset="New", date=AS_OF - pd.Timedelta(days=20), event_type="preventive", downtime_hours=1.0, cost=100.0)]
        r = run("MaintenanceIQ", {"maintenance": pd.DataFrame(rows)})
        assert r.facts["repeat_failure_assets"] == 1 and r.facts["assets_overdue_preventive_with_failures"] == 1

    def test_tender_deadlines_and_price_gap(self):
        rows = [dict(tender_id="T1", title="a", deadline=AS_OF + pd.Timedelta(days=3), value=1e6, status="open"),
                dict(tender_id="T2", title="b", deadline=AS_OF - pd.Timedelta(days=3), value=1e6, status="open")]
        rows += [dict(tender_id=f"L{i}", title="l", deadline=AS_OF - pd.Timedelta(days=40), value=1e6, status="lost", our_bid=1.2e6, winning_bid=1.0e6) for i in range(4)]
        r = run("TenderIQ", {"tenders": pd.DataFrame(rows)})
        assert r.facts["due_within_7_days"] == 1 and r.facts["missed_open_tenders"] == 1 and r.facts["median_lost_bid_above_winner_pct"] == 20.0

    def test_esg_states_its_assumptions(self):
        e = pd.DataFrame({"month": pd.date_range("2025-11-01", periods=12, freq="MS"), "electricity_kwh": [1000.0] * 12})
        r = run("ESGIQ", {"esg": e})
        assert r.facts["grid_factor_assumed_kg_per_kwh"] == 0.71 and any("ESTIMATE" in n for n in r.notes)


class TestRisk:
    def test_split_payments_and_missing_approver_found(self):
        rows = [dict(expense_id=f"E{i}", date=AS_OF - pd.Timedelta(days=i), category="x", vendor=f"v{i}", amount=4000.0 + i, approved_by="Boss") for i in range(40)]
        rows += [dict(expense_id="S1", date=AS_OF - pd.Timedelta(days=2), category="x", vendor="Splitter", amount=30000.0, approved_by="Boss"),
                 dict(expense_id="S2", date=AS_OF - pd.Timedelta(days=2), category="x", vendor="Splitter", amount=29000.0, approved_by="Boss"),
                 dict(expense_id="B1", date=AS_OF - pd.Timedelta(days=1), category="x", vendor="Big", amount=80000.0, approved_by=None)]
        r = run("FraudIQ", {"expenses": pd.DataFrame(rows)})
        assert r.facts["possible_split_payment_groups"] == 1 and r.facts["large_payments_without_approver"] == 1
        assert any("INDICATORS" in n for n in r.notes)

    def test_benford_skipped_on_small_data(self):
        rows = [dict(expense_id=f"E{i}", date=AS_OF, category="x", vendor="v", amount=100.0 + i, approved_by="B") for i in range(50)]
        assert "first_digit_conformity" not in run("FraudIQ", {"expenses": pd.DataFrame(rows)}).facts

    def test_vendor_concentration_and_sole_source(self):
        rows = []
        for i in range(20):
            rows.append(dict(supplier="Big", product="P1", order_date=AS_OF, promised_date=AS_OF, delivered_date=AS_OF, qty_ordered=100.0, qty_received=100.0, unit_price=100.0))
        rows.append(dict(supplier="Small", product="P2", order_date=AS_OF, promised_date=AS_OF, delivered_date=AS_OF, qty_ordered=10.0, qty_received=10.0, unit_price=100.0))
        r = run("VendorRiskIQ", {"supplier_deliveries": pd.DataFrame(rows)})
        assert r.facts["top_supplier"] == "Big" and r.facts["top_supplier_share_pct"] > 90 and r.facts["sole_source_products"] == 2


def hr_frame(n=60, gender=True, ratings=True):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        g = "female" if i % 2 else "male"
        sal = 50000 + rng.integers(-3000, 3000) - (8000 if g == "female" else 0)
        rows.append(dict(employee_id=f"E{i:03d}", department="Sales", monthly_salary=float(sal), join_date=AS_OF - pd.Timedelta(days=int(rng.integers(100, 2000))),
                         **({"gender": g} if gender else {}), **({"performance_rating": 3.0} if ratings else {})))
    return pd.DataFrame(rows)


class TestPeople:
    def test_gender_gap_reported_only_with_enough_people_and_never_individuals(self):
        r = run("SalaryIQ", {"hr": hr_frame(60)})
        assert r.facts["average_gender_pay_gap_within_department_pct"] > 10
        small = run("SalaryIQ", {"hr": hr_frame(8)})                                   # 4 and 4: below the minimum group of 5
        assert "average_gender_pay_gap_within_department_pct" not in small.facts and len(small.tables["gender_pay_gap_by_department"]) == 0

    def test_salary_works_without_optional_columns(self):
        assert run("SalaryIQ", {"hr": hr_frame(40, gender=False)}).facts["active_employees"] == 40

    def test_identical_ratings_flagged_as_not_separating(self):
        r = run("PerformanceReviewIQ", {"hr": hr_frame(40)})
        assert "same rating" in titles(r)

    def test_review_module_with_no_ratings_column_does_not_crash(self):
        r = run("PerformanceReviewIQ", {"hr": hr_frame(40, ratings=False).assign(performance_rating=np.nan)})
        assert r.facts["employees_rated"] == 0

    def test_hiring_flags_stalled_and_uses_no_candidate_ids_in_facts(self):
        rows = [dict(candidate_id=f"C{i}", role="R", applied_date=AS_OF - pd.Timedelta(days=60), stage="interview", source="Portal",
                     last_stage_date=AS_OF - pd.Timedelta(days=40)) for i in range(10)]
        r = run("HiringIQ", {"candidates": pd.DataFrame(rows)})
        assert r.facts["candidates_waiting_over_threshold"] == 10 and "C0" not in str(r.facts)


class TestChannels:
    def test_distributor_stock_building_detected(self):
        rows = []
        for m in range(3):
            mon = (AS_OF - pd.DateOffset(months=2 - m)).replace(day=1)
            rows.append(dict(distributor="Stuffed", month=mon, primary_sales=200.0, secondary_sales=100.0, closing_stock=500.0))
            rows.append(dict(distributor="Fine", month=mon, primary_sales=100.0, secondary_sales=100.0, closing_stock=100.0))
        r = run("DistributorIQ", {"distributors": pd.DataFrame(rows)})
        assert r.facts["worst_distributor"] == "Stuffed" and r.facts["distributors_building_stock"] == 1

    def test_whatsapp_high_opt_out_red(self):
        m = pd.DataFrame([dict(date=AS_OF - pd.Timedelta(days=i), campaign="c", sent=1000, delivered=950, read=500, replied=50, opted_out=40) for i in range(5)])
        assert run("WhatsAppIQ", {"whatsapp_messages": m}).worst_level() == "red"

    def test_store_compares_with_median(self):
        rows = []
        for s, mult in (("A", 1.0), ("B", 1.0), ("C", 1.0), ("Weak", 0.4)):
            for i in range(60):
                rows.append(dict(store=s, date=AS_OF - pd.Timedelta(days=i), sales=1000.0 * mult, transactions=10))
        r = run("StoreIQ", {"store_daily": pd.DataFrame(rows)})
        assert r.facts["worst_store"] == "Weak" and r.facts["stores_well_below_peers"] == 1


class TestUploadOfNewDatasets:
    def test_messy_marketplace_export_maps_and_loads(self, tmp_path):
        from biziq import data_layer as d
        n = 12
        raw = pd.DataFrame({"Order ID": [f"A{i}" for i in range(n)] + ["A1"], "Order Date": [f"{i + 1:02d}/10/2026" for i in range(n)] + ["02/10/2026"], "Marketplace": ["Amazon"] * (n + 1),
                            "Gross Amount": ["₹1,200"] * (n - 1) + ["x", "₹1,200"], "Commission": ["120"] * (n + 1)})
        p = tmp_path / "m.csv"
        raw.to_csv(p, index=False)
        df, rep = d.clean_and_validate(d.read_upload(p, dataset="marketplace_orders"), "marketplace_orders")
        assert rep.ok and len(df) == n - 1 and rep.rows_rejected >= 1 and df["gross_amount"].iloc[0] == 1200
