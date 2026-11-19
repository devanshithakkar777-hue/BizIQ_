"""The fifteen planning, books and business-model modules: planted findings on sample data, numbers checked by hand, and edge cases."""
import numpy as np
import pandas as pd
import pytest

from biziq import registry
from biziq.registry import Context

AS_OF = pd.Timestamp("2026-10-31")
NEW = ["BudgetIQ", "ScenarioIQ", "UnitEconomicsIQ", "DiscountIQ", "AnomalyIQ", "PayablesIQ", "ReconIQ", "ComplianceCalendarIQ", "AssetIQ",
       "InventoryIQ", "ProjectIQ", "SubscriptionIQ", "AppointmentIQ", "RecipeIQ", "SurveyIQ"]


def run(name, data, as_of=AS_OF, **settings):
    spec = registry.get(name)
    res = spec.run(data, Context(as_of=as_of, today=as_of + pd.Timedelta(days=1), settings=settings))
    assert registry.check_result(spec, res) == [], name
    assert spec.recommend(res.facts, res.flags)
    return res


def titles(res):
    return " | ".join(f.title for f in res.flags)


def ts(*a):
    return pd.Timestamp(*a)


class TestOnSampleData:
    def test_all_fifteen_run_and_meet_contract(self, pr):
        for n in NEW:
            assert n in pr.modules, (n, pr.skipped.get(n), pr.failed.get(n))
            assert registry.check_result(registry.get(n), pr.modules[n]) == []
        assert not pr.failed

    @pytest.mark.parametrize("name,needle", [
        ("BudgetIQ", "Marketing is"), ("DiscountIQ", "30% or more"), ("PayablesIQ", "early"), ("ReconIQ", "repeat a reference"), ("ReconIQ", "amounts differ"),
        ("ReconIQ", "not in your books"), ("AssetIQ", "past their useful life"), ("InventoryIQ", "run short"), ("ProjectIQ", "overrun"), ("ProjectIQ", "past their due date"),
        ("SubscriptionIQ", "churn"), ("AppointmentIQ", "Monday"), ("RecipeIQ", "Chicken Biryani"), ("SurveyIQ", "NPS fell"), ("UnitEconomicsIQ", "cost to win")])
    def test_planted_problem_found(self, pr, name, needle):
        if name == "UnitEconomicsIQ":
            needle = "return"
        assert needle in titles(pr.modules[name]), titles(pr.modules[name])

    def test_recon_finds_exactly_what_was_planted(self, pr):
        f = pr.modules["ReconIQ"].facts
        assert f["in_bank_not_in_books"] == 5 and f["amount_mismatches"] == 3 and f["duplicate_bank_lines"] == 2

    def test_facts_hold_no_nan(self, pr):
        for n in NEW:
            for k, v in pr.modules[n].facts.items():
                assert not (isinstance(v, float) and np.isnan(v)), (n, k)


# ------------------------------------------------------------------ hand-built cases with answers worked out by hand
class TestBudget:
    def test_over_and_under(self):
        months = pd.date_range("2026-08-01", periods=3, freq="MS")
        b = pd.DataFrame([dict(month=m, category=c, budget=100000) for m in months for c in ("Marketing", "Software")])
        e = pd.DataFrame([dict(expense_id=f"{m}{c}", date=m + pd.Timedelta(days=5), category=c, vendor="v", amount=a)
                          for m in months for c, a in (("Marketing", 150000), ("Software", 40000))])
        r = run("BudgetIQ", {"budgets": b, "expenses": e})
        assert r.facts["total_budget"] == "Rs 6.00 L" and r.facts["total_actual"] == "Rs 5.70 L"
        assert r.facts["worst_category"] == "Marketing" and r.facts["worst_category_overspend_pct"] == 50.0
        assert "Marketing is 50% over" in titles(r) and "below 75%" in titles(r)

    def test_open_month_is_ignored(self):
        b = pd.DataFrame([dict(month=ts("2026-10-01"), category="X", budget=1000), dict(month=ts("2026-11-01"), category="X", budget=1000)])
        e = pd.DataFrame([dict(expense_id="1", date=ts("2026-10-03"), category="X", vendor="v", amount=900)])
        r = run("BudgetIQ", {"budgets": b, "expenses": e}, as_of=ts("2026-10-31"))
        assert r.facts["months_compared"] == 1

    def test_no_overlap_is_skipped_not_failed(self):
        b = pd.DataFrame([dict(month=ts("2026-01-01"), category="X", budget=1000)])
        e = pd.DataFrame([dict(expense_id="1", date=ts("2026-05-03"), category="X", vendor="v", amount=900)])
        with pytest.raises(ValueError, match="^needs"):
            run("BudgetIQ", {"budgets": b, "expenses": e})


class TestScenario:
    def _fin(self, rev=1000, cogs=600, opex=300, n=12):
        return pd.DataFrame(dict(month=pd.date_range("2025-11-01", periods=n, freq="MS"), revenue=rev, cogs=cogs, operating_expenses=opex, cash=500))

    def test_cushions_by_hand(self):
        r = run("ScenarioIQ", {"financials": self._fin()})
        # annual: revenue 12000, cogs 7200, opex 3600 -> profit 1200. Contribution 4800 -> volume cushion 25%; price cushion 10%
        assert r.facts["volume_fall_that_erases_profit_pct"] == 25.0 and r.facts["price_cut_that_erases_profit_pct"] == 10.0
        assert r.facts["base_operating_profit"] == "Rs 1,200"
        sc = r.tables["scenarios"].set_index("scenario")
        assert sc.loc["Price down 5%, volume unchanged", "operating_profit"] == 600.0       # 12000*0.95 - 7200 - 3600
        assert sc.loc["Volume down 10%", "operating_profit"] == 720.0                       # 10800 - 6480 - 3600

    def test_loss_making_and_short_history(self):
        assert "loss" in titles(run("ScenarioIQ", {"financials": self._fin(opex=500)}))
        with pytest.raises(ValueError, match="^needs"):
            run("ScenarioIQ", {"financials": self._fin(n=2)})

    def test_custom_scenarios(self):
        r = run("ScenarioIQ", {"financials": self._fin()}, scenarios={"My test": dict(price_pct=10)})
        assert r.tables["scenarios"].set_index("scenario").loc["My test", "operating_profit"] == 2400.0


class TestUnitEconomics:
    def test_cac_by_hand(self):
        rows = []
        for i in range(10):                                    # 10 genuinely new customers in the last 6 months, each one order of 1000
            rows.append(dict(order_id=f"N{i}", date=ts("2026-07-01") + pd.Timedelta(days=i), customer_id=f"new{i}", product="P", region="R", quantity=1, unit_price=1000))
        rows.append(dict(order_id="old", date=ts("2025-11-01"), customer_id="old", product="P", region="R", quantity=1, unit_price=1000))
        camp = pd.DataFrame([dict(date=ts("2026-08-01"), channel="Meta", campaign="c", spend=50000, impressions=1, clicks=1, conversions=10, revenue=20000),
                             dict(date=ts("2026-08-02"), channel="Google", campaign="c", spend=50000, impressions=1, clicks=1, conversions=2, revenue=5000)])
        r = run("UnitEconomicsIQ", {"sales": pd.DataFrame(rows), "campaigns": camp})
        assert r.facts["new_customers_6m"] == 10 and r.facts["blended_cac"] == "Rs 10,000"
        assert r.facts["recovery_ratio_90d"] == 0.1 and "return" in titles(r)
        assert r.facts["costliest_channel"] == "Google" and r.facts["costliest_vs_cheapest_x"] == 5.0

    def test_no_spend_or_new_customers_is_skipped(self):
        s = pd.DataFrame([dict(order_id="1", date=ts("2026-09-01"), customer_id="a", product="P", region="R", quantity=1, unit_price=1)])
        c = pd.DataFrame([dict(date=ts("2026-09-01"), channel="x", campaign="c", spend=0, impressions=1, clicks=1, conversions=1, revenue=1)])
        with pytest.raises(ValueError, match="^needs"):
            run("UnitEconomicsIQ", {"sales": s, "campaigns": c})


class TestDiscount:
    def _sales(self, deep_qty):
        rows = []
        for i in range(120):
            rows.append(dict(order_id=f"F{i}", date=ts("2026-09-01"), customer_id="c", product="P", region="R", quantity=2, unit_price=100, discount_pct=0))
        for i in range(60):
            rows.append(dict(order_id=f"D{i}", date=ts("2026-09-02"), customer_id="c", product="P", region="R", quantity=deep_qty, unit_price=70, discount_pct=30))
        return pd.DataFrame(rows)

    def test_discount_cost_by_hand_and_no_lift(self):
        r = run("DiscountIQ", {"sales": self._sales(2)})
        # a 30% discount leaves price 70 -> list price 100 -> 30 given away per unit; 60 lines x 2 units
        assert r.facts["deep_discount_cost"] == "Rs 3,600" and r.facts["quantity_difference_at_30pct_plus"] == 0.0
        assert "30% or more" in titles(r)

    def test_real_lift_is_not_flagged(self):
        r = run("DiscountIQ", {"sales": self._sales(4)})
        assert r.facts["quantity_difference_at_30pct_plus"] == 100.0 and "not any bigger" not in titles(r)

    def test_below_cost_found_with_product_costs(self):
        costs = pd.DataFrame([dict(product="P", unit_cost=80, channel_fee_pct=0, shipping_cost_per_order=0, packaging_cost=0)])
        r = run("DiscountIQ", {"sales": self._sales(2), "product_costs": costs})
        assert r.facts["lines_sold_below_cost"] == 60 and "below cost" in titles(r)

    def test_missing_discount_column_is_skipped(self):
        s = self._sales(2).drop(columns="discount_pct")
        with pytest.raises(ValueError, match="^needs"):
            run("DiscountIQ", {"sales": s})


class TestAnomaly:
    def test_planted_spike_found_and_flat_data_is_quiet(self):
        days = pd.date_range("2026-07-01", "2026-10-31")
        flat = pd.DataFrame([dict(order_id=str(i), date=d, customer_id="c", product="P", region="R", quantity=10 + (i % 3), unit_price=100) for i, d in enumerate(days)])
        assert run("AnomalyIQ", {"sales": flat}).facts["unusual_values_found"] == 0
        spike = flat.copy()
        spike.loc[spike["date"] == ts("2026-10-20"), "quantity"] = 400
        r = run("AnomalyIQ", {"sales": spike})
        assert r.facts["biggest_period"] == "2026-10-20" and r.facts["unusual_values_last_30_days"] >= 1

    def test_needs_some_data(self):
        with pytest.raises(ValueError, match="^needs"):
            run("AnomalyIQ", {})


class TestPayables:
    def test_ageing_by_hand(self):
        inv = pd.DataFrame([
            dict(invoice_id="1", party="A", type="payable", invoice_date=ts("2026-08-01"), due_date=ts("2026-09-01"), amount=1000, paid_date=pd.NaT),     # 60 days overdue
            dict(invoice_id="2", party="B", type="payable", invoice_date=ts("2026-10-01"), due_date=ts("2026-11-10"), amount=500, paid_date=pd.NaT),      # due in 10 days
            dict(invoice_id="3", party="C", type="payable", invoice_date=ts("2026-07-01"), due_date=ts("2026-07-31"), amount=200, paid_date=ts("2026-08-20")),  # paid 20 days late
            dict(invoice_id="4", party="D", type="receivable", invoice_date=ts("2026-07-01"), due_date=ts("2026-07-31"), amount=9999, paid_date=pd.NaT)])
        r = run("PayablesIQ", {"invoices": inv})
        assert r.facts["open_payables"] == "Rs 1,500" and r.facts["overdue_amount"] == "Rs 1,000" and r.facts["due_next_14_days"] == "Rs 500"
        assert r.facts["paid_late_pct"] == 100.0 and "overdue" in titles(r)

    def test_duplicate_bill_and_cash_shortfall(self):
        inv = pd.DataFrame([dict(invoice_id=f"X{i}", party="V", type="payable", invoice_date=ts("2026-10-01") + pd.Timedelta(days=i), due_date=ts("2026-11-15"), amount=700, paid_date=pd.NaT) for i in range(2)])
        fin = pd.DataFrame([dict(month=ts("2026-10-01"), cash=1000)])
        r = run("PayablesIQ", {"invoices": inv, "financials": fin})
        assert r.facts["possible_duplicate_bills"] == 1 and "exceed the cash" in titles(r)

    def test_no_payables_is_skipped(self):
        inv = pd.DataFrame([dict(invoice_id="1", party="A", type="receivable", invoice_date=ts("2026-08-01"), due_date=ts("2026-09-01"), amount=1, paid_date=pd.NaT)])
        with pytest.raises(ValueError, match="^needs"):
            run("PayablesIQ", {"invoices": inv})


class TestRecon:
    def _pair(self):
        led = pd.DataFrame([dict(date=ts("2026-10-01"), description="a", amount=-100.0, reference="R1"), dict(date=ts("2026-10-02"), description="b", amount=250.0, reference="R2"),
                            dict(date=ts("2026-10-03"), description="c", amount=-75.5, reference=None), dict(date=ts("2026-10-04"), description="d", amount=-40.0, reference="R4")])
        bank = pd.DataFrame([dict(date=ts("2026-10-02"), description="a", amount=-100.0, reference="R1"), dict(date=ts("2026-10-04"), description="b", amount=249.0, reference="R2"),
                             dict(date=ts("2026-10-05"), description="c", amount=-75.5, reference=None), dict(date=ts("2026-10-05"), description="fee", amount=-10.0, reference="B9")])
        return bank, led

    def test_each_kind_of_difference(self):
        bank, led = self._pair()
        r = run("ReconIQ", {"bank_statement": bank, "ledger_entries": led})
        assert r.facts["matched"] == 2                              # R1 by reference, the no-reference line by amount and date
        assert r.facts["amount_mismatches"] == 1 and r.facts["mismatch_net"] == "-Rs 1"
        assert r.facts["in_bank_not_in_books"] == 1 and r.facts["in_books_not_on_statement"] == 1

    def test_repeated_bank_line_is_a_duplicate(self):
        bank, led = self._pair()
        bank = pd.concat([bank, bank.iloc[[0]]], ignore_index=True)
        assert run("ReconIQ", {"bank_statement": bank, "ledger_entries": led}).facts["duplicate_bank_lines"] == 1

    def test_identical_files_are_clean(self):
        _, led = self._pair()
        r = run("ReconIQ", {"bank_statement": led.copy(), "ledger_entries": led})
        assert r.facts["matched_pct"] == 100.0 and "agree" in titles(r)


class TestComplianceCalendar:
    def test_dates_by_hand(self):
        r = run("ComplianceCalendarIQ", {}, as_of=ts("2026-11-05"))
        t = r.tables["upcoming"].set_index("obligation")["due_date"]
        assert t["GSTR-1 (outward supplies)"] == "2026-11-11" and t["GSTR-3B (summary return and tax payment)"] == "2026-11-20"
        assert t["Advance tax instalment"] == "2026-12-15" and t["TDS return (quarterly)"] == "2027-01-31"
        assert t["TDS / TCS deposit for last month"] == "2026-11-07" and t["Income tax return"] == "2027-07-31"

    def test_due_today_counts_and_profile_filters(self):
        r = run("ComplianceCalendarIQ", {}, as_of=ts("2026-11-11"))
        assert r.tables["upcoming"].set_index("obligation").loc["GSTR-1 (outward supplies)", "days_left"] == 0 and "within 3 days" in titles(r)
        no_staff = run("ComplianceCalendarIQ", {}, profile=["goods"])
        assert "PF contribution" not in set(no_staff.tables["upcoming"]["obligation"])
        with_staff = run("ComplianceCalendarIQ", {}, profile=["goods", "employees"])
        assert "PF contribution" in set(with_staff.tables["upcoming"]["obligation"])

    def test_year_rollover(self):
        r = run("ComplianceCalendarIQ", {}, as_of=ts("2026-12-28"))
        assert r.tables["upcoming"].set_index("obligation").loc["GSTR-1 (outward supplies)", "due_date"] == "2027-01-11"


class TestAssets:
    def test_book_value_by_hand(self):
        a = pd.DataFrame([dict(asset_id="A1", category="Laptop", purchase_date=ts("2024-10-31"), cost=100000, useful_life_years=4, salvage_value=10000, status="in_use"),
                          dict(asset_id="A2", category="Laptop", purchase_date=ts("2020-01-01"), cost=50000, useful_life_years=4, salvage_value=0, status="in_use"),
                          dict(asset_id="A3", category="Van", purchase_date=ts("2019-01-01"), cost=900000, useful_life_years=8, salvage_value=0, status="disposed")])
        r = run("AssetIQ", {"assets": a})
        # A1 is 730 days (1.9986 years) into a 4-year life: 100000 - 90000 * 0.4997 = about 55,031. A2 is fully depreciated (book value 0). A3 is disposed and ignored.
        assert r.facts["assets_on_register"] == 2 and r.facts["fully_depreciated_in_use"] == 1
        assert abs(float(r.tables["by_category"]["book_value"].sum()) - 55031) < 5

    def test_missing_useful_life_is_skipped(self):
        a = pd.DataFrame([dict(asset_id="A1", category="X", purchase_date=ts("2024-01-01"), cost=1000)])
        with pytest.raises(ValueError, match="^needs"):
            run("AssetIQ", {"assets": a.assign(useful_life_years=np.nan)})


class TestInventory:
    def _data(self, stock, demand_units=90):
        inv = pd.DataFrame([dict(product="P", stock_on_hand=stock, lead_time_days=10)])
        days = pd.date_range("2026-08-03", "2026-10-31")
        sales = pd.DataFrame([dict(order_id=str(i), date=d, customer_id="c", product="P", region="R", quantity=demand_units / 90 if False else 1, unit_price=100) for i, d in enumerate(days)])
        return {"inventory": inv, "sales": sales}

    def test_reorder_logic_by_hand(self):
        # 1 unit sold per day, lead time 10 days -> reorder point 10 (+0 safety: constant demand)
        assert run("InventoryIQ", self._data(5)).tables["products"].iloc[0]["status"] == "will run out before resupply"
        assert run("InventoryIQ", self._data(10)).tables["products"].iloc[0]["status"] in ("below reorder point", "ok")
        assert run("InventoryIQ", self._data(60)).tables["products"].iloc[0]["status"] == "ok"
        assert run("InventoryIQ", self._data(500)).tables["products"].iloc[0]["status"] == "overstocked"

    def test_no_recent_sales_is_dead_stock(self):
        d = self._data(40)
        d["sales"] = d["sales"].assign(date=ts("2025-12-01"))
        d["inventory"] = pd.DataFrame([dict(product="P", stock_on_hand=40, lead_time_days=10)])
        assert run("InventoryIQ", d).tables["products"].iloc[0]["status"] == "no recent sales"


class TestProject:
    def test_forecast_overrun_by_hand(self):
        p = pd.DataFrame([dict(project_id="P1", client="C", budget=100000, start_date=ts("2026-08-01"), due_date=ts("2026-12-01"), status="active", billed=20000,
                               cost_to_date=60000, hours_budget=100, hours_spent=60, percent_complete=40.0),
                          dict(project_id="P2", client="C", budget=50000, start_date=ts("2026-01-01"), due_date=ts("2026-06-01"), completed_date=ts("2026-06-01"), status="completed", billed=50000,
                               cost_to_date=35000, hours_budget=50, hours_spent=50, percent_complete=100.0)])
        r = run("ProjectIQ", {"projects": p})
        assert r.facts["projects_forecast_to_overrun"] == 1 and r.facts["worst_forecast_cost_overrun_pct"] == 50.0       # 60000/0.4 = 150000 vs 100000
        assert r.facts["completed_margin_pct"] == 30.0 and r.facts["unbilled_work_value"] == "Rs 20,000"                 # 100000*0.4 - 20000

    def test_under_ten_percent_is_not_forecast_and_late_is_found(self):
        p = pd.DataFrame([dict(project_id="P1", client="C", budget=100000, start_date=ts("2026-08-01"), due_date=ts("2026-10-01"), status="active", billed=0, cost_to_date=9000, hours_budget=100, hours_spent=9, percent_complete=5.0)])
        r = run("ProjectIQ", {"projects": p})
        assert r.facts["projects_forecast_to_overrun"] == 0 and r.facts["late_projects"] == 1


class TestSubscription:
    def test_mrr_and_churn_by_hand(self):
        rows = [dict(subscription_id=f"S{i}", customer=f"c{i}", plan="Pro", start_date=ts("2026-01-01"), end_date=pd.NaT, mrr=1000) for i in range(20)]
        for i in range(2):                                                       # two cancel in October
            rows[i]["end_date"] = ts("2026-10-15")
        r = run("SubscriptionIQ", {"subscriptions": pd.DataFrame(rows)})
        m = r.tables["monthly"].set_index("month")
        assert m.loc["2026-10", "mrr"] == 18000 and m.loc["2026-09", "mrr"] == 20000
        assert m.loc["2026-10", "logo_churn_pct"] == 10.0 and m.loc["2026-10", "churned_mrr"] == 2000
        assert r.facts["current_mrr"] == "Rs 18,000" and r.facts["annualised_run_rate"] == "Rs 2.16 L"

    def test_future_end_dates_do_not_count_and_too_few_is_skipped(self):
        rows = [dict(subscription_id=f"S{i}", customer=f"c{i}", plan="Pro", start_date=ts("2026-01-01"), end_date=ts("2027-03-01"), mrr=1000) for i in range(12)]
        assert run("SubscriptionIQ", {"subscriptions": pd.DataFrame(rows)}).facts["active_subscriptions"] == 12
        with pytest.raises(ValueError, match="^needs"):
            run("SubscriptionIQ", {"subscriptions": pd.DataFrame(rows[:5])})


class TestAppointment:
    def test_rates_by_hand(self):
        rows = []
        for i in range(100):
            st = "no_show" if i < 20 else ("cancelled" if i < 30 else "completed")
            rows.append(dict(appointment_id=f"A{i}", date=ts("2026-10-05") + pd.Timedelta(days=i % 20), customer=f"c{i % 40}", service="S", status=st, price=500))
        r = run("AppointmentIQ", {"appointments": pd.DataFrame(rows)})
        assert r.facts["no_show_rate_pct"] == 20.0 and r.facts["cancellation_rate_pct"] == 10.0 and r.facts["revenue_lost_to_no_shows"] == "Rs 10,000"

    def test_status_spellings_and_small_input(self):
        rows = [dict(appointment_id=f"A{i}", date=ts("2026-10-05"), customer="c", service="S", status=s) for i, s in enumerate(["No-Show", "no show", "NOSHOW", "Done"] * 15)]
        r = run("AppointmentIQ", {"appointments": pd.DataFrame(rows)})
        assert r.facts["no_show_rate_pct"] == 75.0
        with pytest.raises(ValueError, match="^needs"):
            run("AppointmentIQ", {"appointments": pd.DataFrame(rows[:10])})


class TestRecipe:
    def _r(self):
        rows = []
        for dish, price, sold, ings in (("Star", 200, 100, [("a", 1, 40)]), ("Plow", 100, 100, [("a", 1, 70)]), ("Puzzle", 400, 10, [("b", 1, 100)]), ("Dog", 100, 10, [("b", 1, 90)])):
            for ing, q, c in ings:
                rows.append(dict(dish=dish, ingredient=ing, qty_per_portion=q, unit_cost=c, selling_price=price, portions_sold=sold))
        return pd.DataFrame(rows)

    def test_cost_and_menu_grid_by_hand(self):
        r = run("RecipeIQ", {"recipes": self._r()})
        t = r.tables["dishes"].set_index("dish")
        assert t.loc["Star", "food_cost_pct"] == 20.0 and t.loc["Plow", "food_cost_pct"] == 70.0
        assert t.loc["Star", "class"] == "Star" and t.loc["Plow", "class"] == "Plowhorse" and t.loc["Puzzle", "class"] == "Puzzle" and t.loc["Dog", "class"] == "Dog"
        assert "Plow runs a 70%" in titles(r)

    def test_dish_below_cost_is_red(self):
        rec = self._r()
        rec.loc[rec["dish"] == "Dog", "selling_price"] = 50
        r = run("RecipeIQ", {"recipes": rec})
        assert any(f.level == "red" and "Dog" in f.title for f in r.flags)

    def test_missing_prices_skipped_and_missing_sales_still_costs(self):
        with pytest.raises(ValueError, match="^needs"):
            run("RecipeIQ", {"recipes": self._r().assign(selling_price=np.nan)})
        r = run("RecipeIQ", {"recipes": self._r().drop(columns="portions_sold")})
        assert "stars" not in r.facts and r.facts["dishes"] == 4


class TestSurvey:
    def _s(self, scores, start="2026-09-01"):
        return pd.DataFrame([dict(response_id=f"R{i}", date=ts(start) + pd.Timedelta(days=i % 40), score=s, comment="delivery was late" if s <= 6 else "great quality") for i, s in enumerate(scores)])

    def test_nps_by_hand(self):
        r = run("SurveyIQ", {"survey_responses": self._s([10] * 50 + [8] * 25 + [3] * 25)})
        assert r.facts["nps"] == 25.0 and r.facts["promoters_pct"] == 50.0 and r.facts["detractors_pct"] == 25.0
        assert r.facts["top_detractor_theme"] == "Delivery"

    def test_fall_between_windows(self):
        old = self._s([10] * 40 + [5] * 10, start="2026-07-01").assign(date=lambda d: ts("2026-07-20") + pd.to_timedelta(np.arange(len(d)) % 25, unit="D"))
        new = self._s([6] * 40 + [10] * 10, start="2026-10-01").assign(response_id=lambda d: "N" + d["response_id"], date=lambda d: ts("2026-10-01") + pd.to_timedelta(np.arange(len(d)) % 25, unit="D"))
        r = run("SurveyIQ", {"survey_responses": pd.concat([old, new], ignore_index=True)})
        assert r.facts["nps_change_points"] < -8 and "NPS fell" in titles(r)

    def test_scores_out_of_range_and_too_few(self):
        with pytest.raises(ValueError, match="^needs"):
            run("SurveyIQ", {"survey_responses": self._s([10] * 10)})
        r = run("SurveyIQ", {"survey_responses": self._s([10] * 40 + [99] * 20)})
        assert r.facts["responses"] == 40                                                # impossible scores are dropped, not averaged in


def test_duplicate_product_rows_do_not_crash_inventory_or_discount_modules():
    """Found by the slow sweep: a cost list or stock list that repeats a product used to crash two modules."""
    import pandas as pd
    from biziq import pipeline
    from biziq import data_layer as d
    base = {n: pd.read_csv(d.SAMPLE_DIR / f"{n}.csv") for n in ("sales", "inventory", "product_costs") if (d.SAMPLE_DIR / f"{n}.csv").exists()}
    for n in base:
        for c in base[n].columns:
            if "date" in c:
                base[n][c] = pd.to_datetime(base[n][c], errors="coerce")
    for n in ("product_costs", "inventory"):
        data = dict(base)
        data[n] = pd.concat([base[n], base[n].head(3)], ignore_index=True)
        res = pipeline.run_all(data)
        assert not [f.title for f in res.failed.values() if "needs " not in f.title]


class TestPeopleSafeguards:
    def _hr(self, n_sales=8, n_tiny=2):
        import pandas as pd
        rows = []
        for i in range(n_sales):
            rows.append(dict(employee_id=f"S{i}", department="Sales", join_date="2023-01-01", exit_date=None, monthly_salary=30000, overtime_hours_month=30 if i < 4 else 5, satisfaction_score=4 if i < 4 else 8, performance_rating=5))
        for i in range(n_tiny):
            rows.append(dict(employee_id=f"T{i}", department="Tiny", join_date="2025-06-01", exit_date=None, monthly_salary=40000, overtime_hours_month=40, satisfaction_score=2, performance_rating=1))
        df = pd.DataFrame(rows)
        for c in ("join_date", "exit_date"):
            df[c] = pd.to_datetime(df[c])
        return df

    def test_small_groups_are_not_reported_and_no_person_is_listed(self):
        import pandas as pd
        from biziq import peopleiq
        r = peopleiq.run(self._hr(), None, pd.Timestamp("2026-06-30"))
        d = r.tables["departments"].set_index("department")
        assert pd.isna(d.loc["Tiny", "avg_overtime"]) and pd.isna(d.loc["Tiny", "people_with_strain_signals"]) and d.loc["Tiny", "suggested_hires"] == 0
        assert d.loc["Sales", "people_with_strain_signals"] >= 1
        assert "Tiny" not in " ".join(f.title for f in r.flags) and r.facts.get("highest_overtime_department") == "Sales"
        assert all("employee_id" not in t.columns for t in r.tables.values())

    def test_performance_rating_does_not_change_the_strain_count(self):
        import pandas as pd
        from biziq import peopleiq
        a, b = self._hr(), self._hr()
        b["performance_rating"] = 1
        ra, rb = peopleiq.run(a, None, pd.Timestamp("2026-06-30")), peopleiq.run(b, None, pd.Timestamp("2026-06-30"))
        assert ra.metrics["staff_with_strain_signals"] == rb.metrics["staff_with_strain_signals"]

    def test_appointment_module_does_not_break_results_down_by_staff(self):
        import pandas as pd
        from biziq import data_layer as d, pipeline
        ap, rep = d.clean_and_validate(d.read_upload(d.SAMPLE_DIR / "appointments.csv"), "appointments")
        assert rep.ok
        assert "staff" in ap.columns                                   # the data has staff names; the module must still not rank them
        res = run("AppointmentIQ", {"appointments": ap})
        assert "by_staff" not in res.tables
        assert all("staff" not in f.title.lower() for f in res.flags)


def test_salary_module_names_nobody_and_lists_no_individual_salary():
    import pandas as pd
    from biziq import registry
    from biziq.registry import Context
    rows = [dict(employee_id=f"E{i}", department="Ops", monthly_salary=30000 + i * 500, join_date=pd.Timestamp("2021-01-01"), exit_date=pd.NaT, gender="female" if i % 2 else "male") for i in range(9)]
    rows.append(dict(employee_id="BOSS1", department="Ops", monthly_salary=400000, join_date=pd.Timestamp("2019-01-01"), exit_date=pd.NaT, gender="male"))
    spec = registry.get("SalaryIQ")
    res = spec.run({"hr": pd.DataFrame(rows)}, Context(as_of=pd.Timestamp("2026-06-30"), today=pd.Timestamp("2026-07-01"), settings={}))
    t = res.tables["pay_outliers_by_department"]
    assert list(t["salaries_far_from_department_median"]) == [1] and list(t["people_in_department"]) == [10]
    assert res.facts["pay_outliers_for_review"] == 1
    for tb in res.tables.values():
        assert not ({"employee_id", "monthly_salary"} & set(tb.columns)), tb.columns
    blob = " ".join(f.title + f.detail for f in res.flags) + " ".join(res.notes) + " ".join(spec.recommend(res.facts, res.flags))
    assert "BOSS1" not in blob and "400000" not in blob
