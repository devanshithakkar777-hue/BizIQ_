"""BizIQ test suite. Run:  python -m pytest -q"""
import math

import numpy as np
import pandas as pd
import pytest

from biziq import (automation, cashflowiq, clientiq, data_layer, decision, pipeline, reportiq, reports,
                   riskiq, salesiq, supplychainiq, taxiq, wealthiq)
from biziq.core import safe_div, score_between


# ----------------------------------------------------------------------------- data layer
class TestDataLayer:
    def test_missing_required_column_is_a_clear_error(self):
        _, rep = data_layer.clean_and_validate(pd.DataFrame({"product": ["a"]}), "inventory")
        assert not rep.ok and "stock_on_hand" in rep.errors[0]

    def test_cleaning_dupes_negatives_text_numbers_and_dates(self):
        good = [dict(order_id=f"g{i}", date="2026-01-05", customer_id="c", product="Cap", region="S",
                     quantity="2", unit_price="10") for i in range(10)]
        first = dict(order_id="1", date="2026-01-01", customer_id="c1", product=" Tee ", region="N",
                     quantity="1,000", unit_price="Rs 50")
        rows = good + [first, dict(first),                                              # exact duplicate
                       dict(order_id="2", date="not a date", customer_id="c2", product="Cap", region="S",
                            quantity="2", unit_price="10"),                              # unparseable date
                       dict(order_id="3", date="01/02/2026", customer_id="c3", product="Cap", region="S",
                            quantity="-3", unit_price="10")]                             # negative quantity
        df, rep = data_layer.clean_and_validate(pd.DataFrame(rows).rename(columns={
            "order_id": "Order ID", "customer_id": "Customer-ID", "unit_price": "Unit Price"}), "sales")
        assert rep.ok
        # 10 good + 2 identical lines (KEPT: two identical sale lines can be two real sales, so only warned about);
        # the bad date and the negative quantity are set aside and listed in rep.rejects
        assert len(df) == 12
        r = df[df["order_id"] == "1"].iloc[0]
        assert r["product"] == "Tee" and r["quantity"] == 1000 and r["unit_price"] == 50 and r["revenue"] == 50_000
        assert any("identical" in w for w in rep.warnings)
        assert set(rep.rejects["reason"]) == {"blank required field: date", "negative quantity"} and rep.rows_rejected == 2
        assert rep.rows_in == 14 and rep.rows_out == 12

    def test_exact_duplicates_are_dropped_where_rows_have_ids(self):
        row = dict(invoice_id="I1", party="A", type="receivable", invoice_date="2026-01-01", due_date="2026-02-01", amount=10)
        df, rep = data_layer.clean_and_validate(pd.DataFrame([row, dict(row)]), "invoices")
        assert len(df) == 1 and rep.rejects["reason"].iloc[0].startswith("exact duplicate")

    def test_mostly_blank_required_column_blocks_load(self):
        df = pd.DataFrame({"product": ["a", "b", "c"], "stock_on_hand": [None, None, 5], "lead_time_days": [1, 2, 3]})
        _, rep = data_layer.clean_and_validate(df, "inventory")
        assert not rep.ok

    def test_unbalanced_balance_sheet_warns(self, sample):
        f = sample["financials"].copy()
        f.loc[5, "equity"] += 5_000_000
        _, rep = data_layer.clean_and_validate(f, "financials")
        assert any("balance" in w.lower() for w in rep.warnings)

    def test_duplicate_months_rejected(self, sample):
        f = pd.concat([sample["financials"], sample["financials"].tail(1)])
        _, rep = data_layer.clean_and_validate(f.drop_duplicates(keep="first").assign(revenue=lambda d: d["revenue"]), "financials")
        # exact duplicate rows are dropped as cleaning; a conflicting duplicate month must error
        g = sample["financials"].copy()
        row = g.tail(1).copy()
        row["revenue"] += 1
        _, rep2 = data_layer.clean_and_validate(pd.concat([g, row]), "financials")
        assert not rep2.ok

    def test_sqlite_roundtrip(self, tmp_db, sample):
        rep = data_layer.ingest(sample["inventory"], "inventory")
        assert rep.ok
        back = data_layer.load_dataset("inventory")
        assert len(back) == len(sample["inventory"])
        assert data_layer.dataset_meta().iloc[0]["dataset"] == "inventory"


# ----------------------------------------------------------------------------- helpers
def test_score_between_directions_and_clipping():
    assert score_between(0.5, 0, 1) == 50
    assert score_between(2, 3, 1) == 50           # lower-is-better
    assert score_between(10, 0, 1) == 100 and score_between(-5, 0, 1) == 0
    assert math.isnan(score_between(None, 0, 1))
    assert math.isnan(safe_div(1, 0))


# ----------------------------------------------------------------------------- WealthIQ
class TestWealthIQ:
    def test_known_ratios_on_a_tiny_company(self):
        n = 12
        fin = pd.DataFrame({
            "month": pd.date_range("2025-11-01", periods=n, freq="MS"), "revenue": 1000.0, "cogs": 600.0,
            "operating_expenses": 200.0, "depreciation": 0.0, "interest_expense": 10.0, "tax": 25.0, "net_income": 165.0,
            "cash": 500.0, "receivables": 300.0, "inventory": 200.0, "other_current_assets": 0.0, "fixed_assets": 1000.0,
            "total_assets": 2000.0, "payables": 100.0, "short_term_debt": 100.0, "current_liabilities": 400.0,
            "long_term_debt": 400.0, "total_liabilities": 800.0, "retained_earnings": 500.0, "equity": 1200.0,
            "operating_cash_flow": 150.0})
        r = wealthiq.compute_ratios(fin)
        assert r["current_ratio"] == pytest.approx(1000 / 400)
        assert r["gross_margin"] == pytest.approx(0.40)
        assert r["debt_to_equity"] == pytest.approx(500 / 1200)
        assert r["operating_margin"] == pytest.approx(0.20)
        assert r["interest_coverage"] == pytest.approx(200 * 12 / (10 * 12))
        assert r["dio"] == pytest.approx(200 / (600 * 12) * 365)
        score, _ = wealthiq.health_score(r)
        assert 0 <= score <= 100

    def test_sample_company_is_flagged_not_perfect(self, pr):
        res = pr.modules["WealthIQ"]
        assert 40 < res.score < 85
        assert res.metrics["altman_zone"] in ("grey", "safe", "distress")
        assert any(f.level == "amber" for f in res.flags)

    def test_zone_boundaries(self):
        assert wealthiq.altman_zone(3.0) == "safe" and wealthiq.altman_zone(2.0) == "grey" and wealthiq.altman_zone(1.0) == "distress"


# ----------------------------------------------------------------------------- SalesIQ / ClientIQ
class TestSalesAndClients:
    def test_basket_lift(self):
        rows = []
        for i in range(40):                       # A and B always together
            rows += [dict(order_id=f"o{i}", product="A"), dict(order_id=f"o{i}", product="B")]
        for i in range(40, 80):                   # C alone
            rows.append(dict(order_id=f"o{i}", product="C"))
        p = salesiq.basket_pairs(pd.DataFrame(rows), min_support=5)
        assert len(p) == 1 and p.iloc[0]["lift"] == pytest.approx(2.0)        # (40/80)/((40/80)*(40/80))

    def test_monthly_trend_sums_to_total(self, sample):
        s = sample["sales"]
        assert salesiq.monthly_trend(s)["revenue"].sum() == pytest.approx(s["revenue"].sum())

    def test_sales_slump_detected(self, pr):
        sr = pr.modules["SalesIQ"]
        assert any("below the recent baseline" in f.title for f in sr.flags)

    def test_clientiq_outputs_are_sane(self, pr):
        cl = pr.modules["ClientIQ"]
        c = cl.tables["customers"]
        assert c["churn_prob"].between(0, 1).all() and not c["churn_prob"].isna().any()
        assert set(c["segment"]) <= set(clientiq.SEGMENT_ORDER)
        assert cl.metrics["at_risk_customers"] == int((c["churn_prob"] >= clientiq.AT_RISK_P).sum())
        assert (cl.tables["at_risk"]["churn_prob"] >= clientiq.AT_RISK_P).all()
        assert "[DRAFT" in cl.tables["drafted_emails"].iloc[0]["draft"]       # drafts are labelled for review

    def test_campaign_uplift_is_labelled_estimate(self, pr):
        assert any("ESTIMATE" in n for n in pr.modules["SalesIQ"].notes)


# ----------------------------------------------------------------------------- SupplyChainIQ
class TestSupplyChain:
    def test_reorder_point_formula(self, pr):
        inv = pr.modules["SupplyChainIQ"].tables["inventory"].iloc[0]
        ss = round(supplychainiq.Z_SERVICE * inv["demand_std"] * math.sqrt(inv["lead_time_days"]))
        assert inv["safety_stock"] == ss
        assert inv["reorder_point"] == round(inv["avg_daily_demand"] * inv["lead_time_days"] + ss)

    def test_status_detects_planted_cases(self, pr):
        inv = pr.modules["SupplyChainIQ"].tables["inventory"].set_index("product")
        assert inv.loc["Hoodie", "status"] == "Stock-out risk (critical)"
        assert inv.loc["Wrist Watch", "status"] == "Overstock"
        assert inv.loc["Classic Tee", "status"] == "OK"

    def test_abc_covers_everything_and_a_is_80_percent(self, sample):
        a = supplychainiq.abc_analysis(sample["sales"])
        assert a["share"].sum() == pytest.approx(1.0) and a.iloc[0]["abc_class"] == "A"
        assert a[a["abc_class"] == "A"]["share"].sum() >= 0.80

    def test_forecast_is_non_negative_and_follows_trend(self):
        up, _ = supplychainiq.holt_damped(np.arange(10, 22, dtype=float), 3)
        assert (up >= 0).all() and up[0] > 20
        flat, _ = supplychainiq.holt_damped(np.full(12, 50.0), 3)
        assert flat == pytest.approx([50, 50, 50], abs=1)

    def test_supplier_ranking_matches_generated_quality(self, pr):
        s = pr.modules["SupplyChainIQ"].tables["suppliers"]
        assert s.iloc[0]["supplier"] == "Aarav Textiles" and s.iloc[-1]["supplier"] == "Delta Accessories"


# ----------------------------------------------------------------------------- Extended modules
class TestExtended:
    def test_ageing_buckets(self):
        assert cashflowiq._bucket(0) == "Not due" and cashflowiq._bucket(31) == "31-60 days" and cashflowiq._bucket(120) == "90+ days"

    def test_cash_forecast_has_three_scenarios_ordered(self, pr):
        f = pr.modules["CashFlowIQ"].tables["forecast"]
        assert (f["Downside"] <= f["Base"]).all() and (f["Base"] <= f["Upside"]).all() and len(f) == 6

    def test_tax_fy_and_deadlines(self):
        s, e = taxiq.fy_bounds(pd.Timestamp("2026-10-31"))
        assert s == pd.Timestamp("2026-04-01") and e == pd.Timestamp("2027-03-31")
        d = taxiq.deadlines(pd.Timestamp("2026-11-01"), taxiq.TaxConfig())
        assert (d["obligation"].str.contains("GSTR-3B") & (d["due_date"] == pd.Timestamp("2026-11-20"))).any()
        assert d["due_date"].is_monotonic_increasing

    def test_tax_outputs_are_labelled_as_estimates(self, pr):
        assert any("ESTIMATES" in n for n in pr.modules["TaxIQ"].notes)

    def test_anomaly_detection_finds_exactly_the_planted_issues(self, pr):
        an = pr.modules["RiskIQ"].tables["anomalies"]
        assert len(an) == 2
        assert an["item"].str.contains("Operating expenses").any() and an["item"].str.contains("Sales drop").any()

    def test_credit_risk_ranks_worst_payer_first(self, pr):
        assert pr.modules["RiskIQ"].tables["customer_credit_risk"].iloc[0]["party"] == "Kapoor & Sons"

    def test_investiq_surplus_never_negative_and_allocations_sum_to_100(self, pr):
        r = pr.modules["InvestIQ"]
        assert r.metrics["surplus"] >= 0
        for name, a in __import__("biziq.investiq", fromlist=["x"]).ALLOCATIONS.items():
            assert sum(a.values()) == 100, name
        assert any("not investment advice" in n.lower() or "illustrative" in n.lower() for n in r.notes)

    def test_peopleiq_never_lists_or_scores_individuals(self, pr):
        r = pr.modules["PeopleIQ"]
        assert "employees" not in r.tables
        for t in r.tables.values():
            assert "employee_id" not in t.columns and "performance_rating" not in t.columns
        d = r.tables["departments"]
        assert d["avg_risk"].dropna().between(0, 1).all()
        assert any("Signals, not judgements" in n for n in r.notes)

    def test_sentiment_with_negation(self):
        from biziq import brandiq
        assert brandiq.sentiment("Great quality, love it") > 0
        assert brandiq.sentiment("Terrible packaging, item damaged") < 0
        assert brandiq.sentiment("not good") < 0

    def test_purchase_orders_are_drafts_only(self, pr):
        docs = pr.modules["ProcureIQ"].tables["purchase_order_documents"]
        assert len(docs) > 0 and docs["document"].str.contains("DRAFT PURCHASE ORDER").all()


# ----------------------------------------------------------------------------- pipeline
class TestPipeline:
    def test_all_modules_run_on_sample(self, pr):
        from biziq import registry
        expected = pipeline.MODULE_ORDER + [s.name for s in registry.all_specs()]
        assert list(pr.modules) == expected and not pr.skipped and not pr.failed
        assert not [f for f in pr.all_flags() if "could not run" in f.title]

    def test_facts_are_plain_python(self, pr):
        import json
        for m in pr.modules.values():
            json.dumps(m.facts)                      # raises on numpy scalars / NaN-unsafe types

    def test_missing_data_skips_instead_of_crashing(self, sample):
        out = pipeline.run_all({"financials": sample["financials"]})
        assert "WealthIQ" in out.modules and "SalesIQ" in out.skipped and "SupplyChainIQ" in out.skipped

    def test_one_failing_module_does_not_stop_the_rest(self, sample, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("kaboom")
        monkeypatch.setattr(wealthiq, "run", boom)
        out = pipeline.run_all(sample)
        assert "WealthIQ" in out.failed and "WealthIQ" not in out.modules
        assert any("could not run" in f.title and f.level == "red" for f in out.all_flags())
        assert "SalesIQ" in out.modules and "CashFlowIQ" in out.modules

    def test_failed_client_module_does_not_crash_summaries(self, sample, monkeypatch):
        """Regression: with scikit-learn missing, ClientIQ fails; summaries and recommendations must still work."""
        def boom(*a, **k):
            raise ModuleNotFoundError("No module named 'sklearn'")
        monkeypatch.setattr(clientiq, "run", boom)
        out = pipeline.run_all(sample)
        assert "ClientIQ" in out.failed
        ex = reportiq.executive_summary(out)
        assert all(k[0] != "Revenue at churn risk" for k in ex["kpis"])
        assert reportiq.board_narrative(out)
        assert "ClientIQ" not in decision.get_recommendations(out)


# ----------------------------------------------------------------------------- decision layer
class TestDecisionLayer:
    def test_number_guard_accepts_only_traceable_numbers(self):
        src = "cash Rs 53.35 L buffer Rs 19.35 L 40.4 percent"
        assert decision.number_guard("Cash is Rs 53.35 L, buffer Rs 19.35 L.", src)[0]
        ok, bad = decision.number_guard("Cash will hit Rs 60 L.", src)
        assert not ok and bad == [60.0]
        assert decision.number_guard("1. Collect the 40.4 percent overdue.\n2. Review in 2026.", src)[0]   # list numbers and years ignored

    def test_template_numbers_all_trace_to_facts(self, pr):
        for name, res in pr.modules.items():
            allowed = decision._allowed_source(res.facts, res.flags)
            for item in decision.template_recommendations(name, res):
                ok, bad = decision.number_guard(item, allowed)
                assert ok, f"{name}: untraceable numbers {bad} in: {item}"

    def test_llm_off_without_key(self, monkeypatch, pr):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        recs = decision.get_recommendations(pr, use_llm=True)
        assert all(v["source"].startswith("template") for v in recs.values())

    def test_groq_provider_selected_and_parsed(self, monkeypatch):
        import io, json as _json
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("BIZIQ_PROVIDER", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        assert decision.llm_provider() == "groq" and decision.llm_available()
        seen = {}

        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_open(req, timeout=0):
            seen["auth"] = req.get_header("Authorization")
            seen["body"] = _json.loads(req.data)
            return R(_json.dumps({"choices": [{"message": {"content": " ok text "}}]}).encode())
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", fake_open)
        assert decision._call_llm("hi") == "ok text"
        assert seen["auth"] == "Bearer gsk_test" and seen["body"]["messages"][1]["content"] == "hi"

    def test_llm_output_with_invented_number_is_rejected(self, monkeypatch, pr):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setattr(decision, "llm_available", lambda: True)
        monkeypatch.setattr(decision, "_call_llm", lambda *a, **k: "1. Cash will triple to Rs 999 Cr next month.")
        items, status = decision.llm_recommendations("CashFlowIQ", pr.modules["CashFlowIQ"])
        assert items is None and "rejected" in status

    def test_llm_output_with_verified_numbers_is_used(self, monkeypatch, pr):
        res = pr.modules["CashFlowIQ"]
        good = f"1. Keep cash above the {res.facts['minimum_cash_buffer']} buffer and chase overdue receivables worth {res.facts['receivables_overdue']}."
        monkeypatch.setattr(decision, "llm_available", lambda: True)
        monkeypatch.setattr(decision, "_call_llm", lambda *a, **k: good)
        items, status = decision.llm_recommendations("CashFlowIQ", res)
        assert items and "verified" in status


# ----------------------------------------------------------------------------- ReportIQ and PDF
class TestReporting:
    @pytest.mark.parametrize("q,module", [("How is cash?", "CashFlowIQ"), ("Which products need reordering?", "SupplyChainIQ"),
                                          ("Which customers might churn?", "ClientIQ"), ("when is the GST deadline", "TaxIQ"),
                                          ("any anomalies or fraud", "RiskIQ"), ("how is attrition", "PeopleIQ")])
    def test_question_routing(self, pr, q, module):
        assert reportiq.ask(q, pr)["module"] == module

    def test_unknown_question_gets_help_not_a_guess(self, pr):
        a = reportiq.ask("tell me a joke", pr)
        assert a["module"] is None and "I can answer" in a["answer"]

    @pytest.mark.parametrize("kind", ["weekly", "board"])
    def test_pdf_is_created(self, pr, tmp_path, kind):
        p = reports.build_report(pr, tmp_path / f"{kind}.pdf", kind=kind, data_note="synthetic")
        data = p.read_bytes()
        assert data[:5] == b"%PDF-" and len(data) > 8000


# ----------------------------------------------------------------------------- automation
class TestAutomation:
    def test_alerts_are_deduplicated_between_runs(self, tmp_db, pr):
        alerts = automation.build_alerts(pr)
        assert len(alerts) == len({a.key for a in alerts}) > 10
        first = automation.sync_alerts(alerts)
        automation.mark_notified(first)
        assert len(first) == len(alerts)
        assert automation.sync_alerts(alerts) == []              # same alerts, within cooldown: nothing new

    def test_cleared_alerts_resolve_and_renotify_if_they_return(self, tmp_db, pr):
        alerts = automation.build_alerts(pr)
        automation.mark_notified(automation.sync_alerts(alerts))
        automation.sync_alerts(alerts[1:])
        h = automation.alert_history()
        assert (h["status"] == "resolved").sum() == 1
        back = automation.sync_alerts(alerts)
        assert [a.key for a in back] == [alerts[0].key]

    def test_email_defaults_to_dry_run_and_never_sends(self, tmp_db, monkeypatch):
        for k in ("BIZIQ_SEND_EMAIL", "BIZIQ_SMTP_HOST"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setattr("smtplib.SMTP", lambda *a, **k: (_ for _ in ()).throw(AssertionError("SMTP must not be used")))
        msg = automation.send_email("Test subject", "body")
        assert msg.startswith("DRY RUN")
        assert len(list(automation.OUTBOX.glob("*.eml"))) == 1

    def test_live_email_needs_explicit_opt_in(self, monkeypatch):
        monkeypatch.setenv("BIZIQ_SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("BIZIQ_EMAIL_FROM", "a@b.c")
        monkeypatch.setenv("BIZIQ_EMAIL_TO", "d@e.f")
        monkeypatch.delenv("BIZIQ_SEND_EMAIL", raising=False)
        assert not automation.EmailConfig.from_env().can_send()
        monkeypatch.setenv("BIZIQ_SEND_EMAIL", "1")
        assert automation.EmailConfig.from_env().can_send()

    def test_jobs_run_and_are_logged(self, tmp_db, sample):
        info = automation.weekly_report_job(data=sample, send=True)
        assert info["pdf"].endswith(".pdf") and info["email"].startswith("DRY RUN")
        automation.followup_drafts_job(data=sample)
        automation.purchase_order_job(data=sample)
        log = automation.run_log()
        assert set(log["job"]) >= {"weekly_report", "followup_drafts", "purchase_order_drafts"} and (log["status"] == "ok").all()

    def test_measurement_only_reports_measured_numbers(self, tmp_db):
        t = automation.load_tasks()
        s = automation.measurement_summary(t)
        assert s["time_reduction"] is None and s["measured_tasks"] == 0           # nothing invented
        t.loc[0, ["manual_minutes", "biziq_minutes"]] = [60, 15]
        t.loc[1, ["manual_minutes", "biziq_minutes"]] = [40, 10]
        s = automation.measurement_summary(t)
        assert s["time_reduction"] == pytest.approx(0.75) and s["measured_tasks"] == 2
