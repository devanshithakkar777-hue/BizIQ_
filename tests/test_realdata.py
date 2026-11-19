"""Behaviour on messy, real-world business files and repeat loads."""
import io

import pandas as pd
import pytest

from biziq import data_layer as d
from biziq import ingest, mapping

SALES_COLS = ["order_id", "date", "customer_id", "product", "region", "quantity", "unit_price"]


def sales_csv(rows, header="order_id,date,customer_id,product,region,quantity,unit_price"):
    return io.BytesIO((header + "\n" + "\n".join(rows) + "\n").encode())


def load(buf, dataset="sales", name="x.csv", **kw):
    return d.clean_and_validate(d.read_upload(buf, name, dataset=dataset), dataset, **kw)


@pytest.fixture()
def db(tmp_path):
    d.use_db(tmp_path / "t.db")
    yield
    d.use_db(None)


# ---------------------------------------------------------------- dates (the silent-corruption risk)
class TestDates:
    def test_indian_day_first_dates_are_read_correctly(self):
        df, rep = load(sales_csv(["A1,03/04/2026,C1,Tee,MH,1,100", "A2,25/04/2026,C1,Tee,MH,1,100", "A3,04/05/2026,C2,Tee,MH,1,100"]))
        assert [x.strftime("%Y-%m-%d") for x in df["date"]] == ["2026-04-03", "2026-04-25", "2026-05-04"]
        assert rep.date_order == "dmy"

    def test_ambiguous_dates_are_flagged_not_silent(self):
        df, rep = load(sales_csv(["A1,03/04/2026,C1,Tee,MH,1,100", "A2,04/05/2026,C1,Tee,MH,1,100"]))
        assert df["date"].iloc[0] == pd.Timestamp("2026-04-03")
        assert any("two ways" in w for w in rep.warnings)

    def test_month_first_file_detected_from_values(self):
        df, rep = load(sales_csv(["A1,04/25/2026,C1,Tee,MH,1,100", "A2,03/04/2026,C1,Tee,MH,1,100"]))
        assert df["date"].iloc[0] == pd.Timestamp("2026-04-25") and df["date"].iloc[1] == pd.Timestamp("2026-03-04")

    @pytest.mark.parametrize("raw,exp", [("20260403", "2026-04-03"), ("46115", "2026-04-03"), ("01-Apr-2026", "2026-04-01"),
                                         ("2026-4-5 10:30", "2026-04-05"), ("April 3, 2026", "2026-04-03")])
    def test_other_formats(self, raw, exp):
        out, info = ingest.parse_dates(pd.Series([raw]))
        assert out.iloc[0].strftime("%Y-%m-%d") == exp and not info["bad"]

    def test_bad_and_implausible_dates_reported(self):
        rows = [f"A{i},2026-01-05,C,Tee,MH,1,1" for i in range(10)] + ["B1,2026-02-30,C,Tee,MH,1,1", "B2,1975-01-01,C,Tee,MH,1,1"]
        df, rep = load(sales_csv(rows))
        assert any("not valid dates" in w for w in rep.warnings) and any("before 1990" in w for w in rep.warnings)


# ---------------------------------------------------------------- numbers
class TestNumbers:
    def test_indian_and_accounting_formats(self):
        vals = ["1,23,456.00", "(5,000)", "12,000 Dr", "Rs. 2,500.50", "₹ 99", "1,000-", "12"]
        out, info = ingest.parse_numbers(pd.Series(vals))
        assert list(out) == [123456.0, -5000.0, 12000.0, 2500.5, 99.0, -1000.0, 12.0] and info["drcr"] == 1

    def test_placeholders_are_blank_not_errors(self):
        out, info = ingest.parse_numbers(pd.Series(["-", "N/A", "nil", "", "abc"]))
        assert out.isna().all() and info["bad"] == 1 and info["examples"] == ["abc"]


# ---------------------------------------------------------------- file shapes
class TestFileShapes:
    def test_title_rows_and_total_row(self):
        raw = ("Sales Register,,,,,,\nAcme Traders Pvt Ltd,,,,,,\nPeriod: 01-Apr-2026 to 30-Apr-2026,,,,,,\n,,,,,,\n"
               "Order No.,Date,Party Name,Item,State,Qty,Rate\nA1,03/04/2026,C1,Tee,MH,2,100\nA2,04/04/2026,C2,Cap,KA,1,50\nGrand Total,,,,,3,150\n")
        df, rep = load(io.BytesIO(raw.encode()))
        assert rep.ok and len(df) == 2 and set(df["order_id"]) == {"A1", "A2"}
        assert any("title row" in c for c in rep.cleaning) and any("total" in c.lower() for c in rep.cleaning)
        assert rep.rejects.empty and df["revenue"].sum() == 250

    def test_windows_1252_and_semicolons(self):
        raw = "order_id;date;customer_id;product;region;quantity;unit_price\nA1;2026-04-03;Café Ünion;Tee;MH;1;100\n".encode("cp1252")
        df, rep = load(io.BytesIO(raw))
        assert df["customer_id"].iloc[0] == "Café Ünion" and any("Windows-1252" in c for c in rep.cleaning)

    def test_utf16_with_bom(self):
        raw = "order_id,date,customer_id,product,region,quantity,unit_price\nA1,2026-04-03,C1,Tee,MH,1,100\n".encode("utf-16")
        assert len(load(io.BytesIO(raw))[0]) == 1

    def test_leading_zeros_in_ids_survive(self):
        df, _ = load(sales_csv(["00123,2026-04-03,C1,Tee,MH,1,100"]))
        assert df["order_id"].iloc[0] == "00123"

    def test_excel_multi_sheet_picks_data_sheet_and_numeric_ids(self, tmp_path):
        p = tmp_path / "book.xlsx"
        with pd.ExcelWriter(p) as w:
            pd.DataFrame({"Notes": ["read me"]}).to_excel(w, sheet_name="Cover", index=False)
            title = pd.DataFrame([["Company X", None, None, None, None, None, None], [None] * 7])
            body = pd.DataFrame([["Order No", "Date", "Customer", "Item", "State", "Qty", "Rate"],
                                 [1001.0, "2026-04-03", "C1", "Tee", "MH", 2, 100], ["Total", None, None, None, None, 2, 200]])
            pd.concat([title, body]).to_excel(w, sheet_name="Sales", index=False, header=False)
        df, rep = load(open(p, "rb"), name="book.xlsx")
        assert rep.ok and df["order_id"].iloc[0] == "1001" and any("sheets" in c for c in rep.cleaning)

    def test_file_size_limit(self, monkeypatch):
        monkeypatch.setattr(ingest, "MAX_UPLOAD_MB", 0.0001)
        with pytest.raises(ValueError, match="limit"):
            d.read_upload(io.BytesIO(b"a,b\n" + b"1,2\n" * 1000), "x.csv")


# ---------------------------------------------------------------- column matching
class TestMapping:
    def test_real_world_headings_map_automatically(self):
        raw = pd.DataFrame({"Invoice No.": ["A"], "Party Name": ["x"], "Voucher Type": ["Sales"], "Invoice Date": ["2026-04-01"],
                            "Due On": ["2026-05-01"], "Amount (Rs)": ["1,000"], "Remarks": ["hi"]})
        df, rep = d.clean_and_validate(raw, "invoices")
        assert rep.ok and df["type"].iloc[0] == "receivable" and df["amount"].iloc[0] == 1000
        assert any("Matched column 'Party Name'" in c for c in rep.cleaning)

    def test_fuzzy_match_is_suggested_not_applied(self):
        raw = pd.DataFrame({"order_id": ["1"], "date": ["2026-04-01"], "customer_id": ["c"], "Item Namee": ["Tee"], "region": ["x"], "quantity": [1], "unit_price": [5]})
        df, rep = d.clean_and_validate(raw, "sales")
        assert not rep.ok and "Item Namee" in rep.errors[0] and "confirm" in rep.errors[0]
        assert rep.suggestions[0]["target"] == "product"

    def test_manual_mapping_fixes_it(self):
        raw = pd.DataFrame({"order_id": ["1"], "date": ["2026-04-01"], "customer_id": ["c"], "Item Namee": ["Tee"], "region": ["x"], "quantity": [1], "unit_price": [5]})
        df, rep = d.clean_and_validate(raw, "sales", column_map={"Item Namee": "product"})
        assert rep.ok and df["product"].iloc[0] == "Tee"

    def test_exact_columns_are_never_remapped(self):
        raw = pd.DataFrame({"revenue": [5.0], "sales": [9.0], **{c: ["1"] for c in []}})
        m, _ = mapping.auto_map(["revenue", "sales"], "financials", ["revenue", "cogs"])
        assert "sales" not in m          # revenue already present, so 'sales' must not overwrite it


# ---------------------------------------------------------------- nothing is dropped silently
class TestRejects:
    def test_every_dropped_row_is_listed_with_reason_and_source_row(self):
        raw = ("Title,,,,,,\norder_id,date,customer_id,product,region,quantity,unit_price\n"
               + "\n".join(f"A{i},2026-01-05,C,Tee,MH,1,10" for i in range(10))
               + "\nB1,2026-01-05,C,Tee,MH,-2,10\nB2,,C,Tee,MH,1,10\n")
        df, rep = load(io.BytesIO(raw.encode()))
        assert len(df) == 10 and rep.rows_rejected == 2
        r = rep.rejects.set_index("order_id")
        assert r.loc["B1", "reason"] == "negative quantity" and r.loc["B1", "source_row"] == 13      # sheet row number of the original file
        assert r.loc["B2", "reason"].startswith("blank required field")

    def test_source_row_matches_the_users_file_even_with_blank_lines(self):
        raw = "Title,,,,,,\n\norder_id,date,customer_id,product,region,quantity,unit_price\nA1,2026-01-05,C,Tee,MH,1,10\n\nB1,2026-01-05,C,Tee,MH,-1,10\n"
        _, rep = load(io.BytesIO(raw.encode()))
        assert int(rep.rejects["source_row"].iloc[0]) == 6          # B1 really is on line 6 of the file

    def test_same_invoice_twice_keeps_later_version(self):
        rows = [dict(invoice_id="I1", party="A", type="receivable", invoice_date="2026-01-01", due_date="2026-02-01", amount=10, paid_date=None),
                dict(invoice_id="I1", party="A", type="receivable", invoice_date="2026-01-01", due_date="2026-02-01", amount=10, paid_date="2026-01-20")]
        df, rep = d.clean_and_validate(pd.DataFrame(rows), "invoices")
        assert len(df) == 1 and pd.notna(df["paid_date"].iloc[0]) and any("repeat" in w for w in rep.warnings)

    def test_names_written_two_ways_are_flagged(self):
        rows = [f"A{i},2026-01-05,{n},Tee,MH,1,10" for i, n in enumerate(["Acme Traders", "ACME  TRADERS", "Beta"] * 4)]
        _, rep = load(sales_csv(rows))
        assert any("written more than one way" in w for w in rep.warnings)


# ---------------------------------------------------------------- repeat loads
def inv(i, paid=None, amount=100):
    return dict(invoice_id=i, party="A", type="receivable", invoice_date="2026-01-01", due_date="2026-02-01", amount=amount, paid_date=paid)


class TestMergeAndUndo:
    def test_invoice_upsert_updates_status_without_duplicating(self, db):
        d.ingest(pd.DataFrame([inv("I1"), inv("I2")]), "invoices")
        r = d.ingest(pd.DataFrame([inv("I1", paid="2026-01-20"), inv("I2"), inv("I3")]), "invoices", mode="merge")
        s = r.save
        assert (s.added, s.updated, s.unchanged, s.total) == (1, 1, 1, 3)
        got = d.load_dataset("invoices").set_index("invoice_id")
        assert pd.notna(got.loc["I1", "paid_date"]) and len(got) == 3

    def test_history_survives_a_merge(self, db):
        d.ingest(pd.DataFrame([inv("OLD")]), "invoices")
        d.ingest(pd.DataFrame([inv("NEW")]), "invoices", mode="merge")
        assert set(d.load_dataset("invoices")["invoice_id"]) == {"OLD", "NEW"}

    def test_sales_append_is_idempotent_and_respects_repeated_lines(self, db):
        a = sales_csv(["A1,2026-04-03,C1,Tee,MH,1,100", "A1,2026-04-03,C1,Tee,MH,1,100", "A2,2026-04-04,C1,Cap,MH,1,50"])
        first = d.ingest(d.read_upload(a, "a.csv", dataset="sales"), "sales")
        assert first.save.total == 3
        again = d.ingest(d.read_upload(sales_csv(["A1,2026-04-03,C1,Tee,MH,1,100", "A1,2026-04-03,C1,Tee,MH,1,100", "A2,2026-04-04,C1,Cap,MH,1,50"]), "a.csv", dataset="sales"),
                         "sales", mode="merge")
        assert again.save.added == 0 and again.save.unchanged == 3 and len(d.load_dataset("sales")) == 3     # re-uploading changes nothing
        more = d.ingest(d.read_upload(sales_csv(["A1,2026-04-03,C1,Tee,MH,1,100", "A1,2026-04-03,C1,Tee,MH,1,100", "A1,2026-04-03,C1,Tee,MH,1,100", "A9,2026-05-01,C2,Cap,MH,1,50"]),
                                      "b.csv", dataset="sales"), "sales", mode="merge")
        assert more.save.added == 2 and len(d.load_dataset("sales")) == 5                                      # one extra A1 line and A9

    def test_undo_last_load_and_redo(self, db):
        d.ingest(pd.DataFrame([inv("I1")]), "invoices")
        d.ingest(pd.DataFrame([inv("I9"), inv("I8")]), "invoices")
        assert d.has_previous("invoices") and len(d.load_dataset("invoices")) == 2
        d.restore_previous("invoices")
        assert list(d.load_dataset("invoices")["invoice_id"]) == ["I1"] and d.dataset_meta().set_index("dataset").loc["invoices", "rows"] == 1
        d.restore_previous("invoices")                                                                           # redo
        assert len(d.load_dataset("invoices")) == 2

    def test_undo_without_history_is_a_clear_error(self, db):
        with pytest.raises(ValueError, match="no earlier version"):
            d.restore_previous("invoices")

    def test_connections_are_closed_after_use(self, db):
        import gc, warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            for _ in range(30):
                d.dataset_meta()
            gc.collect()


class TestDataHealth:
    def test_flags_stale_and_missing(self, db):
        old = pd.DataFrame([dict(order_id="A", date="2025-01-01", customer_id="c", product="p", region="r", quantity=1, unit_price=1)])
        d.ingest(old, "sales")
        h = d.data_health(pd.Timestamp("2026-11-18")).set_index("dataset")
        assert h.loc["sales", "status"].startswith("stale") and h.loc["invoices", "status"] == "not loaded"


class TestDataDir:
    def test_data_dir_env_moves_everything(self, tmp_path):
        import subprocess, sys
        code = "import sys; sys.path.insert(0,'.'); from biziq import data_layer, platform_db, automation; print(data_layer.DB_PATH, platform_db.OUT, automation.OUT)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"BIZIQ_DATA_DIR": str(tmp_path), "PATH": "/usr/bin:/usr/local/bin"},
                             cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent)).stdout.split()
        assert out == [str(tmp_path / "biziq.db"), str(tmp_path), str(tmp_path)]


# ---------------------------------------------------------------- UI helpers and page
class TestUI:
    def test_csv_download_cannot_run_formulas(self):
        from biziq.ui import safe_csv
        df = pd.DataFrame({"party": ["=HYPERLINK(\"http://x\")", "+1+1", "@SUM(A1)", "-cmd", "Acme"], "qty": ["-2", "3", "4", "5", "6"], "n": [1, 2, 3, 4, 5]})
        import csv
        cells = [r[0] for r in list(csv.reader(io.StringIO(safe_csv(df))))[1:]]
        assert all(c.startswith(("'=", "'+", "'@", "'-", "Acme")) for c in cells)
        assert "'-2" not in safe_csv(df)                  # genuine negative numbers are left alone

    def test_data_page_loads_sample_and_shows_health(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BIZIQ_AUTH", "0")
        monkeypatch.setenv("BIZIQ_DATA_DIR", str(tmp_path))
        from streamlit.testing.v1 import AppTest
        root = __import__("pathlib").Path(__file__).resolve().parent.parent
        at = AppTest.from_file(str(root / "app.py"), default_timeout=120).run()
        at.sidebar.radio[0].set_value("Data").run()
        next(b for b in at.button if b.label.startswith("Load synthetic")).click().run()
        assert not at.exception
        assert "Data health" in [x.value for x in at.subheader]


# ---------------------------------------------------------------- degenerate data must never crash a module
class TestDegenerateData:
    HONEST = ("needs ",)      # clear, user-readable reasons

    @staticmethod
    def _base():
        out = {}
        for n in d.SCHEMAS:
            f = d.SAMPLE_DIR / f"{n}.csv"
            if f.exists():
                df, rep = d.clean_and_validate(d.read_upload(f, f.name, dataset=n), n)
                if rep.ok:
                    out[n] = df
        return out

    def test_empty_tables_are_treated_as_missing(self):
        from biziq import pipeline
        base = self._base()
        for ds in ("sales", "invoices", "financials", "quotes"):
            data = dict(base)
            data[ds] = base[ds].head(0)
            res = pipeline.run_all(data)                                      # must not raise
            assert not [f for f in res.failed.values() if not any(h in f.title for h in self.HONEST)]

    @pytest.mark.slow
    def test_modules_survive_tiny_constant_zero_and_empty_tables(self):
        import numpy as np
        from biziq import pipeline
        base = self._base()
        bad = []
        for ds, df in base.items():
            num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            zero = df.copy()
            zero[num] = 0.0
            const = df.copy()
            for c in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(const[c]):
                    const[c] = const[c].iloc[0]
            for label, v in (("one_row", df.head(1)), ("two_rows", df.head(2)), ("empty", df.head(0)), ("all_zero", zero), ("constant", const)):
                data = dict(base)
                data[ds] = v.reset_index(drop=True)
                res = pipeline.run_all(data)                                  # must not raise
                for m, flag in res.failed.items():
                    if not any(h in flag.title for h in self.HONEST):
                        bad.append(f"{ds}/{label}: {flag.title[:120]}")
        assert not bad, "\n".join(bad)


# ---------------------------------------------------------------- operations
@pytest.mark.skipif(__import__("biziq.db", fromlist=["x"]).IS_PG, reason="file-copy backup is the SQLite path; PostgreSQL uses pg_dump (tested separately)")
class TestBackup:
    def _load(self):
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location("backup", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "backup.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_backup_then_restore_roundtrip(self, tmp_path):
        b = self._load()
        data = tmp_path / "data"
        d.use_db(data / "orgs" / "o1" / "biziq.db")
        d.ingest(pd.DataFrame([inv("I1"), inv("I2")]), "invoices")
        d.use_db(None)
        out = b.make_backup(data, tmp_path / "bk", keep=5)
        assert (out / "orgs" / "o1" / "biziq.db").exists()
        d.use_db(data / "orgs" / "o1" / "biziq.db")
        d.ingest(pd.DataFrame([inv("X")]), "invoices")                     # change the live data after the backup
        assert list(d.load_dataset("invoices")["invoice_id"]) == ["X"]
        d.use_db(None)
        b.restore(data, tmp_path / "bk", out.name)
        d.use_db(data / "orgs" / "o1" / "biziq.db")
        assert sorted(d.load_dataset("invoices")["invoice_id"]) == ["I1", "I2"]
        d.use_db(None)
        assert list((data / "backups").glob("pre-restore-*"))               # what was there before is kept

    def test_retention_keeps_newest(self, tmp_path):
        b = self._load()
        dest = tmp_path / "bk"
        for s in ("20260101-000000", "20260102-000000", "20260103-000000"):
            (dest / s).mkdir(parents=True)
        b.prune(dest, 2)
        assert sorted(p.name for p in dest.iterdir()) == ["20260102-000000", "20260103-000000"]

    def test_restore_needs_confirmation(self, tmp_path, capsys):
        b = self._load()
        assert b.main(["--data", str(tmp_path), "--restore", "20260101-000000"]) == 2


# ---------------------------------------------------------------- PostgreSQL specifics (need a server: set BIZIQ_TEST_PG_URL)
PG_URL = __import__("os").environ.get("BIZIQ_TEST_PG_URL", "")
needs_pg = pytest.mark.skipif(not PG_URL, reason="set BIZIQ_TEST_PG_URL=postgresql://user@host/db to run PostgreSQL tests")


def _fresh_pg_database() -> str:
    """Create an empty database on the test server and return its URL."""
    import uuid
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    u = make_url(PG_URL.replace("postgresql://", "postgresql+psycopg://", 1))
    name = "t_" + uuid.uuid4().hex[:10]
    eng = create_engine(u, isolation_level="AUTOCOMMIT")
    with eng.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"'))
    eng.dispose()
    return str(u.set(database=name)).replace("postgresql+psycopg://", "postgresql://", 1).replace("***", u.password or "")


def _run(code: str, env: dict, cwd=None):
    import subprocess, sys, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    e = {"PATH": "/usr/bin:/usr/local/bin", "HOME": "/tmp", **env}
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r)\n" % str(root) + code], capture_output=True, text=True, env=e, cwd=cwd or str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@needs_pg
class TestPostgres:
    def test_org_data_is_in_separate_schemas_and_cannot_cross(self):
        url = _fresh_pg_database()
        out = _run('''
import pandas as pd
from biziq import accounts, data_layer as d, db
a = accounts.create_org("Alpha Co", "a@alpha.test", "Ann", "Passw0rdOK!x")
b = accounts.create_org("Beta Co", "b@beta.test", "Bob", "Passw0rdOK!x")
inv = lambda i: dict(invoice_id=i, party="P", type="receivable", invoice_date="2026-01-01", due_date="2026-02-01", amount=5, paid_date=None)
accounts.activate_org(a.org_id); d.ingest(pd.DataFrame([inv("A1"), inv("A2")]), "invoices")
accounts.activate_org(b.org_id); d.ingest(pd.DataFrame([inv("B1")]), "invoices")
accounts.activate_org(a.org_id); ids_a = sorted(d.load_dataset("invoices")["invoice_id"])
accounts.activate_org(b.org_id); ids_b = sorted(d.load_dataset("invoices")["invoice_id"])
with db.pg_connect("platform") as c:
    schemas = sorted(r[0] for r in c.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'org_%'").fetchall())
print(ids_a, ids_b, len(schemas))
''', {"BIZIQ_DATABASE_URL": url, "BIZIQ_DATA_DIR": "/tmp/pgt_" + url[-10:]})
        assert out.strip() == "['A1', 'A2'] ['B1'] 2"

    def test_concurrent_audit_writers_keep_one_unbroken_chain(self):
        url = _fresh_pg_database()
        out = _run('''
import threading
from biziq import accounts, audit
u = accounts.create_org("Gamma", "g@gamma.test", "Gus", "Passw0rdOK!x")
def work(k):
    for i in range(15): audit.log(u.org_id, u.email, "t.%d" % k, "x", str(i))
ts = [threading.Thread(target=work, args=(k,)) for k in range(6)]
[t.start() for t in ts]; [t.join() for t in ts]
ok, msg = audit.verify_chain(u.org_id); print(ok, msg)
''', {"BIZIQ_DATABASE_URL": url, "BIZIQ_DATA_DIR": "/tmp/pgt_" + url[-10:]})
        assert out.strip().startswith("True Audit log intact (91 entries)")           # 90 writes + org.create

    def test_audit_log_cannot_be_truncated_updated_or_deleted(self):
        url = _fresh_pg_database()
        out = _run('''
from biziq import accounts, audit, platform_db
u = accounts.create_org("Delta", "d@delta.test", "Dee", "Passw0rdOK!x")
res = []
for stmt in ("UPDATE audit_log SET detail='x'", "DELETE FROM audit_log", "TRUNCATE audit_log"):
    c = platform_db.connect()
    try: c.execute(stmt); res.append("ALLOWED")
    except Exception as e: res.append("blocked" if "append-only" in str(e) else "other:" + str(e)[:80])
    finally: c.close()
print(res)
''', {"BIZIQ_DATABASE_URL": url, "BIZIQ_DATA_DIR": "/tmp/pgt_" + url[-10:]})
        assert out.strip() == "['blocked', 'blocked', 'blocked']"

    def test_migrate_sqlite_install_to_postgres(self, tmp_path):
        build = '''
import pandas as pd
from biziq import accounts, approvals, audit, data_layer as d
u = accounts.create_org("Migrate Me", "m@mig.test", "Mia", "Passw0rdOK!x")
accounts.activate_org(u.org_id)
for n in ("sales", "invoices", "financials"):
    f = d.SAMPLE_DIR / (n + ".csv"); d.ingest(d.read_upload(f, f.name, dataset=n), n)
d.ingest(d.read_upload(d.SAMPLE_DIR / "invoices.csv", "i.csv", dataset="invoices"), "invoices")          # second load makes a _prev_ table
approvals.request(u, "email", "Chase overdue", {"n": 3})
print(u.org_id)
'''
        org = _run(build, {"BIZIQ_DATA_DIR": str(tmp_path)}).strip().splitlines()[-1]
        url = _fresh_pg_database()
        import pathlib
        script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "migrate_to_postgres.py"
        out = _run(f"import sys; sys.argv=['m','--url',{url!r},'--sqlite-dir',{str(tmp_path)!r}]; import runpy; runpy.run_path({str(script)!r}, run_name='__main__')", {})
        assert "Migration complete" in out and "PROBLEMS" not in out
        check = _run(f'''
import pandas as pd
from biziq import accounts, audit, approvals, data_layer as d
accounts.activate_org({org!r})
print(len(d.load_dataset("sales")), len(d.load_dataset("invoices")), d.has_previous("invoices"), audit.verify_chain({org!r})[0])
u = accounts.authenticate("m@mig.test", "Passw0rdOK!x")
print(u.role, len(approvals.list_requests(u)))
''', {"BIZIQ_DATABASE_URL": url, "BIZIQ_DATA_DIR": str(tmp_path / "unused")})
        lines = check.strip().splitlines()
        assert lines[0].endswith("True True") and lines[1] == "owner 1"

    def test_bulk_copy_handles_nulls_text_ids_and_dates(self):
        url = _fresh_pg_database()
        out = _run('''
import pandas as pd, numpy as np
from biziq import data_layer as d
df = pd.DataFrame({"order_id": ["00123", "00124", "x,y"], "date": pd.to_datetime(["2026-01-05", "2026-01-06", None]),
                   "customer_id": ['He said "hi"', "multi\\nline", None], "product": ["a", "b", "c"], "region": ["N", "S", "E"],
                   "quantity": [1.0, np.nan, 3.0], "unit_price": [9.5, 10.0, 11.0]})
d.save_dataset(df, "sales")
g = d.load_dataset("sales")
print(list(g.order_id), g.customer_id.fillna('NULL').tolist(), g.quantity.isna().tolist(), str(g.date.iloc[0].date()), g.date.isna().tolist())
''', {"BIZIQ_DATABASE_URL": url, "BIZIQ_DATA_DIR": "/tmp/pgt_" + url[-10:]})
        assert out.strip() == "['00123', '00124', 'x,y'] ['He said \"hi\"', 'multi\\nline', 'NULL'] [False, True, False] 2026-01-05 [False, False, True]"


@needs_pg
def test_pg_dump_backup_and_restore_roundtrip(tmp_path):
    url = _fresh_pg_database()
    env = {"BIZIQ_DATABASE_URL": url, "BIZIQ_DATA_DIR": str(tmp_path)}
    _run('''
import pandas as pd
from biziq import data_layer as d
d.ingest(pd.DataFrame([dict(invoice_id=i, party="P", type="receivable", invoice_date="2026-01-01", due_date="2026-02-01", amount=5, paid_date=None) for i in ("I1", "I2")]), "invoices")
''', env)
    import pathlib
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "backup.py"
    run = lambda args: _run(f"import sys, runpy; sys.argv=['b']+{args!r}; sys.exit(runpy.run_path({str(script)!r}, run_name='__main__') or 0)", env)
    out = run(["--data", str(tmp_path), "--dest", str(tmp_path / "bk")])
    stamp = out.strip().rsplit("/", 1)[-1]
    _run('''
import pandas as pd
from biziq import data_layer as d
d.ingest(pd.DataFrame([dict(invoice_id="ZZ", party="P", type="receivable", invoice_date="2026-01-01", due_date="2026-02-01", amount=9, paid_date=None)]), "invoices")
''', env)
    run(["--data", str(tmp_path), "--dest", str(tmp_path / "bk"), "--restore", stamp, "--yes"])
    got = _run("from biziq import data_layer as d; print(sorted(d.load_dataset('invoices').invoice_id))", env)
    assert got.strip() == "['I1', 'I2']"


@needs_pg
def test_encrypted_datasets_migrate_to_postgres_without_being_decrypted(tmp_path, monkeypatch):
    """SQLite (encrypted) -> PostgreSQL: ciphertext is copied byte for byte and still opens with the same master key."""
    import subprocess, sys, os
    from pathlib import Path
    url = _fresh_pg_database()
    key = __import__("biziq.crypto", fromlist=["x"]).generate_master_key()
    env = {**os.environ, "BIZIQ_DATA_DIR": str(tmp_path), "BIZIQ_MASTER_KEY": key, "BIZIQ_ENCRYPT_DATA": "1", "BIZIQ_AUTH": "1"}
    env.pop("BIZIQ_DATABASE_URL", None)
    root = str(Path(__file__).resolve().parent.parent)
    seed = ("from biziq import accounts, data_layer as d; import pandas as pd\n"
            "u=accounts.create_org('Acme','o@acme.test','Olive','Passw0rdOK'); accounts.activate_org(u.org_id)\n"
            "d.save_dataset(pd.DataFrame([dict(invoice_id='I1',party='SECRETPARTY',type='receivable',invoice_date=pd.Timestamp('2026-01-01'),due_date=pd.Timestamp('2026-02-01'),amount=5.0,paid_date=pd.NaT)]),'invoices')\n"
            "print(u.org_id)")
    org = subprocess.run([sys.executable, "-c", seed], env=env, cwd=root, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    r = subprocess.run([sys.executable, "scripts/migrate_to_postgres.py", "--url", url, "--sqlite-dir", str(tmp_path)], env=env, cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    read = ("from biziq import accounts, data_layer as d\naccounts.activate_org(%r)\n"
            "print(d.load_dataset('invoices').party.iloc[0])" % org)
    out = subprocess.run([sys.executable, "-c", read], env={**env, "BIZIQ_DATABASE_URL": url}, cwd=root, capture_output=True, text=True)
    assert out.stdout.strip().endswith("SECRETPARTY"), out.stdout + out.stderr
