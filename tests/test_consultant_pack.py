"""The consultant pack: people off by default, no individual rows, names hideable, honest about limits."""
import io

import pandas as pd
import pytest

from biziq import consultant_pack as cp, decision


@pytest.fixture(scope="module")
def recs(pr):
    return decision.get_recommendations(pr)


def non_people(pr):
    return [x["name"] for x in cp.sections(pr) if not x["people"]]


class TestSections:
    def test_people_sections_are_listed_but_off_by_default(self, pr):
        secs = {x["name"]: x for x in cp.sections(pr)}
        for n in ("PeopleIQ", "HiringIQ", "SalaryIQ", "PerformanceReviewIQ"):
            if n in secs:
                assert secs[n]["people"] and not secs[n]["default"]
        assert secs["SalesIQ"]["default"] and not secs["SalesIQ"]["people"]
        assert any(x["people"] for x in secs.values())

    def test_no_table_in_the_pack_lists_individuals_even_when_people_sections_are_chosen(self, pr, sample, recs):
        names = [x["name"] for x in cp.sections(pr)]
        xl = cp.build_xlsx(pr, recs, names, sample, "Acme")
        book = pd.read_excel(io.BytesIO(xl), sheet_name=None)
        for sheet, df in book.items():
            assert not ({"employee_id", "candidate_id", "candidate"} & set(df.columns)), sheet
        assert any(s.startswith("SalaryIQ") for s in book) or "SalaryIQ" not in pr.modules


class TestExcel:
    def test_has_cover_findings_and_confidence_and_a_sheet_per_table(self, pr, sample, recs):
        xl = cp.build_xlsx(pr, recs, non_people(pr), sample, "Acme Retail")
        book = pd.read_excel(io.BytesIO(xl), sheet_name=None)
        assert list(book)[:3] == ["Cover", "Findings", "How sure"]
        assert "not an audit" in " ".join(book["Cover"].iloc[:, 0].astype(str)).lower()
        assert set(book["Findings"]["confidence"].dropna()) <= {"High", "Medium", "Low", "Rule-based", ""}
        assert len(book) > 10

    def test_cells_cannot_run_as_formulas(self, pr, sample, recs):
        from openpyxl import load_workbook
        evil = {k: v.copy() for k, v in sample.items()}
        evil["sales"].loc[evil["sales"].index[0], "product"] = "=HYPERLINK(\"http://x\")"
        xl = cp.build_xlsx(pr, recs, non_people(pr), evil, "Acme")
        for ws in load_workbook(io.BytesIO(xl)).worksheets:
            for row in ws.iter_rows(values_only=True):
                for v in row:
                    assert not (isinstance(v, str) and v.startswith("=")), (ws.title, v)


class TestHidingNames:
    def test_names_are_replaced_in_tables_and_text_and_consistently(self, sample):
        h = cp.Hider(sample, True)
        real = str(sample["invoices"]["party"].dropna().iloc[0])
        assert real in h.map and h.text(f"{real} owes money") == f"{h.map[real]} owes money"
        t = h.frame(sample["invoices"][["party", "amount"]].head(5))
        assert real not in set(t["party"]) or real not in h.map
        assert h.frame(sample["invoices"][["party"]].head(3))["party"].iloc[0] == h.map[str(sample["invoices"]["party"].iloc[0])]

    def test_off_means_untouched(self, sample):
        h = cp.Hider(sample, False)
        df = sample["invoices"][["party"]].head(3)
        assert h.frame(df).equals(df) and h.text("Acme owes") == "Acme owes"

    def test_hidden_names_do_not_survive_in_the_excel(self, pr, sample, recs):
        xl = cp.build_xlsx(pr, recs, non_people(pr), sample, "Acme", hide_names=True)
        blob = " ".join(str(v) for df in pd.read_excel(io.BytesIO(xl), sheet_name=None).values() for v in df.astype(str).to_numpy().ravel())
        h = cp.Hider(sample, True)
        leaked = [k for k in h.map if len(k) >= 8 and k in blob]
        assert not leaked[:3], leaked[:3]


class TestPdf:
    def test_pdf_builds_with_cover_confidence_and_limits(self, pr, sample, recs, tmp_path):
        p = cp.build_pdf(pr, recs, non_people(pr), sample, "Acme Retail", path=tmp_path / "pack.pdf")
        raw = p.read_bytes()
        assert raw.startswith(b"%PDF") and len(raw) > 20_000
        try:
            from pypdf import PdfReader
        except ImportError:
            pytest.skip("pypdf not installed")
        text = " ".join(pg.extract_text() for pg in PdfReader(str(p)).pages)
        assert "not an audit" in text.lower() and "What BizIQ did not check" in text and "Confidence:" in text
        assert "People sections" in text                       # says that they were left out
