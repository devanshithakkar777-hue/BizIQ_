"""A full report pack the owner can hand to an accountant, banker or consultant: PDF to read, Excel to recompute.

Rules it keeps:
* People modules (pay, performance, hiring, PeopleIQ) are OFF unless the owner ticks them, and even then any table that lists individuals is left out.
* The pack says how sure BizIQ is about each page (the evidence level) and what BizIQ did not check.
* "Hide names" replaces customer, vendor, supplier, client and staff names with neutral labels, best effort: a person should still glance through before sending.
* It is not an audit and says so on the cover.
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import decision, evidence, registry, reportiq
from . import reports as rp
from .exporter import _safe
from .pipeline import PipelineResult

PRIVATE_COLS = ("customer_id", "customer", "party", "vendor", "supplier", "client", "staff", "distributor", "carrier")
PERSON_COLS = {"employee_id", "candidate_id", "candidate", "employee", "name"}
PEOPLE_BUILTIN = {"PeopleIQ"}

DISCLAIMER = ("Prepared by BizIQ from data supplied by the business. It is not an audit, a valuation or advice. Figures marked as estimates rest on stated assumptions; "
              "the confidence level on each section says how much data stood behind it. Check anything important against the source records before relying on it.")


def is_people(name: str) -> bool:
    spec = registry.get(name)
    return name in PEOPLE_BUILTIN or bool(spec and (spec.sensitive or spec.group == "People"))


def sections(pr: PipelineResult) -> list[dict]:
    """What can go into the pack, with people sections off by default."""
    return [dict(name=n, people=is_people(n), default=not is_people(n)) for n in pr.modules]


class Hider:
    """Consistent neutral labels for names in the chosen columns, applied to tables and to free text."""

    def __init__(self, data: dict[str, pd.DataFrame], on: bool):
        self.on, self.map = on, {}
        if not on:
            return
        for df in (data or {}).values():
            for col in PRIVATE_COLS:
                if col in df.columns:
                    for v in df[col].dropna().astype(str).unique():
                        if len(v) >= 3 and v not in self.map:
                            self.map[v] = f"{col.replace('_id', '').title()} {sum(1 for k in self.map if self.map[k].startswith(col.replace('_id', '').title())) + 1}"
        self._rx = re.compile("|".join(re.escape(k) for k in sorted(self.map, key=len, reverse=True))) if self.map else None

    def text(self, s):
        if not self.on or not isinstance(s, str) or self._rx is None:
            return s
        return self._rx.sub(lambda m: self.map[m.group(0)], s)

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.on:
            return df
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == object or str(out[c].dtype).startswith("str"):
                out[c] = out[c].map(lambda v: self.map.get(v, self.text(v)) if isinstance(v, str) else v)
        return out


def _clean_tables(res, hider: Hider) -> dict[str, pd.DataFrame]:
    out = {}
    for k, t in res.tables.items():
        if not isinstance(t, pd.DataFrame) or t.empty:
            continue
        if PERSON_COLS & set(map(str, t.columns)):
            continue                                              # never ship a table that lists individuals
        out[k] = hider.frame(t)
    return out


def _evidence_lines(name, data, today):
    ev = evidence.assess(name, data, today)
    return ev


def build_xlsx(pr: PipelineResult, recs: dict, chosen: list[str], data: dict, org_name: str, hide_names: bool = False) -> bytes:
    hider = Hider(data, hide_names)
    buf = io.BytesIO()
    summary, ev_rows = [], []
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name in chosen:
            res = pr.modules.get(name)
            if res is None:
                continue
            ev = evidence.assess(name, data, pr.as_of)
            level = ev.level if ev else ""
            for f in res.flags:
                summary.append(dict(module=name, confidence=level, level=f.level, finding=hider.text(f.title), detail=hider.text(f.detail)))
            for r in recs.get(name, {}).get("items", []):
                summary.append(dict(module=name, confidence=level, level="action", finding=hider.text(r), detail=""))
            if ev is not None:
                for why in ev.reasons:
                    ev_rows.append(dict(module=name, confidence=level, reason=hider.text(why)))
            for k, t in _clean_tables(res, hider).items():
                _safe(t.head(5000)).to_excel(xw, sheet_name=f"{name}-{k}"[:31], index=False)
        cover = pd.DataFrame({"": [f"BizIQ consultant pack for {org_name}", f"Data as at {pr.as_of:%d %B %Y}", f"Generated {datetime.now():%d %b %Y %H:%M}",
                                   "Sections: " + ", ".join(chosen), "Names hidden: " + ("yes (best effort)" if hide_names else "no"), DISCLAIMER]})
        cover.to_excel(xw, sheet_name="Cover", index=False)
        _safe(pd.DataFrame(summary or [dict(module="", confidence="", level="", finding="Nothing flagged", detail="")])).to_excel(xw, sheet_name="Findings", index=False)
        _safe(pd.DataFrame(ev_rows or [dict(module="", confidence="", reason="")])).to_excel(xw, sheet_name="How sure", index=False)
        wb = xw.book
        order = ["Cover", "Findings", "How sure"]
        wb._sheets = [wb[n] for n in order] + [s for s in wb._sheets if s.title not in order]
    return buf.getvalue()


def build_pdf(pr: PipelineResult, recs: dict, chosen: list[str], data: dict, org_name: str, hide_names: bool = False, path: str | Path | None = None) -> Path:
    st = rp._styles()
    hider = Hider(data, hide_names)
    rp.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(path) if path else rp.OUT_DIR / f"BizIQ_consultant_pack_{pr.as_of:%Y-%m-%d}.pdf"
    names = [n for n in chosen if n in pr.modules]
    excluded_people = [n for n in pr.modules if is_people(n) and n not in names]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(rp.MUTED)
        canvas.drawString(18 * mm, 10 * mm, f"BizIQ consultant pack | {hider.text(org_name) if hide_names else org_name} | data as at {pr.as_of:%d %b %Y} | not an audit")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="BizIQ consultant pack", author="BizIQ")
    s = [Paragraph("Business review pack", st["title"]), Paragraph(f"{org_name} | data as at {pr.as_of:%d %B %Y} | generated {datetime.now():%d %b %Y %H:%M}", st["sub"]),
         Spacer(1, 6), Paragraph("<b>About this pack</b>", st["h2"]), Paragraph(DISCLAIMER, st["body"]), Spacer(1, 4)]
    scope = [f"Sections included ({len(names)}): " + ", ".join(names) + "."]
    if excluded_people:
        scope.append("People sections (" + ", ".join(excluded_people) + ") are not included.")
    elif any(is_people(n) for n in names):
        scope.append("People sections are included at group level only; any table listing individuals has been left out.")
    scope.append("Names of customers, vendors, suppliers, clients and staff have been replaced with neutral labels (best effort)." if hide_names else "Names are shown as in the business's data.")
    s += [Paragraph(x, st["bullet"], bulletText="•") for x in scope]
    s += [Spacer(1, 6), Paragraph("How to read the confidence labels", st["h2"])]
    for lv, txt in (("High", "plenty of recent, complete data behind the page."), ("Medium", "usable as a guide; the data is thin, old or has gaps."),
                    ("Low", "a rough hint only until more or better data is loaded."), ("Rule-based", "statutory calendar rules, not the business's data.")):
        s.append(Paragraph(f"<b>{lv}</b>: {txt}", st["bullet"], bulletText="•"))
    ex = reportiq.executive_summary(pr)
    keep = [f for f in ex["top_issues"] if f.module in names]
    s += [Spacer(1, 8), Paragraph("Needs attention", st["h1"]), rp._flag_table([_hide_flag(f, hider) for f in keep], st, limit=12), PageBreak()]
    for name in names:
        res = pr.modules[name]
        ev = evidence.assess(name, data, pr.as_of)
        head = [Paragraph(name, st["h1"]), Paragraph(hider.text(reportiq.module_summary(name, res)), st["body"])]
        if ev is not None:
            head.append(Paragraph(f"<b>Confidence: {ev.level}.</b> " + " ".join(hider.text(r) for r in ev.reasons[:3]), st["small"]))
        s.append(KeepTogether(head))
        s.append(rp._flag_table([_hide_flag(f, hider) for f in res.flags], st, limit=8))
        for ch in rp._module_charts(name, res):
            s.append(ch)
        tabs = _clean_tables(res, hider)
        for k, t in list(tabs.items())[:2]:
            s += [Spacer(1, 3), Paragraph(k.replace("_", " ").capitalize(), st["h2"]), rp.df_table(t.iloc[:, :7].head(8), st, max_rows=8)]      # first 7 columns keep the page readable; the Excel has everything
        items = recs.get(name, {}).get("items", [])
        if items:
            s.append(Paragraph("Suggested actions", st["h2"]))
            s += [Paragraph(hider.text(i), st["bullet"], bulletText="•") for i in items[:5]]
        for n in res.notes[:2]:
            s.append(Paragraph(f"<i>Method and limits: {hider.text(n)}</i>", st["small"]))
        s.append(Spacer(1, 10))
    s += [PageBreak(), Paragraph("What BizIQ did not check", st["h1"])]
    for x in ("It did not audit or reconcile the books to bank statements or tax filings, and it cannot see transactions that were never entered.",
              "It did not verify that the data supplied is complete or correct; the confidence labels only say how much there was and how recent.",
              "Thresholds (for example 'too much overtime', target margins, normal payment days) are stated assumptions, not industry standards for this specific business.",
              "Forecasts and scenarios extend past patterns and do not know about planned changes, seasonality outside the history, or events.",
              "Tax and compliance dates are a calendar aid: confirm them with a qualified professional.",
              "Nothing here is a judgement about any individual. Where people data appears it is at group level, and a number cannot know the circumstances behind it."):
        s.append(Paragraph(x, st["bullet"], bulletText="•"))
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    return path


def _hide_flag(f, hider: Hider):
    if not hider.on:
        return f
    import dataclasses
    return dataclasses.replace(f, title=hider.text(f.title), detail=hider.text(f.detail))
