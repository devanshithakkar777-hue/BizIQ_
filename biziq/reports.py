"""PDF report generation (ReportLab): weekly digest and board-ready monthly report."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from . import decision, reportiq
from .core import inr
from .pipeline import PipelineResult

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "reports"
INK, MUTED, ACCENT = colors.HexColor("#1f2a37"), colors.HexColor("#6b7280"), colors.HexColor("#1d4ed8")
LEVEL_COL = {"red": colors.HexColor("#dc2626"), "amber": colors.HexColor("#d97706"), "green": colors.HexColor("#16a34a")}
WEEKLY = ["WealthIQ", "SalesIQ", "SupplyChainIQ", "ClientIQ", "CashFlowIQ", "TaxIQ", "RiskIQ"]


def _styles():
    ss = getSampleStyleSheet()
    return dict(
        title=ParagraphStyle("t", parent=ss["Title"], fontSize=22, textColor=INK, alignment=0, spaceAfter=4),
        sub=ParagraphStyle("s", parent=ss["Normal"], fontSize=10, textColor=MUTED, spaceAfter=10),
        h1=ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=ACCENT, spaceBefore=10, spaceAfter=5),
        h2=ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, textColor=INK, spaceBefore=6, spaceAfter=3),
        body=ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13, textColor=INK),
        bullet=ParagraphStyle("bl", parent=ss["BodyText"], fontSize=9.5, leading=13, leftIndent=10, bulletIndent=0),
        small=ParagraphStyle("sm", parent=ss["BodyText"], fontSize=8, leading=10, textColor=MUTED),
        cell=ParagraphStyle("c", parent=ss["BodyText"], fontSize=8, leading=10),
    )


def _fmt(v):
    import numpy as np
    if isinstance(v, np.integer):
        v = int(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%d %b %Y")
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def df_table(df: pd.DataFrame, st, max_rows: int = 8, col_widths=None) -> Table:
    df = df.head(max_rows)
    data = [[Paragraph(f"<b>{c}</b>", st["cell"]) for c in df.columns]]
    for _, r in df.iterrows():
        data.append([Paragraph(_fmt(v), st["cell"]) for v in r.values])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def line_chart(title, labels, series: dict[str, list[float]], w=170 * mm, h=55 * mm, fmt="%.0f") -> Drawing:
    d = Drawing(w, h + 12)
    d.add(String(0, h + 2, title, fontSize=9, fillColor=INK))
    ch = HorizontalLineChart()
    ch.x, ch.y, ch.width, ch.height = 35, 18, w - 50, h - 18
    ch.data = list(series.values())
    ch.categoryAxis.categoryNames = labels
    ch.categoryAxis.labels.fontSize = 6
    ch.categoryAxis.labels.angle = 30
    ch.categoryAxis.labels.dy = -8
    ch.valueAxis.labels.fontSize = 7
    ch.valueAxis.labelTextFormat = fmt
    palette = [ACCENT, colors.HexColor("#dc2626"), colors.HexColor("#16a34a"), colors.HexColor("#9333ea")]
    for i in range(len(series)):
        ch.lines[i].strokeColor = palette[i % 4]
        ch.lines[i].strokeWidth = 1.4
        ch.lines[i].symbol = None
    d.add(ch)
    return d


def bar_chart(title, labels, values, w=170 * mm, h=55 * mm) -> Drawing:
    d = Drawing(w, h + 12)
    d.add(String(0, h + 2, title, fontSize=9, fillColor=INK))
    ch = VerticalBarChart()
    ch.x, ch.y, ch.width, ch.height = 35, 22, w - 50, h - 22
    ch.data = [values]
    ch.categoryAxis.categoryNames = labels
    ch.categoryAxis.labels.fontSize = 6
    ch.categoryAxis.labels.angle = 20
    ch.categoryAxis.labels.dy = -8
    ch.valueAxis.labels.fontSize = 7
    ch.valueAxis.valueMin = 0
    ch.bars[0].fillColor = ACCENT
    d.add(ch)
    return d


def _flag_table(flags, st, limit=10):
    if not flags:
        return Paragraph("No flags raised.", st["body"])
    rows = [[Paragraph(f"<font color='{LEVEL_COL[f.level].hexval()}'><b>{f.level.upper()}</b></font>", st["cell"]),
             Paragraph(f"<b>{f.module}</b>", st["cell"]),
             Paragraph(f"{f.title}" + (f" - {f.detail}" if f.detail else ""), st["cell"])] for f in flags[:limit]]
    t = Table(rows, colWidths=[16 * mm, 26 * mm, 128 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def _module_charts(name, res):
    out = []
    try:
        if name == "WealthIQ" and "trend" in res.tables:
            t = res.tables["trend"]
            out.append(line_chart("Financial health score (0-100)", [m.strftime("%b %y") for m in t["month"]],
                                  {"Health score": t["health_score"].round(1).tolist()}, fmt="%.0f"))
        if name == "SalesIQ" and "monthly" in res.tables:
            m = res.tables["monthly"]
            out.append(bar_chart("Monthly revenue (Rs lakh)", [d.strftime("%b %y") for d in m["date"]],
                                 (m["revenue"] / 1e5).round(1).tolist()))
        if name == "CashFlowIQ" and "forecast" in res.tables:
            f = res.tables["forecast"]
            out.append(line_chart("Cash forecast (Rs lakh)", [m.strftime("%b %y") for m in f["month"]],
                                  {s: (f[s] / 1e5).round(1).tolist() for s in ("Base", "Downside", "Upside")}, fmt="%.0f"))
    except Exception:
        pass
    return out


def _module_table(name, res, st):
    pick = {"SupplyChainIQ": ("reorder", ["product", "stock_on_hand", "days_of_cover", "suggested_order_qty", "status"]),
            "ClientIQ": ("at_risk", ["customer_id", "segment", "recency_days", "churn_prob", "annual_revenue"]),
            "TaxIQ": ("deadlines", ["due_date", "obligation", "days_left"]),
            "RiskIQ": ("anomalies", ["date", "item", "severity", "note"]),
            "ProcureIQ": ("draft_purchase_orders", ["supplier", "product", "quantity", "line_total"]),
            "SalesIQ": ("top_products", ["product", "revenue", "share"]),
            "PeopleIQ": ("departments", ["department", "headcount", "avg_overtime", "attrition_pct"]),
            "BrandIQ": ("by_product", ["product", "reviews", "avg_rating", "negative_share"]),
            "InvestIQ": ("allocation_plan", ["asset_class", "weight_pct", "amount"]),
            "CashFlowIQ": ("receivables_by_party", ["party", "open_amount", "overdue_amount", "avg_days_late"])}
    if name not in pick:
        return None
    tname, cols = pick[name]
    if tname not in res.tables or not len(res.tables[tname]):
        return None
    df = res.tables[tname]
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f" and ("share" in c or "prob" in c or "rate" in c or c == "attrition_pct" or "negative" in c):
            d[c] = (d[c] * 100).round(1).astype(str) + "%"
    return df_table(d, st, max_rows=6)


def build_report(pr: PipelineResult, path: str | Path | None = None, kind: str = "weekly",
                 recs: dict | None = None, data_note: str = "") -> Path:
    """kind = 'weekly' (concise) or 'board' (all modules)."""
    st = _styles()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(path) if path else OUT_DIR / f"BizIQ_{kind}_{pr.as_of:%Y-%m-%d}.pdf"
    recs = recs or decision.get_recommendations(pr)
    names = [n for n in (WEEKLY if kind == "weekly" else list(pr.modules)) if n in pr.modules]
    ex = reportiq.executive_summary(pr)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10 * mm, f"BizIQ {kind} report | as at {pr.as_of:%d %b %Y} | estimates are labelled as estimates")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title=f"BizIQ {kind} report", author="BizIQ")
    s = []
    s.append(Paragraph("BizIQ Weekly Business Report" if kind == "weekly" else "BizIQ Monthly Board Report", st["title"]))
    s.append(Paragraph(f"Data as at {pr.as_of:%d %B %Y} | generated {datetime.now():%d %b %Y %H:%M}", st["sub"]))
    if data_note:
        s.append(Paragraph(f"<b>Data note:</b> {data_note}", st["small"]))
        s.append(Spacer(1, 4))
    k = ex["kpis"]
    if k:
        cells = [[Paragraph(f"<font size=8 color='#6b7280'>{a}</font><br/><b><font size=12>{b}</font></b>", st["body"]) for a, b in k[i:i + 3]]
                 for i in range(0, len(k), 3)]
        kt = Table([c + [""] * (3 - len(c)) for c in cells], colWidths=[57 * mm] * 3)
        kt.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f9fafb")),
                                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        s += [kt, Spacer(1, 6)]
    s.append(Paragraph("Needs attention", st["h1"]))
    s.append(Paragraph(ex["headline"], st["body"]))
    s.append(Spacer(1, 3))
    s.append(_flag_table(ex["top_issues"], st, limit=10 if kind == "weekly" else 14))
    for name in names:
        res = pr.modules[name]
        block = [Paragraph(name, st["h1"]), Paragraph(reportiq.module_summary(name, res), st["body"]), Spacer(1, 3)]
        for ch in _module_charts(name, res):
            block.append(ch)
        tb = _module_table(name, res, st)
        if tb is not None:
            block += [Spacer(1, 3), tb]
        items = recs.get(name, {}).get("items", [])
        if items:
            block.append(Paragraph("Recommended actions", st["h2"]))
            block += [Paragraph(i, st["bullet"], bulletText="•") for i in items[:5]]
        for n in res.notes[:1]:
            block.append(Paragraph(f"<i>Method: {n}</i>", st["small"]))
        s.append(KeepTogether(block[:3]))
        s += block[3:]
    s.append(Spacer(1, 8))
    s.append(Paragraph("Figures are computed by BizIQ from the uploaded data. Projections, campaign uplift, tax and "
                       "allocation figures are estimates on stated assumptions, not advice.", st["small"]))
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    return path
