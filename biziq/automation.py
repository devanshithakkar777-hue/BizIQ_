"""Automation layer: rules and triggers that act on the insights.

* Alert rules (reorder, anomaly, overdue receivables, tax deadlines, at-risk clients, ...)
  with de-duplication so the same alert is not sent every run.
* Email via SMTP. DRY-RUN BY DEFAULT: messages are written to output/outbox/ as .eml files
  and only sent when BIZIQ_SEND_EMAIL=1 and SMTP settings are configured.
* Jobs: weekly report, alert check, follow-up drafts, purchase-order drafts, on-upload
  monthly financial summary. Every run is logged (with duration) for the measurement
  exercise in the plan (Section 5).
* Drafts (follow-up emails, purchase orders) are saved for HUMAN REVIEW and never auto-sent.
"""
from __future__ import annotations

import hashlib
import mimetypes
import os
import smtplib
import sqlite3
import time
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from . import db, platform_db, data_layer, decision, pipeline, reportiq, reports, wealthiq
from .pipeline import PipelineResult

OUT = platform_db.OUT
OUTBOX, DRAFTS, SUMMARIES = OUT / "outbox", OUT / "drafts", OUT / "summaries"
COOLDOWN_DAYS = 7


@dataclass
class Alert:
    key: str
    level: str
    module: str
    title: str
    detail: str = ""
    action: str = ""


def _key(module: str, title: str) -> str:
    # strip digits so "3 customers" and "4 customers" are the same alert over time
    base = "".join(ch for ch in title if not ch.isdigit())
    return hashlib.sha1(f"{module}|{base}".encode()).hexdigest()[:12]


# ------------------------------------------------------------------ alert rules
def build_alerts(pr: PipelineResult) -> list[Alert]:
    alerts: dict[str, Alert] = {}

    def add(level, module, title, detail="", action="", key_title=None):
        k = _key(module, key_title or title)
        alerts.setdefault(k, Alert(k, level, module, title, detail, action))

    for f in pr.all_flags():
        if f.level in ("red", "amber"):
            add(f.level, f.module, f.title, f.detail)

    sc = pr.modules.get("SupplyChainIQ")
    if sc is not None and "reorder" in sc.tables:
        for _, r in sc.tables["reorder"].iterrows():
            lvl = "red" if "critical" in r["status"] else "amber"
            add(lvl, "SupplyChainIQ", f"Reorder {r['product']}: order {int(r['suggested_order_qty'])} units",
                f"{r['days_of_cover']:.0f} days of cover against a {int(r['lead_time_days'])}-day lead time.",
                "Approve the drafted purchase order.", key_title=f"Reorder {r['product']}")
    cf = pr.modules.get("CashFlowIQ")
    if cf is not None and "receivables_by_party" in cf.tables:
        for _, r in cf.tables["receivables_by_party"].head(3).iterrows():
            if r["overdue_amount"] > 0 and r["oldest_days_overdue"] > 30:
                add("amber", "CashFlowIQ", f"{r['party']} has {r['overdue_amount']:,.0f} overdue ({int(r['oldest_days_overdue'])} days)",
                    "Follow up on payment.", "Send a payment reminder.", key_title=f"Overdue {r['party']}")
    cl = pr.modules.get("ClientIQ")
    if cl is not None and "at_risk" in cl.tables:
        for _, r in cl.tables["at_risk"].head(5).iterrows():
            add("amber", "ClientIQ", f"High-value customer {r['customer_id']} at risk of churn",
                f"Churn probability {r['churn_prob']:.0%}, annual revenue {r['annual_revenue']:,.0f}.",
                r["next_best_action"], key_title=f"At risk {r['customer_id']}")
    ri = pr.modules.get("RiskIQ")
    if ri is not None and "anomalies" in ri.tables:
        for _, r in ri.tables["anomalies"].head(5).iterrows():
            add("red" if r["severity"] == "High" else "amber", "RiskIQ", f"Anomaly: {r['item']}", r["note"],
                "Investigate the cause.", key_title=f"Anomaly {r['item']}")
    order = {"red": 0, "amber": 1}
    return sorted(alerts.values(), key=lambda a: (order[a.level], a.module))


# ------------------------------------------------------------------ state
def _ddl(c) -> None:
    c.execute("""CREATE TABLE IF NOT EXISTS alerts (key TEXT PRIMARY KEY, level TEXT, module TEXT, title TEXT,
                 detail TEXT, action TEXT, first_seen TEXT, last_seen TEXT, last_notified TEXT, status TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS automation_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, job TEXT,
                 status TEXT, seconds REAL, details TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS automation_tasks (task TEXT PRIMARY KEY, biziq_feature TEXT,
                 automation_level TEXT, manual_minutes REAL, biziq_minutes REAL, notes TEXT)""")


def _conn():
    c = data_layer._conn()
    db.ensure(c, "automation", _ddl)
    return c


def sync_alerts(alerts: list[Alert]) -> list[Alert]:
    """Store alerts, resolve cleared ones, and return alerts that should be notified now."""
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    due = []
    with _conn() as c:
        active_keys = {a.key for a in alerts}
        for a in alerts:
            row = c.execute("SELECT last_notified, status FROM alerts WHERE key=?", (a.key,)).fetchone()
            if row is None:
                c.execute("INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (a.key, a.level, a.module, a.title, a.detail, a.action, now, now, None, "active"))
                due.append(a)
            else:
                last, status = row
                c.execute("UPDATE alerts SET level=?, title=?, detail=?, action=?, last_seen=?, status='active' WHERE key=?",
                          (a.level, a.title, a.detail, a.action, now, a.key))
                stale = last is None or status == "resolved" or \
                    (pd.Timestamp.now() - pd.Timestamp(last)).days >= COOLDOWN_DAYS
                if stale:
                    due.append(a)
        for (k,) in c.execute("SELECT key FROM alerts WHERE status='active'").fetchall():
            if k not in active_keys:
                c.execute("UPDATE alerts SET status='resolved' WHERE key=?", (k,))
    return due


def mark_notified(alerts: list[Alert]) -> None:
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as c:
        for a in alerts:
            c.execute("UPDATE alerts SET last_notified=? WHERE key=?", (now, a.key))


def alert_history(limit: int = 200) -> pd.DataFrame:
    with _conn() as c:
        return db.read_df(c, "SELECT level, module, title, detail, action, first_seen, last_seen, last_notified, status "
                          "FROM alerts ORDER BY status, first_seen DESC LIMIT ?", (limit,))


def log_run(job: str, status: str, seconds: float, details: str = "") -> None:
    with _conn() as c:
        c.execute("INSERT INTO automation_log (ts, job, status, seconds, details) VALUES (?,?,?,?,?)",
                  (pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), job, status, seconds, details))


def run_log(limit: int = 100) -> pd.DataFrame:
    with _conn() as c:
        return db.read_df(c, "SELECT ts, job, status, ROUND(CAST(seconds AS NUMERIC), 2) AS seconds, details FROM automation_log "
                          "ORDER BY id DESC LIMIT ?", (limit,))


# ------------------------------------------------------------------ email
@dataclass
class EmailConfig:
    host: str = ""
    port: int = 587
    user: str = ""
    password: str = ""
    sender: str = ""
    recipients: tuple[str, ...] = ()
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "EmailConfig":
        rec = tuple(x.strip() for x in os.environ.get("BIZIQ_EMAIL_TO", "").split(",") if x.strip())
        return cls(os.environ.get("BIZIQ_SMTP_HOST", ""), int(os.environ.get("BIZIQ_SMTP_PORT", "587")),
                   os.environ.get("BIZIQ_SMTP_USER", ""), os.environ.get("BIZIQ_SMTP_PASSWORD", ""),
                   os.environ.get("BIZIQ_EMAIL_FROM", os.environ.get("BIZIQ_SMTP_USER", "")), rec,
                   os.environ.get("BIZIQ_SEND_EMAIL", "0") == "1")

    def can_send(self) -> bool:
        return self.enabled and bool(self.host and self.sender and self.recipients)


def send_email(subject: str, body: str, attachments: list[Path] | None = None, cfg: EmailConfig | None = None,
               recipients: tuple[str, ...] | None = None) -> str:
    cfg = cfg or EmailConfig.from_env()
    to = recipients or cfg.recipients or ("owner@example.com",)
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, cfg.sender or "biziq@localhost", ", ".join(to)
    msg.set_content(body)
    for p in attachments or []:
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        main, sub = ctype.split("/", 1)
        msg.add_attachment(Path(p).read_bytes(), maintype=main, subtype=sub, filename=Path(p).name)
    if cfg.can_send():
        with smtplib.SMTP(cfg.host, cfg.port, timeout=30) as s:
            s.starttls()
            if cfg.user:
                s.login(cfg.user, cfg.password)
            s.send_message(msg)
        return f"sent to {', '.join(to)}"
    OUTBOX.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in subject)[:50]
    path = OUTBOX / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_{safe}.eml"
    path.write_bytes(bytes(msg))
    return f"DRY RUN - saved to {path} (set BIZIQ_SEND_EMAIL=1 and SMTP settings to send)"


def compose_digest(pr: PipelineResult, alerts: list[Alert], recs: dict) -> str:
    ex = reportiq.executive_summary(pr)
    lines = [f"BizIQ weekly digest - data as at {pr.as_of:%d %b %Y}", "", ex["headline"], ""]
    lines += [f"{a}: {b}" for a, b in ex["kpis"]]
    if "SalesIQ" in pr.modules:
        lines += ["", "Sales", reportiq.module_summary("SalesIQ", pr.modules["SalesIQ"])]
    lines += ["", "Alerts needing attention"]
    lines += [f"- [{a.level.upper()}] {a.module}: {a.title}" + (f" ({a.action})" if a.action else "") for a in alerts[:12]] or ["- None"]
    lines += ["", "Top recommended actions"]
    for name in ("SupplyChainIQ", "CashFlowIQ", "ClientIQ", "WealthIQ"):
        if name in recs and recs[name]["items"]:
            lines.append(f"- {name}: {recs[name]['items'][0]}")
    lines += ["", "Full report attached. Estimates are labelled as estimates."]
    return "\n".join(lines)


# ------------------------------------------------------------------ jobs
def _timed(job: str, fn):
    t0 = time.perf_counter()
    try:
        info = fn()
        log_run(job, "ok", time.perf_counter() - t0, str(info)[:300])
        return info
    except Exception as e:
        log_run(job, "failed", time.perf_counter() - t0, f"{type(e).__name__}: {e}")
        raise


def weekly_report_job(data=None, send: bool = True, use_llm: bool = False, kind: str = "weekly") -> dict:
    def go():
        d = data if data is not None else data_layer.load_all()
        pr = pipeline.run_all(d)
        recs = decision.get_recommendations(pr, use_llm=use_llm)
        meta = data_layer.dataset_meta()
        note = "Figures are based on SYNTHETIC sample data." if (len(meta) and meta["source"].str.contains("synthetic").any()) else ""
        pdf = reports.build_report(pr, kind=kind, recs=recs, data_note=note)
        alerts = build_alerts(pr)
        due = sync_alerts(alerts)
        status = "not emailed"
        if send:
            status = send_email(f"BizIQ {kind} report - {pr.as_of:%d %b %Y}", compose_digest(pr, alerts, recs), [pdf])
            mark_notified(due)
        return dict(pdf=str(pdf), alerts=len(alerts), new_alerts=len(due), email=status, seconds=round(pr.total_seconds, 2))
    return _timed("weekly_report", go)


def alert_job(data=None, send: bool = True) -> dict:
    def go():
        pr = pipeline.run_all(data if data is not None else data_layer.load_all())
        due = sync_alerts(build_alerts(pr))
        status = "none new"
        if due:
            body = "New BizIQ alerts:\n\n" + "\n".join(
                f"[{a.level.upper()}] {a.module}: {a.title}\n   {a.detail}" + (f"\n   Action: {a.action}" if a.action else "") for a in due)
            status = send_email(f"BizIQ: {len(due)} new alert(s)", body) if send else "not emailed"
            mark_notified(due)
        return dict(new_alerts=len(due), email=status)
    return _timed("alert_check", go)


def followup_drafts_job(data=None) -> dict:
    def go():
        pr = pipeline.run_all(data if data is not None else data_layer.load_all())
        cl = pr.modules.get("ClientIQ")
        if cl is None:
            return dict(drafts=0)
        DRAFTS.mkdir(parents=True, exist_ok=True)
        df = cl.tables["drafted_emails"]
        p = DRAFTS / f"followup_drafts_{pr.as_of:%Y-%m-%d}.csv"
        df.to_csv(p, index=False)
        return dict(drafts=len(df), file=str(p), note="For human review; never auto-sent")
    return _timed("followup_drafts", go)


def purchase_order_job(data=None) -> dict:
    def go():
        pr = pipeline.run_all(data if data is not None else data_layer.load_all())
        pq = pr.modules.get("ProcureIQ")
        if pq is None or not len(pq.tables["purchase_order_documents"]):
            return dict(purchase_orders=0)
        DRAFTS.mkdir(parents=True, exist_ok=True)
        files = []
        for _, r in pq.tables["purchase_order_documents"].iterrows():
            p = DRAFTS / f"PO_DRAFT_{r['supplier'].replace(' ', '_')}_{pr.as_of:%Y-%m-%d}.txt"
            p.write_text(r["document"])
            files.append(str(p))
        return dict(purchase_orders=len(files), files=files, note="Drafts for human approval")
    return _timed("purchase_order_drafts", go)


def on_new_financials_job(fin: pd.DataFrame | None = None) -> dict:
    """Trigger on upload of a new month's financials: ratios, score and a written summary."""
    def go():
        f = fin if fin is not None else data_layer.load_dataset("financials")
        res = wealthiq.run(f)
        SUMMARIES.mkdir(parents=True, exist_ok=True)
        p = SUMMARIES / f"financial_summary_{res.metrics['latest_month']}.txt"
        p.write_text(wealthiq.summary_text(res) + "\n\n" + "\n".join(fl.title for fl in res.flags))
        return dict(file=str(p), health_score=res.metrics["health_score"])
    return _timed("monthly_financial_summary", go)


JOBS = {"weekly_report": weekly_report_job, "alert_check": alert_job, "followup_drafts": followup_drafts_job,
        "purchase_order_drafts": purchase_order_job, "monthly_financial_summary": on_new_financials_job}


# ------------------------------------------------------------------ measuring the automation claim
TASK_TEMPLATE = [
    ("Monthly financial ratios and health summary", "WealthIQ (runs on upload)", "Fully automated"),
    ("Weekly sales summary", "SalesIQ weekly digest", "Fully automated"),
    ("Stock check and reorder list", "SupplyChainIQ + reorder alerts", "Fully automated"),
    ("Purchase order drafting", "ProcureIQ draft POs", "Assisted (human review)"),
    ("At-risk client list and follow-up emails", "ClientIQ drafts", "Assisted (human review)"),
    ("Receivables ageing and chase list", "CashFlowIQ", "Fully automated"),
    ("GST / tax deadline tracking", "TaxIQ reminders", "Fully automated"),
    ("Unusual expense / sales-drop monitoring", "RiskIQ anomaly alerts", "Fully automated"),
    ("Cash flow forecast", "CashFlowIQ", "Fully automated"),
    ("Leadership report", "One-click PDF / scheduled email", "Fully automated"),
]


def load_tasks() -> pd.DataFrame:
    with _conn() as c:
        df = db.read_df(c, "SELECT * FROM automation_tasks")
        if df.empty:
            df = pd.DataFrame(TASK_TEMPLATE, columns=["task", "biziq_feature", "automation_level"])
            df["manual_minutes"], df["biziq_minutes"], df["notes"] = float("nan"), float("nan"), ""
            db.append_df(c, df, "automation_tasks")
    return df


def save_tasks(df: pd.DataFrame) -> None:
    with _conn() as c:
        c.execute("DELETE FROM automation_tasks")
        db.append_df(c, df, "automation_tasks")


def measurement_summary(df: pd.DataFrame) -> dict:
    """Only MEASURED numbers are reported. Returns None for anything not yet measured."""
    n = len(df)
    full = int((df["automation_level"] == "Fully automated").sum())
    assisted = int(df["automation_level"].str.startswith("Assisted").sum())
    both = df.dropna(subset=["manual_minutes", "biziq_minutes"])
    out = dict(tasks=n, fully_automated=full, assisted=assisted, measured_tasks=len(both),
               share_fully_automated=full / n if n else None, share_at_least_assisted=(full + assisted) / n if n else None,
               manual_minutes=None, biziq_minutes=None, time_reduction=None)
    if len(both):
        m, b = float(both["manual_minutes"].sum()), float(both["biziq_minutes"].sum())
        out.update(manual_minutes=m, biziq_minutes=b, time_reduction=(m - b) / m if m else None)
    return out
