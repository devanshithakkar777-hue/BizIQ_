"""Tiny dependency-free scheduler for BizIQ automation jobs.

    python scripts/run_scheduler.py            # run forever
    python scripts/run_scheduler.py --once     # run every job once now, then exit (handy for testing)

Edit SCHEDULE to change when things run. Emails stay in DRY-RUN mode unless
BIZIQ_SEND_EMAIL=1 and SMTP settings are configured (see README).
"""
import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os  # noqa: E402

from biziq import accounts, automation, logconfig  # noqa: E402

logconfig.setup()

# (job name, weekday 0=Mon or None for daily, "HH:MM" local time)
SCHEDULE = [
    ("weekly_report", 0, "08:00"),
    ("alert_check", None, "09:00"),
    ("followup_drafts", 0, "08:15"),
    ("purchase_order_drafts", 0, "08:20"),
    ("live_sync", None, "06:30"),          # pulls Shopify / Zoho / Tally for every organisation that has connected them
]


def due(now: datetime, weekday, hhmm: str, last_run: dict, key: str) -> bool:
    if weekday is not None and now.weekday() != weekday:
        return False
    if now.strftime("%H:%M") < hhmm:
        return False
    return last_run.get(key) != now.date()


def run_job(name: str):
    """Run one job for every organisation (or once against the local database when BIZIQ_AUTH=0)."""
    orgs = accounts.list_org_ids() if os.environ.get("BIZIQ_AUTH", "1") != "0" else [None]
    for org in orgs or [None]:
        if org:
            accounts.activate_org(org)
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] running {name}" + (f" for {org}" if org else "") + " ...", flush=True)
        try:
            if name == "live_sync":
                from biziq.live import sync as live_sync
                print("   ->", live_sync.sync_all(org or "local"), flush=True)
            else:
                print("   ->", automation.JOBS[name](), flush=True)
        except Exception as e:  # keep the scheduler alive
            print(f"   !! {name} failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if args.once:
        for name, _, _ in SCHEDULE:
            run_job(name)
        sys.exit(0)
    last = {}
    print("BizIQ scheduler started. Ctrl+C to stop.")
    while True:
        now = datetime.now()
        for name, wd, hhmm in SCHEDULE:
            if due(now, wd, hhmm, last, name):
                run_job(name)
                last[name] = now.date()
        time.sleep(30)
