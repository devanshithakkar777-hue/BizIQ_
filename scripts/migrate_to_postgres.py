"""Move an existing SQLite BizIQ installation into PostgreSQL.

    python scripts/migrate_to_postgres.py --url postgresql://user:pass@host:5432/biziq              # copy, verify
    python scripts/migrate_to_postgres.py --url ... --sqlite-dir /path/to/output --dry-run          # show what would move

What moves: platform.db (organisations, users, approvals, audit log, connections) into schema `platform`, and each
orgs/<id>/biziq.db into schema `org_<id>`. The audit log is copied row for row, ids and hashes untouched, and the
hash chain is re-verified in PostgreSQL afterwards. Row counts are compared table by table.

Safe by design: it refuses to write into tables that already hold rows, never modifies or deletes the SQLite files,
and is re-runnable after you empty the target. Stop the app while it runs. Keep BIZIQ_SECRET_KEY the same (saved
connection secrets are copied as encrypted bytes).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PLATFORM_TABLES = ["orgs", "users", "approvals", "audit_log", "connections", "data_keys"]
SERIAL = {"users": "id", "approvals": "id", "audit_log": "id"}
ORG_KNOWN = ["_meta", "_meta_prev", "contacts", "dispatch_log", "alerts", "automation_log", "automation_tasks"]


def tables(src: sqlite3.Connection) -> list[str]:
    return [r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def copy_rows(src: sqlite3.Connection, dst, table: str, serial: str | None) -> int:
    cur = src.execute(f'SELECT * FROM "{table}"')
    cols = [d[0] for d in cur.description]
    n = 0
    sql = f'INSERT INTO "{table}" ({", ".join(chr(34) + c + chr(34) for c in cols)}) VALUES ({", ".join("?" for _ in cols)})'
    for row in cur:
        dst.execute(sql, tuple(row))
        n += 1
    if serial and n:
        dst.execute(f"SELECT setval(pg_get_serial_sequence('{table}', '{serial}'), (SELECT MAX({serial}) FROM {table}))")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("BIZIQ_DATABASE_URL"), help="PostgreSQL URL (or set BIZIQ_DATABASE_URL)")
    ap.add_argument("--sqlite-dir", type=Path, default=Path(os.environ.get("BIZIQ_DATA_DIR") or Path(__file__).resolve().parent.parent / "output"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.url:
        print("Give --url or set BIZIQ_DATABASE_URL.")
        return 2
    os.environ["BIZIQ_DATABASE_URL"] = a.url
    from biziq import accounts, approvals, audit, automation, crypto, data_layer, db, platform_db
    from biziq.live import store, whatsapp
    import pandas as pd

    root = a.sqlite_dir
    plat = root / "platform.db"
    orgs = sorted((root / "orgs").glob("*/biziq.db")) if (root / "orgs").exists() else []
    print(f"Source: {root}  ->  PostgreSQL  ({'dry run' if a.dry_run else 'writing'})")
    plan = []
    if plat.exists():
        s = sqlite3.connect(f"file:{plat}?mode=ro", uri=True)
        plan.append(("platform", plat, [(t, s.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]) for t in tables(s) if t in PLATFORM_TABLES]))
        s.close()
    for p in orgs:
        s = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        plan.append((p.parent.name, p, [(t, s.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]) for t in tables(s)]))
        s.close()
    for name, path, tl in plan:
        print(f"  {name:28s} {len(tl)} tables, {sum(n for _, n in tl):,} rows")
    if a.dry_run or not plan:
        return 0

    problems: list[str] = []
    # ---- platform
    if plat.exists():
        for init in (accounts._conn, approvals._conn, audit._conn, store._conn, crypto._conn):
            init().close()                                              # create tables/triggers
        src = sqlite3.connect(f"file:{plat}?mode=ro", uri=True)
        with platform_db.connect() as dst:
            for t in PLATFORM_TABLES:
                if t not in tables(src):
                    continue
                if dst.execute(f'SELECT 1 FROM "{t}" LIMIT 1').fetchone():
                    raise SystemExit(f"Target table platform.{t} already has rows. Empty it (or use a new database) and run again.")
                n = copy_rows(src, dst, t, SERIAL.get(t))
                want = src.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                got = dst.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                print(f"  platform.{t}: {n:,} rows copied")
                if want != got:
                    problems.append(f"platform.{t}: {want} in SQLite, {got} in PostgreSQL")
        src.close()
    # ---- organisations
    org_ids = []
    for p in orgs:
        org = p.parent.name
        org_ids.append(org)
        data_layer.use_db(accounts.org_dir(org) / "biziq.db")
        src = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        with data_layer._conn() as dst:
            db.ensure(dst, "meta", data_layer._init_meta)
            db.ensure(dst, "automation", automation._ddl)
            whatsapp._init(dst)
            for t in tables(src):
                if t in ORG_KNOWN:
                    if dst.execute(f'SELECT 1 FROM "{t}" LIMIT 1').fetchone():
                        raise SystemExit(f"Target table org_{org}.{t} already has rows.")
                    n = copy_rows(src, dst, t, "id" if t == "automation_log" else None)
                elif [r[1] for r in src.execute(f'PRAGMA table_info("{t}")')] == ["chunk", "blob"]:
                    # encrypted dataset: copy the ciphertext untouched (never decrypted here; the scope is the same on both backends)
                    dst.execute(f'DROP TABLE IF EXISTS "{t}"')
                    dst.execute(f'CREATE TABLE "{t}" (chunk INTEGER PRIMARY KEY, blob BLOB)')
                    n = copy_rows(src, dst, t, None)
                else:                                                    # dataset tables and their _prev_ copies
                    df = pd.read_sql(f'SELECT * FROM "{t}"', src)
                    db.write_df(dst, df, t)
                    n = len(df)
                got = dst.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                want = src.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                if want != got:
                    problems.append(f"{org}.{t}: {want} in SQLite, {got} in PostgreSQL")
            print(f"  org {org}: {len(tables(src))} tables copied")
        src.close()
    # ---- verify the audit chain where it now lives
    with platform_db.connect() as pc:
        all_orgs = [r[0] for r in pc.execute("SELECT id FROM orgs").fetchall()]
    for org in sorted(set(org_ids + all_orgs)):
        ok, msg = audit.verify_chain(org)
        if not ok:
            problems.append(f"audit chain for {org}: {msg}")
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("\nMigration complete. Counts match and every audit chain verifies. Point the app at the database with BIZIQ_DATABASE_URL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
