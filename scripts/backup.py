"""Back up (and restore) every BizIQ SQLite database.

    python scripts/backup.py                    # copy all databases to <data dir>/backups/<timestamp>/
    python scripts/backup.py --keep 14          # keep the 14 newest backups (default 14), delete older ones
    python scripts/backup.py --dest /mnt/nas    # write somewhere else (a different disk is the point of a backup)
    python scripts/backup.py --list             # show backups
    python scripts/backup.py --restore 20260301-020000 --yes     # put a backup back (stop the app first)

With BIZIQ_DATABASE_URL set (PostgreSQL) the same commands use pg_dump / pg_restore: one custom-format dump per backup,
restored into a clean slate (BizIQ's own schemas are dropped first, nothing else is touched). pg_dump and pg_restore must be installed (postgresql-client).

Uses SQLite's online backup API, so it is safe while the app is running. Each copy is integrity-checked before it counts.
Schedule it daily (cron: `0 2 * * * python /app/scripts/backup.py`) and copy the folder off the machine.
Not backed up: the BIZIQ_SECRET_KEY used to encrypt saved connection secrets. Keep that in your password manager, or saved
connections cannot be decrypted after a restore.
"""
import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from biziq import platform_db  # noqa: E402

DATA = platform_db.OUT


def find_databases(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.db") if "backups" not in p.relative_to(root).parts)


def backup_one(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    a, b = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30), sqlite3.connect(dst)
    try:
        a.backup(b)
    finally:
        a.close(); b.close()
    c = sqlite3.connect(dst)
    try:
        ok = c.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        c.close()
    if ok != "ok":
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"integrity check failed for {src.name}: {ok}")


def _pg_url() -> str:
    from urllib.parse import urlsplit  # noqa: F401
    return os.environ["BIZIQ_DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)


def _pg_run(cmd: list[str]) -> None:
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{cmd[0]} failed: {r.stderr.strip()[:400]}")


def make_backup(data: Path, dest: Path, keep: int) -> Path:
    if os.environ.get("BIZIQ_DATABASE_URL"):
        out = dest / datetime.now().strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        _pg_run(["pg_dump", "--format=custom", "--no-owner", f"--file={out / 'biziq.pgdump'}", f"--dbname={_pg_url()}"])
        _pg_run(["pg_restore", "--list", str(out / "biziq.pgdump")])           # readable dump = a usable backup
        prune(dest, keep)
        return out
    dbs = find_databases(data)
    if not dbs:
        raise SystemExit(f"No databases found under {data}. Nothing to back up.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = dest / stamp
    for db in dbs:
        backup_one(db, out / db.relative_to(data))
    prune(dest, keep)
    return out


def prune(dest: Path, keep: int) -> None:
    runs = sorted(p for p in dest.iterdir() if p.is_dir() and p.name[:8].isdigit())
    for old in runs[:-keep] if keep > 0 else []:
        shutil.rmtree(old)


def restore(data: Path, dest: Path, stamp: str) -> list[str]:
    src = dest / stamp
    if not src.is_dir():
        raise SystemExit(f"No backup named {stamp} in {dest}.")
    if os.environ.get("BIZIQ_DATABASE_URL"):
        import psycopg
        with psycopg.connect(_pg_url(), autocommit=True) as conn:        # drop only BizIQ's own schemas, then load the dump
            names = [r[0] for r in conn.execute("SELECT schema_name FROM information_schema.schemata")]
            for s in names:
                if s in ("platform", "local") or s.startswith(("org_", "x_")):
                    conn.execute(f'DROP SCHEMA "{s}" CASCADE')
        _pg_run(["pg_restore", "--no-owner", f"--dbname={_pg_url()}", str(src / "biziq.pgdump")])
        return ["biziq.pgdump"]
    done = []
    safety = data / "backups" / f"pre-restore-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    for f in sorted(src.rglob("*.db")):
        target = data / f.relative_to(src)
        if target.exists():                                   # keep what is there, in case the restore was a mistake
            safety_copy = safety / f.relative_to(src)
            safety_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, safety_copy)
        target.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("-wal", "-shm"):
            Path(str(target) + ext).unlink(missing_ok=True)
        shutil.copy2(f, target)
        done.append(str(f.relative_to(src)))
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA, help="data folder (default: BIZIQ_DATA_DIR or ./output)")
    ap.add_argument("--dest", type=Path, default=None, help="where backups go (default: <data>/backups)")
    ap.add_argument("--keep", type=int, default=int(os.environ.get("BIZIQ_BACKUP_KEEP", "14")))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--restore", metavar="STAMP")
    ap.add_argument("--yes", action="store_true", help="confirm a restore")
    a = ap.parse_args(argv)
    dest = a.dest or a.data / "backups"
    if a.list:
        runs = sorted(p.name for p in dest.iterdir() if p.is_dir()) if dest.exists() else []
        print("\n".join(runs) or "No backups yet.")
        return 0
    if a.restore:
        if not a.yes:
            print("Restoring overwrites the live databases (a copy of the current ones is kept under backups/pre-restore-*). Stop BizIQ first, then add --yes.")
            return 2
        print("Restored:", ", ".join(restore(a.data, dest, a.restore)))
        return 0
    out = make_backup(a.data, dest, a.keep)
    n = len(list(out.rglob("*.db"))) + len(list(out.rglob("*.pgdump")))
    print(f"Backed up {n} database(s) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
