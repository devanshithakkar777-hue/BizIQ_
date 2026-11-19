"""Storage backend: SQLite (default, zero setup) or PostgreSQL (set BIZIQ_DATABASE_URL).

    BIZIQ_DATABASE_URL=postgresql://user:password@host:5432/biziq

Everything else in BizIQ talks to a connection object that behaves like sqlite3's (`c.execute(sql, params)` with `?`
placeholders, `fetchone`, `fetchall`, and `with c:` to commit and close). On PostgreSQL:

* the platform tables (organisations, users, approvals, audit log, connections) live in the schema `platform`;
* each organisation's business data lives in its own schema `org_<id>`, so one customer's tables are separate
  objects from another's (use database roles per schema if you need defence in depth, see docs/SECURITY.md);
* the search path is set per transaction, so a pooled connection can never leak from one organisation to another.

The path-based API of the SQLite days (`data_layer.use_db(path)`, `platform_db.PLATFORM_DB`) still works: a path is
mapped to a schema name, which is how the same test-suite runs on both backends.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd

URL = os.environ.get("BIZIQ_DATABASE_URL", "").strip()
IS_PG = URL.startswith(("postgres://", "postgresql://", "postgresql+psycopg://"))

_SAFE = re.compile(r"[^a-z0-9_]")


def schema_for(path: Path | str, default_platform: Path | None = None, default_local: Path | None = None) -> str:
    """Stable PostgreSQL schema name for a database path."""
    p = Path(path)
    if default_platform is not None and p == Path(default_platform):
        return "platform"
    if default_local is not None and p == Path(default_local):
        return "local"
    parts = p.parts
    if len(parts) >= 3 and parts[-3] == "orgs" and parts[-1] == "biziq.db":
        return "org_" + _SAFE.sub("_", parts[-2].lower())
    return "x_" + hashlib.sha1(str(p.resolve() if p.is_absolute() else p).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ PostgreSQL
_engine = None
_engine_lock = threading.Lock()
_initialised: set = set()
_schemas: set = set()


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from sqlalchemy import create_engine
            url = URL.replace("postgres://", "postgresql://", 1)
            if url.startswith("postgresql://"):
                url = "postgresql+psycopg://" + url[len("postgresql://"):]
            _engine = create_engine(url, pool_size=int(os.environ.get("BIZIQ_DB_POOL", "5")), max_overflow=10,
                                    pool_pre_ping=True, pool_recycle=1800)
        return _engine


def dispose() -> None:
    """Drop pooled connections (used by tests and after a fork)."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine.dispose()
            _engine = None
    _schemas.clear()
    _initialised.clear()


def _translate(sql: str, has_params: bool) -> str:
    sql = (sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
              .replace(" BLOB", " BYTEA").replace(" REAL", " DOUBLE PRECISION"))
    sql = sql.replace("%", "%%")                      # the driver reads % as a placeholder marker even with no parameters
    if has_params:
        sql = sql.replace("?", "%s")
    return sql


def _py(v):
    return v.item() if isinstance(v, np.generic) else v


class PGConn:
    """sqlite3-style connection over PostgreSQL. `with PGConn(...) as c:` commits (or rolls back) and releases it."""

    def __init__(self, sa_conn, schema: str):
        self.sa, self.schema = sa_conn, schema

    @property
    def raw(self):
        return self.sa

    def execute(self, sql: str, params=()):
        params = tuple(_py(p) for p in params) if params else ()
        return self.sa.exec_driver_sql(_translate(sql, bool(params)), params or None)

    def commit(self):
        self.sa.commit()

    def rollback(self):
        self.sa.rollback()

    def close(self):
        self.sa.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.sa.commit()
            else:
                self.sa.rollback()
        finally:
            self.sa.close()
        return False


def pg_connect(schema: str) -> PGConn:
    if not re.fullmatch(r"[a-z0-9_]{1,60}", schema):
        raise ValueError("bad schema name")
    eng = _get_engine()
    if schema not in _schemas:
        with eng.connect() as boot:
            boot.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            boot.commit()
        _schemas.add(schema)
    sa = eng.connect()
    sa.exec_driver_sql(f'SET LOCAL search_path TO "{schema}"')
    return PGConn(sa, schema)


# ------------------------------------------------------------------------------------------------ helpers both backends share
def ensure(c, name: str, fn) -> None:
    """Run table-creation `fn(c)`. SQLite: every time (cheap). PostgreSQL: once per process per schema."""
    if not isinstance(c, PGConn):
        fn(c)
        return
    key = (c.schema, name)
    if key in _initialised:
        return
    c.execute("SELECT pg_advisory_xact_lock(727)")           # one process creates tables at a time
    fn(c)
    c.commit()
    c.sa.exec_driver_sql(f'SET LOCAL search_path TO "{c.schema}"')
    _initialised.add(key)


def table_exists(c, name: str) -> bool:
    if isinstance(c, PGConn):
        return c.execute("SELECT 1 FROM information_schema.tables WHERE table_schema=? AND table_name=?", (c.schema, name)).fetchone() is not None
    return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def columns(c, table: str) -> list[str]:
    if isinstance(c, PGConn):
        return [r[0] for r in c.execute("SELECT column_name FROM information_schema.columns WHERE table_schema=? AND table_name=? ORDER BY ordinal_position",
                                        (c.schema, table)).fetchall()]
    return [r[1] for r in c.execute(f'PRAGMA table_info("{table}")').fetchall()]


def add_column_if_missing(c, table: str, coldef: str) -> None:
    """coldef like 'totp_secret BLOB'. Works on both backends; used for upgrades of existing installs."""
    name = coldef.split()[0]
    if name not in columns(c, table):
        c.execute(f"ALTER TABLE {table} ADD COLUMN {coldef}")


def read_df(c, sql: str, params=()) -> pd.DataFrame:
    if isinstance(c, PGConn):
        if params:
            cur = c.execute(sql, params)
            return pd.DataFrame(cur.fetchall(), columns=list(cur.keys()))
        return pd.read_sql(sql, c.raw)
    return pd.read_sql(sql, c, params=list(params) if params else None)


def write_df(c, df: pd.DataFrame, table: str, chunk: int = 100_000) -> None:
    """Replace `table` with the frame. PostgreSQL uses COPY (about 50x faster than INSERT for large uploads)."""
    if not isinstance(c, PGConn):
        df.to_sql(table, c, if_exists="replace", index=False, chunksize=chunk)
        return
    c.execute(f'DROP TABLE IF EXISTS "{table}"')
    df.head(0).to_sql(table, c.raw, schema=c.schema, if_exists="fail", index=False)
    if not len(df):
        return
    dbapi = c.raw.connection.driver_connection
    cols = ", ".join(f'"{x}"' for x in df.columns)
    with dbapi.cursor() as cur, cur.copy(f'COPY "{c.schema}"."{table}" ({cols}) FROM STDIN WITH (FORMAT csv)') as cp:
        for i in range(0, len(df), chunk):
            buf = io.StringIO()
            df.iloc[i:i + chunk].to_csv(buf, header=False, index=False, na_rep="")
            cp.write(buf.getvalue())


def append_df(c, df: pd.DataFrame, table: str) -> None:
    if isinstance(c, PGConn):
        df.to_sql(table, c.raw, schema=c.schema, if_exists="append", index=False, method="multi", chunksize=5000)
    else:
        df.to_sql(table, c, if_exists="append", index=False)


def upsert_sql(table: str, cols: list[str], key: list[str]) -> str:
    """INSERT ... ON CONFLICT DO UPDATE with `?` placeholders; valid on SQLite 3.24+ and PostgreSQL."""
    upd = ", ".join(f'{x}=excluded.{x}' for x in cols if x not in key)
    return (f'INSERT INTO {table} ({", ".join(cols)}) VALUES ({", ".join("?" for _ in cols)}) '
            f'ON CONFLICT({", ".join(key)}) DO ' + (f"UPDATE SET {upd}" if upd else "NOTHING"))


def backend_name() -> str:
    return "postgresql" if IS_PG else "sqlite"
