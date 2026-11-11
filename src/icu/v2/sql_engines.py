"""Load a target's observations into an SQL engine and derive its event tokens there (review of 2026-11-09, item 1).

Engines: ``duckdb`` (the paper's dialect), ``duckdb_portable``, ``postgres`` and ``sqlite`` (the portable dialect of
``sql_registry``). Every engine receives the same long table ``obs(episode_id, t, feature, value)`` (a dense grid, value
NULL where unobserved) and ``edges(feature, c1, c2, c3)``; replicated cohorts are built inside the engine. The row stores
get the indexes a database administrator would add for these queries: ``obs(feature, episode_id, t)`` and
``tokens(token, episode_id, t)``; DuckDB needs none. PostgreSQL tables are UNLOGGED (no write-ahead log for a staging
table) and sessions use ``work_mem = 256MB``; SQLite runs with journaling off and a 1 GB page cache.
"""
from __future__ import annotations

import math
import sqlite3
import time
from typing import Any, AbstractSet, Mapping

import numpy as np
import pandas as pd

from src.icu.v2.sql_registry import edges_rows, obs_rows, register_sqlite_functions, token_sql

ENGINES = ("duckdb", "duckdb_portable", "postgres", "sqlite")


def dialect(engine: str) -> str:
    return "duckdb" if engine == "duckdb" else "portable"


def connect(engine: str, *, threads: int = 6, pg_uri: str | None = None, path: str | None = None):
    """A connection with ``execute(sql).fetchall()``. ``path``: a database file (DuckDB, SQLite); ``pg_uri``: PostgreSQL."""
    if engine in ("duckdb", "duckdb_portable"):
        import duckdb

        con = duckdb.connect(path) if path else duckdb.connect()
        con.execute(f"SET threads TO {threads}")
        return con
    if engine == "sqlite":
        con = sqlite3.connect(path or ":memory:", isolation_level=None)
        for pragma in ("journal_mode = OFF", "synchronous = OFF", "temp_store = MEMORY", "cache_size = -1048576"):
            con.execute(f"PRAGMA {pragma}")
        register_sqlite_functions(con)
        return con
    if engine == "postgres":
        import psycopg

        con = psycopg.connect(pg_uri, autocommit=True)
        con.execute("SET work_mem = '256MB'")
        return con
    raise ValueError(engine)


def _drop(con, table: str) -> None:
    con.execute(f"DROP TABLE IF EXISTS {table}")


def load(con, engine: str, arrays: Mapping[str, tuple], schema: AbstractSet[str], bins: Mapping[str, np.ndarray], replicate: int = 1) -> dict[str, Any]:
    """Create ``obs`` and ``edges`` (and the row stores' index on ``obs``); returns the load and index seconds."""
    t0 = time.perf_counter()
    ep, tt, ff, vv = obs_rows(arrays, schema)
    for table in ("obs", "obs0", "rep", "edges", "tokens"):
        _drop(con, table)
    unlogged = "UNLOGGED " if engine == "postgres" else ""
    if engine in ("duckdb", "duckdb_portable"):
        frame = pd.DataFrame({"episode_id": ep, "t": np.asarray(tt, dtype=np.int32), "feature": ff, "value": np.asarray(vv, dtype=float)})
        con.register("obs_df", frame)
        con.execute("CREATE TABLE obs0 AS SELECT episode_id, t, feature, CASE WHEN isnan(value) THEN NULL ELSE value END AS value FROM obs_df")
        con.unregister("obs_df")
    elif engine == "sqlite":
        con.execute("CREATE TABLE obs0 (episode_id TEXT, t INTEGER, feature TEXT, value REAL)")
        con.execute("BEGIN")
        con.executemany("INSERT INTO obs0 VALUES (?, ?, ?, ?)",
                        ((e, int(t), f, None if math.isnan(v) else float(v)) for e, t, f, v in zip(ep, tt, ff, vv)))
        con.execute("COMMIT")
    else:
        con.execute(f"CREATE {unlogged}TABLE obs0 (episode_id text, t integer, feature text, value double precision)")
        with con.cursor() as cur:
            with cur.copy("COPY obs0 (episode_id, t, feature, value) FROM STDIN") as copy:
                for e, t, f, v in zip(ep, tt, ff, vv):
                    copy.write_row((e, int(t), f, None if math.isnan(v) else float(v)))
    if replicate > 1:
        con.execute(f"CREATE {unlogged}TABLE rep (i INTEGER)")
        if engine == "postgres":
            con.execute(f"INSERT INTO rep SELECT generate_series(0, {replicate - 1})")
        else:
            con.executemany("INSERT INTO rep VALUES (?)", [(i,) for i in range(replicate)])
        con.execute(f"CREATE {unlogged}TABLE obs AS SELECT o.episode_id || '#' || CAST(r.i AS VARCHAR) AS episode_id, o.t, o.feature, o.value "
                    f"FROM obs0 o CROSS JOIN rep r")
        _drop(con, "obs0")
        _drop(con, "rep")
    else:
        con.execute("ALTER TABLE obs0 RENAME TO obs")
    con.execute(f"CREATE {unlogged}TABLE edges (feature VARCHAR, c1 DOUBLE PRECISION, c2 DOUBLE PRECISION, c3 DOUBLE PRECISION)")
    rows = edges_rows(bins)
    if rows:
        if engine == "postgres":
            with con.cursor() as cur:
                cur.executemany("INSERT INTO edges VALUES (%s, %s, %s, %s)", rows)
        else:
            con.executemany("INSERT INTO edges VALUES (?, ?, ?, ?)", rows)
    seconds_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    if engine in ("postgres", "sqlite"):
        con.execute("CREATE INDEX obs_fet ON obs (feature, episode_id, t)")
        con.execute("ANALYZE obs" if engine == "postgres" else "ANALYZE")
    seconds_index = time.perf_counter() - t1
    n_obs = con.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    return {"seconds_load": seconds_load, "seconds_index": seconds_index, "n_obs_rows": int(n_obs)}


def tokens(con, engine: str) -> dict[str, Any]:
    """Derive ``tokens`` in the engine (and its index on the row stores); returns the seconds and the row count."""
    t0 = time.perf_counter()
    if engine != "duckdb":
        _drop(con, "tokens")
    sql = token_sql(dialect=dialect(engine))
    if engine == "postgres":
        sql = sql.replace("CREATE TABLE tokens AS", "CREATE UNLOGGED TABLE tokens AS", 1)
    con.execute(sql)
    if engine in ("postgres", "sqlite"):
        con.execute("CREATE INDEX tok_idx ON tokens (token, episode_id, t)")
        con.execute("ANALYZE tokens" if engine == "postgres" else "ANALYZE")
    seconds = time.perf_counter() - t0
    n_tok = con.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]
    return {"seconds_tokens": seconds, "n_tokens": int(n_tok)}


def versions(con, engine: str) -> str:
    if engine in ("duckdb", "duckdb_portable"):
        import duckdb

        return f"DuckDB {duckdb.__version__}"
    if engine == "sqlite":
        return f"SQLite {sqlite3.sqlite_version}"
    return con.execute("SELECT version()").fetchone()[0].split(",")[0]
