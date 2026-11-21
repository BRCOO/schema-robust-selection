#!/usr/bin/env python3
"""Fused (one-scan) execution of a registry and of the K-statistics model (third review of 2026-11-11, substance item 2;
post hoc). Same eICU AKI cohort, shipped registry, K-statistics model, engines and fresh-process protocol as
sql_engine_timing.py, whose cache it reads (run ``sql_engine_timing.py prep`` first).

  python scripts/v2/paper_v3/sql_fused_timing.py check           # exactness of the fused queries, x1, all engines
  python scripts/v2/paper_v3/sql_fused_timing.py drive 5 15      # rounds, pause (s) -> artifacts/v2/sql_fused/timing.json
  (the driver calls ``run <engine> <replicate>`` in fresh processes)

check: (i) the shipped AKI registry resolved at eICU: fused firings against the Python reference firings, fused registry
scores and fused K-statistics scores against the Python models; (ii) every predicate of the AKI union pool that executes at
eICU: one fused query against the per-program portable queries; both in DuckDB, PostgreSQL and SQLite.
run: per-program and fused scoring of the registry and of the K-statistics model, and the export of the observations the
registry reads to the client (what scoring outside the database would first move), per engine and replicate.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "artifacts/v2/sql_engines/timing_cache.pkl"
OUT = ROOT / "artifacts/v2/sql_fused"
WORK = OUT / "work"
PG_SITE = ROOT / "tmp/pg_site"
POOL_RUN = ("union", "f_mimic_iv_aki_gap6h")
CONFIGS = [("duckdb_memory", 1), ("duckdb_memory", 10), ("duckdb_disk", 1), ("duckdb_disk", 10), ("duckdb_disk", 30), ("duckdb_disk", 56),
           ("postgres", 1), ("postgres", 10), ("sqlite", 1), ("sqlite", 10)]


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _cache():
    with open(SRC, "rb") as h:
        return pickle.load(h)


def _connect(engine: str, pg_uri: str | None):
    from src.icu.v2 import sql_engines as eng

    base = "duckdb" if engine.startswith("duckdb") else engine
    shutil.rmtree(WORK / engine, ignore_errors=True)
    (WORK / engine).mkdir(parents=True)
    path = str(WORK / engine / "db.duckdb") if engine == "duckdb_disk" else (str(WORK / engine / "db.sqlite") if engine == "sqlite" else None)
    if base == "postgres":
        sys.path.insert(0, str(PG_SITE))
    con = eng.connect(base, threads=6, pg_uri=pg_uri, path=path)
    if engine == "duckdb_disk":
        con.execute("SET memory_limit = '8GB'")
        con.execute(f"SET temp_directory = '{(WORK / engine / 'spill').as_posix()}'")
    return base, con


def _close(con, engine: str) -> None:
    for table in ("tokens", "edges", "obs"):
        con.execute(f"DROP TABLE IF EXISTS {table}")
    con.close()
    shutil.rmtree(WORK / engine, ignore_errors=True)


def _firing_sets(rows, n: int) -> list[set]:
    out = [set() for _ in range(n)]
    for row in rows:
        e = row[0]
        for i in range(n):
            if int(row[1 + i]) == 1:
                out[i].add(e)
    return out


def check(pg_uri: str | None) -> dict:
    from src.icu.v2 import sql_engines as eng
    from src.icu.v2.sql_fused import fused_predicates_sql, fused_registry_score_sql, fused_statistic_score_sql
    from src.icu.v2.sql_registry import compilable, fire

    c = _cache()
    ctx = _module("e3_context", ROOT / "scripts/v2/e3_context.py")
    run = ctx.load_run(*ctx.run_dirs(*POOL_RUN))
    pool = [p for p in run.candidates if p.kind == "predicate" and frozenset(p.features) <= frozenset(c["schema"]) and compilable(p)]
    shipped = [b for b in c["resolved"] if b is not None]
    assert all(b.kind == "predicate" for b in shipped)
    k = c["kstats"]
    report = {"n_pool_predicates": len(pool), "n_episodes": len(c["eids"]), "engines": {}}
    for engine, dia in (("duckdb_memory", "duckdb"), ("duckdb_memory", "portable"), ("postgres", "portable"), ("sqlite", "portable")):
        base, con = _connect(engine, pg_uri)
        eng.load(con, base, c["arrays"], c["schema"], c["bins"])
        r = {"dialect": dia}
        fused = _firing_sets(con.execute(fused_predicates_sql(shipped, dia)).fetchall(), len(shipped))
        r["shipped_firing_mismatches"] = sum(fused[i] != c["reg_fired"][b.program_id] for i, b in enumerate(shipped))
        reg = con.execute(fused_registry_score_sql(c["resolved"], c["reg_coef"], c["reg_intercept"], dia)).fetchall()
        r["registry_score_max_abs_diff"] = max(abs(float(s) - c["reg_ref"][e]) for e, s in reg)
        kst = con.execute(fused_statistic_score_sql(k["columns"], k["defaults"], k["means"], k["scales"], k["coef"], k["intercept"], dia)).fetchall()
        r["kstats_score_max_abs_diff"] = max(abs(float(s) - c["k_ref"][e]) for e, s in kst)
        r["n_scored"] = [len(reg), len(kst)]
        t0 = time.perf_counter()
        per_program = fire(con, pool, c["schema"], dialect=dia)
        r["seconds_pool_per_program"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        fused_pool = _firing_sets(con.execute(fused_predicates_sql(pool, dia)).fetchall(), len(pool))
        r["seconds_pool_fused"] = time.perf_counter() - t0
        r["pool_firing_mismatches"] = sum(fused_pool[i] != set(per_program[p.program_id]) for i, p in enumerate(pool))
        r["pool_decisions"] = len(pool) * len(c["eids"])
        r["version"] = eng.versions(con, base)
        _close(con, engine)
        report["engines"][f"{engine}:{dia}"] = r
        print(engine, dia, json.dumps(r), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exactness.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def _peak_sampler(proc, peak, stop):
    while not stop.is_set():
        peak["rss"] = max(peak["rss"], proc.memory_info().rss)
        time.sleep(0.1)


def measure(engine: str, replicate: int) -> None:
    from src.icu.v2 import sql_engines as eng
    from src.icu.v2.sql_fused import fused_registry_score_sql, fused_statistic_score_sql, per_program_score_sql
    from src.icu.v2.sql_models import statistic_score_sql

    c = _cache()
    proc = psutil.Process()
    peak, stop = {"rss": proc.memory_info().rss}, threading.Event()
    th = threading.Thread(target=_peak_sampler, args=(proc, peak, stop), daemon=True)
    row = {"engine": engine, "replicate": replicate, "n_episodes": len(c["eids"]) * replicate}
    th.start()
    base, con = _connect(engine, os.environ.get("PG_URI"))
    row.update(eng.load(con, base, c["arrays"], c["schema"], c["bins"], replicate=replicate))
    k = c["kstats"]
    dia = eng.dialect(base)  # each engine in its best dialect: DuckDB's own, the portable one in the row stores
    row["dialect"] = dia
    queries = {"registry_per_program": per_program_score_sql(c["resolved"], c["reg_coef"], c["reg_intercept"], dia),
               "registry_fused": fused_registry_score_sql(c["resolved"], c["reg_coef"], c["reg_intercept"], dia),
               "kstats_per_statistic": statistic_score_sql(k["columns"], k["defaults"], k["means"], k["scales"], k["coef"], k["intercept"]),
               "kstats_fused": fused_statistic_score_sql(k["columns"], k["defaults"], k["means"], k["scales"], k["coef"], k["intercept"], dia)}
    order = list(queries)
    random.Random(replicate).shuffle(order)
    scores = {}
    for name in order:
        t0 = time.perf_counter()
        scores[name] = con.execute(queries[name]).fetchall()
        row[f"seconds_{name}"] = time.perf_counter() - t0
    feats = sorted({b.params["feature"] for b in c["resolved"] if b is not None})
    export = "SELECT episode_id, t, feature, value FROM obs WHERE feature IN (" + ", ".join("'" + f + "'" for f in feats) + ")"
    t0 = time.perf_counter()
    n_rows = 0
    if base == "duckdb":  # streamed in Arrow batches, so that a million episodes do not need their rows in memory at once
        for batch in con.sql(export).to_arrow_reader(1_000_000):
            n_rows += batch.num_rows
    else:
        cur = con.execute(export)
        while True:
            chunk = cur.fetchmany(200_000)
            if not chunk:
                break
            n_rows += len(chunk)
    row["seconds_export_registry_rows"] = time.perf_counter() - t0
    row["n_export_rows"] = int(n_rows)
    # the fused and per-program scores must agree at every replicate
    a = dict((e, float(s)) for e, s in scores["registry_per_program"])
    row["registry_fused_vs_per_program_max_abs_diff"] = max(abs(float(s) - a[e]) for e, s in scores["registry_fused"])
    b = dict((e, float(s)) for e, s in scores["kstats_per_statistic"])
    row["kstats_fused_vs_per_statistic_max_abs_diff"] = max(abs(float(s) - b[e]) for e, s in scores["kstats_fused"])
    row["version"] = eng.versions(con, base)
    _close(con, engine)
    stop.set()
    th.join()
    row.update({"peak_rss_gb": peak["rss"] / 2**30, "cpu_mhz": psutil.cpu_freq().current})
    print(json.dumps(row), flush=True)


def _pg_server():
    sys.path.insert(0, str(PG_SITE))
    import pgserver

    pgdata = OUT / "pgdata"
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(pgdata, ignore_errors=True)
    return pgserver.get_server(str(pgdata), cleanup_mode="delete")


def drive(rounds: int, pause: float) -> None:
    srv = _pg_server()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", PG_URI=srv.get_uri())
    out = OUT / "timing.json"
    rows, rng = [], random.Random(0)
    try:
        for r in range(rounds):
            order = CONFIGS[:]
            rng.shuffle(order)
            for engine, rep in order:
                res = subprocess.run([sys.executable, __file__, "run", engine, str(rep)], env=env, capture_output=True, text=True, encoding="utf-8")
                line = next((x for x in res.stdout.splitlines()[::-1] if x.startswith("{")), None)
                if res.returncode != 0 or line is None:
                    print(f"[fused] round {r} {engine} x{rep}: FAILED\n{res.stderr[-2000:]}", flush=True)
                    rows.append({"engine": engine, "replicate": rep, "round": r, "failed": True})
                else:
                    row = json.loads(line)
                    row["round"] = r
                    rows.append(row)
                    print(f"[fused] round {r} {engine} x{rep}: registry {row['seconds_registry_per_program']:.2f} -> {row['seconds_registry_fused']:.2f} s, "
                          f"kstats {row['seconds_kstats_per_statistic']:.2f} -> {row['seconds_kstats_fused']:.2f} s, export {row['seconds_export_registry_rows']:.2f} s",
                          flush=True)
                out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
                time.sleep(pause)
    finally:
        srv.cleanup()
        shutil.rmtree(OUT / "pgdata", ignore_errors=True)


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "check":
        srv = _pg_server()
        try:
            check(srv.get_uri())
        finally:
            srv.cleanup()
            shutil.rmtree(OUT / "pgdata", ignore_errors=True)
    elif cmd == "run":
        measure(sys.argv[2], int(sys.argv[3]))
    elif cmd == "drive":
        drive(int(sys.argv[2]), float(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
