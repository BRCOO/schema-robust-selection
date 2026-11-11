#!/usr/bin/env python3
"""In-database execution on four engines (review of 2026-11-09, item 1), fresh-process protocol as sql_timing_driver.py.

The shipped SRS+T registry of the AKI run (K = 10) resolved at eICU, and the K-statistics model of the same run (the ten
window statistics of the L1 path, refitted; run_stat_baselines.py), both scored entirely in SQL on eICU's AKI cohort
replicated up to 10 times (and 30 times on the disk-backed engines), in DuckDB (in memory and disk-backed, 8 GB limit),
PostgreSQL 16 (pgserver, 127.0.0.1 only, UNLOGGED tables) and SQLite, plus the Python reference engine. Each measurement
runs in a fresh process; configurations are interleaved in a random order per round with a pause between runs.

  python scripts/v2/paper_v3/sql_engine_timing.py prep          # model parameters, reference scores -> cache
  python scripts/v2/paper_v3/sql_engine_timing.py drive 5 15    # rounds, pause (s) -> artifacts/v2/sql_engines/timing.json
  (the driver calls ``run <engine> <replicate>`` in fresh processes)

At replicate 1 each engine's registry firings and both models' scores are checked against the Python reference.
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

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts/v2/sql_engines"
CACHE = OUT / "timing_cache.pkl"
WORK = OUT / "timing_work"
PG_SITE = ROOT / "tmp/pg_site"
RUN = ("union", "f_mimic_iv_aki_gap6h")
CONFIGS = [("duckdb_memory", 1), ("duckdb_memory", 10), ("duckdb_disk", 1), ("duckdb_disk", 10), ("duckdb_disk", 30),
           ("postgres", 1), ("postgres", 10), ("sqlite", 1), ("sqlite", 10), ("python", 1), ("python", 10)]


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def prep() -> None:
    ctx = _module("e3_context", ROOT / "scripts/v2/e3_context.py")
    sqlx = _module("sqlx", ROOT / "scripts/v2/run_sql_execution.py")
    stats = _module("run_stat_baselines", ROOT / "scripts/v2/run_stat_baselines.py")
    from src.icu.v2.sql_models import statistic_default
    from src.icu.v2.supervised import FileArrayLoader, build_external_cohort, build_supervised_inputs, load_episode_list, read_feature_names

    run = ctx.load_run(*ctx.run_dirs(*RUN))
    config = json.loads((run.pools_path.parent / "config.json").read_text(encoding="utf-8"))
    train = sqlx.load_arrays(config["episode_dir"], run.ids["train"])
    names = next(iter(train.values()))[2]
    bins = sqlx.fit_feature_bins([a[0] for a in train.values()], names, n_bins=int(config["representation"]["n_bins"]),
                                 missing_masks=[a[1] for a in train.values()])
    del train
    target = next(t for t in run.targets if t.name == "eicu")
    eids = [e.episode_id for e in target.episodes]
    arrays = sqlx.load_arrays(config["targets"]["eicu"]["episode_dir"], eids)
    schema = frozenset(target.features)
    comp = run.registries["srs_ap_t"]
    resolved = comp.resolve(schema)
    bridge = run.fit(list(comp.primaries))
    model = bridge["model_bundle"]["model"]
    X = run.matrix(resolved, eids, run.external["eicu"])
    reg_ref = dict(zip(eids, model.decision_function(X)))
    reg_fired = {b.program_id: run.external["eicu"].get(b.program_id, frozenset()) & frozenset(eids) for b in resolved if b is not None}
    # the K-statistics model of the same run, refitted exactly as run_stat_baselines.py fits it
    tag = "aki_gap6h"
    src_dir = ROOT / f"data/processed/v2/extract/mimic_iv_{tag}_full"
    episodes = load_episode_list(ROOT / f"artifacts/v2/task/f_mimic_iv_{tag}/episodes_split.json")
    inputs = build_supervised_inputs(episodes, read_feature_names(src_dir, episodes[0].episode_id), FileArrayLoader(src_dir), excluded_features=())
    tr, va = inputs.splits["train"], inputs.splits["val"]
    cols = stats.l1_columns(tr.tabular, tr.y, 10)
    kmodel, info = stats.fit_lr(tr.tabular[:, cols], tr.y, va.tabular[:, cols], va.y)
    saved = json.loads((ROOT / f"artifacts/v2/stat_baselines/f_mimic_iv_{tag}/rows.json").read_text(encoding="utf-8"))["details"]["stats_l1_k10"]
    col_names = [inputs.tabular_feature_names[c] for c in cols]
    assert col_names == saved["columns"], (col_names, saved["columns"])
    ext_dir = ROOT / f"data/processed/v2/extract/eicu_{tag}_full"
    ext_eps = load_episode_list(ROOT / f"artifacts/v2/task/f_eicu_{tag}/episodes_all.json")
    cohort = build_external_cohort("eicu", ext_eps, FileArrayLoader(ext_dir), inputs, feature_names=read_feature_names(ext_dir, ext_eps[0].episode_id))
    k_ref = dict(zip(cohort.arrays.episode_ids, kmodel.decision_function(cohort.arrays.tabular[:, cols])))
    assert set(k_ref) == set(eids), "the K-statistics cohort and the registry's eICU episodes differ"
    scaler, clf = kmodel.named_steps["scaler"], kmodel.named_steps["clf"]
    columns = [tuple(n.split("__", 1)) for n in col_names]
    spec = {"columns": columns, "defaults": [statistic_default(s, inputs.preprocessing.tabular_medians[c]) for (f, s), c in zip(columns, cols)],
            "means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as h:
        pickle.dump({"eids": eids, "arrays": arrays, "bins": bins, "schema": schema, "resolved": resolved, "reg_coef": model.coef_[0].tolist(),
                     "reg_intercept": float(model.intercept_[0]), "reg_ref": reg_ref, "reg_fired": reg_fired, "kstats": spec, "k_ref": k_ref,
                     "k_info": {"columns": col_names, "C": info.get("C", info.get("c"))}}, h)
    print("wrote", CACHE, len(eids), "episodes;", sum(b is not None for b in resolved), "resolved programs;", "K-statistics columns", col_names)


def _peak_sampler(proc, peak, stop):
    while not stop.is_set():
        peak["rss"] = max(peak["rss"], proc.memory_info().rss)
        time.sleep(0.1)


def measure(engine: str, replicate: int) -> None:
    from src.icu.v2 import sql_engines as eng
    from src.icu.v2.sql_models import registry_score_sql, statistic_score_sql
    from src.icu.v2.sql_registry import compilable, fire

    with open(CACHE, "rb") as h:
        c = pickle.load(h)
    proc = psutil.Process()
    peak, stop = {"rss": proc.memory_info().rss}, threading.Event()
    th = threading.Thread(target=_peak_sampler, args=(proc, peak, stop), daemon=True)
    row = {"engine": engine, "replicate": replicate, "n_episodes": len(c["eids"]) * replicate}
    resolved = [b for b in c["resolved"] if b is not None and compilable(b)]
    if engine == "python":
        from src.icu.v2.baselines.predicates import predicate_firing_sets, statistic_table
        from src.icu.v2.events import eventize_episode
        from src.icu.v2.numeric_firing import numeric_firing_sets
        from src.icu.v2.patterns import firing_sets

        episodes = {f"{e}#{i}": c["arrays"][e] for i in range(replicate) for e in c["eids"]} if replicate > 1 else c["arrays"]
        symbolic = [b for b in resolved if b.kind in ("sequence", "ordered_rule", "unordered_rule")]
        numeric = [b for b in resolved if b.kind in ("motif", "shapelet")]
        predicate = [b for b in resolved if b.kind == "predicate"]
        th.start()
        t0 = time.perf_counter()
        seqs = {e: eventize_episode(v, m, f, c["bins"], observation_tokens=True, trend_tokens=False) for e, (v, m, f) in episodes.items()}
        t1 = time.perf_counter()
        fired = {**firing_sets(symbolic, seqs), **numeric_firing_sets(numeric, episodes)}
        if predicate:
            ids, columns, table = statistic_table(episodes, sorted({b.params["feature"] for b in predicate}))
            fired.update(predicate_firing_sets(predicate, ids, columns, table))
        row.update({"seconds_tokens": t1 - t0, "seconds_registry": time.perf_counter() - t1})
    else:
        base = "duckdb" if engine.startswith("duckdb") else engine
        shutil.rmtree(WORK / engine, ignore_errors=True)
        (WORK / engine).mkdir(parents=True)
        path = None
        if engine == "duckdb_disk":
            path = str(WORK / engine / "db.duckdb")
        elif engine == "sqlite":
            path = str(WORK / engine / "db.sqlite")
        if base == "postgres":
            sys.path.insert(0, str(PG_SITE))
        th.start()
        con = eng.connect(base, threads=6, pg_uri=os.environ.get("PG_URI"), path=path)
        if engine == "duckdb_disk":
            con.execute("SET memory_limit = '8GB'")
            con.execute(f"SET temp_directory = '{(WORK / engine / 'spill').as_posix()}'")
        row.update(eng.load(con, base, c["arrays"], c["schema"], c["bins"], replicate=replicate))
        row.update(eng.tokens(con, base))
        t0 = time.perf_counter()
        fired = fire(con, resolved, c["schema"], dialect=eng.dialect(base))
        row["seconds_registry"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        reg = con.execute(registry_score_sql(c["resolved"], c["reg_coef"], c["reg_intercept"])).fetchall()
        row["seconds_score_registry"] = time.perf_counter() - t0
        k = c["kstats"]
        t0 = time.perf_counter()
        kst = con.execute(statistic_score_sql(k["columns"], k["defaults"], k["means"], k["scales"], k["coef"], k["intercept"])).fetchall()
        row["seconds_score_kstats"] = time.perf_counter() - t0
        row["version"] = eng.versions(con, base)
        if replicate == 1:  # exactness against the Python reference
            row["firing_mismatches"] = sum(fired[b.program_id] != c["reg_fired"][b.program_id] for b in resolved)
            row["registry_score_max_abs_diff"] = max(abs(float(s) - c["reg_ref"][e]) for e, s in reg)
            row["kstats_score_max_abs_diff"] = max(abs(float(s) - c["k_ref"][e]) for e, s in kst)
            row["n_scored"] = [len(reg), len(kst)]
        for table in ("tokens", "edges", "obs"):
            con.execute(f"DROP TABLE IF EXISTS {table}")
        con.close()
        shutil.rmtree(WORK / engine, ignore_errors=True)
    stop.set()
    th.join()
    row.update({"peak_rss_gb": peak["rss"] / 2**30, "cpu_mhz": psutil.cpu_freq().current})
    print(json.dumps(row), flush=True)


def drive(rounds: int, pause: float) -> None:
    sys.path.insert(0, str(PG_SITE))
    import pgserver

    pgdata = OUT / "pgdata_timing"
    shutil.rmtree(pgdata, ignore_errors=True)
    srv = pgserver.get_server(str(pgdata), cleanup_mode="delete")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", PG_URI=srv.get_uri())
    out = OUT / "timing.json"
    rows, rng = [], random.Random(0)
    try:
        for r in range(rounds):
            order = CONFIGS[:]
            rng.shuffle(order)
            for engine, rep in order:
                res = subprocess.run([sys.executable, str(Path(__file__).resolve()), "run", engine, str(rep)], capture_output=True, text=True, env=env, cwd=str(ROOT))
                line = [x for x in res.stdout.splitlines() if x.startswith("{")]
                if not line:
                    print("failed", engine, rep, res.stderr[-800:], flush=True)
                    continue
                row = json.loads(line[-1])
                row["round"] = r
                rows.append(row)
                print(json.dumps(row), flush=True)
                out.write_text(json.dumps({"protocol": "fresh process per measurement, random order per round, pause between runs",
                                           "pause_s": pause, "rows": rows}, indent=1), encoding="utf-8")
                time.sleep(pause)
    finally:
        srv.cleanup()
        shutil.rmtree(pgdata, ignore_errors=True)
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    elif sys.argv[1] == "run":
        measure(sys.argv[2], int(sys.argv[3]))
    else:
        drive(int(sys.argv[2]), float(sys.argv[3]))
