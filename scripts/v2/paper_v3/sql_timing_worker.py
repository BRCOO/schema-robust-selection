"""Timing worker for registry execution (review item M10, second protocol): one measurement per fresh process.

  prep:   python sql_timing_worker.py prep            -> tmp/v2_helpers/sql_timing_cache.pkl (eICU AKI arrays, bins, resolved registry)
  run:    python sql_timing_worker.py run <engine> <replicate>   engine in duckdb_memory | duckdb_disk | python; prints one JSON row

The cache holds the eICU episodes of the AKI K = 10 run, the frozen TRAIN bin edges and the shipped SRS+T registry resolved
under the eICU schema, so that every measurement starts from the same state without reloading the pools.
"""
import importlib.util
import json
import pickle
import shutil
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "tmp/v2_helpers/sql_timing_cache.pkl"


def sqlx_module():
    spec = importlib.util.spec_from_file_location("sqlx", ROOT / "scripts/v2/run_sql_execution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prep() -> None:
    sqlx = sqlx_module()
    ctx = sqlx.ctx
    run = ctx.load_run(*ctx.run_dirs("union", "f_mimic_iv_aki_gap6h"))
    config = json.loads((run.pools_path.parent / "config.json").read_text(encoding="utf-8"))
    train = sqlx.load_arrays(config["episode_dir"], run.ids["train"])
    names = next(iter(train.values()))[2]
    bins = sqlx.fit_feature_bins([a[0] for a in train.values()], names, n_bins=int(config["representation"]["n_bins"]),
                                 missing_masks=[a[1] for a in train.values()])
    target = next(t for t in run.targets if t.name == "eicu")
    eids = [e.episode_id for e in target.episodes]
    arrays = sqlx.load_arrays(config["targets"]["eicu"]["episode_dir"], eids)
    schema = frozenset(target.features)
    resolved = [b for b in run.registries["srs_ap_t"].resolve(schema) if b is not None]
    with open(CACHE, "wb") as h:
        pickle.dump({"eids": eids, "arrays": arrays, "bins": bins, "schema": schema, "resolved": resolved}, h)
    print("wrote", CACHE, len(eids), "episodes,", len(resolved), "programs,", sum(sqlx.compilable(b) for b in resolved), "compiled")


def measure(engine: str, replicate: int) -> None:
    with open(CACHE, "rb") as h:
        c = pickle.load(h)
    proc = psutil.Process()
    peak, stop = {"rss": proc.memory_info().rss}, threading.Event()

    def sampler():
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], proc.memory_info().rss)
            time.sleep(0.1)

    th = threading.Thread(target=sampler, daemon=True)
    freq0 = psutil.cpu_freq().current
    row = {"engine": engine, "replicate": replicate, "n_episodes": len(c["eids"]) * replicate}
    if engine.startswith("duckdb"):
        import duckdb

        sqlx = sqlx_module()
        work = ROOT / f"tmp/v2_helpers/duckdb_worker_{engine}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        th.start()
        if engine == "duckdb_disk":
            con = duckdb.connect(str(work / "scale.duckdb"))
            con.execute("SET memory_limit = '8GB'")
            con.execute(f"SET temp_directory = '{(work / 'spill').as_posix()}'")
        else:
            con = duckdb.connect()
        con.execute("SET threads TO 6")
        loaded = sqlx.load_obs(con, c["arrays"], c["schema"], c["bins"], replicate=replicate)
        t0 = time.perf_counter()
        fired = sqlx.fire(con, [b for b in c["resolved"] if sqlx.compilable(b)], c["schema"])
        row.update({"seconds_load": loaded["seconds_load"], "seconds_tokens": loaded["seconds_tokens"], "seconds_registry": time.perf_counter() - t0,
                    "n_obs_rows": loaded["n_obs_rows"], "duckdb": duckdb.__version__})
        con.close()
        shutil.rmtree(work, ignore_errors=True)
    else:
        from src.icu.v2.events import eventize_episode
        from src.icu.v2.numeric_firing import numeric_firing_sets
        from src.icu.v2.patterns import firing_sets

        arrays = c["arrays"]
        episodes = {f"{e}#{i}": arrays[e] for i in range(replicate) for e in c["eids"]} if replicate > 1 else arrays
        symbolic = [b for b in c["resolved"] if b.kind in ("sequence", "ordered_rule", "unordered_rule")]
        numeric = [b for b in c["resolved"] if b.kind in ("motif", "shapelet")]
        th.start()
        t0 = time.perf_counter()
        seqs = {e: eventize_episode(v, m, f, c["bins"], observation_tokens=True, trend_tokens=False) for e, (v, m, f) in episodes.items()}
        t1 = time.perf_counter()
        fired = {**firing_sets(symbolic, seqs), **numeric_firing_sets(numeric, episodes)}
        row.update({"seconds_tokens": t1 - t0, "seconds_registry": time.perf_counter() - t1})
    stop.set()
    th.join()
    row.update({"peak_rss_gb": peak["rss"] / 2**30, "n_fired": int(sum(len(v) for v in fired.values())), "cpu_mhz_start": freq0,
                "cpu_mhz_end": psutil.cpu_freq().current})
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    else:
        measure(sys.argv[2], int(sys.argv[3]))
