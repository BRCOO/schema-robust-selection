"""SQL execution timings with repeats, in-memory and disk-backed DuckDB, and the Python reference engine on the same
episodes (review of the v3 experiments, 2026-11-05, item M10). The eICU AKI cohort of the K = 10 run, replicated x1, x10
(and x30 for DuckDB), the shipped SRS+T registry resolved under the eICU schema. Every DuckDB configuration runs in a fresh
connection; ``--repeats`` runs per configuration (median and IQR reported). The Python engine eventizes the episodes and fires
the resolved programs with the code that built the pools (src/icu/v2/events.py, patterns.py, numeric_firing.py).

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/sql_timing_repeats.py [--repeats 5] [--scales 1,10,30]
"""
import argparse
import importlib.util
import json
import platform
import shutil
import sys
import threading
import time
from pathlib import Path

import duckdb
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("sqlx", ROOT / "scripts/v2/run_sql_execution.py")
sqlx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sqlx)
ctx = sqlx.ctx

from src.icu.v2.events import eventize_episode  # noqa: E402
from src.icu.v2.numeric_firing import numeric_firing_sets  # noqa: E402
from src.icu.v2.patterns import firing_sets  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--label", default="f_mimic_iv_aki_gap6h")
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--scales", default="1,10,30")
parser.add_argument("--memory-scales", default="1,10", help="in-memory DuckDB stays below the free RAM")
parser.add_argument("--python-scales", default="1,10")
args = parser.parse_args()

run = ctx.load_run(*ctx.run_dirs("union", args.label))
config = json.loads((run.pools_path.parent / "config.json").read_text(encoding="utf-8"))
train = sqlx.load_arrays(config["episode_dir"], run.ids["train"])
names = next(iter(train.values()))[2]
bins = sqlx.fit_feature_bins([a[0] for a in train.values()], names, n_bins=int(config["representation"]["n_bins"]),
                             missing_masks=[a[1] for a in train.values()])
del train
target = next(t for t in run.targets if t.name == "eicu")
eids = [e.episode_id for e in target.episodes]
schema = frozenset(target.features)
arrays = sqlx.load_arrays(config["targets"]["eicu"]["episode_dir"], eids)
comp = run.registries["srs_ap_t"]
branches = [b for b in comp.resolve(schema) if b is not None]
resolved = [b for b in branches if sqlx.compilable(b)]
info = {"label": args.label, "n_episodes": len(eids), "n_coordinates": len(comp.primaries), "n_resolved": len(branches),
        "n_compiled": len(resolved), "not_compiled": [b.program_id for b in branches if not sqlx.compilable(b)],
        "kinds": sorted({b.kind for b in resolved}), "duckdb": duckdb.__version__, "python": platform.python_version(),
        "cpu": platform.processor(), "cores_logical": psutil.cpu_count(), "ram_gb": psutil.virtual_memory().total / 2**30}
print(json.dumps(info), flush=True)
proc = psutil.Process()
work = ROOT / "tmp/v2_helpers/duckdb_repeats"
rows = []


def sampler(peak, stop):
    while not stop.is_set():
        peak["rss"] = max(peak["rss"], proc.memory_info().rss)
        time.sleep(0.1)


for mode in ("memory", "disk"):
    for r in [int(x) for x in (args.memory_scales if mode == "memory" else args.scales).split(",")]:
        for rep in range(args.repeats):
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            if mode == "disk":
                con = duckdb.connect(str(work / "scale.duckdb"))
                con.execute("SET memory_limit = '8GB'")
                con.execute(f"SET temp_directory = '{(work / 'spill').as_posix()}'")
            else:
                con = duckdb.connect()
            con.execute("SET threads TO 6")
            peak, stop = {"rss": 0}, threading.Event()
            th = threading.Thread(target=sampler, args=(peak, stop), daemon=True)
            th.start()
            loaded = sqlx.load_obs(con, arrays, schema, bins, replicate=r)
            t0 = time.perf_counter()
            fired = sqlx.fire(con, resolved, schema)
            reg = time.perf_counter() - t0
            stop.set()
            th.join()
            con.close()
            row = {"engine": f"duckdb_{mode}", "replicate": r, "repeat": rep, "n_episodes": len(eids) * r, "n_obs_rows": loaded["n_obs_rows"],
                   "seconds_load": loaded["seconds_load"], "seconds_tokens": loaded["seconds_tokens"], "seconds_registry": reg,
                   "peak_rss_gb": peak["rss"] / 2**30, "n_fired": int(sum(len(v) for v in fired.values()))}
            rows.append(row)
            print(json.dumps(row), flush=True)
shutil.rmtree(work, ignore_errors=True)

symbolic = [b for b in resolved if b.kind in ("sequence", "ordered_rule", "unordered_rule")]
numeric = [b for b in resolved if b.kind in ("motif", "shapelet")]
for r in [int(x) for x in args.python_scales.split(",")]:
    episodes = {f"{e}#{i}": arrays[e] for i in range(r) for e in eids} if r > 1 else arrays
    for rep in range(args.repeats):
        t0 = time.perf_counter()
        seqs = {e: eventize_episode(v, m, f, bins, observation_tokens=True, trend_tokens=False) for e, (v, m, f) in episodes.items()}
        t_tok = time.perf_counter() - t0
        t1 = time.perf_counter()
        fired = {**firing_sets(symbolic, seqs), **numeric_firing_sets(numeric, episodes)}
        t_reg = time.perf_counter() - t1
        row = {"engine": "python", "replicate": r, "repeat": rep, "n_episodes": len(episodes), "seconds_tokens": t_tok,
               "seconds_registry": t_reg, "peak_rss_gb": proc.memory_info().rss / 2**30, "n_fired": int(sum(len(v) for v in fired.values()))}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del seqs
out = ROOT / f"artifacts/v2/sql_execution/timing_repeats_{args.label}.json"
out.write_text(json.dumps({"info": info, "rows": rows}, indent=1), encoding="utf-8")
print("wrote", out)
