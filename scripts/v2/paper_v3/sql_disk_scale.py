"""SQL scale-up with a disk-backed DuckDB database (review of the v3 experiments, 2026-11-05): the eICU episodes of a run
replicated x1, x10, x30 as in run_sql_execution.py, but the database lives in a file with a memory limit, so DuckDB
can spill instead of exhausting RAM. Times the event-token derivation and the shipped SRS+T registry; records peak RSS.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/sql_disk_scale.py <label> <memory|disk> [scales]
"""
import importlib.util
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import duckdb
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("sqlx", ROOT / "scripts/v2/run_sql_execution.py")
sqlx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sqlx)
ctx = sqlx.ctx

label, mode = sys.argv[1], sys.argv[2]
scales = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,10,30").split(",")]
run = ctx.load_run(*ctx.run_dirs("union", label))
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
resolved = [b for b in comp.resolve(schema) if b is not None and sqlx.compilable(b)]
proc = psutil.Process()
work = ROOT / "tmp/v2_helpers/duckdb_disk"
out_rows = []
for r in scales:
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
    peak = {"rss": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], proc.memory_info().rss)
            time.sleep(0.2)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    info = sqlx.load_obs(con, arrays, schema, bins, replicate=r)
    t0 = time.perf_counter()
    sqlx.fire(con, resolved, schema)
    reg = time.perf_counter() - t0
    stop.set()
    th.join()
    con.close()
    row = {"label": label, "mode": mode, "replicate": r, "n_episodes": len(eids) * r, "n_obs_rows": info["n_obs_rows"],
           "seconds_tokens": info["seconds_tokens"], "seconds_registry": reg, "peak_rss_gb": peak["rss"] / 2**30}
    out_rows.append(row)
    print(json.dumps(row), flush=True)
out = ROOT / f"artifacts/v2/sql_execution/disk_scale_{label}_{mode}.json"
out.write_text(json.dumps(out_rows, indent=1), encoding="utf-8")
shutil.rmtree(work, ignore_errors=True)
print("wrote", out)
