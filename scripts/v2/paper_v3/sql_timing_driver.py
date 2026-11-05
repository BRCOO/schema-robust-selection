"""Driver of the second timing protocol (review item M10): every measurement in a fresh process, configurations interleaved
in a random order per round, a pause between runs; writes artifacts/v2/sql_execution/timing_repeats_f_mimic_iv_aki_gap6h.json.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/sql_timing_driver.py [rounds] [pause_s]
"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable
WORKER = ROOT / "scripts/v2/paper_v3/sql_timing_worker.py"
CONFIGS = [("duckdb_memory", 1), ("duckdb_memory", 10), ("duckdb_disk", 1), ("duckdb_disk", 10), ("duckdb_disk", 30), ("python", 1), ("python", 10)]
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
pause = float(sys.argv[2]) if len(sys.argv) > 2 else 15
out = ROOT / "artifacts/v2/sql_execution/timing_repeats_f_mimic_iv_aki_gap6h.json"
rows = []
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
rng = random.Random(0)
for r in range(rounds):
    order = CONFIGS[:]
    rng.shuffle(order)
    for engine, rep in order:
        res = subprocess.run([PY, str(WORKER), "run", engine, str(rep)], capture_output=True, text=True, env=env, cwd=str(ROOT))
        line = [x for x in res.stdout.splitlines() if x.startswith("{")]
        if not line:
            print("failed", engine, rep, res.stderr[-500:], flush=True)
            continue
        row = json.loads(line[-1])
        row["round"] = r
        rows.append(row)
        print(json.dumps(row), flush=True)
        out.write_text(json.dumps({"protocol": "fresh process per measurement, random order per round, pause between runs",
                                   "pause_s": pause, "rows": rows}, indent=1), encoding="utf-8")
        time.sleep(pause)
print("wrote", out)
