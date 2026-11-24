"""Drive domain, post hoc (review of 2026-11-09): SRS+T against SRS (no target schemas) at every budget, deployed with surrogate
chains and bootstrapped exactly as the ablations of run_union_ap.py (same n_boot and seed). Backblaze data only. Run with four
BLAS threads, the configuration of the reported evaluation (average precision splits exact ties by floating-point noise).

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \\
      python scripts/v2/paper_v3/drive_srs_vs_srst.py

Output: artifacts/v2/union_ap/drive_k{K}/f_backblaze_seagate_gap2d/srs_pair.csv (read by tables.py, Table 6).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("run_union_ap", ROOT / "scripts/v2/run_union_ap.py")
ap = importlib.util.module_from_spec(spec)
sys.modules["run_union_ap"] = ap
spec.loader.exec_module(ap)
ctx = ap.ctx
sb = ctx.selection_baselines_module()

for k in (5, 10, 20):
    label = f"drive_k{k}/f_backblaze_seagate_gap2d"
    out = ap.OUT / label
    run = ctx.load_run(*ctx.drive_dirs(k))
    saved = json.loads((out / "registries.json").read_text(encoding="utf-8"))
    regs = {name: ctx._composite(saved[name], run.by_id) for name in ("srs_ap_t", "srs_ap")}
    probs = {name: ap.deploy(run, comp) for name, comp in regs.items()}
    y_test = np.asarray(run.labels["test"])
    g_test = np.array([e.subject_id for e in run.grouped["test"]])
    rows = []
    d = ap.paired_cluster_bootstrap_diff(y_test, probs["srs_ap_t"][0], probs["srs_ap"][0], g_test, metric="auprc", n_boot=1000, seed=0)
    rows.append({"run": label, "a": "srs_ap_t", "b": "srs_ap", "target": "internal", **d, "metric": "internal"})
    for record in sb.paired_target_bootstrap(run.targets, probs["srs_ap_t"][1], probs["srs_ap"][1], n_boot=1000, seed=0):
        rows.append({"run": label, "a": "srs_ap_t", "b": "srs_ap", **record, "metric": "target"})
    ctx.write_csv(out / "srs_pair.csv", rows)
    for r in rows:
        print(f"K={k} {r['target']:>18}: {100 * r['point_diff']:+.2f} [{100 * r['low']:+.2f}, {100 * r['high']:+.2f}]", flush=True)
