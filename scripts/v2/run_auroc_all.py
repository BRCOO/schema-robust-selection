#!/usr/bin/env python3
"""AUROC (and AUPRC, Brier) point estimates of every registry of Table 2 of the v3 paper, over the 16 ICU runs:
the twelve established selectors, SRS+T, SRS, the likelihood-decision SRS+T, the schema-unaware wrappers (likelihood:
fs_cv; AUPRC: srs_ap_unaware from union_ap/ablations.json) and the hybrid NMIFS+AP (artifacts/v2/nmifs_ap).
Deployed with surrogate chains exactly as in run_union_ap.py. Output: artifacts/v2/extra_metrics/<root>/<label>/rows_all.csv.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_auroc_all.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("e3_context", ROOT / "scripts/v2/e3_context.py")
ctx = importlib.util.module_from_spec(spec)
sys.modules["e3_context"] = ctx
spec.loader.exec_module(ctx)
spec = importlib.util.spec_from_file_location("run_union_ap", ROOT / "scripts/v2/run_union_ap.py")
ap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ap)

from src.icu.v2.stats import metric_value  # noqa: E402

OUT = ROOT / "artifacts/v2/extra_metrics"
TABLE = ("srs_ap_t", "srs_ap", "nmifs", "mrmr", "mrmr_stratified", "jmi", "cmim", "mmrfs", "mmr_ig", "mms_xin_ig", "dpp_ig",
         "l1_logistic", "l1_dropout", "mrmr_filter_known", "srs_c_val", "fs_cv", "srs_ap_unaware", "nmifs_ap_t")


def main() -> int:
    for root, label in ctx.ICU_RUNS:
        out = OUT / root / label
        if (out / "rows_all.csv").exists():
            print(f"[auroc] {root}/{label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(*ctx.run_dirs(root, label))
        regs = dict(run.registries)
        for name, reg in json.loads((ROOT / "artifacts/v2/union_ap" / root / label / "ablations.json").read_text(encoding="utf-8")).items():
            regs[name] = ctx._composite(reg, run.by_id)
        for name, reg in json.loads((ROOT / "artifacts/v2/nmifs_ap" / root / label / "registries.json").read_text(encoding="utf-8")).items():
            regs[name] = ctx._composite(reg, run.by_id)
        y_test = np.asarray(run.labels["test"])
        rows = []
        for name in TABLE:
            p_test, per_target = ap.deploy(run, regs[name])
            row = {"run": f"{root}/{label}", "registry": name, "internal_auprc": metric_value(y_test, p_test, "auprc"),
                   "internal_auroc": metric_value(y_test, p_test, "auroc")}
            for t in run.targets:
                yt = [e.label for e in t.episodes]
                row[f"{t.name}_auprc"] = metric_value(yt, per_target[t.name], "auprc")
                row[f"{t.name}_auroc"] = metric_value(yt, per_target[t.name], "auroc")
            row["mean_target_auprc"] = float(np.mean([row[f"{t.name}_auprc"] for t in run.targets]))
            row["mean_target_auroc"] = float(np.mean([row[f"{t.name}_auroc"] for t in run.targets]))
            rows.append(row)
        out.mkdir(parents=True, exist_ok=True)
        ctx.write_csv(out / "rows_all.csv", rows)
        print(f"[auroc] {root}/{label} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
