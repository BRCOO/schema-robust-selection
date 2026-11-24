#!/usr/bin/env python3
"""Extra metrics for the v3 experiments (review, 2026-11-05): AUROC and Brier score of the main registries, and paired
patient-clustered bootstraps per real target (eICU, HiRID) and per controlled target, which the SRS-AP evaluation only
summarized as the mean over targets. Registries: SRS+T, SRS, NMIFS, mRMR, L1+dropout, Filter+mRMR and the
likelihood-decision SRS+T, deployed with surrogate chains exactly as in run_union_ap.py.

Writes artifacts/v2/extra_metrics/<root>/<label>/{rows.csv, paired_targets.csv}.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_extra_metrics.py
"""
from __future__ import annotations

import argparse
import importlib.util
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
REGS = ("srs_ap_t", "srs_ap", "nmifs", "mrmr", "l1_dropout", "mrmr_filter_known", "srs_c_val")
PAIRS = (("srs_ap_t", "nmifs"), ("srs_ap_t", "mrmr"), ("srs_ap", "nmifs"), ("srs_ap_t", "l1_dropout"))


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="", help="comma list of root:label (default: the 16 ICU runs)")
    args = parser.parse_args(argv)
    sb = ctx.selection_baselines_module()
    for root, label in [tuple(x.split(":")) for x in args.runs.split(",") if x] or ctx.ICU_RUNS:
        out = OUT / root / label
        if (out / "paired_targets.csv").exists():
            print(f"[extra] {root}/{label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(*ctx.run_dirs(root, label))
        y_test = np.asarray(run.labels["test"])
        probs, rows = {}, []
        for name in REGS:
            p_test, per_target = ap.deploy(run, run.registries[name])
            probs[name] = (p_test, per_target)
            row = {"run": f"{root}/{label}", "registry": name, "internal_auprc": metric_value(y_test, p_test, "auprc"),
                   "internal_auroc": metric_value(y_test, p_test, "auroc"), "internal_brier": brier(y_test, p_test)}
            for t in run.targets:
                yt = [e.label for e in t.episodes]
                row[f"{t.name}_auprc"] = metric_value(yt, per_target[t.name], "auprc")
                row[f"{t.name}_auroc"] = metric_value(yt, per_target[t.name], "auroc")
                row[f"{t.name}_brier"] = brier(yt, per_target[t.name])
            row["mean_target_auprc"] = float(np.mean([row[f"{t.name}_auprc"] for t in run.targets]))
            row["mean_target_auroc"] = float(np.mean([row[f"{t.name}_auroc"] for t in run.targets]))
            row["mean_target_brier"] = float(np.mean([row[f"{t.name}_brier"] for t in run.targets]))
            rows.append(row)
        paired = []
        for a, b in PAIRS:
            for record in sb.paired_target_bootstrap(run.targets, probs[a][1], probs[b][1], n_boot=1000, seed=0):
                paired.append({"run": f"{root}/{label}", "a": a, "b": b, **record})
        out.mkdir(parents=True, exist_ok=True)
        ctx.write_csv(out / "rows.csv", rows)
        ctx.write_csv(out / "paired_targets.csv", paired)
        print(f"[extra] {root}/{label} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
