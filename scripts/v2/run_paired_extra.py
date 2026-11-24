#!/usr/bin/env python3
"""Paired comparisons missing from the earlier drivers (review of the v3 experiments, 2026-11-05): SRS (source schema
only) against NMIFS+AP, and SRS+T against SRS, on the 16 ICU runs and the drive runs. Registries are read, never
re-selected; deployment and bootstrap as in run_union_ap.py --phase evaluate.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_paired_extra.py
"""
from __future__ import annotations

import argparse
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

from src.icu.v2.stats import paired_cluster_bootstrap_diff  # noqa: E402

OUT = ROOT / "artifacts/v2/paired_extra"
PAIRS = (("srs_ap", "nmifs_ap_t"), ("srs_ap_t", "srs_ap"), ("srs_ap_t", "srs_c_val"), ("srs_ap_t", "nmifs_ap_t"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="", help="comma list of root:label (default: the 16 ICU runs; none: no ICU run)")
    parser.add_argument("--drive-k", default="5,10,20")
    args = parser.parse_args(argv)
    sb = ctx.selection_baselines_module()
    for label, e3_dir, extra_dir, icu in ap.jobs(args):
        out = OUT / label
        if (out / "paired.csv").exists() and "srs_c_val" in (out / "paired.csv").read_text(encoding="utf-8"):
            print(f"[pairs] {label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(e3_dir, extra_dir)
        regs = {name: run.registries[name] for name in ("srs_ap_t", "srs_ap", "srs_c_val")}
        hyb = json.loads((ROOT / "artifacts/v2/nmifs_ap" / label / "registries.json").read_text(encoding="utf-8"))
        regs["nmifs_ap_t"] = ctx._composite(hyb["nmifs_ap_t"], run.by_id)
        probs = {name: ap.deploy(run, comp) for name, comp in regs.items()}
        y_test = np.asarray(run.labels["test"])
        g_test = np.array([e.subject_id for e in run.grouped["test"]])
        paired = []
        for a, b in PAIRS:
            d = paired_cluster_bootstrap_diff(y_test, probs[a][0], probs[b][0], g_test, metric="auprc", n_boot=1000, seed=0)
            paired.append({"run": label, "a": a, "b": b, "target": "internal", **d, "metric": "internal"})
            for record in sb.paired_target_bootstrap(run.targets, probs[a][1], probs[b][1], n_boot=1000, seed=0):
                paired.append({"run": label, "a": a, "b": b, **record, "metric": "mean_target" if record["target"] == "mean_over_targets" else "target"})
        out.mkdir(parents=True, exist_ok=True)
        ctx.write_csv(out / "paired.csv", paired)
        print(f"[pairs] {label} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
