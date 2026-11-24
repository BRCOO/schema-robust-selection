#!/usr/bin/env python3
"""Wrapper decomposition (review of the v3 experiments, round 2, item M3): paired comparisons along the path
NMIFS -> Wrapper, lik. (fs_cv) -> Wrapper, AUPRC (srs_ap_unaware) -> SRS (srs_ap: + dropout scenarios, surrogates in the
objective) -> SRS+T (srs_ap_t: + target schemas), and SRS+T against its variant without surrogates in the objective, on the
16 ICU runs. For every pair: internal TEST (patient-clustered paired bootstrap), every target with the mean over all targets,
and the mean over the two real databases (eICU, HiRID). Registries are read, never re-selected; deployment with surrogate
chains as in run_union_ap.py --phase evaluate.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_decomposition_pairs.py
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

OUT = ROOT / "artifacts/v2/decomposition"
PAIRS = (("fs_cv", "nmifs"), ("srs_ap_unaware", "fs_cv"), ("srs_ap", "srs_ap_unaware"), ("srs_ap_t", "srs_ap"),
         ("srs_ap_t", "srs_ap_t_nosurr"), ("srs_ap_t", "nmifs"))
#: ablations.json names of the ablation registries used here
ABLATION = {"srs_ap_unaware": "srs_ap_unaware", "srs_ap_t_nosurr": "srs_ap_t_nosur"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="", help="comma list of root:label (default: the 16 ICU runs)")
    args = parser.parse_args(argv)
    sb = ctx.selection_baselines_module()
    for root, label in [tuple(x.split(":")) for x in args.runs.split(",") if x] or ctx.ICU_RUNS:
        run_id = f"{root}/{label}"
        out = OUT / run_id
        if (out / "paired.csv").exists():
            print(f"[decomp] {run_id}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(*ctx.run_dirs(root, label))
        regs = dict(run.registries)
        abl = json.loads((ROOT / "artifacts/v2/union_ap" / run_id / "ablations.json").read_text(encoding="utf-8"))
        for name, key in ABLATION.items():
            if key in abl:
                regs.setdefault(name, ctx._composite(abl[key], run.by_id))
        if "nmifs" not in regs:
            from src.icu.v2.selection_baselines import nmifs
            regs["nmifs"] = run.composite(nmifs(run.inp, run.k))
        names = sorted({n for pair in PAIRS for n in pair})
        missing = [n for n in names if n not in regs]
        if missing:
            print(f"[decomp] {run_id}: missing {missing}; available ablations {sorted(abl)}", flush=True)
        probs = {n: ap.deploy(run, regs[n]) for n in names if n in regs}
        y_test = np.asarray(run.labels["test"])
        g_test = np.array([e.subject_id for e in run.grouped["test"]])
        real = [t for t in run.targets if t.name in ("eicu", "hirid")]
        paired = []
        for a, b in PAIRS:
            if a not in probs or b not in probs:
                continue
            d = paired_cluster_bootstrap_diff(y_test, probs[a][0], probs[b][0], g_test, metric="auprc", n_boot=1000, seed=0)
            paired.append({"run": run_id, "a": a, "b": b, "target": "internal", "scope": "internal", **d})
            for record in sb.paired_target_bootstrap(run.targets, probs[a][1], probs[b][1], n_boot=1000, seed=0):
                paired.append({"run": run_id, "a": a, "b": b, "scope": "all_targets", **record})
            for record in sb.paired_target_bootstrap(real, {k: probs[a][1][k] for k in ("eicu", "hirid")},
                                                     {k: probs[b][1][k] for k in ("eicu", "hirid")}, n_boot=1000, seed=0):
                if record["target"] == "mean_over_targets":
                    paired.append({"run": run_id, "a": a, "b": b, "scope": "real_targets", **record})
        out.mkdir(parents=True, exist_ok=True)
        ctx.write_csv(out / "paired.csv", paired)
        print(f"[decomp] {run_id} ({time.perf_counter() - t0:.0f}s), {len(probs)} registries, {len(paired)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
