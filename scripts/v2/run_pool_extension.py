#!/usr/bin/env python3
"""Pool extension (predicate integration, 2026-11-08): what adding the window-statistic predicates as a sixth generator did
to every selector. For each of the 16 ICU runs, the shipped (CV seed 0) registries of the five-generator pools (the archived
joint-bootstrap dumps, artifacts/v2/_five_generators/joint_bootstrap) and of the six-generator pools (artifacts/v2/
joint_bootstrap, run_joint_bootstrap.py --phase dump) are scored on the same episodes; a joint patient bootstrap (MIMIC-IV,
eICU and HiRID patients drawn once per replicate for every run, as in run_joint_bootstrap.py) gives 95% percentile intervals
for the difference of the 16-run means (six minus five) per registry and endpoint, and per-run verdicts.

Writes artifacts/v2/pool_extension/{boot.npz, summary.csv, per_run.csv} and paper/v3_assets/source_data/q6_pool_extension.csv.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_pool_extension.py
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
spec = importlib.util.spec_from_file_location("run_joint_bootstrap", ROOT / "scripts/v2/run_joint_bootstrap.py")
jb = importlib.util.module_from_spec(spec)
sys.modules["run_joint_bootstrap"] = jb
spec.loader.exec_module(jb)
ctx = jb.ctx

FIVE = ROOT / "artifacts/v2/_five_generators/joint_bootstrap"
SIX = ROOT / "artifacts/v2/joint_bootstrap"
OUT = ROOT / "artifacts/v2/pool_extension"
SRC = ROOT / "paper/v3_assets/source_data"
ENDPOINTS = ("internal", "real", "controlled", "targets")


def main(argv=None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    datasets = ["internal", *jb.REAL, *jb.CONTROLLED]
    runs = []
    for root, label in ctx.ICU_RUNS:
        five = np.load(FIVE / root / f"{label}.npz", allow_pickle=False)
        six = np.load(SIX / root / f"{label}.npz", allow_pickle=False)
        for d in datasets:  # the same episodes, in the same order, in both dumps
            assert (five[f"g_{d}"] == six[f"g_{d}"]).all() and (five[f"y_{d}"] == six[f"y_{d}"]).all(), (root, label, d)
        runs.append((f"{root}/{label}", five, six))
    methods = [m for m in (str(x) for x in runs[0][2]["methods"]) if m in {str(x) for x in runs[0][1]["methods"]}]
    index: dict[str, dict[str, int]] = {"mimic": {}, "eicu": {}, "hirid": {}}
    for _, five, _ in runs:
        for d in datasets:
            ix = index[jb.database_of(d)]
            for g in five[f"g_{d}"]:
                ix.setdefault(str(g), len(ix))
    sizes = {db: len(ix) for db, ix in index.items()}
    prep = []
    for run, five, six in runs:
        mf = {str(m): j for j, m in enumerate(five["methods"])}
        ms = {str(m): j for j, m in enumerate(six["methods"])}
        for d in datasets:
            y = six[f"y_{d}"].astype(int)
            g = np.array([index[jb.database_of(d)][str(x)] for x in six[f"g_{d}"]])
            cols = [five[f"p_{d}"][:, mf[m]] for m in methods] + [six[f"p_{d}"][:, ms[m]] for m in methods]
            prep.append((run, d, g, [jb.ap_prepare(y, p) for p in cols]))
    R, M, D = len(runs), 2 * len(methods), len(datasets)
    point = np.zeros((R, M, D))
    for i, (_, _, g, per_m) in enumerate(prep):
        r, k = divmod(i, D)
        for j, (o, e, ys) in enumerate(per_m):
            point[r, j, k] = jb.weighted_ap(np.ones((1, len(g))), o, e, ys)[0]
    rng = np.random.default_rng(args.seed)
    boot = np.zeros((args.n_boot, R, M, D))
    t0 = time.perf_counter()
    for start in range(0, args.n_boot, args.chunk):
        nb = min(args.chunk, args.n_boot - start)
        counts = {db: np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(nb)]).astype(np.float64) for db, n in sizes.items()}
        for i, (_, d, g, per_m) in enumerate(prep):
            r, k = divmod(i, D)
            W = counts[jb.database_of(d)][:, g]
            for j, (o, e, ys) in enumerate(per_m):
                boot[start:start + nb, r, j, k] = jb.weighted_ap(W, o, e, ys)
        print(f"[extension] {start + nb}/{args.n_boot} ({time.perf_counter() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "boot.npz", boot=boot, point=point, runs=np.array([r for r, _, _ in runs]), methods=np.array(methods),
                        datasets=np.array(datasets))
    rows, per_run = [], []
    n = len(methods)
    for j, m in enumerate(methods):
        for e in ENDPOINTS:
            v5, v6 = jb.endpoint_values(point[:, j], datasets, e), jb.endpoint_values(point[:, n + j], datasets, e)
            b5, b6 = jb.endpoint_values(boot[:, :, j], datasets, e), jb.endpoint_values(boot[:, :, n + j], datasets, e)
            d, db = v6 - v5, b6 - b5
            lo, hi = np.percentile(db.mean(axis=1), [2.5, 97.5])
            lo_r, hi_r = np.percentile(db, [2.5, 97.5], axis=0)
            rows.append({"registry": m, "endpoint": e, "five": float(v5.mean()), "six": float(v6.mean()), "diff": float(d.mean()),
                         "low": float(lo), "high": float(hi), "better": int((lo_r > 0).sum()), "ns": int(((lo_r <= 0) & (hi_r >= 0)).sum()),
                         "worse": int((hi_r < 0).sum())})
            for r, (run, _, _) in enumerate(runs):
                per_run.append({"registry": m, "endpoint": e, "run": run, "five": float(v5[r]), "six": float(v6[r]), "diff": float(d[r]),
                                "low": float(lo_r[r]), "high": float(hi_r[r])})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(per_run).to_csv(OUT / "per_run.csv", index=False)
    SRC.mkdir(parents=True, exist_ok=True)
    S.to_csv(SRC / "q6_pool_extension.csv", index=False)
    show = S[S.endpoint.isin(["internal", "real", "targets"])].copy()
    for c in ("five", "six", "diff", "low", "high"):
        show[c] = (100 * show[c]).round(1)
    pd.set_option("display.width", 200)
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
