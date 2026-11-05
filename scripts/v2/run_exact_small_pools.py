#!/usr/bin/env python3
"""Reviewer item E: how close is greedy to the exact optimum on instances small enough to solve?

Theorem 1's constant can be small, so we measure the approximation ratio directly. From the union
pool of a finished E3 run, draw sub-pools of ``--m`` candidates (``top``: uniformly from the
``--top`` candidates with the highest TRAIN information gain, where the choice matters; ``any``:
uniformly from the whole pool). On each sub-pool and budget K, enumerate every K-subset to find
the exact optimum of

* ``deployed``: the objective SRS optimizes (cross-validated frozen probe, scenario-weighted,
  composites with surrogate chains over the whole pool), for the scenario distributions of q = 0
  and q = 0.25 (with the known target schemas, as SRS+T);
* ``analyzable``: Theorem 1's variant (projected per scenario, in-sample, no surrogates),

and compare with exact greedy restricted to the same sub-pool (firing duplicates skipped, as in
SRS). Reports J(greedy) / J(optimum) and whether greedy found the optimum.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_exact_small_pools.py
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _context():
    name = "e3_context"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "v2" / "e3_context.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


ctx = _context()

from src.icu.v2.robust_selection import _Objective, grouped_folds, information_gain, scenario_set  # noqa: E402
from src.icu.v2.surrogates import branch_index, surrogate_chains  # noqa: E402


def objectives(run, known: dict, max_surrogates: int):
    """The deployed objectives (q = 0, 0.25) and the analyzable one, over the whole union pool."""
    cands = list(run.candidates)
    ids = sorted(run.train_ids)
    row = {e: i for i, e in enumerate(ids)}
    X = np.zeros((len(ids), len(cands)))
    for j, p in enumerate(cands):
        for e in run.train_firings[p.program_id]:
            X[row[e], j] = 1.0
    y = np.array([1.0 if e in run.train_pos else 0.0 for e in ids])
    folds = grouped_folds(ids, run.subject_of, 3, 0)
    chains = surrogate_chains(cands, X, max_surrogates)
    feature_sets = [frozenset(p.features) for p in cands]
    out = {}
    for name, q, mode in (("deployed_q0", 0.0, "frozen"), ("deployed_q25", 0.25, "frozen"), ("analyzable", 0.0, "projected")):
        scen = scenario_set(run.source, known, dropout_q=q)
        w = np.array([s.weight for s in scen], dtype=float)
        w = w / w.sum()
        executable = np.array([[1.0 if f <= s.features else 0.0 for f in feature_sets] for s in scen])
        if mode == "frozen":
            branch = branch_index(chains, cands, [s.features for s in scen])
            out[name] = _Objective(X, y, folds, executable, w, 1.0, "frozen", branch)
        else:
            out[name] = _Objective(X, y, [(np.arange(len(ids)), np.arange(len(ids)))], executable, w, 1.0, "projected", None)
    signature = [X[:, j].tobytes() for j in range(len(cands))]
    ig = np.array([information_gain(run.train_firings[p.program_id], run.train_ids, run.train_pos) for p in cands])
    return out, signature, ig


def greedy(objective, pool: list[int], k: int, signature) -> tuple[list[int], float]:
    selected, value = [], 0.0
    for _ in range(k):
        taken = {signature[j] for j in selected}
        best = None
        for j in pool:
            if j in selected or signature[j] in taken:
                continue
            v = objective(selected + [j])
            if best is None or v > best[0] + 1e-12:
                best = (v, j)
        if best is None:
            break
        selected.append(best[1])
        value = best[0]
    return selected, value


def optimum(objective, pool: list[int], k: int, signature) -> tuple[tuple[int, ...], float]:
    best = (None, -np.inf)
    for subset in itertools.combinations(pool, k):
        if len({signature[j] for j in subset}) < k:
            continue
        v = objective(list(subset))
        if v > best[1]:
            best = (subset, v)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="union:f_mimic_iv_gap1h,union:f_mimic_iv_aki_gap6h")
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--ks", default="3,5")
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "artifacts/v2/exact_small_pools/rows.csv"))
    args = parser.parse_args(argv)
    records = []
    for spec in [x for x in args.runs.split(",") if x]:
        root, label = spec.split(":")
        e3_dir, extra_dir = ctx.run_dirs(root, label)
        run = ctx.load_run(e3_dir, extra_dir)
        known_names = json.loads((e3_dir / "manifest.json").read_text(encoding="utf-8"))["args"]["known_targets"].split(",")
        known = {t.name: t.features for t in run.targets if t.name in known_names}
        t0 = time.perf_counter()
        objs, signature, ig = objectives(run, known, run.max_surrogates)
        print(f"[exact] {label}: objectives built ({time.perf_counter() - t0:.0f}s)", flush=True)
        top = list(np.argsort(-ig, kind="stable")[: args.top])
        for sampler in ("top", "any"):
            frame = top if sampler == "top" else list(range(len(run.candidates)))
            for draw in range(args.draws):
                rng = np.random.default_rng([args.seed, draw, 0 if sampler == "top" else 1])
                pool = sorted(int(j) for j in rng.choice(frame, size=args.m, replace=False))
                for k in [int(x) for x in args.ks.split(",") if x]:
                    for name, objective in objs.items():
                        t1 = time.perf_counter()
                        g_sel, g_val = greedy(objective, pool, k, signature)
                        o_sel, o_val = optimum(objective, pool, k, signature)
                        records.append({"run": f"{root}/{label}", "sampler": sampler, "draw": draw, "k": k, "objective": name,
                                        "greedy": g_val, "optimum": o_val, "ratio": g_val / o_val if o_val > 0 else float("nan"),
                                        "greedy_is_optimal": sorted(g_sel) == sorted(o_sel), "seconds": time.perf_counter() - t1})
                print(f"[exact] {label} {sampler} draw {draw}: " + ", ".join(
                    f"{r['objective']}/K{r['k']} {r['ratio']:.4f}" for r in records[-len(objs) * len(args.ks.split(',')):]), flush=True)
    ctx.write_csv(Path(args.out), records, first=("run", "sampler", "draw", "k", "objective"))
    print(f"[exact] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
