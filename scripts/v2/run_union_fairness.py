#!/usr/bin/env python3
"""Reviewer item F: are the score-based baselines handicapped by generator scores on different scales?

MMR, Xin's MMS and DPP rank by the generator's own score, min-max normalized over the union pool,
where scores of different generators are not on one scale. On the same union pools and TRAIN-only,
they are re-run with two common relevance scales (``selection_baselines.with_relevance``):
``rank`` (percentile rank within the generator) and ``ig`` (TRAIN information gain), each with its
validation-tuned hyper-parameter grid as in E3. Two further selectors test the mechanism of
Section 6.5: NMIFS (redundancy normalized by entropy, i.e. by firing rate) and generator-stratified
mRMR (redundancy judged only within one language). Every registry is deployed with surrogates;
paired patient-clustered bootstrap of SRS+T (``srs_c_val``) and SRS (``srs_c_unknown_val``) against each.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_union_fairness.py [--runs ...]
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

from src.icu.v2.selection_baselines import BASELINES, mrmr_stratified, nmifs, run_baseline, with_relevance  # noqa: E402
from src.icu.v2.stats import metric_value, paired_cluster_bootstrap_diff  # noqa: E402
from src.icu.v2.surrogates import refire_composite  # noqa: E402

SCORE_BASED = ("mmr", "mms_xin", "dpp")


def deploy(run, comp):
    bridge = run.fit(list(comp.primaries))
    p_test = np.array([bridge["test_probabilities"][e] for e in run.ids["test"]])
    per_target = {}
    for target in run.targets:
        eids = [e.episode_id for e in target.episodes]
        out = refire_composite(bridge["model_bundle"], comp, target.features, run.external[target.name], eids)
        per_target[target.name] = np.array([out["probabilities"][e] for e in eids])
    return bridge["validation_score"], p_test, per_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default=str(ROOT / "artifacts/v2/union_fairness"))
    parser.add_argument("--fast", action="store_true", help="only NMIFS and stratified mRMR (skip the score-based re-runs)")
    args = parser.parse_args(argv)
    sb = ctx.selection_baselines_module()
    runs = [tuple(x.split(":")) for x in args.runs.split(",") if x] or list(ctx.ICU_RUNS)
    for root, label in runs:
        out = Path(args.out_root) / root / label
        if (out / "manifest.json").exists():
            print(f"[fair] {root}/{label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(*ctx.run_dirs(root, label))
        inp = run.inp
        registries, meta, tuning = {}, {}, []
        for relevance in (() if args.fast else ("rank", "ig")):
            rescored = with_relevance(inp, relevance)
            for name in SCORE_BASED:
                _, param, grid = BASELINES[name]
                best = None
                for value in grid:
                    picked = [run.by_id[p.program_id] for p in run_baseline(name, rescored, run.k, value)]
                    score = run.fit(picked)["validation_score"] if picked else -np.inf
                    tuning.append({"registry": f"{name}_{relevance}", "param": param, "value": value, "validation_score": score})
                    if best is None or score > best[0] + 1e-12:
                        best = (score, value, picked)
                registries[f"{name}_{relevance}"] = run.composite(best[2])
                meta[f"{name}_{relevance}"] = {"param": param, "value": best[1]}
        registries["nmifs"] = run.composite(nmifs(inp, run.k))
        registries["mrmr_stratified"] = run.composite(mrmr_stratified(inp, run.k))
        for name in (("srs_c_val", "srs_c_unknown_val", "mrmr") if args.fast else ("srs_c_val", "srs_c_unknown_val", "mrmr", "mmr", "mms_xin", "dpp")):
            registries[name] = run.registries[name]
        y_test = np.asarray(run.labels["test"])
        g_test = np.array([e.subject_id for e in run.grouped["test"]])
        rows, probs = [], {}
        for name, comp in registries.items():
            val, p_test, per_target = deploy(run, comp)
            probs[name] = (p_test, per_target)
            target_auprc = {t.name: metric_value([e.label for e in t.episodes], per_target[t.name], "auprc") for t in run.targets}
            rows.append({"run": f"{root}/{label}", "registry": name, "validation_score": val, "internal_auprc": metric_value(y_test, p_test, "auprc"),
                         "mean_target_auprc": float(np.mean(list(target_auprc.values()))), "generators": ",".join(sorted({p.method for p in comp.primaries})),
                         **{f"{t}_auprc": v for t, v in target_auprc.items()}, **meta.get(name, {})})
        paired = []
        for a in ("srs_c_val", "srs_c_unknown_val"):
            for b in registries:
                if b in ("srs_c_val", "srs_c_unknown_val"):
                    continue
                d = paired_cluster_bootstrap_diff(y_test, probs[a][0], probs[b][0], g_test, metric="auprc", n_boot=args.n_boot, seed=args.seed)
                paired.append({"run": f"{root}/{label}", "a": a, "b": b, **d, "metric": "internal"})
                for record in sb.paired_target_bootstrap(run.targets, probs[a][1], probs[b][1], n_boot=args.n_boot, seed=args.seed):
                    if record["target"] == "mean_over_targets":
                        paired.append({"run": f"{root}/{label}", "a": a, "b": b, "metric": "mean_target", **record})
        ctx.write_csv(out / "rows.csv", rows)
        ctx.write_csv(out / "paired.csv", paired)
        ctx.write_csv(out / "tuning.csv", tuning, first=("registry",))
        saved = {name: {"primaries": [p.program_id for p in comp.primaries],
                        "surrogates": {pid: [q.program_id for q in qs] for pid, qs in comp.surrogates.items()}} for name, comp in registries.items()}
        (out / "registries.json").write_text(json.dumps(saved, indent=1), encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps({"args": vars(args), "meta": meta, "elapsed_seconds": time.perf_counter() - t0}, indent=2), encoding="utf-8")
        best = {r["registry"]: (r["internal_auprc"], r["mean_target_auprc"]) for r in rows}
        print(f"[fair] {root}/{label} ({time.perf_counter() - t0:.0f}s) " + " ".join(f"{n}={v[0]:.3f}/{v[1]:.3f}" for n, v in best.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
