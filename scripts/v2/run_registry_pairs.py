#!/usr/bin/env python3
"""Paired comparison of registries saved by different runs (internal test and every target).

Each registry is named ``<registries.json>:<selector>``; registries.json files are written by
run_selection_baselines.py and run_srs_ablation.py ({method: {selector: [program_id, ...]}}).
For every generator pool, the frozen bridge of each registry is fit exactly as in
evaluate_registry, and A - B is bootstrapped (patient-clustered, paired) on the internal test
split and on every target, plus the mean over targets (joint replicates).

  /d/BRCoo/miniconda/python scripts/v2/run_registry_pairs.py --pools .../pools.pkl \\
      --a artifacts/v2/srs_ablation/f_mimic_iv_gap1h/registries.json:srs_source_only \\
      --b artifacts/v2/selection_baselines/f_mimic_iv_gap1h/registries.json:mrmr \\
      --out artifacts/v2/registry_pairs/f_mimic_iv_gap1h_fs_vs_mrmr.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.pipeline import episodes_by_split  # noqa: E402
from src.icu.v2.stats import paired_cluster_bootstrap_diff  # noqa: E402


def selection_baselines_module():
    spec = importlib.util.spec_from_file_location("run_selection_baselines", ROOT / "scripts" / "v2" / "run_selection_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(spec: str) -> tuple[dict, str]:
    path, selector = spec.rsplit(":", 1)
    return json.loads(Path(path).read_text(encoding="utf-8")), selector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--a", required=True, help="registries.json:selector")
    parser.add_argument("--b", action="append", required=True, help="registries.json:selector (repeatable)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    sb = selection_baselines_module()
    with open(args.pools, "rb") as handle:
        state = pickle.load(handle)
    grouped = episodes_by_split(state["episodes"])
    test = grouped["test"]
    y = np.array([e.label for e in test], dtype=int)
    groups = np.array([e.subject_id for e in test])
    a_registries, a_name = load(args.a)
    records = []
    for b_spec in args.b:
        b_registries, b_name = load(b_spec)
        for pool in state["pools"]:
            by_id = {p.program_id: p for p in pool.candidates}
            ra = [by_id[i] for i in a_registries.get(pool.method, {}).get(a_name, [])]
            rb = [by_id[i] for i in b_registries.get(pool.method, {}).get(b_name, [])]
            if not ra or not rb:
                continue
            pa = sb.target_probabilities(ra, pool, state, grouped)
            pb = sb.target_probabilities(rb, pool, state, grouped)
            # internal test: refit gives the same probabilities as evaluate_registry
            from src.icu.v2.bridge import fit_frozen_bridge_v2
            from src.icu.v2.pipeline import restrict_firings

            def internal(registry):
                ids = {name: [e.episode_id for e in rows] for name, rows in grouped.items()}
                labels = {name: [e.label for e in rows] for name, rows in grouped.items()}
                firings = {p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) for p in registry}
                bridge = fit_frozen_bridge_v2(list(registry), {n: restrict_firings(firings, frozenset(ids[n])) for n in ids}, ids, labels)
                return np.array([bridge["test_probabilities"][e.episode_id] for e in test])

            diff = paired_cluster_bootstrap_diff(y, internal(ra), internal(rb), groups, metric="auprc", n_boot=args.n_boot, seed=args.seed)
            records.append({"method": pool.method, "a": a_name, "b": b_name, "target": "internal_test", **{k: diff[k] for k in ("point_diff", "low", "high", "frac_le_zero")}})
            for record in sb.paired_target_bootstrap(state["targets"], pa, pb, n_boot=args.n_boot, seed=args.seed):
                records.append({"method": pool.method, "a": a_name, "b": b_name, **record})
        print(f"[registry-pairs] {a_name} vs {b_name} done", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "a", "b", "target", "point_diff", "low", "high", "frac_le_zero"])
        writer.writeheader()
        writer.writerows(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
