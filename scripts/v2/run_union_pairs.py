#!/usr/bin/env python3
"""Paired comparisons between registries saved by run_union_selection.py (E3 registries.json).

Every named registry is rebuilt (primaries + surrogates), its frozen probe is fit as in E3, and
A - B is bootstrapped (patient-clustered, paired): internal test AUPRC, and mean target AUPRC
with both registries deployed with surrogates (joint replicates over the targets).

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_union_pairs.py \\
      --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl --e3-dir artifacts/v2/union/f_mimic_iv_gap1h \\
      --a fs_cv --b mrmr --b mmrfs --b l1_logistic --out artifacts/v2/union_pairs/f_mimic_iv_gap1h_fs_cv.csv
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

from src.icu.v2.bridge import fit_frozen_bridge_v2  # noqa: E402
from src.icu.v2.pipeline import episodes_by_split, restrict_firings  # noqa: E402
from src.icu.v2.stats import paired_cluster_bootstrap_diff  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, refire_composite  # noqa: E402
from src.icu.v2.union import union_pool  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--e3-dir", required=True)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location("sb", ROOT / "scripts" / "v2" / "run_selection_baselines.py")
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    with open(args.pools, "rb") as handle:
        state = pickle.load(handle)
    grouped = episodes_by_split(state["episodes"])
    ids = {n: [e.episode_id for e in rows] for n, rows in grouped.items()}
    labels = {n: [e.label for e in rows] for n, rows in grouped.items()}
    cands, firings, external = union_pool(state)
    by_id = {c.program_id: c for pool in state["pools"] for c in pool.candidates}
    saved = json.loads((Path(args.e3_dir) / "registries.json").read_text(encoding="utf-8"))
    test_y = np.array(labels["test"], dtype=int)
    test_g = np.array([e.subject_id for e in grouped["test"]])

    def probabilities(name):
        entry = saved[name]
        comp = CompositeRegistry(tuple(by_id[i] for i in entry["primaries"]), {pid: tuple(by_id[q] for q in qs) for pid, qs in entry["surrogates"].items()})
        registry = list(comp.primaries)
        f = {p.program_id: firings[p.program_id] for p in registry}
        bridge = fit_frozen_bridge_v2(registry, {s: restrict_firings(f, frozenset(ids[s])) for s in ids}, ids, labels)
        internal = np.array([bridge["test_probabilities"][e] for e in ids["test"]])
        targets = {}
        for t in state["targets"]:
            eids = [e.episode_id for e in t.episodes]
            out = refire_composite(bridge["model_bundle"], comp, t.features, external[t.name], eids)
            targets[t.name] = np.array([out["probabilities"][e] for e in eids])
        return internal, targets

    a_int, a_tgt = probabilities(args.a)
    records = []
    for b in args.b:
        b_int, b_tgt = probabilities(b)
        diff = paired_cluster_bootstrap_diff(test_y, a_int, b_int, test_g, metric="auprc", n_boot=args.n_boot, seed=args.seed)
        records.append({"a": args.a, "b": b, "target": "internal_test", "point_diff": diff["point_diff"], "low": diff["low"], "high": diff["high"]})
        for r in sb.paired_target_bootstrap(state["targets"], a_tgt, b_tgt, n_boot=args.n_boot, seed=args.seed):
            if r["target"] == "mean_over_targets":
                records.append({"a": args.a, "b": b, "target": "mean_over_targets", "point_diff": r["point_diff"], "low": r["low"], "high": r["high"]})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["a", "b", "target", "point_diff", "low", "high"])
        writer.writeheader()
        writer.writerows(records)
    for r in records:
        star = "*" if r["low"] > 0 or r["high"] < 0 else ""
        print(f"{r['a']} - {r['b']:18s} {r['target']:18s} {r['point_diff']:+.3f}{star} [{r['low']:+.3f}, {r['high']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
