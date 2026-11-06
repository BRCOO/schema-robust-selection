#!/usr/bin/env python3
"""Surrogate programs (composite registries) against plain registries, and SRS-C.

For every generator pool (same pools, K and frozen bridge as E1):

* the E1 registries of the established selectors and of SRS (``--selectors``, from E1's
  registries.json, validation-tuned as in E1), each deployed two ways:
  ``plain`` (non-executable programs zeroed) and ``composite`` (every program ships its CART
  surrogates chosen on TRAIN firings, ``src/icu/v2/surrogates.py``; at a target each coordinate
  uses its first executable branch);
* ``srs_c``: SRS-C, SRS whose objective already uses the composite coordinates
  (``schema_robust_select(..., surrogates=True)``), scenarios = source + ``--known-targets``,
  deployed as a composite (and, for reference, plain).

The probe of every registry is fit once on the primaries' source firings (TRAIN, C on
validation); composites change only how a target computes the coordinates. Paired
patient-clustered bootstrap of the mean target AUPRC: every registry composite vs plain,
SRS-C vs every other composite and plain registry.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_composite_selection.py \\
      --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl \\
      --e1-registries artifacts/v2/selection_baselines/f_mimic_iv_gap1h/registries.json \\
      --out-dir artifacts/v2/composite/f_mimic_iv_gap1h
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.bridge import fit_frozen_bridge_v2, refire_external  # noqa: E402
from src.icu.v2.patterns import dedup_programs  # noqa: E402
from src.icu.v2.pipeline import episodes_by_split, restrict_firings  # noqa: E402
from src.icu.v2.robust_selection import scenario_set, schema_robust_select  # noqa: E402
from src.icu.v2.selection_baselines import SelectionInput  # noqa: E402
from src.icu.v2.stats import metric_value  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, composite_registry, refire_composite  # noqa: E402


def selection_baselines_module():
    spec = importlib.util.spec_from_file_location("run_selection_baselines", ROOT / "scripts" / "v2" / "run_selection_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--e1-registries", required=True, help="registries.json of run_selection_baselines.py (same K)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selectors", default="mrmr,mmrfs,l1_logistic,srs_known")
    parser.add_argument("--known-targets", default="eicu,hirid")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-surrogates", type=int, default=5, help="surrogates per program (rpart default 5)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    started_all = time.perf_counter()
    sb = selection_baselines_module()
    with open(args.pools, "rb") as handle:
        state = pickle.load(handle)
    e1 = json.loads(Path(args.e1_registries).read_text(encoding="utf-8"))
    selectors = [s for s in args.selectors.split(",") if s]
    source = sb.source_schema(state, Path(args.pools).parent / "config.json")
    grouped = episodes_by_split(state["episodes"])
    ids = {name: [e.episode_id for e in rows] for name, rows in grouped.items()}
    labels = {name: [e.label for e in rows] for name, rows in grouped.items()}
    train_ids = frozenset(ids["train"])
    train_pos = frozenset(e.episode_id for e in grouped["train"] if e.label == 1)
    subject_of = {e.episode_id: e.subject_id for e in grouped["train"]}
    known = {t.name: t.features for t in state["targets"] if t.name in [x for x in args.known_targets.split(",") if x]}
    targets = list(state["targets"])
    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    saved: dict[str, Any] = {}

    for pool in state["pools"]:
        candidates = tuple(dedup_programs(pool.candidates))
        by_id = {c.program_id: c for c in pool.candidates}
        train_firings = {c.program_id: frozenset(pool.firings.get(c.program_id, frozenset())) & train_ids for c in candidates}
        inp = SelectionInput(candidates, train_firings, train_ids, train_pos)
        registries: dict[str, CompositeRegistry] = {}
        seconds: dict[str, float] = {}
        for name in selectors:
            registry = [by_id[i] for i in e1[pool.method][name]]
            t0 = time.perf_counter()
            registries[name] = composite_registry(registry, candidates, train_firings, sorted(train_ids), args.max_surrogates)
            seconds[name] = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = schema_robust_select(inp, args.k, scenario_set(source, known), subject_of, surrogates=True, max_surrogates=args.max_surrogates)
        seconds["srs_c"] = time.perf_counter() - t0
        registries["srs_c"] = CompositeRegistry(tuple(result.registry), result.surrogates)

        probs: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        for name, comp in registries.items():
            registry = list(comp.primaries)
            if not registry:
                continue
            firings = {p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) for p in registry}
            bridge = fit_frozen_bridge_v2(registry, {s: restrict_firings(firings, frozenset(ids[s])) for s in ids}, ids, labels)
            internal = metric_value(labels["test"], [bridge["test_probabilities"][e] for e in ids["test"]], "auprc")
            for variant in ("plain", "composite"):
                per_target, row = {}, {"method": pool.method, "registry": name, "variant": variant, "n_programs": len(registry),
                                       "n_with_surrogates": sum(1 for p in registry if comp.surrogates.get(p.program_id)),
                                       "selection_seconds": seconds.get(name, 0.0), "internal_auprc": internal,
                                       "validation_score": bridge["validation_score"]}
                for target in targets:
                    episode_ids = [e.episode_id for e in target.episodes]
                    external = pool.external_firings.get(target.name, {})
                    if variant == "plain":
                        refired = refire_external(bridge["model_bundle"], registry, target.features, external, episode_ids)
                        row[f"{target.name}_surrogate_fraction"] = 0.0
                    else:
                        refired = refire_composite(bridge["model_bundle"], comp, target.features, external, episode_ids)
                        row[f"{target.name}_surrogate_fraction"] = refired["surrogate_fraction"]
                    p = np.array([refired["probabilities"][e] for e in episode_ids])
                    per_target[target.name] = p
                    row[f"{target.name}_auprc"] = metric_value([e.label for e in target.episodes], p, "auprc")
                    row[f"{target.name}_survival"] = refired["schema_survival"]
                row["mean_target_auprc"] = float(np.mean([row[f"{t.name}_auprc"] for t in targets]))
                rows.append(row)
                probs[(name, variant)] = per_target
        comparisons = [((name, "composite"), (name, "plain")) for name in registries]
        comparisons += [(("srs_c", "composite"), (name, variant)) for name in selectors for variant in ("composite", "plain")]
        for a, b in comparisons:
            if a not in probs or b not in probs:
                continue
            for record in sb.paired_target_bootstrap(targets, probs[a], probs[b], n_boot=args.n_boot, seed=args.seed):
                paired.append({"method": pool.method, "a": f"{a[0]}:{a[1]}", "b": f"{b[0]}:{b[1]}", **record})
        saved[pool.method] = {name: {"primaries": [p.program_id for p in comp.primaries],
                                     "surrogates": {pid: [q.program_id for q in qs] for pid, qs in comp.surrogates.items()}}
                              for name, comp in registries.items()}
        print(f"[composite] {pool.method} done ({time.perf_counter() - started_all:.0f}s)", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for file_name, records in (("rows.csv", rows), ("paired.csv", paired)):
        columns = sorted({k for r in records for k in r}, key=lambda c: (c not in ("method", "registry", "variant", "a", "b", "target"), c))
        with open(out / file_name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    (out / "registries.json").write_text(json.dumps(saved, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"pools": args.pools, "args": vars(args), "elapsed_seconds": time.perf_counter() - started_all}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
