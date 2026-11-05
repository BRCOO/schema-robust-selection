#!/usr/bin/env python3
"""SRS ablation: which part of SRS matters (schema awareness, frozen vs projected objective,
cross-validation, budget).

Variants (all K = 10 unless named otherwise; no validation tuning anywhere):

* ``srs_known``             main: frozen masked probe, 3-fold patient-grouped CV, scenarios =
                            source + known targets (eICU, HiRID)
* ``srs_source_only``       same, scenarios = source only -> plain forward stepwise selection on
                            the cross-validated log-likelihood (no schema awareness)
* ``srs_projected``         per-scenario refit on the executable programs (source-side), CV
* ``srs_projected_insample`` same, fit and scored on all TRAIN rows: the variant with the
                            (1 - e^-gamma) guarantee of greedy forward selection
* ``srs_frozen_insample``   frozen objective without CV
* ``srs_known_k5`` / ``srs_known_k20``  budget sensitivity
* ``srs_known_k20_fill``    K = 20 without the min_gain stop: always returns K programs, like
                            the established selectors (the stop never triggers at K = 10)

Each registry: evaluate_registry (internal test + every target); target-level paired bootstrap
of ``srs_known`` against each variant (per target and mean over targets).

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_srs_ablation.py \\
      --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl --out-dir artifacts/v2/srs_ablation/f_mimic_iv_gap1h
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.patterns import dedup_programs  # noqa: E402
from src.icu.v2.pipeline import episodes_by_split, evaluate_registry, flatten_row  # noqa: E402
from src.icu.v2.robust_selection import scenario_set, schema_robust_select  # noqa: E402
from src.icu.v2.selection_baselines import SelectionInput  # noqa: E402

#: name -> (k, known targets used?, mode, cv, fill the budget (no min_gain stop)?)
VARIANTS: dict[str, tuple[int, bool, str, bool, bool]] = {
    "srs_known": (10, True, "frozen", True, False),
    "srs_source_only": (10, False, "frozen", True, False),
    "srs_projected": (10, True, "projected", True, False),
    "srs_projected_insample": (10, True, "projected", False, False),
    "srs_frozen_insample": (10, True, "frozen", False, False),
    "srs_known_k5": (5, True, "frozen", True, False),
    "srs_known_k20": (20, True, "frozen", True, False),
    "srs_known_k20_fill": (20, True, "frozen", True, True),
}
REFERENCE = "srs_known"


def selection_baselines_module():
    spec = importlib.util.spec_from_file_location("run_selection_baselines", ROOT / "scripts" / "v2" / "run_selection_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--known-targets", default="eicu,hirid")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", default=",".join(VARIANTS), help="comma list (default: all)")
    args = parser.parse_args(argv)
    variants = {name: VARIANTS[name] for name in args.variants.split(",") if name}
    started_all = time.perf_counter()
    sb = selection_baselines_module()
    with open(args.pools, "rb") as handle:
        state = pickle.load(handle)
    source = sb.source_schema(state, Path(args.pools).parent / "config.json")
    grouped = episodes_by_split(state["episodes"])
    train_ids = frozenset(e.episode_id for e in grouped["train"])
    train_pos = frozenset(e.episode_id for e in grouped["train"] if e.label == 1)
    subject_of = {e.episode_id: e.subject_id for e in grouped["train"]}
    names = [t for t in args.known_targets.split(",") if t]
    known = {t.name: t.features for t in state["targets"] if t.name in names}
    rows: list[dict[str, Any]] = []
    target_paired: list[dict[str, Any]] = []
    registry_ids: dict[str, dict[str, list[str]]] = {}
    for pool in state["pools"]:
        candidates = tuple(dedup_programs(pool.candidates))
        inp = SelectionInput(candidates, {p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) & train_ids for p in candidates}, train_ids, train_pos)
        registries = {}
        for name, (k, use_known, mode, cv, fill) in variants.items():
            started = time.perf_counter()
            result = schema_robust_select(inp, k, scenario_set(source, known if use_known else {}), subject_of, mode=mode, cv=cv,
                                          min_gain=-math.inf if fill else 1e-4)
            seconds = time.perf_counter() - started
            registries[name] = result.registry
            evaluation = evaluate_registry(result.registry, pool.firings, state["episodes"], targets=state["targets"], external_firings=pool.external_firings, n_boot=200, seed=args.seed)
            rows.append({**flatten_row(pool.method, name, evaluation), "variant": name, "k": k, "mode": mode, "cv": cv, "known_targets": use_known,
                         "fill_budget": fill, "selection_seconds": seconds, "stop": result.stop_reason})
        registry_ids[pool.method] = {n: [p.program_id for p in r] for n, r in registries.items()}
        probabilities = {n: sb.target_probabilities(r, pool, state, grouped) for n, r in registries.items() if r}
        for name in registries:
            if name == REFERENCE or name not in probabilities or REFERENCE not in probabilities:
                continue
            for record in sb.paired_target_bootstrap(state["targets"], probabilities[REFERENCE], probabilities[name], n_boot=args.n_boot, seed=args.seed):
                target_paired.append({"method": pool.method, "selector": REFERENCE, "reference": name, **record})
        print(f"[srs-ablation] {pool.method} done ({time.perf_counter() - started_all:.0f}s)", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for file_name, records in (("rows.csv", rows), ("target_paired.csv", target_paired)):
        columns = sorted({k for r in records for k in r}, key=lambda c: (c not in ("method", "variant", "selector", "reference", "target"), c))
        with open(out / file_name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    (out / "registries.json").write_text(json.dumps(registry_ids, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"pools": args.pools, "args": vars(args), "variants": variants, "elapsed_seconds": time.perf_counter() - started_all}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
