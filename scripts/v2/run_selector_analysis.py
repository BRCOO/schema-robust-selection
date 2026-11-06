#!/usr/bin/env python3
"""Selector analyses on cached candidate pools (no re-mining).

Loads the pools pickled by ``run_discovery_comparison.py --save-pools`` and runs:

* ``k``          K in {5, 10, 20} for native and v2 selection (v2 with caps fixed at their
                 K = 10 values, and with caps and pool views scaled to K)
* ``ablation``   v2 selector with one component removed at a time, and utility-only plus one
                 component at a time (K = 10)
* ``weights``    a weight grid (redundancy x family x anchor x schema) for Pareto fronts; per
                 method the grid point with the best VALIDATION score is flagged
* ``tuned``      per method, native top-K vs the v2 selector with the validation-chosen grid
                 weights, with a paired patient-clustered bootstrap of test AUPRC (needs
                 ``weights`` in the same run or a weights.csv in --out-dir)
* ``stability``  selection stability under patient-level subsampling of TRAIN
* ``registry``   the selected programs (native and v2, K = 10) with readable descriptions

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_selector_analysis.py \\
      --pools artifacts/v2/comparison/x_main/pools.pkl --out-dir artifacts/v2/selector_analysis/x_main \\
      --selector-targets eicu,hirid
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import pickle
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.patterns import contrast_score, dedup_programs  # noqa: E402
from src.icu.v2.pipeline import MethodPool, compare_methods, restrict_firings  # noqa: E402
from src.icu.v2.selector import SelectorConfig, select  # noqa: E402
from src.icu.v2.types import ProgramRecord  # noqa: E402

# v2 main configuration: the v1 locked profile's hard rule "no care-process program in the
# top 5" is switched off (vasopressor_on is no longer an input; with large pools the rule left
# the selector unable to fill K, e.g. 2 of 10 PrefixSpan programs). It is ablated below.
BASE_OVERRIDES: dict[str, Any] = {"care_process_cap": None}

_DEFAULT = SelectorConfig()
UTILITY_ONLY: dict[str, Any] = {
    "w_family": 0.0, "w_anchor": 0.0, "w_schema": 0.0, "w_care": 0.0, "w_redundancy": 0.0,
    "max_per_anchor_group": None, "max_per_family": None, "care_process_cap": None,
    "min_anchor_coverage": None, "max_repair_swaps": 0,
}

ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {},
    # leave one out of the main configuration
    "no_family_term": {"w_family": 0.0},
    "no_anchor_term": {"w_anchor": 0.0},
    "no_schema_term": {"w_schema": 0.0},
    "no_care_penalty": {"w_care": 0.0},
    "v1_care_cap_top5": {"care_process_cap": 0},
    "no_hard_caps": {"max_per_anchor_group": None, "max_per_family": None, "care_process_cap": None},
    "no_repair": {"min_anchor_coverage": None, "max_repair_swaps": 0},
    "plus_redundancy_0.5": {"w_redundancy": 0.5},
    "large_pool_200": {"pool_size": 200},
    "no_schema_view": {"pool_per_schema": None},
    "utility_only": UTILITY_ONLY,
    # add one in to utility_only (main-configuration value of that component)
    "utility+family_term": {**UTILITY_ONLY, "w_family": _DEFAULT.w_family},
    "utility+anchor_term": {**UTILITY_ONLY, "w_anchor": _DEFAULT.w_anchor},
    "utility+schema_term": {**UTILITY_ONLY, "w_schema": _DEFAULT.w_schema},
    "utility+care_penalty": {**UTILITY_ONLY, "w_care": _DEFAULT.w_care},
    "utility+redundancy_0.5": {**UTILITY_ONLY, "w_redundancy": 0.5},
    "utility+anchor_cap": {**UTILITY_ONLY, "max_per_anchor_group": _DEFAULT.max_per_anchor_group},
    "utility+repair": {**UTILITY_ONLY, "min_anchor_coverage": _DEFAULT.min_anchor_coverage, "max_repair_swaps": _DEFAULT.max_repair_swaps},
}
WEIGHT_GRID = {"w_redundancy": (0.0, 0.5, 1.0), "w_family": (0.0, 0.8), "w_anchor": (0.0, 0.3), "w_schema": (0.0, 0.35, 1.0)}


def base(k: int = 10, **overrides: Any) -> SelectorConfig:
    return replace(SelectorConfig(k=k), **{**BASE_OVERRIDES, **overrides})


def _annotate(rows: Sequence[Mapping[str, Any]], **tags: Any) -> list[dict[str, Any]]:
    return [{**tags, **row} for row in rows]


def run_compare(state: Mapping[str, Any], config: SelectorConfig, selections: Sequence[str], n_boot: int, seed: int, selector_targets: Sequence[str] | None) -> dict[str, Any]:
    return compare_methods(
        state["pools"], state["episodes"], targets=state["targets"], selector_config=config, k=config.k,
        selections=selections, n_boot=n_boot, seed=seed, selector_target_names=selector_targets,
    )


def budget_scaled(k: int) -> dict[str, Any]:
    """Pool views, anchor cap and coverage gate proportional to K (the defaults at K = 10).

    With the K = 10 values, K = 20 is infeasible by construction: 9 coarse groups x 2 per
    anchor signature, from a pool of about 20 units.
    """
    return {
        "pool_size": 2 * k,
        "pool_per_anchor_group": math.ceil(0.3 * k),
        "pool_per_family": math.ceil(0.3 * k),
        "pool_per_schema": 2 * k,
        "max_per_anchor_group": math.ceil(0.2 * k),
        "min_anchor_coverage": math.ceil(0.6 * k),
    }


def k_sweep(state, n_boot, seed, targets, ks=(5, 10, 20)) -> list[dict[str, Any]]:
    """Native top-K (independent of the caps) and the v2 selector with fixed and K-scaled caps."""
    rows: list[dict[str, Any]] = []
    for k in ks:
        out = run_compare(state, base(k), ("native", "v2_selector"), n_boot, seed, targets)
        rows += _annotate(out["rows"], analysis="k_sweep", k=k, caps="fixed")
        out = run_compare(state, base(k, **budget_scaled(k)), ("v2_selector",), n_boot, seed, targets)
        rows += _annotate(out["rows"], analysis="k_sweep", k=k, caps="scaled")
    return rows


def ablation(state, n_boot, seed, targets) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, overrides in ABLATIONS.items():
        config = base(10, **overrides)
        out = run_compare(state, config, ("v2_selector",), n_boot, seed, targets)
        rows += _annotate(out["rows"], analysis="ablation", variant=name, overrides=json.dumps(overrides))
    return rows


def mark_validation_choice(rows: list[dict[str, Any]]) -> None:
    """Flag per method the grid point with the best frozen-bridge VALIDATION score.

    Ties go to the default configuration, then to grid order. Test metrics play no part, so
    the flagged row is a legitimate "weights tuned on validation" result.
    """
    for method in dict.fromkeys(r["method"] for r in rows):
        indexed = [(i, r) for i, r in enumerate(rows) if r["method"] == method]
        usable = [(i, r) for i, r in indexed if not r.get("error") and r.get("validation_score") is not None]
        best = max(usable, key=lambda item: (round(float(item[1]["validation_score"]), 12), bool(item[1]["is_default"]), -item[0]))[0] if usable else None
        for i, r in indexed:
            r["val_selected"] = i == best


def weight_grid(state, n_boot, seed, targets) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = list(WEIGHT_GRID)
    default = asdict(base(10))
    for values in itertools.product(*(WEIGHT_GRID[k] for k in keys)):
        overrides = dict(zip(keys, values))
        out = run_compare(state, base(10, **overrides), ("v2_selector",), n_boot, seed, targets)
        is_default = all(default[key] == value for key, value in overrides.items())
        rows += _annotate(out["rows"], analysis="weights", is_default=is_default, **overrides)
    mark_validation_choice(rows)
    return rows


def tuned(state, n_boot, seed, targets, weight_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Native top-K vs the v2 selector with each method's validation-chosen weights.

    The weights come from :func:`mark_validation_choice` (validation score only); the test
    split is touched once, here, for the reported numbers and the paired difference.
    """
    rows: list[dict[str, Any]] = []
    for pool in state["pools"]:
        chosen = next((r for r in weight_rows if r["method"] == pool.method and str(r.get("val_selected")) == "True"), None)
        if chosen is None:
            continue
        overrides = {key: float(chosen[key]) for key in WEIGHT_GRID}
        out = compare_methods(
            [pool], state["episodes"], targets=state["targets"], selector_config=base(10, **overrides), k=10,
            selections=("native", "v2_selector"), reference=(pool.method, "native"), n_boot=n_boot, seed=seed,
            selector_target_names=targets,
        )
        paired = out["paired_vs_reference"][0] if out["paired_vs_reference"] else {}
        for row in out["rows"]:
            extra = {}
            if row["selection"] == "v2_selector" and paired:
                extra = {"diff_vs_native": paired["point_diff"], "diff_low": paired["low"], "diff_high": paired["high"], "diff_frac_le_zero": paired["frac_le_zero"]}
            rows.append({"analysis": "tuned", **overrides, **row, **extra})
    return rows


def _hash(seed: int, value: object) -> str:
    return hashlib.sha256(f"{seed}:stability:{value}".encode()).hexdigest()


def _rescored(pool: MethodPool, episode_ids: frozenset[str], positives: frozenset[str]) -> list[ProgramRecord]:
    pos_total = len(positives)
    neg_total = len(episode_ids) - pos_total
    out = []
    for program in pool.candidates:
        hits = frozenset(pool.firings.get(program.program_id, frozenset())) & episode_ids
        pos = len(hits & positives)
        out.append(replace(program, score=contrast_score(pos, pos_total, len(hits) - pos, neg_total)))
    return out


def _topk(candidates: Sequence[ProgramRecord], k: int) -> list[str]:
    ranked = sorted(dedup_programs(candidates), key=lambda p: (-p.score, p.canonical_key(), p.program_id))
    return [p.program_id for p in ranked[:k]]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def stability(state, targets, reps: int, fraction: float, seed: int) -> list[dict[str, Any]]:
    """Selections under patient-level TRAIN subsampling, scored by one contrast rule for all methods.

    Candidate pools are fixed (no re-mining); each repetition re-scores candidates on the
    subsample with the smoothed log2 contrast and re-selects (top-K by score, and the v2
    selector with subsample firings). Reports Jaccard to the full-TRAIN selection and the
    mean pairwise Jaccard between repetitions.
    """
    episodes = state["episodes"]
    train = [e for e in episodes if e.split == "train"]
    patients = sorted({e.subject_id for e in train})
    schemas = {t.name: t.features for t in state["targets"] if not targets or t.name in targets}
    config = base(10)
    full_ids = frozenset(e.episode_id for e in train)
    full_pos = frozenset(e.episode_id for e in train if e.label == 1)
    rows: list[dict[str, Any]] = []
    for pool in state["pools"]:
        selections: dict[str, list[list[str]]] = {"topk_contrast": [], "v2_selector": []}
        reference: dict[str, list[str]] = {}
        for rep in range(-1, reps):
            if rep < 0:
                ids, pos = full_ids, full_pos
            else:
                chosen = set(sorted(patients, key=lambda s: _hash(seed + rep, s))[: int(round(fraction * len(patients)))])
                ids = frozenset(e.episode_id for e in train if e.subject_id in chosen)
                pos = frozenset(e.episode_id for e in train if e.subject_id in chosen and e.label == 1)
            rescored = _rescored(pool, ids, pos)
            topk = _topk(rescored, config.k)
            registry, _ = select(rescored, config, restrict_firings({p.program_id: pool.firings.get(p.program_id, frozenset()) for p in rescored}, ids), schemas)
            v2 = [p.program_id for p in registry]
            if rep < 0:
                reference = {"topk_contrast": topk, "v2_selector": v2}
            else:
                selections["topk_contrast"].append(topk)
                selections["v2_selector"].append(v2)
        for name, sels in selections.items():
            to_full = [_jaccard(s, reference[name]) for s in sels]
            pairwise = [_jaccard(a, b) for a, b in itertools.combinations(sels, 2)]
            rows.append(
                {
                    "analysis": "stability",
                    "method": pool.method,
                    "selection": name,
                    "reps": reps,
                    "fraction": fraction,
                    "jaccard_to_full_mean": float(np.mean(to_full)),
                    "jaccard_to_full_min": float(np.min(to_full)),
                    "pairwise_jaccard_mean": float(np.mean(pairwise)) if pairwise else float("nan"),
                    "distinct_programs_selected": len({pid for s in sels for pid in s}),
                }
            )
    return rows


def describe(program: ProgramRecord) -> str:
    params = program.params
    if program.kind in ("motif", "shapelet"):
        template = ", ".join(f"{float(v):+.2f}" for v in params.get("template", ()))
        direction = params.get("direction")
        relation = "" if direction is None else (" near" if direction == "le" else " far from")
        return f"{program.kind}[{params.get('feature')}, L={params.get('length')}]{relation} z-shape ({template}) within {float(params.get('threshold', 0.0)):.2f}"
    def fmt(itemsets):
        return " -> ".join("{" + ", ".join(sorted(i)) + "}" for i in itemsets)
    if program.consequent:
        return f"{fmt(program.elements)} => {fmt(program.consequent)}"
    return fmt(program.elements)


def registry_export(state, targets) -> list[dict[str, Any]]:
    from src.icu.v2.pipeline import native_registry, selector_registry

    episodes = state["episodes"]
    train_ids = frozenset(e.episode_id for e in episodes if e.split == "train")
    train_pos = frozenset(e.episode_id for e in episodes if e.split == "train" and e.label == 1)
    test_ids = frozenset(e.episode_id for e in episodes if e.split == "test")
    test_pos = frozenset(e.episode_id for e in episodes if e.split == "test" and e.label == 1)
    schemas = {t.name: t.features for t in state["targets"] if not targets or t.name in targets}
    rows: list[dict[str, Any]] = []
    for pool in state["pools"]:
        regs = {"native": native_registry(pool, 10), "v2_selector": selector_registry(pool, base(10), train_ids, schemas)[0]}
        for selection, registry in regs.items():
            for rank, program in enumerate(registry, start=1):
                hits = frozenset(pool.firings.get(program.program_id, frozenset()))
                rows.append(
                    {
                        "method": pool.method,
                        "selection": selection,
                        "rank": rank,
                        "program_id": program.program_id,
                        "kind": program.kind,
                        "description": describe(program),
                        "features": sorted(program.features),
                        "anchor_signature": sorted(program.anchor_signature),
                        "train_pos": len(hits & train_pos),
                        "train_neg": len((hits & train_ids) - train_pos),
                        "test_pos": len(hits & test_pos),
                        "test_neg": len((hits & test_ids) - test_pos),
                        "score": program.score,
                    }
                )
    return rows


def write_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns = sorted({key for row in rows for key in row}, key=lambda c: (c not in ("analysis", "method", "selection", "variant", "k"), c))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--analyses", default="k,ablation,weights,tuned,stability,registry")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--tuned-n-boot", type=int, default=2000, help="bootstrap replicates for the tuned analysis (paper CIs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selector-targets", default="", help="comma list of target schemas the selector may use")
    parser.add_argument("--stability-reps", type=int, default=20)
    parser.add_argument("--stability-fraction", type=float, default=0.8)
    args = parser.parse_args(argv)
    with open(args.pools, "rb") as handle:
        state = pickle.load(handle)
    targets = [t for t in args.selector_targets.split(",") if t] or None
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    weight_rows: list[dict[str, Any]] | None = None
    for analysis in [a for a in args.analyses.split(",") if a]:
        started = time.perf_counter()
        if analysis == "k":
            rows = k_sweep(state, args.n_boot, args.seed, targets)
        elif analysis == "ablation":
            rows = ablation(state, args.n_boot, args.seed, targets)
        elif analysis == "weights":
            rows = weight_rows = weight_grid(state, args.n_boot, args.seed, targets)
        elif analysis == "tuned":
            if weight_rows is None:
                if not (out / "weights.csv").exists():
                    raise SystemExit("tuned needs the weights analysis (same run) or weights.csv in --out-dir")
                with open(out / "weights.csv", encoding="utf-8") as handle:
                    weight_rows = list(csv.DictReader(handle))
            rows = tuned(state, args.tuned_n_boot, args.seed, targets, weight_rows)
        elif analysis == "stability":
            rows = stability(state, targets, args.stability_reps, args.stability_fraction, args.seed)
        elif analysis == "registry":
            rows = registry_export(state, targets)
        else:
            raise SystemExit(f"unknown analysis {analysis!r}")
        write_rows(rows, out / f"{analysis}.csv")
        timings[analysis] = time.perf_counter() - started
        print(f"[selector-analysis] {analysis}: {len(rows)} rows in {timings[analysis]:.0f}s", flush=True)
    manifest_path = out / "selector_analysis_manifest.json"
    runs = json.loads(manifest_path.read_text(encoding="utf-8")).get("runs", []) if manifest_path.exists() else []
    runs.append({"args": vars(args), "timings_seconds": timings})
    manifest_path.write_text(
        json.dumps({"pools": args.pools, "runs": runs, "ablations": ABLATIONS, "weight_grid": WEIGHT_GRID, "base_config": asdict(base(10)), "base_overrides": BASE_OVERRIDES}, indent=2, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
