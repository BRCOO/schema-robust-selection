#!/usr/bin/env python3
"""Selection-method comparison: the v2 selector vs established pattern-set selection methods.

For each candidate pool in a cached ``pools.pkl`` (``run_discovery_comparison.py
--save-pools``) and K = 10, these registries are built from TRAIN data only:

* ``native``      the generator's own top-K (or FMG's v1 selection)
* ``v2_apriori``  the v2 selector with the a-priori weights (``run_selector_analysis.base``)
* ``v2_tuned``    the v2 selector with the weights chosen on validation (``tuned.csv``)
* the baselines of :mod:`src.icu.v2.selection_baselines` (MMR, Xin et al. MMS, MMRFS, mRMR,
  DPP greedy MAP, L1 logistic), each with its hyper-parameter chosen on VALIDATION with the
  same criterion as the selector's weights: the frozen bridge's validation AUPRC (C chosen
  on validation, model fit on TRAIN). Ties go to the earlier grid value.

Every registry is evaluated with :func:`src.icu.v2.pipeline.evaluate_registry` (internal test,
real and controlled targets, patient-clustered bootstrap), and every non-reference registry
gets a paired patient-clustered bootstrap of test AUPRC against ``v2_tuned`` and ``native``.
All baselines see the same de-duplicated candidate list as ``native`` top-K.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_selection_baselines.py \\
      --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl \\
      --selector-analysis-dir artifacts/v2/selector_analysis/f_mimic_iv_gap1h \\
      --out-dir artifacts/v2/selection_baselines/f_mimic_iv_gap1h --selector-targets eicu,hirid
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.bridge import fit_frozen_bridge_v2, refire_external  # noqa: E402
from src.icu.v2.patterns import dedup_programs  # noqa: E402
from src.icu.v2.pipeline import MethodPool, episodes_by_split, evaluate_registry, flatten_row, native_registry, restrict_firings, selector_registry  # noqa: E402
from src.icu.v2.selection_baselines import BASELINES, SelectionInput, information_gain, run_baseline  # noqa: E402
from src.icu.v2.robust_selection import scenario_set, schema_robust_select  # noqa: E402
from src.icu.v2.stats import paired_cluster_bootstrap_diff  # noqa: E402
from src.icu.v2.types import ProgramRecord  # noqa: E402

REFERENCES = ("v2_tuned", "v2_auto", "native", "mrmr", "srs_known", "srs_robust25")
TARGET_COMPARISONS = (
    ("srs_known", "mrmr"), ("srs_known", "mmrfs"), ("srs_known", "l1_logistic"), ("srs_known", "v2_auto"),
    ("srs_known", "mrmr_filter_known"), ("srs_robust25", "mrmr"), ("srs_robust25", "v2_auto"),
)
SRS_VARIANTS = (("srs_known", 0.0), ("srs_robust25", 0.25), ("srs_robust50", 0.5))


def selector_analysis_module():
    spec = importlib.util.spec_from_file_location("run_selector_analysis", ROOT / "scripts" / "v2" / "run_selector_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v2_config(sa, k: int, **overrides):
    """v2 selector config with pool views, anchor cap and coverage gate scaled to K
    (``run_selector_analysis.budget_scaled``; identical to the defaults at K = 10), so the
    selector can fill larger budgets; explicit overrides win."""
    return sa.base(k, **{**sa.budget_scaled(k), **overrides})


def selection_input(pool: MethodPool, train_ids: frozenset[str], train_positives: frozenset[str]) -> SelectionInput:
    candidates = tuple(dedup_programs(pool.candidates))
    firings = {p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) & train_ids for p in candidates}
    return SelectionInput(candidates=candidates, train_firings=firings, train_ids=train_ids, train_positives=train_positives)


def validation_score(registry: Sequence[ProgramRecord], pool: MethodPool, grouped: Mapping[str, list]) -> float:
    """Frozen-bridge validation AUPRC of a registry (the criterion used for every choice)."""
    if not registry:
        return -math.inf
    ids = {name: [e.episode_id for e in rows] for name, rows in grouped.items()}
    labels = {name: [e.label for e in rows] for name, rows in grouped.items()}
    firings = {p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) for p in registry}
    try:
        bridge = fit_frozen_bridge_v2(list(registry), {name: restrict_firings(firings, frozenset(ids[name])) for name in ids}, ids, labels)
    except ValueError:
        return -math.inf
    return float(bridge["validation_score"])


def ig_rescored_pool(pool: MethodPool, train_ids: frozenset[str], train_positives: frozenset[str]) -> MethodPool:
    """The same candidates, scored by the information gain of their TRAIN firing indicator."""
    rescored = [
        replace(p, score=information_gain(frozenset(pool.firings.get(p.program_id, frozenset())) & train_ids, train_ids, train_positives))
        for p in pool.candidates
    ]
    rescored.sort(key=lambda p: -p.score)
    return replace(pool, candidates=rescored, native_registry=None)


def tuned_overrides(tuned_rows: Sequence[Mapping[str, str]], method: str, keys: Sequence[str]) -> dict[str, float] | None:
    row = next((r for r in tuned_rows if r["method"] == method and r["selection"] == "v2_selector"), None)
    return None if row is None else {key: float(row[key]) for key in keys}


def source_schema(state: Mapping[str, Any], config_path: Path) -> frozenset[str]:
    """The source (internal) schema: ``internal_features`` of the comparison config next to the
    pools, else the union of all target schemas and of the candidates' features."""
    if config_path.exists():
        features = json.loads(config_path.read_text(encoding="utf-8")).get("internal_features")
        if features:
            return frozenset(features)
    union: set[str] = set()
    for target in state["targets"]:
        union |= set(target.features)
    for pool in state["pools"]:
        for program in pool.candidates:
            union |= set(program.features)
    return frozenset(union)


def target_probabilities(registry: Sequence[ProgramRecord], pool: MethodPool, state: Mapping[str, Any], grouped: Mapping[str, list]) -> dict[str, np.ndarray]:
    """Frozen-bridge probabilities on every target (rows in target.episodes order); the bridge
    is fit exactly as in evaluate_registry (TRAIN fit, C on validation), then refired."""
    ids = {name: [e.episode_id for e in rows] for name, rows in grouped.items()}
    labels = {name: [e.label for e in rows] for name, rows in grouped.items()}
    firings = {p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) for p in registry}
    bridge = fit_frozen_bridge_v2(list(registry), {name: restrict_firings(firings, frozenset(ids[name])) for name in ids}, ids, labels)
    out = {}
    for target in state["targets"]:
        episode_ids = [e.episode_id for e in target.episodes]
        refired = refire_external(bridge["model_bundle"], list(registry), target.features, pool.external_firings.get(target.name, {}), episode_ids)
        out[target.name] = np.array([refired["probabilities"][e] for e in episode_ids])
    return out


def paired_target_bootstrap(targets, probs_a: Mapping[str, np.ndarray], probs_b: Mapping[str, np.ndarray], n_boot: int, seed: int) -> list[dict[str, Any]]:
    """Per-target and mean-over-targets AUPRC difference (a - b) with a patient-clustered
    bootstrap; the mean uses one joint replicate per bootstrap draw (each target resampled
    independently, the same draw for a and b)."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    prepared = []
    for target in targets:
        y = np.array([e.label for e in target.episodes], dtype=int)
        g = np.array([str(e.subject_id) for e in target.episodes])
        order = np.argsort(g, kind="stable")
        _, starts, counts = np.unique(g[order], return_index=True, return_counts=True)
        prepared.append((target.name, y, probs_a[target.name], probs_b[target.name], order, starts, counts))
    points = {name: average_precision_score(y, pa) - average_precision_score(y, pb) for name, y, pa, pb, *_ in prepared}
    draws: dict[str, list[float]] = {name: [] for name, *_ in prepared}
    means = []
    for _ in range(n_boot):
        diffs = []
        for name, y, pa, pb, order, starts, counts in prepared:
            pick = rng.integers(0, len(starts), len(starts))
            c = counts[pick]
            idx = order[np.repeat(starts[pick] - np.cumsum(np.r_[0, c[:-1]]), c) + np.arange(c.sum())]
            yy = y[idx]
            if yy.min() == yy.max():
                continue
            d = average_precision_score(yy, pa[idx]) - average_precision_score(yy, pb[idx])
            draws[name].append(d)
            diffs.append(d)
        if len(diffs) == len(prepared):
            means.append(float(np.mean(diffs)))
    records = []
    for name, *_ in prepared:
        values = np.array(draws[name])
        records.append({"target": name, "point_diff": points[name], "low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5)), "frac_le_zero": float(np.mean(values <= 0))})
    values = np.array(means)
    records.append({"target": "mean_over_targets", "point_diff": float(np.mean(list(points.values()))), "low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5)), "frac_le_zero": float(np.mean(values <= 0))})
    return records


def build_registries(state, pool: MethodPool, k: int, targets: Sequence[str] | None, tuned_rows, tuning_log: list, source_features: frozenset[str]) -> tuple[dict[str, list[ProgramRecord]], dict[str, dict[str, Any]]]:
    sa = selector_analysis_module()
    grouped = episodes_by_split(state["episodes"])
    train_ids = frozenset(e.episode_id for e in grouped["train"])
    train_pos = frozenset(e.episode_id for e in grouped["train"] if e.label == 1)
    schemas = {t.name: t.features for t in state["targets"] if not targets or t.name in targets}
    registries: dict[str, list[ProgramRecord]] = {}
    info: dict[str, dict[str, Any]] = {}

    started = time.perf_counter()
    registries["native"] = native_registry(pool, k)
    info["native"] = {"hparam": "", "selection_seconds": time.perf_counter() - started}
    for name, overrides in (("v2_apriori", {}), ("v2_tuned", tuned_overrides(tuned_rows, pool.method, list(sa.WEIGHT_GRID)))):
        if overrides is None:
            continue
        started = time.perf_counter()
        registries[name] = selector_registry(pool, v2_config(sa, k, **overrides), train_ids, schemas)[0]
        info[name] = {"hparam": json.dumps(overrides), "selection_seconds": time.perf_counter() - started}

    # v2 selector with a label-aware utility: each candidate's information gain on TRAIN (the
    # relevance of MMRFS and, for binary indicators, mRMR) instead of the generator's own score;
    # weights tuned on validation over the same grid. ``v2_auto`` picks, on validation, between
    # the native-score and the information-gain utility (each with its own tuned weights).
    ig_pool = ig_rescored_pool(pool, train_ids, train_pos)
    if "v2_tuned" in registries:
        registries["v2_ig_apriori"] = selector_registry(ig_pool, v2_config(sa, k), train_ids, schemas)[0]
        info["v2_ig_apriori"] = {"hparam": "{}", "selection_seconds": float("nan")}
        best_ig: tuple[float, dict[str, float], list[ProgramRecord]] | None = None
        for values in itertools.product(*sa.WEIGHT_GRID.values()):
            overrides = dict(zip(sa.WEIGHT_GRID, values))
            registry = selector_registry(ig_pool, v2_config(sa, k, **overrides), train_ids, schemas)[0]
            score = validation_score(registry, pool, grouped)
            tuning_log.append({"method": pool.method, "selector": "v2_ig_tuned", "param": "weights", "value": json.dumps(overrides), "n_programs": len(registry), "validation_score": score, "selection_seconds": float("nan")})
            if best_ig is None or score > best_ig[0] + 1e-12:
                best_ig = (score, overrides, registry)
        registries["v2_ig_tuned"] = best_ig[2]
        info["v2_ig_tuned"] = {"hparam": json.dumps(best_ig[1]), "selection_seconds": float("nan")}
        native_score = validation_score(registries["v2_tuned"], pool, grouped)
        use_ig = best_ig[0] > native_score + 1e-12
        registries["v2_auto"] = registries["v2_ig_tuned"] if use_ig else registries["v2_tuned"]
        info["v2_auto"] = {"hparam": json.dumps({"utility": "information_gain" if use_ig else "generator_score", **(best_ig[1] if use_ig else json.loads(info["v2_tuned"]["hparam"]))}), "selection_seconds": float("nan")}
        # the same, without the structural constraints (no per-anchor cap, no coverage repair):
        # isolates what the hard caps and the repair cost in utility
        best_nc: tuple[float, dict[str, float], list[ProgramRecord]] | None = None
        for values in itertools.product(*sa.WEIGHT_GRID.values()):
            overrides = dict(zip(sa.WEIGHT_GRID, values))
            config = v2_config(sa, k, **overrides, max_per_anchor_group=None, max_per_family=None, min_anchor_coverage=None, max_repair_swaps=0)
            registry = selector_registry(ig_pool, config, train_ids, schemas)[0]
            score = validation_score(registry, pool, grouped)
            tuning_log.append({"method": pool.method, "selector": "v2_ig_nocaps", "param": "weights", "value": json.dumps(overrides), "n_programs": len(registry), "validation_score": score, "selection_seconds": float("nan")})
            if best_nc is None or score > best_nc[0] + 1e-12:
                best_nc = (score, overrides, registry)
        registries["v2_ig_nocaps"] = best_nc[2]
        info["v2_ig_nocaps"] = {"hparam": json.dumps({**best_nc[1], "caps": None, "repair": None}), "selection_seconds": float("nan")}

    inp = selection_input(pool, train_ids, train_pos)
    for name, (_, param, grid) in BASELINES.items():
        best: tuple[float, Any, list[ProgramRecord], float] | None = None
        for value in grid:
            started = time.perf_counter()
            registry = run_baseline(name, inp, k, value)
            seconds = time.perf_counter() - started
            score = validation_score(registry, pool, grouped)
            tuning_log.append({"method": pool.method, "selector": name, "param": param or "", "value": value, "n_programs": len(registry), "validation_score": score, "selection_seconds": seconds})
            if best is None or score > best[0] + 1e-12:
                best = (score, value, registry, seconds)
        registries[name] = best[2]
        info[name] = {"hparam": "" if param is None else json.dumps({param: best[1]}), "selection_seconds": best[3]}

    # SRS (src/icu/v2/robust_selection.py): deployment-aware greedy selection over the source
    # schema, the known target schemas and, for the robust variants, random group-dropout schemas.
    # No validation tuning: the scenario set is the deployer's assumption, reported per variant.
    subject_of = {e.episode_id: e.subject_id for e in grouped["train"]}
    for name, q in SRS_VARIANTS:
        scenarios = scenario_set(source_features, schemas, dropout_q=q, n_dropout=20, seed=0)
        started = time.perf_counter()
        result = schema_robust_select(inp, k, scenarios, subject_of)
        registries[name] = result.registry
        info[name] = {
            "hparam": json.dumps({"dropout_q": q, "known_targets": sorted(schemas), "stop": result.stop_reason, "screened": result.screened}),
            "selection_seconds": time.perf_counter() - started,
        }
    # schema-aware baseline: mRMR restricted to candidates executable on the source and on every known target
    common = frozenset(source_features).intersection(*schemas.values()) if schemas else frozenset(source_features)
    filtered = SelectionInput(tuple(p for p in inp.candidates if frozenset(p.features) <= common), inp.train_firings, inp.train_ids, inp.train_positives)
    started = time.perf_counter()
    registries["mrmr_filter_known"] = run_baseline("mrmr", filtered, k) if filtered.candidates else []
    info["mrmr_filter_known"] = {"hparam": json.dumps({"executable_on": sorted(common)}), "selection_seconds": time.perf_counter() - started}
    return registries, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--selector-analysis-dir", required=True, help="folder with tuned.csv (validation-chosen selector weights)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selector-targets", default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    started_all = time.perf_counter()
    with open(args.pools, "rb") as handle:
        state = pickle.load(handle)
    with open(Path(args.selector_analysis_dir) / "tuned.csv", encoding="utf-8") as handle:
        tuned_rows = list(csv.DictReader(handle))
    targets = [t for t in args.selector_targets.split(",") if t] or None
    source_features = source_schema(state, Path(args.pools).parent / "config.json")
    grouped = episodes_by_split(state["episodes"])
    test = grouped["test"]
    y = np.array([e.label for e in test], dtype=int)
    groups = np.array([e.subject_id for e in test])

    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    tuning: list[dict[str, Any]] = []
    target_paired: list[dict[str, Any]] = []
    registry_ids: dict[str, dict[str, list[str]]] = {}
    for pool in state["pools"]:
        registries, info = build_registries(state, pool, args.k, targets, tuned_rows, tuning, source_features)
        registry_ids[pool.method] = {name: [p.program_id for p in registry] for name, registry in registries.items()}
        evaluations = {}
        for name, registry in registries.items():
            evaluation = evaluate_registry(registry, pool.firings, state["episodes"], targets=state["targets"], external_firings=pool.external_firings, n_boot=args.n_boot, seed=args.seed)
            evaluations[name] = evaluation
            rows.append({**flatten_row(pool.method, name, evaluation), "selector": name, **info[name]})
        probabilities = {
            name: target_probabilities(registries[name], pool, state, grouped)
            for name in {n for pair in TARGET_COMPARISONS for n in pair}
            if name in registries and registries[name]
        }
        for a, b in TARGET_COMPARISONS:
            if a not in probabilities or b not in probabilities:
                continue
            for record in paired_target_bootstrap(state["targets"], probabilities[a], probabilities[b], n_boot=min(args.n_boot, 1000), seed=args.seed):
                target_paired.append({"method": pool.method, "selector": a, "reference": b, **record})
        for ref in REFERENCES:
            ref_eval = evaluations.get(ref)
            if ref_eval is None or "error" in ref_eval:
                continue
            ref_p = np.array([ref_eval["test_probabilities"][e.episode_id] for e in test])
            for name, evaluation in evaluations.items():
                if name == ref or "error" in evaluation:
                    continue
                other = np.array([evaluation["test_probabilities"][e.episode_id] for e in test])
                diff = paired_cluster_bootstrap_diff(y, other, ref_p, groups, metric="auprc", n_boot=args.n_boot, seed=args.seed)
                paired.append({"method": pool.method, "selector": name, "reference": ref, **diff})
        print(f"[selection-baselines] {pool.method}: {len(registries)} registries", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, records: list[dict[str, Any]]) -> None:
        columns = sorted({k for r in records for k in r}, key=lambda c: (c not in ("method", "selector", "selection", "reference", "n_programs"), c))
        with open(out / name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)

    write("rows.csv", rows)
    write("paired.csv", paired)
    write("tuning.csv", tuning)
    write("target_paired.csv", target_paired)
    (out / "registries.json").write_text(json.dumps(registry_ids, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps({"pools": args.pools, "args": vars(args), "baselines": {k: {"param": v[1], "grid": [str(x) for x in v[2]]} for k, v in BASELINES.items()}, "target_comparisons": TARGET_COMPARISONS, "elapsed_seconds": time.perf_counter() - started_all}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
