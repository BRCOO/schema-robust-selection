#!/usr/bin/env python3
"""Schema-aware selection: what the schema term buys when the target schema is known.

Deployment setting: the target source's feature list (its schema) is known before
deployment, its data and labels are not. Selection may use the schema only. For each
candidate pool and each target t (real: eICU, HiRID; controlled: the internal test split
with features dropped), registries of K programs:

* ``native``        generator top-K, schema-unaware
* ``filter_topk``   candidates restricted to programs executable on t (sigma_f within F_t), then
                    generator top-K -- the obvious hard-filter baseline
* ``v2_agnostic``   v2 selector, validation-tuned weights, schema term over the real targets
                    (eICU + HiRID), i.e. the main registry (not specialized to t)
* ``v2_aware``      v2 selector with the schema term on t only; the other weights are the tuned
                    ones and w_schema is chosen on VALIDATION scored under t's schema
                    (non-executable coordinates zeroed on validation episodes)
* ``v2_filter``     v2 selector (tuned weights) on the executable candidates only
* ``mmr_filter``    MMR (lam chosen on validation) on the executable candidates only
* ``mrmr_filter`` / ``mmrfs_filter`` / ``l1_logistic_filter``  the strongest established selectors
                    (E1) on the executable candidates only; MMRFS's delta chosen on validation
* ``fs_cv``         SRS with the source schema only = schema-unaware forward selection on the
                    cross-validated log-likelihood (in both settings)
* ``srs_aware``     SRS (src/icu/v2/robust_selection.py) with the source schema and t as its
                    scenarios (equal weight); no validation tuning
* with ``--unaware-registries`` (E1's registries.json): the validation-tuned schema-unaware
  established selectors (default mRMR, MMRFS, L1), one registry for every target

Multi-target setting (one registry for ALL targets): ``native``; ``filter_topk`` on the
intersection of the target schemas; ``v2_agnostic``; ``v2_aware_all`` (schema term averaged
over all targets, w_schema chosen on the mean validation AUPRC under the target schemas);
``v2_filter`` on the intersection; ``srs_all`` (SRS with every target schema as a scenario).
Paired bootstrap of ``srs_all`` against the unaware registries on each target
(multi_target_paired.csv). registries.json holds every registry's program ids.

Each registry: frozen bridge fit on TRAIN (C on validation), refired on the target with
non-executable coordinates zeroed (no repair, no refit). Target AUPRC with a
patient-clustered bootstrap CI; paired bootstrap of ``v2_aware`` against ``filter_topk`` and
``native`` on the target episodes.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_schema_aware_selection.py \\
      --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl \\
      --selector-analysis-dir artifacts/v2/selector_analysis/f_mimic_iv_gap1h \\
      --out-dir artifacts/v2/schema_aware/f_mimic_iv_gap1h --agnostic-targets eicu,hirid
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
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.bridge import fit_frozen_bridge_v2, refire_external  # noqa: E402
from src.icu.v2.patterns import dedup_programs  # noqa: E402
from src.icu.v2.pipeline import MethodPool, TargetCohort, episodes_by_split, native_registry, restrict_firings, selector_registry  # noqa: E402
from src.icu.v2.program_metrics import set_redundancy  # noqa: E402
from src.icu.v2.selection_baselines import BASELINES, SelectionInput, run_baseline  # noqa: E402
from src.icu.v2.robust_selection import scenario_set, schema_robust_select  # noqa: E402
from src.icu.v2.stats import metric_value, paired_cluster_bootstrap_diff, patient_cluster_bootstrap  # noqa: E402
from src.icu.v2.types import ProgramRecord  # noqa: E402

W_SCHEMA_GRID = (0.35, 1.0, 3.0, 10.0)


def selector_analysis_module():
    spec = importlib.util.spec_from_file_location("run_selector_analysis", ROOT / "scripts" / "v2" / "run_selector_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def executable(program: ProgramRecord, features: frozenset[str]) -> bool:
    return frozenset(program.features) <= features


def filtered_pool(pool: MethodPool, features: frozenset[str]) -> MethodPool:
    keep = [p for p in pool.candidates if executable(p, features)]
    return replace(pool, candidates=keep, native_registry=None)


class Context:
    """Splits, labels and per-split firings shared by every fit."""

    def __init__(self, state: Mapping[str, Any]):
        self.state = state
        self.grouped = episodes_by_split(state["episodes"])
        self.ids = {name: [e.episode_id for e in rows] for name, rows in self.grouped.items()}
        self.labels = {name: [e.label for e in rows] for name, rows in self.grouped.items()}
        self.train_ids = frozenset(self.ids["train"])
        self.train_pos = frozenset(e.episode_id for e in self.grouped["train"] if e.label == 1)

    def fit(self, registry: Sequence[ProgramRecord], pool: MethodPool) -> dict | None:
        if not registry:
            return None
        firings = {p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) for p in registry}
        try:
            return fit_frozen_bridge_v2(list(registry), {name: restrict_firings(firings, frozenset(self.ids[name])) for name in self.ids}, self.ids, self.labels)
        except ValueError:
            return None

    def val_score_under(self, bridge: dict | None, registry: Sequence[ProgramRecord], pool: MethodPool, feature_sets: Sequence[frozenset[str]]) -> float:
        """Mean validation AUPRC with each schema's non-executable coordinates zeroed."""
        if bridge is None:
            return -math.inf
        val_firings = restrict_firings({p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) for p in registry}, frozenset(self.ids["val"]))
        scores = []
        for features in feature_sets:
            refired = refire_external(bridge["model_bundle"], list(registry), features, val_firings, self.ids["val"])
            scores.append(metric_value(self.labels["val"], [refired["probabilities"][e] for e in self.ids["val"]], "auprc"))
        return float(np.mean(scores))

    def score_target(self, bridge: dict, registry: Sequence[ProgramRecord], pool: MethodPool, target: TargetCohort, n_boot: int, seed: int) -> dict[str, Any]:
        refired = refire_external(bridge["model_bundle"], list(registry), target.features, pool.external_firings.get(target.name, {}), [e.episode_id for e in target.episodes])
        y = np.array([e.label for e in target.episodes], dtype=int)
        p = np.array([refired["probabilities"][e.episode_id] for e in target.episodes])
        g = np.array([e.subject_id for e in target.episodes])
        boot = patient_cluster_bootstrap(y, p, g, "auprc", n_boot=n_boot, seed=seed)
        return {"auprc": boot["point"], "auprc_low": boot["low"], "auprc_high": boot["high"], "auroc": metric_value(y, p, "auroc"), "schema_survival": refired["schema_survival"], "probabilities": p, "labels": y, "groups": g}

    def internal_auprc(self, bridge: dict) -> float:
        test = self.grouped["test"]
        return metric_value([e.label for e in test], [bridge["test_probabilities"][e.episode_id] for e in test], "auprc")


def tuned_weights(tuned_rows: Sequence[Mapping[str, str]], method: str, keys: Sequence[str]) -> dict[str, float]:
    row = next((r for r in tuned_rows if r["method"] == method and r["selection"] == "v2_selector"), None)
    if row is None:
        raise SystemExit(f"no tuned weights for {method}")
    return {key: float(row[key]) for key in keys}


def choose_w_schema(ctx: Context, pool: MethodPool, sa, k: int, weights: dict[str, float], schemas: Mapping[str, frozenset[str]], log: list, tag: str) -> tuple[list[ProgramRecord], float]:
    best = None
    for w in W_SCHEMA_GRID:
        registry = selector_registry(pool, sa.base(k, **{**weights, "w_schema": w}), ctx.train_ids, schemas)[0]
        bridge = ctx.fit(registry, pool)
        score = ctx.val_score_under(bridge, registry, pool, list(schemas.values()))
        log.append({"method": pool.method, "scenario": tag, "registry": "v2_aware", "param": "w_schema", "value": w, "n_programs": len(registry), "validation_score_under_schema": score})
        if best is None or score > best[0] + 1e-12:
            best = (score, w, registry)
    return best[2], best[1]


def choose_baseline(ctx: Context, pool: MethodPool, k: int, name: str, log: list, tag: str) -> tuple[list[ProgramRecord], Any]:
    """Established selector ``name`` (selection_baselines.BASELINES) on ``pool``; its hyper-parameter,
    if any, chosen on validation like in E1."""
    candidates = tuple(dedup_programs(pool.candidates))
    inp = SelectionInput(candidates, {p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) & ctx.train_ids for p in candidates}, ctx.train_ids, ctx.train_pos)
    _, param, grid = BASELINES[name]
    best = None
    for value in grid:
        registry = run_baseline(name, inp, k, value)
        bridge = ctx.fit(registry, pool)
        score = -math.inf if bridge is None else float(bridge["validation_score"])
        if param is not None:
            log.append({"method": pool.method, "scenario": tag, "registry": f"{name}_filter", "param": param, "value": value, "n_programs": len(registry), "validation_score_under_schema": score})
        if best is None or score > best[0] + 1e-12:
            best = (score, value, registry)
    return best[2], best[1]


def choose_mmr(ctx: Context, pool: MethodPool, k: int, log: list, tag: str) -> tuple[list[ProgramRecord], float]:
    return choose_baseline(ctx, pool, k, "mmr", log, tag)


FILTER_BASELINES = ("mrmr", "mmrfs", "l1_logistic")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--selector-analysis-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--agnostic-targets", default="eicu,hirid", help="schemas the agnostic (main) registry was selected with")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unaware-registries", default="", help="E1 registries.json (run_selection_baselines.py, same K)")
    parser.add_argument("--unaware-selectors", default="mrmr,mmrfs,l1_logistic")
    args = parser.parse_args(argv)
    unaware_ids = json.loads(Path(args.unaware_registries).read_text(encoding="utf-8")) if args.unaware_registries else {}
    unaware_names = [s for s in args.unaware_selectors.split(",") if s] if unaware_ids else []
    started = time.perf_counter()
    with open(args.pools, "rb") as handle:
        state = pickle.load(handle)
    with open(Path(args.selector_analysis_dir) / "tuned.csv", encoding="utf-8") as handle:
        tuned_rows = list(csv.DictReader(handle))
    sa = selector_analysis_module()
    ctx = Context(state)
    targets: list[TargetCohort] = list(state["targets"])
    agnostic_names = [t for t in args.agnostic_targets.split(",") if t]
    agnostic_schemas = {t.name: t.features for t in targets if t.name in agnostic_names}
    config_path = Path(args.pools).parent / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")).get("internal_features"):
        source_features = frozenset(json.loads(config_path.read_text(encoding="utf-8"))["internal_features"])
    else:
        source_features = frozenset().union(*(t.features for t in targets))
    subject_of = {e.episode_id: e.subject_id for e in ctx.grouped["train"]}

    def srs(pool: MethodPool, schemas: Mapping[str, frozenset[str]]) -> list[ProgramRecord]:
        candidates = tuple(dedup_programs(pool.candidates))
        inp = SelectionInput(candidates, {p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) & ctx.train_ids for p in candidates}, ctx.train_ids, ctx.train_pos)
        return schema_robust_select(inp, args.k, scenario_set(source_features, schemas), subject_of).registry
    single_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    multi_rows: list[dict[str, Any]] = []
    multi_paired: list[dict[str, Any]] = []
    tuning: list[dict[str, Any]] = []
    registry_ids: dict[str, dict[str, list[str]]] = {}

    def record(rows: list, base: dict, registry, pool, bridge, target, extra=None) -> dict | None:
        if bridge is None:
            rows.append({**base, "n_programs": len(registry), "error": "empty or unfit registry"})
            return None
        scored = ctx.score_target(bridge, registry, pool, target, args.n_boot, args.seed)
        train_firings = restrict_firings({p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) for p in registry}, ctx.train_ids)
        rows.append({
            **base, "n_programs": len(registry), "target_auprc": scored["auprc"], "target_auprc_low": scored["auprc_low"], "target_auprc_high": scored["auprc_high"],
            "target_auroc": scored["auroc"], "target_schema_survival": scored["schema_survival"], "internal_auprc": ctx.internal_auprc(bridge),
            "redundancy": set_redundancy(registry, train_firings), **(extra or {}),
        })
        return scored

    for pool in state["pools"]:
        weights = tuned_weights(tuned_rows, pool.method, list(sa.WEIGHT_GRID))
        agnostic = selector_registry(pool, sa.base(args.k, **weights), ctx.train_ids, agnostic_schemas)[0]
        native = native_registry(pool, args.k)
        # schema-unaware forward selection (SRS with the source schema only): isolates what knowing the
        # target schema adds on top of the cross-validated likelihood objective itself
        fs_cv = srs(pool, {})
        fitted = {"native": (native, ctx.fit(native, pool)), "v2_agnostic": (agnostic, ctx.fit(agnostic, pool)), "fs_cv": (fs_cv, ctx.fit(fs_cv, pool))}
        by_id = {c.program_id: c for c in pool.candidates}
        for name in unaware_names:
            registry = [by_id[i] for i in unaware_ids[pool.method][name]]
            fitted[name] = (registry, ctx.fit(registry, pool))
        ids_out = registry_ids.setdefault(pool.method, {})
        ids_out.update({name: [c.program_id for c in reg] for name, (reg, _) in fitted.items()})
        # ---------------- single known target
        for target in targets:
            f_pool = filtered_pool(pool, target.features)
            regs = dict(fitted)
            filter_topk = dedup_programs(f_pool.candidates)[: args.k]
            regs["filter_topk"] = (filter_topk, ctx.fit(filter_topk, pool))
            aware, w_aware = choose_w_schema(ctx, pool, sa, args.k, weights, {target.name: target.features}, tuning, target.name)
            regs["v2_aware"] = (aware, ctx.fit(aware, pool))
            v2_filter = selector_registry(f_pool, sa.base(args.k, **weights), ctx.train_ids, {target.name: target.features})[0] if f_pool.candidates else []
            regs["v2_filter"] = (v2_filter, ctx.fit(v2_filter, pool))
            mmr_f, lam = choose_mmr(ctx, f_pool, args.k, tuning, target.name) if f_pool.candidates else ([], None)
            regs["mmr_filter"] = (mmr_f, ctx.fit(mmr_f, pool))
            chosen = {}
            for name in FILTER_BASELINES:
                registry, chosen[name] = choose_baseline(ctx, f_pool, args.k, name, tuning, target.name) if f_pool.candidates else ([], None)
                regs[f"{name}_filter"] = (registry, ctx.fit(registry, pool))
            srs_aware = srs(pool, {target.name: target.features})
            regs["srs_aware"] = (srs_aware, ctx.fit(srs_aware, pool))
            ids_out.update({f"{name}@{target.name}": [c.program_id for c in reg] for name, (reg, _) in regs.items() if name not in fitted})
            scored = {}
            for name, (registry, bridge) in regs.items():
                extra = {"w_schema": w_aware} if name == "v2_aware" else ({"lam": lam} if name == "mmr_filter" else {})
                if name == "mmrfs_filter":
                    extra = {"delta": chosen["mmrfs"]}
                scored[name] = record(single_rows, {"method": pool.method, "target": target.name, "registry": name, "n_executable_candidates": len(f_pool.candidates)}, registry, pool, bridge, target, extra)
            for lead, ref in (("v2_aware", "filter_topk"), ("v2_aware", "native"), ("v2_aware", "v2_agnostic"),
                              ("srs_aware", "filter_topk"), ("srs_aware", "mmr_filter"), ("srs_aware", "v2_aware"), ("srs_aware", "native"),
                              ("srs_aware", "fs_cv"), *(("srs_aware", name) for name in unaware_names),
                              *(("srs_aware", f"{name}_filter") for name in FILTER_BASELINES)):
                if scored.get(lead) is None or scored.get(ref) is None:
                    continue
                a, b = scored[lead], scored[ref]
                diff = paired_cluster_bootstrap_diff(a["labels"], a["probabilities"], b["probabilities"], a["groups"], metric="auprc", n_boot=args.n_boot, seed=args.seed)
                paired_rows.append({"method": pool.method, "target": target.name, "comparison": f"{lead} - {ref}", **diff})
        # ---------------- one registry for all targets
        all_schemas = {t.name: t.features for t in targets}
        intersection = frozenset.intersection(*all_schemas.values())
        i_pool = filtered_pool(pool, intersection)
        regs = dict(fitted)
        filter_topk = dedup_programs(i_pool.candidates)[: args.k]
        regs["filter_topk_intersection"] = (filter_topk, ctx.fit(filter_topk, pool))
        aware_all, w_all = choose_w_schema(ctx, pool, sa, args.k, weights, all_schemas, tuning, "all_targets")
        regs["v2_aware_all"] = (aware_all, ctx.fit(aware_all, pool))
        v2_int = selector_registry(i_pool, sa.base(args.k, **weights), ctx.train_ids, all_schemas)[0] if i_pool.candidates else []
        regs["v2_filter_intersection"] = (v2_int, ctx.fit(v2_int, pool))
        srs_all = srs(pool, all_schemas)
        regs["srs_all"] = (srs_all, ctx.fit(srs_all, pool))
        ids_out.update({f"{name}@all": [c.program_id for c in reg] for name, (reg, _) in regs.items() if name not in fitted})
        scored_all: dict[str, dict] = {}
        for name, (registry, bridge) in regs.items():
            row: dict[str, Any] = {"method": pool.method, "registry": name, "n_programs": len(registry), "n_intersection_candidates": len(i_pool.candidates), "w_schema": w_all if name == "v2_aware_all" else ""}
            if bridge is None:
                multi_rows.append({**row, "error": "empty or unfit registry"})
                continue
            per = {t.name: ctx.score_target(bridge, registry, pool, t, 200, args.seed) for t in targets}
            scored_all[name] = per
            row.update({f"{t}_auprc": s["auprc"] for t, s in per.items()})
            row.update({f"{t}_survival": s["schema_survival"] for t, s in per.items()})
            row.update({"mean_target_auprc": float(np.mean([s["auprc"] for s in per.values()])), "min_target_auprc": float(min(s["auprc"] for s in per.values())), "internal_auprc": ctx.internal_auprc(bridge)})
            multi_rows.append(row)
        for ref in ("fs_cv", "filter_topk_intersection", *unaware_names):
            if "srs_all" not in scored_all or ref not in scored_all:
                continue
            for target in targets:
                a, b = scored_all["srs_all"][target.name], scored_all[ref][target.name]
                diff = paired_cluster_bootstrap_diff(a["labels"], a["probabilities"], b["probabilities"], a["groups"], metric="auprc", n_boot=args.n_boot, seed=args.seed)
                multi_paired.append({"method": pool.method, "target": target.name, "comparison": f"srs_all - {ref}", **diff})
        print(f"[schema-aware] {pool.method} done ({time.perf_counter() - started:.0f}s)", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, records: list[dict[str, Any]]) -> None:
        columns = sorted({k for r in records for k in r}, key=lambda c: (c not in ("method", "target", "registry", "scenario", "comparison"), c))
        with open(out / name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)

    write("single_target.csv", single_rows)
    write("single_target_paired.csv", paired_rows)
    write("multi_target.csv", multi_rows)
    write("multi_target_paired.csv", multi_paired)
    (out / "registries.json").write_text(json.dumps(registry_ids, indent=1), encoding="utf-8")
    write("tuning.csv", tuning)
    (out / "manifest.json").write_text(json.dumps({"pools": args.pools, "args": vars(args), "w_schema_grid": W_SCHEMA_GRID, "elapsed_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
