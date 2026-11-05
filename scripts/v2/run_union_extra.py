#!/usr/bin/env python3
"""E3 add-on: stronger and schema-robust baselines, the unknown-target setting, and schema-loss
stress curves, for one finished E3 run (``run_union_selection.py``).

The E3 registries are read from ``--e3-dir`` (registries.json; pools, K, known targets and the
surrogate / shortlist settings come from its manifest) and are not re-selected. Added registries,
all on the same union pool and TRAIN-only, deployed with surrogates like every E3 registry:

* ``jmi`` / ``cmim``: the information-theoretic filters of ``selection_baselines.EXTRA_BASELINES``;
* ``l1_dropout``: L1-logistic selection trained with schema-dropout augmentation
  (``selection_baselines.l1_schema_dropout``), q chosen on validation from ``--l1-dropout-grid``;
* ``srs_c_unknown25``: SRS-C with the source schema and random group-dropout scenarios (q = 0.25)
  and no known target; ``srs_c_unknown_val`` picks ``fs_cv`` (q = 0) or ``srs_c_unknown25`` on
  validation, the same rule as ``srs_c_val`` -- SRS-C-V when nothing is known about the targets.

Outputs (``--out-dir``): ``rows.csv`` (every registry, plain and composite, as in E3);
``paired_internal.csv`` / ``paired_targets.csv`` (patient-clustered paired bootstrap; composite
against composite): ``srs_c_val`` against the added registries, and ``srs_c_unknown_val`` against
every registry that uses no target schema (plus ``mrmr_filter_known`` for reference);
``stress.csv``: internal TEST AUPRC under random schemas that drop every source feature
(``feature``) or every coarse group (``group``) independently with probability l -- the same
draws for every registry, plain and composite deployment; ``tuning.csv``, ``registries.json``,
``manifest.json``.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_union_extra.py \\
      --e3-dir artifacts/v2/union/f_mimic_iv_gap1h --out-dir artifacts/v2/union_extra/union/f_mimic_iv_gap1h
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
from src.icu.v2.pipeline import episodes_by_split, restrict_firings  # noqa: E402
from src.icu.v2.robust_selection import scenario_set, schema_robust_select  # noqa: E402
from src.icu.v2.selection_baselines import SelectionInput, l1_schema_dropout, run_baseline  # noqa: E402
from src.icu.v2.stats import metric_value, paired_cluster_bootstrap_diff  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, composite_registry, refire_composite  # noqa: E402
from src.icu.v2.types import coarse_group  # noqa: E402
from src.icu.v2.union import union_pool  # noqa: E402

#: registries that use no target schema (fair opponents in the unknown-target setting)
NO_TARGET = ("fs_cv", "mrmr", "jmi", "cmim", "mmrfs", "l1_logistic", "l1_dropout", "mmr", "dpp", "mms_xin")


def selection_baselines_module():
    spec = importlib.util.spec_from_file_location("run_selection_baselines", ROOT / "scripts" / "v2" / "run_selection_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = sorted({k for r in records for k in r}, key=lambda c: (c not in ("registry", "variant", "a", "b", "target", "family", "level"), c))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e3-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--l1-dropout-grid", default="0.25,0.5")
    parser.add_argument("--stress-levels", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--stress-draws", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    e3 = Path(args.e3_dir)
    e3_manifest = json.loads((e3 / "manifest.json").read_text(encoding="utf-8"))
    e3_args = e3_manifest["args"]
    k, max_surrogates, shortlist = int(e3_args["k"]), int(e3_args["max_surrogates"]), int(e3_args["shortlist"]) or None
    sb = selection_baselines_module()
    pools_path = ROOT / e3_args["pools"]
    with open(pools_path, "rb") as handle:
        state = pickle.load(handle)
    source = sb.source_schema(state, pools_path.parent / "config.json")
    grouped = episodes_by_split(state["episodes"])
    ids = {name: [e.episode_id for e in rows] for name, rows in grouped.items()}
    labels = {name: [e.label for e in rows] for name, rows in grouped.items()}
    train_ids = frozenset(ids["train"])
    train_pos = frozenset(e.episode_id for e in grouped["train"] if e.label == 1)
    subject_of = {e.episode_id: e.subject_id for e in grouped["train"]}
    targets = list(state["targets"])
    candidates, firings, external = union_pool(state)
    by_id = {c.program_id: c for pool in state["pools"] for c in pool.candidates}
    train_firings = {c.program_id: firings[c.program_id] & train_ids for c in candidates}
    inp = SelectionInput(tuple(candidates), train_firings, train_ids, train_pos)
    print(f"[extra] {e3}: {len(candidates)} candidates, K = {k}", flush=True)

    def fit(registry):
        f = {p.program_id: firings.get(p.program_id, frozenset()) for p in registry}
        return fit_frozen_bridge_v2(list(registry), {s: restrict_firings(f, frozenset(ids[s])) for s in ids}, ids, labels)

    def composite(registry):
        return composite_registry(registry, candidates, train_firings, sorted(train_ids), max_surrogates)

    saved = json.loads((e3 / "registries.json").read_text(encoding="utf-8"))
    registries: dict[str, CompositeRegistry] = {
        name: CompositeRegistry(tuple(by_id[i] for i in reg["primaries"]),
                                {pid: tuple(by_id[q] for q in qs) for pid, qs in reg["surrogates"].items()})
        for name, reg in saved.items()}
    meta: dict[str, dict[str, Any]] = {}
    tuning: list[dict[str, Any]] = []
    for name in ("jmi", "cmim"):
        t0 = time.perf_counter()
        registries[name] = composite(run_baseline(name, inp, k))
        meta[name] = {"selection_seconds": time.perf_counter() - t0}
    t0, best = time.perf_counter(), None
    for q in [float(x) for x in args.l1_dropout_grid.split(",") if x]:
        registry = l1_schema_dropout(inp, k, q=q, source_features=source, seed=args.seed)
        score = fit(registry)["validation_score"] if registry else -np.inf
        tuning.append({"registry": "l1_dropout", "param": "q", "value": q, "n_programs": len(registry), "validation_score": score})
        if best is None or score > best[0] + 1e-12:
            best = (score, q, registry)
    registries["l1_dropout"] = composite(best[2])
    meta["l1_dropout"] = {"param": "q", "value": best[1], "selection_seconds": time.perf_counter() - t0}
    t0 = time.perf_counter()
    result = schema_robust_select(inp, k, scenario_set(source, {}, dropout_q=0.25), subject_of, surrogates=True,
                                  max_surrogates=max_surrogates, shortlist=shortlist)
    registries["srs_c_unknown25"] = CompositeRegistry(tuple(result.registry), result.surrogates)
    meta["srs_c_unknown25"] = {"selection_seconds": time.perf_counter() - t0, "stop": result.stop_reason}
    val = {name: fit(list(registries[name].primaries))["validation_score"] for name in ("fs_cv", "srs_c_unknown25")}
    chosen = "srs_c_unknown25" if val["srs_c_unknown25"] > val["fs_cv"] + 1e-12 else "fs_cv"
    registries["srs_c_unknown_val"] = registries[chosen]
    meta["srs_c_unknown_val"] = {"chosen": chosen, "dropout_q": 0.25 if chosen == "srs_c_unknown25" else 0.0, "validation_scores": val}
    print(f"[extra] selection done ({time.perf_counter() - started:.0f}s); l1_dropout q = {best[1]}, unknown-target pick = {chosen}", flush=True)

    rows: list[dict[str, Any]] = []
    bundles: dict[str, dict] = {}
    internal_probs: dict[str, np.ndarray] = {}
    target_probs: dict[str, dict[str, np.ndarray]] = {}
    test_y = np.array(labels["test"], dtype=int)
    test_g = np.array([e.subject_id for e in grouped["test"]])
    for name, comp in registries.items():
        registry = list(comp.primaries)
        bridge = fit(registry)
        bundles[name] = bridge["model_bundle"]
        p_test = np.array([bridge["test_probabilities"][e] for e in ids["test"]])
        internal_probs[name] = p_test
        base = {"registry": name, "n_programs": len(registry), "generators": ",".join(sorted({p.method for p in registry})),
                "internal_auprc": metric_value(test_y, p_test, "auprc"), "internal_auroc": metric_value(test_y, p_test, "auroc"),
                "validation_score": bridge["validation_score"], **meta.get(name, {})}
        for variant in ("plain", "composite"):
            row, per_target = dict(base, variant=variant), {}
            for target in targets:
                eids = [e.episode_id for e in target.episodes]
                if variant == "plain":
                    refired = refire_external(bridge["model_bundle"], registry, target.features, external[target.name], eids)
                else:
                    refired = refire_composite(bridge["model_bundle"], comp, target.features, external[target.name], eids)
                p = np.array([refired["probabilities"][e] for e in eids])
                per_target[target.name] = p
                row[f"{target.name}_auprc"] = metric_value([e.label for e in target.episodes], p, "auprc")
                row[f"{target.name}_survival"] = refired["schema_survival"]
            row["mean_target_auprc"] = float(np.mean([row[f"{t.name}_auprc"] for t in targets]))
            row["worst_target_auprc"] = float(min(row[f"{t.name}_auprc"] for t in targets))
            rows.append(row)
            if variant == "composite":
                target_probs[name] = per_target

    comparisons = [("srs_c_val", b) for b in ("jmi", "cmim", "l1_dropout", "srs_c_unknown_val")]
    comparisons += [("srs_c_unknown_val", b) for b in (*NO_TARGET, "mrmr_filter_known") if b in registries]
    paired_internal, paired_targets = [], []
    for a, b in comparisons:
        diff = paired_cluster_bootstrap_diff(test_y, internal_probs[a], internal_probs[b], test_g, metric="auprc", n_boot=args.n_boot, seed=args.seed)
        paired_internal.append({"a": a, "b": b, **diff})
        for record in sb.paired_target_bootstrap(targets, target_probs[a], target_probs[b], n_boot=args.n_boot, seed=args.seed):
            paired_targets.append({"a": f"{a}:composite", "b": f"{b}:composite", **record})
    print(f"[extra] paired tests done ({time.perf_counter() - started:.0f}s)", flush=True)

    stress: list[dict[str, Any]] = []
    test_ids = ids["test"]
    test_set = frozenset(test_ids)
    used = {q.program_id for comp in registries.values() for p in comp.primaries for q in comp.branches(p)}
    test_firings = {pid: firings.get(pid, frozenset()) & test_set for pid in used}
    groups = sorted({coarse_group(f) for f in source})
    levels = [float(x) for x in args.stress_levels.split(",") if x]
    for fi, family in enumerate(("feature", "group")):
        for li, level in enumerate(levels):
            for draw in range(args.stress_draws):
                rng = np.random.default_rng([args.seed, fi, li, draw])
                if family == "feature":
                    kept = frozenset(f for f in sorted(source) if rng.random() >= level)
                else:
                    kept_groups = {g for g in groups if rng.random() >= level}
                    kept = frozenset(f for f in source if coarse_group(f) in kept_groups)
                for name, comp in registries.items():
                    for variant in ("plain", "composite"):
                        if variant == "plain":
                            out = refire_external(bundles[name], list(comp.primaries), kept, test_firings, test_ids)
                        else:
                            out = refire_composite(bundles[name], comp, kept, test_firings, test_ids)
                        p = np.array([out["probabilities"][e] for e in test_ids])
                        stress.append({"family": family, "level": level, "draw": draw, "registry": name, "variant": variant,
                                       "n_kept_features": len(kept), "survival": out["schema_survival"],
                                       "auprc": metric_value(test_y, p, "auprc")})
    print(f"[extra] stress done ({time.perf_counter() - started:.0f}s)", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for file_name, records in (("rows.csv", rows), ("paired_internal.csv", paired_internal), ("paired_targets.csv", paired_targets),
                               ("tuning.csv", tuning), ("stress.csv", stress)):
        write_csv(out / file_name, records)
    new = ("jmi", "cmim", "l1_dropout", "srs_c_unknown25", "srs_c_unknown_val")
    (out / "registries.json").write_text(json.dumps({name: {"primaries": [p.program_id for p in registries[name].primaries],
                                                            "surrogates": {pid: [q.program_id for q in qs] for pid, qs in registries[name].surrogates.items()}}
                                                     for name in new}, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"e3_dir": str(e3), "pools": e3_args["pools"], "k": k, "args": vars(args), "meta": meta,
                                                   "n_candidates": len(candidates), "elapsed_seconds": time.perf_counter() - started},
                                                  indent=2, default=str), encoding="utf-8")
    print(f"[extra] wrote {out} ({time.perf_counter() - started:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
