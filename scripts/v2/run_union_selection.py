#!/usr/bin/env python3
"""E3: one registry from the union of all generators' pools (cross-generator selection).

Candidates: the de-duplicated candidates of every generator pool of ``--pools`` (program ids are
generator-prefixed, so the union is disjoint); firings and target firings are the pools' own.
Registries of K programs on the union:

* established selectors (``selection_baselines.BASELINES``: MMR, Xin MMS, MMRFS, mRMR, DPP,
  L1), hyper-parameters chosen on validation exactly as in E1;
* ``mrmr_filter_known``: mRMR on the candidates executable on every known target (the
  schema-informed established baseline);
* ``fs_cv``: SRS with the source schema only (schema-unaware forward selection);
* ``srs_known``: SRS, scenarios = source + ``--known-targets``;
* ``srs_c``: SRS-C, the same with composite candidates (CART surrogates, ``surrogates.py``);
* ``srs_c_robust25``: SRS-C plus random coarse-group dropout scenarios (q = 0.25);
* ``srs_c_val`` (the main method, the default ``--reference``): SRS-C with the schema-dropout rate
  q chosen on validation from {0, 0.25}, i.e. ``srs_c`` or ``srs_c_robust25`` -- the established
  selectors are tuned on validation too;
* ``srs_projected_insample``: SRS (known targets) with the projected in-sample objective, the
  variant with the (1 - e^-gamma) guarantee;
* ``best_single_srs`` / ``best_single_mrmr``: the E1 registry (``--e1-dir``) of the generator
  whose registry has the best validation AUPRC -- what a single-generator system would ship.

Every registry is deployed ``plain`` (non-executable programs zeroed) and ``composite``
(surrogates chosen among the union candidates on TRAIN firings; SRS-C keeps its own). The
probe is fit once per registry on the primaries' source firings (TRAIN, C on validation).
Paired patient-clustered bootstrap: internal test AUPRC of ``--reference`` against every
other registry; mean target AUPRC (joint replicates) of the reference's composite against
every registry's composite and plain deployment, and of every composite against its plain.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_union_selection.py \\
      --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl \\
      --e1-dir artifacts/v2/selection_baselines/f_mimic_iv_gap1h --out-dir artifacts/v2/union/f_mimic_iv_gap1h
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
from src.icu.v2.program_metrics import set_redundancy  # noqa: E402
from src.icu.v2.robust_selection import scenario_set, schema_robust_select  # noqa: E402
from src.icu.v2.selection_baselines import BASELINES, SelectionInput, run_baseline  # noqa: E402
from src.icu.v2.stats import metric_value, paired_cluster_bootstrap_diff  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, composite_registry, refire_composite  # noqa: E402


def selection_baselines_module():
    spec = importlib.util.spec_from_file_location("run_selection_baselines", ROOT / "scripts" / "v2" / "run_selection_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--e1-dir", default="", help="E1 output dir of the same run (rows.csv + registries.json) for the best single-generator registries")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--known-targets", default="eicu,hirid")
    parser.add_argument("--reference", default="srs_c_val")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-surrogates", type=int, default=5)
    parser.add_argument("--shortlist", type=int, default=10, help="fast SRS shortlist (0 = exact greedy); the frozen variants only")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    sb = selection_baselines_module()
    with open(args.pools, "rb") as handle:
        state = pickle.load(handle)
    source = sb.source_schema(state, Path(args.pools).parent / "config.json")
    grouped = episodes_by_split(state["episodes"])
    ids = {name: [e.episode_id for e in rows] for name, rows in grouped.items()}
    labels = {name: [e.label for e in rows] for name, rows in grouped.items()}
    train_ids = frozenset(ids["train"])
    train_pos = frozenset(e.episode_id for e in grouped["train"] if e.label == 1)
    subject_of = {e.episode_id: e.subject_id for e in grouped["train"]}
    known = {t.name: t.features for t in state["targets"] if t.name in [x for x in args.known_targets.split(",") if x]}
    targets = list(state["targets"])

    candidates, firings, external = [], {}, {}
    for pool in state["pools"]:
        for c in dedup_programs(pool.candidates):
            candidates.append(c)
            firings[c.program_id] = frozenset(pool.firings.get(c.program_id, frozenset()))
            for name, fired in pool.external_firings.items():
                external.setdefault(name, {})[c.program_id] = frozenset(fired.get(c.program_id, frozenset()))
    by_id = {c.program_id: c for pool in state["pools"] for c in pool.candidates}
    train_firings = {c.program_id: firings[c.program_id] & train_ids for c in candidates}
    inp = SelectionInput(tuple(candidates), train_firings, train_ids, train_pos)
    print(f"[union] {len(candidates)} candidates from {len(state['pools'])} generators", flush=True)

    def fit(registry):
        f = {p.program_id: firings[p.program_id] for p in registry}
        return fit_frozen_bridge_v2(list(registry), {s: restrict_firings(f, frozenset(ids[s])) for s in ids}, ids, labels)

    registries: dict[str, CompositeRegistry] = {}
    meta: dict[str, dict[str, Any]] = {}
    tuning: list[dict[str, Any]] = []
    for name, (_, param, grid) in BASELINES.items():
        t0 = time.perf_counter()
        best = None
        for value in grid:
            registry = run_baseline(name, inp, args.k, value)
            score = fit(registry)["validation_score"] if registry else -np.inf
            tuning.append({"registry": name, "param": param, "value": value, "n_programs": len(registry), "validation_score": score})
            if best is None or score > best[0] + 1e-12:
                best = (score, value, registry)
        registries[name] = composite_registry(best[2], candidates, train_firings, sorted(train_ids), args.max_surrogates)
        meta[name] = {"param": param, "value": best[1], "selection_seconds": time.perf_counter() - t0}
    known_exec = [c for c in candidates if all(frozenset(c.features) <= f for f in known.values())]
    if known_exec:
        inp_known = SelectionInput(tuple(known_exec), {c.program_id: train_firings[c.program_id] for c in known_exec}, train_ids, train_pos)
        t0 = time.perf_counter()
        registry = run_baseline("mrmr", inp_known, args.k, None)
        registries["mrmr_filter_known"] = composite_registry(registry, candidates, train_firings, sorted(train_ids), args.max_surrogates)
        meta["mrmr_filter_known"] = {"n_candidates": len(known_exec), "selection_seconds": time.perf_counter() - t0}
    srs_variants = (
        ("fs_cv", {}, 0.0, False, "frozen", True),
        ("srs_known", known, 0.0, False, "frozen", True),
        ("srs_c", known, 0.0, True, "frozen", True),
        ("srs_c_robust25", known, 0.25, True, "frozen", True),  # + random coarse-group dropout scenarios
        ("srs_projected_insample", known, 0.0, False, "projected", False),  # the variant with the guarantee
    )
    for name, scen, dropout_q, use_surrogates, mode, cv in srs_variants:
        t0 = time.perf_counter()
        result = schema_robust_select(inp, args.k, scenario_set(source, scen, dropout_q=dropout_q), subject_of, surrogates=use_surrogates,
                                      max_surrogates=args.max_surrogates, mode=mode, cv=cv,
                                      shortlist=(args.shortlist or None) if mode == "frozen" else None)
        registries[name] = (CompositeRegistry(tuple(result.registry), result.surrogates) if use_surrogates
                            else composite_registry(result.registry, candidates, train_firings, sorted(train_ids), args.max_surrogates))
        meta[name] = {"selection_seconds": time.perf_counter() - t0, "stop": result.stop_reason}
    val = {name: fit(list(registries[name].primaries))["validation_score"] for name in ("srs_c", "srs_c_robust25")}
    chosen = "srs_c_robust25" if val["srs_c_robust25"] > val["srs_c"] + 1e-12 else "srs_c"
    registries["srs_c_val"] = registries[chosen]
    meta["srs_c_val"] = {"chosen": chosen, "dropout_q": 0.25 if chosen == "srs_c_robust25" else 0.0, "validation_scores": val,
                         "selection_seconds": meta["srs_c"]["selection_seconds"] + meta["srs_c_robust25"]["selection_seconds"]}
    if args.e1_dir:
        e1_rows = [r for r in csv.DictReader(open(Path(args.e1_dir) / "rows.csv", encoding="utf-8")) if not r.get("error")]
        e1_regs = json.loads((Path(args.e1_dir) / "registries.json").read_text(encoding="utf-8"))
        for selector in ("srs_known", "mrmr"):
            rows_s = [r for r in e1_rows if r["selector"] == selector]
            if not rows_s:
                continue
            chosen = max(rows_s, key=lambda r: float(r["validation_score"]))
            registry = [by_id[i] for i in e1_regs[chosen["method"]][selector]]
            registries[f"best_single_{'srs' if selector == 'srs_known' else selector}"] = composite_registry(
                registry, candidates, train_firings, sorted(train_ids), args.max_surrogates)
            meta[f"best_single_{'srs' if selector == 'srs_known' else selector}"] = {"generator": chosen["method"], "e1_validation_score": float(chosen["validation_score"])}

    rows: list[dict[str, Any]] = []
    internal_probs: dict[str, np.ndarray] = {}
    target_probs: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    test_y = np.array(labels["test"], dtype=int)
    test_g = np.array([e.subject_id for e in grouped["test"]])
    for name, comp in registries.items():
        registry = list(comp.primaries)
        bridge = fit(registry)
        p_test = np.array([bridge["test_probabilities"][e] for e in ids["test"]])
        internal_probs[name] = p_test
        base = {"registry": name, "n_programs": len(registry), "generators": ",".join(sorted({p.method for p in registry})),
                "n_generators": len({p.method for p in registry}), "internal_auprc": metric_value(test_y, p_test, "auprc"),
                "internal_auroc": metric_value(test_y, p_test, "auroc"), "validation_score": bridge["validation_score"],
                "redundancy": set_redundancy(registry, {p.program_id: train_firings[p.program_id] for p in registry}),
                **{k: v for k, v in meta.get(name, {}).items()}}
        for variant in ("plain", "composite"):
            row, per_target = dict(base, variant=variant), {}
            for target in targets:
                eids = [e.episode_id for e in target.episodes]
                if variant == "plain":
                    refired = refire_external(bridge["model_bundle"], registry, target.features, external[target.name], eids)
                    row[f"{target.name}_surrogate_fraction"] = 0.0
                else:
                    refired = refire_composite(bridge["model_bundle"], comp, target.features, external[target.name], eids)
                    row[f"{target.name}_surrogate_fraction"] = refired["surrogate_fraction"]
                p = np.array([refired["probabilities"][e] for e in eids])
                per_target[target.name] = p
                row[f"{target.name}_auprc"] = metric_value([e.label for e in target.episodes], p, "auprc")
                row[f"{target.name}_survival"] = refired["schema_survival"]
            row["mean_target_auprc"] = float(np.mean([row[f"{t.name}_auprc"] for t in targets]))
            row["worst_target_auprc"] = float(min(row[f"{t.name}_auprc"] for t in targets))
            rows.append(row)
            target_probs[(name, variant)] = per_target
        print(f"[union] {name}: internal {base['internal_auprc']:.3f}, mean target plain/composite "
              f"{rows[-2]['mean_target_auprc']:.3f}/{rows[-1]['mean_target_auprc']:.3f} ({base['generators']})", flush=True)

    ref = args.reference
    paired_internal = []
    for name in registries:
        if name == ref:
            continue
        diff = paired_cluster_bootstrap_diff(test_y, internal_probs[ref], internal_probs[name], test_g, metric="auprc", n_boot=args.n_boot, seed=args.seed)
        paired_internal.append({"a": ref, "b": name, **diff})
    paired_targets = []
    comparisons = [((ref, "composite"), (name, variant)) for name in registries if name != ref for variant in ("composite", "plain")]
    comparisons += [((name, "composite"), (name, "plain")) for name in registries]
    for a, b in comparisons:
        for record in sb.paired_target_bootstrap(targets, target_probs[a], target_probs[b], n_boot=args.n_boot, seed=args.seed):
            paired_targets.append({"a": f"{a[0]}:{a[1]}", "b": f"{b[0]}:{b[1]}", **record})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for file_name, records in (("rows.csv", rows), ("paired_internal.csv", paired_internal), ("paired_targets.csv", paired_targets), ("tuning.csv", tuning)):
        columns = sorted({k for r in records for k in r}, key=lambda c: (c not in ("registry", "variant", "a", "b", "target"), c))
        with open(out / file_name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    saved = {name: {"primaries": [p.program_id for p in comp.primaries],
                    "surrogates": {pid: [q.program_id for q in qs] for pid, qs in comp.surrogates.items()}} for name, comp in registries.items()}
    (out / "registries.json").write_text(json.dumps(saved, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"pools": args.pools, "args": vars(args), "n_candidates": len(candidates), "meta": meta,
                                                   "elapsed_seconds": time.perf_counter() - started}, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
