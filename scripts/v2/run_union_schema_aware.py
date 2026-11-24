#!/usr/bin/env python3
"""E2 on the union of generator pools: the target schema is known before deployment.

For every target t (eICU, HiRID and the five controlled shifts), registries of K programs from
the union of the five generators' pools (``src/icu/v2/union.py``):

* ``<selector>_filter``: the strongest established selectors (mRMR, MMRFS, L1) on the candidates
  executable on t only, hyper-parameters chosen on validation -- a target-specialized registry;
* ``<selector>``: the same selectors on the whole union (schema-unaware), deployed with
  surrogates (taken from ``--e3-dir``/registries.json, i.e. E3's registries);
* ``srs_aware``: SRS with the source schema and t as scenarios, deployed with surrogates;
* ``srs_c_aware``: SRS-C with the same scenarios (selection already counts surrogates);
* ``srs_c_w<w>`` (``--target-weights``): SRS-C with scenario weights 1 - w (source) and w (t), tracing
  the source/target trade-off; ``frontier.csv`` records, for every other registry, whether some
  SRS-C point is at least as good on both the target and the source AUPRC.

Every registry: frozen probe fit on the primaries' source firings (TRAIN, C on validation);
target AUPRC with surrogates (filtered registries are executable on t anyway); source (internal
test) AUPRC. Paired patient-clustered bootstrap of ``srs_c_aware`` against every other
registry on t's episodes.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_union_schema_aware.py \\
      --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl --e3-dir artifacts/v2/union/f_mimic_iv_gap1h \\
      --out-dir artifacts/v2/union_schema_aware/f_mimic_iv_gap1h
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

from src.icu.v2.bridge import fit_frozen_bridge_v2  # noqa: E402
from src.icu.v2.pipeline import episodes_by_split, restrict_firings  # noqa: E402
from src.icu.v2.robust_selection import SchemaScenario, scenario_set, schema_robust_select  # noqa: E402
from src.icu.v2.selection_baselines import BASELINES, SelectionInput, run_baseline  # noqa: E402
from src.icu.v2.stats import metric_value, paired_cluster_bootstrap_diff  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, composite_registry, refire_composite  # noqa: E402
from src.icu.v2.union import union_pool  # noqa: E402

FILTER_SELECTORS = ("mrmr", "mmrfs", "l1_logistic")


def selection_baselines_module():
    spec = importlib.util.spec_from_file_location("run_selection_baselines", ROOT / "scripts" / "v2" / "run_selection_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--e3-dir", required=True, help="run_union_selection.py output of the same run (unaware registries)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-surrogates", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-weights", default="", help="comma list of target weights w for the SRS-C trade-off (e.g. 0.75,0.9,1.0)")
    parser.add_argument("--shortlist", type=int, default=10, help="fast SRS shortlist (0 = exact greedy)")
    args = parser.parse_args(argv)
    target_weights = [float(w) for w in args.target_weights.split(",") if w]
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
    candidates, firings, external = union_pool(state)
    by_id = {c.program_id: c for c in candidates}
    train_firings = {c.program_id: firings[c.program_id] & train_ids for c in candidates}
    e3 = json.loads((Path(args.e3_dir) / "registries.json").read_text(encoding="utf-8"))
    test_y = np.array(labels["test"], dtype=int)

    def fit(registry):
        f = {p.program_id: firings[p.program_id] for p in registry}
        return fit_frozen_bridge_v2(list(registry), {s: restrict_firings(f, frozenset(ids[s])) for s in ids}, ids, labels)

    def tuned(name, pool_cands):
        inp = SelectionInput(tuple(pool_cands), {c.program_id: train_firings[c.program_id] for c in pool_cands}, train_ids, train_pos)
        _, _, grid = BASELINES[name]
        best = None
        for value in grid:
            registry = run_baseline(name, inp, args.k, value)
            score = fit(registry)["validation_score"] if registry else -np.inf
            if best is None or score > best[0] + 1e-12:
                best = (score, value, registry)
        return best[2], best[1]

    unaware = {name: CompositeRegistry(tuple(by_id[i] for i in e3[name]["primaries"]),
                                       {pid: tuple(by_id[q] for q in qs) for pid, qs in e3[name]["surrogates"].items()})
               for name in FILTER_SELECTORS if name in e3}
    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    inp_all = SelectionInput(tuple(candidates), train_firings, train_ids, train_pos)
    for target in state["targets"]:
        t0 = time.perf_counter()
        executable = [c for c in candidates if frozenset(c.features) <= target.features]
        regs: dict[str, CompositeRegistry] = dict(unaware)
        params: dict[str, Any] = {}
        for name in FILTER_SELECTORS:
            registry, params[f"{name}_filter"] = tuned(name, executable) if executable else ([], None)
            regs[f"{name}_filter"] = CompositeRegistry(tuple(registry), {})
        for name, use_surrogates in (("srs_aware", False), ("srs_c_aware", True)):
            result = schema_robust_select(inp_all, args.k, scenario_set(source, {target.name: target.features}), subject_of,
                                          surrogates=use_surrogates, max_surrogates=args.max_surrogates, shortlist=args.shortlist or None)
            regs[name] = (CompositeRegistry(tuple(result.registry), result.surrogates) if use_surrogates
                          else composite_registry(result.registry, candidates, train_firings, sorted(train_ids), args.max_surrogates))
        for w in target_weights:
            scen = [s for s in (SchemaScenario("source", frozenset(source), 1.0 - w), SchemaScenario(target.name, target.features, w)) if s.weight > 0]
            result = schema_robust_select(inp_all, args.k, scen, subject_of, surrogates=True, max_surrogates=args.max_surrogates,
                                          shortlist=args.shortlist or None)
            regs[f"srs_c_w{w:g}"] = CompositeRegistry(tuple(result.registry), result.surrogates)
        eids = [e.episode_id for e in target.episodes]
        y = np.array([e.label for e in target.episodes], dtype=int)
        g = np.array([e.subject_id for e in target.episodes])
        probs = {}
        for name, comp in regs.items():
            if not comp.primaries:
                rows.append({"target": target.name, "registry": name, "n_programs": 0, "error": "no executable candidate"})
                continue
            bridge = fit(list(comp.primaries))
            refired = refire_composite(bridge["model_bundle"], comp, target.features, external[target.name], eids)
            p = np.array([refired["probabilities"][e] for e in eids])
            probs[name] = p
            rows.append({
                "target": target.name, "registry": name, "n_programs": len(comp.primaries),
                "generators": ",".join(sorted({q.method for q in comp.primaries})), "param": params.get(name, ""),
                "target_auprc": metric_value(y, p, "auprc"), "internal_auprc": metric_value(test_y, [bridge["test_probabilities"][e] for e in ids["test"]], "auprc"),
                "target_schema_survival": refired["schema_survival"], "surrogate_fraction": refired["surrogate_fraction"],
                "n_executable_candidates": len(executable),
            })
        for name in regs:
            if name == "srs_c_aware" or name not in probs or "srs_c_aware" not in probs:
                continue
            diff = paired_cluster_bootstrap_diff(y, probs["srs_c_aware"], probs[name], g, metric="auprc", n_boot=args.n_boot, seed=args.seed)
            paired.append({"target": target.name, "comparison": f"srs_c_aware - {name}", **diff})
        print(f"[union-e2] {target.name} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frontier = []
    for target_name in dict.fromkeys(r["target"] for r in rows):
        points = [r for r in rows if r["target"] == target_name and not r.get("error")]
        srs_points = [r for r in points if r["registry"] == "srs_c_aware" or r["registry"].startswith("srs_c_w")]
        for r in points:
            if r in srs_points:
                continue
            dominated_by = [s["registry"] for s in srs_points if s["target_auprc"] >= r["target_auprc"] - 1e-12 and s["internal_auprc"] >= r["internal_auprc"] - 1e-12]
            frontier.append({"target": target_name, "registry": r["registry"], "target_auprc": r["target_auprc"], "internal_auprc": r["internal_auprc"],
                             "dominated": bool(dominated_by), "dominated_by": ",".join(dominated_by)})
    for file_name, records in (("single_target.csv", rows), ("single_target_paired.csv", paired), ("frontier.csv", frontier)):
        columns = sorted({k for r in records for k in r}, key=lambda c: (c not in ("target", "registry", "comparison"), c))
        with open(out / file_name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    (out / "manifest.json").write_text(json.dumps({"pools": args.pools, "args": vars(args), "elapsed_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
