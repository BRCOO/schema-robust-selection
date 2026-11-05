#!/usr/bin/env python3
"""Reviewer items H and I: SRS-AP and the strongest baselines on restricted candidate pools.

* ``--restrict care`` (item H): drop every program that uses a care-process feature
  (ventilation, sedation or vasopressor status); ``care_obs`` also drops programs with the
  measurement-process tokens ``:missing`` and ``:sparse_measurement``. Tests whether the edge of
  SRS rests on care-process shortcuts.
* ``--restrict generator:<name>`` (item I, on a large-pool run): keep one generator's candidates
  only (about 1,000 for the SPMF miners, i.e. more than the standard union pool of about 510).
  Tests whether the union's advantage is just pool size; compare with the standard union run.

Selection is TRAIN-only as everywhere: SRS+T-AP and SRS-AP (q on VAL), mRMR, NMIFS, MMRFS (delta on
VAL), L1 and L1 + schema dropout (q on VAL). Every registry is deployed with surrogates chosen in
the restricted pool; paired patient-clustered bootstrap of SRS+T-AP against each.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_restricted_pools.py --restrict care
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _context():
    name = "e3_context"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "v2" / "e3_context.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


ctx = _context()

from src.icu.v2.robust_selection import scenario_set, schema_robust_select  # noqa: E402
from src.icu.v2.selection_baselines import BASELINES, SelectionInput, l1_schema_dropout, nmifs, run_baseline  # noqa: E402
from src.icu.v2.stats import metric_value, paired_cluster_bootstrap_diff  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, composite_registry, refire_composite, surrogate_chains  # noqa: E402

CARE = frozenset({"ventilation_on", "sedation_on", "vasopressor_on"})


def keep(program, restrict: str) -> bool:
    if restrict.startswith("generator:"):
        return program.method == restrict.split(":", 1)[1]
    if frozenset(program.features) & CARE:
        return False
    if restrict == "care_obs":
        tokens = [t for itemset in (*program.elements, *program.consequent) for t in itemset]
        return not any(t.endswith(":missing") or t.endswith(":sparse_measurement") for t in tokens)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--restrict", required=True)
    parser.add_argument("--runs", default="union:f_mimic_iv_gap1h,union:f_mimic_iv_gap2h,union:f_mimic_iv_gap4h,union:f_mimic_iv_aki_gap6h")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default=str(ROOT / "artifacts/v2/restricted_pools"))
    args = parser.parse_args(argv)
    sb = ctx.selection_baselines_module()
    for spec in [x for x in args.runs.split(",") if x]:
        root, label = spec.split(":")
        out = Path(args.out_root) / args.restrict.replace(":", "_") / root / label
        if (out / "manifest.json").exists():
            print(f"[restrict] {spec}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        e3_dir, extra_dir = ctx.run_dirs(root, label)
        run = ctx.load_run(e3_dir, extra_dir)
        known_names = json.loads((e3_dir / "manifest.json").read_text(encoding="utf-8"))["args"]["known_targets"].split(",")
        known = {t.name: t.features for t in run.targets if t.name in known_names}
        cands = [c for c in run.candidates if keep(c, args.restrict)]
        train_ids = sorted(run.train_ids)
        tf = {c.program_id: run.train_firings[c.program_id] for c in cands}
        inp = SelectionInput(tuple(cands), tf, run.train_ids, run.train_pos)
        X = run.matrix(cands, train_ids)
        chains = surrogate_chains(cands, X, run.max_surrogates)

        def composite(registry):
            return composite_registry(registry, cands, tf, train_ids, run.max_surrogates)

        registries = {}
        for name, uses_known in (("srs_ap_t", True), ("srs_ap", False)):
            best = None
            for q in (0.0, 0.25):
                res = schema_robust_select(inp, run.k, scenario_set(run.source, known if uses_known else {}, dropout_q=q), run.subject_of,
                                           surrogates=True, max_surrogates=run.max_surrogates, shortlist=10, chains=chains, decision="auprc")
                comp = CompositeRegistry(tuple(res.registry), res.surrogates)
                val = run.fit(list(comp.primaries))["validation_score"]
                if best is None or val > best[0] + 1e-12:
                    best = (val, comp)
            registries[name] = best[1]
        registries["mrmr"] = composite(run_baseline("mrmr", inp, run.k))
        registries["nmifs"] = composite(nmifs(inp, run.k))
        registries["l1_logistic"] = composite(run_baseline("l1_logistic", inp, run.k))
        for name, param_grid in (("mmrfs", BASELINES["mmrfs"][2]),):
            best = None
            for value in param_grid:
                reg = run_baseline(name, inp, run.k, value)
                val = run.fit(reg)["validation_score"] if reg else -np.inf
                if best is None or val > best[0] + 1e-12:
                    best = (val, reg)
            registries[name] = composite(best[1])
        best = None
        for q in (0.25, 0.5):
            reg = l1_schema_dropout(inp, run.k, q=q, source_features=run.source, seed=args.seed)
            val = run.fit(reg)["validation_score"] if reg else -np.inf
            if best is None or val > best[0] + 1e-12:
                best = (val, reg)
        registries["l1_dropout"] = composite(best[1])

        y_test = np.asarray(run.labels["test"])
        g_test = np.array([e.subject_id for e in run.grouped["test"]])
        rows, probs = [], {}
        for name, comp in registries.items():
            bridge = run.fit(list(comp.primaries))
            p_test = np.array([bridge["test_probabilities"][e] for e in run.ids["test"]])
            per_target = {}
            for target in run.targets:
                eids = [e.episode_id for e in target.episodes]
                res = refire_composite(bridge["model_bundle"], comp, target.features, run.external[target.name], eids)
                per_target[target.name] = np.array([res["probabilities"][e] for e in eids])
            probs[name] = (p_test, per_target)
            ta = {t.name: metric_value([e.label for e in t.episodes], per_target[t.name], "auprc") for t in run.targets}
            rows.append({"run": spec, "restrict": args.restrict, "registry": name, "n_candidates": len(cands),
                         "internal_auprc": metric_value(y_test, p_test, "auprc"), "mean_target_auprc": float(np.mean(list(ta.values()))),
                         "test_ids": "", **{f"{t}_auprc": v for t, v in ta.items()}})
        paired = []
        for b in registries:
            if b == "srs_ap_t":
                continue
            d = paired_cluster_bootstrap_diff(y_test, probs["srs_ap_t"][0], probs[b][0], g_test, metric="auprc", n_boot=args.n_boot, seed=args.seed)
            paired.append({"run": spec, "a": "srs_ap_t", "b": b, **d, "metric": "internal"})
            for record in sb.paired_target_bootstrap(run.targets, probs["srs_ap_t"][1], probs[b][1], n_boot=args.n_boot, seed=args.seed):
                if record["target"] == "mean_over_targets":
                    paired.append({"run": spec, "a": "srs_ap_t", "b": b, **record, "metric": "mean_target"})
        ctx.write_csv(out / "rows.csv", rows)
        ctx.write_csv(out / "paired.csv", paired)
        np.savez_compressed(out / "probabilities.npz", test_ids=np.array(run.ids["test"]),
                            **{f"{name}__internal": p[0] for name, p in probs.items()},
                            **{f"{name}__{t}": v for name, p in probs.items() for t, v in p[1].items()})
        (out / "registries.json").write_text(json.dumps({n: {"primaries": [p.program_id for p in c.primaries],
                                                             "surrogates": {pid: [q.program_id for q in qs] for pid, qs in c.surrogates.items()}}
                                                         for n, c in registries.items()}, indent=1), encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps({"args": vars(args), "n_candidates": len(cands), "n_union": len(run.candidates),
                                                       "seconds": time.perf_counter() - t0}, indent=2), encoding="utf-8")
        got = {r["registry"]: (r["internal_auprc"], r["mean_target_auprc"]) for r in rows}
        print(f"[restrict] {args.restrict} {spec}: {len(cands)}/{len(run.candidates)} candidates ({time.perf_counter() - t0:.0f}s) "
              + " ".join(f"{n}={v[0]:.3f}/{v[1]:.3f}" for n, v in got.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
