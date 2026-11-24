#!/usr/bin/env python3
"""Reference registries and seed replicates for the v3 experiments (review of 2026-11-05, items M3, M11, M12).

Two phases, both reading the saved E3 runs (``e3_context``) and deploying every registry with surrogate chains, exactly as
``run_union_ap.py --phase evaluate``:

* ``--phase refs`` (all 16 ICU runs and the drive runs): reference registries
  - ``filter_nmifs``: NMIFS on the candidates executable at every known target (the target-aware version of the strongest
    filter, built like Filter+mRMR in ``run_union_selection.py``);
  - ``all_programs``: every candidate of the pool as a primary (one probe on the whole pool, C on VAL);
  - ``random_k``: ``--n-random`` registries of K candidates drawn uniformly (seeded); per-draw rows.
  Paired patient-clustered bootstrap (1,000) of SRS+T and SRS against ``filter_nmifs``: internal, mean target and per target.
* ``--phase seeds`` (the four K = 10 standard runs): NMIFS+AP and Wrapper, AUPRC re-selected with CV seeds 1-4, as the
  SRS+T seeds of ``run_srs_sensitivity.py`` (the seed moves the patient-grouped CV folds; q on VAL for NMIFS+AP,
  q = 0 for the wrapper). Rows only.

Writes artifacts/v2/references/<root>/<label>/{registries.json, rows.csv, paired.csv} and
artifacts/v2/seed_replicates/<label>/rows.csv.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_reference_registries.py --phase refs
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
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("e3_context", ROOT / "scripts/v2/e3_context.py")
ctx = importlib.util.module_from_spec(spec)
sys.modules["e3_context"] = ctx
spec.loader.exec_module(ctx)
spec = importlib.util.spec_from_file_location("run_union_ap", ROOT / "scripts/v2/run_union_ap.py")
ap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ap)

from src.icu.v2.robust_selection import scenario_set, schema_robust_select  # noqa: E402
from src.icu.v2.selection_baselines import SelectionInput, nmifs  # noqa: E402
from src.icu.v2.stats import metric_value, paired_cluster_bootstrap_diff  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, surrogate_chains  # noqa: E402

OUT_REFS = ROOT / "artifacts/v2/references"
OUT_SEEDS = ROOT / "artifacts/v2/seed_replicates"


def only_runs(args, root, labels):
    """The labels of ``root`` named in ``--runs`` (root:label pairs, any label), or ``labels`` when ``--runs`` is empty."""
    if not args.runs:
        return list(labels)
    return [label for r, label in (x.split(":") for x in args.runs.split(",") if x) if r == root]


def known_schemas(run, e3_dir):
    names = json.loads((Path(e3_dir) / "manifest.json").read_text(encoding="utf-8"))["args"]["known_targets"].split(",")
    return {t.name: t.features for t in run.targets if t.name in names}


def rows_of(run, label, name, comp, extra=None):
    p_test, per_target = ap.deploy(run, comp)
    y_test = np.asarray(run.labels["test"])
    tgt = {t.name: metric_value([e.label for e in t.episodes], per_target[t.name], "auprc") for t in run.targets}
    row = {"run": label, "registry": name, "n_programs": len(comp.primaries), "internal_auprc": metric_value(y_test, p_test, "auprc"),
           "mean_target_auprc": float(np.mean(list(tgt.values()))), **{f"{t}_auprc": v for t, v in tgt.items()}, **(extra or {})}
    return row, (p_test, per_target)


def phase_refs(args) -> None:
    sb = ctx.selection_baselines_module()
    for label, e3_dir, extra_dir, icu in ap.jobs(args):
        out = OUT_REFS / label
        if (out / "paired.csv").exists():
            print(f"[refs] {label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(e3_dir, extra_dir)
        known = known_schemas(run, e3_dir)
        regs = {}
        known_exec = [c for c in run.candidates if all(frozenset(c.features) <= f for f in known.values())]
        inp_known = SelectionInput(tuple(known_exec), {c.program_id: run.train_firings[c.program_id] for c in known_exec}, run.train_ids, run.train_pos)
        regs["filter_nmifs"] = run.composite(nmifs(inp_known, run.k))
        regs["all_programs"] = run.composite(list(run.candidates))
        rng = np.random.default_rng(args.seed)
        draws = [run.composite([run.candidates[i] for i in rng.choice(len(run.candidates), size=run.k, replace=False)])
                 for _ in range(args.n_random)]
        rows, probs = [], {}
        for name, comp in regs.items():
            row, probs[name] = rows_of(run, label, name, comp)
            rows.append(row)
        for i, comp in enumerate(draws):
            row, _ = rows_of(run, label, "random_k", comp, {"draw": i})
            rows.append(row)
        for name in ("srs_ap_t", "srs_ap"):
            _, probs[name] = rows_of(run, label, name, run.registries[name])
        y_test = np.asarray(run.labels["test"])
        g_test = np.array([e.subject_id for e in run.grouped["test"]])
        paired = []
        for a in ("srs_ap_t", "srs_ap"):
            b = "filter_nmifs"
            d = paired_cluster_bootstrap_diff(y_test, probs[a][0], probs[b][0], g_test, metric="auprc", n_boot=args.n_boot, seed=0)
            paired.append({"run": label, "a": a, "b": b, "target": "internal", **d, "metric": "internal"})
            for record in sb.paired_target_bootstrap(run.targets, probs[a][1], probs[b][1], n_boot=args.n_boot, seed=0):
                paired.append({"run": label, "a": a, "b": b, **record, "metric": "mean_target" if record["target"] == "mean_over_targets" else "target"})
        out.mkdir(parents=True, exist_ok=True)
        (out / "registries.json").write_text(json.dumps({k: ap.saved(v) for k, v in regs.items()}, indent=1), encoding="utf-8")
        ctx.write_csv(out / "rows.csv", rows)
        ctx.write_csv(out / "paired.csv", paired)
        rk = [r for r in rows if r["registry"] == "random_k"]
        print(f"[refs] {label} ({time.perf_counter() - t0:.0f}s): filter_nmifs={rows[0]['internal_auprc']:.3f}/{rows[0]['mean_target_auprc']:.3f} "
              f"all={rows[1]['internal_auprc']:.3f}/{rows[1]['mean_target_auprc']:.3f} random={np.mean([r['internal_auprc'] for r in rk]):.3f}/"
              f"{np.mean([r['mean_target_auprc'] for r in rk]):.3f}", flush=True)


def phase_seeds(args) -> None:
    root = args.root
    base = Path(args.seeds_out) if args.seeds_out else OUT_SEEDS
    out_root = base if root == "union" else base / root
    labels = only_runs(args, root, ctx.LARGE if root.startswith("union_large") else ctx.STANDARD)
    names = [x for x in args.configs.split(",") if x] if args.configs else (["srs_ap_t"] if args.with_srs else []) + ["nmifs_ap_t", "srs_ap_unaware"]
    fname = f"rows_{args.tag}.csv" if args.tag else "rows.csv"
    for label in labels:
        out = out_root / label
        if (out / fname).exists():
            print(f"[seeds] {root}/{label} {fname}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        e3_dir, extra_dir = ctx.run_dirs(root, label)
        run = ctx.load_run(e3_dir, extra_dir)
        known = known_schemas(run, e3_dir)
        X = run.matrix(run.candidates, sorted(run.train_ids))
        chains = surrogate_chains(run.candidates, X, run.max_surrogates)
        rows = []
        # name: (target schemas, screener, dropout rates, decision, surrogates in the objective)
        known_or_none = {"srs_ap_t": (known, "newton", (0.0, 0.25), "auprc", True), "srs_ap": ({}, "newton", (0.0, 0.25), "auprc", True),
                         "nmifs_ap_t": (known, "nmifs", (0.0, 0.25), "auprc", True), "srs_ap_unaware": ({}, "newton", (0.0,), "auprc", True),
                         # the likelihood-decision versions, as in run_union_selection.py: SRS+T, lik. (srs_c_val, q on VAL) and
                         # Wrapper, lik. (fs_cv: source schema only, no surrogates in the objective, chains built afterwards)
                         "srs_c_val": (known, "newton", (0.0, 0.25), "loglik", True), "fs_cv": ({}, "newton", (0.0,), "loglik", False)}
        configs = [(name, *known_or_none[name]) for name in names]
        saved = {}
        for seed in (1, 2, 3, 4):
            for name, schemas, screener, qs, decision, use_surrogates in configs:
                val = {}
                for q in qs:
                    res = schema_robust_select(run.inp, run.k, scenario_set(run.source, schemas, dropout_q=q), run.subject_of,
                                               surrogates=use_surrogates, max_surrogates=run.max_surrogates, shortlist=10,
                                               chains=chains if use_surrogates else None, decision=decision, screener=screener, seed=seed)
                    comp = CompositeRegistry(tuple(res.registry), res.surrogates) if use_surrogates else run.composite(res.registry)
                    val[q] = (run.fit(list(comp.primaries))["validation_score"], comp)
                q = 0.25 if 0.25 in val and val[0.25][0] > val[0.0][0] + 1e-12 else 0.0
                row, _ = rows_of(run, f"{root}/{label}", name, val[q][1], {"seed": seed, "q": q})
                rows.append(row)
                saved[f"{name}_seed{seed}"] = ap.saved(val[q][1])
            print(f"[seeds] {root}/{label} seed {seed}: " + " ".join(f"{r['registry']}={r['internal_auprc']:.3f}/{r['mean_target_auprc']:.3f}"
                                                                     for r in rows[-len(configs):]), flush=True)
        out.mkdir(parents=True, exist_ok=True)
        ctx.write_csv(out / fname, rows)
        (out / fname.replace("rows", "registries").replace(".csv", ".json")).write_text(json.dumps(saved, indent=1), encoding="utf-8")
        print(f"[seeds] {root}/{label} ({time.perf_counter() - t0:.0f}s)", flush=True)


def phase_compilable(args) -> None:
    """SRS+T restricted to the programs that compile to SQL (FMG's rule-id matched rules removed from the pool and from the
    surrogate chains); q on VAL; paired against the shipped SRS+T."""
    from src.icu.v2.sql_registry import compilable

    sb = ctx.selection_baselines_module()
    root = args.root
    labels = only_runs(args, root, ctx.LARGE if root.startswith("union_large") else ctx.STANDARD)
    for label in labels:
        out = ROOT / "artifacts/v2/compilable_only" / root / label
        if (out / "paired.csv").exists():
            print(f"[compilable] {root}/{label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        e3_dir, extra_dir = ctx.run_dirs(root, label)
        run = ctx.load_run(e3_dir, extra_dir)
        known = known_schemas(run, e3_dir)
        cands = [c for c in run.candidates if compilable(c)]
        inp = SelectionInput(tuple(cands), {c.program_id: run.train_firings[c.program_id] for c in cands}, run.train_ids, run.train_pos)
        X = run.matrix(cands, sorted(run.train_ids))
        chains = surrogate_chains(cands, X, run.max_surrogates)
        val = {}
        for q in (0.0, 0.25):
            res = schema_robust_select(inp, run.k, scenario_set(run.source, known, dropout_q=q), run.subject_of, surrogates=True,
                                       max_surrogates=run.max_surrogates, shortlist=10, chains=chains, decision="auprc")
            comp = CompositeRegistry(tuple(res.registry), res.surrogates)
            val[q] = (run.fit(list(comp.primaries))["validation_score"], comp)
        q = 0.25 if val[0.25][0] > val[0.0][0] + 1e-12 else 0.0
        comp = val[q][1]
        rows, probs = [], {}
        for name, reg in (("srs_ap_t_compilable", comp), ("srs_ap_t", run.registries["srs_ap_t"])):
            row, probs[name] = rows_of(run, f"{root}/{label}", name, reg, {"q": q if name != "srs_ap_t" else None,
                                                                            "n_candidates": len(cands) if name != "srs_ap_t" else len(run.candidates)})
            rows.append(row)
        y_test = np.asarray(run.labels["test"])
        g_test = np.array([e.subject_id for e in run.grouped["test"]])
        a, b = "srs_ap_t", "srs_ap_t_compilable"
        d = paired_cluster_bootstrap_diff(y_test, probs[a][0], probs[b][0], g_test, metric="auprc", n_boot=args.n_boot, seed=0)
        paired = [{"run": f"{root}/{label}", "a": a, "b": b, "target": "internal", **d, "metric": "internal"}]
        for record in sb.paired_target_bootstrap(run.targets, probs[a][1], probs[b][1], n_boot=args.n_boot, seed=0):
            paired.append({"run": f"{root}/{label}", "a": a, "b": b, **record, "metric": "mean_target" if record["target"] == "mean_over_targets" else "target"})
        out.mkdir(parents=True, exist_ok=True)
        (out / "registries.json").write_text(json.dumps({"srs_ap_t_compilable": ap.saved(comp)}, indent=1), encoding="utf-8")
        ctx.write_csv(out / "rows.csv", rows)
        ctx.write_csv(out / "paired.csv", paired)
        print(f"[compilable] {root}/{label} ({time.perf_counter() - t0:.0f}s): " +
              " ".join(f"{r['registry']}={r['internal_auprc']:.3f}/{r['mean_target_auprc']:.3f}" for r in rows), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("refs", "seeds", "compilable"), required=True)
    parser.add_argument("--runs", default="")
    parser.add_argument("--drive-k", default="5,10,20")
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default="union", help="seeds phase: E3 root (union = K 10, union_k20 = K 20)")
    parser.add_argument("--with-srs", action="store_true", help="seeds phase: also re-select SRS+T")
    parser.add_argument("--configs", default="", help="seeds phase: comma list of srs_ap_t, srs_ap, nmifs_ap_t, srs_ap_unaware")
    parser.add_argument("--tag", default="", help="seeds phase: output rows_<tag>.csv instead of rows.csv")
    parser.add_argument("--seeds-out", default="", help="seeds phase: output root (default artifacts/v2/seed_replicates)")
    args = parser.parse_args(argv)
    {"refs": phase_refs, "seeds": phase_seeds, "compilable": phase_compilable}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
