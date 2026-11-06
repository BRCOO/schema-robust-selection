#!/usr/bin/env python3
"""SRS-AP, the revision pre-specified in docs/reviews/v2_draft_review_2026-11-03.md section 6.

SRS-AP differs from SRS only in the shortlist decision (``schema_robust_select(decision="auprc")``).
Three phases, run in order:

* ``--phase select``: per run, SRS+T-AP (known target schemas) and SRS-AP (source schema only),
  each for q = 0 and q = 0.25, q picked on VALIDATION AUPRC as in SRS; records the validation
  AUPRC of the shipped probe for SRS-AP and for the original SRS. Nothing on TEST or a target.
* ``--phase decide``: the pre-specified rule over the 16 ICU runs and both configurations:
  adopt SRS-AP iff mean(VAL AUPRC of SRS-AP - VAL AUPRC of SRS) > 0. Writes ``decision.json``.
* ``--phase evaluate`` (needs ``decision.json``): internal TEST and every target, every registry
  deployed with surrogates; paired patient-clustered bootstrap of SRS+T-AP and SRS-AP against
  every saved registry (E3, add-on, fairness re-runs incl. NMIFS) and against the original SRS.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_union_ap.py --phase select
"""

from __future__ import annotations

import argparse
import csv
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
from src.icu.v2.selection_baselines import nmifs  # noqa: E402
from src.icu.v2.stats import metric_value, paired_cluster_bootstrap_diff  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, refire_composite, surrogate_chains  # noqa: E402

OUT = ROOT / "artifacts/v2/union_ap"
#: configuration -> (uses known targets, the original SRS registry it revises)
CONFIGS = {"srs_ap_t": (True, "srs_c_val"), "srs_ap": (False, "srs_c_unknown_val")}


def jobs(args):
    runs = [tuple(x.split(":")) for x in args.runs.split(",") if x] or list(ctx.ICU_RUNS)
    out = [(f"{root}/{label}", *ctx.run_dirs(root, label), True) for root, label in runs]
    out += [(f"drive_k{k}/f_backblaze_seagate_gap2d", *ctx.drive_dirs(int(k)), False) for k in args.drive_k.split(",") if k]
    return out


def saved(comp: CompositeRegistry) -> dict:
    return {"primaries": [p.program_id for p in comp.primaries], "surrogates": {pid: [q.program_id for q in qs] for pid, qs in comp.surrogates.items()}}


def phase_select(args) -> None:
    for label, e3_dir, extra_dir, icu in jobs(args):
        out = OUT / label
        if (out / "select.json").exists():
            print(f"[ap] {label}: selected, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(e3_dir, extra_dir)
        known_names = json.loads((e3_dir / "manifest.json").read_text(encoding="utf-8"))["args"]["known_targets"].split(",")
        known = {t.name: t.features for t in run.targets if t.name in known_names}
        X = run.matrix(run.candidates, sorted(run.train_ids))
        chains = surrogate_chains(run.candidates, X, run.max_surrogates)
        record, registries = {"run": label, "icu": icu}, {}
        for config, (uses_known, original) in CONFIGS.items():
            val = {}
            for q in (0.0, 0.25):
                result = schema_robust_select(run.inp, run.k, scenario_set(run.source, known if uses_known else {}, dropout_q=q), run.subject_of,
                                              surrogates=True, max_surrogates=run.max_surrogates, shortlist=10, chains=chains, decision="auprc")
                comp = CompositeRegistry(tuple(result.registry), result.surrogates)
                val[q] = (run.fit(list(comp.primaries))["validation_score"], comp, result.stop_reason)
            q = 0.25 if val[0.25][0] > val[0.0][0] + 1e-12 else 0.0
            registries[config] = val[q][1]
            record[config] = {"q": q, "val_auprc": val[q][0], "val_q0": val[0.0][0], "val_q25": val[0.25][0], "stop": val[q][2],
                              "n_programs": len(val[q][1].primaries),
                              "original": original, "val_original": run.fit(list(run.registries[original].primaries))["validation_score"]}
        out.mkdir(parents=True, exist_ok=True)
        (out / "registries.json").write_text(json.dumps({k: saved(v) for k, v in registries.items()}, indent=1), encoding="utf-8")
        record["seconds"] = time.perf_counter() - t0
        (out / "select.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[ap] {label} ({record['seconds']:.0f}s) " + " ".join(
            f"{c}: VAL {record[c]['val_auprc']:.4f} vs original {record[c]['val_original']:.4f} (q={record[c]['q']:g})" for c in CONFIGS), flush=True)


def phase_decide(args) -> None:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.rglob("select.json"))]
    paper_runs = {f"{root}/{label}" for root, label in ctx.ICU_RUNS}
    icu = [r for r in records if r["icu"] and r["run"] in paper_runs]  # the 16 runs of the rule, not re-splits
    diffs = [r[c]["val_auprc"] - r[c]["val_original"] for r in icu for c in CONFIGS]
    decision = {"rule": "adopt SRS-AP iff mean over the 16 ICU runs and both configurations of VAL AUPRC(SRS-AP) - VAL AUPRC(SRS) > 0",
                "n_runs": len(icu), "n_differences": len(diffs), "mean_difference": float(np.mean(diffs)) if diffs else None,
                "per_config": {c: float(np.mean([r[c]["val_auprc"] - r[c]["val_original"] for r in icu])) for c in CONFIGS} if icu else {},
                "adopt": bool(diffs) and len(icu) == len(ctx.ICU_RUNS) and float(np.mean(diffs)) > 0}
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


def deploy(run, comp):
    bridge = run.fit(list(comp.primaries))
    p_test = np.array([bridge["test_probabilities"][e] for e in run.ids["test"]])
    per_target = {}
    for target in run.targets:
        eids = [e.episode_id for e in target.episodes]
        out = refire_composite(bridge["model_bundle"], comp, target.features, run.external[target.name], eids)
        per_target[target.name] = np.array([out["probabilities"][e] for e in eids])
    return p_test, per_target


def phase_evaluate(args) -> None:
    if not (OUT / "decision.json").exists():
        raise SystemExit("run --phase decide first: TEST is evaluated only after the decision is recorded")
    sb = ctx.selection_baselines_module()
    for label, e3_dir, extra_dir, icu in jobs(args):
        out = OUT / label
        if (out / "rows.csv").exists():
            print(f"[ap] {label}: evaluated, skipped", flush=True)
            continue
        run = ctx.load_run(e3_dir, extra_dir)
        regs = dict(run.registries)
        ap = json.loads((out / "registries.json").read_text(encoding="utf-8"))
        for name, reg in ap.items():
            regs[name] = ctx._composite(reg, run.by_id)
        fair = ROOT / "artifacts/v2/union_fairness" / label / "registries.json"
        if fair.exists():
            for name, reg in json.loads(fair.read_text(encoding="utf-8")).items():
                regs.setdefault(name, ctx._composite(reg, run.by_id))
        if "nmifs" not in regs:
            regs["nmifs"] = run.composite(nmifs(run.inp, run.k))
        y_test = np.asarray(run.labels["test"])
        g_test = np.array([e.subject_id for e in run.grouped["test"]])
        rows, probs = [], {}
        for name, comp in regs.items():
            p_test, per_target = deploy(run, comp)
            probs[name] = (p_test, per_target)
            target_auprc = {t.name: metric_value([e.label for e in t.episodes], per_target[t.name], "auprc") for t in run.targets}
            rows.append({"run": label, "registry": name, "internal_auprc": metric_value(y_test, p_test, "auprc"),
                         "mean_target_auprc": float(np.mean(list(target_auprc.values()))), **{f"{t}_auprc": v for t, v in target_auprc.items()}})
        paired = []
        for a in CONFIGS:
            for b in regs:
                if b in CONFIGS:
                    continue
                d = paired_cluster_bootstrap_diff(y_test, probs[a][0], probs[b][0], g_test, metric="auprc", n_boot=args.n_boot, seed=args.seed)
                paired.append({"run": label, "a": a, "b": b, **d, "metric": "internal"})
                for record in sb.paired_target_bootstrap(run.targets, probs[a][1], probs[b][1], n_boot=args.n_boot, seed=args.seed):
                    if record["target"] == "mean_over_targets":
                        paired.append({"run": label, "a": a, "b": b, **record, "metric": "mean_target"})
        ctx.write_csv(out / "rows.csv", rows)
        ctx.write_csv(out / "paired.csv", paired)
        get = {r["registry"]: (r["internal_auprc"], r["mean_target_auprc"]) for r in rows}
        print(f"[ap] {label} evaluated: " + " ".join(f"{n}={get[n][0]:.3f}/{get[n][1]:.3f}" for n in ("srs_ap_t", "srs_c_val", "srs_ap", "srs_c_unknown_val", "nmifs", "mrmr") if n in get), flush=True)


def phase_ablate(args) -> None:
    """Ablations of the adopted SRS-AP: ``srs_ap_unaware`` (source schema only, q = 0: the
    schema-unaware wrapper) and ``srs_ap_t_nosur`` (known targets, no surrogates in the
    objective, chains attached afterwards; q on VAL). Paired per target against SRS+T-AP."""
    sb = ctx.selection_baselines_module()
    for label, e3_dir, extra_dir, icu in jobs(args):
        out = OUT / label
        if (out / "ablation_paired.csv").exists():
            print(f"[ap] {label}: ablations done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(e3_dir, extra_dir)
        known_names = json.loads((e3_dir / "manifest.json").read_text(encoding="utf-8"))["args"]["known_targets"].split(",")
        known = {t.name: t.features for t in run.targets if t.name in known_names}
        X = run.matrix(run.candidates, sorted(run.train_ids))
        chains = surrogate_chains(run.candidates, X, run.max_surrogates)
        result = schema_robust_select(run.inp, run.k, scenario_set(run.source, {}, dropout_q=0.0), run.subject_of, surrogates=True,
                                      max_surrogates=run.max_surrogates, shortlist=10, chains=chains, decision="auprc")
        ablations = {"srs_ap_unaware": CompositeRegistry(tuple(result.registry), result.surrogates)}
        best = None
        for q in (0.0, 0.25):
            result = schema_robust_select(run.inp, run.k, scenario_set(run.source, known, dropout_q=q), run.subject_of, surrogates=False,
                                          max_surrogates=run.max_surrogates, shortlist=10, decision="auprc")
            val = run.fit(list(result.registry))["validation_score"]
            if best is None or val > best[0] + 1e-12:
                best = (val, result.registry)
        ablations["srs_ap_t_nosur"] = run.composite(best[1])
        (out / "ablations.json").write_text(json.dumps({k: saved(v) for k, v in ablations.items()}, indent=1), encoding="utf-8")
        ap = {name: ctx._composite(reg, run.by_id) for name, reg in json.loads((out / "registries.json").read_text(encoding="utf-8")).items()}
        regs = {**ap, **ablations}
        y_test = np.asarray(run.labels["test"])
        g_test = np.array([e.subject_id for e in run.grouped["test"]])
        probs = {name: deploy(run, comp) for name, comp in regs.items()}
        paired = []
        for a, b in (("srs_ap_t", "srs_ap_unaware"), ("srs_ap_t", "srs_ap_t_nosur"), ("srs_ap", "srs_ap_unaware")):
            d = paired_cluster_bootstrap_diff(y_test, probs[a][0], probs[b][0], g_test, metric="auprc", n_boot=args.n_boot, seed=args.seed)
            paired.append({"run": label, "a": a, "b": b, "target": "internal", **d, "metric": "internal"})
            for record in sb.paired_target_bootstrap(run.targets, probs[a][1], probs[b][1], n_boot=args.n_boot, seed=args.seed):
                paired.append({"run": label, "a": a, "b": b, **record, "metric": "target"})
        exec_share = {}
        for target in run.targets:
            prim = ablations["srs_ap_unaware"].primaries
            exec_share[target.name] = float(np.mean([frozenset(p.features) <= frozenset(target.features) for p in prim]))
        ctx.write_csv(out / "ablation_paired.csv", paired)
        (out / "ablation_manifest.json").write_text(json.dumps({"executable_share_unaware": exec_share, "seconds": time.perf_counter() - t0}, indent=2), encoding="utf-8")
        print(f"[ap] {label} ablations ({time.perf_counter() - t0:.0f}s)", flush=True)


def main(argv: list[str] | None = None) -> int:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("select", "decide", "evaluate", "ablate"), required=True)
    parser.add_argument("--runs", default="")
    parser.add_argument("--drive-k", default="5,10,20")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default=str(OUT), help="output root (default artifacts/v2/union_ap)")
    args = parser.parse_args(argv)
    OUT = Path(args.out_root)
    {"select": phase_select, "decide": phase_decide, "evaluate": phase_evaluate, "ablate": phase_ablate}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
