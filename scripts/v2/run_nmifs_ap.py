#!/usr/bin/env python3
"""Hybrid baseline NMIFS+AP (review of the v3 experiments, 2026-11-05): SRS+T's machinery (scenarios with the known
target schemas and dropout, surrogate chains in the objective, shortlist of m = 10 decided by the scenario-weighted
held-out AUPRC, min-gain stop, q in {0, 0.25} chosen on VAL) with the one-step Newton screening replaced by NMIFS's
conditional score. It isolates the screening by estimated deployed value from the AUPRC decision.

Writes artifacts/v2/nmifs_ap/<root>/<label>/{registries.json, select.json, rows.csv, paired.csv}; paired = SRS+T vs
NMIFS+AP and NMIFS+AP vs NMIFS (internal and mean target, patient/drive-clustered paired bootstrap, 1,000 replicates).

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_nmifs_ap.py
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
from src.icu.v2.selection_baselines import nmifs  # noqa: E402
from src.icu.v2.stats import metric_value, paired_cluster_bootstrap_diff  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, surrogate_chains  # noqa: E402

OUT = ROOT / "artifacts/v2/nmifs_ap"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="")
    parser.add_argument("--drive-k", default="5,10,20")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)
    sb = ctx.selection_baselines_module()
    for label, e3_dir, extra_dir, icu in ap.jobs(args):
        out = OUT / label
        if (out / "paired.csv").exists():
            print(f"[nmifs-ap] {label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(e3_dir, extra_dir)
        known_names = json.loads((e3_dir / "manifest.json").read_text(encoding="utf-8"))["args"]["known_targets"].split(",")
        known = {t.name: t.features for t in run.targets if t.name in known_names}
        X = run.matrix(run.candidates, sorted(run.train_ids))
        chains = surrogate_chains(run.candidates, X, run.max_surrogates)
        val = {}
        for q in (0.0, 0.25):
            res = schema_robust_select(run.inp, run.k, scenario_set(run.source, known, dropout_q=q), run.subject_of, surrogates=True,
                                       max_surrogates=run.max_surrogates, shortlist=10, chains=chains, decision="auprc", screener="nmifs")
            comp = CompositeRegistry(tuple(res.registry), res.surrogates)
            val[q] = (run.fit(list(comp.primaries))["validation_score"], comp, res.stop_reason)
        q = 0.25 if val[0.25][0] > val[0.0][0] + 1e-12 else 0.0
        comp = val[q][1]
        out.mkdir(parents=True, exist_ok=True)
        (out / "registries.json").write_text(json.dumps({"nmifs_ap_t": ap.saved(comp)}, indent=1), encoding="utf-8")
        (out / "select.json").write_text(json.dumps({"run": label, "q": q, "val_q0": val[0.0][0], "val_q25": val[0.25][0], "stop": val[q][2],
                                                     "n_programs": len(comp.primaries)}, indent=2), encoding="utf-8")
        nmifs_reg = run.registries["nmifs"] if "nmifs" in run.registries else run.composite(nmifs(run.inp, run.k))  # drives: as run_union_ap
        regs = {"nmifs_ap_t": comp, "srs_ap_t": run.registries["srs_ap_t"], "nmifs": nmifs_reg}
        y_test = np.asarray(run.labels["test"])
        g_test = np.array([e.subject_id for e in run.grouped["test"]])
        probs, rows = {}, []
        for name, reg in regs.items():
            p_test, per_target = ap.deploy(run, reg)
            probs[name] = (p_test, per_target)
            tgt = {t.name: metric_value([e.label for e in t.episodes], per_target[t.name], "auprc") for t in run.targets}
            rows.append({"run": label, "registry": name, "internal_auprc": metric_value(y_test, p_test, "auprc"),
                         "mean_target_auprc": float(np.mean(list(tgt.values()))), **{f"{t}_auprc": v for t, v in tgt.items()}})
        paired = []
        for a, b in (("srs_ap_t", "nmifs_ap_t"), ("nmifs_ap_t", "nmifs"), ("srs_ap_t", "nmifs")):
            d = paired_cluster_bootstrap_diff(y_test, probs[a][0], probs[b][0], g_test, metric="auprc", n_boot=args.n_boot, seed=0)
            paired.append({"run": label, "a": a, "b": b, **d, "metric": "internal"})
            for record in sb.paired_target_bootstrap(run.targets, probs[a][1], probs[b][1], n_boot=args.n_boot, seed=0):
                if record["target"] == "mean_over_targets":
                    paired.append({"run": label, "a": a, "b": b, **record, "metric": "mean_target"})
        ctx.write_csv(out / "rows.csv", rows)
        ctx.write_csv(out / "paired.csv", paired)
        get = {r["registry"]: (r["internal_auprc"], r["mean_target_auprc"]) for r in rows}
        print(f"[nmifs-ap] {label} ({time.perf_counter() - t0:.0f}s) q={q:g}: " +
              " ".join(f"{n}={get[n][0]:.3f}/{get[n][1]:.3f}" for n in regs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
