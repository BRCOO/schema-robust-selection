#!/usr/bin/env python3
"""Decision diagnostics for SRS+T (review of the method section, 2026-11-05).

* ``--phase trace``: re-runs SRS+T on every ICU run at the dropout rate VAL chose (artifacts/v2/union_ap/<run>/select.json) with
  a per-step trace, checks that the registry equals the shipped one, and records per step whether the AUPRC decision departed
  from the likelihood order, the chosen candidate's own likelihood gain, and whether the likelihood estimate fell.
* ``--phase shortlist``: SRS+T with shortlists m = 5 and m = 20 (dropout rate chosen on VAL as in run_union_ap.py) at K = 10 on
  the four standard settings; internal TEST and every target, deployed with surrogate chains.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_decision_diagnostics.py --phase trace
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("e3_context", ROOT / "scripts/v2/e3_context.py")
ctx = importlib.util.module_from_spec(spec)
sys.modules["e3_context"] = ctx
spec.loader.exec_module(ctx)
spec = importlib.util.spec_from_file_location("run_reference_registries", ROOT / "scripts/v2/run_reference_registries.py")
refs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refs)

from src.icu.v2.robust_selection import scenario_set, schema_robust_select  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, surrogate_chains  # noqa: E402

OUT = ROOT / "artifacts/v2/decision_diagnostics"


def phase_trace() -> None:
    out = OUT / "trace.csv"
    rows = []
    for root, label in ctx.ICU_RUNS:
        run_id = f"{root}/{label}"
        t0 = time.perf_counter()
        e3_dir, extra_dir = ctx.run_dirs(root, label)
        run = ctx.load_run(e3_dir, extra_dir)
        known = refs.known_schemas(run, e3_dir)
        sel = json.loads((ROOT / "artifacts/v2/union_ap" / run_id / "select.json").read_text(encoding="utf-8"))
        shipped = json.loads((ROOT / "artifacts/v2/union_ap" / run_id / "registries.json").read_text(encoding="utf-8"))["srs_ap_t"]["primaries"]
        q = float(sel["srs_ap_t"]["q"])
        X = run.matrix(run.candidates, sorted(run.train_ids))
        chains = surrogate_chains(run.candidates, X, run.max_surrogates)
        trace = []
        res = schema_robust_select(run.inp, run.k, scenario_set(run.source, known, dropout_q=q), run.subject_of, surrogates=True,
                                   max_surrogates=run.max_surrogates, shortlist=10, chains=chains, decision="auprc", trace=trace)
        same = [p.program_id for p in res.registry] == list(shipped)
        for t in trace:
            rows.append({"run": run_id, "q": q, "reproduced": same, **t})
        print(f"[trace] {run_id} q={q:g} reproduced={same} steps={len(trace)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ctx.write_csv(out, rows)


def phase_shortlist() -> None:
    out = OUT / "shortlist.csv"
    rows = []
    for label in ctx.STANDARD:
        root = "union"
        run_id = f"{root}/{label}"
        t0 = time.perf_counter()
        e3_dir, extra_dir = ctx.run_dirs(root, label)
        run = ctx.load_run(e3_dir, extra_dir)
        known = refs.known_schemas(run, e3_dir)
        X = run.matrix(run.candidates, sorted(run.train_ids))
        chains = surrogate_chains(run.candidates, X, run.max_surrogates)
        for m in (5, 20):
            val = {}
            for q in (0.0, 0.25):
                res = schema_robust_select(run.inp, run.k, scenario_set(run.source, known, dropout_q=q), run.subject_of, surrogates=True,
                                           max_surrogates=run.max_surrogates, shortlist=m, chains=chains, decision="auprc")
                comp = CompositeRegistry(tuple(res.registry), res.surrogates)
                val[q] = (run.fit(list(comp.primaries))["validation_score"], comp)
            q = 0.25 if val[0.25][0] > val[0.0][0] + 1e-12 else 0.0
            row, _ = refs.rows_of(run, run_id, f"srs_ap_t_m{m}", val[q][1], {"m": m, "q": q, "val_auprc": val[q][0]})
            rows.append(row)
            print(f"[shortlist] {run_id} m={m} q={q:g}: VAL {val[q][0]:.4f}, internal {row['internal_auprc']:.3f}, targets {row['mean_target_auprc']:.3f} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ctx.write_csv(out, rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("trace", "shortlist"), required=True)
    args = parser.parse_args(argv)
    {"trace": phase_trace, "shortlist": phase_shortlist}[args.phase]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
