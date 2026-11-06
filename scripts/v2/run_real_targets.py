#!/usr/bin/env python3
"""Real-target endpoint (review of the v3 experiments, round 2): every registry of Table 2 deployed at the two real target
databases (eICU, HiRID), with paired patient-clustered bootstraps of SRS+T and SRS against each comparator on eICU, HiRID and
their mean; plus each registry's AUPRC lift over prevalence at every target (for retention on lift). Registries are read, never
re-selected; deployment with surrogate chains as in run_union_ap.py --phase evaluate.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_real_targets.py
"""
from __future__ import annotations

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

from src.icu.v2.stats import metric_value  # noqa: E402

OUT = ROOT / "artifacts/v2/real_targets"
NAMES = ("srs_ap_t", "srs_ap", "nmifs_ap_t", "srs_ap_unaware", "fs_cv", "nmifs", "mrmr", "mrmr_stratified", "jmi", "cmim", "mmrfs", "mmr_ig",
         "mms_xin_ig", "dpp_ig", "l1_logistic", "l1_dropout", "mrmr_filter_known", "filter_nmifs")
#: E3 names of the registries whose paper names differ
ALIASES = {"mmr_ig": "mmr", "mms_xin_ig": "mms_xin", "dpp_ig": "dpp"}


def main() -> int:
    sb = ctx.selection_baselines_module()
    for root, label in ctx.ICU_RUNS:
        run_id = f"{root}/{label}"
        out = OUT / run_id
        if (out / "paired.csv").exists():
            print(f"[real] {run_id}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(*ctx.run_dirs(root, label))
        regs = dict(run.registries)
        for extra in ("nmifs_ap", "references"):
            path = ROOT / "artifacts/v2" / extra / run_id / "registries.json"
            for name, reg in json.loads(path.read_text(encoding="utf-8")).items():
                regs.setdefault(name, ctx._composite(reg, run.by_id))
        abl = json.loads((ROOT / "artifacts/v2/union_ap" / run_id / "ablations.json").read_text(encoding="utf-8"))
        regs.setdefault("srs_ap_unaware", ctx._composite(abl["srs_ap_unaware"], run.by_id))
        if "nmifs" not in regs:
            from src.icu.v2.selection_baselines import nmifs
            regs["nmifs"] = run.composite(nmifs(run.inp, run.k))
        real = [t for t in run.targets if t.name in ("eicu", "hirid")]
        prevalence = {t.name: float(np.mean([e.label for e in t.episodes])) for t in run.targets}
        prevalence["internal_test"] = float(np.mean(run.labels["test"]))
        probs, rows = {}, []
        for name in NAMES:
            comp = regs.get(name)
            if comp is None:
                print(f"[real] {run_id}: missing {name}", flush=True)
                continue
            p_test, per_target = ap.deploy(run, comp)
            probs[name] = per_target
            row = {"run": run_id, "registry": name, "internal_lift": metric_value(run.labels["test"], p_test, "auprc") - prevalence["internal_test"]}
            for t in run.targets:
                row[f"{t.name}_lift"] = metric_value([e.label for e in t.episodes], per_target[t.name], "auprc") - prevalence[t.name]
            rows.append(row)
        paired = []
        for a in ("srs_ap_t", "srs_ap"):
            for b in NAMES:
                if b in (a, "srs_ap_t") or b not in probs:
                    continue
                for record in sb.paired_target_bootstrap(real, {k: probs[a][k] for k in ("eicu", "hirid")}, {k: probs[b][k] for k in ("eicu", "hirid")},
                                                        n_boot=1000, seed=0):
                    paired.append({"run": run_id, "a": a, "b": b, **record})
        out.mkdir(parents=True, exist_ok=True)
        ctx.write_csv(out / "lift.csv", rows)
        ctx.write_csv(out / "paired.csv", paired)
        (out / "prevalence.json").write_text(json.dumps(prevalence, indent=1), encoding="utf-8")
        print(f"[real] {run_id} ({time.perf_counter() - t0:.0f}s), {len(probs)} registries", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
