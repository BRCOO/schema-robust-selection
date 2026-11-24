#!/usr/bin/env python3
"""Schema-loss stress curves for any saved registries of the 16 ICU runs (SRS-AP and NMIFS included).

The draws are exactly those of ``run_union_extra.py``: for family ``feature`` every source feature,
for ``group`` every coarse group, is dropped independently with probability l, with the generator
seeded by (seed, family index, level index, draw). Every registry is deployed with surrogates
(frozen probe, ``surrogates.refire_composite``) on the internal TEST split.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_stress.py
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

from src.icu.v2.stats import metric_value  # noqa: E402
from src.icu.v2.surrogates import refire_composite  # noqa: E402
from src.icu.v2.types import coarse_group  # noqa: E402

REGISTRIES = "srs_ap_t,srs_ap,srs_c_val,srs_c_unknown_val,nmifs,mrmr,mrmr_filter_known,l1_dropout,fs_cv,cmim"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="")
    parser.add_argument("--registries", default=REGISTRIES)
    parser.add_argument("--levels", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default=str(ROOT / "artifacts/v2/stress_ap"))
    args = parser.parse_args(argv)
    runs = [tuple(x.split(":")) for x in args.runs.split(",") if x] or list(ctx.ICU_RUNS)
    levels = [float(x) for x in args.levels.split(",") if x]
    for root, label in runs:
        out = Path(args.out_root) / root / label
        if (out / "stress.csv").exists():
            print(f"[stress] {root}/{label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(*ctx.run_dirs(root, label))
        names = [n for n in args.registries.split(",") if n in run.registries]
        test_ids = run.ids["test"]
        test_set = frozenset(test_ids)
        y = np.asarray(run.labels["test"])
        bundles = {n: run.fit(list(run.registries[n].primaries))["model_bundle"] for n in names}
        used = {q.program_id for n in names for p in run.registries[n].primaries for q in run.registries[n].branches(p)}
        test_firings = {pid: run.firings.get(pid, frozenset()) & test_set for pid in used}
        groups = sorted({coarse_group(f) for f in run.source})
        records = []
        for fi, family in enumerate(("feature", "group")):
            for li, level in enumerate(levels):
                for draw in range(args.draws):
                    rng = np.random.default_rng([args.seed, fi, li, draw])
                    if family == "feature":
                        kept = frozenset(f for f in sorted(run.source) if rng.random() >= level)
                    else:
                        kept_groups = {g for g in groups if rng.random() >= level}
                        kept = frozenset(f for f in run.source if coarse_group(f) in kept_groups)
                    for n in names:
                        res = refire_composite(bundles[n], run.registries[n], kept, test_firings, test_ids)
                        p = np.array([res["probabilities"][e] for e in test_ids])
                        records.append({"run": f"{root}/{label}", "family": family, "level": level, "draw": draw, "registry": n,
                                        "variant": "composite", "n_kept_features": len(kept), "auprc": metric_value(y, p, "auprc")})
        ctx.write_csv(out / "stress.csv", records, first=("run", "family", "level", "draw", "registry"))
        (out / "manifest.json").write_text(json.dumps({"args": vars(args), "registries": names, "seconds": time.perf_counter() - t0}, indent=2), encoding="utf-8")
        print(f"[stress] {root}/{label} ({time.perf_counter() - t0:.0f}s) {len(names)} registries", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
