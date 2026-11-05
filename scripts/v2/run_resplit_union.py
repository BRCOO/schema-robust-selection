#!/usr/bin/env python3
"""Robustness to the patient-level split: the union-pool experiment (E3) on new TRAIN/VAL/TEST splits.

For every seed: a new patient-level split of the internal cohort (``split_extracted_cohort.py
--seed``; the original split uses seed 20261021), candidate pools re-mined on the new TRAIN with
the four non-FMG generators (FMG, the weakest generator, needs ~25 GB and hours per split), E3
(``run_union_selection.py``, no single-generator E1 references) and its add-on
(``run_union_extra.py``). Everything else -- extraction, features, external cohorts, generator
settings -- is the original run's (its comparison ``config.json``). Resumable: a step is skipped
when its output exists.

Outputs: ``artifacts/v2/task/r<seed>_mimic_iv_<tag>/``, ``artifacts/v2/comparison/r<seed>_mimic_iv_<tag>/``,
``artifacts/v2/union_resplit/r<seed>_mimic_iv_<tag>/``, ``artifacts/v2/union_resplit_extra/r<seed>_mimic_iv_<tag>/``;
logs in ``artifacts/v2/logs/resplit_r<seed>_<tag>.log``.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_resplit_union.py --tag gap1h --seeds 1,2,3,4,5
  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_resplit_union.py --tag aki_gap6h --seeds 1,2,3,4,5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="")


def sh(args: list[str], log: Path) -> None:
    started = time.perf_counter()
    with open(log, "a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n$ {' '.join(args)}\n")
        handle.flush()
        code = subprocess.run(args, cwd=str(ROOT), env=ENV, stdout=handle, stderr=subprocess.STDOUT).returncode
        handle.write(f"[exit {code} after {time.perf_counter() - started:.0f}s]\n")
    if code != 0:
        raise SystemExit(f"step failed ({code}): {' '.join(args)}; see {log}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="run tag of the original track, e.g. gap1h or aki_gap6h")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--stress-draws", type=int, default=10)
    args = parser.parse_args(argv)
    original = json.loads((ROOT / f"artifacts/v2/comparison/f_mimic_iv_{args.tag}/config.json").read_text(encoding="utf-8"))
    extract_dir = original["episode_dir"]
    (ROOT / "artifacts/v2/logs").mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        name = f"r{seed}_mimic_iv_{args.tag}"
        log = ROOT / f"artifacts/v2/logs/resplit_r{seed}_{args.tag}.log"
        task, comparison = ROOT / f"artifacts/v2/task/{name}", ROOT / f"artifacts/v2/comparison/{name}"
        union, extra = ROOT / f"artifacts/v2/union_resplit/{name}", ROOT / f"artifacts/v2/union_resplit_extra/{name}"
        print(f"[resplit] {name}", flush=True)
        if not (task / "episodes_split.json").exists():
            sh([PY, "scripts/v2/split_extracted_cohort.py", "--extract-dir", extract_dir, "--out-dir", str(task), "--seed", str(seed)], log)
        if not (comparison / "pools.pkl").exists():
            config = dict(original)
            config["_note"] = f"Re-split robustness run (seed {seed}) of f_mimic_iv_{args.tag}: new patient-level split, pools re-mined on its TRAIN without FMG."
            config["cohort"] = str(task / "episodes_split.json")
            config["out_dir"] = str(comparison)
            config["methods"] = {m: v for m, v in original["methods"].items() if m != "fmg_bridge"}
            config["targets"] = {t: {k: v for k, v in spec.items() if k != "fmg_contrastive_inputs"} for t, spec in original["targets"].items()}
            config["reference"] = None
            config["n_boot"] = 200
            path = ROOT / f"artifacts/v2/comparison/configs/{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            sh([PY, "scripts/v2/run_discovery_comparison.py", "--config", str(path), "--save-pools", str(comparison / "pools.pkl")], log)
        if not (union / "manifest.json").exists():
            sh([PY, "scripts/v2/run_union_selection.py", "--pools", str(comparison / "pools.pkl"), "--e1-dir", "", "--k", str(args.k), "--out-dir", str(union)], log)
        if not (extra / "manifest.json").exists():
            sh([PY, "scripts/v2/run_union_extra.py", "--e3-dir", str(union), "--out-dir", str(extra), "--stress-draws", str(args.stress_draws)], log)
        print(f"[resplit] {name} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
