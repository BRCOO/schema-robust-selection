#!/usr/bin/env python3
"""Second domain, end to end: Backblaze drive failure (source Seagate; targets HGST, Toshiba, WDC).

Pre-registered in ``docs/reviews/v2_drive_domain_preregistration_2026-11-02.md``; the method and every
setting are the ICU ones. Steps (resumable, each skipped when its output exists):

  compare   comparison config + ``run_discovery_comparison.py`` with the four non-FMG generators
            (pools mined on the Seagate TRAIN split, fired on all episodes and the three targets)
  union     E3 (``run_union_selection.py``) at K = 5, 10, 20, known targets = the three manufacturers
  extra     the E3 add-on (``run_union_extra.py``) at every K

Inputs from ``scripts/v2/extract_backblaze_cohort.py``. Outputs under ``artifacts/v2/drive/``.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_drive_track.py
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
TARGETS = ("hgst", "toshiba", "wdc")


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
    parser.add_argument("--tag", default="gap2d")
    parser.add_argument("--source", default="seagate")
    parser.add_argument("--budgets", default="10,5,20")
    args = parser.parse_args(argv)
    extract = lambda v: ROOT / f"data/processed/v2/extract/backblaze_{v}_{args.tag}_full"  # noqa: E731
    task = lambda v: ROOT / f"artifacts/v2/task/f_backblaze_{v}_{args.tag}"  # noqa: E731
    name = f"f_backblaze_{args.source}_{args.tag}"
    comparison = ROOT / f"artifacts/v2/drive/comparison/{name}"
    log = ROOT / f"artifacts/v2/logs/drive_{args.tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    manifests = {v: json.loads((extract(v) / "extraction_manifest.json").read_text(encoding="utf-8")) for v in (args.source, *TARGETS)}
    if not (comparison / "pools.pkl").exists():
        icu = json.loads((ROOT / "artifacts/v2/comparison/f_mimic_iv_gap1h/config.json").read_text(encoding="utf-8"))
        config = {
            "_note": f"Second domain (pre-registered 2026-11-02): Backblaze drive failure, {args.tag}, source {args.source}; generator settings of the ICU runs, no FMG.",
            "cohort": str(task(args.source) / "episodes_split.json"),
            "episode_dir": str(extract(args.source)),
            "out_dir": str(comparison),
            "k": 10, "n_boot": 200, "seed": 0,
            "representation": icu["representation"],
            "internal_features": manifests[args.source]["available_features"],
            "methods": {m: v for m, v in icu["methods"].items() if m != "fmg_bridge"},
            "targets": {v: {"episodes": str(task(v) / "episodes_all.json"), "episode_dir": str(extract(v)), "features": manifests[v]["available_features"]} for v in TARGETS},
            "schema_shift_targets": {},
            "selector_targets": list(TARGETS),
            "selector": icu.get("selector", {}),
            "reference": None,
        }
        path = ROOT / f"artifacts/v2/drive/comparison/configs/{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        sh([PY, "scripts/v2/run_discovery_comparison.py", "--config", str(path), "--save-pools", str(comparison / "pools.pkl")], log)
    for k in [int(x) for x in args.budgets.split(",") if x]:
        union = ROOT / f"artifacts/v2/drive/union_k{k}/{name}"
        extra = ROOT / f"artifacts/v2/drive/union_extra_k{k}/{name}"
        if not (union / "manifest.json").exists():
            sh([PY, "scripts/v2/run_union_selection.py", "--pools", str(comparison / "pools.pkl"), "--e1-dir", "", "--known-targets", ",".join(TARGETS),
                "--k", str(k), "--out-dir", str(union)], log)
        if not (extra / "manifest.json").exists():
            sh([PY, "scripts/v2/run_union_extra.py", "--e3-dir", str(union), "--out-dir", str(extra)], log)
        print(f"[drive] K = {k} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
