#!/usr/bin/env python3
"""Extract a leakage-safe v2 cohort from raw MIMIC-IV / eICU / HiRID tables.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/extract_v2_cohort.py \\
      --source mimic_iv --raw-dir data/raw/mimic-iv/mimic-iv-3.1 \\
      --cache-dir data/processed/v2/cache/mimic_iv \\
      --out-dir data/processed/v2/extract/mimic_iv_gap1h --n-positive-patients 3000 --fmg-view

``--task aki_onset`` extracts the second task (KDIGO creatinine AKI onset; see the module
docstring of src/icu/v2/extract/common.py), e.g. with ``--gap-hours 6``.

Writes the v1-layout episode directory (raw canonical units, NaN = not observed) and,
with --fmg-view, a sibling ``<out-dir>_fmg`` with FMG-Bridge's per-episode-scaled
representation for v1 per-episode mining.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.extract.common import TASKS, ExtractionSpec, extract_cohort, write_fmg_representation  # noqa: E402

ADAPTERS = {
    "mimic_iv": "src.icu.v2.extract.mimic_iv:MimicIVAdapter",
    "eicu": "src.icu.v2.extract.eicu:EICUAdapter",
    "hirid": "src.icu.v2.extract.hirid:HiRIDAdapter",
}


def load_adapter(source: str, raw_dir: str, cache_dir: str):
    module_name, class_name = ADAPTERS[source].split(":")
    return getattr(importlib.import_module(module_name), class_name)(raw_dir, cache_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-hours", type=int, default=12)
    parser.add_argument("--gap-hours", type=float, default=1.0)
    parser.add_argument("--controls-per-positive", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-positive-patients", type=int, default=None)
    parser.add_argument("--fmg-view", action="store_true")
    parser.add_argument("--match-columns", default="", help="comma list of stays columns controls must share (eICU: hospitalid)")
    parser.add_argument("--required-vital", default="HR", help="vital that must be observed in the last bins before the cutoff ('' disables)")
    parser.add_argument("--required-vital-last-bins", type=int, default=1)
    parser.add_argument("--max-positives-per-patient", type=int, default=1, help="0 keeps all positive stays")
    parser.add_argument("--chunk-stays", type=int, default=2000)
    parser.add_argument("--task", choices=TASKS, default="vasopressor_initiation")
    parser.add_argument("--aki-max-first-creatinine", type=float, default=4.0,
                        help="aki_onset: exclude stays whose first ICU creatinine is >= this (mg/dL); 0 disables")
    args = parser.parse_args(argv)
    spec = ExtractionSpec(
        window_hours=args.window_hours,
        prediction_gap_hours=args.gap_hours,
        controls_per_positive=args.controls_per_positive,
        seed=args.seed,
        max_positives_per_patient=None if args.max_positives_per_patient == 0 else args.max_positives_per_patient,
        match_columns=tuple(c for c in args.match_columns.split(",") if c),
        required_vital=args.required_vital or None,
        required_vital_last_bins=args.required_vital_last_bins if args.required_vital else 0,
        task=args.task,
        aki_max_first_creatinine=args.aki_max_first_creatinine or None,
    )
    started = time.perf_counter()
    adapter = load_adapter(args.source, args.raw_dir, args.cache_dir)
    manifest = extract_cohort(adapter, spec, args.out_dir, n_positive_patients=args.n_positive_patients, chunk_stays=args.chunk_stays)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    if args.fmg_view:
        manifest["fmg_view"] = write_fmg_representation(args.out_dir, str(Path(args.out_dir)) + "_fmg")
    Path(args.out_dir, "extraction_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("source", "counts", "unavailable_features", "elapsed_seconds")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
