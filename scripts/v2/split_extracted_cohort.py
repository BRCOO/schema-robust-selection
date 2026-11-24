#!/usr/bin/env python3
"""Turn an extraction directory into a cohort file for the v2 runners.

Internal source (default): patient-level train/val/test split + pre-registration.
External source (--external): every episode is a held-out target episode (split "external").

  PYTHONDONTWRITEBYTECODE=1 python scripts/v2/split_extracted_cohort.py \\
      --extract-dir data/processed/v2/extract/mimic_iv_gap1h --out-dir artifacts/v2/task/x_mimic_iv_gap1h
  ... --extract-dir data/processed/v2/extract/eicu_gap1h --out-dir artifacts/v2/task/x_eicu_gap1h --external
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.cohort import preregistration_record, sample_patient_ids  # noqa: E402
from src.icu.v2.extract.common import episode_records  # noqa: E402
from src.icu.v2.splits import patient_level_split, split_summary  # noqa: E402
from src.icu.v2.task import TaskSpec  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--fractions", default="0.7,0.1,0.2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes-per-patient", type=int, default=None)
    parser.add_argument("--sample-patients", type=int, default=None)
    args = parser.parse_args(argv)
    extract_dir, out = Path(args.extract_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((extract_dir / "extraction_manifest.json").read_text(encoding="utf-8"))
    episodes = episode_records(extract_dir / "shock_episode_index.json")
    if args.sample_patients is not None:
        chosen = sample_patient_ids({e.subject_id for e in episodes}, args.sample_patients, args.seed)
        episodes = [e for e in episodes if e.subject_id in chosen]
    if args.external:
        rows = [asdict(replace(e, split="external")) for e in episodes]
        summary = {"external": {"episodes": len(rows), "positives": sum(r["label"] for r in rows), "patients": len({r["subject_id"] for r in rows})}}
        (out / "episodes_all.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    else:
        fractions = tuple(float(x) for x in args.fractions.split(","))
        split, split_manifest = patient_level_split(episodes, fractions=fractions, seed=args.seed, max_episodes_per_patient=args.max_episodes_per_patient)
        rows = [asdict(e) for e in split]
        summary = split_summary(split)
        (out / "episodes_split.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        (out / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        # AKI task: vasopressor_on is an input and the anchors are KDIGO onsets
        task = {"excluded_features": (), "anchor_kinds": ("aki_onset",)} if manifest["spec"].get("task") == "aki_onset" else {}
        prereg = preregistration_record(
            TaskSpec(window_hours=float(manifest["spec"]["window_hours"]), prediction_gap_hours=float(manifest["spec"]["prediction_gap_hours"]), **task),
            {"fractions": list(fractions), "seed": args.seed, "max_episodes_per_patient": args.max_episodes_per_patient, "grouping": "subject_id", "sample_patients": args.sample_patients},
            root=ROOT,
            extra={"extraction_manifest": manifest},
        )
        (out / "preregistration.json").write_text(json.dumps(prereg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "cohort_summary.json").write_text(json.dumps({"source": manifest["source"], "extract_dir": str(extract_dir), "summary": summary, "available_features": manifest["available_features"]}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
