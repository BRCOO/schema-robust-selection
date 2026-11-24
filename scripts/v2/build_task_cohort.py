#!/usr/bin/env python3
"""Build the v2 cohort (task eligibility + patient-level split + pre-registration).

Streams the v1 episode index, applies a TaskSpec, splits whole patients, and writes
(in --out-dir):
  episodes_split.json   EpisodeRecord rows with split; input for scripts/v2/run_supervised_v2.py
  task_spec.json        the TaskSpec used
  split_manifest.json   per-split episode/patient counts, seed, cap, patient lists
  cohort_summary.json   input/eligible counts, split summary, exclusions by reason
  exclusions.json       every excluded episode with its reasons
  preregistration.json  frozen settings + content hash; commit it before running models

Only when the machine is free (it reads the 560 MB index once, streaming):
  PYTHONDONTWRITEBYTECODE=1 python scripts/v2/build_task_cohort.py \\
      --index data/processed/mimic-iv-full-cw3-gap0-real/mimic-iv/shock_episode_index.json \\
      --task-spec <task_spec.json> --max-episodes-per-patient 10 \\
      --out-dir artifacts/v2/task/mimic_iv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.cohort import (  # noqa: E402
    build_cohort,
    iter_index_records,
    preregistration_record,
    task_spec_from_dict,
    trim_record,
    write_cohort_outputs,
)
from src.icu.v2.task import TaskSpec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="v1 shock_episode_index.json (top-level JSON array)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task-spec", default=None, help="TaskSpec JSON (bare or under 'task_spec'); default TaskSpec()")
    parser.add_argument("--fractions", default="0.7,0.1,0.2", help="train,val,test episode fractions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes-per-patient", type=int, default=None)
    parser.add_argument("--sample-patients", type=int, default=None, help="uniform random patient sample drawn after eligibility (hash-ranked, order-independent)")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--extra", default=None, help="JSON file with other frozen settings (metrics, selector weights, primary gap)")
    parser.add_argument("--hash-index", action="store_true", help="also sha256 the index file (slow for 560 MB)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="SMOKE TESTS ONLY: stop after N index records. The index is sorted by stay_id, "
        "so a prefix is not a sample (v1's first-500 rows held 31 patients).",
    )
    return parser


def _load_spec(path: str | None) -> TaskSpec:
    if not path:
        return TaskSpec()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("task_spec"), dict):
        payload = payload["task_spec"]
    return task_spec_from_dict(payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = _load_spec(args.task_spec)
    fractions = tuple(float(part) for part in args.fractions.split(","))
    records = []
    for i, record in enumerate(iter_index_records(args.index)):
        if args.limit is not None and i >= args.limit:
            break
        records.append(trim_record(record))
    result = build_cohort(
        records,
        spec,
        fractions=fractions,
        seed=args.seed,
        max_episodes_per_patient=args.max_episodes_per_patient,
        sample_patients=args.sample_patients,
        sample_seed=args.sample_seed,
    )
    split_params = {
        "fractions": list(fractions),
        "seed": args.seed,
        "max_episodes_per_patient": args.max_episodes_per_patient,
        "limit": args.limit,
        "sample_patients": args.sample_patients,
        "sample_seed": args.sample_seed,
        "grouping": "subject_id",
    }
    extra = json.loads(Path(args.extra).read_text(encoding="utf-8")) if args.extra else None
    prereg = preregistration_record(
        spec,
        split_params,
        root=ROOT,
        index_path=args.index,
        hash_index=args.hash_index,
        extra=extra,
    )
    written = write_cohort_outputs(result, spec, prereg, args.out_dir)
    summary = {
        "n_input_records": result["n_input_records"],
        "n_eligible": result["n_eligible"],
        "n_patients_eligible": result["n_patients_eligible"],
        "n_patients_sampled": result["n_patients_sampled"],
        "split_summary": result["split_summary"],
        "limit": args.limit,
        "outputs": written,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if args.limit is not None:
        print("WARNING: --limit used; this cohort is a prefix of the index, not a sample.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
