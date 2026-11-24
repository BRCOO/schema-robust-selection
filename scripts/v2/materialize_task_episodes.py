#!/usr/bin/env python3
"""Write v2 task windows as a v1-layout episode directory for FMG per-episode mining.

Inputs: the cohort from scripts/v2/build_task_cohort.py (episodes_split.json and
task_spec.json) and the v1 episode directory. Output: a NEW directory with
shock_episode_index.json, per-episode .npy / _missing_mask.npy / .json, and
materialization_manifest.json. The source directory is read by constructed path only.

Only when the machine is free:
  PYTHONDONTWRITEBYTECODE=1 python scripts/v2/materialize_task_episodes.py \\
      --episodes artifacts/v2/task/mimic_iv/episodes_split.json \\
      --task-spec artifacts/v2/task/mimic_iv/task_spec.json \\
      --src-dir data/processed/mimic-iv-full-cw3-gap0-real/mimic-iv \\
      --dst-dir data/processed/v2/mimic-iv-<task>
Then mine (label-free, per episode) and run the train-only contrastive stage:
  python scripts/run_mimic_experiment.py --input data/processed/v2/mimic-iv-<task> \\
      --out-dir artifacts/v2/fmg_mining/<task> --disable-episode-type-balance --stop-after-contrastive-cache
  python scripts/v2/run_fmg_train_only.py --contrastive-inputs artifacts/v2/fmg_mining/<task>/contrastive_inputs.json \\
      --split-manifest artifacts/v2/task/mimic_iv/episodes_split.json --output-dir artifacts/v2/fmg_train_only/<task>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.cohort import task_spec_from_dict  # noqa: E402
from src.icu.v2.materialize import materialize_cohort  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True, help="episodes_split.json from build_task_cohort.py")
    parser.add_argument("--task-spec", required=True, help="task_spec.json from build_task_cohort.py")
    parser.add_argument("--src-dir", required=True, help="v1 episode directory (read only)")
    parser.add_argument("--dst-dir", required=True, help="new directory to write")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="smoke tests only: first N listed episodes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(Path(args.task_spec).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("task_spec"), dict):
        payload = payload["task_spec"]
    spec = task_spec_from_dict(payload)
    rows = json.loads(Path(args.episodes).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("episodes", [])
    episode_ids = [str(row["episode_id"]) for row in rows]
    if args.limit is not None:
        episode_ids = episode_ids[: args.limit]
    manifest = materialize_cohort(args.src_dir, args.dst_dir, episode_ids, spec, overwrite=args.overwrite)
    print(json.dumps({key: manifest[key] for key in ("n_requested", "n_written", "n_failed", "positives", "controls")}, indent=2))
    return 0 if manifest["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
