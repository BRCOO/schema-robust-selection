#!/usr/bin/env python3
"""Leakage gate for a materialized v2 cohort: run before any miner or model.

Loads the cohort's episodes from the materialized directory (already cut at the v2
cutoff, so no spec is re-applied), then runs the single-feature probes and the
multivariate probes ("all", "observation_only", "values_only") with patient-grouped
cross-validation. Writes cohort_leakage_audit.json and exits with status 2 when any
single feature reaches the separability threshold, so a pipeline can stop there.

Only when the machine is free:
  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_cohort_leakage_audit.py \\
      --episodes artifacts/v2/task/mimic_iv_gap1/episodes_split.json \\
      --episode-dir data/processed/v2/mimic-iv-gap1 \\
      --out-json artifacts/v2/leakage_audit/mimic_iv_gap1_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.leakage_audit import (  # noqa: E402
    DEFAULT_SEPARABILITY_THRESHOLD,
    collect_summaries,
    flag_leaks,
    load_episodes,
    multivariate_probe,
    single_feature_probes,
)

VIEWS: tuple[str, ...] = ("all", "observation_only", "values_only")


def audit_cohort(rows: list[dict], episode_dir: str | Path, *, threshold: float, n_bootstrap: int) -> dict[str, Any]:
    by_id = {str(row["episode_id"]): row for row in rows}
    episodes = load_episodes(episode_dir, list(by_id))
    collected = collect_summaries(episodes, None, apply_eligibility=False)
    summaries, labels = collected["summaries"], collected["labels"]
    mismatched = [eid for eid, label in labels.items() if int(by_id[eid].get("label", label)) != int(label)]
    if mismatched:
        raise ValueError(f"{len(mismatched)} episodes have labels that disagree with their metadata, e.g. {mismatched[:3]}")
    probes = single_feature_probes(summaries, labels)
    flagged = flag_leaks(probes, threshold)
    train_ids = [eid for eid, row in by_id.items() if row.get("split") in ("train", "val") and eid in summaries]
    test_ids = [eid for eid, row in by_id.items() if row.get("split") == "test" and eid in summaries]
    groups = {eid: row["subject_id"] for eid, row in by_id.items()}
    multivariate: dict[str, Any] = {}
    for view in VIEWS:
        try:
            multivariate[view] = multivariate_probe(
                summaries, labels, train_ids, test_ids, view=view, groups=groups, n_bootstrap=n_bootstrap
            )
        except ValueError as exc:  # e.g. no observation columns vary in this cohort
            multivariate[view] = {"error": str(exc)}
    return {
        "n_episodes": len(summaries),
        "threshold": threshold,
        "top_probes": probes[:25],
        "flagged": flagged,
        "multivariate": multivariate,
        "notes": [
            "train for the multivariate probe = train + val splits; test = test split",
            "summaries come from the materialized (already cut) arrays; no spec is re-applied",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True, help="episodes_split.json from build_task_cohort.py")
    parser.add_argument("--episode-dir", required=True, help="materialized v2 episode directory")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SEPARABILITY_THRESHOLD)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = json.loads(Path(args.episodes).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("episodes", [])
    result = audit_cohort(rows, args.episode_dir, threshold=args.threshold, n_bootstrap=args.n_bootstrap)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"n_episodes": result["n_episodes"], "n_flagged": len(result["flagged"]), "top": result["top_probes"][:3]}, indent=2, default=str))
    return 2 if result["flagged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
