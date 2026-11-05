#!/usr/bin/env python3
"""Write a support-pre-filtered, train-only copy of an FMG contrastive cache.

Two streaming passes over --contrastive-inputs: count rule support on the mining split,
then write that split's episodes with only the rules that can pass v1's support
thresholds (see src/icu/v2/fmg_prefilter.py). The output feeds
scripts/v2/run_fmg_train_only.py unchanged.

--verify-on <v1 contrastive_inputs.json> first checks exactness against v1 itself: it
mines the given v1 cache (all episodes, as v1 did) with and without the pre-filter and
requires identical ranked programs, gates and QC; the run stops if they differ.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/prefilter_fmg_inputs.py \\
      --contrastive-inputs artifacts/v2/fmg_mining/mimic_iv_gap1h/contrastive_inputs.json \\
      --split-manifest artifacts/v2/task/mimic_iv_gap1h/episodes_split.json \\
      --out artifacts/v2/fmg_mining/mimic_iv_gap1h/contrastive_inputs_train_prefiltered.json \\
      --verify-on artifacts/results/mimic_iv/fmgbridge_contract_internal_v1/contrastive_inputs.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2 import fmg_adapter as fa  # noqa: E402
from src.icu.v2.fmg_prefilter import (  # noqa: E402
    count_rule_support,
    iter_filtered_inputs,
    kept_rule_ids,
    prefilter_thresholds,
)


def verify_against_v1(v1_inputs: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Mine the v1 cache with and without the pre-filter; both runs must be identical."""
    min_pos, min_ctl = prefilter_thresholds(fa.resolve_v1_profile(config))
    full = list(fa.iter_json_array(v1_inputs))
    positives, controls, _ = count_rule_support(v1_inputs)
    keep = kept_rule_ids(positives, controls, min_pos, min_ctl)
    filtered = list(iter_filtered_inputs(v1_inputs, keep))
    a = fa.run_fmg_contrastive_stage(full, config)
    b = fa.run_fmg_contrastive_stage(filtered, config)
    # qc["lead_time_hours"] summarizes ALL raw rule records (count and mean), so it changes
    # by construction when records are dropped; it is descriptive and feeds no gate here
    # (gates are compared directly). Every other qc field must be identical.
    descriptive = ("lead_time_hours",)
    same = {
        "programs": a.programs == b.programs,
        "gates": a.gates == b.gates,
        "selector": a.selector == b.selector,
        "stages": a.stages == b.stages,
        "rule_count": a.rule_count == b.rule_count,
        "qc_except_descriptive": {k: v for k, v in a.qc.items() if k not in descriptive}
        == {k: v for k, v in b.qc.items() if k not in descriptive},
    }
    return {
        "identical": all(same.values()),
        "components": same,
        "descriptive_qc_full": {k: a.qc.get(k) for k in descriptive},
        "descriptive_qc_prefiltered": {k: b.qc.get(k) for k in descriptive},
        "rule_records_before": sum(len(e.get("rules", ())) for e in full),
        "rule_records_after": sum(len(e.get("rules", ())) for e in filtered),
        "thresholds": [min_pos, min_ctl],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contrastive-inputs", required=True)
    parser.add_argument("--split-manifest", required=True, help="episodes_split.json or a v1 split manifest")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--miner-config", default=None, help="JSON profile; default: the v1 locked reference profile")
    parser.add_argument("--verify-on", default=None, help="v1 contrastive_inputs.json used for the exactness check")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.miner_config).read_text(encoding="utf-8")) if args.miner_config else fa.v1_locked_reference_profile()
    report: dict[str, Any] = {"input": args.contrastive_inputs, "split": args.split}
    if args.verify_on:
        started = time.perf_counter()
        report["verification"] = verify_against_v1(Path(args.verify_on), config)
        report["verification"]["elapsed_seconds"] = time.perf_counter() - started
        print(json.dumps({"verification": report["verification"]}, indent=2))
        if not report["verification"]["identical"]:
            print("pre-filter is NOT exact for this configuration; stopping", file=sys.stderr)
            return 3
    min_pos, min_ctl = prefilter_thresholds(fa.resolve_v1_profile(config))
    split_of = fa.load_split_manifest(args.split_manifest)
    started = time.perf_counter()
    positives, controls, episodes = count_rule_support(args.contrastive_inputs, split_of, args.split)
    keep = kept_rule_ids(positives, controls, min_pos, min_ctl)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_episodes = n_rules = 0
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("[")
        for i, item in enumerate(iter_filtered_inputs(args.contrastive_inputs, keep, split_of, args.split)):
            handle.write("," if i else "")
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            n_episodes += 1
            n_rules += len(item["rules"])
        handle.write("]")
    report.update(
        {
            "thresholds": {"min_rule_positive_count": min_pos, "min_rule_control_count": min_ctl},
            "episodes": episodes,
            "n_episodes_written": n_episodes,
            "distinct_rules_before": len(set(positives) | set(controls)),
            "distinct_rules_kept": len(keep),
            "rule_records_written": n_rules,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    Path(str(out) + ".manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "verification"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
