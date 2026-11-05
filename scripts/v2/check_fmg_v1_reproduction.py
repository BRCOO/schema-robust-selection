#!/usr/bin/env python3
"""Check that the v2 FMG adapter reproduces the locked v1 FMG top-10 on the v1 inputs.

This is the real-data parity check behind ``src/icu/v2/fmg_adapter.py``: it re-runs the
v1 contrastive stage through the adapter on ALL 119 v1 discovery episodes (exactly what v1
did, test episodes included) and compares the top-10 (program_id, representative_rule_id)
with the stored v1 summary. A match means the adapter is faithful, so differences under
the train-only setting come from the data split, not from the glue code.

Loads the 55 MB summary and streams the 280 MB inputs (about 1.5-3 GB RAM). Only when the
machine is free:
  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/check_fmg_v1_reproduction.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2 import fmg_adapter as fa  # noqa: E402

DEFAULT_RUN = ROOT / "artifacts" / "results" / "mimic_iv" / "fmgbridge_contract_internal_v1"


def locked_config_from_summary(filters: dict) -> dict:
    """The adapter-accepted subset of the stored v1 ``contrastive_filters``."""
    keys = set(fa.resolve_v1_profile({})) - {"exclude_sparse_measurement_dominant", "upstream_only_keys_without_effect"}
    config = {key: value for key, value in filters.items() if key in keys}
    config.pop("stability_auto_adaptive_history_glob", None)
    config["hard_exclude_sparse_measurement_dominant"] = bool(filters.get("exclude_sparse_measurement_dominant", False))
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "mimic_iv_summary.json").read_text(encoding="utf-8"))
    config = locked_config_from_summary(summary["contrastive_filters"])
    want = [(p["program_id"], p["representative_rule_id"]) for p in summary["contrastive_programs"][: args.top_k]]
    transfer = {key: config.get(key) for key in sorted(config) if key.startswith("transfer_")}
    del summary
    inputs = list(fa.iter_json_array(run_dir / "contrastive_inputs.json"))
    got = [(p["program_id"], p["representative_rule_id"]) for p in fa.mine_fmg_programs(inputs, config)[: args.top_k]]
    report = {"reproduces_locked_top_k": got == want, "transfer_settings": transfer, "expected": want, "got": got}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if got == want else 1


if __name__ == "__main__":
    raise SystemExit(main())
