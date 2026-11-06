#!/usr/bin/env python3
"""Mine FMG-Bridge programs on the TRAIN split only (v2 protocol, audit item P0-2).

Streams v1's ``contrastive_inputs.json`` (280 MB for the locked run), keeps only the
episodes that the split manifest assigns to ``--split`` (default ``train``), reruns the v1
contrastive scoring and program selection on them through :mod:`src.icu.v2.fmg_adapter`,
checks that every support count comes from those episodes, and writes to ``--output-dir``:

- ``fmg_programs_ranked.json``: v1 ``contrastive_programs`` order (constrained window first)
- ``fmg_programs_selected.json``: the first ``--top-k`` programs
- ``fmg_program_records.json``: the selected programs as ``ProgramRecord`` JSON
- ``fmg_train_firings.json``: program_id -> episodes of the mining split where it fires (v1 semantics)
- ``fmg_profile_effective.json``: profile of the final pass (reusable as ``--miner-config``)
- ``provenance.json``: input hashes, split counts, mining episode ids, config, gates, code hashes

Example (run only when the machine is free)::

    python scripts/v2/run_fmg_train_only.py \
        --contrastive-inputs artifacts/results/mimic_iv/fmgbridge_contract_internal_v1/contrastive_inputs.json \
        --split-manifest docs/experiments/icde_comparator_split_manifest.json \
        --output-dir artifacts/v2/fmg_train_only/internal_v1_windows

Nothing from the project is imported at module import time, so ``--help`` is cheap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CODE_FILES = (
    "src/icu/shock_contrastive_miner.py",
    "src/icu/rule_significance.py",
    "scripts/run_mimic_experiment.py",
    "scripts/run_component_ablation_upgrade.py",
    "docs/experiments/strict_profile_icde_locked.json",
    "src/icu/v2/fmg_adapter.py",
    "src/icu/v2/types.py",
    "scripts/v2/run_fmg_train_only.py",
)
TRANSFER_KEYS = (
    "transfer_baseline_top5_unique_groups",
    "transfer_baseline_top10_unique_groups",
    "transfer_baseline_top5_score_min",
    "transfer_max_drop_top5_unique_groups",
    "transfer_max_drop_top10_unique_groups",
    "transfer_max_drop_top5_score_min",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerun FMG-Bridge contrastive program selection on one split (train) of v1 contrastive inputs."
    )
    parser.add_argument(
        "--contrastive-inputs",
        required=True,
        help="v1 contrastive_inputs.json (a JSON list of {episode_id, episode_type, rules}); streamed.",
    )
    parser.add_argument(
        "--split-manifest",
        required=True,
        help="v1 manifest {'split': {'train': [...], 'val': [...], 'test': [...]}} or a list of {episode_id, split}.",
    )
    parser.add_argument(
        "--miner-config",
        default=None,
        help=(
            "v1 --profile-json keys, as an inline JSON object or a JSON file path. Default: the locked "
            "reference profile (scripts/run_component_ablation_upgrade.py build_reference_profile)."
        ),
    )
    parser.add_argument(
        "--transfer-target",
        choices=("two-pass", "off"),
        default="two-pass",
        help=(
            "two-pass (default): first pass without the transfer target, second pass with baselines from the "
            "first pass and v1 TRANSFER_TOLERANCES (the paper's full selector). off: use --miner-config as is."
        ),
    )
    parser.add_argument("--split", default="train", help="Split whose episodes are mined (default: train).")
    parser.add_argument("--top-k", type=int, default=10, help="Number of selected programs (default: 10).")
    parser.add_argument(
        "--domain",
        default="mimic_iv",
        help="Domain name v1 would infer from the input directory; used only with confidence-threshold files.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for the outputs listed in the module docstring.")
    parser.add_argument(
        "--compact-memory",
        action="store_true",
        help=(
            "Intern rule ids and share feature tuples while loading (src/icu/v2/fmg_prefilter.compact_episode). "
            "Values are unchanged; parity with v1 was verified on the v1 cache (2026-10-21)."
        ),
    )
    parser.add_argument(
        "--verify-v1-parity",
        action="store_true",
        help=(
            "Also run v1's own run_mimic_experiment.main on a train-only cache under OUTPUT_DIR/v1_parity and "
            "require identical programs (writes a copy of the mined inputs; heavy on real data)."
        ),
    )
    return parser


def _json_ready(value: Any) -> Any:
    """Strict-JSON view: non-finite floats -> None, sets -> sorted lists, numpy scalars -> Python."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_ready(item) for item in value)
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, allow_nan=False), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    """Read-only ``git rev-parse HEAD`` (empty string when git is unavailable)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _load_miner_config(value: str | None, fa: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if value is None:
        source = {
            "kind": "v1_locked_reference_profile",
            "builder": "scripts/run_component_ablation_upgrade.py:build_reference_profile",
            "profile_json": "docs/experiments/strict_profile_icde_locked.json",
        }
        return fa.v1_locked_reference_profile(), source
    if value.strip().startswith("{"):
        payload = json.loads(value)
        source = {"kind": "inline_json"}
    else:
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        source = {"kind": "file", "path": str(path), "sha256": _sha256_file(path)}
    if not isinstance(payload, dict):
        raise SystemExit("--miner-config must be a JSON object of v1 profile keys")
    return payload, source


def _pass_summary(result: Any, config: dict[str, Any]) -> dict[str, Any]:
    diversity = dict(result.qc.get("mechanism_diversity", {}) or {})
    return {
        "transfer_baselines": {key: config.get(key) for key in TRANSFER_KEYS},
        "ranked_program_ids": [program.get("program_id") for program in result.programs],
        "top5_unique_groups": diversity.get("top5_unique_groups"),
        "top10_unique_groups": diversity.get("top10_unique_groups"),
        "top5_score_min": diversity.get("top5_score_min"),
        "gates_overall_pass": bool(result.gates.get("overall_pass", False)),
        "gate_failures": list(dict(result.gates.get("quality", {})).get("failed", [])),
        "autotune_attempts": len(dict(result.selector.get("autotune", {})).get("attempts", [])),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    from src.icu.v2 import fmg_adapter as fa  # project imports stay out of --help

    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = Path(args.contrastive_inputs)
    manifest_path = Path(args.split_manifest)

    split_of = fa.load_split_manifest(manifest_path)
    mining_ids = {episode_id for episode_id, name in split_of.items() if name == args.split}
    if not mining_ids:
        raise SystemExit(f"split manifest {manifest_path} has no {args.split!r} episodes")
    requested, config_source = _load_miner_config(args.miner_config, fa)

    hasher = hashlib.sha256()
    counts: dict[str, Counter] = {}
    seen: set[str] = set()
    kept: list[dict] = []
    feature_cache: dict = {}
    if args.compact_memory:
        from src.icu.v2.fmg_prefilter import compact_episode
    for episode in fa.iter_json_array(inputs_path, hasher=hasher):
        episode_id = str(episode.get("episode_id", "") or "")
        if episode_id in seen:
            raise SystemExit(f"duplicate episode_id {episode_id!r} in {inputs_path}")
        seen.add(episode_id)
        split = split_of.get(episode_id, "unassigned")
        counts.setdefault(split, Counter())[str(episode.get("episode_type"))] += 1
        if split == args.split:
            kept.append(compact_episode(episode, feature_cache) if args.compact_memory else episode)
    mining_inputs = fa.filter_contrastive_inputs(kept, mining_ids)
    if not mining_inputs:
        raise SystemExit(f"no {args.split!r} episodes of the manifest occur in {inputs_path}")
    print(f"[INFO] streamed {len(seen)} episodes; mining on {len(mining_inputs)} {args.split!r} episodes", flush=True)

    if args.transfer_target == "two-pass":
        first_config = fa.without_transfer_target(requested)
        first = fa.run_fmg_contrastive_stage(mining_inputs, first_config, domain=args.domain)
        first_pass = _pass_summary(first, first_config)
        final_config = fa.transfer_target_config(requested, first)
    else:
        first_pass = None
        final_config = dict(requested)
    result = fa.run_fmg_contrastive_stage(mining_inputs, final_config, domain=args.domain)
    ranked = result.programs
    selected = ranked[: args.top_k]
    fa.assert_train_only_supports(ranked, mining_ids, mining_inputs)
    records = fa.fmg_programs_to_records(selected)
    firings = fa.fmg_firing_sets(records, {str(episode["episode_id"]): episode for episode in mining_inputs})

    parity = None
    if args.verify_v1_parity:
        work_dir = output_dir / "v1_parity"
        reference = fa.mine_fmg_programs_via_v1_entrypoint(mining_inputs, final_config, work_dir)
        canonical = [json.dumps(_json_ready(rows), sort_keys=True) for rows in (reference, ranked)]
        identical = canonical[0] == canonical[1]
        parity = {"identical": identical, "work_dir": str(work_dir), "reference_program_count": len(reference)}

    outputs = {
        "ranked_programs": output_dir / "fmg_programs_ranked.json",
        "selected_programs": output_dir / "fmg_programs_selected.json",
        "records": output_dir / "fmg_program_records.json",
        "train_firings": output_dir / "fmg_train_firings.json",
        "effective_profile": output_dir / "fmg_profile_effective.json",
        "provenance": output_dir / "provenance.json",
    }
    _write_json(outputs["ranked_programs"], ranked)
    _write_json(outputs["selected_programs"], selected)
    _write_json(outputs["records"], [fa.program_record_to_json(record) for record in records])
    _write_json(outputs["train_firings"], {pid: sorted(ids) for pid, ids in firings.items()})
    _write_json(outputs["effective_profile"], final_config)

    mining_episode_ids = sorted(str(episode["episode_id"]) for episode in mining_inputs)
    provenance = {
        "tool": "scripts/v2/run_fmg_train_only.py",
        "purpose": "FMG-Bridge contrastive scoring and program selection on one split only (audit P0-2)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "inputs": {
            "contrastive_inputs": {
                "path": str(inputs_path),
                "sha256": hasher.hexdigest(),
                "bytes": inputs_path.stat().st_size,
                "episode_count": len(seen),
            },
            "split_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
                "episode_count": len(split_of),
            },
            "miner_config": config_source,
        },
        "mining_split": args.split,
        "episode_counts_by_split": {split: dict(counter) for split, counter in sorted(counts.items())},
        "mining_episodes": {
            "count": len(mining_inputs),
            "positive": sum(1 for episode in mining_inputs if episode.get("episode_type") == "positive"),
            "control": sum(1 for episode in mining_inputs if episode.get("episode_type") == "control"),
            "manifest_ids_absent_from_inputs": len(mining_ids) - len(mining_inputs),
            "ids_sha256": hashlib.sha256("\n".join(mining_episode_ids).encode("utf-8")).hexdigest(),
            "episode_ids": mining_episode_ids,
        },
        "config": {
            "requested": requested,
            "final_pass": final_config,
            "resolved_v1_parameters": result.config,
            "transfer_target": {"mode": args.transfer_target, "first_pass": first_pass},
        },
        "result": {
            "top_k": args.top_k,
            "ranked_rule_count": result.rule_count,
            "ranked_program_count": len(ranked),
            "selected_program_ids": [record.program_id for record in records],
            "selected_program_labels": [record.params.get("program_label") for record in records],
            "train_firing_counts": {pid: len(ids) for pid, ids in firings.items()},
            "final_pass": _pass_summary(result, final_config),
            "stages": result.stages,
        },
        "guards": {"train_only_supports": "passed", "checked_programs": len(ranked)},
        "v1_parity": parity,
        "code": {
            "git_head": _git_head(),
            "sha256": {name: _sha256_file(ROOT / name) for name in CODE_FILES if (ROOT / name).exists()},
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    _write_json(outputs["provenance"], provenance)
    print(
        f"[INFO] {len(ranked)} ranked / {len(selected)} selected programs from {len(mining_inputs)} "
        f"{args.split!r} episodes -> {output_dir}",
        flush=True,
    )
    if parity is not None and not parity["identical"]:
        raise SystemExit(f"v1 parity check failed; see {outputs['provenance']}")
    return provenance


if __name__ == "__main__":
    main()
