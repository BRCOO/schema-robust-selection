#!/usr/bin/env python3
"""Run the v2 discovery comparison: every method x {native, v2_selector} on one cohort.

Everything is driven by one JSON config (see docs/reviews/v2_runbook.md for a full
example):

  {
    "cohort": "artifacts/v2/task/mimic_iv/episodes_split.json",
    "episode_dir": "data/processed/v2/mimic-iv-gap1",
    "out_dir": "artifacts/v2/comparison/gap1",
    "k": 10, "n_boot": 2000, "seed": 0,
    "representation": {"n_bins": 4, "observation_tokens": true, "trend_tokens": false},
    "methods": {
      "fmg_bridge": {"fmg_dir": "...", "contrastive_inputs": "..."},
      "prefixspan": {"minsup": 0.05, "max_pattern_length": 3, "pool_cap": 200},
      "erminer": {"minsup": 0.05, "minconf": 0.5, "max_antecedent": 2, "max_consequent": 1},
      "matrix_profile": {"lengths": [3, 4, 5], "top_k_per_feature": 3},
      "shapelet": {"lengths": [3, 4, 5], "top_k_per_feature": 3}
    },
    "targets": {
      "eicu": {"episodes": "...", "episode_dir": "...", "features": ["HR", "..."],
               "fmg_contrastive_inputs": "..."}
    },
    "selector": {"w_schema": 0.35},
    "reference": ["fmg_bridge", "v2_selector"]
  }

Outputs in out_dir: comparison_rows.csv, comparison_full.json (details, selector traces,
test probabilities), config.json, provenance.json. Only when the machine is free.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.extract.common import CANONICAL_FEATURES  # noqa: E402
from src.icu.v2.pipeline import TargetCohort, compare_methods  # noqa: E402
from src.icu.v2.pools import (  # noqa: E402
    NUMERIC_MINERS,
    SPMF_MINERS,
    build_fmg_pool,
    build_numeric_pool,
    build_sequences,
    build_symbolic_pool,
    labels_and_groups,
    load_arrays,
)
from src.icu.v2.selector import SelectorConfig  # noqa: E402
from src.icu.v2.types import EpisodeRecord  # noqa: E402

FMG_METHOD = "fmg_bridge"


def _rows(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("episodes", payload) if isinstance(payload, dict) else payload


def _episode(row: dict, *, split: str | None = None) -> EpisodeRecord:
    label = row.get("label")
    if label is None:
        label = 1 if row.get("episode_type") == "positive" else 0
    return EpisodeRecord(
        episode_id=str(row["episode_id"]),
        subject_id=int(row["subject_id"]),
        stay_id=int(row.get("stay_id", 0)),
        label=int(label),
        anchor_kind=str(row.get("anchor_kind", "")),
        anchor_time=str(row.get("anchor_time", "")),
        split=str(row.get("split", "")) if split is None else split,
    )


def _tuple_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: tuple(value) if isinstance(value, list) else value for key, value in params.items()}


def add_schema_shift_targets(config: dict[str, Any], episodes, arrays, pools, targets: list[TargetCohort]) -> None:
    """Controlled schema shift on the internal TEST split (config["schema_shift_targets"]).

    Each entry ``{"name": {"drop": [features]}}`` becomes a target whose schema lacks the
    dropped features; its episodes are the internal test episodes and its firings are the
    internal ones restricted to them. ``refire_external`` then zeroes every program that
    needs a dropped feature (no repair), so only the schema changes, not the population.
    """
    specs = config.get("schema_shift_targets") or {}
    if not specs:
        return
    test = tuple(replace(e, split="") for e in episodes if e.split == "test")
    test_ids = frozenset(e.episode_id for e in test)
    if config.get("internal_features"):
        base = frozenset(config["internal_features"])
    elif arrays:
        base = frozenset(next(iter(arrays.values()))[2])
    else:
        base = frozenset(CANONICAL_FEATURES)
    for name, spec in specs.items():
        drop = frozenset(spec["drop"])
        unknown = sorted(drop - base)
        if unknown:
            raise ValueError(f"schema_shift_targets[{name!r}] drops unknown features {unknown}")
        targets.append(TargetCohort(name=name, features=base - drop, episodes=test))
        for pool in pools:
            pool.external_firings[name] = {pid: frozenset(ids) & test_ids for pid, ids in pool.firings.items()}


def build_state(config: dict[str, Any]) -> dict[str, Any]:
    """Mine every method's TRAIN-only candidate pool and compute all firing sets.

    This is the expensive part (mining, ranking, firing); the returned state can be pickled
    with --save-pools and re-evaluated under other selector settings without re-mining.
    """
    episodes = [_episode(row) for row in _rows(config["cohort"])]
    labels, groups = labels_and_groups(episodes)
    train_ids = [e.episode_id for e in episodes if e.split == "train"]
    methods: dict[str, dict] = dict(config["methods"])
    unknown = sorted(set(methods) - set(SPMF_MINERS) - set(NUMERIC_MINERS) - {FMG_METHOD})
    if unknown:
        raise ValueError(f"unknown methods {unknown}")

    targets: list[TargetCohort] = []
    target_arrays: dict[str, dict] = {}
    for name, spec in (config.get("targets") or {}).items():
        target_episodes = tuple(_episode(row, split="") for row in _rows(spec["episodes"]))
        targets.append(TargetCohort(name=name, features=frozenset(spec["features"]), episodes=target_episodes))
        if any(m != FMG_METHOD for m in methods):
            target_arrays[name] = load_arrays(spec["episode_dir"], [e.episode_id for e in target_episodes])

    arrays = load_arrays(config["episode_dir"], [e.episode_id for e in episodes]) if any(m != FMG_METHOD for m in methods) else {}
    sequences, external_sequences = {}, {}
    if any(m in SPMF_MINERS for m in methods):
        sequences, external_sequences, _ = build_sequences(arrays, train_ids, extra=target_arrays, **config.get("representation", {}))

    pools = []
    timings: dict[str, float] = {}
    for method, params in methods.items():
        params = _tuple_params(dict(params))
        started = time.perf_counter()
        if method == FMG_METHOD:
            pools.append(
                build_fmg_pool(
                    params["fmg_dir"],
                    params["contrastive_inputs"],
                    episode_ids=[e.episode_id for e in episodes],
                    external_inputs={
                        name: spec["fmg_contrastive_inputs"]
                        for name, spec in (config.get("targets") or {}).items()
                        if spec.get("fmg_contrastive_inputs")
                    },
                )
            )
        elif method in SPMF_MINERS:
            pool_cap = int(params.pop("pool_cap", 200))
            pools.append(
                build_symbolic_pool(
                    method, sequences, labels, train_ids,
                    mining_params=params, pool_cap=pool_cap, external_sequences=external_sequences,
                )
            )
        else:
            pools.append(
                build_numeric_pool(
                    method, arrays, labels, train_ids,
                    groups=groups if method == "matrix_profile" else None,
                    mining_params=params, external_arrays=target_arrays,
                )
            )
        timings[method] = time.perf_counter() - started

    add_schema_shift_targets(config, episodes, arrays, pools, targets)
    return {
        "pools": pools,
        "episodes": episodes,
        "targets": targets,
        "timings": timings,
        "n_candidates": {pool.method: len(pool.candidates) for pool in pools},
    }


def evaluate_state(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Selection, bridge, metrics and paired comparisons on a built (or loaded) state."""
    k = int(config.get("k", 10))
    selector = SelectorConfig(k=k, **_tuple_params(dict(config.get("selector", {}))))
    reference = tuple(config["reference"]) if config.get("reference") else None
    started = time.perf_counter()
    result = compare_methods(
        state["pools"],
        state["episodes"],
        targets=state["targets"],
        selector_config=selector,
        k=k,
        reference=reference,
        n_boot=int(config.get("n_boot", 2000)),
        seed=int(config.get("seed", 0)),
        selector_target_names=config.get("selector_targets"),
    )
    # Wall time per method covers mining (JVM time for SPMF), ranking, and firing on every
    # internal and external episode; shared loading and eventization are excluded.
    result["timings_seconds"] = {
        "pool_build": state["timings"],
        "selection_bridge_metrics_total": time.perf_counter() - started,
        "n_candidates": state["n_candidates"],
    }
    return result


def run(config: dict[str, Any]) -> dict[str, Any]:
    return evaluate_state(build_state(config), config)


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_outputs(result: dict[str, Any], config: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    columns = sorted({key for row in rows for key in row}, key=lambda c: (c not in ("method", "selection"), c))
    with open(out / "comparison_rows.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (out / "comparison_full.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "git_head": _git_head(),
        "n_rows": len(rows),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {name: str(out / name) for name in ("comparison_rows.csv", "comparison_full.json", "config.json", "provenance.json")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="comparison config JSON")
    parser.add_argument("--out-dir", default=None, help="overrides config['out_dir']")
    parser.add_argument("--save-pools", default=None, help="pickle the built candidate pools / firings for later re-evaluation")
    parser.add_argument("--load-pools", default=None, help="skip mining: load pools pickled by --save-pools")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = args.out_dir or config.get("out_dir")
    if not out_dir:
        raise SystemExit("an output directory is required (--out-dir or config['out_dir'])")
    if args.load_pools:
        with open(args.load_pools, "rb") as handle:
            state = pickle.load(handle)
    else:
        state = build_state(config)
    if args.save_pools:
        Path(args.save_pools).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_pools, "wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    result = evaluate_state(state, config)
    written = write_outputs(result, config, out_dir)
    print(json.dumps({"rows": len(result["rows"]), "outputs": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
