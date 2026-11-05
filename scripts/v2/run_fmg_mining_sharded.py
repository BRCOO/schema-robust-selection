#!/usr/bin/env python3
"""Run v1 per-episode FMG mining in parallel shards and merge the contrastive cache.

``scripts/run_mimic_experiment.py`` mines episodes one by one and keeps every episode's
rule records in memory until it writes one indented ``contrastive_inputs.json`` (about
2.4 MB per episode in v1), which does not fit in RAM for thousands of episodes. Per-episode
mining is label-free and independent across episodes, so sharding does not change any
episode's rules: each shard is a separate, unmodified v1 process on a subset of the
materialized episodes, and the shard caches are streamed into one compact JSON array.

Shard inputs are hard links (same volume) or copies of the materialized episode files plus
a shard ``shock_episode_index.json``; the materialized directory itself is only read.

Only when the machine is free:
  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_fmg_mining_sharded.py \\
      --input data/processed/v2/mimic-iv-gap1h --out-dir artifacts/v2/fmg_mining/mimic_iv_gap1h \\
      --shards 16 --workers 6 --delete-shard-caches
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.fmg_adapter import iter_json_array  # noqa: E402
from src.icu.v2.materialize import episode_paths  # noqa: E402

RME = ROOT / "scripts" / "run_mimic_experiment.py"


def shard_metas(metas: Sequence[dict], n_shards: int) -> list[list[dict]]:
    """Round-robin over episodes sorted by id, so shards are balanced and deterministic."""
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    ordered = sorted(metas, key=lambda meta: str(meta["episode_id"]))
    shards: list[list[dict]] = [[] for _ in range(min(n_shards, max(1, len(ordered))))]
    for i, meta in enumerate(ordered):
        shards[i % len(shards)].append(meta)
    return shards


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_shard_input(src_dir: Path, shard_dir: Path, metas: Sequence[dict]) -> Path:
    """Shard input dir with linked episode files and its own index."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    for meta in metas:
        src = episode_paths(src_dir, str(meta["episode_id"]))
        dst = episode_paths(shard_dir, str(meta["episode_id"]))
        for key in ("values", "mask", "meta"):
            _link_or_copy(src[key], dst[key])
    (shard_dir / "shock_episode_index.json").write_text(json.dumps(list(metas), ensure_ascii=False), encoding="utf-8")
    return shard_dir


def run_shard(python: str, shard_input: Path, shard_out: Path, extra_args: Sequence[str] = ()) -> dict[str, Any]:
    """One unmodified v1 process; returns status, elapsed time and its cache manifest."""
    shard_out.mkdir(parents=True, exist_ok=True)
    command = [
        python, str(RME), "--input", str(shard_input), "--out-dir", str(shard_out),
        "--disable-episode-type-balance", "--stop-after-contrastive-cache", *extra_args,
    ]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="")
    started = time.perf_counter()
    with open(shard_out / "shard.log", "w", encoding="utf-8", errors="replace") as log:
        status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env).returncode
    manifest_path = shard_out / "contrastive_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    return {"shard": shard_out.parent.name, "returncode": status, "elapsed_seconds": time.perf_counter() - started, "manifest": manifest}


def merge_shard_caches(cache_paths: Sequence[Path], merged_path: Path, keep_rule_ids: frozenset[str] | None = None) -> dict[str, Any]:
    """Stream shard ``contrastive_inputs.json`` arrays into one compact array.

    Raises if an episode id appears in two shards. With ``keep_rule_ids`` only those rule
    records are kept (every episode is still written), which is all firing needs on an
    external source. Returns counts and the id list.
    """
    seen: set[str] = set()
    tmp_path = merged_path.with_suffix(".json.partial")
    with open(tmp_path, "w", encoding="utf-8") as out:
        out.write("[")
        first = True
        for path in cache_paths:
            if not path.exists():
                continue
            for item in iter_json_array(path):
                episode_id = str(item.get("episode_id"))
                if episode_id in seen:
                    raise ValueError(f"episode {episode_id!r} appears in more than one shard")
                seen.add(episode_id)
                if keep_rule_ids is not None:
                    item = {**item, "rules": [r for r in item.get("rules", ()) if str(r.get("rule_id")) in keep_rule_ids]}
                out.write("" if first else ",")
                out.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
                first = False
        out.write("]")
    tmp_path.replace(merged_path)
    return {"n_episodes": len(seen), "episode_ids": sorted(seen)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="materialized v2 episode dir with shock_episode_index.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None, help="timing probes only: first N episodes by id")
    parser.add_argument("--delete-shard-caches", action="store_true", help="delete shard caches and shard inputs after a successful merge")
    parser.add_argument("--keep-rule-ids", default=None, help="JSON list of rule ids to keep when merging (e.g. the FMG candidates' member rules)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src_dir, out_dir = Path(args.input), Path(args.out_dir)
    metas = json.loads((src_dir / "shock_episode_index.json").read_text(encoding="utf-8"))
    if args.limit is not None:
        metas = sorted(metas, key=lambda meta: str(meta["episode_id"]))[: args.limit]
    shards = shard_metas(metas, args.shards)
    started = time.perf_counter()
    jobs = []
    for k, shard in enumerate(shards):
        shard_root = out_dir / "shards" / f"shard_{k:03d}"
        jobs.append((prepare_shard_input(src_dir, shard_root / "input", shard), shard_root / "out"))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(lambda job: run_shard(args.python, job[0], job[1]), jobs))
    failed = [r for r in results if r["returncode"] != 0 or not r["manifest"]]
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(src_dir),
        "n_requested": len(metas),
        "n_shards": len(shards),
        "workers": args.workers,
        "shards": results,
        "failed_shards": [r["shard"] for r in failed],
        "mining_elapsed_seconds": time.perf_counter() - started,
    }
    if not failed:
        keep = frozenset(json.loads(Path(args.keep_rule_ids).read_text(encoding="utf-8"))) if args.keep_rule_ids else None
        merged = merge_shard_caches([out / "contrastive_inputs.json" for _, out in jobs], out_dir / "contrastive_inputs.json", keep)
        report["kept_rule_ids"] = None if keep is None else len(keep)
        report["n_merged_episodes"] = merged["n_episodes"]
        report["n_skipped_by_v1"] = len(metas) - merged["n_episodes"]
        if args.delete_shard_caches:
            for shard_input, shard_out in jobs:
                (shard_out / "contrastive_inputs.json").unlink(missing_ok=True)
                shutil.rmtree(shard_input, ignore_errors=True)
            report["shard_caches_deleted"] = True
    (out_dir / "sharded_mining_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "shards"}, indent=2, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
