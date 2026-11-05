"""Materialize v2 task windows as v1-layout episode files.

FMG-Bridge's per-episode rule mining (``scripts/run_mimic_experiment.py``) reads an
episode directory: ``shock_episode_index.json`` plus, per episode,
``shock_episode_<id>.npy``, ``shock_episode_<id>_missing_mask.npy`` and a metadata
``.json``. To mine under the leakage-safe v2 task, each kept v1 episode is cut at the
prediction cutoff with :func:`~src.icu.v2.task.transform_episode` (excluded features
dropped, imputation and scaling refit on the kept rows) and written to a NEW directory in
the same layout. Episode ids are unchanged so split manifests still apply.

The source directory is only read, by constructed path, and never listed (it holds about
950k files). Writing into the source directory or one of its subdirectories is refused.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.icu.shock_episode_dataset import CORE_OBSERVATION_FEATURES
from src.icu.v2.task import TaskSpec, transform_episode

FEATURE_LIST_KEYS: tuple[str, ...] = (
    "intervention_features",
    "missingness_features",
    "observation_density_features",
)


def spec_sha256(spec: TaskSpec) -> str:
    canonical = json.dumps(spec.to_dict(), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def episode_paths(directory: str | Path, episode_id: str) -> dict[str, Path]:
    """v1 file names for one episode (the id already starts with ``stay_``)."""
    directory = Path(directory)
    stem = f"shock_episode_{episode_id}"
    return {
        "values": directory / f"{stem}.npy",
        "mask": directory / f"{stem}_missing_mask.npy",
        "meta": directory / f"{stem}.json",
    }


def _check_destination(src_dir: Path, dst_dir: Path) -> None:
    src = src_dir.resolve()
    dst = dst_dir.resolve()
    if dst == src or src in dst.parents:
        raise ValueError(f"refusing to write into the source directory {src}")


def core_observation_counts(mask: np.ndarray, feature_names: list[str]) -> tuple[int, int]:
    """(observed core channels, observed core points) on the kept rows, v1 definition."""
    core = [j for j, name in enumerate(feature_names) if name in CORE_OBSERVATION_FEATURES]
    if not core or mask.size == 0:
        return 0, 0
    observed = (~np.asarray(mask, dtype=bool)[:, core]).sum(axis=0)
    return int((observed > 0).sum()), int(observed.sum())


def build_v2_meta(meta: Mapping[str, Any], values: np.ndarray, mask: np.ndarray, names: list[str], info: Mapping[str, Any], spec: TaskSpec) -> dict[str, Any]:
    """Metadata for a materialized window; v1 keys are kept and updated, provenance under ``v2``."""
    step = pd.Timedelta(minutes=int(spec.resample_minutes))
    first_bin = pd.Timestamp(meta["start_time"]) + step * int(info["rows_trimmed_start"])
    rows = int(values.shape[0])
    channels, points = core_observation_counts(mask, names)
    new_meta = dict(meta)
    new_meta.update(
        {
            "feature_names": names,
            "shape": [rows, int(values.shape[1])],
            "start_time": first_bin.isoformat(),
            "end_time": (first_bin + step * max(0, rows - 1)).isoformat(),
            "window_start": meta.get("window_start") if int(info["rows_trimmed_start"]) == 0 else first_bin.isoformat(),
            "window_end": info["cutoff_time"],
            "missing_mask_file": f"shock_episode_{meta['episode_id']}_missing_mask.npy",
            "core_observed_channels": channels,
            "core_observed_points": points,
        }
    )
    for key in FEATURE_LIST_KEYS:
        if meta.get(key) is not None:
            new_meta[key] = [name for name in meta[key] if name in names]
    new_meta["v2"] = {
        "source_shape": list(meta.get("shape") or []),
        "source_window_end": meta.get("window_end"),
        "reference_time": info["reference_time"],
        "cutoff_time": info["cutoff_time"],
        "dropped_features": list(info["dropped_features"]),
        "rows_trimmed_start": int(info["rows_trimmed_start"]),
        "reimputed": bool(info["reimputed"]),
        "task_spec_sha256": spec_sha256(spec),
    }
    return new_meta


def materialize_episode(src_dir: str | Path, dst_dir: str | Path, episode_id: str, spec: TaskSpec, *, overwrite: bool = False) -> dict[str, Any]:
    """Cut one v1 episode at the v2 cutoff and write it to ``dst_dir``; returns its new metadata.

    Raises ``ValueError`` if the window is not full under ``spec`` (the cohort builder
    should already have excluded it) or if a destination file exists and ``overwrite`` is off.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    _check_destination(src_dir, dst_dir)
    src = episode_paths(src_dir, episode_id)
    meta = json.loads(src["meta"].read_text(encoding="utf-8"))
    values = np.load(src["values"])
    mask = np.load(src["mask"])
    new_values, new_mask, names, info = transform_episode(values, mask, meta, spec)
    if spec.require_full_window and not info["full_window"]:
        raise ValueError(f"episode {episode_id!r} is not a full window under the task spec")
    dst = episode_paths(dst_dir, episode_id)
    if not overwrite:
        existing = [str(path) for path in dst.values() if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite {existing}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    new_meta = build_v2_meta(meta, new_values, new_mask, names, info, spec)
    np.save(dst["values"], new_values)
    np.save(dst["mask"], new_mask)
    dst["meta"].write_text(json.dumps(new_meta, ensure_ascii=False), encoding="utf-8")
    return new_meta


def materialize_cohort(
    src_dir: str | Path,
    dst_dir: str | Path,
    episode_ids: Iterable[str],
    spec: TaskSpec,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize every listed episode and write ``shock_episode_index.json`` + a manifest.

    Failures (missing files, non-full windows, existing files) are collected, not raised,
    so one bad episode does not abort a long run; the manifest lists them.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    _check_destination(src_dir, dst_dir)
    metas: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for episode_id in episode_ids:
        try:
            metas.append(materialize_episode(src_dir, dst_dir, str(episode_id), spec, overwrite=overwrite))
        except (OSError, ValueError) as exc:
            failures.append({"episode_id": str(episode_id), "error": f"{type(exc).__name__}: {exc}"})
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "shock_episode_index.json").write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "task_spec": spec.to_dict(),
        "task_spec_sha256": spec_sha256(spec),
        "n_requested": len(metas) + len(failures),
        "n_written": len(metas),
        "n_failed": len(failures),
        "failures": failures,
        "positives": sum(1 for meta in metas if meta.get("episode_type") == "positive"),
        "controls": sum(1 for meta in metas if meta.get("episode_type") == "control"),
    }
    (dst_dir / "materialization_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
