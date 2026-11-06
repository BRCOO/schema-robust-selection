from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import json
import numpy as np

import pandas as pd

from .icu_eventizer import DEFAULT_INTERVENTION_FEATURES, DEFAULT_MISSINGNESS_FEATURES
from .mimic_iv_features import build_feature_matrix
from .mimic_iv_schema import get_feature_specs

CORE_OBSERVATION_FEATURES = {
    "HR",
    "RR",
    "SpO2",
    "SBP",
    "DBP",
    "MAP",
    "Temp",
}


def build_shock_episode_index(
    anchors: pd.DataFrame,
    icustays: pd.DataFrame,
    pre_hours: int = 12,
    max_control_windows_per_anchor: int = 1,
    control_window_gap_hours: int = 0,
    min_control_windows_per_anchor: int = 0,
    control_fallback_stride_hours: int = 0,
) -> pd.DataFrame:
    if anchors.empty:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "episode_type",
                "stay_id",
                "subject_id",
                "hadm_id",
                "anchor_kind",
                "anchor_time",
                "window_start",
                "window_end",
                "control_windows_requested",
                "control_windows_built",
            ]
        )

    window = timedelta(hours=int(pre_hours))
    control_window_cap = max(0, int(max_control_windows_per_anchor))
    min_control_target = max(0, int(min_control_windows_per_anchor))
    control_gap = timedelta(hours=max(0, int(control_window_gap_hours)))
    fallback_stride = timedelta(hours=max(0, int(control_fallback_stride_hours)))
    control_step = window + control_gap
    stay_frame = icustays.copy()
    stay_frame["stay_id"] = stay_frame["stay_id"].astype(int)
    stay_frame["intime"] = pd.to_datetime(stay_frame["intime"], errors="coerce")
    stay_frame["outtime"] = pd.to_datetime(stay_frame["outtime"], errors="coerce")
    stay_by_id = stay_frame.set_index("stay_id")

    anchor_frame = anchors.copy()
    anchor_frame["stay_id"] = anchor_frame["stay_id"].astype(int)
    anchor_frame["anchor_time"] = pd.to_datetime(anchor_frame["anchor_time"], errors="coerce")
    anchor_frame = anchor_frame.sort_values(["stay_id", "anchor_time"]).reset_index(drop=True)

    episodes: list[dict] = []
    for anchor in anchor_frame.itertuples(index=False):
        if anchor.stay_id not in stay_by_id.index:
            continue
        stay = stay_by_id.loc[anchor.stay_id]
        stay_start = pd.Timestamp(stay["intime"])
        stay_end = pd.Timestamp(stay["outtime"])
        anchor_time = pd.Timestamp(anchor.anchor_time)
        positive_start = max(stay_start, anchor_time - window)
        positive_end = min(anchor_time, stay_end)
        if positive_end <= positive_start:
            continue

        anchor_key = anchor_time.strftime("%Y%m%d%H%M%S")
        base = {
            "stay_id": int(anchor.stay_id),
            "subject_id": int(stay["subject_id"]) if pd.notna(stay["subject_id"]) else None,
            "hadm_id": int(stay["hadm_id"]) if pd.notna(stay["hadm_id"]) else None,
            "anchor_kind": anchor.anchor_kind,
            "anchor_time": anchor_time,
        }

        controls: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        control_window_keys: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
        if control_window_cap > 0 and positive_start - stay_start >= window:
            for control_idx in range(control_window_cap):
                control_end = positive_start - (control_step * control_idx)
                control_start = control_end - window
                if control_start < stay_start:
                    break
                window_key = (control_start, control_end)
                if window_key in control_window_keys:
                    continue
                controls.append(window_key)
                control_window_keys.add(window_key)

            if len(controls) < control_window_cap and fallback_stride > timedelta(0):
                fallback_shift = 1
                while len(controls) < control_window_cap:
                    control_end = positive_start - (fallback_stride * fallback_shift)
                    control_start = control_end - window
                    if control_start < stay_start:
                        break
                    window_key = (control_start, control_end)
                    fallback_shift += 1
                    if window_key in control_window_keys:
                        continue
                    controls.append(window_key)
                    control_window_keys.add(window_key)

        control_built = int(len(controls))
        if control_built < min_control_target:
            continue

        base_with_control_stats = {
            **base,
            "control_windows_requested": int(control_window_cap),
            "control_windows_built": int(control_built),
        }
        episodes.append(
            {
                **base_with_control_stats,
                "episode_id": f"stay_{anchor.stay_id}_{anchor_key}_positive",
                "episode_type": "positive",
                "window_start": positive_start,
                "window_end": positive_end,
            }
        )

        for control_idx, (control_start, control_end) in enumerate(controls):
            control_suffix = "" if control_idx == 0 else f"_{control_idx + 1}"
            episodes.append(
                {
                    **base_with_control_stats,
                    "episode_id": f"stay_{anchor.stay_id}_{anchor_key}_control{control_suffix}",
                    "episode_type": "control",
                    "window_start": control_start,
                    "window_end": control_end,
                }
            )

    return pd.DataFrame(episodes)


def _feature_fill_map(feature_names: list[str]) -> dict[str, str]:
    by_name = {spec.name: spec.fill for spec in get_feature_specs("core")}
    return {name: by_name.get(name, "interpolate") for name in feature_names}


def _normalize_observed_frame(
    observations: pd.DataFrame,
    feature_subset: set[str] | None = None,
) -> pd.DataFrame:
    frame = observations.copy()
    frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce")
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce")
    frame = frame.dropna(subset=["stay_id", "event_time"]).copy()
    if feature_subset is not None and "feature_name" in frame.columns:
        frame = frame[frame["feature_name"].isin(feature_subset)].copy()
    frame["stay_id"] = frame["stay_id"].astype(int)
    return frame


def _build_stay_observation_groups(observed: pd.DataFrame) -> dict[int, dict]:
    groups: dict[int, dict] = {}
    if observed.empty:
        return groups
    for stay_id, stay_rows in observed.groupby("stay_id", sort=False):
        sorted_rows = stay_rows.sort_values("event_time").reset_index(drop=True)
        groups[int(stay_id)] = {
            "rows": sorted_rows,
            "times": sorted_rows["event_time"].to_numpy(dtype="datetime64[ns]"),
            "features": sorted_rows["feature_name"].astype(str).to_numpy() if "feature_name" in sorted_rows.columns else np.array([]),
        }
    return groups


def _window_slice_indices(
    times: np.ndarray,
    window_start,
    window_end,
) -> tuple[int, int]:
    if times.size == 0:
        return (0, 0)
    start = np.datetime64(pd.Timestamp(window_start).to_datetime64())
    end = np.datetime64(pd.Timestamp(window_end).to_datetime64())
    left = int(np.searchsorted(times, start, side="left"))
    right = int(np.searchsorted(times, end, side="right"))
    return (left, right)


def filter_episode_index_by_observation_support(
    observations: pd.DataFrame,
    episode_index: pd.DataFrame,
    core_features: set[str] | None = None,
    min_core_observed_channels: int = 2,
    min_core_observed_points: int = 6,
) -> pd.DataFrame:
    if episode_index.empty:
        cols = list(episode_index.columns) + ["core_observed_channels", "core_observed_points"]
        return pd.DataFrame(columns=cols)

    core_set = set(core_features or CORE_OBSERVATION_FEATURES)
    observed_core = _normalize_observed_frame(observations, feature_subset=core_set)
    stay_groups = _build_stay_observation_groups(observed_core)

    kept: list[dict] = []
    for episode in episode_index.itertuples(index=False):
        stay_group = stay_groups.get(int(episode.stay_id))
        if not stay_group:
            continue
        left, right = _window_slice_indices(
            stay_group["times"],
            episode.window_start,
            episode.window_end,
        )
        points = int(max(0, right - left))
        if points <= 0:
            continue
        window_features = stay_group["features"][left:right]
        channels = int(pd.unique(window_features).size) if window_features.size else 0
        if channels < int(min_core_observed_channels) or points < int(min_core_observed_points):
            continue
        entry = dict(episode._asdict())
        entry["core_observed_channels"] = channels
        entry["core_observed_points"] = points
        kept.append(entry)

    if not kept:
        cols = list(episode_index.columns) + ["core_observed_channels", "core_observed_points"]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(kept)


def materialize_shock_episode_matrices(
    observations: pd.DataFrame,
    episode_index: pd.DataFrame,
    out_dir: Path | str,
    resample_minutes: int = 60,
    feature_names: list[str] | None = None,
) -> list[dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ordered_features = list(feature_names or [])
    fill_map = _feature_fill_map(ordered_features)
    intervention_features = [name for name in ordered_features if name in DEFAULT_INTERVENTION_FEATURES]
    missingness_features = [name for name in ordered_features if name in DEFAULT_MISSINGNESS_FEATURES]
    observed = _normalize_observed_frame(observations)
    stay_groups = _build_stay_observation_groups(observed)

    entries: list[dict] = []
    for episode in episode_index.itertuples(index=False):
        stay_group = stay_groups.get(int(episode.stay_id))
        if stay_group:
            left, right = _window_slice_indices(
                stay_group["times"],
                episode.window_start,
                episode.window_end,
            )
            rows = stay_group["rows"].iloc[left:right].copy()
        else:
            rows = observed.iloc[0:0].copy()
        X, meta = build_feature_matrix(
            rows,
            resample_minutes=resample_minutes,
            feature_names=ordered_features,
            window_start=episode.window_start,
            window_end=episode.window_end,
            fill_map=fill_map,
            include_missing_mask=True,
        )
        missing_mask = meta.pop("missing_mask")
        stem = f"shock_episode_{episode.episode_id}"
        missing_mask_name = f"{stem}_missing_mask.npy"
        meta.update(
            {
                "dataset": "mimic-iv-shock",
                "episode_id": episode.episode_id,
                "episode_type": episode.episode_type,
                "stay_id": int(episode.stay_id),
                "subject_id": int(episode.subject_id) if pd.notna(episode.subject_id) else None,
                "hadm_id": int(episode.hadm_id) if pd.notna(episode.hadm_id) else None,
                "anchor_kind": episode.anchor_kind,
                "anchor_time": pd.Timestamp(episode.anchor_time).isoformat(),
                "window_start": pd.Timestamp(episode.window_start).isoformat(),
                "window_end": pd.Timestamp(episode.window_end).isoformat(),
                "missing_mask_file": missing_mask_name,
                "intervention_features": intervention_features,
                "missingness_features": missingness_features,
                "observation_density_features": missingness_features,
            }
        )
        core_observed_channels = getattr(episode, "core_observed_channels", None)
        core_observed_points = getattr(episode, "core_observed_points", None)
        if core_observed_channels is not None and pd.notna(core_observed_channels):
            meta["core_observed_channels"] = int(core_observed_channels)
        if core_observed_points is not None and pd.notna(core_observed_points):
            meta["core_observed_points"] = int(core_observed_points)
        control_windows_requested = getattr(episode, "control_windows_requested", None)
        control_windows_built = getattr(episode, "control_windows_built", None)
        if control_windows_requested is not None and pd.notna(control_windows_requested):
            meta["control_windows_requested"] = int(control_windows_requested)
        if control_windows_built is not None and pd.notna(control_windows_built):
            meta["control_windows_built"] = int(control_windows_built)
        npy_path = out_path / f"{stem}.npy"
        meta_path = out_path / f"{stem}.json"
        np.save(npy_path, X)
        np.save(out_path / missing_mask_name, missing_mask)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        entries.append(meta)

    with open(out_path / "shock_episode_index.json", "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
    return entries
