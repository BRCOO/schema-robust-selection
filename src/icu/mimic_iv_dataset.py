from collections import defaultdict
from pathlib import Path

import json
import numpy as np
import pandas as pd

from .mimic_iv_cohort import build_icu_cohort
from .mimic_iv_features import build_feature_matrix
from .mimic_iv_io import detect_mimic_root, iter_table_chunks, list_required_tables
from .mimic_iv_io import (
    _prepare_cohort_for_lab_join,
    normalize_chart_chunk,
    normalize_input_chunk,
    normalize_lab_chunk,
    normalize_output_chunk,
)
from .mimic_iv_schema import get_feature_names, get_feature_specs
from .shock_anchor_builder import extract_vasopressor_anchors
from .shock_episode_dataset import (
    build_shock_episode_index,
    filter_episode_index_by_observation_support,
    materialize_shock_episode_matrices,
)

TABLE_USECOLS = {
    "icu/icustays.csv.gz": ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit", "last_careunit"],
    "icu/chartevents.csv.gz": ["stay_id", "itemid", "charttime", "valuenum", "value"],
    "hosp/labevents.csv.gz": ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "value"],
    "icu/inputevents.csv.gz": ["stay_id", "itemid", "starttime", "rate", "amount", "statusdescription"],
    "icu/outputevents.csv.gz": ["stay_id", "itemid", "charttime", "value"],
}


def load_icustays(root: Path | str) -> pd.DataFrame:
    chunks = list(
        iter_table_chunks(
            root,
            "icu/icustays.csv.gz",
            usecols=TABLE_USECOLS["icu/icustays.csv.gz"],
            chunksize=50000,
        )
    )
    if not chunks:
        raise FileNotFoundError("icustays.csv.gz not found or empty")
    return pd.concat(chunks, ignore_index=True)


def collect_inputevents(root: Path | str, cohort: pd.DataFrame, itemids: set[int], chunksize: int = 100000) -> pd.DataFrame:
    stay_ids = set(cohort["stay_id"].astype(int).tolist())
    outputs = []
    for chunk in iter_table_chunks(
        root,
        "icu/inputevents.csv.gz",
        usecols=TABLE_USECOLS["icu/inputevents.csv.gz"],
        chunksize=chunksize,
    ):
        frame = chunk.copy()
        frame = frame[frame["itemid"].isin(itemids)]
        if "stay_id" in frame.columns:
            frame = frame[frame["stay_id"].isin(stay_ids)]
        if frame.empty:
            continue
        outputs.append(frame)
    if not outputs:
        return pd.DataFrame(columns=["stay_id", "itemid", "starttime", "rate", "amount", "statusdescription"])
    data = pd.concat(outputs, ignore_index=True)
    data["stay_id"] = pd.to_numeric(data["stay_id"], errors="coerce").astype("Int64")
    data = data.dropna(subset=["stay_id", "starttime"]).copy()
    data["stay_id"] = data["stay_id"].astype(int)
    return data


def _feature_rows_from_normalized(normalized: pd.DataFrame, specs) -> pd.DataFrame:
    rows = []
    for spec in specs:
        block = normalized[normalized["itemid"].isin(spec.itemids)].copy()
        if block.empty:
            continue
        if spec.transform == "temperature_c":
            mask_f = block["itemid"].astype(int) == 223761
            block.loc[mask_f, "value"] = (block.loc[mask_f, "value"] - 32.0) / 1.8
        if spec.kind == "presence":
            block["value"] = 1.0
        block["feature_name"] = spec.name
        block["agg"] = spec.agg
        block["fill"] = spec.fill
        block["source_table"] = spec.source_table
        rows.append(block[["stay_id", "event_time", "feature_name", "value", "agg", "fill", "source_table"]])
    if not rows:
        return pd.DataFrame(columns=["stay_id", "event_time", "feature_name", "value", "agg", "fill", "source_table"])
    return pd.concat(rows, ignore_index=True)


def collect_feature_observations(root: Path | str, cohort: pd.DataFrame, feature_set: str = "core", chunksize: int = 100000) -> pd.DataFrame:
    specs = get_feature_specs(feature_set)
    by_table = defaultdict(list)
    for spec in specs:
        by_table[spec.source_table].append(spec)

    stay_ids = set(cohort["stay_id"].astype(int).tolist())
    lab_cohort_lookup = _prepare_cohort_for_lab_join(cohort) if "hosp/labevents.csv.gz" in by_table else None
    outputs = []
    for table_name, table_specs in by_table.items():
        allowed = {itemid for spec in table_specs for itemid in spec.itemids}
        table_usecols = TABLE_USECOLS.get(table_name)
        if table_name == "icu/chartevents.csv.gz":
            for chunk in iter_table_chunks(root, table_name, usecols=table_usecols, chunksize=chunksize):
                norm = normalize_chart_chunk(chunk, allowed_itemids=allowed, stay_ids=stay_ids)
                rows = _feature_rows_from_normalized(norm, table_specs)
                if not rows.empty:
                    outputs.append(rows)
        elif table_name == "hosp/labevents.csv.gz":
            for chunk in iter_table_chunks(root, table_name, usecols=table_usecols, chunksize=chunksize):
                norm = normalize_lab_chunk(
                    chunk,
                    cohort=None,
                    allowed_itemids=allowed,
                    cohort_lookup=lab_cohort_lookup,
                )
                rows = _feature_rows_from_normalized(norm, table_specs)
                if not rows.empty:
                    outputs.append(rows)
        elif table_name == "icu/inputevents.csv.gz":
            for chunk in iter_table_chunks(root, table_name, usecols=table_usecols, chunksize=chunksize):
                norm = normalize_input_chunk(chunk, allowed_itemids=allowed, stay_ids=stay_ids)
                rows = _feature_rows_from_normalized(norm, table_specs)
                if not rows.empty:
                    outputs.append(rows)
        elif table_name == "icu/outputevents.csv.gz":
            for chunk in iter_table_chunks(root, table_name, usecols=table_usecols, chunksize=chunksize):
                norm = normalize_output_chunk(chunk, allowed_itemids=allowed, stay_ids=stay_ids)
                rows = _feature_rows_from_normalized(norm, table_specs)
                if not rows.empty:
                    outputs.append(rows)
    if not outputs:
        return pd.DataFrame(columns=["stay_id", "event_time", "feature_name", "value", "agg", "fill", "source_table"])
    data = pd.concat(outputs, ignore_index=True)
    data["stay_id"] = data["stay_id"].astype(int)
    data = data.sort_values(["stay_id", "event_time", "feature_name"]).reset_index(drop=True)
    return data


def write_stay_matrices(observations: pd.DataFrame, cohort: pd.DataFrame, out_dir: Path | str, resample_minutes: int = 60, feature_set: str = "core") -> list[dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    feature_names = get_feature_names(feature_set)
    entries = []
    for _, stay in cohort.iterrows():
        stay_id = int(stay["stay_id"])
        rows = observations[observations["stay_id"] == stay_id].copy()
        X, meta = build_feature_matrix(rows, resample_minutes=resample_minutes, feature_names=feature_names)
        meta.update({
            "dataset": "mimic-iv",
            "stay_id": stay_id,
            "subject_id": int(stay["subject_id"]) if "subject_id" in stay and pd.notna(stay["subject_id"]) else None,
            "hadm_id": int(stay["hadm_id"]) if pd.notna(stay["hadm_id"]) else None,
            "intime": pd.to_datetime(stay["intime"]).isoformat(),
            "outtime": pd.to_datetime(stay["outtime"]).isoformat(),
        })
        stem = f"mimic_iv_stay_{stay_id}"
        np.save(out_path / f"{stem}.npy", X)
        with open(out_path / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        entries.append(meta)
    with open(out_path / "mimic_iv_index.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    return entries


def prepare_mimic_iv_shock_dataset(
    raw_dir: Path | str,
    out_dir: Path | str,
    cohort_limit: int = 0,
    resample_minutes: int = 60,
    mimic_root: str | None = None,
    feature_set: str = "core",
    min_hours: int = 12,
    shock_pre_hours: int = 12,
    shock_min_core_channels: int = 2,
    shock_min_core_points: int = 6,
    shock_control_windows_per_anchor: int = 1,
    shock_control_window_gap_hours: int = 0,
    shock_min_controls_per_anchor: int = 0,
    shock_control_fallback_stride_hours: int = 0,
    chunksize: int = 200000,
) -> list[dict]:
    source = Path(mimic_root) if mimic_root else Path(raw_dir)
    root = detect_mimic_root(source)
    icustays = load_icustays(root)
    cohort = build_icu_cohort(icustays, min_hours=min_hours, cohort_limit=cohort_limit)
    vasopressor_ids = set()
    for spec in get_feature_specs(feature_set):
        if spec.name == "vasopressor_on":
            vasopressor_ids.update(spec.itemids)
    inputevents = collect_inputevents(root, cohort, vasopressor_ids, chunksize=chunksize)
    anchors = extract_vasopressor_anchors(inputevents)
    if anchors.empty:
        empty_episode_index = build_shock_episode_index(
            anchors,
            cohort,
            pre_hours=shock_pre_hours,
            max_control_windows_per_anchor=shock_control_windows_per_anchor,
            control_window_gap_hours=shock_control_window_gap_hours,
            min_control_windows_per_anchor=shock_min_controls_per_anchor,
            control_fallback_stride_hours=shock_control_fallback_stride_hours,
        )
        empty_observations = pd.DataFrame(
            columns=["stay_id", "event_time", "feature_name", "value", "agg", "fill", "source_table"]
        )
        return materialize_shock_episode_matrices(
            empty_observations,
            empty_episode_index,
            out_dir,
            resample_minutes=resample_minutes,
            feature_names=get_feature_names(feature_set),
        )
    anchor_stay_ids = set(pd.to_numeric(anchors["stay_id"], errors="coerce").dropna().astype(int).tolist())
    cohort_stay_ids = pd.to_numeric(cohort["stay_id"], errors="coerce")
    cohort_for_observations = cohort[cohort_stay_ids.isin(anchor_stay_ids)].copy()
    observations = collect_feature_observations(
        root,
        cohort_for_observations,
        feature_set=feature_set,
        chunksize=chunksize,
    )
    episode_index = build_shock_episode_index(
        anchors,
        cohort_for_observations,
        pre_hours=shock_pre_hours,
        max_control_windows_per_anchor=shock_control_windows_per_anchor,
        control_window_gap_hours=shock_control_window_gap_hours,
        min_control_windows_per_anchor=shock_min_controls_per_anchor,
        control_fallback_stride_hours=shock_control_fallback_stride_hours,
    )
    episode_index = filter_episode_index_by_observation_support(
        observations,
        episode_index,
        min_core_observed_channels=shock_min_core_channels,
        min_core_observed_points=shock_min_core_points,
    )
    return materialize_shock_episode_matrices(
        observations,
        episode_index,
        out_dir,
        resample_minutes=resample_minutes,
        feature_names=get_feature_names(feature_set),
    )


def prepare_mimic_iv_dataset(
    raw_dir: Path | str,
    out_dir: Path | str,
    cohort_limit: int = 0,
    resample_minutes: int = 60,
    mimic_root: str | None = None,
    feature_set: str = "core",
    min_hours: int = 12,
    chunksize: int = 200000,
    mode: str = "stay",
    shock_pre_hours: int = 12,
    shock_min_core_channels: int = 2,
    shock_min_core_points: int = 6,
    shock_control_windows_per_anchor: int = 1,
    shock_control_window_gap_hours: int = 0,
    shock_min_controls_per_anchor: int = 0,
    shock_control_fallback_stride_hours: int = 0,
) -> list[dict]:
    if mode == "shock-episodes":
        return prepare_mimic_iv_shock_dataset(
            raw_dir=raw_dir,
            out_dir=out_dir,
            cohort_limit=cohort_limit,
            resample_minutes=resample_minutes,
            mimic_root=mimic_root,
            feature_set=feature_set,
            min_hours=min_hours,
            shock_pre_hours=shock_pre_hours,
            shock_min_core_channels=shock_min_core_channels,
            shock_min_core_points=shock_min_core_points,
            shock_control_windows_per_anchor=shock_control_windows_per_anchor,
            shock_control_window_gap_hours=shock_control_window_gap_hours,
            shock_min_controls_per_anchor=shock_min_controls_per_anchor,
            shock_control_fallback_stride_hours=shock_control_fallback_stride_hours,
            chunksize=chunksize,
        )
    source = Path(mimic_root) if mimic_root else Path(raw_dir)
    root = detect_mimic_root(source)
    icustays = load_icustays(root)
    cohort = build_icu_cohort(icustays, min_hours=min_hours, cohort_limit=cohort_limit)
    observations = collect_feature_observations(root, cohort, feature_set=feature_set, chunksize=chunksize)
    return write_stay_matrices(observations, cohort, out_dir, resample_minutes=resample_minutes, feature_set=feature_set)
