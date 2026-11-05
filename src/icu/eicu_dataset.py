from pathlib import Path

import pandas as pd

from .mimic_iv_cohort import build_icu_cohort
from .mimic_iv_schema import get_feature_names
from .shock_episode_dataset import (
    build_shock_episode_index,
    filter_episode_index_by_observation_support,
    materialize_shock_episode_matrices,
)

EICU_BASE_TIME = pd.Timestamp("2000-01-01T00:00:00")

EICU_REQUIRED_FILES = (
    "patient.csv.gz",
    "vitalPeriodic.csv.gz",
    "vitalAperiodic.csv.gz",
    "lab.csv.gz",
    "infusionDrug.csv.gz",
    "respiratoryCharting.csv.gz",
    "intakeOutput.csv.gz",
)

VASOPRESSOR_KEYWORDS = (
    "norepinephrine",
    "noradrenaline",
    "epinephrine",
    "adrenaline",
    "vasopressin",
    "phenylephrine",
    "dopamine",
    "dobutamine",
)

SEDATION_KEYWORDS = (
    "propofol",
    "dexmedetomidine",
    "midazolam",
    "fentanyl",
)

VENTILATION_KEYWORDS = (
    "vent",
    "peep",
    "fio2",
    "ett",
    "endotracheal",
)


def detect_eicu_root(path: Path | str) -> Path:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"eICU path not found: {root}")
    if root.is_file():
        raise FileNotFoundError(f"unsupported eICU file path: {root}")

    if all((root / name).exists() for name in EICU_REQUIRED_FILES):
        return root

    for child in root.iterdir():
        if not child.is_dir():
            continue
        if all((child / name).exists() for name in EICU_REQUIRED_FILES):
            return child

    raise FileNotFoundError(f"could not detect eICU root under: {root}")


def _read_eicu_table(root: Path, name: str, usecols: list[str], chunksize: int = 200000):
    path = root / name
    if not path.exists():
        return
    header = pd.read_csv(path, nrows=0, compression="infer")
    selected_cols = [col for col in usecols if col in set(header.columns)]
    if not selected_cols:
        return
    for chunk in pd.read_csv(path, usecols=selected_cols, chunksize=chunksize, low_memory=False, compression="infer"):
        yield chunk


def load_eicu_icustays(root: Path) -> pd.DataFrame:
    chunks = list(
        _read_eicu_table(
            root,
            "patient.csv.gz",
            [
                "patientunitstayid",
                "patienthealthsystemstayid",
                "uniquepid",
                "unitadmitoffset",
                "unitdischargeoffset",
            ],
            chunksize=100000,
        )
    )
    if not chunks:
        raise FileNotFoundError("patient.csv.gz not found or empty")

    frame = pd.concat(chunks, ignore_index=True)
    if "patientunitstayid" not in frame.columns:
        raise ValueError("patient.csv.gz missing required column: patientunitstayid")
    if "unitdischargeoffset" not in frame.columns:
        raise ValueError("patient.csv.gz missing required column: unitdischargeoffset")

    frame["stay_id"] = pd.to_numeric(frame["patientunitstayid"], errors="coerce").astype("Int64")
    frame["hadm_id"] = pd.to_numeric(frame.get("patienthealthsystemstayid"), errors="coerce").astype("Int64")
    frame["subject_id"] = pd.to_numeric(frame.get("uniquepid"), errors="coerce").astype("Int64")
    if "unitadmitoffset" in frame.columns:
        frame["unitadmitoffset"] = pd.to_numeric(frame["unitadmitoffset"], errors="coerce")
    else:
        frame["unitadmitoffset"] = 0.0
    frame["unitdischargeoffset"] = pd.to_numeric(frame["unitdischargeoffset"], errors="coerce")

    frame = frame.dropna(subset=["stay_id", "unitdischargeoffset"]).copy()
    frame["stay_id"] = frame["stay_id"].astype(int)
    frame["hadm_id"] = frame["hadm_id"].fillna(frame["stay_id"]).astype(int)
    frame["subject_id"] = frame["subject_id"].fillna(frame["stay_id"]).astype(int)
    frame["unitadmitoffset"] = frame["unitadmitoffset"].fillna(0.0)
    frame["intime"] = EICU_BASE_TIME + pd.to_timedelta(frame["unitadmitoffset"], unit="m")
    frame["outtime"] = EICU_BASE_TIME + pd.to_timedelta(frame["unitadmitoffset"] + frame["unitdischargeoffset"], unit="m")
    frame = frame[frame["outtime"] > frame["intime"]].copy()
    return frame[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "unitadmitoffset"]]


def _build_stay_admit_offset_map(icustays: pd.DataFrame) -> dict[int, float]:
    return {
        int(row.stay_id): float(row.unitadmitoffset)
        for row in icustays[["stay_id", "unitadmitoffset"]].itertuples(index=False)
    }


def _offset_to_time(stay_offsets: dict[int, float], stay_id: int, offset_minutes) -> pd.Timestamp | None:
    if stay_id not in stay_offsets:
        return None
    if pd.isna(offset_minutes):
        return None
    return EICU_BASE_TIME + pd.to_timedelta(float(stay_offsets[stay_id]) + float(offset_minutes), unit="m")


def _append_row(rows: list[dict], stay_id: int, event_time, feature_name: str, value, agg: str, fill: str, source: str) -> None:
    if event_time is None or pd.isna(value):
        return
    rows.append(
        {
            "stay_id": int(stay_id),
            "event_time": pd.Timestamp(event_time),
            "feature_name": feature_name,
            "value": float(value),
            "agg": agg,
            "fill": fill,
            "source_table": source,
        }
    )


def _extract_vitals(root: Path, stay_ids: set[int], stay_offsets: dict[int, float], chunksize: int) -> list[dict]:
    rows: list[dict] = []
    periodic_map = {
        "heartrate": "HR",
        "respiration": "RR",
        "sao2": "SpO2",
        "systemicsystolic": "SBP",
        "systemicdiastolic": "DBP",
        "systemicmean": "MAP",
        "temperature": "Temp",
    }
    for chunk in _read_eicu_table(
        root,
        "vitalPeriodic.csv.gz",
        ["patientunitstayid", "observationoffset", *periodic_map.keys()],
        chunksize=chunksize,
    ):
        frame = chunk.copy()
        for col in periodic_map:
            if col not in frame.columns:
                frame[col] = pd.NA
        frame["patientunitstayid"] = pd.to_numeric(frame["patientunitstayid"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["patientunitstayid", "observationoffset"]).copy()
        frame["patientunitstayid"] = frame["patientunitstayid"].astype(int)
        frame = frame[frame["patientunitstayid"].isin(stay_ids)]
        if frame.empty:
            continue
        for entry in frame.itertuples(index=False):
            stay_id = int(entry.patientunitstayid)
            event_time = _offset_to_time(stay_offsets, stay_id, entry.observationoffset)
            for source_col, feature_name in periodic_map.items():
                _append_row(rows, stay_id, event_time, feature_name, getattr(entry, source_col), "mean", "interpolate", "vitalPeriodic")

    aperiodic_map = {
        "noninvasivesystolic": "SBP",
        "noninvasivediastolic": "DBP",
        "noninvasivemean": "MAP",
    }
    for chunk in _read_eicu_table(
        root,
        "vitalAperiodic.csv.gz",
        ["patientunitstayid", "observationoffset", *aperiodic_map.keys()],
        chunksize=chunksize,
    ):
        frame = chunk.copy()
        for col in aperiodic_map:
            if col not in frame.columns:
                frame[col] = pd.NA
        frame["patientunitstayid"] = pd.to_numeric(frame["patientunitstayid"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["patientunitstayid", "observationoffset"]).copy()
        frame["patientunitstayid"] = frame["patientunitstayid"].astype(int)
        frame = frame[frame["patientunitstayid"].isin(stay_ids)]
        if frame.empty:
            continue
        for entry in frame.itertuples(index=False):
            stay_id = int(entry.patientunitstayid)
            event_time = _offset_to_time(stay_offsets, stay_id, entry.observationoffset)
            for source_col, feature_name in aperiodic_map.items():
                _append_row(rows, stay_id, event_time, feature_name, getattr(entry, source_col), "mean", "interpolate", "vitalAperiodic")
    return rows


def _extract_labs(root: Path, stay_ids: set[int], stay_offsets: dict[int, float], chunksize: int) -> list[dict]:
    rows: list[dict] = []

    def map_lab_feature(name: str) -> str | None:
        text = str(name or "").strip().lower()
        if "lactate" in text:
            return "Lactate"
        if text == "ph" or text.startswith("ph "):
            return "pH"
        if "creatinine" in text:
            return "Creatinine"
        if "glucose" in text:
            return "Glucose"
        if "bilirubin" in text and "total" in text:
            return "Bilirubin_Total"
        if "wbc" in text or "white blood" in text:
            return "WBC"
        if "platelet" in text:
            return "Platelets"
        return None

    for chunk in _read_eicu_table(
        root,
        "lab.csv.gz",
        ["patientunitstayid", "labresultoffset", "labname", "labresult"],
        chunksize=chunksize,
    ):
        frame = chunk.copy()
        for col in ("labresultoffset", "labname", "labresult"):
            if col not in frame.columns:
                frame[col] = pd.NA
        frame["patientunitstayid"] = pd.to_numeric(frame["patientunitstayid"], errors="coerce").astype("Int64")
        frame["labresult"] = pd.to_numeric(frame["labresult"], errors="coerce")
        frame = frame.dropna(subset=["patientunitstayid", "labresultoffset", "labresult"]).copy()
        frame["patientunitstayid"] = frame["patientunitstayid"].astype(int)
        frame = frame[frame["patientunitstayid"].isin(stay_ids)]
        if frame.empty:
            continue
        for entry in frame.itertuples(index=False):
            stay_id = int(entry.patientunitstayid)
            feature_name = map_lab_feature(entry.labname)
            if not feature_name:
                continue
            event_time = _offset_to_time(stay_offsets, stay_id, entry.labresultoffset)
            _append_row(rows, stay_id, event_time, feature_name, entry.labresult, "mean", "interpolate", "lab")
    return rows


def _classify_infusion(drugname: str) -> tuple[bool, bool]:
    text = str(drugname or "").strip().lower()
    is_pressor = any(token in text for token in VASOPRESSOR_KEYWORDS)
    is_sedation = any(token in text for token in SEDATION_KEYWORDS)
    return is_pressor, is_sedation


def _collect_vasopressor_stay_ids(root: Path, chunksize: int = 200000) -> set[int]:
    hit_ids: set[int] = set()
    for chunk in _read_eicu_table(
        root,
        "infusionDrug.csv.gz",
        ["patientunitstayid", "drugname"],
        chunksize=chunksize,
    ):
        frame = chunk.copy()
        if "patientunitstayid" not in frame.columns:
            continue
        if "drugname" not in frame.columns:
            frame["drugname"] = ""
        frame["patientunitstayid"] = pd.to_numeric(frame["patientunitstayid"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["patientunitstayid"]).copy()
        if frame.empty:
            continue
        text = frame["drugname"].astype(str).str.lower()
        mask = False
        for token in VASOPRESSOR_KEYWORDS:
            mask = mask | text.str.contains(token, na=False)
        if mask is False:
            continue
        hits = frame.loc[mask, "patientunitstayid"].astype(int).tolist()
        hit_ids.update(hits)
    return hit_ids


def _select_anchor_priority_cohort(root: Path, cohort: pd.DataFrame, cohort_limit: int, chunksize: int) -> pd.DataFrame:
    limit = int(cohort_limit)
    if limit <= 0 or len(cohort) <= limit:
        return cohort

    vasopressor_ids = _collect_vasopressor_stay_ids(root, chunksize=chunksize)
    if not vasopressor_ids:
        return cohort.head(limit).reset_index(drop=True)

    frame = cohort.copy()
    frame["stay_id"] = frame["stay_id"].astype(int)
    prioritized = frame[frame["stay_id"].isin(vasopressor_ids)].copy()
    if len(prioritized) >= limit:
        return prioritized.head(limit).reset_index(drop=True)

    remaining = frame[~frame["stay_id"].isin(set(prioritized["stay_id"].tolist()))].copy()
    selected = pd.concat([prioritized, remaining.head(limit - len(prioritized))], ignore_index=True)
    return selected.reset_index(drop=True)


def _extract_infusion_observations(root: Path, stay_ids: set[int], stay_offsets: dict[int, float], chunksize: int) -> tuple[list[dict], pd.DataFrame]:
    rows: list[dict] = []
    anchor_rows: list[dict] = []
    for chunk in _read_eicu_table(
        root,
        "infusionDrug.csv.gz",
        ["patientunitstayid", "infusionoffset", "drugname", "drugrate", "infusionrate", "drugamount"],
        chunksize=chunksize,
    ):
        frame = chunk.copy()
        for col in ("drugname", "drugrate", "infusionrate", "drugamount"):
            if col not in frame.columns:
                frame[col] = pd.NA
        frame["patientunitstayid"] = pd.to_numeric(frame["patientunitstayid"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["patientunitstayid", "infusionoffset"]).copy()
        frame["patientunitstayid"] = frame["patientunitstayid"].astype(int)
        frame = frame[frame["patientunitstayid"].isin(stay_ids)]
        if frame.empty:
            continue
        for entry in frame.itertuples(index=False):
            stay_id = int(entry.patientunitstayid)
            event_time = _offset_to_time(stay_offsets, stay_id, entry.infusionoffset)
            is_pressor, is_sedation = _classify_infusion(entry.drugname)

            def _first_numeric(*candidates) -> float:
                for candidate in candidates:
                    value = pd.to_numeric(candidate, errors="coerce")
                    if pd.notna(value):
                        return float(value)
                return 1.0

            numeric = _first_numeric(entry.drugrate, entry.infusionrate, entry.drugamount)
            if is_pressor:
                _append_row(rows, stay_id, event_time, "vasopressor_on", 1.0, "max", "zero", "infusionDrug")
                anchor_rows.append({"stay_id": stay_id, "event_time": event_time, "rate": max(0.0, numeric)})
            if is_sedation:
                _append_row(rows, stay_id, event_time, "sedation_on", 1.0, "max", "zero", "infusionDrug")
    anchors = pd.DataFrame(anchor_rows)
    return rows, anchors


def _extract_ventilation(root: Path, stay_ids: set[int], stay_offsets: dict[int, float], chunksize: int) -> list[dict]:
    rows: list[dict] = []
    for chunk in _read_eicu_table(
        root,
        "respiratoryCharting.csv.gz",
        ["patientunitstayid", "respchartoffset", "respcharttypecat", "respchartvaluelabel", "respchartvalue"],
        chunksize=chunksize,
    ):
        frame = chunk.copy()
        for col in ("respcharttypecat", "respchartvaluelabel", "respchartvalue"):
            if col not in frame.columns:
                frame[col] = pd.NA
        frame["patientunitstayid"] = pd.to_numeric(frame["patientunitstayid"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["patientunitstayid", "respchartoffset"]).copy()
        frame["patientunitstayid"] = frame["patientunitstayid"].astype(int)
        frame = frame[frame["patientunitstayid"].isin(stay_ids)]
        if frame.empty:
            continue
        for entry in frame.itertuples(index=False):
            stay_id = int(entry.patientunitstayid)
            text = " ".join(
                [
                    str(getattr(entry, "respcharttypecat", "") or ""),
                    str(getattr(entry, "respchartvaluelabel", "") or ""),
                    str(getattr(entry, "respchartvalue", "") or ""),
                ]
            ).lower()
            if not any(token in text for token in VENTILATION_KEYWORDS):
                continue
            event_time = _offset_to_time(stay_offsets, stay_id, entry.respchartoffset)
            _append_row(rows, stay_id, event_time, "ventilation_on", 1.0, "max", "zero", "respiratoryCharting")
    return rows


def _extract_urine_output(root: Path, stay_ids: set[int], stay_offsets: dict[int, float], chunksize: int) -> list[dict]:
    rows: list[dict] = []
    for chunk in _read_eicu_table(
        root,
        "intakeOutput.csv.gz",
        ["patientunitstayid", "intakeoutputoffset", "outputtotal", "celllabel", "cellvaluenumeric"],
        chunksize=chunksize,
    ):
        frame = chunk.copy()
        for col in ("outputtotal", "celllabel", "cellvaluenumeric"):
            if col not in frame.columns:
                frame[col] = pd.NA
        frame["patientunitstayid"] = pd.to_numeric(frame["patientunitstayid"], errors="coerce").astype("Int64")
        frame["outputtotal"] = pd.to_numeric(frame["outputtotal"], errors="coerce")
        frame["cellvaluenumeric"] = pd.to_numeric(frame["cellvaluenumeric"], errors="coerce")
        frame = frame.dropna(subset=["patientunitstayid", "intakeoutputoffset"]).copy()
        frame["patientunitstayid"] = frame["patientunitstayid"].astype(int)
        frame = frame[frame["patientunitstayid"].isin(stay_ids)]
        if frame.empty:
            continue
        for entry in frame.itertuples(index=False):
            stay_id = int(entry.patientunitstayid)
            event_time = _offset_to_time(stay_offsets, stay_id, entry.intakeoutputoffset)
            urine_value = entry.outputtotal
            if pd.isna(urine_value):
                label_text = str(getattr(entry, "celllabel", "") or "").lower()
                if "urine" not in label_text:
                    continue
                urine_value = entry.cellvaluenumeric
            _append_row(rows, stay_id, event_time, "urine_output", urine_value, "sum", "zero", "intakeOutput")
    return rows


def collect_eicu_feature_observations(root: Path, cohort: pd.DataFrame, chunksize: int = 200000) -> tuple[pd.DataFrame, pd.DataFrame]:
    stay_ids = set(cohort["stay_id"].astype(int).tolist())
    stay_offsets = _build_stay_admit_offset_map(cohort)

    rows: list[dict] = []
    rows.extend(_extract_vitals(root, stay_ids, stay_offsets, chunksize=chunksize))
    rows.extend(_extract_labs(root, stay_ids, stay_offsets, chunksize=chunksize))

    infusion_rows, anchor_seed = _extract_infusion_observations(root, stay_ids, stay_offsets, chunksize=chunksize)
    rows.extend(infusion_rows)
    rows.extend(_extract_ventilation(root, stay_ids, stay_offsets, chunksize=chunksize))
    rows.extend(_extract_urine_output(root, stay_ids, stay_offsets, chunksize=chunksize))

    if not rows:
        empty_rows = pd.DataFrame(columns=["stay_id", "event_time", "feature_name", "value", "agg", "fill", "source_table"])
        empty_anchors = pd.DataFrame(columns=["stay_id", "anchor_time", "anchor_kind", "previous_rate", "current_rate"])
        return empty_rows, empty_anchors

    observations = pd.DataFrame(rows)
    observations = observations.sort_values(["stay_id", "event_time", "feature_name"]).reset_index(drop=True)
    anchors = extract_eicu_vasopressor_anchors(anchor_seed)
    return observations, anchors


def extract_eicu_vasopressor_anchors(seed_rows: pd.DataFrame, escalation_ratio: float = 1.5) -> pd.DataFrame:
    if seed_rows.empty:
        return pd.DataFrame(columns=["stay_id", "anchor_time", "anchor_kind", "previous_rate", "current_rate"])

    frame = seed_rows.copy()
    frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce").astype("Int64")
    frame["rate"] = pd.to_numeric(frame["rate"], errors="coerce")
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce")
    frame = frame.dropna(subset=["stay_id", "event_time", "rate"]).copy()
    frame["stay_id"] = frame["stay_id"].astype(int)
    frame = (
        frame.groupby(["stay_id", "event_time"], as_index=False)["rate"]
        .sum()
        .sort_values(["stay_id", "event_time"])
        .reset_index(drop=True)
    )

    anchors: list[dict] = []
    for stay_id, group in frame.groupby("stay_id", sort=True):
        previous_rate = 0.0
        for row in group.itertuples(index=False):
            current_rate = float(row.rate)
            if current_rate <= 0.0:
                previous_rate = 0.0
                continue
            anchor_kind = None
            if previous_rate <= 0.0:
                anchor_kind = "initiation"
            elif current_rate >= previous_rate * float(escalation_ratio):
                anchor_kind = "escalation"
            if anchor_kind:
                anchors.append(
                    {
                        "stay_id": int(stay_id),
                        "anchor_time": pd.Timestamp(row.event_time),
                        "anchor_kind": anchor_kind,
                        "previous_rate": float(previous_rate),
                        "current_rate": current_rate,
                    }
                )
            previous_rate = current_rate

    return pd.DataFrame(anchors)


def prepare_eicu_dataset(
    raw_dir: Path | str,
    out_dir: Path | str,
    cohort_limit: int = 0,
    resample_minutes: int = 60,
    eicu_root: str | None = None,
    min_hours: int = 12,
    mode: str = "shock-episodes",
    shock_pre_hours: int = 12,
    shock_min_core_channels: int = 2,
    shock_min_core_points: int = 6,
    shock_control_windows_per_anchor: int = 1,
    shock_control_window_gap_hours: int = 0,
    shock_min_controls_per_anchor: int = 0,
    shock_control_fallback_stride_hours: int = 0,
    chunksize: int = 200000,
) -> list[dict]:
    if mode != "shock-episodes":
        raise ValueError("eICU preparation currently supports mode='shock-episodes' only")

    source = Path(eicu_root) if eicu_root else Path(raw_dir)
    root = detect_eicu_root(source)
    icustays = load_eicu_icustays(root)
    cohort_full = build_icu_cohort(icustays, min_hours=min_hours, cohort_limit=0)
    cohort = _select_anchor_priority_cohort(root, cohort_full, cohort_limit=cohort_limit, chunksize=chunksize)

    observations, anchors = collect_eicu_feature_observations(root, cohort, chunksize=chunksize)
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
        return materialize_shock_episode_matrices(
            observations,
            empty_episode_index,
            out_dir,
            resample_minutes=resample_minutes,
            feature_names=get_feature_names("core"),
        )

    anchor_stays = set(pd.to_numeric(anchors["stay_id"], errors="coerce").dropna().astype(int).tolist())
    cohort_anchor = cohort[cohort["stay_id"].astype(int).isin(anchor_stays)].copy()
    observations_anchor = observations[observations["stay_id"].astype(int).isin(anchor_stays)].copy()

    episode_index = build_shock_episode_index(
        anchors,
        cohort_anchor,
        pre_hours=shock_pre_hours,
        max_control_windows_per_anchor=shock_control_windows_per_anchor,
        control_window_gap_hours=shock_control_window_gap_hours,
        min_control_windows_per_anchor=shock_min_controls_per_anchor,
        control_fallback_stride_hours=shock_control_fallback_stride_hours,
    )
    episode_index = filter_episode_index_by_observation_support(
        observations_anchor,
        episode_index,
        min_core_observed_channels=shock_min_core_channels,
        min_core_observed_points=shock_min_core_points,
    )
    return materialize_shock_episode_matrices(
        observations_anchor,
        episode_index,
        out_dir,
        resample_minutes=resample_minutes,
        feature_names=get_feature_names("core"),
    )
