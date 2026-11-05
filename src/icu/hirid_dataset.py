from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd

from .mimic_iv_cohort import build_icu_cohort
from .mimic_iv_schema import get_feature_names, get_feature_specs
from .shock_anchor_builder import extract_vasopressor_anchors
from .shock_episode_dataset import (
    build_shock_episode_index,
    filter_episode_index_by_observation_support,
    materialize_shock_episode_matrices,
)

HIRID_STAGE_DIRS = ("merged_stage", "imputed_stage", "raw_stage")
HIRID_BASE_TIME = pd.Timestamp("2000-01-01T00:00:00")
CORE_FEATURE_NAMES = set(get_feature_names("core"))

_FEATURE_META = {spec.name: spec for spec in get_feature_specs("core")}
_NORMALIZED_FEATURE_ALIASES = {
    "hr": "HR",
    "heartrate": "HR",
    "heart_rate": "HR",
    "rr": "RR",
    "respiration": "RR",
    "respiratoryrate": "RR",
    "spo2": "SpO2",
    "sao2": "SpO2",
    "sbp": "SBP",
    "systolicbp": "SBP",
    "systolicbpinvasive": "SBP",
    "invasivesystolicarterialpressure": "SBP",
    "dbp": "DBP",
    "diastolicbp": "DBP",
    "diastolicbpinvasive": "DBP",
    "invasivediastolicarterialpressure": "DBP",
    "map": "MAP",
    "mapinvasive": "MAP",
    "invasivemeanarterialpressure": "MAP",
    "noninvasivemeanarterialpressure": "MAP",
    "meanarterialpressure": "MAP",
    "temp": "Temp",
    "temperature": "Temp",
    "corebodytemperature": "Temp",
    "lactate": "Lactate",
    "lactatearterial": "Lactate",
    "lactatevenous": "Lactate",
    "ph": "pH",
    "creatinine": "Creatinine",
    "glucose": "Glucose",
    "serumglucose": "Glucose",
    "bilirubin_total": "Bilirubin_Total",
    "bilirubintotal": "Bilirubin_Total",
    "wbc": "WBC",
    "platelets": "Platelets",
    "vasopressor_on": "vasopressor_on",
    "vasopressoron": "vasopressor_on",
    "vasopressor": "vasopressor_on",
    "sedation_on": "sedation_on",
    "sedationon": "sedation_on",
    "ventilation_on": "ventilation_on",
    "ventilationon": "ventilation_on",
    "urine_output": "urine_output",
    "urineoutput": "urine_output",
}

_HIRID_REFERENCE_FILE_CANDIDATES = (
    "hirid_variable_reference_preprocessed.csv",
    "hirid_variable_reference.csv",
)
_HIRID_VASOPRESSOR_NAME_TOKENS = (
    "noradren",
    "norepi",
    "epineph",
    "adrenalin",
    "dopamine",
    "vasopressin",
    "phenylephrine",
    "dobutamine",
)
_HIRID_SEDATION_NAME_TOKENS = (
    "midazolam",
    "propofol",
    "dexmedetomidine",
    "fentanyl",
    "sufentanil",
    "remifentanil",
    "morphine",
)
_HIRID_INTERVENTION_FEATURES = {"vasopressor_on", "sedation_on"}
_HIRID_UNIT_HINTS = {
    "HR": ("/min", "bpm"),
    "RR": ("/min", "rpm"),
    "SpO2": ("%",),
    "SBP": ("mmhg",),
    "DBP": ("mmhg",),
    "MAP": ("mmhg",),
    "Temp": ("c", "degc", "fahrenheit", "f"),
    "Lactate": ("mmol", "mg/dl"),
    "pH": ("ph",),
    "Creatinine": ("mg/dl", "umol", "µmol"),
    "Glucose": ("mmol", "mg/dl"),
    "Bilirubin_Total": ("umol", "µmol", "mg/dl"),
    "WBC": ("10^9/l", "g/l", "k/ul"),
    "Platelets": ("10^9/l", "k/ul"),
    "urine_output": ("ml", "mL", "cc"),
}
_HIRID_PHARMA_UNIT_TOKENS = ("mg", "mcg", "ug", "ml", "dose", "rate", "/min", "/h", "iu", "u")

for feature_name in CORE_FEATURE_NAMES:
    _NORMALIZED_FEATURE_ALIASES.setdefault(feature_name.lower(), feature_name)


def _normalize_feature_token(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch == "_")


def _canonicalize_variable_id_token(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return str(numeric)
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def _infer_feature_name_from_description(name: str | None) -> str | None:
    if name is None:
        return None
    text = str(name).strip().lower()
    if not text:
        return None
    tokens = set(re.findall(r"[a-z0-9]+", text))
    compact = "".join(ch for ch in text if ch.isalnum())

    if "heart" in tokens and "rate" in tokens:
        return "HR"
    if "respiratory" in tokens and "rate" in tokens:
        return "RR"
    if ("oxygen" in tokens and "saturation" in tokens) or "spo2" in compact:
        return "SpO2"
    if "pulmonary" not in tokens and "systolic" in tokens and "pressure" in tokens:
        return "SBP"
    if "pulmonary" not in tokens and "diastolic" in tokens and "pressure" in tokens:
        return "DBP"
    if "pulmonary" not in tokens and "mean" in tokens and "arterial" in tokens and "pressure" in tokens:
        return "MAP"
    if "temperature" in tokens:
        return "Temp"
    if "lactate" in tokens:
        return "Lactate"
    if "ph" in tokens:
        return "pH"
    if "creatinine" in tokens:
        return "Creatinine"
    if "bilirubin" in tokens:
        if "direct" in tokens and "total" not in tokens:
            return None
        return "Bilirubin_Total"
    if "glucose" in tokens:
        return "Glucose"
    if "leukocytes" in tokens or "leukocyte" in tokens or "wbc" in tokens:
        return "WBC"
    if "platelets" in tokens or "platelet" in tokens:
        return "Platelets"
    if "urine" in tokens and ("volume" in tokens or "output" in tokens):
        return "urine_output"
    return None


def _register_feature_alias(alias_map: dict[str, str], token: str, feature_name: str) -> None:
    normalized = _normalize_feature_token(token)
    if not normalized:
        return
    alias_map[normalized] = feature_name
    compact = normalized.replace("_", "")
    if compact:
        alias_map[compact] = feature_name


def _is_unit_sane_for_feature(feature_name: str, unit_text: str | None, is_pharma_source: bool) -> bool:
    unit = str(unit_text or "").strip().lower()
    if is_pharma_source:
        if feature_name not in _HIRID_INTERVENTION_FEATURES:
            return False
        if not unit:
            return True
        return any(token in unit for token in _HIRID_PHARMA_UNIT_TOKENS)

    if not unit:
        return True
    hints = _HIRID_UNIT_HINTS.get(feature_name)
    if not hints:
        return True
    return any(hint.lower() in unit for hint in hints)


def _load_hirid_reference_feature_aliases(
    root: Path,
    return_diagnostics: bool = False,
) -> dict[str, str] | tuple[dict[str, str], dict[str, object]]:
    aliases: dict[str, str] = {}
    selected_aliases: dict[str, dict[str, object]] = {}
    quarantined_aliases: dict[str, str] = {}
    diagnostics: dict[str, object] = {
        "precedence": {
            "source_reliability_priority": ["observation", "preprocessed", "pharma"],
            "evidence_priority": ["name_alias", "name_keyword", "description_infer"],
            "pharma_conflict_rule": "prefer_non_pharma_for_non_intervention; prefer_pm_prefix_intervention_for_pharma",
            "unit_sanity_gate_enabled": True,
        },
        "total_rows": 0,
        "mapped_rows": 0,
        "quarantined_rows": 0,
        "quarantine_reasons": {},
        "conflict_resolved_count": 0,
        "quarantined_aliases_count": 0,
        "unmapped_rows": 0,
    }

    def _text_or_empty(value) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value).strip()

    def _mark_quarantine(reason: str) -> None:
        diagnostics["quarantined_rows"] = int(diagnostics["quarantined_rows"]) + 1
        reasons = dict(diagnostics["quarantine_reasons"])
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        diagnostics["quarantine_reasons"] = reasons

    def _evidence_bonus(kind: str) -> int:
        if kind == "name_alias":
            return 70
        if kind == "name_keyword":
            return 60
        return 45

    def _source_bonus(kind: str) -> int:
        if kind == "observation":
            return 30
        if kind == "preprocessed":
            return 25
        if kind == "pharma":
            return 20
        return 10

    def _choose_winner(
        alias_key: str,
        existing: dict[str, object],
        candidate: dict[str, object],
    ) -> dict[str, object] | None:
        existing_feature = str(existing.get("feature_name"))
        candidate_feature = str(candidate.get("feature_name"))
        existing_source = str(existing.get("source_kind"))
        candidate_source = str(candidate.get("source_kind"))
        existing_intervention = existing_feature in _HIRID_INTERVENTION_FEATURES
        candidate_intervention = candidate_feature in _HIRID_INTERVENTION_FEATURES

        if existing_feature == candidate_feature:
            if int(candidate.get("confidence", 0)) > int(existing.get("confidence", 0)):
                return candidate
            return existing

        mixed_pharma = {existing_source, candidate_source} == {"pharma", "observation"}
        if mixed_pharma:
            if alias_key.startswith("pm"):
                if candidate_intervention and not existing_intervention:
                    diagnostics["conflict_resolved_count"] = int(diagnostics["conflict_resolved_count"]) + 1
                    return candidate
                if existing_intervention and not candidate_intervention:
                    diagnostics["conflict_resolved_count"] = int(diagnostics["conflict_resolved_count"]) + 1
                    return existing
            if candidate_source == "observation" and not candidate_intervention:
                diagnostics["conflict_resolved_count"] = int(diagnostics["conflict_resolved_count"]) + 1
                return candidate
            if existing_source == "observation" and not existing_intervention:
                diagnostics["conflict_resolved_count"] = int(diagnostics["conflict_resolved_count"]) + 1
                return existing

        candidate_conf = int(candidate.get("confidence", 0))
        existing_conf = int(existing.get("confidence", 0))
        if candidate_conf > existing_conf:
            diagnostics["conflict_resolved_count"] = int(diagnostics["conflict_resolved_count"]) + 1
            return candidate
        if existing_conf > candidate_conf:
            diagnostics["conflict_resolved_count"] = int(diagnostics["conflict_resolved_count"]) + 1
            return existing

        quarantined_aliases[alias_key] = f"conflict_unresolved:{existing_feature}|{candidate_feature}"
        return None

    def _register_alias_candidate(token: str, candidate: dict[str, object]) -> None:
        normalized = _normalize_feature_token(token)
        if not normalized:
            return
        for key in (normalized, normalized.replace("_", "")):
            if not key:
                continue
            if key in quarantined_aliases:
                continue
            existing = selected_aliases.get(key)
            if existing is None:
                selected_aliases[key] = candidate
                continue
            winner = _choose_winner(key, existing, candidate)
            if winner is None:
                selected_aliases.pop(key, None)
                continue
            selected_aliases[key] = winner

    for filename in _HIRID_REFERENCE_FILE_CANDIDATES:
        ref_path = root / filename
        if not ref_path.exists():
            continue
        try:
            ref = pd.read_csv(ref_path)
        except Exception:
            continue
        if ref.empty:
            continue

        normalized_cols = {str(col).strip().lower(): str(col) for col in ref.columns}
        id_col = normalized_cols.get("variable id") or normalized_cols.get("id")
        name_col = normalized_cols.get("name") or normalized_cols.get("variable name")
        source_col = normalized_cols.get("source table") or normalized_cols.get("source")
        raw_ids_col = (
            normalized_cols.get("raw variable ids")
            or normalized_cols.get("raw_variable_ids")
            or normalized_cols.get("raw variable id")
        )
        if id_col is None:
            continue

        for _, row in ref.iterrows():
            diagnostics["total_rows"] = int(diagnostics["total_rows"]) + 1
            raw_id = row.get(id_col)
            if raw_id is None:
                diagnostics["unmapped_rows"] = int(diagnostics["unmapped_rows"]) + 1
                continue
            variable_id = _canonicalize_variable_id_token(raw_id)
            if not variable_id:
                diagnostics["unmapped_rows"] = int(diagnostics["unmapped_rows"]) + 1
                continue

            feature_name = None
            variable_name = ""
            source_name = _text_or_empty(row.get(source_col)).lower() if source_col is not None else ""
            is_preprocessed_source = "preprocessed" in filename.lower()
            source_kind = "pharma" if ("pharma" in source_name or variable_id.lower().startswith("pm")) else "observation"
            if is_preprocessed_source and source_kind != "pharma":
                source_kind = "preprocessed"
            is_pharma_source = source_kind == "pharma"
            evidence_kind = "description_infer"
            if name_col is not None:
                variable_name = _text_or_empty(row.get(name_col))
                feature_name = _canonical_feature_name(variable_name)
                if feature_name is not None:
                    evidence_kind = "name_alias"

            if feature_name is None and name_col is not None:
                variable_name_lower = variable_name.lower()
                if any(token in variable_name_lower for token in _HIRID_VASOPRESSOR_NAME_TOKENS):
                    feature_name = "vasopressor_on"
                    evidence_kind = "name_keyword"
                elif any(token in variable_name_lower for token in _HIRID_SEDATION_NAME_TOKENS):
                    feature_name = "sedation_on"
                    evidence_kind = "name_keyword"

            if feature_name is None and name_col is not None:
                feature_name = _infer_feature_name_from_description(variable_name)
                if feature_name is not None:
                    evidence_kind = "description_infer"

            if is_pharma_source and feature_name not in {"vasopressor_on", "sedation_on"}:
                diagnostics["unmapped_rows"] = int(diagnostics["unmapped_rows"]) + 1
                feature_name = None

            if feature_name is None:
                diagnostics["unmapped_rows"] = int(diagnostics["unmapped_rows"]) + 1
                continue

            unit_text = ""
            for unit_key in ("unit", "units", "doseunit"):
                unit_col = normalized_cols.get(unit_key)
                if unit_col is not None:
                    unit_text = _text_or_empty(row.get(unit_col))
                    if unit_text:
                        break
            if not _is_unit_sane_for_feature(feature_name, unit_text, is_pharma_source=is_pharma_source):
                _mark_quarantine("unit_mismatch")
                continue

            confidence = _source_bonus(source_kind) + _evidence_bonus(evidence_kind)
            if confidence < 60:
                _mark_quarantine("low_confidence_mapping")
                continue

            candidate_meta = {
                "feature_name": feature_name,
                "source_kind": source_kind,
                "evidence_kind": evidence_kind,
                "confidence": int(confidence),
            }
            _register_alias_candidate(variable_id, candidate_meta)
            if name_col is not None:
                _register_alias_candidate(variable_name, candidate_meta)
            if raw_ids_col is not None:
                raw_ids_text = _text_or_empty(row.get(raw_ids_col))
                for raw_candidate in re.findall(r"\d+(?:\.\d+)?", raw_ids_text):
                    canonical = _canonicalize_variable_id_token(raw_candidate)
                    if canonical:
                        _register_alias_candidate(canonical, candidate_meta)
            diagnostics["mapped_rows"] = int(diagnostics["mapped_rows"]) + 1

    for alias_key, candidate in selected_aliases.items():
        feature_name = str(candidate.get("feature_name", ""))
        if feature_name:
            aliases[alias_key] = feature_name

    diagnostics["quarantined_aliases_count"] = int(len(quarantined_aliases))
    if quarantined_aliases:
        diagnostics["quarantined_aliases"] = dict(sorted(quarantined_aliases.items())[:100])

    if return_diagnostics:
        return aliases, diagnostics
    return aliases


def _canonical_feature_name(name: str | None, extra_aliases: dict[str, str] | None = None) -> str | None:
    if name is None:
        return None
    if isinstance(name, (int, float)) and not pd.isna(name):
        numeric = float(name)
        if numeric.is_integer():
            text = str(int(numeric))
        else:
            text = str(numeric)
    else:
        text = str(name).strip()
    if not text:
        return None
    if text in CORE_FEATURE_NAMES:
        return text
    normalized = _normalize_feature_token(text)
    if extra_aliases:
        mapped = extra_aliases.get(normalized)
        if mapped is not None:
            return mapped
    if normalized in _NORMALIZED_FEATURE_ALIASES:
        return _NORMALIZED_FEATURE_ALIASES[normalized]
    compact = normalized.replace("_", "")
    if extra_aliases:
        mapped = extra_aliases.get(compact)
        if mapped is not None:
            return mapped
    if compact in _NORMALIZED_FEATURE_ALIASES:
        return _NORMALIZED_FEATURE_ALIASES[compact]
    inferred = _infer_feature_name_from_description(text)
    if inferred is not None:
        return inferred
    return None


def detect_hirid_root(path: Path | str) -> Path:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"HiRID path not found: {root}")
    if root.is_file():
        raise FileNotFoundError(f"unsupported HiRID file path: {root}")

    if any((root / name).is_dir() for name in HIRID_STAGE_DIRS):
        return root

    for child in root.iterdir():
        if not child.is_dir():
            continue
        if any((child / name).is_dir() for name in HIRID_STAGE_DIRS):
            return child

    raise FileNotFoundError(f"could not detect HiRID stage layout under: {root}")


def _resolve_hirid_stage_dir(root: Path) -> Path:
    for stage_name in HIRID_STAGE_DIRS:
        stage_dir = root / stage_name
        if stage_dir.is_dir():
            return stage_dir
    return root


def _iter_hirid_table_paths(stage_dir: Path) -> list[Path]:
    table_paths: list[Path] = []
    for pattern in ("*.csv", "*.csv.gz", "*.parquet"):
        table_paths.extend(path for path in stage_dir.rglob(pattern) if path.is_file())
    table_paths.sort()
    return table_paths


def _iter_hirid_chunks(table_path: Path, chunksize: int):
    suffixes = [part.lower() for part in table_path.suffixes]
    if ".parquet" in suffixes:
        yield pd.read_parquet(table_path)
        return
    if ".csv" in suffixes:
        if int(chunksize) > 0:
            for chunk in pd.read_csv(table_path, chunksize=int(chunksize), low_memory=False):
                yield chunk
            return
        yield pd.read_csv(table_path, low_memory=False)


def _resolve_stay_id_column(columns: list[str]) -> str | None:
    aliases = (
        "stay_id",
        "patientunitstayid",
        "icustay_id",
        "patientid",
    )
    lower_map = {col.lower(): col for col in columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def _extract_event_time(frame: pd.DataFrame) -> pd.Series | None:
    datetime_candidates = (
        "event_time",
        "charttime",
        "datetime",
        "givenat",
        "enteredentryat",
        "timestamp",
        "starttime",
        "observation_time",
    )
    lower_map = {col.lower(): col for col in frame.columns}
    for alias in datetime_candidates:
        col = lower_map.get(alias)
        if col is None:
            continue
        parsed = pd.to_datetime(frame[col], errors="coerce")
        if parsed.notna().any():
            return parsed

    offset_candidates = (
        "offset_min",
        "offset_minutes",
        "observationoffset",
        "minutes_from_admit",
        "time",
        "minute",
    )
    for alias in offset_candidates:
        col = lower_map.get(alias)
        if col is None:
            continue
        minutes = pd.to_numeric(frame[col], errors="coerce")
        if minutes.notna().any():
            return HIRID_BASE_TIME + pd.to_timedelta(minutes, unit="m")

    return None


def _build_observation_rows(
    frame: pd.DataFrame,
    source_name: str,
    feature_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["stay_id", "event_time", "feature_name", "value", "agg", "fill", "source_table"])

    stay_col = _resolve_stay_id_column(list(frame.columns))
    if stay_col is None:
        return pd.DataFrame(columns=["stay_id", "event_time", "feature_name", "value", "agg", "fill", "source_table"])

    event_time = _extract_event_time(frame)
    if event_time is None:
        return pd.DataFrame(columns=["stay_id", "event_time", "feature_name", "value", "agg", "fill", "source_table"])

    base = frame.copy()
    base["stay_id"] = pd.to_numeric(base[stay_col], errors="coerce")
    base["event_time"] = pd.to_datetime(event_time, errors="coerce")
    base = base.dropna(subset=["stay_id", "event_time"]).copy()
    if base.empty:
        return pd.DataFrame(columns=["stay_id", "event_time", "feature_name", "value", "agg", "fill", "source_table"])

    outputs: list[pd.DataFrame] = []
    lower_map = {col.lower(): col for col in base.columns}
    long_feature_col = None
    for alias in ("feature_name", "variable", "var_name", "measurement", "variableid", "pharmaid", "itemid"):
        if alias in lower_map:
            long_feature_col = lower_map[alias]
            break
    long_value_col = None
    for alias in (
        "value",
        "valuenum",
        "measurement_value",
        "givendose",
        "cumulativedose",
        "fluidamount_calc",
        "cumulfluidamount_calc",
    ):
        if alias in lower_map:
            long_value_col = lower_map[alias]
            break

    if long_feature_col and long_value_col:
        long_frame = base[["stay_id", "event_time", long_feature_col, long_value_col]].copy()
        long_frame["feature_name"] = long_frame[long_feature_col].map(
            lambda value: _canonical_feature_name(value, extra_aliases=feature_aliases)
        )
        long_frame["value"] = pd.to_numeric(long_frame[long_value_col], errors="coerce")
        long_frame = long_frame.dropna(subset=["feature_name", "value"]).copy()
        if not long_frame.empty:
            outputs.append(long_frame[["stay_id", "event_time", "feature_name", "value"]])

    excluded = {
        stay_col.lower(),
        "stay_id",
        "event_time",
        "charttime",
        "datetime",
        "givenat",
        "enteredentryat",
        "timestamp",
        "starttime",
        "observation_time",
        "offset_min",
        "offset_minutes",
        "observationoffset",
        "minutes_from_admit",
        "time",
        "minute",
        "feature_name",
        "variable",
        "var_name",
        "measurement",
        "variableid",
        "pharmaid",
        "itemid",
        "value",
        "valuenum",
        "measurement_value",
        "givendose",
        "cumulativedose",
        "fluidamount_calc",
        "cumulfluidamount_calc",
    }
    for col in base.columns:
        if col.lower() in excluded:
            continue
        feature_name = _canonical_feature_name(col, extra_aliases=feature_aliases)
        if not feature_name:
            continue
        block = base[["stay_id", "event_time", col]].copy()
        block["feature_name"] = feature_name
        block["value"] = pd.to_numeric(block[col], errors="coerce")
        block = block.dropna(subset=["value"])
        if block.empty:
            continue
        outputs.append(block[["stay_id", "event_time", "feature_name", "value"]])

    if not outputs:
        return pd.DataFrame(columns=["stay_id", "event_time", "feature_name", "value", "agg", "fill", "source_table"])

    observed = pd.concat(outputs, ignore_index=True)
    observed["stay_id"] = observed["stay_id"].astype(int)
    observed["agg"] = observed["feature_name"].map(lambda name: _FEATURE_META[name].agg)
    observed["fill"] = observed["feature_name"].map(lambda name: _FEATURE_META[name].fill)
    observed["source_table"] = source_name
    observed = observed.drop_duplicates(subset=["stay_id", "event_time", "feature_name", "value", "source_table"]).copy()
    observed = observed.sort_values(["stay_id", "event_time", "feature_name"]).reset_index(drop=True)
    return observed


def collect_hirid_feature_observations(
    stage_dir: Path,
    chunksize: int = 200000,
    feature_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    table_paths = _iter_hirid_table_paths(stage_dir)
    for table_path in table_paths:
        source_name = str(table_path.relative_to(stage_dir)).replace("\\", "/")
        for chunk in _iter_hirid_chunks(table_path, chunksize=chunksize):
            rows = _build_observation_rows(
                chunk,
                source_name=source_name,
                feature_aliases=feature_aliases,
            )
            if not rows.empty:
                outputs.append(rows)
    if not outputs:
        return pd.DataFrame(columns=["stay_id", "event_time", "feature_name", "value", "agg", "fill", "source_table"])
    observed = pd.concat(outputs, ignore_index=True)
    observed = observed.sort_values(["stay_id", "event_time", "feature_name"]).reset_index(drop=True)
    return observed


def _build_hirid_icustays(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])

    bounds = observations.groupby("stay_id", as_index=False).agg(
        intime=("event_time", "min"),
        outtime=("event_time", "max"),
    )
    bounds["subject_id"] = bounds["stay_id"]
    bounds["hadm_id"] = bounds["stay_id"]
    return bounds[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]]


def _build_vasopressor_seed(observations: pd.DataFrame) -> pd.DataFrame:
    vaso_rows = observations[observations["feature_name"] == "vasopressor_on"].copy()
    if vaso_rows.empty:
        return pd.DataFrame(columns=["stay_id", "itemid", "starttime", "rate", "amount", "statusdescription"])
    vaso_rows = vaso_rows[vaso_rows["value"] > 0].copy()
    if vaso_rows.empty:
        return pd.DataFrame(columns=["stay_id", "itemid", "starttime", "rate", "amount", "statusdescription"])
    vaso_rows["itemid"] = 221906
    vaso_rows["starttime"] = vaso_rows["event_time"]
    vaso_rows["rate"] = pd.to_numeric(vaso_rows["value"], errors="coerce").fillna(0.0)
    vaso_rows["amount"] = vaso_rows["rate"]
    vaso_rows["statusdescription"] = "Running"
    return vaso_rows[["stay_id", "itemid", "starttime", "rate", "amount", "statusdescription"]]


def _persist_hirid_mapping_diagnostics(
    out_dir: Path | str,
    diagnostics: dict[str, object],
    entries: list[dict] | None = None,
) -> None:
    payload = dict(diagnostics)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "hirid_mapping_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    if entries is None:
        return
    for entry in entries:
        if isinstance(entry, dict):
            entry["hirid_mapping_diagnostics"] = {
                "quarantined_rows": int(payload.get("quarantined_rows", 0)),
                "quarantine_reasons": dict(payload.get("quarantine_reasons", {})),
                "quarantined_aliases_count": int(payload.get("quarantined_aliases_count", 0)),
            }


def prepare_hirid_dataset(
    raw_dir: Path | str,
    out_dir: Path | str,
    cohort_limit: int = 0,
    resample_minutes: int = 60,
    hirid_root: str | None = None,
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
        raise ValueError("HiRID preparation currently supports mode='shock-episodes' only")

    source = Path(hirid_root) if hirid_root else Path(raw_dir)
    root = detect_hirid_root(source)
    stage_dir = _resolve_hirid_stage_dir(root)
    feature_aliases, mapping_diagnostics = _load_hirid_reference_feature_aliases(root, return_diagnostics=True)
    observations = collect_hirid_feature_observations(
        stage_dir,
        chunksize=chunksize,
        feature_aliases=feature_aliases,
    )
    icustays = _build_hirid_icustays(observations)
    cohort = build_icu_cohort(icustays, min_hours=min_hours, cohort_limit=cohort_limit)

    if cohort.empty:
        empty_episode_index = build_shock_episode_index(
            pd.DataFrame(columns=["stay_id", "anchor_time", "anchor_kind", "previous_rate", "current_rate"]),
            cohort,
            pre_hours=shock_pre_hours,
            max_control_windows_per_anchor=shock_control_windows_per_anchor,
            control_window_gap_hours=shock_control_window_gap_hours,
            min_control_windows_per_anchor=shock_min_controls_per_anchor,
            control_fallback_stride_hours=shock_control_fallback_stride_hours,
        )
        entries = materialize_shock_episode_matrices(
            observations.iloc[0:0].copy(),
            empty_episode_index,
            out_dir,
            resample_minutes=resample_minutes,
            feature_names=get_feature_names("core"),
        )
        _persist_hirid_mapping_diagnostics(out_dir, mapping_diagnostics, entries=entries)
        return entries

    cohort_stays = set(cohort["stay_id"].astype(int).tolist())
    observations_cohort = observations[observations["stay_id"].astype(int).isin(cohort_stays)].copy()
    anchors = extract_vasopressor_anchors(_build_vasopressor_seed(observations_cohort))

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
        entries = materialize_shock_episode_matrices(
            observations_cohort,
            empty_episode_index,
            out_dir,
            resample_minutes=resample_minutes,
            feature_names=get_feature_names("core"),
        )
        _persist_hirid_mapping_diagnostics(out_dir, mapping_diagnostics, entries=entries)
        return entries

    anchor_stays = set(pd.to_numeric(anchors["stay_id"], errors="coerce").dropna().astype(int).tolist())
    cohort_anchor = cohort[cohort["stay_id"].astype(int).isin(anchor_stays)].copy()
    observations_anchor = observations_cohort[observations_cohort["stay_id"].astype(int).isin(anchor_stays)].copy()
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
    entries = materialize_shock_episode_matrices(
        observations_anchor,
        episode_index,
        out_dir,
        resample_minutes=resample_minutes,
        feature_names=get_feature_names("core"),
    )
    _persist_hirid_mapping_diagnostics(out_dir, mapping_diagnostics, entries=entries)
    return entries
