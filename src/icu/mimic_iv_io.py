import gzip
import io
import zipfile
from pathlib import Path

import pandas as pd

from .mimic_iv_schema import get_feature_specs

MIMIC_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TABLE_DEFAULT_DTYPES = {
    "icu/icustays.csv.gz": {"subject_id": "Int64", "hadm_id": "Int64", "stay_id": "Int64"},
    "icu/chartevents.csv.gz": {"stay_id": "Int64", "itemid": "Int64", "valuenum": "float64"},
    "hosp/labevents.csv.gz": {"subject_id": "Int64", "hadm_id": "Int64", "itemid": "Int64", "valuenum": "float64"},
    "icu/inputevents.csv.gz": {"stay_id": "Int64", "itemid": "Int64", "rate": "float64", "amount": "float64"},
    "icu/outputevents.csv.gz": {"stay_id": "Int64", "itemid": "Int64", "value": "float64"},
}


def list_required_tables() -> list[str]:
    tables = {"icu/icustays.csv.gz"}
    for spec in get_feature_specs("core"):
        tables.add(spec.source_table)
    return sorted(tables)


def _is_mimic_root(path: Path) -> bool:
    return (path / "icu").is_dir() and (path / "hosp").is_dir()


def detect_mimic_root(path: Path | str) -> Path:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"MIMIC-IV path not found: {root}")

    if root.is_file():
        if root.suffix.lower() == ".zip":
            return root
        raise FileNotFoundError(f"unsupported MIMIC-IV file path: {root}")

    if _is_mimic_root(root):
        return root

    extracted_roots = sorted(
        child for child in root.iterdir()
        if child.is_dir() and child.name.startswith("mimic-iv-") and _is_mimic_root(child)
    )
    if extracted_roots:
        return extracted_roots[0]

    zip_archives = sorted(root.glob("mimic-iv-*.zip"))
    if zip_archives:
        return zip_archives[0]

    raise FileNotFoundError(f"could not detect a MIMIC-IV root under: {root}")


def _resolve_zip_member(zf: zipfile.ZipFile, table_name: str) -> str:
    matches = [name for name in zf.namelist() if name.endswith(table_name)]
    if not matches:
        raise FileNotFoundError(f"table not found in MIMIC-IV zip: {table_name}")
    return matches[0]


def iter_table_chunks(root: Path | str, table_name: str, usecols=None, chunksize: int = 100000, dtypes: dict | None = None):
    base = Path(root)
    resolved_dtypes = dict(dtypes or TABLE_DEFAULT_DTYPES.get(table_name, {}))
    if base.is_file() and base.suffix.lower() == ".zip":
        with zipfile.ZipFile(base) as zf:
            member_name = _resolve_zip_member(zf, table_name)
            with zf.open(member_name, "r") as raw:
                if member_name.endswith(".gz"):
                    file_obj = gzip.open(raw, "rt", encoding="utf-8")
                else:
                    file_obj = io.TextIOWrapper(raw, encoding="utf-8")
                with file_obj as text_stream:
                    for chunk in pd.read_csv(
                        text_stream,
                        usecols=usecols,
                        chunksize=chunksize,
                        low_memory=False,
                        dtype=resolved_dtypes,
                    ):
                        yield chunk
        return

    path = base / table_name
    if not path.exists():
        raise FileNotFoundError(f"table not found under MIMIC-IV root: {path}")
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        compression="infer",
        dtype=resolved_dtypes,
    ):
        yield chunk


def _parse_mimic_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format=MIMIC_DATETIME_FORMAT, errors="coerce")
    unresolved = parsed.isna() & values.notna()
    if unresolved.any():
        # Fallback for non-standard rows (for example fractional seconds).
        parsed.loc[unresolved] = pd.to_datetime(values[unresolved], errors="coerce")
    return parsed


def normalize_chart_chunk(chunk: pd.DataFrame, allowed_itemids: set[int], stay_ids: set[int] | None = None) -> pd.DataFrame:
    frame = chunk.copy()
    frame = frame[frame["itemid"].isin(allowed_itemids)]
    if stay_ids is not None and "stay_id" in frame.columns:
        frame = frame[frame["stay_id"].isin(stay_ids)]
    if frame.empty:
        return pd.DataFrame(columns=["stay_id", "itemid", "event_time", "value"])
    frame["event_time"] = _parse_mimic_datetime(frame["charttime"])
    frame["value"] = pd.to_numeric(frame.get("valuenum"), errors="coerce")
    frame = frame.dropna(subset=["stay_id", "event_time", "value"]).copy()
    frame["stay_id"] = frame["stay_id"].astype(int)
    frame["itemid"] = frame["itemid"].astype(int)
    return frame[["stay_id", "itemid", "event_time", "value"]]


def _prepare_cohort_for_lab_join(cohort: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"hadm_id", "stay_id", "intime", "outtime"}
    if not required_cols.issubset(cohort.columns):
        columns = {
            "hadm_id": pd.Series(dtype="int64"),
            "stay_id": pd.Series(dtype="int64"),
            "intime": pd.Series(dtype="datetime64[ns]"),
            "outtime": pd.Series(dtype="datetime64[ns]"),
        }
        if "subject_id" in cohort.columns:
            columns["subject_id"] = pd.Series(dtype="int64")
            return pd.DataFrame(columns)[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]]
        return pd.DataFrame(columns)

    cohort_cols = [col for col in ["subject_id", "hadm_id", "stay_id", "intime", "outtime"] if col in cohort.columns]
    cohort_frame = cohort[cohort_cols].copy()
    cohort_frame["hadm_id"] = pd.to_numeric(cohort_frame["hadm_id"], errors="coerce").astype("Int64")
    cohort_frame["stay_id"] = pd.to_numeric(cohort_frame["stay_id"], errors="coerce").astype("Int64")
    if "subject_id" in cohort_frame.columns:
        cohort_frame["subject_id"] = pd.to_numeric(cohort_frame["subject_id"], errors="coerce").astype("Int64")
    cohort_frame["intime"] = pd.to_datetime(cohort_frame["intime"], errors="coerce")
    cohort_frame["outtime"] = pd.to_datetime(cohort_frame["outtime"], errors="coerce")
    cohort_frame = cohort_frame.dropna(subset=["hadm_id", "stay_id", "intime", "outtime"]).copy()
    cohort_frame["hadm_id"] = cohort_frame["hadm_id"].astype(int)
    cohort_frame["stay_id"] = cohort_frame["stay_id"].astype(int)
    if "subject_id" in cohort_frame.columns:
        cohort_frame["subject_id"] = cohort_frame["subject_id"].astype(int)
    return cohort_frame


def normalize_lab_chunk(
    chunk: pd.DataFrame,
    cohort: pd.DataFrame | None,
    allowed_itemids: set[int],
    cohort_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = chunk.copy()
    frame = frame[frame["itemid"].isin(allowed_itemids)]
    if frame.empty:
        return pd.DataFrame(columns=["stay_id", "itemid", "event_time", "value"])
    frame["event_time"] = _parse_mimic_datetime(frame["charttime"])
    frame["value"] = pd.to_numeric(frame.get("valuenum"), errors="coerce")
    frame = frame.dropna(subset=["hadm_id", "event_time", "value"]).copy()
    frame["hadm_id"] = frame["hadm_id"].astype(int)
    frame["itemid"] = frame["itemid"].astype(int)

    if cohort_lookup is None:
        if cohort is None:
            raise ValueError("cohort or cohort_lookup is required for lab normalization")
        cohort_frame = _prepare_cohort_for_lab_join(cohort)
    else:
        cohort_frame = cohort_lookup

    merge_keys = ["hadm_id"]
    if "subject_id" in frame.columns and "subject_id" in cohort_frame.columns:
        frame["subject_id"] = pd.to_numeric(frame["subject_id"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["subject_id"]).copy()
        frame["subject_id"] = frame["subject_id"].astype(int)
        merge_keys = ["hadm_id", "subject_id"]

    merged = frame.merge(cohort_frame, on=merge_keys, how="inner", sort=False)
    merged = merged[(merged["event_time"] >= merged["intime"]) & (merged["event_time"] <= merged["outtime"])].copy()
    return merged[["stay_id", "itemid", "event_time", "value"]]


def normalize_input_chunk(chunk: pd.DataFrame, allowed_itemids: set[int], stay_ids: set[int] | None = None) -> pd.DataFrame:
    frame = chunk.copy()
    frame = frame[frame["itemid"].isin(allowed_itemids)]
    if stay_ids is not None and "stay_id" in frame.columns:
        frame = frame[frame["stay_id"].isin(stay_ids)]
    if "statusdescription" in frame.columns:
        mask = frame["statusdescription"].astype(str).str.contains("rewrite|stop|paused", case=False, na=False)
        frame = frame[~mask]
    if frame.empty:
        return pd.DataFrame(columns=["stay_id", "itemid", "event_time", "value"])
    frame["event_time"] = _parse_mimic_datetime(frame["starttime"])
    value_col = "rate" if "rate" in frame.columns else "amount"
    frame["value"] = pd.to_numeric(frame.get(value_col), errors="coerce")
    frame = frame.dropna(subset=["stay_id", "event_time"]).copy()
    frame["value"] = frame["value"].fillna(1.0)
    frame["stay_id"] = frame["stay_id"].astype(int)
    frame["itemid"] = frame["itemid"].astype(int)
    return frame[["stay_id", "itemid", "event_time", "value"]]


def normalize_output_chunk(chunk: pd.DataFrame, allowed_itemids: set[int], stay_ids: set[int] | None = None) -> pd.DataFrame:
    frame = chunk.copy()
    frame = frame[frame["itemid"].isin(allowed_itemids)]
    if stay_ids is not None and "stay_id" in frame.columns:
        frame = frame[frame["stay_id"].isin(stay_ids)]
    if frame.empty:
        return pd.DataFrame(columns=["stay_id", "itemid", "event_time", "value"])
    frame["event_time"] = _parse_mimic_datetime(frame["charttime"])
    frame["value"] = pd.to_numeric(frame.get("value"), errors="coerce")
    frame = frame.dropna(subset=["stay_id", "event_time", "value"]).copy()
    frame["stay_id"] = frame["stay_id"].astype(int)
    frame["itemid"] = frame["itemid"].astype(int)
    return frame[["stay_id", "itemid", "event_time", "value"]]
