"""MIMIC-IV 3.1 source adapter for the v2 raw-data extraction.

Converts the raw MIMIC-IV tables into the four frames of ``src/icu/v2/extract/common.py``
(stays, vasopressor intervals, events, interventions). Everything that defines the task
(anchors, windows, controls) lives in ``common.py``; this module only decides which raw
rows mean what. Mappings follow ``src/icu/v2/harmonization.py`` (MIMIC_IV_TABLE,
CANONICAL_SPECS) except where a decision below says otherwise.

Raw checks (``docs/reviews/v2_harmonization_spec.md`` section 8, run on 2026-10-21; outputs
in ``artifacts/v2/harmonization/mimic_iv_*``) and the decisions they support
------------------------------------------------------------------------------------------

R-M1 outputevents (5,359,395 rows, 91,510 stays):
  * urine_output = the 10 mimic-code urine items 226559 Foley, 226560 Void, 226561, 226584,
    226563, 226564, 226565, 226567, 226557, 226558 plus the GU-irrigation pair. They hold
    4,145,339 rows in 90,471 stays; Foley + Void are 96.2 % of the rows. The v1 items
    (226566, 226627 OR Urine, 226631 PACU Urine, 227489) hold 33,090 rows in 22,286 stays
    (226566 has no rows in 3.1), which confirms defect N1. All units are mL.
  * GU irrigation: 227488 (irrigant in) is charted at the same (stay, charttime) as 227489
    (irrigant + urine out) for 6,576 of 7,199 rows (91 %). v2 emits the net
    out - in per (stay, charttime) for paired rows only; 1,271 of 7,824 pairs are negative
    (more irrigant in than out: urine undetermined) and are dropped, as are unpaired rows.
    Netting before the plausibility screen avoids keeping one half of a 6 L pair.
  * Rows outside CANONICAL_SPECS (0-5000 mL per record) are dropped (24 negative, 10 > 5 L
    among the urine items).

R-M2 inputevents (drug items):
  * statusdescription values for every drug item are ChangeDose/Rate, FinishedRunning,
    Paused, Stopped and Bolus; there are no Rewritten rows in 3.1 (still excluded if they
    appear). In MetaVision a row is a delivery period [starttime, endtime) and the status
    says how it ended: Paused and Stopped rows have amount > 0 in 100 % of rows (median 72
    and 88 min), so they are KEPT. v1 dropped them (``mimic_iv_io.py``,
    ``shock_anchor_builder.py``); for 2,716 (Paused) and 2,607 (Stopped) of 28,483 stays the
    first vasopressor row has that status, so dropping it would move the anchor after the
    true initiation and leak post-initiation data into the positive window.
  * status "Bolus" inside Continuous Med rows is a bolus given from the drip (median 1-6.5
    min; norepinephrine 26 rows/14 stays, phenylephrine 18, dopamine 2, propofol 821):
    excluded (no boluses).
  * Vasopressors (continuous, rate > 0): 221906 norepinephrine, 221289 epinephrine,
    221662 dopamine, 221749/229630/229632/229631 phenylephrine, 222315 vasopressin and
    angiotensin II 229709/229764 (option ``include_angiotensin``, default True; 52 stays,
    never the first vasopressor). 229630 (61,013 rows, 3,778 stays) and 229632 (12,840
    rows, 453 stays) are continuous mcg/kg/min drips -> included; 229631 has no rows.
    229617 "Epinephrine." is Drug Push only (234 rows, 107 stays) -> excluded, as are
    dobutamine 221653 and milrinone 221986. All other vasopressor rows are "Continuous Med"
    and only 2 have rate <= 0. First continuous vasopressor of the 28,481 stays:
    phenylephrine 13,802, norepinephrine 11,417, epinephrine 1,450, dopamine 1,386,
    vasopressin 426.
  * Sedation (task decision, differs from harmonization.py which lists fentanyl and not
    lorazepam): continuous rate > 0 infusions of propofol 222168 (402,988 rows, 28,237
    stays; 95,823 pushes excluded), midazolam 221668 (62,335 rows, 7,627 stays; 64,479
    pushes excluded), dexmedetomidine 225150 (48,368 rows, 3,459 stays) and 229420 (92,549
    rows, 4,757 stays), lorazepam 221385 (233 continuous rows, 42 stays; 42,035 pushes
    excluded). Opioid infusions only with ``include_opioids``: fentanyl 221744 (11,992
    continuous rows) and 225942 (138,861), hydromorphone 221833 (15,332), morphine 225154
    (8,008). Fentanyl (Push) 225972 is never used; ketamine 221712 (13,685 continuous rows,
    610 stays) is outside the set. v1 counted the pushes (rate NaN filled with 1.0).

R-M3 labevents (158,374,764 rows):
  * Included items carry one unit each (50813 mmol/L, 50820 units, 50912/50931/50885
    mg/dL, 51301/51300/51265 K/uL); one row per item has an empty unit and is read in the
    declared unit.
  * The pending second itemids (53154, 52546, 52569, 53089, 51755, 51756, 53189; 52442 has
    no rows) have <= 1,309 rows overall, never share a specimen with the primary item and
    have 0 rows inside any ICU stay -> excluded, no effect. 52024 whole-blood creatinine
    (blood gas; 3,160 ICU rows vs 575,869 for 50912) and 50809 blood-gas glucose (183,527
    ICU rows) are excluded.
  * pH 50820 specimen types (52033), all rows: arterial 70.2 %, venous 22.0 %, mixed 2.9 %,
    central venous 1.9 %, none 3.0 %; rows inside ICU stays (548,061): arterial 77.7 %,
    venous 16.1 %, mixed 2.6 %, central venous 2.2 %, none 1.4 %. Default pools them
    (harmonization spec: blood-gas pH); option ``ph_specimen="arterial"`` keeps arterial
    specimens only.
  * Labs have no stay_id: a row is assigned to every ICU stay of the same subject whose
    [intime, outtime] contains charttime (no MIMIC stays overlap). hadm_id is null for
    ~0.2 % of these rows, so the join does not use hadm_id. Time = charttime (specimen
    collection), as in mimic-code.

R-M4 chartevents (432,997,491 rows):
  * HR 220045 (8,752,069 rows, bpm); RR 220210 (8,636,655) + 224690 Respiratory Rate
    (Total) (753,531; mimic-code resp_rate), excluding 224688 set and 224689 spontaneous;
    SpO2 220277 (8,567,015), excluding SaO2/SvO2/ScvO2; Temp 223761 degF (2,055,040,
    converted) + 223762 degC (394,843).
  * BP: invasive 220050/220051/220052 (~3.09 M rows each) and 225309/225310/225312
    (~388 k rows each, medians 111/56/75 vs 116/57/76 mmHg -> included), non-invasive
    220179/220180/220181 (~5.37 M each); every row carries its modality and common.py
    applies BP_MERGE_RULE per bin. Stay-hours with BP: non-invasive only 4,645,681,
    invasive only 2,535,725, both 412,264 -> 86 % of arterial-line hours have no cuff
    reading, which confirms the mechanism of spec section 3.2 (v1 used cuff BP only).
    38,260 of 94,353 stays have invasive BP.
  * Glucose = lab 50931 + point-of-care finger stick 225664 (task decision: POC glucose
    included in every database for cross-database consistency). 225664 has 1,024,965 rows
    and an empty valueuom on every row; it is read as mg/dL (median 140). R-M6
    (``mimic_iv_R-M6_poc_glucose.txt``, ICU rows, nearest lab row of the stay within 2 h):
    220621 "Glucose (serum)" repeats a lab 50931 value at the same minute for 96.0 % of its
    584,841 rows and 226537 "Glucose (whole blood)" repeats a blood-gas 50809 value for
    89.5 % of its 204,656 rows -> both are nurse-charted copies and are excluded (50809
    itself is excluded). The finger stick shares its value with a lab or blood-gas result
    for only 0.5 % / 0.2 % of rows -> an independent measurement.
  * Values outside CANONICAL_SPECS are dropped, not repaired (e.g. HR ranges from -241,395 to
    1e7 in the raw table; 813 of 394,843 values of the Celsius item 223762 lie in 90-110,
    i.e. Fahrenheit typed into the Celsius item, and 712 of 2,055,040 values of the
    Fahrenheit item 223761 lie in 25-46).

R-M5 procedureevents (ventilation, task decision):
  * ventilation_on = 225792 Invasive Ventilation intervals (35,479 rows, 31,969 stays;
    statuses FinishedRunning/Stopped/Paused, no missing end). 225794 Non-invasive
    Ventilation (3,238 rows, 2,696 stays) only with ``include_noninvasive_ventilation``.
  * Coverage against chartevents documentation: 84.1 % of stay-hours with an invasive
    ventilator mode and 85.1 % of hours with "Endotracheal tube" charted lie inside a
    225792 interval; 86.7 % of 225792 interval-hours have such documentation within +-4 h.
    procedureevents therefore misses ~15 % of documented invasive ventilation. 225794
    covers only 21.9 % of hours with a BiPAP/CPAP mask charted.

Adults: every MIMIC-IV patient has anchor_age >= 18 (minimum 18) and no stay starts before
anchor_year, so the age filter (anchor_age and age at intime >= ``min_age``) keeps all
94,444 stays with a valid [intime, outtime] (14 stays without outtime are dropped).

Caching (``cache_dir``)
-----------------------
Every big table is streamed once with ``usecols`` and an itemid filter and the rows of a
fixed itemid SUPERSET (mapped items plus the candidates the checks compare) are stored for
ALL ICU stays as zstd parquet sorted by (stay_id, time): ``chartevents.parquet`` (~58 M
rows), ``labevents.parquet`` (already assigned to stays, with the blood-gas specimen type),
``inputevents.parquet``, ``outputevents.parquet``, ``procedureevents.parquet`` and
``stays.parquet``. ``cache_manifest.json`` records, per table, a key (cache version, file
name, size and mtime of the source file or CRC of the zip member, itemid superset); a table
is rebuilt only when its key changes. The mapping (itemid -> feature, units, plausibility,
options) is applied when :meth:`MimicIVAdapter.events` is called, so options never require a
rebuild. Each cache is loaded into memory once per adapter (chartevents 1.7 GB) and
:meth:`events` slices the requested stays with a binary search on the sorted stay_id column,
so the chunked calls of ``common.extract_cohort`` never re-read the 3.5 GB chartevents file.
Building all caches takes ~8 min (chartevents 294 s, labevents 139 s; peak RSS 5.2 GB, 340
MB on disk); an extraction from warm caches takes ~1 min.
"""

from __future__ import annotations

import contextlib
import json
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.icu.v2.extract.common import CANONICAL_FEATURES
from src.icu.v2.harmonization import CANONICAL_SPECS, conversion_for, normalize_unit

SOURCE_VERSION = "MIMIC-IV 3.1"
CACHE_VERSION = 1

# --------------------------------------------------------------------------------------
# Item maps (labels from icu/d_items and hosp/d_labitems of MIMIC-IV 3.1)
# --------------------------------------------------------------------------------------

#: chartevents numeric items: itemid -> (feature, source unit, BP modality)
CHART_ITEMS: dict[int, tuple[str, str, str | None]] = {
    220045: ("HR", "bpm", None),
    220210: ("RR", "insp/min", None),
    224690: ("RR", "insp/min", None),
    220277: ("SpO2", "%", None),
    220179: ("SBP", "mmHg", "noninvasive"),
    220180: ("DBP", "mmHg", "noninvasive"),
    220181: ("MAP", "mmHg", "noninvasive"),
    220050: ("SBP", "mmHg", "invasive"),
    220051: ("DBP", "mmHg", "invasive"),
    220052: ("MAP", "mmHg", "invasive"),
    225309: ("SBP", "mmHg", "invasive"),
    225310: ("DBP", "mmHg", "invasive"),
    225312: ("MAP", "mmHg", "invasive"),
    223762: ("Temp", "°C", None),
    223761: ("Temp", "°F", None),
    225664: ("Glucose", "mg/dL", None),  # point-of-care finger stick (option poc_glucose)
}
POC_GLUCOSE_ITEMS: frozenset[int] = frozenset({225664})
#: cached for the checks, never mapped: nurse-charted copies of lab / blood-gas glucose
CHART_CHECK_ITEMS: dict[int, str] = {220621: "Glucose (serum)", 226537: "Glucose (whole blood)"}

#: labevents items: itemid -> (feature, source unit)
LAB_ITEMS: dict[int, tuple[str, str]] = {
    50813: ("Lactate", "mmol/L"),
    50820: ("pH", "units"),
    50912: ("Creatinine", "mg/dL"),
    50931: ("Glucose", "mg/dL"),
    50885: ("Bilirubin_Total", "mg/dL"),
    51301: ("WBC", "K/uL"),
    51300: ("WBC", "K/uL"),
    51265: ("Platelets", "K/uL"),
}
#: cached for the checks, never mapped
LAB_CHECK_ITEMS: dict[int, str] = {50809: "Glucose (Blood, Blood Gas)"}
SPECIMEN_TYPE_ITEM = 52033  # "Specimen Type" of a blood-gas specimen (ART./VEN./MIX./CENTRAL VENOUS.)
ARTERIAL_SPECIMEN = "ART."
PH_ITEM = 50820

URINE_ITEMS: dict[int, str] = {
    226559: "Foley", 226560: "Void", 226561: "Condom Cath", 226584: "Ileoconduit", 226563: "Suprapubic",
    226564: "R Nephrostomy", 226565: "L Nephrostomy", 226567: "Straight Cath", 226557: "R Ureteral Stent",
    226558: "L Ureteral Stent",
}
GU_IRRIGANT_OUT = 227489  # GU Irrigant/Urine Volume Out (irrigant + urine)
GU_IRRIGANT_IN = 227488  # GU Irrigant Volume In

VASOPRESSOR_ITEMS: dict[int, str] = {
    221906: "norepinephrine", 221289: "epinephrine", 221662: "dopamine", 221749: "phenylephrine",
    229630: "phenylephrine", 229632: "phenylephrine", 229631: "phenylephrine", 222315: "vasopressin",
}
ANGIOTENSIN_ITEMS: dict[int, str] = {229709: "angiotensin_ii", 229764: "angiotensin_ii"}
#: cached so the exclusions can be counted, never used
EXCLUDED_DRUG_ITEMS: dict[int, str] = {
    229617: "Epinephrine. (Drug Push only)", 221653: "Dobutamine (inotrope)", 221986: "Milrinone (inodilator)",
    225972: "Fentanyl (Push)", 221712: "Ketamine (outside the sedation set)",
}
SEDATIVE_ITEMS: dict[int, str] = {
    222168: "propofol", 221668: "midazolam", 225150: "dexmedetomidine", 229420: "dexmedetomidine",
    221385: "lorazepam",
}
OPIOID_ITEMS: dict[int, str] = {221744: "fentanyl", 225942: "fentanyl", 221833: "hydromorphone", 225154: "morphine"}
CONTINUOUS_CATEGORIES: tuple[str, ...] = ("Continuous Med", "Continuous IV")
EXCLUDED_STATUSES: tuple[str, ...] = ("Rewritten", "Bolus")

VENT_INVASIVE_ITEM = 225792
VENT_NONINVASIVE_ITEM = 225794

EVENT_OUTPUT_COLUMNS = ("stay_id", "time", "feature", "value", "modality")
INTERVENTION_OUTPUT_COLUMNS = ("stay_id", "start", "end", "feature")

#: raw file of each cached table, relative to raw_dir (or to the root folder of the zip)
TABLE_FILES: dict[str, tuple[str, ...]] = {
    "stays": ("icu/icustays.csv.gz", "hosp/patients.csv.gz"),
    "inputevents": ("icu/inputevents.csv.gz",),
    "outputevents": ("icu/outputevents.csv.gz",),
    "procedureevents": ("icu/procedureevents.csv.gz",),
    "chartevents": ("icu/chartevents.csv.gz",),
    "labevents": ("hosp/labevents.csv.gz",),
}
DICTIONARY_FILES = ("icu/d_items.csv.gz", "hosp/d_labitems.csv.gz")

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _cache_items(table: str) -> list[int]:
    """Itemid superset stored in the cache of ``table`` (part of the cache key)."""
    if table == "chartevents":
        return sorted(set(CHART_ITEMS) | set(CHART_CHECK_ITEMS))
    if table == "labevents":
        return sorted(set(LAB_ITEMS) | set(LAB_CHECK_ITEMS) | {SPECIMEN_TYPE_ITEM})
    if table == "inputevents":
        return sorted(set(VASOPRESSOR_ITEMS) | set(ANGIOTENSIN_ITEMS) | set(EXCLUDED_DRUG_ITEMS) | set(SEDATIVE_ITEMS) | set(OPIOID_ITEMS))
    if table == "outputevents":
        return sorted(set(URINE_ITEMS) | {GU_IRRIGANT_OUT, GU_IRRIGANT_IN})
    if table == "procedureevents":
        return [VENT_INVASIVE_ITEM, VENT_NONINVASIVE_ITEM]
    return []


def _parse_time(values: pd.Series) -> pd.Series:
    """MIMIC timestamps (``YYYY-MM-DD HH:MM:SS``; ISO fallback for fractional seconds)."""
    parsed = pd.to_datetime(values, format=_TIME_FORMAT, errors="coerce")
    retry = parsed.isna() & values.notna()
    if retry.any():
        parsed.loc[retry] = pd.to_datetime(values[retry], format="ISO8601", errors="coerce")
    return parsed.astype("datetime64[ns]")


def _slice_sorted(frame: pd.DataFrame, sorted_ids: np.ndarray, wanted: np.ndarray) -> pd.DataFrame:
    """Rows of ``frame`` (sorted by stay_id, ids in ``sorted_ids``) whose stay is in ``wanted``."""
    wanted = np.unique(np.asarray(wanted, dtype=np.int64))
    starts = np.searchsorted(sorted_ids, wanted, side="left")
    ends = np.searchsorted(sorted_ids, wanted, side="right")
    lengths = ends - starts
    total = int(lengths.sum())
    if total == 0:
        return frame.iloc[0:0]
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    index = np.arange(total) + np.repeat(starts - offsets, lengths)
    return frame.iloc[index]


class _Codes:
    """Small string -> int16 code table (keeps 60 M unit strings out of memory)."""

    def __init__(self) -> None:
        self.index: dict[str, int] = {}

    def encode(self, values: pd.Series) -> np.ndarray:
        text = values.fillna("").astype(str)
        for value in text.unique():
            if value not in self.index:
                self.index[value] = len(self.index)
        return text.map(self.index).to_numpy(dtype=np.int16)

    def dictionary(self) -> list[str]:
        return [value for value, _ in sorted(self.index.items(), key=lambda kv: kv[1])]


def _dictionary_column(codes: np.ndarray, dictionary: list[str]) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(pa.array(codes, type=pa.int16()), pa.array(dictionary, type=pa.string()))


def _write_parquet(frame_or_table: pd.DataFrame | pa.Table, path: Path) -> int:
    table = frame_or_table if isinstance(frame_or_table, pa.Table) else pa.Table.from_pandas(frame_or_table, preserve_index=False)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=1_000_000)
    tmp.replace(path)
    return int(table.num_rows)


class MimicIVAdapter:
    """MIMIC-IV 3.1 adapter (see the module docstring for every mapping decision).

    ``raw_dir`` holds the extracted ``icu/*.csv.gz`` and ``hosp/labevents.csv.gz``; tables
    that are not extracted (``hosp/patients``, the dictionaries) are read as single members
    of ``zip_path`` (default: ``<raw_dir>.zip``) without extracting the archive.
    """

    name = "mimic_iv"

    def __init__(
        self,
        raw_dir: str | Path,
        cache_dir: str | Path,
        *,
        zip_path: str | Path | None = None,
        include_noninvasive_ventilation: bool = False,
        include_opioids: bool = False,
        include_angiotensin: bool = True,
        poc_glucose: bool = True,
        ph_specimen: str = "blood_gas",
        min_age: float = 18,
        chunksize: int = 5_000_000,
        rebuild: bool = False,
        verbose: bool = True,
    ) -> None:
        if ph_specimen not in ("blood_gas", "arterial"):
            raise ValueError("ph_specimen must be 'blood_gas' (pooled) or 'arterial'")
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir)
        self.zip_path = Path(zip_path) if zip_path is not None else Path(str(self.raw_dir).rstrip("/\\") + ".zip")
        self.options: dict[str, Any] = {
            "include_noninvasive_ventilation": bool(include_noninvasive_ventilation),
            "include_opioids": bool(include_opioids),
            "include_angiotensin": bool(include_angiotensin),
            "poc_glucose": bool(poc_glucose),
            "ph_specimen": ph_specimen,
            "min_age": float(min_age),
            "excluded_statuses": list(EXCLUDED_STATUSES),
            "continuous_categories": list(CONTINUOUS_CATEGORIES),
        }
        self.chunksize = int(chunksize)
        self.rebuild = bool(rebuild)
        self.verbose = bool(verbose)
        self._frames: dict[str, pd.DataFrame] = {}
        self._sorted_ids: dict[str, np.ndarray] = {}
        self._rebuilt: set[str] = set()
        self._stay_counts: dict[str, int] = {}
        self._vaso_counts: dict[str, Any] = {}
        self._event_counts: dict[str, Counter] = defaultdict(Counter)
        self._feature_pass_counts: dict[str, Counter] = defaultdict(Counter)
        self._intervention_counts: Counter = Counter()

    # ------------------------------------------------------------------ raw-file access

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[mimic_iv {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

    def _zip_member(self, zf: zipfile.ZipFile, relative: str) -> str:
        for name in zf.namelist():
            if name == relative or name.endswith("/" + relative):
                return name
        raise FileNotFoundError(f"{relative} is neither under {self.raw_dir} nor in {self.zip_path}")

    @contextlib.contextmanager
    def _open(self, relative: str) -> Iterator[Any]:
        """Path of an extracted table, else a binary stream of the zip member."""
        path = self.raw_dir / relative
        if path.exists():
            yield path
            return
        if not self.zip_path.exists():
            raise FileNotFoundError(f"{relative} not found under {self.raw_dir} and no archive at {self.zip_path}")
        with zipfile.ZipFile(self.zip_path) as zf, zf.open(self._zip_member(zf, relative)) as stream:
            yield stream

    def _signature(self, relative: str) -> dict[str, Any]:
        path = self.raw_dir / relative
        if path.exists():
            stat = path.stat()
            return {"file": relative, "path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        if self.zip_path.exists():
            with zipfile.ZipFile(self.zip_path) as zf:
                info = zf.getinfo(self._zip_member(zf, relative))
                return {"file": relative, "zip": str(self.zip_path), "member": info.filename, "size": int(info.file_size), "crc": int(info.CRC)}
        return {"file": relative, "missing": True}

    def _read_csv(self, relative: str, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """Chunks of a gzipped MIMIC csv (``chunksize`` rows)."""
        with self._open(relative) as source:
            yield from pd.read_csv(source, compression="gzip", chunksize=self.chunksize, low_memory=False, **kwargs)

    # ------------------------------------------------------------------ cache management

    @property
    def _manifest_path(self) -> Path:
        return self.cache_dir / "cache_manifest.json"

    def _read_manifest(self) -> dict[str, Any]:
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return {}

    def _cache_key(self, table: str) -> dict[str, Any]:
        """Cache version, source identity (name, size, mtime or zip CRC; not the spelling of
        raw_dir) and the cached itemid superset."""
        location = ("path", "zip")
        return {
            "cache_version": CACHE_VERSION,
            "sources": [{k: v for k, v in self._signature(r).items() if k not in location} for r in TABLE_FILES[table]],
            "items": _cache_items(table),
        }

    def _cache_path(self, table: str) -> Path:
        return self.cache_dir / f"{table}.parquet"

    def _ensure(self, table: str) -> None:
        key = self._cache_key(table)
        fresh = self._cache_path(table).exists() and self._read_manifest().get(table, {}).get("key") == key
        if fresh and not (self.rebuild and table not in self._rebuilt):
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"building cache {table} ...")
        started = time.perf_counter()
        builder = getattr(self, f"_build_{table}")
        stats = builder()
        seconds = time.perf_counter() - started
        manifest = self._read_manifest()  # re-read: another table may have been written meanwhile
        manifest[table] = {
            "key": key,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(seconds, 1),
            "file_bytes": int(self._cache_path(table).stat().st_size),
            **stats,
        }
        tmp = self._manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(self._manifest_path)
        self._rebuilt.add(table)
        self._log(f"built {table}: {stats.get('rows', '?')} rows in {seconds:.0f} s")

    def build_cache(self, tables: Sequence[str] | None = None) -> dict[str, Any]:
        """Build (or validate) the parquet caches; returns the cache manifest."""
        order = ["stays", "inputevents", "outputevents", "procedureevents", "chartevents", "labevents"]
        for table in tables or order:
            self._ensure(table)
        return self._read_manifest()

    def _load(self, table: str) -> pd.DataFrame:
        if table not in self._frames:
            self._ensure(table)
            started = time.perf_counter()
            frame = pd.read_parquet(self._cache_path(table))
            for column in frame.columns:
                if pd.api.types.is_datetime64_any_dtype(frame[column]):
                    frame[column] = frame[column].astype("datetime64[ns]")
            self._frames[table] = frame
            if "stay_id" in frame.columns and table != "stays":
                ids = frame["stay_id"].to_numpy(dtype=np.int64)
                if len(ids) and not (np.diff(ids) >= 0).all():
                    raise ValueError(f"cache {table} is not sorted by stay_id; rebuild it")
                self._sorted_ids[table] = ids
            self._log(f"loaded {table}: {len(frame)} rows in {time.perf_counter() - started:.1f} s")
        return self._frames[table]

    # ------------------------------------------------------------------ cache builders

    def _build_stays(self) -> dict[str, Any]:
        icu = pd.concat(
            self._read_csv("icu/icustays.csv.gz", usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"]),
            ignore_index=True,
        )
        patients = pd.concat(
            self._read_csv("hosp/patients.csv.gz", usecols=["subject_id", "anchor_age", "anchor_year", "anchor_year_group"]),
            ignore_index=True,
        )
        frame = icu.merge(patients, on="subject_id", how="left")
        frame["intime"] = _parse_time(frame["intime"])
        frame["outtime"] = _parse_time(frame["outtime"])
        frame["age_at_intime"] = frame["anchor_age"] + (frame["intime"].dt.year - frame["anchor_year"])
        frame["subject_id"] = frame["subject_id"].astype(np.int64)
        frame["stay_id"] = frame["stay_id"].astype(np.int64)
        frame = frame.sort_values("stay_id").reset_index(drop=True)
        rows = _write_parquet(frame, self._cache_path("stays"))
        return {"rows": rows, "subjects": int(frame["subject_id"].nunique()), "without_patient_row": int(frame["anchor_age"].isna().sum())}

    def _build_inputevents(self) -> dict[str, Any]:
        items = set(_cache_items("inputevents"))
        columns = ["stay_id", "itemid", "starttime", "endtime", "rate", "rateuom", "amount", "amountuom",
                   "ordercategorydescription", "statusdescription"]
        parts = []
        read = 0
        for chunk in self._read_csv("icu/inputevents.csv.gz", usecols=columns,
                                    dtype={"stay_id": "float64", "itemid": "int64", "rate": "float64", "amount": "float64"}):
            read += len(chunk)
            part = chunk[chunk["itemid"].isin(items) & chunk["stay_id"].notna()]
            if len(part):
                parts.append(part)
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
        frame["stay_id"] = frame["stay_id"].astype(np.int64)
        frame["itemid"] = frame["itemid"].astype(np.int64)
        frame["starttime"] = _parse_time(frame["starttime"])
        frame["endtime"] = _parse_time(frame["endtime"])
        for column in ("rateuom", "amountuom", "ordercategorydescription", "statusdescription"):
            frame[column] = frame[column].astype("category")
        frame = frame.sort_values(["stay_id", "starttime"], kind="stable").reset_index(drop=True)
        rows = _write_parquet(frame, self._cache_path("inputevents"))
        return {"rows": rows, "rows_read": read, "rows_by_item": {int(k): int(v) for k, v in frame["itemid"].value_counts().items()}}

    def _build_outputevents(self) -> dict[str, Any]:
        items = set(_cache_items("outputevents"))
        parts = []
        read = 0
        for chunk in self._read_csv("icu/outputevents.csv.gz", usecols=["stay_id", "itemid", "charttime", "value", "valueuom"],
                                    dtype={"stay_id": "float64", "itemid": "int64", "value": "float64"}):
            read += len(chunk)
            part = chunk[chunk["itemid"].isin(items) & chunk["stay_id"].notna()]
            if len(part):
                parts.append(part)
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["stay_id", "itemid", "charttime", "value", "valueuom"])
        frame["stay_id"] = frame["stay_id"].astype(np.int64)
        frame["itemid"] = frame["itemid"].astype(np.int64)
        frame["time"] = _parse_time(frame["charttime"])
        frame["valueuom"] = frame["valueuom"].astype("category")
        frame = frame.drop(columns=["charttime"]).sort_values(["stay_id", "time"], kind="stable").reset_index(drop=True)
        rows = _write_parquet(frame, self._cache_path("outputevents"))
        return {"rows": rows, "rows_read": read, "rows_by_item": {int(k): int(v) for k, v in frame["itemid"].value_counts().items()}}

    def _build_procedureevents(self) -> dict[str, Any]:
        items = set(_cache_items("procedureevents"))
        columns = ["stay_id", "itemid", "starttime", "endtime", "statusdescription"]
        parts = []
        for chunk in self._read_csv("icu/procedureevents.csv.gz", usecols=columns, dtype={"stay_id": "float64", "itemid": "int64"}):
            part = chunk[chunk["itemid"].isin(items) & chunk["stay_id"].notna()]
            if len(part):
                parts.append(part)
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
        frame["stay_id"] = frame["stay_id"].astype(np.int64)
        frame["itemid"] = frame["itemid"].astype(np.int64)
        frame["starttime"] = _parse_time(frame["starttime"])
        frame["endtime"] = _parse_time(frame["endtime"])
        frame["statusdescription"] = frame["statusdescription"].astype("category")
        frame = frame.sort_values(["stay_id", "starttime"], kind="stable").reset_index(drop=True)
        rows = _write_parquet(frame, self._cache_path("procedureevents"))
        return {"rows": rows, "rows_by_item": {int(k): int(v) for k, v in frame["itemid"].value_counts().items()}}

    def _build_chartevents(self) -> dict[str, Any]:
        """One streaming pass over chartevents: rows of the cached itemids, all stays."""
        items = np.array(_cache_items("chartevents"), dtype=np.int64)
        units = _Codes()
        stay_parts, time_parts, item_parts, value_parts, unit_parts = [], [], [], [], []
        raw_counts: Counter = Counter()
        read = 0
        started = time.perf_counter()
        for number, chunk in enumerate(
            self._read_csv("icu/chartevents.csv.gz", usecols=["stay_id", "itemid", "charttime", "valuenum", "valueuom"],
                           dtype={"stay_id": "float64", "itemid": "int64", "valuenum": "float64", "valueuom": "object"})
        ):
            read += len(chunk)
            part = chunk[chunk["itemid"].isin(items)]
            for (itemid, uom, missing), n in part.groupby(
                [part["itemid"], part["valueuom"].fillna(""), part["valuenum"].isna()]
            ).size().items():
                raw_counts[(int(itemid), str(uom), bool(missing))] += int(n)
            part = part[part["valuenum"].notna() & part["stay_id"].notna()]
            times = _parse_time(part["charttime"])
            keep = times.notna().to_numpy()
            part, times = part[keep], times[keep]
            stay_parts.append(part["stay_id"].to_numpy(dtype=np.int64))
            time_parts.append(times.to_numpy(dtype="datetime64[ns]").astype(np.int64))
            item_parts.append(part["itemid"].to_numpy(dtype=np.int32))
            value_parts.append(part["valuenum"].to_numpy(dtype=np.float64))
            unit_parts.append(units.encode(part["valueuom"]))
            if number % 10 == 9:
                self._log(f"  chartevents: {read:,} rows read, {sum(len(p) for p in stay_parts):,} kept, {time.perf_counter() - started:.0f} s")
        table = self._sorted_table(stay_parts, time_parts, item_parts, value_parts, unit_parts, units.dictionary(), "valueuom")
        rows = _write_parquet(table, self._cache_path("chartevents"))
        return {
            "rows": rows,
            "rows_read": read,
            "raw_counts": [{"itemid": i, "valueuom": u, "valuenum_missing": m, "n": n} for (i, u, m), n in sorted(raw_counts.items())],
        }

    @staticmethod
    def _sorted_table(stay_parts, time_parts, item_parts, value_parts, unit_parts, dictionary, unit_name, extra=None) -> pa.Table:
        """Concatenate the per-chunk arrays and sort by (stay_id, time) with bounded copies."""
        stay = np.concatenate(stay_parts) if stay_parts else np.zeros(0, dtype=np.int64)
        stamp = np.concatenate(time_parts) if time_parts else np.zeros(0, dtype=np.int64)
        order = np.lexsort((stamp, stay))
        stay, stamp = stay[order], stamp[order]
        columns = {
            "stay_id": pa.array(stay, type=pa.int64()),
            "time": pa.array(stamp.view("datetime64[ns]"), type=pa.timestamp("ns")),
        }
        del stay, stamp
        item = (np.concatenate(item_parts) if item_parts else np.zeros(0, dtype=np.int32))[order]
        columns["itemid"] = pa.array(item, type=pa.int32())
        del item
        value = (np.concatenate(value_parts) if value_parts else np.zeros(0))[order]
        columns["valuenum"] = pa.array(value, type=pa.float64())
        del value
        codes = (np.concatenate(unit_parts) if unit_parts else np.zeros(0, dtype=np.int16))[order]
        columns[unit_name] = _dictionary_column(codes, dictionary or [""])
        for name, (parts, kind) in (extra or {}).items():
            data = (np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64))[order]
            columns[name] = pa.array(data, type=kind)
        return pa.table(columns)

    def _build_labevents(self) -> dict[str, Any]:
        """One streaming pass over labevents: cached items of ICU subjects, assigned to stays."""
        stays = self._load("stays")
        stays = stays[stays["intime"].notna() & stays["outtime"].notna()][["subject_id", "stay_id", "intime", "outtime"]]
        subjects = np.unique(stays["subject_id"].to_numpy(dtype=np.int64))
        items = np.array([i for i in _cache_items("labevents") if i != SPECIMEN_TYPE_ITEM], dtype=np.int64)
        wanted = np.array(_cache_items("labevents"), dtype=np.int64)
        units = _Codes()
        stay_parts, time_parts, item_parts, value_parts, unit_parts, specimen_parts = [], [], [], [], [], []
        specimen_types = []
        raw_counts: Counter = Counter()
        read = in_window = 0
        started = time.perf_counter()
        for number, chunk in enumerate(
            self._read_csv("hosp/labevents.csv.gz",
                           usecols=["subject_id", "specimen_id", "itemid", "charttime", "value", "valuenum", "valueuom"],
                           dtype={"subject_id": "float64", "specimen_id": "float64", "itemid": "int64", "valuenum": "float64",
                                  "value": "object", "valueuom": "object"})
        ):
            read += len(chunk)
            chunk = chunk[chunk["itemid"].isin(wanted) & chunk["subject_id"].notna()]
            chunk = chunk[np.isin(chunk["subject_id"].to_numpy(dtype=np.int64), subjects)]
            chunk = chunk.assign(subject_id=chunk["subject_id"].astype(np.int64),
                                 specimen_id=chunk["specimen_id"].fillna(-1).astype(np.int64))
            types = chunk[chunk["itemid"] == SPECIMEN_TYPE_ITEM]
            if len(types):
                specimen_types.append(types[["specimen_id", "value"]])
            part = chunk[chunk["itemid"].isin(items)]
            for (itemid, uom, missing), n in part.groupby(
                [part["itemid"], part["valueuom"].fillna(""), part["valuenum"].isna()]
            ).size().items():
                raw_counts[(int(itemid), str(uom), bool(missing))] += int(n)
            part = part[part["valuenum"].notna()]
            part = part.assign(time=_parse_time(part["charttime"]))
            part = part[part["time"].notna()]
            joined = part[["subject_id", "specimen_id", "itemid", "time", "valuenum", "valueuom"]].merge(stays, on="subject_id")
            joined = joined[(joined["time"] >= joined["intime"]) & (joined["time"] <= joined["outtime"])]
            in_window += len(joined)
            stay_parts.append(joined["stay_id"].to_numpy(dtype=np.int64))
            time_parts.append(joined["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64))
            item_parts.append(joined["itemid"].to_numpy(dtype=np.int32))
            value_parts.append(joined["valuenum"].to_numpy(dtype=np.float64))
            unit_parts.append(units.encode(joined["valueuom"]))
            specimen_parts.append(joined["specimen_id"].to_numpy(dtype=np.int64))
            if number % 10 == 9:
                self._log(f"  labevents: {read:,} rows read, {in_window:,} in ICU stays, {time.perf_counter() - started:.0f} s")
        table = self._sorted_table(stay_parts, time_parts, item_parts, value_parts, unit_parts, units.dictionary(), "valueuom",
                                   extra={"specimen_id": (specimen_parts, pa.int64())})
        type_frame = (pd.concat(specimen_types, ignore_index=True).drop_duplicates("specimen_id")
                      if specimen_types else pd.DataFrame(columns=["specimen_id", "value"]))
        type_frame = type_frame[type_frame["value"].notna()]
        type_map = pd.Series(type_frame["value"].astype(str).to_numpy(), index=type_frame["specimen_id"].to_numpy(dtype=np.int64))
        specimen = pd.Series(table.column("specimen_id").to_numpy())
        specimen_type = specimen.map(type_map)
        table = table.append_column("specimen_type", pa.array(specimen_type.astype(object).where(specimen_type.notna(), None).tolist(),
                                                              type=pa.string()).dictionary_encode())
        rows = _write_parquet(table, self._cache_path("labevents"))
        ph = specimen_type[pd.Series(table.column("itemid").to_numpy()) == PH_ITEM]
        return {
            "rows": rows,
            "rows_read": read,
            "rows_in_icu_stays": int(in_window),
            "ph_specimen_types": {str(k): int(v) for k, v in ph.fillna("<none>").value_counts().items()},
            "raw_counts_icu_subjects": [{"itemid": i, "valueuom": u, "valuenum_missing": m, "n": n} for (i, u, m), n in sorted(raw_counts.items())],
        }

    # ------------------------------------------------------------------ contract: stays / vasopressors

    def stays(self) -> pd.DataFrame:
        """Adult ICU stays: subject_id, stay_id, intime, outtime."""
        frame = self._load("stays")
        valid = frame["intime"].notna() & frame["outtime"].notna() & (frame["outtime"] > frame["intime"])
        adult = (frame["anchor_age"] >= self.options["min_age"]) & (frame["age_at_intime"] >= self.options["min_age"])
        self._stay_counts = {
            "icustays": int(len(frame)),
            "invalid_times": int((~valid).sum()),
            "below_min_age": int((valid & ~adult).sum()),
            "stays": int((valid & adult).sum()),
        }
        out = frame.loc[valid & adult, ["subject_id", "stay_id", "intime", "outtime"]].reset_index(drop=True)
        out["subject_id"] = out["subject_id"].astype(np.int64)
        out["stay_id"] = out["stay_id"].astype(np.int64)
        return out

    def _drug_items(self, kind: str) -> dict[int, str]:
        if kind == "vasopressor":
            return {**VASOPRESSOR_ITEMS, **(ANGIOTENSIN_ITEMS if self.options["include_angiotensin"] else {})}
        if kind == "sedation":
            return {**SEDATIVE_ITEMS, **(OPIOID_ITEMS if self.options["include_opioids"] else {})}
        raise ValueError(kind)

    def _continuous_rows(self, frame: pd.DataFrame, items: dict[int, str]) -> tuple[pd.DataFrame, dict[str, int]]:
        """Continuous infusion periods with rate > 0 (no boluses, no pushes, no rewritten rows)."""
        counts = {}
        rows = frame[frame["itemid"].isin(list(items))]
        counts["rows"] = int(len(rows))
        rows = rows[rows["ordercategorydescription"].isin(CONTINUOUS_CATEGORIES)]
        counts["after_continuous_category"] = int(len(rows))
        rows = rows[rows["rate"] > 0]
        counts["after_rate_positive"] = int(len(rows))
        rows = rows[~rows["statusdescription"].isin(EXCLUDED_STATUSES).fillna(False).astype(bool)]
        counts["after_status"] = int(len(rows))
        rows = rows[rows["starttime"].notna() & (rows["endtime"].isna() | (rows["endtime"] >= rows["starttime"]))]
        counts["kept"] = int(len(rows))
        return rows, counts

    def vasopressor_intervals(self) -> pd.DataFrame:
        """Continuous vasopressor infusions with rate > 0: stay_id, start, end, drug."""
        frame = self._load("inputevents")
        items = self._drug_items("vasopressor")
        rows, counts = self._continuous_rows(frame, items)
        out = pd.DataFrame(
            {
                "stay_id": rows["stay_id"].to_numpy(dtype=np.int64),
                "start": rows["starttime"].to_numpy(dtype="datetime64[ns]"),
                "end": rows["endtime"].to_numpy(dtype="datetime64[ns]"),
                "drug": rows["itemid"].map(items).astype(object).to_numpy(),
            }
        )
        first = out.sort_values(["stay_id", "start"], kind="stable").groupby("stay_id").head(1)
        self._vaso_counts = {
            **counts,
            "stays": int(out["stay_id"].nunique()),
            "first_drug": {str(k): int(v) for k, v in first["drug"].value_counts().items()},
            "rows_by_drug": {str(k): int(v) for k, v in out["drug"].value_counts().items()},
        }
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ contract: events

    def _count(self, feature: str, key: str, n: int) -> None:
        if n:
            self._event_counts[feature][key] += int(n)

    def _convert(self, rows: pd.DataFrame, item_feature: dict[int, str], item_unit: dict[int, str]) -> pd.DataFrame:
        """Canonical units and plausibility screen; rows: stay_id, time, itemid, valuenum, valueuom."""
        if rows.empty:
            return rows.assign(feature=pd.Series(dtype=object), value=pd.Series(dtype=float))
        observed = rows["valueuom"].astype(object).where(rows["valueuom"].notna(), "").astype(str)
        keys = pd.DataFrame({"itemid": rows["itemid"].to_numpy(dtype=np.int64), "uom": observed.to_numpy()})
        factor = np.full(len(rows), np.nan)
        offset = np.zeros(len(rows))
        low = np.full(len(rows), np.nan)
        high = np.full(len(rows), np.nan)
        features = np.empty(len(rows), dtype=object)
        for (itemid, uom), index in keys.groupby(["itemid", "uom"]).indices.items():
            feature = item_feature[int(itemid)]
            features[index] = feature
            low[index] = CANONICAL_SPECS[feature].low
            high[index] = CANONICAL_SPECS[feature].high
            unit = uom if normalize_unit(uom) else item_unit[int(itemid)]  # empty unit: the item's documented unit
            conversion = conversion_for(feature, unit)
            self._count(feature, "rows", len(index))
            if conversion is None:
                self._count(feature, f"unit_rejected:{uom}", len(index))
                continue
            if normalize_unit(unit) != normalize_unit(item_unit[int(itemid)]):
                self._count(feature, f"unit_other:{uom}", len(index))
            if not conversion.is_identity:
                self._count(feature, f"converted:{conversion.name}", len(index))
            factor[index] = conversion.factor
            offset[index] = conversion.offset
        value = rows["valuenum"].to_numpy(dtype=float) * factor + offset
        converted = ~np.isnan(factor)
        plausible = converted & (value >= low) & (value <= high)
        dropped = pd.Series(features[converted & ~plausible]).value_counts()
        for feature, n in dropped.items():
            self._count(str(feature), "implausible", int(n))
        out = rows.loc[plausible].copy()
        out["feature"] = features[plausible]
        out["value"] = value[plausible]
        return out

    def _chart_events(self, stay_ids: np.ndarray, features: Collection[str] | None = None) -> pd.DataFrame:
        items = {k: v for k, v in CHART_ITEMS.items() if self.options["poc_glucose"] or k not in POC_GLUCOSE_ITEMS}
        if features is not None:
            items = {k: v for k, v in items.items() if v[0] in features}
            if not items:
                return pd.DataFrame(columns=list(EVENT_OUTPUT_COLUMNS))
        frame = self._load("chartevents")
        rows = _slice_sorted(frame, self._sorted_ids["chartevents"], stay_ids)
        rows = rows[rows["itemid"].isin(list(items))]
        out = self._convert(rows, {k: v[0] for k, v in items.items()}, {k: v[1] for k, v in items.items()})
        modality = out["itemid"].map({k: v[2] for k, v in items.items()})
        return pd.DataFrame(
            {
                "stay_id": out["stay_id"].to_numpy(dtype=np.int64),
                "time": out["time"].to_numpy(dtype="datetime64[ns]"),
                "feature": out["feature"].to_numpy(dtype=object),
                "value": out["value"].to_numpy(dtype=float),
                "modality": modality.astype(object).where(modality.notna(), None).to_numpy(dtype=object),
            }
        )

    def _lab_events(self, stay_ids: np.ndarray, features: Collection[str] | None = None) -> pd.DataFrame:
        items = LAB_ITEMS if features is None else {k: v for k, v in LAB_ITEMS.items() if v[0] in features}
        if not items:
            return pd.DataFrame(columns=list(EVENT_OUTPUT_COLUMNS))
        frame = self._load("labevents")
        rows = _slice_sorted(frame, self._sorted_ids["labevents"], stay_ids)
        rows = rows[rows["itemid"].isin(list(items))]
        if self.options["ph_specimen"] == "arterial":
            not_arterial = (rows["itemid"] == PH_ITEM) & (rows["specimen_type"].astype(object) != ARTERIAL_SPECIMEN)
            self._count("pH", "not_arterial_specimen", int(not_arterial.sum()))
            rows = rows[~not_arterial]
        out = self._convert(rows, {k: v[0] for k, v in items.items()}, {k: v[1] for k, v in items.items()})
        return pd.DataFrame(
            {
                "stay_id": out["stay_id"].to_numpy(dtype=np.int64),
                "time": out["time"].to_numpy(dtype="datetime64[ns]"),
                "feature": out["feature"].to_numpy(dtype=object),
                "value": out["value"].to_numpy(dtype=float),
                "modality": np.full(len(out), None, dtype=object),
            }
        )

    def _urine_events(self, stay_ids: np.ndarray) -> pd.DataFrame:
        frame = self._load("outputevents")
        rows = _slice_sorted(frame, self._sorted_ids["outputevents"], stay_ids)
        rows = rows[rows["value"].notna() & rows["time"].notna()]
        spec = CANONICAL_SPECS["urine_output"]
        urine = rows[rows["itemid"].isin(list(URINE_ITEMS))]
        self._count("urine_output", "rows", len(urine))
        uom = urine["valueuom"].astype(object)
        normalized = {u: normalize_unit(u) for u in uom.dropna().unique()}
        unit_ok = uom.isna() | uom.map(normalized).isin(["mL", ""])
        self._count("urine_output", "unit_rejected", int((~unit_ok).sum()))
        urine = urine[unit_ok.to_numpy(dtype=bool)]
        irrigation = rows[rows["itemid"].isin([GU_IRRIGANT_OUT, GU_IRRIGANT_IN])]
        if len(irrigation):
            signed = irrigation["value"].where(irrigation["itemid"] == GU_IRRIGANT_OUT, -irrigation["value"])
            pairs = irrigation.assign(signed=signed, is_out=irrigation["itemid"] == GU_IRRIGANT_OUT,
                                      is_in=irrigation["itemid"] == GU_IRRIGANT_IN)
            net = pairs.groupby(["stay_id", "time"], as_index=False).agg(value=("signed", "sum"), has_out=("is_out", "any"),
                                                                           has_in=("is_in", "any"))
            self._count("urine_output", "irrigation_times", len(net))
            paired = net["has_out"] & net["has_in"]
            self._count("urine_output", "irrigation_unpaired_dropped", int((~paired).sum()))
            self._count("urine_output", "irrigation_negative_net_dropped", int((paired & (net["value"] < 0)).sum()))
            net = net[paired & (net["value"] >= 0)]
            self._count("urine_output", "irrigation_net_rows", len(net))
            urine = pd.concat([urine[["stay_id", "time", "value"]], net[["stay_id", "time", "value"]]], ignore_index=True)
        plausible = (urine["value"] >= spec.low) & (urine["value"] <= spec.high)
        self._count("urine_output", "implausible", int((~plausible).sum()))
        urine = urine[plausible]
        return pd.DataFrame(
            {
                "stay_id": urine["stay_id"].to_numpy(dtype=np.int64),
                "time": urine["time"].to_numpy(dtype="datetime64[ns]"),
                "feature": np.full(len(urine), "urine_output", dtype=object),
                "value": urine["value"].to_numpy(dtype=float),
                "modality": np.full(len(urine), None, dtype=object),
            }
        )

    def events(self, stay_ids: Sequence[int]) -> pd.DataFrame:
        """Numeric features in canonical units for the requested stays only.

        Columns stay_id, time, feature, value, modality (``invasive``/``noninvasive`` for
        SBP/DBP/MAP, else None). Implausible values and unconvertible units are dropped and
        counted in :meth:`provenance`.
        """
        ids = np.unique(np.asarray(list(stay_ids), dtype=np.int64))
        frames = [self._chart_events(ids), self._lab_events(ids), self._urine_events(ids)]
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["stay_id", "time"], kind="stable").reset_index(drop=True)
        for feature, n in out["feature"].value_counts().items():
            self._count(str(feature), "kept", int(n))
        self._event_counts["_calls"]["stays_requested"] += int(len(ids))
        return out[list(EVENT_OUTPUT_COLUMNS)]

    def feature_events(self, stay_ids: Sequence[int], features: Iterable[str]) -> pd.DataFrame:
        """:meth:`events` restricted to ``features``, reading only the tables they come from.

        Used by task definitions that scan every stay (the AKI creatinine pass). Its
        conversion counts are reported separately (``feature_events_counts``), so
        ``event_counts`` in :meth:`provenance` still describes the episode extraction only.
        """
        wanted = frozenset(str(f) for f in features)
        ids = np.unique(np.asarray(list(stay_ids), dtype=np.int64))
        saved, self._event_counts = self._event_counts, self._feature_pass_counts
        try:
            frames = [self._chart_events(ids, wanted), self._lab_events(ids, wanted)]
            if "urine_output" in wanted:
                frames.append(self._urine_events(ids))
        finally:
            self._event_counts = saved
        frames = [f for f in frames if len(f)]
        if not frames:
            return pd.DataFrame(columns=list(EVENT_OUTPUT_COLUMNS))
        out = pd.concat(frames, ignore_index=True).sort_values(["stay_id", "time"], kind="stable")
        return out.reset_index(drop=True)[list(EVENT_OUTPUT_COLUMNS)]

    # ------------------------------------------------------------------ contract: interventions

    def interventions(self, stay_ids: Sequence[int]) -> pd.DataFrame:
        """ventilation_on (procedureevents) and sedation_on (continuous infusions) intervals."""
        ids = np.unique(np.asarray(list(stay_ids), dtype=np.int64))
        procedures = self._load("procedureevents")
        vent_items = [VENT_INVASIVE_ITEM] + ([VENT_NONINVASIVE_ITEM] if self.options["include_noninvasive_ventilation"] else [])
        vent = procedures[procedures["stay_id"].isin(ids) & procedures["itemid"].isin(vent_items)]
        vent = vent[~vent["statusdescription"].isin(EXCLUDED_STATUSES).fillna(False).astype(bool) & vent["starttime"].notna()]
        vent = vent[vent["endtime"].isna() | (vent["endtime"] >= vent["starttime"])]
        drugs = self._load("inputevents")
        sedation, _ = self._continuous_rows(drugs[drugs["stay_id"].isin(ids)], self._drug_items("sedation"))
        self._intervention_counts["ventilation_on"] += int(len(vent))
        self._intervention_counts["sedation_on"] += int(len(sedation))
        frames = [
            pd.DataFrame({"stay_id": vent["stay_id"].to_numpy(dtype=np.int64), "start": vent["starttime"].to_numpy(dtype="datetime64[ns]"),
                          "end": vent["endtime"].to_numpy(dtype="datetime64[ns]"), "feature": "ventilation_on"}),
            pd.DataFrame({"stay_id": sedation["stay_id"].to_numpy(dtype=np.int64),
                          "start": sedation["starttime"].to_numpy(dtype="datetime64[ns]"),
                          "end": sedation["endtime"].to_numpy(dtype="datetime64[ns]"), "feature": "sedation_on"}),
        ]
        out = pd.concat(frames, ignore_index=True)
        out["feature"] = out["feature"].astype(object)
        return out[list(INTERVENTION_OUTPUT_COLUMNS)].sort_values(["stay_id", "start"], kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------ contract: metadata

    def available_features(self) -> frozenset[str]:
        """All 17 canonical features have a MIMIC-IV source (module docstring)."""
        mapped = {f for f, _, _ in CHART_ITEMS.values()} | {f for f, _ in LAB_ITEMS.values()} | {"urine_output", "ventilation_on", "sedation_on"}
        return frozenset(f for f in CANONICAL_FEATURES if f in mapped)

    def _labels(self) -> dict[int, str]:
        labels: dict[int, str] = {}
        for relative in DICTIONARY_FILES:
            try:
                for chunk in self._read_csv(relative, usecols=["itemid", "label"]):
                    labels.update(zip(chunk["itemid"].astype(int), chunk["label"].astype(str)))
            except (FileNotFoundError, KeyError, ValueError):
                continue
        return labels

    def provenance(self) -> dict[str, Any]:
        labels = self._labels()

        def named(items: dict[int, Any]) -> dict[str, str]:
            return {str(k): labels.get(k, "") for k in sorted(items)}

        manifest = self._read_manifest()
        chart_items = {k: v for k, v in CHART_ITEMS.items() if self.options["poc_glucose"] or k not in POC_GLUCOSE_ITEMS}
        return {
            "adapter": "src.icu.v2.extract.mimic_iv.MimicIVAdapter",
            "source_version": SOURCE_VERSION,
            "cache_version": CACHE_VERSION,
            "raw_dir": str(self.raw_dir),
            "zip_path": str(self.zip_path),
            "cache_dir": str(self.cache_dir),
            "options": self.options,
            "raw_files": [self._signature(r) for t in TABLE_FILES for r in TABLE_FILES[t]],
            "items": {
                "chartevents": {str(k): {"feature": f, "unit": u, "modality": m, "label": labels.get(k, "")} for k, (f, u, m) in chart_items.items()},
                "labevents": {str(k): {"feature": f, "unit": u, "label": labels.get(k, "")} for k, (f, u) in LAB_ITEMS.items()},
                "urine_output": {**named(URINE_ITEMS), "net_of_irrigation": f"{GU_IRRIGANT_OUT} minus {GU_IRRIGANT_IN} at the same charttime"},
                "vasopressor": named(self._drug_items("vasopressor")),
                "sedation_on": named(self._drug_items("sedation")),
                "ventilation_on": named({VENT_INVASIVE_ITEM: 1, **({VENT_NONINVASIVE_ITEM: 1} if self.options["include_noninvasive_ventilation"] else {})}),
                "excluded_drug_items": {str(k): v for k, v in EXCLUDED_DRUG_ITEMS.items()},
                "cached_not_mapped": {str(k): v for k, v in {**CHART_CHECK_ITEMS, **LAB_CHECK_ITEMS}.items()},
            },
            "decisions": [
                "time = charttime (labs: specimen collection time); labs assigned to stays by subject_id and charttime in [intime, outtime]",
                "BP rows keep their modality; common.episode_matrix prefers invasive values per bin",
                "Temp degF -> degC; values outside CANONICAL_SPECS ranges dropped after conversion",
                "Glucose = lab 50931 + POC finger stick 225664 (empty valueuom read as mg/dL); 220621/226537/50809 excluded",
                "pH 50820 " + ("arterial specimens only" if self.options["ph_specimen"] == "arterial" else "arterial and venous blood gas pooled"),
                "urine_output = 10 urine items + (227489 - 227488) netted per charttime; negative nets and unpaired irrigation rows dropped",
                "drug intervals: [starttime, endtime) of Continuous Med/IV rows with rate > 0; statuses Rewritten and Bolus excluded; Paused/Stopped kept",
                "ventilation_on = procedureevents 225792" + (" + 225794" if self.options["include_noninvasive_ventilation"] else "") + " intervals",
            ],
            "stay_counts": self._stay_counts,
            "vasopressor_counts": self._vaso_counts,
            "event_counts": {k: dict(v) for k, v in self._event_counts.items()},
            "feature_events_counts": {k: dict(v) for k, v in self._feature_pass_counts.items()},
            "intervention_counts": dict(self._intervention_counts),
            "cache": {
                t: {k: v for k, v in entry.items() if k not in ("raw_counts", "raw_counts_icu_subjects", "rows_by_item")}
                for t, entry in manifest.items()
            },
        }


__all__ = ["MimicIVAdapter", "CHART_ITEMS", "LAB_ITEMS", "URINE_ITEMS", "VASOPRESSOR_ITEMS", "SEDATIVE_ITEMS", "OPIOID_ITEMS"]
