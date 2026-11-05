"""HiRID 1.1.1 source adapter for the v2 extraction (external target cohort).

Converts the HiRID raw stage into the four frames of :mod:`src.icu.v2.extract.common`.
Mapping identifiers, units and plausible ranges come from
:data:`src.icu.v2.harmonization.HIRID_TABLE` / :data:`~src.icu.v2.harmonization.CANONICAL_SPECS`
(read-only); this module adds the HiRID record semantics checked on the raw data
(R-H1..R-H3 of ``docs/reviews/v2_harmonization_spec.md``, :func:`run_raw_checks`; outputs
in ``artifacts/v2/harmonization/hirid_*``). Three include rows of HIRID_TABLE are
overridden by the data (RR 5685 and 310, see :data:`DATA_EXCLUDED_OBS_IDS`) and the stay
end uses the last vital sign; both are options.

Data and cache
--------------
Only the **raw stage** is used (``observation_tables`` and ``pharma_records``, 250 parquet
parts each; the 18-variable merged stage is never read). Parts already extracted under
``subset_dir`` (default ``data/raw/hirid_raw_subset/raw_stage``: 200 observation and 120
pharma parts) are read in place; the others are streamed from ``raw_stage/*_parquet.tar.gz``
(``tarfile`` mode ``r|gz``, one member in memory at a time, nothing extracted to disk;
extracted copies are size-checked against the archive members). One pass over each table
writes a compact cache in ``cache_dir`` (per source part: the needed observation rows,
per (patient, variable) summaries, status/value counts; the vasoactive/sedative pharma
rows), then derived tables (patients, infusion and ventilation intervals). Every accessor
is served from the cache; the per-part build is resumable. Full build: 229 s with 4
workers, 2.1 GB.

Evidence (full data: 776,921,131 observation rows, 148 variableids, 33,904 patients;
16,270,399 pharma rows, 533 pharmaids)
--------------------------------------------------------------------------------------
Namespaces. ``variableid`` (observation) and ``pharmaid`` (pharma) are looked up in their
own namespace only. ``pharmaid`` 300 (metronidazole tablets) does occur (12 rows: 10 x 780,
2 x 544) and is never read as RR; observation 300 is RR.

Record filters (codes from hirid.intensivecare.ai "structure of the published data").
Observation ``status`` and pharma ``recordstatus`` are bit fields. Rows with bit 2
(invalidated; 190 observation / 24 pharma rows) or bit 32 (notified, not measured /
administered; 42,258 / 29,986 rows) are dropped; circEWS step1a_2 also drops bit 2.
Out-of-range / bigger-than / smaller-than flags are kept (censored values; the plausibility
screen applies). Lab results are re-entered with later ``entertime`` (creatinine: 114,320 of
190,527 rows are repeats): per (patient, variable, datetime) the most recently entered row
is kept (706,569 repeats dropped in total). ``typeid`` is only 0 fluids / 1 drugs and does
not mark boluses.

Stays. One ICU admission per patient: ``stay_id = subject_id = patientid`` (HiRID gives no
separate stay id), ``intime = admissiontime`` (general_table; observations start at most
1 h before it). ``outtime`` = last HR/BP/SpO2/RR record (``outtime_rule="last_vital"``,
default) rather than the last observation of any kind: the latter runs past the end of
monitoring for 3,002 patients by > 2 h (579 by > 24 h), mostly through APACHE II/IV group
and cumulative fluid-balance entries, and landmark matching then picks such tails: 127 of
1,521 matched controls (8.3%) would get windows reaching past the last vital sign, 49
entirely (empty control episodes). ``outtime_rule="last_observation"`` restores the
literal rule. Adults only (``age >= 18``; the youngest HiRID age is 20).

Numeric features (observation ``variableid``; units from the reference, converted with
CANONICAL_SPECS; values outside the canonical plausible range are dropped and counted):
HR 200; RR 300 only (320 ventilator rate excluded by the table; 310 and 5685 excluded by
the data, ``table_rr_ids=True`` restores them): 5685 is no breaths/min rate (sampled every
~15 s, median 84, 54% > 70, corr 0.19 with 300); 310 behaves as the ventilator's
spontaneous rate (17.4% exact zeros, 97.3% of them under invasive modes; 90.6% of nonzero
values equal 300, which is recorded within 2 min of every 310 row, so nothing is lost);
SpO2 4000/8280 peripheral (4200 SvO2, 20000800 SaO2, 20001000/24000737 ScvO2 excluded);
SBP/DBP/MAP invasive 100/120/110 (54M rows each) and non-invasive 600/620/610 (0.4M) with
``modality`` (PA pressures and CVP excluded); Temp 410 core / 7100 rectal / 400 axillary
degC (medians 37.3 / 37.3 / 37.0); Lactate 24000524 arterial, 24000732/24000485 venous
(median 1.4 mmol/l; 24000564 CSF excluded); pH 20000300 arterial (CSF/drain excluded);
Creatinine 20000600 umol/l -> mg/dL (median 78 umol/l; urine 24000572 excluded, 24000573
absent); Glucose 20005110/24000523/24000585 mmol/l -> mg/dL (medians 7.7/6.8/7.9; the three
ids HiRID itself pools as "serum glucose" vm174; 24000585 carries 98.6% of the rows;
point-of-care/blood-gas glucose is included by the cross-database decision; CSF 24000400
excluded); Bilirubin 20004300 umol/l -> mg/dL (median 13); WBC 20000700 and Platelets
20000110 in ``G/l`` (= 10^9/L, case-sensitive unit; medians 10.3 and 164); urine_output
10020000 (below; urine sodium/urea, drains and fluid balance excluded).

urine_output. 10020000 "Hourly urine volume [ml/h]" is a RATE charted about every 2 h
(501,124 records; interval median 120 min, p25 91, p75 243, p95 485): within patients the
value does not grow with the charting interval (median ratio value@~240 min /
value@~120 min = 1.008 over 17,177 patients). The volume of a record is therefore
``rate x hours since the previous urine record of the patient`` (first record, median
4.2 h after admission: hours since admission), the interval capped at
``urine_max_interval_hours`` (default 12 h; 1.8% of intervals are longer). The event keeps
the record time (never moved back), so a bin sums the volume charted in it, as for
MIMIC-IV outputevents.

Infusions (vasopressor_on, sedation_on). Within one ``infusionid`` the records are a start
(``recordstatus`` 524, dose 0 in 99.9%), periodic records every ~15 min (8, automatic; 520,
manual or rate change) and a stop (776); ``cumulativedose[k] = cumulativedose[k-1] +
givendose[k]`` holds for 99.997% of records (circEWS reads doses the same way), so
``givendose`` at ``givenat[k]`` is the dose delivered in ``(givenat[k-1], givenat[k]]``. A
drug is "on" over the merged segments with dose > 0: an interval starts at the record
before its first positive dose and ends at the last positive-dose record. Instantaneous
records (start and stop bits together, 780) are boluses and never count; single-record
infusions do not either. An infusion whose first record already carries a dose (no start
record; < 0.1%) is back-projected by that dose at the next segment's rate (at most 6 h).
Vasopressors = include rows of HIRID_TABLE: norepinephrine 1000657/1000656/1000462 (35,569 /
13,460 / 21 infusions), epinephrine 1000649/1000650, vasopressin 112/113 (112, "inj 20
U/ml", runs as cv infusions: 62 infusions); 10,634 patients have a vasopressor interval.
Not vasopressors: dobutamine 426 (4,423 patients with infusions), milrinone 1000441,
levosimendan 1000606, terlipressin, and the bolus items 1000658/1000750/1000655/71 (780
records only, no interval). HiRID has no dopamine or phenylephrine. Sedation = propofol
208/1001050 (the 1% items 1000491/1001053 only occur as 780 boluses), midazolam
252/1001051 (251 "inj 1 mg/ml" is 99.7% boluses and stays excluded), dexmedetomidine
1001049/1000847; fentanyl (1000741 Perfusor; 1000251 "inj" is bolus-only) only with
``include_opioids=True``.

ventilation_on (invasive only by default; ``include_niv=True`` adds NIV). Evidence points:
ventilator mode 3845 (15,933 patients, exported about every minute while a ventilator is
connected; 1 Stand By is the most frequent code and means off; NIV codes 12 NIV DGV, 13 NIV
DU, 14 Nasal CPAP, 17 NIV neur adj; invasive codes 2-8, 10, 11, 15; 9, 10, 17 never occur)
and the PEEP setting 2600 (20,386 patients, every ~15 min). A PEEP record takes the class
of the last mode record within 30 min; 4,521 PEEP patients have no mode export and fall
back to the last airway code 15001552 within 24 h (1 intubated / 2 tracheostomy ->
invasive, 3 mask / 4 helmet / 5 mouthpiece -> NIV); PEEP without either (0.3% of PEEP
records) follows ``peep_without_context`` (default: invasive). Points of the requested
class form one interval while consecutive points are <= 60 min apart (0.8% of PEEP gaps
exceed it) and no off/other-class point intervenes; the interval ends 15 min after its last
point or at the next evidence point, whichever is earlier; isolated single points (a
momentary mode switch or one PEEP sample) are dropped. 19,326 patients get invasive
ventilation intervals.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.icu.v2 import harmonization as hz
from src.icu.v2.extract.common import CANONICAL_FEATURES, INTERVENTION_FEATURES, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[4]
SOURCE = "hirid"
ADAPTER_VERSION = "2026-10-21"
#: Layout of the per-part raw caches; bump to force a rebuild of the expensive pass.
RAW_CACHE_VERSION = 1
#: Derived tables (patients, infusion and ventilation intervals); bump to recompute them.
RULES_VERSION = 2

DEFAULT_SUBSET_DIR = ROOT / "data" / "raw" / "hirid_raw_subset" / "raw_stage"
TABLE_LAYOUT: dict[str, dict[str, str]] = {
    "observation": {
        "archive": "raw_stage/observation_tables_parquet.tar.gz",
        "subdir": "observation_tables/parquet",
        "cache": "obs",
    },
    "pharma": {
        "archive": "raw_stage/pharma_records_parquet.tar.gz",
        "subdir": "pharma_records/parquet",
        "cache": "pharma",
    },
}
PART_RE = re.compile(r"^part-(\d+)\.parquet$")
OBS_COLUMNS = ["patientid", "datetime", "entertime", "variableid", "value", "status", "type"]
PHARMA_COLUMNS = [
    "patientid", "pharmaid", "givenat", "enteredentryat", "givendose", "cumulativedose", "fluidamount_calc",
    "cumulfluidamount_calc", "doseunit", "route", "infusionid", "typeid", "subtypeid", "recordstatus",
]

# Status bit fields (hirid.intensivecare.ai/structure-of-the-published-data).
OBS_INVALIDATED = 2
OBS_NOT_MEASURED = 32
PHARMA_INVALIDATED = 2
PHARMA_START = 4
PHARMA_NOT_ADMINISTERED = 32
PHARMA_STOP = 256
OBS_DROP_BITS = OBS_INVALIDATED | OBS_NOT_MEASURED
PHARMA_DROP_BITS = PHARMA_INVALIDATED | PHARMA_NOT_ADMINISTERED


# ---------------------------------------------------------------------------------------
# Identifiers (all taken from harmonization.HIRID_TABLE, namespace-aware)
# ---------------------------------------------------------------------------------------


def _table_rows(namespace: str, feature: str | None = None, decision: str | None = None) -> tuple[hz.MappingRow, ...]:
    return tuple(
        row
        for row in hz.HIRID_TABLE
        if row.namespace == namespace
        and (feature is None or row.feature == feature)
        and (decision is None or row.decision == decision)
    )


#: observation variableid -> include row, for the numeric canonical features.
OBS_FEATURE_ROWS: dict[int, hz.MappingRow] = {
    int(row.identifier): row for row in _table_rows("observation", decision="include") if row.feature in NUMERIC_FEATURES
}
#: HIRID_TABLE include rows that the raw-data checks reject. The adapter drops them unless
#: ``table_rr_ids=True``; recommended change to harmonization.py (both -> exclude).
DATA_EXCLUDED_OBS_IDS: dict[int, str] = {
    5685: "not breaths/min: sampled every ~15 s, median 84, 54% of values > 70, corr 0.19 with id 300 "
    "(R-H2, artifacts/v2/harmonization/hirid_rr_ids_check.csv)",
    310: "ventilator spontaneous rate: 17% exact zeros, 97% of them under invasive modes; otherwise equal to id 300, "
    "which is co-recorded within 2 min for virtually every 310 row (no coverage loss)",
}
VENT_MODE_ID = 3845
AIRWAY_ID = 15001552
PEEP_SET_ID = 2600
VENT_IDS: tuple[int, ...] = (VENT_MODE_ID, AIRWAY_ID, PEEP_SET_ID)
URINE_ID = 10020000
FEATURE_IDS: tuple[int, ...] = tuple(sorted(OBS_FEATURE_ROWS))
OBS_KEEP_IDS: tuple[int, ...] = tuple(sorted(set(FEATURE_IDS) | set(VENT_IDS)))
VITAL_IDS: tuple[int, ...] = tuple(
    sorted(vid for vid, row in OBS_FEATURE_ROWS.items() if row.feature in ("HR", "RR", "SpO2", "SBP", "DBP", "MAP"))
)

#: Ventilator mode codes (ordinal_vars_ref.csv). NIV = names with NIV/CPAP.
VENT_MODES_NIV: dict[int, str] = {
    code: name for code, name in hz.HIRID_VENT_MODE_CODES_ON.items() if "NIV" in name or "CPAP" in name
}
VENT_MODES_INVASIVE: dict[int, str] = {
    code: name for code, name in hz.HIRID_VENT_MODE_CODES_ON.items() if code not in VENT_MODES_NIV
}
VENT_MODES_OFF: dict[int, str] = dict(hz.HIRID_VENT_MODE_CODES_OFF)
AIRWAY_INVASIVE: dict[int, str] = {1: "intubiert", 2: "tracheotomiert"}
AIRWAY_NIV: dict[int, str] = {code: name for code, name in hz.HIRID_AIRWAY_CODES_ON.items() if code not in AIRWAY_INVASIVE}

VASOPRESSOR_ROWS: dict[int, hz.MappingRow] = {
    int(row.identifier): row for row in _table_rows("pharma", "vasopressor_on", "include")
}
SEDATION_ROWS: dict[int, hz.MappingRow] = {int(row.identifier): row for row in _table_rows("pharma", "sedation_on", "include")}
#: Every pharma id the tables mention for the two drug states (include and exclude rows),
#: plus 300 for the namespace check; their raw rows are cached.
RELEVANT_PHARMAIDS: frozenset[int] = frozenset(
    int(row.identifier) for row in hz.HIRID_TABLE if row.namespace == "pharma" and row.feature in ("vasopressor_on", "sedation_on")
) | frozenset({300})

_DRUG_TOKENS: tuple[tuple[str, str], ...] = (
    ("noradrenalin", "norepinephrine"),
    ("adrenalin", "epinephrine"),
    ("vasopressin", "vasopressin"),
    ("dobutrex", "dobutamine"),
    ("corotrop", "milrinone"),
    ("simdax", "levosimendan"),
    ("glypressin", "terlipressin"),
    ("haemopressin", "terlipressin"),
    ("disoprivan", "propofol"),
    ("dormicum", "midazolam"),
    ("dexdor", "dexmedetomidine"),
    ("bupivacain", "bupivacaine"),
    ("sufentanyl", "sufentanil"),
    ("fentanyl", "fentanyl"),
    ("durogesic", "fentanyl patch"),
    ("ultiva", "remifentanil"),
    ("morphin", "morphine"),
    ("temesta", "lorazepam"),
    ("ketalar", "ketamine"),
    ("catapresan", "clonidine"),
    ("pentothal", "thiopental"),
    ("rivotril", "clonazepam"),
    ("metronidazole", "metronidazole"),
)
OPIOID_DRUGS: frozenset[str] = frozenset({"fentanyl"})


def drug_name(label: str) -> str:
    """Generic drug name for a HiRID pharma label (brand names resolved)."""
    text = str(label).casefold()
    for token, name in _DRUG_TOKENS:
        if token in text:
            return name
    return text.strip()


PHARMA_LABELS: dict[int, str] = {
    int(row.identifier): row.label for row in hz.HIRID_TABLE if row.namespace == "pharma"
}


# ---------------------------------------------------------------------------------------
# Raw access: extracted parts and streamed archive members
# ---------------------------------------------------------------------------------------


def part_number(name: str) -> int | None:
    match = PART_RE.match(PurePosixPath(str(name)).name)
    return int(match.group(1)) if match else None


def extracted_parts(directory: str | Path) -> dict[int, Path]:
    """``part-N.parquet`` files of one extracted raw-stage table directory (may not exist)."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    found: dict[int, Path] = {}
    for entry in os.scandir(directory):
        number = part_number(entry.name)
        if number is not None and entry.is_file():
            found[number] = Path(entry.path)
    return found


def iter_archive_parts(
    archive: str | Path, wanted: Callable[[int, int], bool] | None = None, *, bufsize: int = 4 << 20
) -> Iterator[tuple[int, int, bytes | None]]:
    """Stream a raw-stage ``*.tar.gz``: yield ``(part, member size, bytes or None)`` per
    ``part-N.parquet`` member in archive order. Bytes are read (into memory) only when
    ``wanted(part, size)``; nothing is written to disk and member paths are never used."""
    with tarfile.open(archive, mode="r|gz", bufsize=bufsize) as tar:
        for member in tar:
            if not member.isfile():
                continue
            number = part_number(member.name)
            if number is None:
                continue
            data = None
            if wanted is None or wanted(number, int(member.size)):
                handle = tar.extractfile(member)
                data = handle.read() if handle is not None else None
            yield number, int(member.size), data


def _write_parquet(table: pa.Table | pd.DataFrame, path: Path, **kwargs: Any) -> None:
    if isinstance(table, pd.DataFrame):
        table = pa.Table.from_pandas(table, preserve_index=False)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd", **kwargs)
    os.replace(tmp, path)


def _write_json(payload: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _part_stem(part: int) -> str:
    return f"part-{int(part):03d}"


def _status_bits(values: pd.Series | np.ndarray) -> np.ndarray:
    """Status as int64 for bit tests (int16 sources can be negative; low bits survive)."""
    return np.asarray(pd.Series(values).fillna(0).to_numpy(), dtype=np.int64)


OBS_KEPT_SCHEMA = pa.schema(
    [("patientid", pa.int32()), ("variableid", pa.int32()), ("datetime", pa.timestamp("ns")), ("value", pa.float32())]
)


def process_observation_table(table: pa.Table, part: int, out_dir: str | Path) -> dict[str, Any]:
    """Cache one observation part: kept rows + per-(patient, variable) summary + counts."""
    out_dir = Path(out_dir)
    stem = _part_stem(part)
    df = table.select(OBS_COLUMNS).to_pandas()
    del table
    status = _status_bits(df["status"])
    invalid = (status & OBS_INVALIDATED) != 0
    not_measured = ((status & OBS_NOT_MEASURED) != 0) & ~invalid
    valid = ~invalid & ~not_measured
    has_value = df["value"].notna().to_numpy()
    df["type"] = df["type"].fillna("")
    counts = (
        df.assign(has_value=has_value)
        .groupby(["variableid", "status", "type", "has_value"], dropna=False, sort=True)
        .size()
        .rename("n")
        .reset_index()
    )
    counts.insert(0, "part", np.int32(part))
    good = df.loc[valid & has_value, ["variableid", "value"]]
    values = good.groupby("variableid")["value"].agg(n_valid_value="size", vmin="min", vmax="max", vsum="sum")
    values["vsumsq"] = (good["value"].astype(np.float64) ** 2).groupby(good["variableid"]).sum()
    summary = (
        df.loc[valid, ["patientid", "variableid", "datetime"]]
        .groupby(["patientid", "variableid"], sort=True)["datetime"]
        .agg(n_valid="size", first="min", last="max")
        .reset_index()
    )
    summary.insert(0, "part", np.int32(part))
    keep = valid & has_value & df["variableid"].isin(OBS_KEEP_IDS).to_numpy()
    kept = df.loc[keep, ["patientid", "variableid", "datetime", "entertime", "value"]].sort_values(
        ["patientid", "variableid", "datetime", "entertime"], kind="mergesort"
    )
    duplicate = kept.duplicated(["patientid", "variableid", "datetime"], keep="last").to_numpy()
    dup_by_var = pd.Series(duplicate, index=kept["variableid"].to_numpy()).groupby(level=0).sum()
    kept = kept.loc[~duplicate, ["patientid", "variableid", "datetime", "value"]]
    values["n_kept"] = kept.groupby("variableid").size()
    values["n_duplicates_dropped"] = dup_by_var
    values = values.fillna({"n_kept": 0, "n_duplicates_dropped": 0}).reset_index()
    values.insert(0, "part", np.int32(part))
    meta = {
        "part": int(part),
        "raw_cache_version": RAW_CACHE_VERSION,
        "rows": int(len(df)),
        "patients": int(df["patientid"].nunique()),
        "invalidated": int(invalid.sum()),
        "not_measured": int(not_measured.sum()),
        "null_value_valid": int((valid & ~has_value).sum()),
        "kept": int(len(kept)),
        "duplicates_dropped": int(duplicate.sum()),
    }
    _write_parquet(counts, out_dir / f"{stem}.counts.parquet")
    _write_parquet(values, out_dir / f"{stem}.values.parquet")
    _write_parquet(summary, out_dir / f"{stem}.patients.parquet")
    _write_json(meta, out_dir / f"{stem}.json")
    kept_table = pa.Table.from_pandas(kept, preserve_index=False).cast(OBS_KEPT_SCHEMA)
    _write_parquet(
        kept_table,
        out_dir / f"{stem}.parquet",
        row_group_size=131072,
        use_dictionary=["patientid", "variableid"],
        column_encoding={"datetime": "DELTA_BINARY_PACKED"},
    )
    return meta


def process_pharma_table(table: pa.Table, part: int, out_dir: str | Path) -> dict[str, Any]:
    """Cache one pharma part: rows of the relevant pharmaids + per-code counts."""
    out_dir = Path(out_dir)
    stem = _part_stem(part)
    df = table.select(PHARMA_COLUMNS).to_pandas()
    del table
    keys = ["pharmaid", "recordstatus", "typeid", "subtypeid", "doseunit", "route"]
    counts = df.groupby(keys, dropna=False, sort=True).size().rename("n").reset_index()
    counts.insert(0, "part", np.int32(part))
    counts["n_patients"] = df.groupby(keys, dropna=False, sort=True)["patientid"].nunique().to_numpy()
    kept = df[df["pharmaid"].isin(RELEVANT_PHARMAIDS)].reset_index(drop=True)
    kept.insert(0, "part", np.int32(part))
    meta = {
        "part": int(part),
        "raw_cache_version": RAW_CACHE_VERSION,
        "rows": int(len(df)),
        "patients": int(df["patientid"].nunique()),
        "kept": int(len(kept)),
    }
    _write_parquet(counts, out_dir / f"{stem}.counts.parquet")
    _write_json(meta, out_dir / f"{stem}.json")
    _write_parquet(kept, out_dir / f"{stem}.parquet")
    return meta


_PROCESSORS: dict[str, Callable[[pa.Table, int, Path], dict[str, Any]]] = {
    "observation": process_observation_table,
    "pharma": process_pharma_table,
}
_COLUMNS: dict[str, list[str]] = {"observation": OBS_COLUMNS, "pharma": PHARMA_COLUMNS}


def _process_part_file(kind: str, part: int, path: str, out_dir: str) -> dict[str, Any]:
    started = time.perf_counter()
    meta = _PROCESSORS[kind](pq.read_table(path, columns=_COLUMNS[kind]), part, out_dir)
    return {**meta, "source": "extracted", "seconds": round(time.perf_counter() - started, 2)}


def _process_part_bytes(kind: str, part: int, data: bytes, out_dir: str) -> dict[str, Any]:
    started = time.perf_counter()
    meta = _PROCESSORS[kind](pq.read_table(pa.BufferReader(data), columns=_COLUMNS[kind]), part, out_dir)
    return {**meta, "source": "archive", "seconds": round(time.perf_counter() - started, 2)}


# ---------------------------------------------------------------------------------------
# Derivations (pure functions on frames; unit-tested on synthetic data)
# ---------------------------------------------------------------------------------------

_NAT = np.datetime64("NaT", "ns")


def _typed_frame(spec: Mapping[str, str]) -> pd.DataFrame:
    """Empty frame with the given column dtypes (keeps ``.dt`` usable on empty results)."""
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in spec.items()})


INTERVAL_DTYPES = {
    "patientid": "int64", "pharmaid": "int64", "infusionid": "int64", "start": "datetime64[ns]",
    "end": "datetime64[ns]", "dose": "float64", "n_segments": "int64", "backprojected": "bool",
}
VENT_INTERVAL_DTYPES = {"patientid": "int64", "start": "datetime64[ns]", "end": "datetime64[ns]", "n_points": "int64"}
VENT_POINT_DTYPES = {"patientid": "int64", "time": "datetime64[ns]", "source": "object", "context": "object", "cls": "object"}


def derive_infusion_intervals(
    records: pd.DataFrame, *, max_backprojection: pd.Timedelta = pd.Timedelta(hours=6)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Continuous-infusion "on" intervals from HiRID pharma records.

    Records with bit 2 (invalidated) or 32 (not administered) are dropped; instantaneous
    records (start and stop bits, status 780) are boluses. Within an ``infusionid`` (sorted
    by givenat, enteredentryat) record k covers ``(givenat[k-1], givenat[k]]`` with dose
    ``givendose[k]``; consecutive segments with dose > 0 merge into one interval. A first
    record that already carries a dose is back-projected at the rate of the next segment
    (at most ``max_backprojection``). Returns ``(intervals, per_infusion)``.
    """
    cols = ["patientid", "pharmaid", "infusionid", "givenat", "enteredentryat", "givendose", "recordstatus"]
    interval_cols = ["patientid", "pharmaid", "infusionid", "start", "end", "dose", "n_segments", "backprojected"]
    r = records.loc[:, cols].copy()
    status = _status_bits(r["recordstatus"])
    r["dropped"] = (status & PHARMA_DROP_BITS) != 0
    r["instant"] = ~r["dropped"].to_numpy() & ((status & PHARMA_START) != 0) & ((status & PHARMA_STOP) != 0)
    r["has_start"] = ~r["dropped"].to_numpy() & ~r["instant"].to_numpy() & ((status & PHARMA_START) != 0)
    r["has_stop"] = ~r["dropped"].to_numpy() & ~r["instant"].to_numpy() & ((status & PHARMA_STOP) != 0)
    use = r[~r["dropped"] & ~r["instant"]].sort_values(["infusionid", "givenat", "enteredentryat"], kind="mergesort")
    use = use.reset_index(drop=True)
    n = len(use)
    inf = use["infusionid"].to_numpy()
    t = use["givenat"].to_numpy(dtype="datetime64[ns]")
    dose = use["givendose"].fillna(0.0).to_numpy(dtype=float)
    same_prev = np.zeros(n, dtype=bool)
    same_next = np.zeros(n, dtype=bool)
    if n > 1:
        same_prev[1:] = inf[1:] == inf[:-1]
        same_next[:-1] = same_prev[1:]
    seg_start = np.full(n, _NAT)
    if n > 1:
        seg_start[1:] = np.where(same_prev[1:], t[:-1], _NAT)
    # first record that already carries a dose (no zero-dose start record): back-project
    first_dosed = ~same_prev & (dose > 0) & same_next
    backprojected = np.zeros(n, dtype=bool)
    if first_dosed.any():
        idx = np.flatnonzero(first_dosed)
        dt_next = (t[idx + 1] - t[idx]).astype("timedelta64[ns]").astype(np.int64) / 1e9
        dose_next = dose[idx + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where((dt_next > 0) & (dose_next > 0), dose_next / dt_next, np.nan)
            back = np.where(np.isfinite(rate) & (rate > 0), dose[idx] / rate, np.maximum(dt_next, 0.0))
        back = np.clip(np.nan_to_num(back, nan=0.0), 0.0, max_backprojection.total_seconds())
        seg_start[idx] = t[idx] - (back * 1e9).astype("timedelta64[ns]")
        backprojected[idx] = True
    on = (dose > 0) & ~np.isnat(seg_start)
    prev_on = np.zeros(n, dtype=bool)
    if n > 1:
        prev_on[1:] = on[:-1]
    new_run = on & ~(prev_on & same_prev)
    run = np.cumsum(new_run)
    if on.any():
        seg = pd.DataFrame(
            {
                "run": run[on],
                "patientid": use["patientid"].to_numpy()[on],
                "pharmaid": use["pharmaid"].to_numpy()[on],
                "infusionid": inf[on],
                "start": seg_start[on],
                "end": t[on],
                "dose": dose[on],
                "backprojected": backprojected[on],
            }
        )
        intervals = seg.groupby("run", sort=True).agg(
            patientid=("patientid", "first"),
            pharmaid=("pharmaid", "first"),
            infusionid=("infusionid", "first"),
            start=("start", "first"),
            end=("end", "last"),
            dose=("dose", "sum"),
            n_segments=("dose", "size"),
            backprojected=("backprojected", "first"),
        )
        intervals = intervals[intervals["end"] > intervals["start"]].reset_index(drop=True)
        intervals = intervals[interval_cols].astype(INTERVAL_DTYPES)
    else:
        intervals = _typed_frame(INTERVAL_DTYPES)
    per = r.groupby("infusionid", sort=True).agg(
        patientid=("patientid", "first"),
        pharmaid=("pharmaid", "first"),
        n_records=("recordstatus", "size"),
        n_dropped=("dropped", "sum"),
        n_instant=("instant", "sum"),
        has_start=("has_start", "any"),
        has_stop=("has_stop", "any"),
    )
    per["n_usable"] = use.groupby("infusionid").size().reindex(per.index, fill_value=0).astype(int)
    per["first_status"] = use.groupby("infusionid")["recordstatus"].first().reindex(per.index)
    per["n_intervals"] = intervals.groupby("infusionid").size().reindex(per.index, fill_value=0).astype(int)
    on_hours = (intervals["end"] - intervals["start"]).dt.total_seconds() / 3600.0
    per["on_hours"] = on_hours.groupby(intervals["infusionid"].to_numpy()).sum().reindex(per.index, fill_value=0.0)
    per["backprojected"] = (
        intervals.groupby("infusionid")["backprojected"].any().reindex(per.index, fill_value=False).astype(bool)
    )
    return intervals, per.reset_index()


def classify_ventilation_points(
    rows: pd.DataFrame,
    *,
    mode_window: pd.Timedelta = pd.Timedelta(minutes=30),
    airway_window: pd.Timedelta = pd.Timedelta(hours=24),
) -> pd.DataFrame:
    """Evidence points from ventilator mode (3845) and PEEP setting (2600) rows.

    ``rows``: patientid, datetime, variableid, value (valid rows of 3845/15001552/2600).
    Mode points: class invasive / niv / off from the code. PEEP points: class of the last
    mode point within ``mode_window``; otherwise of the last airway code within
    ``airway_window`` (1/2 invasive, 3/4/5 niv); otherwise ``unknown``. Returns patientid,
    time, source (mode/peep), context (mode/airway/none), cls.
    """
    columns = ["patientid", "time", "source", "context", "cls"]
    rows = rows.loc[rows["variableid"].isin(VENT_IDS), ["patientid", "datetime", "variableid", "value"]]
    pid_dtype = np.int64
    mode = rows[rows["variableid"] == VENT_MODE_ID]
    code = mode["value"].round().astype(np.int64).to_numpy()
    mode_cls = np.where(
        np.isin(code, list(VENT_MODES_INVASIVE)), "invasive", np.where(np.isin(code, list(VENT_MODES_NIV)), "niv", "off")
    )
    modes = pd.DataFrame(
        {
            "patientid": mode["patientid"].to_numpy().astype(pid_dtype),
            "time": mode["datetime"].to_numpy(dtype="datetime64[ns]"),
            "source": "mode",
            "context": "mode",
            "cls": mode_cls,
        }
    ).sort_values("time", kind="mergesort")
    air = rows[rows["variableid"] == AIRWAY_ID]
    acode = air["value"].round().astype(np.int64).to_numpy()
    airs = pd.DataFrame(
        {
            "patientid": air["patientid"].to_numpy().astype(pid_dtype),
            "time": air["datetime"].to_numpy(dtype="datetime64[ns]"),
            "air": np.where(
                np.isin(acode, list(AIRWAY_INVASIVE)), "invasive", np.where(np.isin(acode, list(AIRWAY_NIV)), "niv", "none")
            ),
        }
    ).sort_values("time", kind="mergesort")
    peep = rows[rows["variableid"] == PEEP_SET_ID]
    peeps = pd.DataFrame(
        {
            "patientid": peep["patientid"].to_numpy().astype(pid_dtype),
            "time": peep["datetime"].to_numpy(dtype="datetime64[ns]"),
        }
    ).sort_values("time", kind="mergesort")
    if len(peeps):
        joined = pd.merge_asof(
            peeps, modes[["patientid", "time", "cls"]], on="time", by="patientid", direction="backward", tolerance=mode_window
        )
        no_mode = joined["cls"].isna().to_numpy()
        joined["context"] = np.where(no_mode, "none", "mode")
        if no_mode.any():
            fallback = pd.merge_asof(
                joined.loc[no_mode, ["patientid", "time"]].reset_index(),
                airs,
                on="time",
                by="patientid",
                direction="backward",
                tolerance=airway_window,
            ).set_index("index")
            air_cls = fallback["air"].reindex(joined.index[no_mode])
            has_air = air_cls.isin(["invasive", "niv"]).to_numpy()
            joined.loc[joined.index[no_mode], "cls"] = np.where(has_air, air_cls.to_numpy(), "unknown")
            joined.loc[joined.index[no_mode][has_air], "context"] = "airway"
        joined["source"] = "peep"
        peeps = joined[columns]
    else:
        peeps = _typed_frame(VENT_POINT_DTYPES)
    frames = [frame for frame in (modes[columns], peeps) if len(frame)]
    if not frames:
        return _typed_frame(VENT_POINT_DTYPES)
    points = pd.concat(frames, ignore_index=True).astype(VENT_POINT_DTYPES)
    return points.sort_values(["patientid", "time"], kind="mergesort").reset_index(drop=True)


def ventilation_intervals(
    points: pd.DataFrame,
    on_classes: Iterable[str],
    off_classes: Iterable[str],
    *,
    max_gap: pd.Timedelta = pd.Timedelta(minutes=60),
    carry: pd.Timedelta = pd.Timedelta(minutes=15),
    min_points: int = 2,
) -> pd.DataFrame:
    """Merge evidence points into ``[start, end)`` intervals (patientid, start, end, n_points).

    Points of ``on_classes`` form one interval while consecutive points are at most
    ``max_gap`` apart and no point of ``off_classes`` lies between them. The interval ends
    ``carry`` after its last point or at the next evidence point, whichever is earlier
    (and at least one minute after the last point). Runs with fewer than ``min_points``
    points (an isolated mode switch or PEEP sample) are dropped.
    """
    columns = ["patientid", "start", "end", "n_points"]
    on_classes, off_classes = set(on_classes), set(off_classes)
    p = points[points["cls"].isin(on_classes | off_classes)].sort_values(["patientid", "time"], kind="mergesort")
    n = len(p)
    if n == 0 or not p["cls"].isin(on_classes).any():
        return _typed_frame(VENT_INTERVAL_DTYPES)
    pid = p["patientid"].to_numpy()
    t = p["time"].to_numpy(dtype="datetime64[ns]")
    on = p["cls"].isin(on_classes).to_numpy()
    same_prev = np.zeros(n, dtype=bool)
    same_prev[1:] = pid[1:] == pid[:-1]
    prev_on = np.zeros(n, dtype=bool)
    prev_on[1:] = on[:-1]
    close = np.zeros(n, dtype=bool)
    close[1:] = (t[1:] - t[:-1]) <= max_gap.to_timedelta64()
    new_run = on & ~(same_prev & prev_on & close)
    run = np.cumsum(new_run)
    end = t + carry.to_timedelta64()
    same_next = np.zeros(n, dtype=bool)
    same_next[:-1] = same_prev[1:]
    nxt = np.full(n, _NAT)
    nxt[:-1] = t[1:]
    end = np.where(same_next & (nxt < end), nxt, end)
    end = np.maximum(end, t + np.timedelta64(1, "m"))
    frame = pd.DataFrame({"run": run[on], "patientid": pid[on], "t": t[on], "end": end[on]})
    out = frame.groupby("run", sort=True).agg(
        patientid=("patientid", "first"), start=("t", "first"), end=("end", "last"), n_points=("t", "size")
    )
    out = out[out["n_points"] >= int(min_points)]
    return out.reset_index(drop=True)[columns].astype(VENT_INTERVAL_DTYPES)


def urine_volumes(
    patientid: np.ndarray,
    times: np.ndarray,
    rates: np.ndarray,
    intime: np.ndarray,
    *,
    max_interval_hours: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """mL per urine record from 10020000 rates (ml/h): rate x hours since the previous record
    of the same patient (first record: since ``intime``), the interval clipped to
    ``[0, max_interval_hours]``. Inputs must be sorted by patient, then time. Returns
    (volumes, interval hours)."""
    n = len(rates)
    t = np.asarray(times, dtype="datetime64[ns]")
    prev = np.asarray(intime, dtype="datetime64[ns]").copy()
    if n > 1:
        same = np.asarray(patientid)[1:] == np.asarray(patientid)[:-1]
        prev[1:] = np.where(same, t[:-1], prev[1:])
    hours = (t - prev).astype("timedelta64[ns]").astype(np.int64) / 3.6e12
    hours = np.where(np.isnat(prev), np.nan, hours)
    hours = np.clip(hours, 0.0, float(max_interval_hours))
    return np.asarray(rates, dtype=float) * hours, hours


# ---------------------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------------------


class HiRIDAdapter:
    """v2 source adapter for HiRID 1.1.1 (see the module docstring for every decision)."""

    name = SOURCE

    def __init__(
        self,
        raw_dir: str | Path,
        cache_dir: str | Path,
        *,
        subset_dir: str | Path | None = None,
        include_niv: bool = False,
        include_opioids: bool = False,
        peep_without_context: str = "invasive",
        outtime_rule: str = "last_vital",
        urine_max_interval_hours: float = 12.0,
        table_rr_ids: bool = False,
        build_workers: int = 1,
        auto_build: bool = True,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if outtime_rule not in ("last_observation", "last_vital"):
            raise ValueError("outtime_rule must be 'last_observation' or 'last_vital'")
        if peep_without_context not in ("invasive", "niv", "ignore"):
            raise ValueError("peep_without_context must be 'invasive', 'niv' or 'ignore'")
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir)
        self.subset_dir = Path(subset_dir) if subset_dir is not None else DEFAULT_SUBSET_DIR
        self.include_niv = bool(include_niv)
        self.include_opioids = bool(include_opioids)
        self.peep_without_context = peep_without_context
        self.outtime_rule = outtime_rule
        self.urine_max_interval_hours = float(urine_max_interval_hours)
        self.table_rr_ids = bool(table_rr_ids)
        #: numeric observation ids read by events(): HIRID_TABLE include rows minus the
        #: data-rejected ids (DATA_EXCLUDED_OBS_IDS) unless ``table_rr_ids``.
        self.feature_ids: tuple[int, ...] = tuple(
            vid for vid in FEATURE_IDS if self.table_rr_ids or vid not in DATA_EXCLUDED_OBS_IDS
        )
        self.build_workers = max(1, int(build_workers))
        self.auto_build = bool(auto_build)
        self._log = log or (lambda message: print(message, flush=True))
        self._memo: dict[str, Any] = {}
        self._counters: dict[str, Any] = {"events_calls": 0, "events_rows_out": 0, "implausible_dropped": {}, "urine": {}}
        #: counters of feature_events (task-definition passes), kept apart from events'
        self._feature_pass_counters: dict[str, Any] = {"calls": 0, "rows_out": 0, "implausible_dropped": {}, "urine": {}}

    # ----------------------------------------------------------------- cache management
    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    def _read_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def _cache_part_dir(self, kind: str) -> Path:
        return self.cache_dir / TABLE_LAYOUT[kind]["cache"]

    def _done_parts(self, kind: str) -> set[int]:
        directory = self._cache_part_dir(kind)
        done: set[int] = set()
        if not directory.is_dir():
            return done
        for entry in os.scandir(directory):
            number = part_number(entry.name)
            if number is None:
                continue
            meta_path = directory / f"{_part_stem(number)}.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("raw_cache_version") == RAW_CACHE_VERSION:
                    done.add(number)
        return done

    def table_complete(self, kind: str) -> bool:
        info = self._read_manifest().get(kind, {})
        members = info.get("archive_members")
        if not info.get("complete") or not members:
            return False
        return {int(k) for k in members} <= self._done_parts(kind)

    def build_cache(self, *, workers: int | None = None, force_derived: bool = False) -> dict[str, Any]:
        """Run the raw passes (if incomplete) and the derived tables (if outdated)."""
        workers = self.build_workers if workers is None else max(1, int(workers))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        processed = 0
        for kind in ("observation", "pharma"):
            if not self.table_complete(kind):
                processed += self._build_table(kind, workers)
        manifest = self._read_manifest()
        derived = manifest.get("derived", {})
        outdated = derived.get("rules_version") != RULES_VERSION or not (self.cache_dir / "derived").is_dir()
        if processed or force_derived or outdated:
            self._build_derived()
        self._memo.clear()
        return self._read_manifest()

    def ensure_cache(self) -> None:
        if self._memo.get("cache_ok"):
            return
        manifest = self._read_manifest()
        ready = (
            self.table_complete("observation")
            and self.table_complete("pharma")
            and manifest.get("derived", {}).get("rules_version") == RULES_VERSION
        )
        if not ready:
            if not self.auto_build:
                raise RuntimeError(f"HiRID cache in {self.cache_dir} is incomplete; run build_cache()")
            self.build_cache()
        self._memo["cache_ok"] = True

    def _update_manifest(self, kind: str, payload: Mapping[str, Any]) -> None:
        manifest = self._read_manifest()
        manifest.update(
            {
                "source": SOURCE,
                "adapter_version": ADAPTER_VERSION,
                "raw_cache_version": RAW_CACHE_VERSION,
                "raw_dir": str(self.raw_dir),
                "subset_dir": str(self.subset_dir),
            }
        )
        manifest[kind] = {**manifest.get(kind, {}), **payload}
        _write_json(manifest, self.manifest_path)

    def _build_table(self, kind: str, workers: int) -> int:
        """Cache every part of one raw table; returns the number of parts processed now."""
        layout = TABLE_LAYOUT[kind]
        out_dir = self._cache_part_dir(kind)
        out_dir.mkdir(parents=True, exist_ok=True)
        archive = self.raw_dir / layout["archive"]
        local = extracted_parts(self.subset_dir / layout["subdir"])
        done = self._done_parts(kind)
        started = time.perf_counter()
        todo = sorted(n for n in local if n not in done)
        self._log(f"[hirid] {kind}: {len(done)} parts cached, {len(todo)} extracted parts to process, archive {archive}")
        results: dict[int, dict[str, Any]] = {}
        if todo:
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_process_part_file, kind, n, str(local[n]), str(out_dir)): n for n in todo}
                    for future in as_completed(futures):
                        meta = future.result()
                        results[meta["part"]] = meta
                        if len(results) % 25 == 0:
                            self._log(f"[hirid] {kind}: {len(results)}/{len(todo)} extracted parts ({time.perf_counter() - started:.0f} s)")
            else:
                for n in todo:
                    results[n] = _process_part_file(kind, n, str(local[n]), str(out_dir))
                    if len(results) % 25 == 0:
                        self._log(f"[hirid] {kind}: {len(results)}/{len(todo)} extracted parts ({time.perf_counter() - started:.0f} s)")
        done |= set(results)
        members: dict[int, int] = {}
        mismatched: list[int] = []
        if archive.exists():
            local_sizes = {n: p.stat().st_size for n, p in local.items()}

            def wanted(number: int, size: int) -> bool:
                # not cached yet, or the extracted copy differs from the archive member
                return number not in done or (number in local_sizes and local_sizes[number] != size)

            for number, size, data in iter_archive_parts(archive, wanted):
                members[number] = size
                if data is None:
                    continue
                if number in done:
                    mismatched.append(number)
                results[number] = _process_part_bytes(kind, number, data, str(out_dir))
                done.add(number)
                del data
                self._log(f"[hirid] {kind}: part {number} streamed from the archive ({time.perf_counter() - started:.0f} s)")
        else:
            self._log(f"[hirid] {kind}: archive not found; only extracted parts are cached")
        complete = bool(members) and set(members) <= done
        part_meta = {}
        for number in sorted(done):
            meta_path = out_dir / f"{_part_stem(number)}.json"
            if meta_path.exists():
                part_meta[str(number)] = json.loads(meta_path.read_text(encoding="utf-8"))
        for number, meta in results.items():
            part_meta[str(number)] = {**part_meta.get(str(number), {}), "source": meta.get("source"), "seconds": meta.get("seconds")}
        self._update_manifest(
            kind,
            {
                "complete": complete,
                "archive": str(archive),
                "archive_members": {str(k): v for k, v in sorted(members.items())} or self._read_manifest().get(kind, {}).get("archive_members"),
                "extracted_dir": str(self.subset_dir / layout["subdir"]),
                "n_extracted_parts": len(local),
                "size_mismatch_redone": mismatched,
                "parts": part_meta,
                "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "build_seconds": round(time.perf_counter() - started, 1),
            },
        )
        self._log(f"[hirid] {kind}: complete={complete}, {len(done)} parts, {time.perf_counter() - started:.0f} s")
        return len(results)

    def _part_files(self, kind: str, suffix: str) -> list[Path]:
        directory = self._cache_part_dir(kind)
        return [directory / f"{_part_stem(n)}{suffix}" for n in sorted(self._done_parts(kind))]

    def _build_derived(self) -> None:
        started = time.perf_counter()
        derived = self.cache_dir / "derived"
        derived.mkdir(parents=True, exist_ok=True)
        # patients: per (patient, variable) summaries -> first/last observation, last vital
        summary = pd.concat([pd.read_parquet(p) for p in self._part_files("observation", ".patients.parquet")], ignore_index=True)
        _write_parquet(summary, derived / "obs_patient_variable.parquet")
        grouped = summary.groupby("patientid")
        patients = pd.DataFrame(
            {
                "part": grouped["part"].min(),
                "n_parts": grouped["part"].nunique(),
                "n_valid": grouped["n_valid"].sum(),
                "first_obs": grouped["first"].min(),
                "last_obs": grouped["last"].max(),
            }
        )
        vital = summary[summary["variableid"].isin(VITAL_IDS)].groupby("patientid")["last"].max()
        patients["last_vital"] = vital.reindex(patients.index)
        patients = patients.reset_index()
        _write_parquet(patients, derived / "patients.parquet")
        patient_parts = summary[["patientid", "part"]].drop_duplicates().sort_values(["patientid", "part"])
        _write_parquet(patient_parts, derived / "patient_parts.parquet")
        del summary
        # pharma rows -> infusion intervals
        pharma = pd.concat([pd.read_parquet(p) for p in self._part_files("pharma", ".parquet")], ignore_index=True)
        pharma = pharma.sort_values(["patientid", "infusionid", "givenat"], kind="mergesort").reset_index(drop=True)
        _write_parquet(pharma, derived / "pharma_records.parquet")
        intervals, per_infusion = derive_infusion_intervals(pharma)
        intervals["drug"] = [drug_name(PHARMA_LABELS.get(int(p), str(p))) for p in intervals["pharmaid"]]
        per_infusion["drug"] = [drug_name(PHARMA_LABELS.get(int(p), str(p))) for p in per_infusion["pharmaid"]]
        _write_parquet(intervals, derived / "infusion_intervals.parquet")
        _write_parquet(per_infusion, derived / "infusion_summary.parquet")
        del pharma
        # ventilation evidence -> intervals for both definitions
        vent = self._read_obs_rows(None, VENT_IDS)
        points = classify_ventilation_points(vent)
        _write_parquet(points, derived / "ventilation_points.parquet")
        frames = []
        for kind, (on, off) in self._ventilation_rule_sets().items():
            iv = ventilation_intervals(points, on, off)
            iv["kind"] = kind
            frames.append(iv)
        _write_parquet(pd.concat(frames, ignore_index=True), derived / "ventilation_intervals.parquet")
        self._update_manifest(
            "derived",
            {
                "rules_version": RULES_VERSION,
                "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "build_seconds": round(time.perf_counter() - started, 1),
                "n_patients": int(len(patients)),
                "n_infusion_intervals": int(len(intervals)),
                "n_ventilation_points": int(len(points)),
            },
        )
        self._log(f"[hirid] derived tables built in {time.perf_counter() - started:.0f} s")

    @staticmethod
    def _ventilation_rule_sets() -> dict[str, tuple[set[str], set[str]]]:
        """kind -> (on classes, off classes). PEEP points without context are ``unknown``."""
        return {
            "invasive": ({"invasive"}, {"off", "niv"}),
            "invasive_or_unknown": ({"invasive", "unknown"}, {"off", "niv"}),
            "niv_or_unknown": ({"invasive"}, {"off", "niv", "unknown"}),
            "positive_pressure": ({"invasive", "niv", "unknown"}, {"off"}),
        }

    def _ventilation_kind(self) -> str:
        if self.include_niv:
            return "positive_pressure"
        return {"invasive": "invasive_or_unknown", "niv": "niv_or_unknown", "ignore": "invasive"}[self.peep_without_context]

    # --------------------------------------------------------------------- cache reads
    def _derived(self, name: str) -> pd.DataFrame:
        if name not in self._memo:
            self._memo[name] = pd.read_parquet(self.cache_dir / "derived" / f"{name}.parquet")
        return self._memo[name]

    def _read_obs_table(self, patient_ids: Sequence[int] | None, variable_ids: Iterable[int]) -> pa.Table:
        """Cached observation rows (patientid, variableid, datetime, value) of the given
        patients (None = all) and variables; per part sorted by patient, variable, time."""
        directory = self._cache_part_dir("observation")
        variable_ids = sorted(int(v) for v in variable_ids)
        filters: list[tuple[str, str, Any]] = [("variableid", "in", variable_ids)]
        if patient_ids is None:
            per_part: dict[int, list[int] | None] = {n: None for n in sorted(self._done_parts("observation"))}
        else:
            ids = sorted({int(p) for p in patient_ids})
            pp = self._derived("patient_parts")
            pp = pp[pp["patientid"].isin(ids)]
            per_part = {int(n): sorted(int(x) for x in g["patientid"]) for n, g in pp.groupby("part")}
        tables = []
        for n, pids in sorted(per_part.items()):
            part_filters = list(filters) if pids is None else [("patientid", "in", pids), *filters]
            tables.append(pq.read_table(directory / f"{_part_stem(n)}.parquet", filters=part_filters).cast(OBS_KEPT_SCHEMA))
        if not tables:
            return OBS_KEPT_SCHEMA.empty_table()
        return pa.concat_tables(tables)

    def _read_obs_rows(self, patient_ids: Sequence[int] | None, variable_ids: Iterable[int]) -> pd.DataFrame:
        return self._read_obs_table(patient_ids, variable_ids).to_pandas()

    def _general(self) -> pd.DataFrame:
        if "general" not in self._memo:
            general = pd.read_csv(self.raw_dir / "general_table.csv", usecols=["patientid", "admissiontime", "age"])
            general["admissiontime"] = pd.to_datetime(general["admissiontime"])
            self._memo["general"] = general
        return self._memo["general"]

    # ---------------------------------------------------------------------- contract
    def stays(self) -> pd.DataFrame:
        self.ensure_cache()
        general = self._general()
        patients = self._derived("patients")
        df = general.merge(patients[["patientid", "last_obs", "last_vital"]], on="patientid", how="inner")
        out_col = "last_obs" if self.outtime_rule == "last_observation" else "last_vital"
        adult = df["age"] >= 18
        has_out = df[out_col].notna() & (df[out_col] > df["admissiontime"])
        kept = df[adult & has_out]
        self._counters["stays"] = {
            "general_table_patients": int(len(general)),
            "with_observations": int(len(df)),
            "without_observations": int(len(general) - len(df)),
            "under_18": int((~adult).sum()),
            "outtime_not_after_intime": int((adult & ~has_out).sum()),
            "stays": int(len(kept)),
            "outtime_rule": self.outtime_rule,
        }
        return pd.DataFrame(
            {
                "subject_id": kept["patientid"].astype(np.int64).to_numpy(),
                "stay_id": kept["patientid"].astype(np.int64).to_numpy(),
                "intime": kept["admissiontime"].to_numpy(dtype="datetime64[ns]"),
                "outtime": kept[out_col].to_numpy(dtype="datetime64[ns]"),
            }
        )

    def _intervals(self, pharmaids: Iterable[int]) -> pd.DataFrame:
        intervals = self._derived("infusion_intervals")
        return intervals[intervals["pharmaid"].isin(set(int(p) for p in pharmaids))]

    def sedation_pharmaids(self) -> tuple[int, ...]:
        ids = [
            pid
            for pid, row in SEDATION_ROWS.items()
            if self.include_opioids or drug_name(row.label) not in OPIOID_DRUGS
        ]
        return tuple(sorted(ids))

    def vasopressor_intervals(self) -> pd.DataFrame:
        self.ensure_cache()
        iv = self._intervals(VASOPRESSOR_ROWS)
        return pd.DataFrame(
            {
                "stay_id": iv["patientid"].astype(np.int64).to_numpy(),
                "start": iv["start"].to_numpy(dtype="datetime64[ns]"),
                "end": iv["end"].to_numpy(dtype="datetime64[ns]"),
                "drug": iv["drug"].astype(str).to_numpy(),
            }
        ).sort_values(["stay_id", "start"], kind="mergesort").reset_index(drop=True)

    def events(self, stay_ids: Sequence[int]) -> pd.DataFrame:
        self.ensure_cache()
        out = self.canonical_events(self._read_obs_table(stay_ids, self.feature_ids))
        self._counters["events_calls"] += 1
        self._counters["events_rows_out"] += int(len(out))
        return out

    def feature_events(self, stay_ids: Sequence[int], features: Iterable[str]) -> pd.DataFrame:
        """:meth:`events` restricted to ``features``: reads only their observation ids.

        Its counters are reported separately (``feature_events_counters``).
        """
        self.ensure_cache()
        wanted = frozenset(str(f) for f in features)
        vids = [vid for vid in self.feature_ids if OBS_FEATURE_ROWS[int(vid)].feature in wanted]
        if not vids:
            return pd.DataFrame(columns=["stay_id", "time", "feature", "value", "modality"])
        pass_counters = self._feature_pass_counters
        saved = self._counters
        self._counters = {**saved, "implausible_dropped": pass_counters["implausible_dropped"], "urine": pass_counters["urine"]}
        try:
            out = self.canonical_events(self._read_obs_table(stay_ids, vids))
        finally:
            self._counters = saved
        pass_counters["calls"] += 1
        pass_counters["rows_out"] += int(len(out))
        return out

    def canonical_events(self, raw: pa.Table | pd.DataFrame) -> pd.DataFrame:
        """Cached raw rows -> contract events (canonical units, plausibility screen, modality).

        Only ``self.feature_ids`` are mapped; other variable ids are dropped. Rows keep the
        cache order (patient, variable, time within a part); nothing in the contract needs
        a global sort, and skipping it halves the peak memory.
        """
        names = ("patientid", "variableid", "datetime", "value")
        if isinstance(raw, pa.Table):
            cols = {name: raw.column(name).to_numpy() for name in names}
        else:
            cols = {name: raw[name].to_numpy() for name in names}
        del raw
        vid = np.asarray(cols.pop("variableid"))
        ids = np.asarray(self.feature_ids, dtype=np.int64)
        pos = np.clip(np.searchsorted(ids, vid), 0, len(ids) - 1).astype(np.int16)
        known = np.take(ids, pos) == vid
        rows = [OBS_FEATURE_ROWS[int(i)] for i in ids]
        conversions = [hz.conversion_for(r.feature, r.unit) for r in rows]
        features = np.array([r.feature for r in rows], dtype=object)
        factor = np.array([c.factor for c in conversions], dtype=float)
        offset = np.array([c.offset for c in conversions], dtype=float)
        low = np.array([hz.CANONICAL_SPECS[r.feature].low for r in rows], dtype=float)
        high = np.array([hz.CANONICAL_SPECS[r.feature].high for r in rows], dtype=float)
        modality = np.array([r.modality or None for r in rows], dtype=object)
        value = np.asarray(cols.pop("value"), dtype=np.float64)
        value *= np.take(factor, pos)
        value += np.take(offset, pos)
        patient = np.asarray(cols.pop("patientid"))
        times = np.asarray(cols.pop("datetime"), dtype="datetime64[ns]")
        urine = known & (vid == URINE_ID)
        del vid
        if urine.any():
            general = self._general().set_index("patientid")["admissiontime"]
            u_idx = np.flatnonzero(urine)
            u_idx = u_idx[np.lexsort((times[u_idx], patient[u_idx]))]
            intime = general.reindex(patient[u_idx].astype(np.int64)).to_numpy(dtype="datetime64[ns]")
            volumes, hours = urine_volumes(
                patient[u_idx], times[u_idx], value[u_idx], intime, max_interval_hours=self.urine_max_interval_hours
            )
            value[u_idx] = volumes
            stats = self._counters["urine"]
            stats["records"] = stats.get("records", 0) + int(len(u_idx))
            stats["interval_capped"] = stats.get("interval_capped", 0) + int(np.sum(hours >= self.urine_max_interval_hours))
            stats["no_intime"] = stats.get("no_intime", 0) + int(np.isnan(hours).sum())
        plausible = known & np.isfinite(value)
        plausible &= value >= np.take(low, pos)
        plausible &= value <= np.take(high, pos)
        dropped = known & ~plausible
        if dropped.any():
            counts = np.bincount(pos[dropped], minlength=len(ids))
            agg = self._counters["implausible_dropped"]
            for j in np.flatnonzero(counts):
                agg[features[j]] = agg.get(features[j], 0) + int(counts[j])
        kept_pos = pos[plausible]
        out = pd.DataFrame(
            {
                "stay_id": patient[plausible].astype(np.int64),
                "time": times[plausible],
                "feature": features[kept_pos],
                "value": value[plausible],
                "modality": modality[kept_pos],
            },
            copy=False,
        )
        return out

    def interventions(self, stay_ids: Sequence[int]) -> pd.DataFrame:
        self.ensure_cache()
        ids = {int(s) for s in stay_ids}
        vent = self._derived("ventilation_intervals")
        vent = vent[(vent["kind"] == self._ventilation_kind()) & vent["patientid"].isin(ids)]
        sed = self._intervals(self.sedation_pharmaids())
        sed = sed[sed["patientid"].isin(ids)]
        frames = [
            pd.DataFrame({"stay_id": vent["patientid"].to_numpy(), "start": vent["start"].to_numpy(), "end": vent["end"].to_numpy(), "feature": "ventilation_on"}),
            pd.DataFrame({"stay_id": sed["patientid"].to_numpy(), "start": sed["start"].to_numpy(), "end": sed["end"].to_numpy(), "feature": "sedation_on"}),
        ]
        out = pd.concat(frames, ignore_index=True)
        out["stay_id"] = out["stay_id"].astype(np.int64)
        out["start"] = pd.to_datetime(out["start"])
        out["end"] = pd.to_datetime(out["end"])
        return out.sort_values(["stay_id", "start"], kind="mergesort").reset_index(drop=True)

    def feature_row_counts(self) -> dict[str, int]:
        """Valid, de-duplicated rows per numeric feature in the whole cache (read ids only)."""
        values = pd.concat([pd.read_parquet(p) for p in self._part_files("observation", ".values.parquet")], ignore_index=True)
        kept = values.groupby("variableid")["n_kept"].sum()
        counts: dict[str, int] = {}
        for vid in self.feature_ids:
            feature = OBS_FEATURE_ROWS[vid].feature
            counts[feature] = counts.get(feature, 0) + int(kept.get(vid, 0))
        return counts

    def available_features(self) -> frozenset[str]:
        self.ensure_cache()
        if "available" not in self._memo:
            counts = self.feature_row_counts()
            available = {f for f, n in counts.items() if n > 0}
            vent = self._derived("ventilation_intervals")
            if (vent["kind"] == self._ventilation_kind()).any():
                available.add("ventilation_on")
            if len(self._intervals(self.sedation_pharmaids())):
                available.add("sedation_on")
            self._memo["available"] = frozenset(f for f in CANONICAL_FEATURES if f in available)
        return self._memo["available"]

    def provenance(self) -> dict[str, Any]:
        manifest = self._read_manifest()

        def table_info(kind: str) -> dict[str, Any]:
            info = manifest.get(kind, {})
            parts = info.get("parts", {})
            sources: dict[str, int] = {}
            for meta in parts.values():
                sources[str(meta.get("source"))] = sources.get(str(meta.get("source")), 0) + 1
            totals = {
                key: int(sum(int(meta.get(key, 0) or 0) for meta in parts.values()))
                for key in ("rows", "kept", "invalidated", "not_measured", "duplicates_dropped")
            }
            return {
                "complete": info.get("complete"),
                "archive": info.get("archive"),
                "n_archive_members": len(info.get("archive_members") or {}),
                "n_parts_cached": len(parts),
                "parts_by_source": sources,
                "row_totals": totals,
            }

        feature_ids = {}
        for vid in self.feature_ids:
            row = OBS_FEATURE_ROWS[vid]
            conversion = hz.conversion_for(row.feature, row.unit)
            name = conversion.name if conversion else None
            if vid == URINE_ID:
                name = "rate (ml/h) x hours since the previous urine record (see urine_output)"
            feature_ids.setdefault(row.feature, []).append(
                {
                    "variableid": vid,
                    "label": row.label,
                    "unit": row.unit,
                    "conversion": name,
                    "modality": row.modality or None,
                }
            )
        return {
            "adapter": "src.icu.v2.extract.hirid.HiRIDAdapter",
            "adapter_version": ADAPTER_VERSION,
            "raw_cache_version": RAW_CACHE_VERSION,
            "rules_version": RULES_VERSION,
            "raw_dir": str(self.raw_dir),
            "subset_dir": str(self.subset_dir),
            "cache_dir": str(self.cache_dir),
            "stage": "raw_stage only (merged_stage never read)",
            "options": {
                "include_niv": self.include_niv,
                "include_opioids": self.include_opioids,
                "peep_without_context": self.peep_without_context,
                "outtime_rule": self.outtime_rule,
                "urine_max_interval_hours": self.urine_max_interval_hours,
                "table_rr_ids": self.table_rr_ids,
                "ventilation_kind": self._ventilation_kind(),
            },
            "stay_definition": "stay_id = subject_id = patientid; intime = general_table.admissiontime; outtime = "
            + ("last valid observation time" if self.outtime_rule == "last_observation" else "last HR/BP/SpO2/RR record")
            + "; age >= 18",
            "record_filters": {
                "observation_status_dropped_bits": {"2": "invalidated", "32": "notified, not measured"},
                "pharma_recordstatus_dropped_bits": {"2": "invalidated", "32": "notified, not administered"},
                "observation_duplicates": "per (patient, variableid, datetime) the row with the latest entertime is kept",
            },
            "numeric_features": feature_ids,
            "data_excluded_table_ids": {
                str(vid): reason for vid, reason in DATA_EXCLUDED_OBS_IDS.items() if vid not in self.feature_ids
            },
            "urine_output": f"10020000 rate (ml/h) x hours since the previous urine record (first: since admission), interval capped at {self.urine_max_interval_hours} h; plausible range on the per-record mL",
            "vasopressors": {str(pid): {"label": row.label, "drug": drug_name(row.label)} for pid, row in sorted(VASOPRESSOR_ROWS.items())},
            "sedation": {str(pid): {"label": SEDATION_ROWS[pid].label, "drug": drug_name(SEDATION_ROWS[pid].label)} for pid in self.sedation_pharmaids()},
            "infusion_rule": "givendose at givenat[k] covers (givenat[k-1], givenat[k]] within an infusionid; on while dose > 0; "
            "780 (start+stop) records are boluses; single-record infusions ignored; dosed first record back-projected (<= 6 h)",
            "ventilation_rule": {
                "mode_codes_invasive": VENT_MODES_INVASIVE,
                "mode_codes_niv": VENT_MODES_NIV,
                "mode_codes_off": VENT_MODES_OFF,
                "airway_codes_invasive": AIRWAY_INVASIVE,
                "airway_codes_niv": AIRWAY_NIV,
                "peep_context": "last 3845 within 30 min, else last 15001552 within 24 h, else unknown",
                "merge": "points <= 60 min apart without an off/other-class point; end = min(last + 15 min, next point); "
                "runs of a single point dropped",
            },
            "tables": {kind: table_info(kind) for kind in ("observation", "pharma")},
            "derived": manifest.get("derived", {}),
            "counters": self._counters,
            "feature_events_counters": self._feature_pass_counters,
        }


# ---------------------------------------------------------------------------------------
# Raw-data checks R-H1..R-H3 (written to artifacts/v2/harmonization/hirid_*)
# ---------------------------------------------------------------------------------------


def _reference(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "hirid_variable_reference.csv"
    if not path.exists():
        return pd.DataFrame(columns=["namespace", "identifier", "ref_name", "ref_unit"])
    ref = pd.read_csv(path, encoding="utf-8-sig")
    ref.columns = [c.strip() for c in ref.columns]
    return pd.DataFrame(
        {
            "namespace": ref["Source Table"].str.strip().str.lower(),
            "identifier": pd.to_numeric(ref["ID"], errors="coerce").astype("Int64"),
            "ref_name": ref["Variable Name"],
            "ref_unit": ref["Unit"],
        }
    )


def _table_decisions(namespace: str) -> pd.DataFrame:
    rows = [r for r in hz.HIRID_TABLE if r.namespace == namespace]
    return pd.DataFrame(
        {
            "identifier": [int(r.identifier) for r in rows],
            "v2_feature": [r.feature for r in rows],
            "v2_decision": [r.decision for r in rows],
            "v2_confidence": [r.confidence for r in rows],
        }
    )


def _quantiles(values: np.ndarray, qs: Sequence[float]) -> list[float]:
    if values.size == 0:
        return [float("nan")] * len(qs)
    return [float(x) for x in np.quantile(values, qs)]


def run_raw_checks(adapter: HiRIDAdapter, out_dir: str | Path, *, v1_dirs: Sequence[str | Path] = (), v1_sample: int = 300, seed: int = 0) -> dict[str, Any]:
    """R-H1..R-H3 on the full cache; writes ``hirid_*`` files to ``out_dir``; returns headlines."""
    adapter.ensure_cache()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = _reference(adapter.raw_dir)
    headline: dict[str, Any] = {}
    # ---- R-H1: observation variable ids present
    counts = pd.concat([pd.read_parquet(p) for p in adapter._part_files("observation", ".counts.parquet")], ignore_index=True)
    status = _status_bits(counts["status"])
    counts["invalid"] = (status & OBS_INVALIDATED) != 0
    counts["not_measured"] = ((status & OBS_NOT_MEASURED) != 0) & ~counts["invalid"]
    per_var = counts.groupby("variableid").apply(
        lambda g: pd.Series(
            {
                "n_rows": int(g["n"].sum()),
                "n_invalidated": int(g.loc[g["invalid"], "n"].sum()),
                "n_not_measured": int(g.loc[g["not_measured"], "n"].sum()),
                "n_null_value": int(g.loc[~g["has_value"], "n"].sum()),
                "n_type_F": int(g.loc[g["type"] == "F", "n"].sum()),
                "n_type_P": int(g.loc[g["type"] == "P", "n"].sum()),
                "n_type_C": int(g.loc[g["type"] == "C", "n"].sum()),
            }
        ),
        include_groups=False,
    )
    values = pd.concat([pd.read_parquet(p) for p in adapter._part_files("observation", ".values.parquet")], ignore_index=True)
    vagg = values.groupby("variableid").agg(
        n_valid_value=("n_valid_value", "sum"), vmin=("vmin", "min"), vmax=("vmax", "max"), vsum=("vsum", "sum"),
        n_kept=("n_kept", "sum"), n_duplicates_dropped=("n_duplicates_dropped", "sum"),
    )
    vagg["vmean"] = vagg["vsum"] / vagg["n_valid_value"]
    pv = adapter._derived("obs_patient_variable")
    n_pat = pv.groupby("variableid")["patientid"].nunique().rename("n_patients")
    obs_ids = per_var.join(vagg.drop(columns="vsum"), how="left").join(n_pat, how="left").reset_index()
    obs_ref = ref[ref["namespace"] == "observation"].drop(columns="namespace")
    obs_ids = obs_ids.merge(obs_ref, left_on="variableid", right_on="identifier", how="left").drop(columns="identifier")
    obs_ids = obs_ids.merge(_table_decisions("observation"), left_on="variableid", right_on="identifier", how="left").drop(columns="identifier")
    obs_ids = obs_ids.sort_values("n_rows", ascending=False)
    obs_ids.to_csv(out / "hirid_observation_variableids.csv", index=False)
    table_obs = {int(r.identifier) for r in hz.HIRID_TABLE if r.namespace == "observation"}
    include_obs = {int(r.identifier) for r in hz.HIRID_TABLE if r.namespace == "observation" and r.decision == "include"}
    present = set(obs_ids.loc[obs_ids["n_rows"] > 0, "variableid"].astype(int))
    headline["observation_rows_total"] = int(per_var["n_rows"].sum())
    headline["observation_variableids_present"] = int(len(present))
    headline["observation_invalidated_rows"] = int(per_var["n_invalidated"].sum())
    headline["observation_not_measured_rows"] = int(per_var["n_not_measured"].sum())
    headline["included_observation_ids_absent"] = sorted(include_obs - present)
    headline["table_observation_ids_absent"] = sorted(table_obs - present)
    headline["patients_with_observations"] = int(pv["patientid"].nunique())
    status_tab = counts[counts["variableid"].isin(include_obs)].groupby(["variableid", "status"])["n"].sum().reset_index()
    status_tab.to_csv(out / "hirid_observation_status_included_ids.csv", index=False)
    # ---- R-H1: pharma ids, units, record semantics
    pcounts = pd.concat([pd.read_parquet(p) for p in adapter._part_files("pharma", ".counts.parquet")], ignore_index=True)
    keys = ["pharmaid", "recordstatus", "typeid", "subtypeid", "doseunit", "route"]
    psem = pcounts.groupby(keys, dropna=False)["n"].sum().reset_index()
    pharma_ref = ref[ref["namespace"] == "pharma"].drop(columns=["namespace", "ref_unit"])
    psem = psem.merge(pharma_ref, left_on="pharmaid", right_on="identifier", how="left").drop(columns="identifier")
    psem = psem.merge(_table_decisions("pharma"), left_on="pharmaid", right_on="identifier", how="left").drop(columns="identifier")
    psem.sort_values(["pharmaid", "n"], ascending=[True, False]).to_csv(out / "hirid_pharma_semantics.csv", index=False)
    rel = psem[psem["pharmaid"].isin(RELEVANT_PHARMAIDS)]
    rel.sort_values(["v2_feature", "pharmaid", "n"], ascending=[True, True, False]).to_csv(
        out / "hirid_pharma_semantics_vasoactive_sedative.csv", index=False
    )
    headline["pharma_rows_total"] = int(pcounts["n"].sum())
    headline["pharma_recordstatus_counts"] = {str(k): int(v) for k, v in pcounts.groupby("recordstatus")["n"].sum().items()}
    headline["pharma_typeid_counts"] = {str(k): int(v) for k, v in pcounts.groupby("typeid")["n"].sum().items()}
    headline["pharmaid_300_rows"] = int(pcounts.loc[pcounts["pharmaid"] == 300, "n"].sum())
    headline["pharmaid_300_recordstatus"] = {str(k): int(v) for k, v in pcounts[pcounts["pharmaid"] == 300].groupby("recordstatus")["n"].sum().items()}
    ref_pharma = set(pharma_ref["identifier"].dropna().astype(int))
    unknown_pharma = sorted(set(pcounts["pharmaid"].astype(int)) - ref_pharma)
    headline["pharmaids_not_in_reference"] = unknown_pharma[:50]
    headline["n_pharmaids_present"] = int(pcounts["pharmaid"].nunique())
    # infusion structure per pharmaid
    per_inf = adapter._derived("infusion_summary")
    intervals = adapter._derived("infusion_intervals")
    records = adapter._derived("pharma_records")
    rstat = _status_bits(records["recordstatus"])
    records = records.assign(
        dropped=(rstat & PHARMA_DROP_BITS) != 0,
        instant=((rstat & PHARMA_DROP_BITS) == 0) & ((rstat & PHARMA_START) != 0) & ((rstat & PHARMA_STOP) != 0),
    )
    rec_by = records.groupby("pharmaid").agg(
        n_records=("recordstatus", "size"), n_patients=("patientid", "nunique"), n_dropped=("dropped", "sum"), n_bolus_records=("instant", "sum")
    )
    first_dose_zero = (
        records.sort_values(["infusionid", "givenat"]).groupby("infusionid").head(1).groupby("pharmaid")["givendose"].apply(lambda s: float((s == 0).mean()))
    )
    inf_by = per_inf.groupby("pharmaid").agg(
        n_infusions=("infusionid", "size"),
        n_with_interval=("n_intervals", lambda s: int((s > 0).sum())),
        n_single_usable_record=("n_usable", lambda s: int((s == 1).sum())),
        n_without_start=("has_start", lambda s: int((~s).sum())),
        n_without_stop=("has_stop", lambda s: int((~s).sum())),
        n_backprojected=("backprojected", "sum"),
        median_on_hours=("on_hours", lambda s: float(s[s > 0].median()) if (s > 0).any() else float("nan")),
    )
    iv_by = intervals.groupby("pharmaid").agg(n_intervals=("start", "size"), n_patients_with_interval=("patientid", "nunique"))
    structure = rec_by.join(inf_by, how="left").join(iv_by, how="left").join(first_dose_zero.rename("frac_first_record_dose0"), how="left")
    structure = structure.reset_index()
    structure["label"] = [PHARMA_LABELS.get(int(p), "") for p in structure["pharmaid"]]
    structure["drug"] = [drug_name(lbl) for lbl in structure["label"]]
    structure = structure.merge(_table_decisions("pharma"), left_on="pharmaid", right_on="identifier", how="left").drop(columns="identifier")
    structure.sort_values(["v2_feature", "v2_decision", "pharmaid"]).to_csv(out / "hirid_pharma_infusion_structure.csv", index=False)
    use = records[~records["dropped"] & ~records["instant"]].sort_values(["infusionid", "givenat", "enteredentryat"])
    prev_cum = use.groupby("infusionid")["cumulativedose"].shift()
    ok = prev_cum.notna()
    resid = (use["cumulativedose"] - prev_cum - use["givendose"]).abs()
    tol = 1e-3 * np.maximum(1.0, use["cumulativedose"].abs())
    headline["cumulativedose_identity_frac"] = float((resid[ok] <= tol[ok]).mean()) if ok.any() else float("nan")
    gaps = use.groupby("infusionid")["givenat"].diff().dt.total_seconds() / 60.0
    headline["infusion_record_gap_minutes_quantiles"] = dict(zip(["p25", "p50", "p75", "p95"], _quantiles(gaps.dropna().to_numpy(), [0.25, 0.5, 0.75, 0.95])))
    vaso_iv = intervals[intervals["pharmaid"].isin(VASOPRESSOR_ROWS)]
    headline["vasopressor_patients"] = int(vaso_iv["patientid"].nunique())
    headline["vasopressor_intervals"] = int(len(vaso_iv))
    excl_vaso = [int(r.identifier) for r in _table_rows("pharma", "vasopressor_on", "exclude")]
    headline["excluded_vasoactive_patients_with_infusion"] = {
        f"{p} {PHARMA_LABELS.get(p, '')}": int(intervals.loc[intervals["pharmaid"] == p, "patientid"].nunique()) for p in excl_vaso
    }
    # ---- R-H2: value scales of the included numeric ids (full data; vitals subsampled)
    totals = vagg["n_kept"]
    stride = {vid: max(1, int(round(float(totals.get(vid, 0)) / 2_000_000))) for vid in FEATURE_IDS}
    samples: dict[int, list[np.ndarray]] = {vid: [] for vid in FEATURE_IDS}
    urine_parts: list[pd.DataFrame] = []
    for path in adapter._part_files("observation", ".parquet"):
        table = pq.read_table(path, filters=[("variableid", "in", list(FEATURE_IDS))]).to_pandas()
        for vid, group in table.groupby("variableid"):
            vals = group["value"].to_numpy(dtype=float)
            samples[int(vid)].append(vals[:: stride[int(vid)]])
        urine_parts.append(table[table["variableid"] == URINE_ID])
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    scale_rows = []
    for vid in FEATURE_IDS:
        row = OBS_FEATURE_ROWS[vid]
        vals = np.concatenate(samples[vid]) if samples[vid] else np.array([])
        conversion = hz.conversion_for(row.feature, row.unit)
        canon = conversion.to_canonical(vals) if vals.size else vals
        spec = hz.CANONICAL_SPECS[row.feature]
        record = {
            "variableid": vid, "feature": row.feature, "label": row.label, "source_unit": row.unit,
            "conversion": conversion.name, "n_rows_kept": int(totals.get(vid, 0)), "sample_stride": stride[vid],
            "n_sampled": int(vals.size),
        }
        record.update({f"src_q{q:g}": v for q, v in zip(qs, _quantiles(vals, qs))})
        record.update({f"canon_q{q:g}": v for q, v in zip(qs, _quantiles(np.asarray(canon), qs))})
        if row.feature != "urine_output" and vals.size:
            record["frac_outside_plausible"] = float(np.mean((canon < spec.low) | (canon > spec.high)))
        scale_rows.append(record)
    scales = pd.DataFrame(scale_rows)
    scales.to_csv(out / "hirid_value_scales.csv", index=False)
    med = scales.set_index("variableid")["src_q0.5"]
    headline["median_source_values"] = {
        "lactate_24000524_mmol_l": float(med.get(24000524, np.nan)),
        "creatinine_20000600_umol_l": float(med.get(20000600, np.nan)),
        "platelets_20000110_G_l": float(med.get(20000110, np.nan)),
        "wbc_20000700_G_l": float(med.get(20000700, np.nan)),
        "glucose_20005110_mmol_l": float(med.get(20005110, np.nan)),
        "glucose_24000523_mmol_l": float(med.get(24000523, np.nan)),
        "glucose_24000585_mmol_l": float(med.get(24000585, np.nan)),
        "bilirubin_20004300_umol_l": float(med.get(20004300, np.nan)),
        "temp_400_axillary": float(med.get(400, np.nan)),
        "temp_410_core": float(med.get(410, np.nan)),
    }
    # urine record interval and rate-vs-volume evidence
    urine = pd.concat(urine_parts, ignore_index=True).sort_values(["patientid", "datetime"])
    del urine_parts, samples
    # RR ids: which "Respiratory rate" ids are breaths/min (every 10th part)
    rr_parts = [
        pq.read_table(path, filters=[("variableid", "in", [300, 310, 5685, VENT_MODE_ID])]).to_pandas()
        for path in adapter._part_files("observation", ".parquet")[::10]
    ]
    rr = pd.concat(rr_parts, ignore_index=True)
    del rr_parts
    modes = rr[rr["variableid"] == VENT_MODE_ID].rename(columns={"value": "mode"})[["patientid", "datetime", "mode"]].sort_values("datetime")
    rr_rows = []
    for vid in (300, 310, 5685):
        own = rr[rr["variableid"] == vid].sort_values(["patientid", "datetime"])
        gap = own.groupby("patientid")["datetime"].diff().dt.total_seconds() / 60.0
        with_mode = pd.merge_asof(
            own[["patientid", "datetime", "value"]].sort_values("datetime"), modes, on="datetime", by="patientid",
            direction="backward", tolerance=pd.Timedelta(minutes=30),
        )
        invasive = with_mode["mode"].round().isin(list(VENT_MODES_INVASIVE)).to_numpy()
        zero = with_mode["value"].to_numpy() == 0
        record = {
            "variableid": vid, "n_rows_sampled": int(len(own)), "median": float(own["value"].median()),
            "frac_zero": float(zero.mean()) if len(own) else float("nan"),
            "frac_zero_under_invasive_mode": float(invasive[zero].mean()) if zero.any() else float("nan"),
            "frac_gt_70": float((own["value"] > 70).mean()) if len(own) else float("nan"),
            "median_gap_min": float(gap.median()),
        }
        if vid != 300:
            ref300 = rr[rr["variableid"] == 300][["patientid", "datetime", "value"]].rename(columns={"value": "v300"}).sort_values("datetime")
            pair = pd.merge_asof(
                own[["patientid", "datetime", "value"]].sort_values("datetime"), ref300, on="datetime", by="patientid",
                direction="nearest", tolerance=pd.Timedelta(minutes=2),
            )
            paired = pair.dropna(subset=["v300"])
            nonzero = paired[paired["value"] != 0]
            record.update(
                {
                    "frac_rows_with_300_within_2min": float(len(paired) / max(len(own), 1)),
                    "corr_with_300": float(paired[["value", "v300"]].corr().iloc[0, 1]) if len(paired) > 2 else float("nan"),
                    "frac_nonzero_equal_300": float((nonzero["value"] == nonzero["v300"]).mean()) if len(nonzero) else float("nan"),
                }
            )
        rr_rows.append(record)
    pd.DataFrame(rr_rows).to_csv(out / "hirid_rr_ids_check.csv", index=False)
    headline["rr_ids"] = {str(r["variableid"]): {k: v for k, v in r.items() if k != "variableid"} for r in rr_rows}
    headline["rr_ids_data_excluded_by_adapter"] = {str(k): v for k, v in DATA_EXCLUDED_OBS_IDS.items()}
    del rr, modes
    urine["gap_min"] = urine.groupby("patientid")["datetime"].diff().dt.total_seconds() / 60.0
    adm = adapter._general().set_index("patientid")["admissiontime"]
    first = urine.groupby("patientid").head(1)
    first_offset_h = (first["datetime"].to_numpy() - adm.reindex(first["patientid"]).to_numpy()) / np.timedelta64(1, "h")
    buckets = pd.cut(urine["gap_min"], [0, 45, 75, 105, 135, 200, 270, 400, 720, 1e9])
    by_gap = urine.groupby(buckets, observed=True)["value"].agg(["size", "median", "mean"]).reset_index()
    by_gap["gap_bucket_min"] = by_gap["gap_min"].astype(str)
    by_gap.drop(columns="gap_min").to_csv(out / "hirid_urine_value_by_interval.csv", index=False)
    within = (
        urine[urine["gap_min"].between(105, 135)].groupby("patientid")["value"].median().rename("v120").to_frame()
        .join(urine[urine["gap_min"].between(225, 255)].groupby("patientid")["value"].median().rename("v240"), how="inner")
    )
    within = within[(within["v120"] > 0)]
    gap_q = _quantiles(urine["gap_min"].dropna().to_numpy(), [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    headline["urine"] = {
        "records": int(len(urine)),
        "patients": int(urine["patientid"].nunique()),
        "interval_minutes_quantiles": dict(zip(["p5", "p25", "p50", "p75", "p90", "p95", "p99"], gap_q)),
        "frac_interval_le_75min": float((urine["gap_min"] <= 75).mean()),
        "frac_interval_gt_12h": float((urine["gap_min"] > 720).mean()),
        "within_patient_ratio_v240_over_v120_median": float((within["v240"] / within["v120"]).median()) if len(within) else float("nan"),
        "n_patients_ratio": int(len(within)),
        "first_record_hours_after_admission_quantiles": dict(zip(["p5", "p25", "p50", "p75", "p95"], _quantiles(first_offset_h[~np.isnan(first_offset_h)], [0.05, 0.25, 0.5, 0.75, 0.95]))),
        "rate_value_quantiles_ml_h": dict(zip(["p1", "p50", "p99", "p999"], _quantiles(urine["value"].to_numpy(dtype=float), [0.01, 0.5, 0.99, 0.999]))),
    }
    del urine
    # ---- ventilation coding
    vent = adapter._read_obs_rows(None, VENT_IDS)
    codes = vent[vent["variableid"].isin([VENT_MODE_ID, AIRWAY_ID])].assign(code=lambda d: d["value"].round().astype(int))
    code_tab = codes.groupby(["variableid", "code"]).agg(n=("value", "size"), n_patients=("patientid", "nunique")).reset_index()
    names = {VENT_MODE_ID: {**VENT_MODES_INVASIVE, **VENT_MODES_NIV, **VENT_MODES_OFF}, AIRWAY_ID: {**AIRWAY_INVASIVE, **AIRWAY_NIV, 6: "-"}}
    cls = {VENT_MODE_ID: {**{k: "invasive" for k in VENT_MODES_INVASIVE}, **{k: "niv" for k in VENT_MODES_NIV}}, AIRWAY_ID: {**{k: "invasive" for k in AIRWAY_INVASIVE}, **{k: "niv" for k in AIRWAY_NIV}}}
    code_tab["name"] = [names[v].get(c, "?") for v, c in zip(code_tab["variableid"], code_tab["code"])]
    code_tab["v2_class"] = [cls[v].get(c, "off/none") for v, c in zip(code_tab["variableid"], code_tab["code"])]
    code_tab.to_csv(out / "hirid_ventilation_codes.csv", index=False)
    points = adapter._derived("ventilation_points")
    ctx = points[points["source"] == "peep"].groupby(["context", "cls"]).agg(n=("time", "size"), n_patients=("patientid", "nunique")).reset_index()
    ctx.to_csv(out / "hirid_ventilation_peep_context.csv", index=False)
    vint = adapter._derived("ventilation_intervals")
    vsum = vint.assign(hours=(vint["end"] - vint["start"]).dt.total_seconds() / 3600.0).groupby("kind").agg(
        n_intervals=("start", "size"), n_patients=("patientid", "nunique"), total_hours=("hours", "sum"),
        median_hours=("hours", "median"), n_single_point=("n_points", lambda s: int((s == 1).sum())),
    ).reset_index()
    vsum.to_csv(out / "hirid_ventilation_intervals_summary.csv", index=False)
    patients_with_mode = set(vent.loc[vent["variableid"] == VENT_MODE_ID, "patientid"])
    patients_with_peep = set(vent.loc[vent["variableid"] == PEEP_SET_ID, "patientid"])
    headline["ventilation"] = {
        "patients_with_mode_3845": len(patients_with_mode),
        "patients_with_peep_2600": len(patients_with_peep),
        "peep_patients_without_any_mode": len(patients_with_peep - patients_with_mode),
        "peep_points_by_context_class": {f"{c}/{k}": int(n) for c, k, n in zip(ctx["context"], ctx["cls"], ctx["n"])},
        "intervals_by_kind": {r["kind"]: {"patients": int(r["n_patients"]), "intervals": int(r["n_intervals"]), "total_hours": float(r["total_hours"])} for _, r in vsum.iterrows()},
    }
    del vent
    # ---- outtime: last observation vs last vital sign
    pats = adapter._derived("patients")
    tail_h = (pats["last_obs"] - pats["last_vital"]).dt.total_seconds() / 3600.0
    last_var = pv.loc[pv.groupby("patientid")["last"].idxmax(), ["patientid", "variableid"]].set_index("patientid")["variableid"]
    late = pats.loc[tail_h > 2, "patientid"]
    late_vars = last_var.reindex(late).value_counts().head(15)
    late_tab = late_vars.rename("n_patients").reset_index().rename(columns={"index": "variableid"})
    late_tab = late_tab.merge(obs_ref, left_on="variableid", right_on="identifier", how="left").drop(columns="identifier")
    late_tab.to_csv(out / "hirid_outtime_last_variable_when_tail_gt2h.csv", index=False)
    headline["outtime"] = {
        "last_obs_minus_last_vital_hours_quantiles": dict(zip(["p50", "p90", "p95", "p99", "max"], _quantiles(tail_h.dropna().to_numpy(), [0.5, 0.9, 0.95, 0.99, 1.0]))),
        "patients_tail_gt_2h": int((tail_h > 2).sum()),
        "patients_tail_gt_24h": int((tail_h > 24).sum()),
        "patients_without_vitals": int(pats["last_vital"].isna().sum()),
        "first_obs_minus_admission_hours_quantiles": dict(
            zip(["min", "p1", "p50"], _quantiles(((pats["first_obs"] - adm.reindex(pats["patientid"]).to_numpy()).dt.total_seconds() / 3600.0).dropna().to_numpy(), [0.0, 0.01, 0.5]))
        ),
    }
    # ---- R-H3: v1 processed HiRID feature coverage (lazy sample by constructed path)
    v1_rows = []
    for directory in v1_dirs:
        directory = Path(directory)
        index = directory / "shock_episode_index.json"
        if not index.exists():
            continue
        from src.icu.v2.cohort import iter_index_records

        rng = np.random.default_rng(seed)
        reservoir: list[dict[str, Any]] = []
        seen = 0
        for record in iter_index_records(index):
            seen += 1
            if len(reservoir) < v1_sample:
                reservoir.append(record)
            else:
                j = int(rng.integers(0, seen))
                if j < v1_sample:
                    reservoir[j] = record
        observed: dict[tuple[str, str], list[float]] = {}
        n_ok = 0
        for record in reservoir:
            eid = record.get("episode_id")
            meta_path = directory / f"shock_episode_{eid}.json"
            mask_path = directory / f"shock_episode_{eid}_missing_mask.npy"
            if not (meta_path.exists() and mask_path.exists()):
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            mask = np.load(mask_path)
            names_v1 = meta.get("feature_names") or []
            kind = str(meta.get("episode_type", record.get("episode_type", "")))
            for j, feat in enumerate(names_v1):
                frac = float((~mask[:, j].astype(bool)).mean())
                observed.setdefault((feat, kind), []).append(frac)
                observed.setdefault((feat, "all"), []).append(frac)
            n_ok += 1
        for (feat, kind), fr in observed.items():
            v1_rows.append(
                {
                    "v1_dir": str(directory), "episodes_in_index": seen, "episodes_sampled": n_ok, "feature": feat,
                    "episode_type": kind, "n": len(fr), "mean_observed_fraction": float(np.mean(fr)),
                }
            )
    if v1_rows:
        pd.DataFrame(v1_rows).to_csv(out / "hirid_v1_processed_coverage.csv", index=False)
    headline["v1_processed_coverage_dirs"] = [str(d) for d in v1_dirs]
    headline["merged_stage_used"] = False
    (out / "hirid_rchecks_summary.json").write_text(json.dumps(headline, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return headline


__all__ = [
    "HiRIDAdapter",
    "classify_ventilation_points",
    "derive_infusion_intervals",
    "drug_name",
    "iter_archive_parts",
    "run_raw_checks",
    "urine_volumes",
    "ventilation_intervals",
]
