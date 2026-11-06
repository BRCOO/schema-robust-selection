"""Backblaze drive stats as a second, non-clinical domain: drive-failure prediction.

Data: Backblaze Hard Drive Data (public), one CSV per day with one row per drive: ``date``,
``serial_number``, ``model``, ``failure`` (1 on the last day of a drive that failed) and SMART
attributes (``smart_<id>_raw`` / ``smart_<id>_normalized``).

The task mirrors the ICU tasks of ``extract/common.py``:

* **positive**: a drive with ``failure = 1`` on day T (failure is terminal, so one positive per
  drive); anchor T; the window is the ``window_days`` days ending ``gap_days`` before T, i.e. days
  [T - gap - W + 1, T - gap]; the drive must be in the data on the window's first day.
* **control**: per positive, one drive of the same model (else the same manufacturer) that never
  fails in the scanned period, is in the data on the window's first day, and is still there
  ``horizon_days`` after T (or on the last scanned day); landmark T, the same window. Controls are
  drawn without replacement with a seeded generator.
* **features** (``FEATURES``): the SMART attributes reported by the source manufacturer (Seagate);
  raw values for counters and temperatures, normalized values for attributes whose raw value is a
  vendor-specific encoding (1 read error rate, 3 spin-up time, 7 seek error rate). NaN when a drive
  does not report the attribute that day or is missing that day. No imputation, no scaling.
* **schema of a manufacturer**: the features observed (at least once in the window) in >= 95% of
  its episodes.
* **subject = drive**: one integer id per serial number; positives and controls are different
  drives, and splits are drive-level.
* **hard disks only**: solid-state boot drives (``is_ssd``) report another attribute set and are
  excluded as positives and as controls.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

#: (feature name, Backblaze column) in canonical order
FEATURES: tuple[tuple[str, str], ...] = (
    ("smart_1", "smart_1_normalized"), ("smart_3", "smart_3_normalized"), ("smart_4", "smart_4_raw"),
    ("smart_5", "smart_5_raw"), ("smart_7", "smart_7_normalized"), ("smart_9", "smart_9_raw"),
    ("smart_10", "smart_10_raw"), ("smart_12", "smart_12_raw"), ("smart_187", "smart_187_raw"),
    ("smart_188", "smart_188_raw"), ("smart_190", "smart_190_raw"), ("smart_192", "smart_192_raw"),
    ("smart_193", "smart_193_raw"), ("smart_194", "smart_194_raw"), ("smart_197", "smart_197_raw"),
    ("smart_198", "smart_198_raw"), ("smart_199", "smart_199_raw"), ("smart_240", "smart_240_raw"),
    ("smart_241", "smart_241_raw"), ("smart_242", "smart_242_raw"),
)
FEATURE_MEANING = {
    "smart_1": "read error rate", "smart_3": "spin-up time", "smart_4": "start/stop count",
    "smart_5": "reallocated sectors", "smart_7": "seek error rate", "smart_9": "power-on hours",
    "smart_10": "spin retry count", "smart_12": "power cycle count", "smart_187": "reported uncorrectable errors",
    "smart_188": "command timeouts", "smart_190": "airflow temperature", "smart_192": "power-off retracts",
    "smart_193": "load cycle count", "smart_194": "temperature", "smart_197": "current pending sectors",
    "smart_198": "offline uncorrectable sectors", "smart_199": "UDMA CRC errors", "smart_240": "head flying hours",
    "smart_241": "total LBAs written", "smart_242": "total LBAs read",
}
SCHEMA_THRESHOLD = 0.95


SSD_MARKERS = ("SSD", "SA510", "WDS", "MTFDDAV", "CT250", "CT500", "DELLBOSS", " ZA")


def is_ssd(model: str) -> bool:
    """Solid-state boot drives in the Backblaze data (by model name)."""
    m = str(model).upper()
    return any(marker in m for marker in SSD_MARKERS)


def manufacturer(model: str) -> str:
    m = str(model).upper()
    if is_ssd(m):
        return "ssd"
    if m.startswith("ST"):
        return "seagate"
    if m.startswith(("HGST", "HITACHI", "HUH", "HMS", "HDS")):
        return "hgst"
    if m.startswith(("TOSHIBA", "MG0", "MD0", "HDWF")):
        return "toshiba"
    if m.startswith(("WDC", "WUH", "WD")):
        return "wdc"
    return "other"


def daily_files(zips: Iterable[Path]) -> Iterator[tuple[pd.Timestamp, zipfile.ZipFile, str]]:
    """(day, archive, member) for every daily CSV of the archives, in date order."""
    entries = []
    for path in zips:
        archive = zipfile.ZipFile(path)
        for name in archive.namelist():
            if name.endswith(".csv") and not name.startswith("__MACOSX") and "/._" not in name:
                entries.append((pd.Timestamp(Path(name).stem), archive, name))
    yield from sorted(entries, key=lambda e: e[0])


def read_day(archive: zipfile.ZipFile, member: str, columns: list[str]) -> pd.DataFrame:
    with archive.open(member) as handle:
        data = handle.read()
    header = data[: data.index(b"\n")].decode().strip().split(",")
    usecols = [c for c in columns if c in header]
    return pd.read_csv(io.BytesIO(data), usecols=usecols, dtype={"serial_number": str, "model": str}, low_memory=False)


@dataclass
class Drive:
    model: str
    first: int  # day index of the first and last row
    last: int
    failed_on: int = -1


def scan(zips: Iterable[Path]) -> tuple[list[pd.Timestamp], dict[str, Drive]]:
    """Pass 1: every drive's model, first and last day, and failure day (day indices)."""
    days: list[pd.Timestamp] = []
    drives: dict[str, Drive] = {}
    for index, (day, archive, member) in enumerate(daily_files(zips)):
        days.append(day)
        frame = read_day(archive, member, ["serial_number", "model", "failure"])
        for serial, model, failure in zip(frame["serial_number"], frame["model"], frame["failure"]):
            drive = drives.get(serial)
            if drive is None:
                drives[serial] = drive = Drive(model=str(model), first=index, last=index)
            drive.last = index
            if failure == 1 and drive.failed_on < 0:
                drive.failed_on = index
    return days, drives


def select_episodes(days: list[pd.Timestamp], drives: dict[str, Drive], *, gap_days: int, window_days: int, horizon_days: int,
                    seed: int) -> tuple[list[dict], dict[str, int]]:
    """Positives (failures with a full window) and one matched control each (same model, else same
    manufacturer); returns episode rows (serial, role, anchor and window day indices) and counts."""
    rng = np.random.default_rng(seed)
    last_day = len(days) - 1
    never = [s for s, d in sorted(drives.items()) if d.failed_on < 0 and not is_ssd(d.model)]
    by_model: dict[str, list[str]] = {}
    by_vendor: dict[str, list[str]] = {}
    for serial in never:
        by_model.setdefault(drives[serial].model, []).append(serial)
        by_vendor.setdefault(manufacturer(drives[serial].model), []).append(serial)
    used: set[str] = set()
    rows, counts = [], {"failures": 0, "positives": 0, "no_history": 0, "matched_model": 0, "matched_vendor": 0, "unmatched": 0}
    for serial, drive in sorted(drives.items(), key=lambda kv: (kv[1].failed_on, kv[0])):
        if drive.failed_on < 0 or is_ssd(drive.model):
            continue
        counts["failures"] += 1
        anchor = drive.failed_on
        end = anchor - gap_days
        start = end - window_days + 1
        if start < 0 or drive.first > start:
            counts["no_history"] += 1
            continue
        need_until = min(anchor + horizon_days, last_day)

        def eligible(pool: list[str]) -> list[str]:
            return [s for s in pool if s not in used and drives[s].first <= start and drives[s].last >= need_until]

        pool = eligible(by_model.get(drive.model, []))
        kind = "matched_model"
        if not pool:
            pool, kind = eligible(by_vendor.get(manufacturer(drive.model), [])), "matched_vendor"
        if not pool:
            counts["unmatched"] += 1
            continue
        control = pool[int(rng.integers(len(pool)))]
        used.add(control)
        counts["positives"] += 1
        counts[kind] += 1
        for role, s in (("positive", serial), ("control", control)):
            rows.append({"serial": s, "role": role, "model": drives[s].model, "manufacturer": manufacturer(drives[s].model),
                         "anchor": anchor, "start": start, "end": end, "matched_serial": control if role == "positive" else serial})
    return rows, counts


def extract_windows(zips: Iterable[Path], days: list[pd.Timestamp], rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Pass 2: values (episodes x window days x features) with NaN where not reported, and the mask."""
    window = rows[0]["end"] - rows[0]["start"] + 1
    values = np.full((len(rows), window, len(FEATURES)), np.nan)
    needed: dict[int, dict[str, tuple[int, int]]] = {}
    for e, row in enumerate(rows):
        for offset, day in enumerate(range(row["start"], row["end"] + 1)):
            needed.setdefault(day, {})[row["serial"]] = (e, offset)
    columns = ["serial_number", *[column for _, column in FEATURES]]
    for index, (_, archive, member) in enumerate(daily_files(zips)):
        wanted = needed.get(index)
        if not wanted:
            continue
        frame = read_day(archive, member, columns)
        frame = frame[frame["serial_number"].isin(wanted.keys())].copy()
        for _, column in FEATURES:
            if column not in frame.columns:  # an attribute no drive reports that day
                frame[column] = np.nan
        block = frame[[c for _, c in FEATURES]].to_numpy(dtype=float)
        for serial, vector in zip(frame["serial_number"], block):
            e, offset = wanted[serial]
            values[e, offset] = vector
    return values, np.isnan(values)


def manufacturer_schemas(rows: list[dict], mask: np.ndarray) -> dict[str, list[str]]:
    """Features observed at least once in the window in >= SCHEMA_THRESHOLD of each manufacturer's episodes."""
    names = [n for n, _ in FEATURES]
    observed = ~mask.all(axis=1)  # episodes x features
    out = {}
    for vendor in sorted({r["manufacturer"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["manufacturer"] == vendor]
        share = observed[idx].mean(axis=0)
        out[vendor] = [n for n, s in zip(names, share) if s >= SCHEMA_THRESHOLD]
    return out
