"""Shared episode construction for the v2 extraction (MIMIC-IV, eICU, HiRID).

Each source adapter (``src/icu/v2/extract/<source>.py``) only converts its raw tables into
four standardized frames; everything that defines the task lives here, so the three
databases are cut identically:

``stays``         subject_id:int64, stay_id:int64, intime, outtime (datetime64)
``vasopressors``  stay_id, start, end (datetime64), drug:str -- continuous infusions with
                  rate > 0 of the harmonized vasopressor set (no inotropes, no boluses)
``events``        stay_id, time (datetime64), feature:str, value:float in canonical units;
                  optional ``modality`` ("invasive"/"noninvasive") for SBP/DBP/MAP
``interventions`` stay_id, start, end (datetime64), feature in INTERVENTION_FEATURES

Adapters may also provide ``feature_events(stay_ids, features)``: ``events`` restricted to
``features``, reading only the tables they come from (used for the AKI creatinine pass).

Task (defaults are the proposals of docs/reviews/v2_task_decision_memo.md):

* Positive: the FIRST vasopressor start of an ICU stay (initiation), kept only if the stay
  has at least ``window_hours + prediction_gap_hours`` of ICU time before it.
* Cutoff = anchor - gap. The window is ``n_bins`` hourly bins ending exactly at the cutoff,
  half-open, so no observation at or after the cutoff is used.
* Controls: stays with NO vasopressor infusion at all. Each positive is matched to
  ``controls_per_positive`` unused control stays that are still in the ICU at the
  positive's time since admission; the control's pseudo-anchor is its intime plus that
  time (landmark matching), so window length and time since admission carry no label.
* ``vasopressor_on`` is never a model input of this task; it only defines anchors.

Second task, ``ExtractionSpec.task = "aki_onset"`` (acute kidney injury, KDIGO creatinine
criterion on ICU creatinine only):

* AKI onset = the first ICU creatinine measurement that is >= ``aki_abs_rise`` (0.3 mg/dL)
  above the lowest creatinine of the previous ``aki_abs_window_hours`` (48 h), or >=
  ``aki_rel_rise`` (1.5) times the lowest of the previous ``aki_rel_window_days`` (7 d).
  Only measurements inside the ICU stay count (MIMIC labs are assigned to stays by
  charttime in [intime, outtime]; the other sources are restricted the same way), so a
  patient admitted with AKI is only detected by a further rise in the ICU.
* Stays whose first ICU creatinine is >= ``aki_max_first_creatinine`` (4.0 mg/dL; a proxy
  for end-stage kidney disease and chronic dialysis, which no source records uniformly)
  are neither positives nor controls.
* Positive: the onset, with the same history rule as above (window + gap before it) and
  one positive per patient. Controls: stays with at least ``aki_min_creatinine_for_control``
  creatinine measurements and no onset, from patients without AKI onset in any stay,
  landmark-matched as above and at risk at the landmark: an ICU creatinine before it (as
  every positive has before its onset) and one at or after it (so no AKI was observed).
* Inputs: the 17 features plus ``vasopressor_on`` (an ordinary predictor here); creatinine
  stays an input, its values before the cutoff being the pre-onset trajectory.

Values are stored in canonical units with NaN where a feature was not observed in a bin
(``missing_mask`` True). Nothing is imputed or scaled here; FMG-Bridge's v1 per-episode
representation is derived separately (:func:`fmg_representation`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from src.icu.v2.materialize import episode_paths
from src.icu.v2.task import refit_imputation_and_scaling
from src.icu.v2.types import EpisodeRecord

CANONICAL_FEATURES: tuple[str, ...] = (
    "HR", "RR", "SpO2", "SBP", "DBP", "MAP", "Temp", "ventilation_on", "Lactate", "pH",
    "Creatinine", "Glucose", "Bilirubin_Total", "WBC", "Platelets", "sedation_on", "urine_output",
)
INTERVENTION_FEATURES: tuple[str, ...] = ("ventilation_on", "sedation_on")
#: interval flags episode_matrix fills; vasopressor_on is a model input only in the AKI task
FLAG_FEATURES: tuple[str, ...] = INTERVENTION_FEATURES + ("vasopressor_on",)
#: inputs of the AKI task: the 17 canonical features plus vasopressor_on
AKI_FEATURES: tuple[str, ...] = CANONICAL_FEATURES + ("vasopressor_on",)
TASKS: tuple[str, ...] = ("vasopressor_initiation", "aki_onset")
SUM_FEATURES: tuple[str, ...] = ("urine_output",)
BP_FEATURES: tuple[str, ...] = ("SBP", "DBP", "MAP")
NUMERIC_FEATURES: tuple[str, ...] = tuple(f for f in CANONICAL_FEATURES if f not in INTERVENTION_FEATURES)
# v1 lists these 14 as missingness / observation-density features (urine_output is zero-filled in v1)
MISSINGNESS_FEATURES: tuple[str, ...] = tuple(f for f in NUMERIC_FEATURES if f not in SUM_FEATURES)

STAY_COLUMNS = ("subject_id", "stay_id", "intime", "outtime")
VASO_COLUMNS = ("stay_id", "start", "end", "drug")
EVENT_COLUMNS = ("stay_id", "time", "feature", "value")
INTERVENTION_COLUMNS = ("stay_id", "start", "end", "feature")


class SourceAdapter(Protocol):
    """What a database adapter provides. Frames must follow the column contracts above."""

    name: str

    def stays(self) -> pd.DataFrame: ...

    def vasopressor_intervals(self) -> pd.DataFrame: ...

    def events(self, stay_ids: Sequence[int]) -> pd.DataFrame: ...

    def interventions(self, stay_ids: Sequence[int]) -> pd.DataFrame: ...

    def available_features(self) -> frozenset[str]: ...

    def provenance(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ExtractionSpec:
    """Task parameters shared by every source.

    ``max_positives_per_patient``: keep only each patient's earliest eligible positive stay
    (decision memo item 5); ``None`` keeps all. ``match_columns``: extra ``stays`` columns a
    control must share with its positive (e.g. ``("hospitalid",)`` for eICU).
    ``exclude_vasopressor_patients_from_controls``: control stays must come from patients
    with no vasopressor infusion in ANY stay. ``required_vital`` /
    ``required_vital_last_bins``: both classes must have that vital observed in at least
    one of the last bins before the cutoff (removes episodes whose monitoring stops before
    the anchor, e.g. a patient off the unit when the drug is started); 0 disables.
    ``task`` and the ``aki_*`` fields select and parametrize the AKI task (module docstring);
    there ``exclude_vasopressor_patients_from_controls`` excludes patients with an AKI onset.
    """

    window_hours: int = 12
    prediction_gap_hours: float = 1.0
    resample_minutes: int = 60
    controls_per_positive: int = 1
    seed: int = 0
    max_positives_per_patient: int | None = 1
    match_columns: tuple[str, ...] = ()
    exclude_vasopressor_patients_from_controls: bool = True
    required_vital: str | None = "HR"
    required_vital_last_bins: int = 1
    task: str = "vasopressor_initiation"
    aki_abs_rise: float = 0.3
    aki_abs_window_hours: float = 48.0
    aki_rel_rise: float = 1.5
    aki_rel_window_days: float = 7.0
    aki_min_creatinine_for_control: int = 2
    aki_max_first_creatinine: float | None = 4.0

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.aki_abs_rise <= 0 or self.aki_rel_rise <= 1 or self.aki_min_creatinine_for_control < 2:
            raise ValueError("AKI criterion needs aki_abs_rise > 0, aki_rel_rise > 1, aki_min_creatinine_for_control >= 2")
        if int(self.window_hours * 60) % int(self.resample_minutes):
            raise ValueError("window_hours must be a multiple of the bin width")
        if self.prediction_gap_hours < 0 or self.controls_per_positive < 0:
            raise ValueError("gap and controls_per_positive must be non-negative")
        if self.max_positives_per_patient is not None and self.max_positives_per_patient < 1:
            raise ValueError("max_positives_per_patient must be >= 1 or None")
        if not 0 <= self.required_vital_last_bins <= self.n_bins:
            raise ValueError("required_vital_last_bins must lie in [0, n_bins]")

    @property
    def n_bins(self) -> int:
        return int(self.window_hours * 60) // int(self.resample_minutes)

    @property
    def lookback(self) -> pd.Timedelta:
        return pd.Timedelta(hours=self.window_hours) + pd.Timedelta(hours=self.prediction_gap_hours)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} frame is missing columns {missing}")


def _hash_key(seed: int, salt: str, value: object) -> str:
    return hashlib.sha256(f"{seed}:{salt}:{value}".encode()).hexdigest()


def find_positive_anchors(stays: pd.DataFrame, vasopressors: pd.DataFrame, spec: ExtractionSpec) -> tuple[pd.DataFrame, dict[str, int]]:
    """First vasopressor start per stay, kept when enough ICU history precedes it.

    Returns positives (subject_id, stay_id, intime, outtime, anchor_time, t_since_admit)
    and exclusion counts.
    """
    require_columns(stays, STAY_COLUMNS, "stays")
    require_columns(vasopressors, VASO_COLUMNS, "vasopressors")
    first = vasopressors.groupby("stay_id", as_index=False)["start"].min().rename(columns={"start": "anchor_time"})
    merged = stays.merge(first, on="stay_id", how="inner")
    counts = {"stays_with_vasopressor": int(len(merged))}
    inside = (merged["anchor_time"] >= merged["intime"]) & (merged["anchor_time"] <= merged["outtime"])
    counts["anchor_outside_stay"] = int((~inside).sum())
    merged = merged[inside]
    enough = merged["anchor_time"] - merged["intime"] >= spec.lookback
    counts["anchor_too_early"] = int((~enough).sum())
    merged = merged[enough].copy()
    merged["t_since_admit"] = merged["anchor_time"] - merged["intime"]
    counts["positives"] = int(len(merged))
    return merged.reset_index(drop=True), counts


def kdigo_onsets(creatinine: pd.DataFrame, spec: ExtractionSpec) -> pd.DataFrame:
    """First KDIGO creatinine AKI time per stay, as an anchor-event frame (VASO_COLUMNS).

    ``creatinine``: stay_id, time, value (mg/dL), restricted to the ICU stay. A measurement
    at time t is an onset when value(t) - min(values in [t - 48 h, t)) >= aki_abs_rise or
    value(t) >= aki_rel_rise * min(values in [t - 7 d, t)); the first one per stay is kept.
    The windows are open at t, so simultaneous duplicate results never trigger each other.
    """
    abs_window = pd.Timedelta(hours=spec.aki_abs_window_hours).to_timedelta64()
    rel_window = pd.Timedelta(days=spec.aki_rel_window_days).to_timedelta64()
    rows = []
    frame = creatinine.sort_values(["stay_id", "time"], kind="stable")
    for stay_id, group in frame.groupby("stay_id", sort=False):
        times = group["time"].to_numpy(dtype="datetime64[ns]")
        values = group["value"].to_numpy(dtype=float)
        for i in range(1, len(values)):
            t = times[i]
            before = times[:i] < t  # simultaneous duplicates are not a baseline for each other
            in_abs = before & (times[:i] >= t - abs_window)
            in_rel = before & (times[:i] >= t - rel_window)
            hit = False
            if in_abs.any() and values[i] - values[:i][in_abs].min() >= spec.aki_abs_rise - 1e-12:
                hit = True
            elif in_rel.any() and values[i] >= spec.aki_rel_rise * values[:i][in_rel].min() - 1e-12:
                hit = True
            if hit:
                rows.append({"stay_id": int(stay_id), "start": pd.Timestamp(t), "end": pd.Timestamp(t), "drug": "aki_kdigo_creatinine"})
                break
    return pd.DataFrame(rows, columns=list(VASO_COLUMNS))


def creatinine_table(adapter: "SourceAdapter", stays: pd.DataFrame, chunk_stays: int = 2000) -> pd.DataFrame:
    """All ICU creatinine measurements (stay_id, time, value in mg/dL) of ``stays``.

    Uses the adapter's ``feature_events`` when it has one, else ``events``; measurements
    outside [intime, outtime] are dropped.
    """
    getter = getattr(adapter, "feature_events", None)
    ids = [int(s) for s in stays["stay_id"]]
    parts = []
    for begin in range(0, len(ids), chunk_stays):
        chunk = ids[begin : begin + chunk_stays]
        ev = getter(chunk, ("Creatinine",)) if getter is not None else adapter.events(chunk)
        if len(ev):
            ev = ev.loc[ev["feature"].astype(str) == "Creatinine", ["stay_id", "time", "value"]]
            parts.append(ev.astype({"stay_id": np.int64, "value": float}))
    if not parts:
        return pd.DataFrame({"stay_id": pd.Series(dtype=np.int64), "time": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype=float)})
    creat = pd.concat(parts, ignore_index=True)
    creat = creat.merge(stays[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
    inside = (creat["time"] >= creat["intime"]) & (creat["time"] <= creat["outtime"]) & creat["value"].notna()
    creat = creat.loc[inside, ["stay_id", "time", "value"]]
    return creat.sort_values(["stay_id", "time"], kind="stable").reset_index(drop=True)


def aki_cohort(stays: pd.DataFrame, creatinine: pd.DataFrame, spec: ExtractionSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Anchor events and control source of the AKI task (module docstring).

    Returns ``(positive_events, all_onsets, control_stays, counts)``: the onsets of stays
    that may be positives, every onset (so that :func:`match_controls` excludes their
    stays and patients), and the stays :func:`match_controls` draws from, i.e. eligible
    stays plus the onset stays, with ``at_risk_from`` / ``at_risk_until`` = time since
    intime of their first and last ICU creatinine.
    """
    onsets = kdigo_onsets(creatinine, spec)
    frame = creatinine.merge(stays[["stay_id", "intime"]], on="stay_id", how="inner")
    frame = frame.assign(offset=frame["time"] - frame["intime"]).sort_values(["stay_id", "time"], kind="stable")
    grouped = frame.groupby("stay_id", sort=True)
    summary = pd.DataFrame(
        {
            "n": grouped.size(),
            "first_value": grouped["value"].first(),
            "at_risk_from": grouped["offset"].min(),
            "at_risk_until": grouped["offset"].max(),
        }
    )
    if spec.aki_max_first_creatinine is None:
        excluded: set[int] = set()
    else:
        excluded = set(int(s) for s in summary.index[summary["first_value"] >= spec.aki_max_first_creatinine])
    onset_stays = set(int(s) for s in onsets["stay_id"])
    eligible = set(int(s) for s in summary.index[summary["n"] >= spec.aki_min_creatinine_for_control]) - excluded
    positive_events = onsets[~onsets["stay_id"].isin(excluded)].reset_index(drop=True)
    control_stays = stays[stays["stay_id"].isin(eligible | onset_stays)]
    control_stays = control_stays.merge(summary[["at_risk_from", "at_risk_until"]], left_on="stay_id", right_index=True, how="left")
    counts = {
        "creatinine_measurements": int(len(creatinine)),
        "stays_with_creatinine": int(len(summary)),
        "stays_excluded_high_first_creatinine": int(len(excluded)),
        "stays_with_aki_onset": int(len(onsets)),
        "aki_onsets_after_exclusion": int(len(positive_events)),
        "control_source_stays": int(len(eligible - onset_stays)),
    }
    return positive_events, onsets, control_stays.reset_index(drop=True), counts


def limit_positives_per_patient(positives: pd.DataFrame, spec: ExtractionSpec) -> tuple[pd.DataFrame, int]:
    """Keep each patient's earliest positive stays (by ``stay_order`` if the adapter gives
    one, else ``intime``; ties, e.g. eICU's shared synthetic intime, by a seeded hash)."""
    if spec.max_positives_per_patient is None or positives.empty:
        return positives, 0
    order_col = "stay_order" if "stay_order" in positives.columns else "intime"
    ranked = positives.assign(_tie=[_hash_key(spec.seed, "first", s) for s in positives["stay_id"]])
    ranked = ranked.sort_values(["subject_id", order_col, "_tie"], kind="mergesort")
    kept = ranked.groupby("subject_id", sort=False).head(int(spec.max_positives_per_patient)).drop(columns="_tie")
    return kept.reset_index(drop=True), int(len(positives) - len(kept))


def match_controls(stays: pd.DataFrame, vasopressors: pd.DataFrame, positives: pd.DataFrame, spec: ExtractionSpec) -> tuple[pd.DataFrame, dict[str, int]]:
    """Landmark-matched controls from stays without any vasopressor infusion.

    Positives are processed in hash order and take the next unused control stays (hash
    order) that are still in the ICU at the positive's time since admission and share the
    positive's ``spec.match_columns`` values. With
    ``exclude_vasopressor_patients_from_controls`` a control's patient never had a
    vasopressor infusion in any stay. If ``stays`` has ``at_risk_from`` / ``at_risk_until``
    (time since intime; AKI task), a control is only matched at a landmark t with
    at_risk_from < t <= at_risk_until. Returns controls (subject_id, stay_id, intime,
    outtime, anchor_time = pseudo-anchor, t_since_admit, matched_stay_id) and counts.
    """
    match_columns = tuple(spec.match_columns)
    for column in match_columns:
        if column not in stays.columns or column not in positives.columns:
            raise ValueError(f"match column {column!r} is missing from stays or positives")
    vaso_stays = set(vasopressors["stay_id"].unique())
    at_risk = "at_risk_from" in stays.columns and "at_risk_until" in stays.columns
    pool = stays[~stays["stay_id"].isin(vaso_stays)].copy()
    excluded_patients = 0
    if spec.exclude_vasopressor_patients_from_controls:
        vaso_subjects = set(stays.loc[stays["stay_id"].isin(vaso_stays), "subject_id"])
        before = len(pool)
        pool = pool[~pool["subject_id"].isin(vaso_subjects)]
        excluded_patients = before - len(pool)
    pool = pool.assign(los=pool["outtime"] - pool["intime"], key=[_hash_key(spec.seed, "control", s) for s in pool["stay_id"]])
    pool = pool.sort_values("key").reset_index(drop=True)
    los = pool["los"].to_numpy()
    if at_risk:
        risk_from = pool["at_risk_from"].to_numpy(dtype="timedelta64[ns]")
        risk_until = pool["at_risk_until"].to_numpy(dtype="timedelta64[ns]")
    used = np.zeros(len(pool), dtype=bool)
    if match_columns:
        groups = {key: np.asarray(idx) for key, idx in pool.groupby(list(match_columns), sort=False).indices.items()}
    else:
        groups = {(): np.arange(len(pool))}
    empty = np.zeros(0, dtype=np.int64)
    order = sorted(range(len(positives)), key=lambda i: _hash_key(spec.seed, "positive", positives["stay_id"].iat[i]))
    rows: list[dict[str, Any]] = []
    shortfall = 0
    for i in order:
        need = spec.controls_per_positive
        t = positives["t_since_admit"].iat[i]
        if match_columns:
            key = tuple(positives[c].iat[i] for c in match_columns)
            idx = groups.get(key if len(key) > 1 else key[0], empty)
        else:
            idx = groups[()]
        t64 = pd.Timedelta(t).to_timedelta64()
        ok = ~used[idx] & (los[idx] >= t64)
        if at_risk:
            ok &= (risk_from[idx] < t64) & (risk_until[idx] >= t64)
        candidates = idx[ok]
        if len(candidates) < need:
            shortfall += need - len(candidates)
        for j in candidates[:need]:
            used[j] = True
            row = pool.iloc[j]
            rows.append(
                {
                    "subject_id": int(row["subject_id"]),
                    "stay_id": int(row["stay_id"]),
                    "intime": row["intime"],
                    "outtime": row["outtime"],
                    "anchor_time": row["intime"] + t,
                    "t_since_admit": t,
                    "matched_stay_id": int(positives["stay_id"].iat[i]),
                }
            )
    controls = pd.DataFrame(rows, columns=["subject_id", "stay_id", "intime", "outtime", "anchor_time", "t_since_admit", "matched_stay_id"])
    return controls, {
        "control_pool": int(len(pool)),
        "control_stays_excluded_vasopressor_patient": int(excluded_patients),
        "controls": int(len(controls)),
        "control_shortfall": int(shortfall),
    }


def sample_positive_patients(positives: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Uniform, order-independent sample of positive patients (all their positives kept)."""
    if n is None:
        return positives
    subjects = sorted(set(int(s) for s in positives["subject_id"]), key=lambda s: _hash_key(seed, "sample", s))
    chosen = set(subjects[: int(n)])
    return positives[positives["subject_id"].isin(chosen)].reset_index(drop=True)


def bin_edges(cutoff: pd.Timestamp, spec: ExtractionSpec) -> pd.DatetimeIndex:
    """``n_bins + 1`` edges; the last edge is the cutoff (bins are half-open)."""
    step = pd.Timedelta(minutes=int(spec.resample_minutes))
    return pd.DatetimeIndex([cutoff - step * (spec.n_bins - i) for i in range(spec.n_bins + 1)])


def episode_matrix(
    events: pd.DataFrame,
    interventions: pd.DataFrame,
    cutoff: pd.Timestamp,
    spec: ExtractionSpec,
    features: Sequence[str] = CANONICAL_FEATURES,
    available: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hourly (n_bins x features) values in canonical units and the missing mask.

    Numeric features: mean per bin (urine_output: sum). Blood pressure: within a bin the
    mean of invasive readings if any, else of non-invasive readings. Interventions: 1 when
    an interval overlaps the bin, else 0 (never missing), but only for features the
    source provides: a feature outside ``available`` stays NaN and missing in every bin,
    so a source without ventilation data never reads as "not ventilated". Only data in
    ``[cutoff - window, cutoff)`` is used.
    """
    edges = bin_edges(cutoff, spec)
    start, end = edges[0], edges[-1]
    n = spec.n_bins
    values = np.full((n, len(features)), np.nan)
    mask = np.ones((n, len(features)), dtype=bool)
    col = {feature: j for j, feature in enumerate(features)}
    provided = set(features) if available is None else set(available)
    if len(events):
        window = events[(events["time"] >= start) & (events["time"] < end)]
        if len(window):
            idx = np.searchsorted(edges.asi8, window["time"].astype("datetime64[ns]").to_numpy().astype("int64"), side="right") - 1
            window = window.assign(bin=idx, feature=window["feature"].astype(str))
            if "modality" in window.columns:
                modality = window["modality"].astype(object).where(window["modality"].notna(), None).astype(str)
                is_bp = window["feature"].isin(BP_FEATURES).to_numpy()
                if is_bp.any():
                    bp = window[is_bp].assign(inv=(modality[is_bp] == "invasive").to_numpy(dtype=bool))
                    has_inv = bp.groupby(["feature", "bin"], observed=True, sort=False)["inv"].transform("any").to_numpy(dtype=bool)
                    keep = ~has_inv | bp["inv"].to_numpy(dtype=bool)
                    window = pd.concat([window[~is_bp], bp[keep].drop(columns="inv")])
            for (feature, b), group in window.groupby(["feature", "bin"], observed=True, sort=False):
                if feature not in col or feature not in provided or not (0 <= b < n):
                    continue
                vals = group["value"].astype(float)
                values[b, col[feature]] = float(vals.sum() if feature in SUM_FEATURES else vals.mean())
                mask[b, col[feature]] = False
    for feature in FLAG_FEATURES:
        if feature not in col or feature not in provided:
            continue
        j = col[feature]
        values[:, j] = 0.0
        mask[:, j] = False
        if len(interventions):
            rows = interventions[interventions["feature"].astype(str) == feature]
            for s_, e_ in zip(rows["start"], rows["end"]):
                e_ = s_ + pd.Timedelta(minutes=1) if pd.isna(e_) else e_
                for b in range(n):
                    if s_ < edges[b + 1] and e_ > edges[b]:
                        values[b, j] = 1.0
    return values, mask


def vital_observed_near_cutoff(mask: np.ndarray, features: Sequence[str], spec: ExtractionSpec) -> bool:
    """True if ``spec.required_vital`` is observed in one of the last required bins."""
    if not spec.required_vital or spec.required_vital_last_bins == 0:
        return True
    j = list(features).index(spec.required_vital)
    return bool((~np.asarray(mask, dtype=bool)[-spec.required_vital_last_bins :, j]).any())


def episode_id(source: str, kind: str, stay_id: int, cutoff: pd.Timestamp) -> str:
    return f"{source}_stay_{int(stay_id)}_{cutoff.strftime('%Y%m%d%H%M%S')}_{kind}"


def build_episode_meta(source: str, kind: str, row: Mapping[str, Any], spec: ExtractionSpec, features: Sequence[str], available: frozenset[str]) -> dict[str, Any]:
    cutoff = pd.Timestamp(row["anchor_time"]) - pd.Timedelta(hours=spec.prediction_gap_hours)
    edges = bin_edges(cutoff, spec)
    eid = episode_id(source, kind, row["stay_id"], cutoff)
    meta = {
        "dataset": f"{source}-v2",
        "episode_id": eid,
        "episode_type": kind,
        "stay_id": int(row["stay_id"]),
        "subject_id": int(row["subject_id"]),
        "anchor_kind": ("initiation" if spec.task == "vasopressor_initiation" else "aki_onset") if kind == "positive" else "landmark_control",
        "task": spec.task,
        "anchor_time": pd.Timestamp(row["anchor_time"]).isoformat(),
        "window_start": edges[0].isoformat(),
        "window_end": cutoff.isoformat(),
        "start_time": edges[0].isoformat(),
        "end_time": edges[-2].isoformat(),
        "resample_minutes": int(spec.resample_minutes),
        "shape": [spec.n_bins, len(features)],
        "feature_names": list(features),
        "missing_mask_file": f"shock_episode_{eid}_missing_mask.npy",
        "intervention_features": [f for f in FLAG_FEATURES if f in features],
        "missingness_features": [f for f in MISSINGNESS_FEATURES if f in features],
        "observation_density_features": [f for f in MISSINGNESS_FEATURES if f in features],
        "t_since_admit_hours": float(pd.Timedelta(row["t_since_admit"]).total_seconds() / 3600.0),
        "source_available_features": sorted(available),
        "values_units": "canonical (src/icu/v2/harmonization.py); NaN = not observed; not imputed, not scaled",
    }
    if kind == "control":
        meta["matched_positive_stay_id"] = int(row["matched_stay_id"])
    return meta


def write_episode(directory: Path, meta: Mapping[str, Any], values: np.ndarray, mask: np.ndarray) -> None:
    paths = episode_paths(directory, str(meta["episode_id"]))
    np.save(paths["values"], values)
    np.save(paths["mask"], mask)
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def extract_cohort(
    adapter: SourceAdapter,
    spec: ExtractionSpec,
    out_dir: str | Path,
    *,
    n_positive_patients: int | None = None,
    features: Sequence[str] | None = None,
    chunk_stays: int = 2000,
) -> dict[str, Any]:
    """Run the full extraction for one source and write a v1-layout episode directory.

    ``features`` defaults to the task's inputs: CANONICAL_FEATURES, or AKI_FEATURES.
    """
    if features is None:
        features = AKI_FEATURES if spec.task == "aki_onset" else CANONICAL_FEATURES
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stays = adapter.stays()
    vaso = adapter.vasopressor_intervals()
    task_counts: dict[str, Any] = {}
    if spec.task == "aki_onset":
        creat = creatinine_table(adapter, stays, chunk_stays)
        positive_events, anchors, control_stays, task_counts = aki_cohort(stays, creat, spec)
        del creat
    else:
        positive_events, anchors, control_stays = vaso, vaso, stays
    positives, pos_counts = find_positive_anchors(stays, positive_events, spec)
    if spec.task == "aki_onset":
        pos_counts = {**task_counts, "stays_with_anchor_event": pos_counts.pop("stays_with_vasopressor"), **pos_counts}
    positives, n_extra_positive_stays = limit_positives_per_patient(positives, spec)
    pos_counts["positives_after_per_patient_limit"] = int(len(positives))
    pos_counts["positive_stays_dropped_per_patient_limit"] = n_extra_positive_stays
    positives = sample_positive_patients(positives, n_positive_patients, spec.seed)
    controls, ctl_counts = match_controls(control_stays, anchors, positives, spec)
    if spec.task == "aki_onset":
        ctl_counts["control_stays_excluded_aki_patient"] = ctl_counts.pop("control_stays_excluded_vasopressor_patient")
    available = frozenset(adapter.available_features())
    if "vasopressor_on" in features:
        available = available | {"vasopressor_on"}  # every source provides vasopressor infusions
    if spec.required_vital and spec.required_vital not in available:
        raise ValueError(f"required vital {spec.required_vital!r} is not provided by {adapter.name}")
    episodes = [("positive", row) for row in positives.to_dict("records")] + [("control", row) for row in controls.to_dict("records")]
    by_stay: dict[int, list[tuple[str, dict]]] = {}
    for kind, row in episodes:
        by_stay.setdefault(int(row["stay_id"]), []).append((kind, row))
    stay_ids = sorted(by_stay)
    metas: list[dict[str, Any]] = []
    dropped_no_vital = {"positive": 0, "control": 0}
    for begin in range(0, len(stay_ids), chunk_stays):
        chunk = stay_ids[begin : begin + chunk_stays]
        ev = adapter.events(chunk)
        iv = adapter.interventions(chunk)
        if "vasopressor_on" in features:  # AKI task: vasopressor infusions are an input
            v = vaso[vaso["stay_id"].isin(chunk)]
            flags = pd.DataFrame({"stay_id": v["stay_id"].to_numpy(dtype=np.int64), "start": v["start"].to_numpy(dtype="datetime64[ns]"),
                                  "end": v["end"].to_numpy(dtype="datetime64[ns]"), "feature": "vasopressor_on"})
            if len(flags):
                iv = pd.concat([f for f in (iv, flags) if len(f)], ignore_index=True)
        require_columns(ev, EVENT_COLUMNS, "events")
        require_columns(iv, INTERVENTION_COLUMNS, "interventions")
        ev_idx = ev.groupby("stay_id", sort=False).indices if len(ev) else {}
        iv_idx = iv.groupby("stay_id", sort=False).indices if len(iv) else {}
        for stay_id in chunk:
            stay_ev = ev.iloc[ev_idx[stay_id]] if stay_id in ev_idx else ev.iloc[0:0]
            stay_iv = iv.iloc[iv_idx[stay_id]] if stay_id in iv_idx else iv.iloc[0:0]
            for kind, row in by_stay[stay_id]:
                meta = build_episode_meta(adapter.name, kind, row, spec, features, available)
                cutoff = pd.Timestamp(meta["window_end"])
                values, mask = episode_matrix(stay_ev, stay_iv, cutoff, spec, features, available)
                if not vital_observed_near_cutoff(mask, features, spec):
                    dropped_no_vital[kind] += 1
                    continue
                write_episode(out, meta, values, mask)
                metas.append(meta)
        del ev, iv
    (out / "shock_episode_index.json").write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": adapter.name,
        "spec": spec.to_dict(),
        "n_positive_patients_requested": n_positive_patients,
        "counts": {
            **pos_counts,
            **ctl_counts,
            "dropped_no_recent_vital": dropped_no_vital,
            "positives_written": sum(m["episode_type"] == "positive" for m in metas),
            "controls_written": sum(m["episode_type"] == "control" for m in metas),
        },
        "available_features": sorted(available),
        "unavailable_features": [f for f in features if f not in available],
        "adapter": adapter.provenance(),
    }
    (out / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifest


def episode_records(index_path: str | Path, split: str = "") -> list[EpisodeRecord]:
    """EpisodeRecords from an extraction index (labels from episode_type)."""
    metas = json.loads(Path(index_path).read_text(encoding="utf-8"))
    return [
        EpisodeRecord(
            episode_id=m["episode_id"],
            subject_id=int(m["subject_id"]),
            stay_id=int(m["stay_id"]),
            label=1 if m["episode_type"] == "positive" else 0,
            anchor_kind=m.get("anchor_kind", ""),
            anchor_time=m.get("anchor_time", ""),
            window_start=m.get("window_start", ""),
            window_end=m.get("window_end", ""),
            split=split,
        )
        for m in metas
    ]


def fmg_representation(values: np.ndarray, mask: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """FMG-Bridge's v1 input representation: per-episode imputation + robust scaling.

    The v1 miner was designed on per-episode robust-scaled windows (its change threshold
    is in scaled units), so FMG is mined on this view while every other method uses the
    raw canonical values. Masked entries are ignored when refitting.
    """
    filled = np.where(np.asarray(mask, dtype=bool), 0.0, np.nan_to_num(np.asarray(values, dtype=float)))
    return refit_imputation_and_scaling(filled, np.asarray(mask, dtype=bool), list(feature_names))


def write_fmg_representation(src_dir: str | Path, dst_dir: str | Path) -> dict[str, Any]:
    """Copy an extraction directory into FMG's per-episode-scaled representation."""
    src, dst = Path(src_dir), Path(dst_dir)
    if dst.resolve() == src.resolve():
        raise ValueError("destination must differ from the source directory")
    dst.mkdir(parents=True, exist_ok=True)
    metas = json.loads((src / "shock_episode_index.json").read_text(encoding="utf-8"))
    for meta in metas:
        paths = episode_paths(src, meta["episode_id"])
        values = np.load(paths["values"])
        mask = np.load(paths["mask"])
        scaled = fmg_representation(values, mask, meta["feature_names"])
        out_meta = {**meta, "values_units": "FMG v1 representation: per-episode linear imputation + robust scaling"}
        write_episode(dst, out_meta, scaled, mask)
    (dst / "shock_episode_index.json").write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    return {"n_episodes": len(metas), "src": str(src), "dst": str(dst)}
