"""Leakage-safe task definition for the v2 vasopressor-anchor protocol.

Background (audit P0-1, ``docs/reviews/ICDE2027_558_audit_and_revision_plan.md``):

* v1 positives cover ``[anchor - 12h, anchor]`` and include the anchor instant
  (``src/icu/shock_episode_dataset.py`` lines 77-78; ``searchsorted(side="right")`` at
  line 196). The anchor is a vasopressor start (``shock_anchor_builder.py``) and
  ``vasopressor_on`` is built from the same itemids and ``starttime``
  (``mimic_iv_schema.py``), so the last bin of every positive shows the label.
* Positives truncated at ICU admission have fewer than 13 bins; controls never do.
* Controls are back-to-back windows before the positive of the same stay, without
  overlap or exclusion rules.

This module works on the v1 arrays and metadata as they are on disk. It (1) cuts every
episode on whole bins at ``reference_time - prediction_gap`` with one rule for both
classes, (2) drops anchor-derived features, (3) optionally refits v1's imputation and
scaling on the kept rows, and (4) decides eligibility from metadata only.

v1 bin semantics (``mimic_iv_features.build_feature_matrix``): ``start_time`` is
``floor(window_start)``; row ``i`` aggregates events in
``[start_time + i*r, start_time + (i + 1)*r)``; the last row starts at
``floor(window_end)``. A full 12 h window therefore has 13 rows, the last one holding
the anchor instant.

Every default of :class:`TaskSpec` is a proposal for the team, not a decision.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.icu.mimic_iv_schema import get_feature_specs

POSITIVE = "positive"
CONTROL = "control"

ELIGIBLE = "eligible"
REASON_SHORT_WINDOW = "short_window"
REASON_ANCHOR_KIND = "anchor_kind"
REASON_NOT_FIRST_ANCHOR = "not_first_anchor"
REASON_INITIATION_WASHOUT = "initiation_washout"
REASON_ANCHOR_INSIDE_CONTROL = "anchor_inside_control"
REASON_CONTROL_OVERLAPS_POSITIVE = "control_overlaps_positive"
REASON_CONTROL_TOO_CLOSE = "control_too_close_to_anchor"

# v1 fill mode per feature. "interpolate" features are imputed and robust-scaled per
# episode by build_feature_matrix; "zero" features (presence flags, urine) are not.
V1_FILL_BY_FEATURE: dict[str, str] = {spec.name: spec.fill for spec in get_feature_specs("core")}
_IQR_FLOOR = 1e-9  # same constants as mimic_iv_features._robust_scale_columns


@dataclass(frozen=True)
class TaskSpec:
    """Configurable v2 task. All defaults are proposals pending a team decision.

    Attributes:
        window_hours: History a window must cover before its reference time. v1
            materialized 12 h, so larger values make every v1 episode "short".
        prediction_gap_hours: Bins that end after ``reference_time - gap`` are dropped.
            With hourly bins the last kept bin ends between ``gap`` and ``gap + 1 h``
            before the anchor (1 h -> 11 of the 13 v1 rows remain).
        resample_minutes: Bin width; must match the episode metadata.
        excluded_features: Inputs removed because they share the anchor's source
            (``vasopressor_on`` uses the anchor itemids and ``starttime``).
        require_full_window: Exclude windows truncated at ICU admission (length leak).
        anchor_kinds: Positive anchor kinds kept. 89 % of v1 anchors are escalations,
            some comparing different drugs, so ``("initiation",)`` is a candidate.
        initiation_washout_hours: If set, an initiation positive is excluded when another
            anchor of its stay lies in ``[anchor - washout, anchor)``. ``None`` disables it.
        control_min_hours_from_anchor: A control is excluded when a stay anchor lies in
            ``[window_start, window_end + this)``. ``None`` disables the rule. With the
            default (= ``window_hours``) a v1 first control, which ends exactly 12 h before
            its own anchor, is still allowed.
        forbid_control_overlap_with_positive: Exclude a control whose window overlaps, with
            positive length, ``[a - window_hours, a]`` for some stay anchor ``a``.
        forbid_anchor_inside_control: Exclude a control with a stay anchor in
            ``[window_start, window_end]`` (closed: v1 slices include both ends).
        first_anchor_per_stay_only: Keep only the earliest anchor of a stay as a positive.
        reimpute_after_cutoff: Addition to the brief. Refit v1's per-episode linear
            imputation and robust scaling on the kept rows, so no kept value depends on a
            dropped bin. v1 interpolates and scales over the whole window, anchor bin
            included; robust scaling is invariant to v1's own affine scaling, so the refit
            equals what v1 would have produced for the truncated window.
    """

    window_hours: float = 12.0
    prediction_gap_hours: float = 1.0
    resample_minutes: int = 60
    excluded_features: tuple[str, ...] = ("vasopressor_on",)
    require_full_window: bool = True
    anchor_kinds: tuple[str, ...] = ("initiation", "escalation")
    initiation_washout_hours: float | None = None
    control_min_hours_from_anchor: float | None = 12.0
    forbid_control_overlap_with_positive: bool = True
    forbid_anchor_inside_control: bool = True
    first_anchor_per_stay_only: bool = False
    reimpute_after_cutoff: bool = True

    def __post_init__(self) -> None:
        for name in ("excluded_features", "anchor_kinds"):
            value = getattr(self, name)
            object.__setattr__(self, name, (value,) if isinstance(value, str) else tuple(value))
        if not self.window_hours > 0:
            raise ValueError("window_hours must be positive")
        if self.prediction_gap_hours < 0:
            raise ValueError("prediction_gap_hours must be >= 0")
        if int(self.resample_minutes) <= 0:
            raise ValueError("resample_minutes must be positive")
        if not self.anchor_kinds:
            raise ValueError("anchor_kinds must not be empty")
        for name in ("initiation_washout_hours", "control_min_hours_from_anchor"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 or None")

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON reports."""
        return asdict(self)


# --------------------------------------------------------------------------- time helpers


def _timestamp(value: Any, field: str) -> pd.Timestamp:
    """Parse a metadata time (ISO string or Timestamp); reject missing values."""
    if value is None:
        raise ValueError(f"episode metadata lacks {field!r}")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"episode metadata has an invalid {field!r}: {value!r}")
    return stamp


def _hours(value: float) -> pd.Timedelta:
    return pd.Timedelta(hours=float(value))


def _bin(spec: TaskSpec) -> pd.Timedelta:
    return pd.Timedelta(minutes=int(spec.resample_minutes))


def episode_type(meta: Mapping[str, Any]) -> str:
    """Return ``"positive"`` or ``"control"``; raise on anything else."""
    kind = str(meta.get("episode_type", ""))
    if kind not in (POSITIVE, CONTROL):
        raise ValueError(f"unknown episode_type {kind!r} for {meta.get('episode_id')!r}")
    return kind


def episode_label(meta: Mapping[str, Any]) -> int:
    """1 for positives, 0 for controls."""
    return 1 if episode_type(meta) == POSITIVE else 0


def reference_time(meta: Mapping[str, Any]) -> pd.Timestamp:
    """Time the prediction gap is measured from.

    Positives: ``anchor_time`` (the label event). Controls: their own ``window_end`` used
    as a pseudo-anchor, so both classes are cut the same way relative to the window end.
    A control's ``anchor_time`` is the paired positive's anchor and is not used here.
    """
    if episode_type(meta) == POSITIVE:
        return _timestamp(meta.get("anchor_time"), "anchor_time")
    return _timestamp(meta.get("window_end"), "window_end")


def cutoff_time(meta: Mapping[str, Any], spec: TaskSpec) -> pd.Timestamp:
    """``reference_time(meta) - prediction_gap``; kept bins must end at or before it."""
    return reference_time(meta) - _hours(spec.prediction_gap_hours)


def _check_resample(meta: Mapping[str, Any], spec: TaskSpec) -> None:
    declared = meta.get("resample_minutes")
    if declared is not None and int(declared) != int(spec.resample_minutes):
        raise ValueError(
            f"episode {meta.get('episode_id')!r} uses {declared}-minute bins, "
            f"spec expects {spec.resample_minutes}"
        )


def _declared_rows(meta: Mapping[str, Any], spec: TaskSpec) -> int:
    shape = meta.get("shape")
    if shape:
        return int(shape[0])
    if meta.get("start_time") is None or meta.get("end_time") is None:
        return 0
    span = _timestamp(meta["end_time"], "end_time") - _timestamp(meta["start_time"], "start_time")
    return int(span // _bin(spec)) + 1


def _complete_bins_before_cutoff(meta: Mapping[str, Any], spec: TaskSpec, n_rows: int) -> int:
    if n_rows <= 0:
        return 0
    _check_resample(meta, spec)
    start = _timestamp(meta.get("start_time"), "start_time")
    complete = (cutoff_time(meta, spec) - start) // _bin(spec)
    return int(min(n_rows, max(0, complete)))


def cutoff_rows(meta: Mapping[str, Any], spec: TaskSpec) -> int:
    """Number of leading rows whose bin ends at or before the cutoff.

    Row ``i`` is kept iff ``start_time + (i + 1) * resample <= reference_time - gap``.
    The rule is conservative: with anchor 09:40 and gap 1 h the cutoff is 08:40, so the
    last kept bin is 07:00-08:00 and the newest kept data is 1 h 40 min before the anchor.
    """
    return _complete_bins_before_cutoff(meta, spec, _declared_rows(meta, spec))


def expected_rows(spec: TaskSpec) -> int:
    """Rows a full window keeps: ``floor((window - gap) / resample)``.

    Exact when window and gap are multiples of the bin width (default: 11). Otherwise a
    full window keeps this many or one more, depending on the anchor's minute offset;
    :func:`kept_row_slice` trims that extra oldest row so all full windows have equal length.
    """
    return max(0, int((_hours(spec.window_hours) - _hours(spec.prediction_gap_hours)) // _bin(spec)))


def kept_row_slice(meta: Mapping[str, Any], spec: TaskSpec) -> slice:
    """Rows kept by :func:`transform_episode`: the last ``expected_rows`` of the cutoff rows.

    For v1 data under the default spec the slice always starts at 0.
    """
    stop = cutoff_rows(meta, spec)
    return slice(max(0, stop - expected_rows(spec)), stop)


def is_full_window(meta: Mapping[str, Any], spec: TaskSpec) -> bool:
    """True if the window reaches ``window_hours`` back from its reference time and keeps
    at least :func:`expected_rows` rows after the cutoff.

    v1 positives truncated at ICU admission (``window_start = intime``) fail the first
    test; positives cut short by an ICU discharge before the anchor fail the second.
    """
    start = meta.get("window_start") or meta.get("start_time")
    covers = _timestamp(start, "window_start") <= reference_time(meta) - _hours(spec.window_hours)
    return bool(covers and cutoff_rows(meta, spec) >= expected_rows(spec))


# ----------------------------------------------------------------------- array transform


def refit_imputation_and_scaling(
    values: np.ndarray,
    mask: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Redo v1's per-episode imputation and robust scaling using only the given rows.

    Mirrors ``build_feature_matrix``: for "interpolate" features, linear interpolation by
    position with edge clamping between observed rows (``mask`` False), 0.0 for a column
    never observed, then ``(x - median) / (IQR + 1e-9)`` with IQR < 1e-9 replaced by 1.
    Zero-fill features are returned unchanged. Because robust scaling is invariant to a
    prior positive affine map, refitting v1's scaled values equals scaling the raw values,
    except where v1 itself hit the IQR fallback.
    """
    out = np.array(values, dtype=float, copy=True)
    mask = np.asarray(mask, dtype=bool)
    positions = np.arange(out.shape[0], dtype=float)
    for j, name in enumerate(feature_names):
        if V1_FILL_BY_FEATURE.get(str(name), "interpolate") != "interpolate":
            continue
        column = out[:, j]
        observed = ~mask[:, j] & np.isfinite(column)
        if not observed.any():
            out[:, j] = 0.0
            continue
        filled = np.interp(positions, positions[observed], column[observed])
        q1, q3 = np.percentile(filled, [25, 75])
        iqr = q3 - q1
        if iqr < _IQR_FLOOR:
            iqr = 1.0
        out[:, j] = (filled - np.median(filled)) / (iqr + _IQR_FLOOR)
    return out


def transform_episode(
    values: np.ndarray,
    mask: np.ndarray,
    meta: Mapping[str, Any],
    spec: TaskSpec,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Cut one v1 episode at the prediction cutoff and drop excluded features.

    Returns ``(values, mask, feature_names, info)``; the inputs are not modified. ``info``
    has ``rows_before``, ``rows_after``, ``rows_trimmed_start``, ``dropped_features``,
    ``excluded_not_found``, ``reference_time``, ``cutoff_time``, ``full_window`` and
    ``reimputed``.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError(f"values {values.shape} and mask {mask.shape} must be equal 2-D shapes")
    names = [str(name) for name in meta.get("feature_names", ())]
    if len(names) != values.shape[1]:
        raise ValueError(f"{len(names)} feature names for {values.shape[1]} columns")
    declared = meta.get("shape")
    if declared and int(declared[0]) != values.shape[0]:
        raise ValueError(f"metadata declares {declared[0]} rows, array has {values.shape[0]}")

    stop = _complete_bins_before_cutoff(meta, spec, values.shape[0])
    start = max(0, stop - expected_rows(spec))
    excluded = set(spec.excluded_features)
    keep = [j for j, name in enumerate(names) if name not in excluded]
    new_values = values[start:stop][:, keep]
    new_mask = mask[start:stop][:, keep]
    new_names = [names[j] for j in keep]
    reimputed = bool(spec.reimpute_after_cutoff and new_values.shape[0] > 0)
    if reimputed:
        new_values = refit_imputation_and_scaling(new_values, new_mask, new_names)

    window_start = meta.get("window_start") or meta.get("start_time")
    covers = _timestamp(window_start, "window_start") <= reference_time(meta) - _hours(spec.window_hours)
    info = {
        "rows_before": int(values.shape[0]),
        "rows_after": int(new_values.shape[0]),
        "rows_trimmed_start": int(start),
        "dropped_features": [name for name in names if name in excluded],
        "excluded_not_found": [name for name in spec.excluded_features if name not in names],
        "reference_time": reference_time(meta).isoformat(),
        "cutoff_time": cutoff_time(meta, spec).isoformat(),
        "full_window": bool(covers and stop >= expected_rows(spec)),
        "reimputed": reimputed,
    }
    return new_values, new_mask, new_names, info


# --------------------------------------------------------------------------- eligibility


def build_stay_anchor_index(
    metas: Iterable[Mapping[str, Any]],
    *,
    include_control_anchors: bool = False,
) -> dict[int, list[pd.Timestamp]]:
    """Map ``stay_id`` to its sorted, de-duplicated anchor times.

    By default only positives contribute their ``anchor_time``. A v1 control's
    ``anchor_time`` is its paired anchor, also a real anchor of the stay;
    ``include_control_anchors=True`` adds those. The rules are exact only when the index
    covers every anchor of the stay (the full cohort index); a subsample under-counts.
    """
    index: dict[int, set[pd.Timestamp]] = {}
    for meta in metas:
        if episode_type(meta) != POSITIVE and not include_control_anchors:
            continue
        stay_id = int(meta["stay_id"])
        index.setdefault(stay_id, set()).add(_timestamp(meta.get("anchor_time"), "anchor_time"))
    return {stay_id: sorted(times) for stay_id, times in sorted(index.items())}


def exclusion_reasons(
    meta: Mapping[str, Any],
    spec: TaskSpec,
    stay_anchors: Sequence[pd.Timestamp],
) -> list[str]:
    """All reasons that make an episode ineligible, most fundamental first (empty if eligible).

    Positives: ``anchor_kind``, ``not_first_anchor``, ``initiation_washout``.
    Controls: ``anchor_inside_control``, ``control_overlaps_positive``,
    ``control_too_close_to_anchor``. A control's own paired anchor is always added to
    ``stay_anchors``. Control rules use anchors of every kind and only look forward
    (anchors before ``window_start`` are not checked). Both classes: ``short_window``.
    """
    kind = episode_type(meta)
    anchors = sorted({_timestamp(anchor, "stay anchor") for anchor in stay_anchors})
    reasons: list[str] = []
    if kind == POSITIVE:
        anchor = _timestamp(meta.get("anchor_time"), "anchor_time")
        anchor_kind = str(meta.get("anchor_kind", ""))
        if anchor_kind not in spec.anchor_kinds:
            reasons.append(REASON_ANCHOR_KIND)
        if spec.first_anchor_per_stay_only and any(other < anchor for other in anchors):
            reasons.append(REASON_NOT_FIRST_ANCHOR)
        if spec.initiation_washout_hours is not None and anchor_kind == "initiation":
            washout_start = anchor - _hours(spec.initiation_washout_hours)
            if any(washout_start <= other < anchor for other in anchors):
                reasons.append(REASON_INITIATION_WASHOUT)
    else:
        start = _timestamp(meta.get("window_start"), "window_start")
        end = _timestamp(meta.get("window_end"), "window_end")
        if meta.get("anchor_time") is not None:
            anchors = sorted(set(anchors) | {_timestamp(meta["anchor_time"], "anchor_time")})
        window = _hours(spec.window_hours)
        if spec.forbid_anchor_inside_control and any(start <= a <= end for a in anchors):
            reasons.append(REASON_ANCHOR_INSIDE_CONTROL)
        if spec.forbid_control_overlap_with_positive and any(a - window < end and a > start for a in anchors):
            reasons.append(REASON_CONTROL_OVERLAPS_POSITIVE)
        if spec.control_min_hours_from_anchor is not None:
            horizon = end + _hours(spec.control_min_hours_from_anchor)
            if any(start <= a < horizon for a in anchors):
                reasons.append(REASON_CONTROL_TOO_CLOSE)
    if spec.require_full_window and not is_full_window(meta, spec):
        reasons.append(REASON_SHORT_WINDOW)
    return reasons


def episode_eligibility(
    meta: Mapping[str, Any],
    spec: TaskSpec,
    stay_anchors: Sequence[pd.Timestamp],
) -> tuple[bool, str]:
    """``(True, "eligible")`` or ``(False, first reason)``; see :func:`exclusion_reasons`."""
    reasons = exclusion_reasons(meta, spec, stay_anchors)
    return (not reasons, reasons[0] if reasons else ELIGIBLE)


def apply_task(
    metas: Sequence[Mapping[str, Any]],
    spec: TaskSpec,
    *,
    stay_anchors: Mapping[int, Sequence[pd.Timestamp]] | None = None,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Split episode metadata into kept metas and excluded rows (metadata only, no arrays).

    ``stay_anchors`` defaults to :func:`build_stay_anchor_index` over the positives in
    ``metas``. Excluded rows carry ``episode_id``, ``episode_type``, ``stay_id``,
    ``anchor_kind``, ``reason`` (first) and ``reasons`` (all).
    """
    index = build_stay_anchor_index(metas) if stay_anchors is None else stay_anchors
    kept: list[Mapping[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for meta in metas:
        reasons = exclusion_reasons(meta, spec, index.get(int(meta["stay_id"]), ()))
        if not reasons:
            kept.append(meta)
            continue
        excluded.append(
            {
                "episode_id": meta.get("episode_id"),
                "episode_type": episode_type(meta),
                "stay_id": int(meta["stay_id"]),
                "anchor_kind": meta.get("anchor_kind"),
                "reason": reasons[0],
                "reasons": reasons,
            }
        )
    return kept, excluded


def summarize_exclusions(excluded: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts of excluded rows by first reason, by any reason, and first reason per class."""
    primary = Counter(str(row["reason"]) for row in excluded)
    any_reason = Counter(str(reason) for row in excluded for reason in row.get("reasons", (row["reason"],)))
    by_type: dict[str, Counter] = {}
    for row in excluded:
        by_type.setdefault(str(row["episode_type"]), Counter())[str(row["reason"])] += 1
    return {
        "n_excluded": len(excluded),
        "primary": dict(primary.most_common()),
        "any": dict(any_reason.most_common()),
        "primary_by_type": {kind: dict(counts.most_common()) for kind, counts in sorted(by_type.items())},
    }
