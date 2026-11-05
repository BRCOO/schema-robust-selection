from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from src.icu.mimic_iv_features import build_feature_matrix
from src.icu.v2.task import (
    ELIGIBLE,
    REASON_ANCHOR_INSIDE_CONTROL,
    REASON_ANCHOR_KIND,
    REASON_CONTROL_OVERLAPS_POSITIVE,
    REASON_CONTROL_TOO_CLOSE,
    REASON_INITIATION_WASHOUT,
    REASON_NOT_FIRST_ANCHOR,
    REASON_SHORT_WINDOW,
    V1_FILL_BY_FEATURE,
    TaskSpec,
    apply_task,
    build_stay_anchor_index,
    cutoff_rows,
    cutoff_time,
    episode_eligibility,
    episode_label,
    exclusion_reasons,
    expected_rows,
    is_full_window,
    kept_row_slice,
    reference_time,
    refit_imputation_and_scaling,
    summarize_exclusions,
    transform_episode,
)

FEATURES = ["HR", "MAP", "Lactate", "vasopressor_on", "urine_output"]
H = pd.Timedelta(hours=1)


def _bins(window_start: pd.Timestamp, window_end: pd.Timestamp) -> dict:
    start, end = window_start.floor("60min"), window_end.floor("60min")
    n_rows = int((end - start) / H) + 1
    return {
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "shape": [n_rows, len(FEATURES)],
        "resample_minutes": 60,
        "feature_names": list(FEATURES),
    }


def positive(anchor: str, *, stay: int = 1, kind: str = "initiation", window_start: str | None = None) -> dict:
    """v1-style positive: [max(intime, anchor - 12h), anchor]."""
    anchor_ts = pd.Timestamp(anchor)
    start = pd.Timestamp(window_start) if window_start else anchor_ts - 12 * H
    return {
        "episode_id": f"stay_{stay}_{anchor_ts:%Y%m%d%H%M%S}_positive",
        "episode_type": "positive",
        "stay_id": stay,
        "anchor_kind": kind,
        "anchor_time": anchor_ts.isoformat(),
        "window_start": start.isoformat(),
        "window_end": anchor_ts.isoformat(),
        **_bins(start, anchor_ts),
    }


def control(window_end: str, *, paired_anchor: str, stay: int = 1, kind: str = "escalation", suffix: str = "") -> dict:
    """v1-style control: a 12 h window ending at window_end, carrying its paired anchor."""
    end = pd.Timestamp(window_end)
    start = end - 12 * H
    anchor_ts = pd.Timestamp(paired_anchor)
    return {
        "episode_id": f"stay_{stay}_{anchor_ts:%Y%m%d%H%M%S}_control{suffix}",
        "episode_type": "control",
        "stay_id": stay,
        "anchor_kind": kind,
        "anchor_time": anchor_ts.isoformat(),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        **_bins(start, end),
    }


# ----------------------------------------------------------------------------- TaskSpec


def test_task_spec_defaults_validation_and_serialization() -> None:
    spec = TaskSpec()
    assert spec.prediction_gap_hours == 1.0
    assert spec.excluded_features == ("vasopressor_on",)
    assert spec.anchor_kinds == ("initiation", "escalation")
    assert spec.initiation_washout_hours is None
    assert spec.control_min_hours_from_anchor == 12.0
    assert json.loads(json.dumps(spec.to_dict()))["excluded_features"] == ["vasopressor_on"]
    assert TaskSpec(excluded_features="vasopressor_on").excluded_features == ("vasopressor_on",)
    assert TaskSpec(anchor_kinds=["initiation"]).anchor_kinds == ("initiation",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.prediction_gap_hours = 2.0  # type: ignore[misc]
    for bad in (
        {"prediction_gap_hours": -1.0},
        {"window_hours": 0.0},
        {"resample_minutes": 0},
        {"anchor_kinds": ()},
        {"initiation_washout_hours": -1.0},
        {"control_min_hours_from_anchor": -0.5},
    ):
        with pytest.raises(ValueError):
            TaskSpec(**bad)


# ------------------------------------------------------------------------ cutoff arithmetic


def test_cutoff_positive_anchor_off_the_hour() -> None:
    meta = positive("2136-01-15T09:40:00")
    assert meta["start_time"] == "2136-01-14T21:00:00" and meta["shape"][0] == 13
    assert reference_time(meta) == pd.Timestamp("2136-01-15T09:40:00")
    assert cutoff_time(meta, TaskSpec()) == pd.Timestamp("2136-01-15T08:40:00")
    # bins 21:00 ... 07:00 end by 08:00 <= 08:40; the 08:00 bin ends 09:00 > 08:40
    assert cutoff_rows(meta, TaskSpec()) == 11
    assert kept_row_slice(meta, TaskSpec()) == slice(0, 11)
    assert is_full_window(meta, TaskSpec())


def test_cutoff_positive_anchor_on_the_hour_is_inclusive_at_bin_end() -> None:
    meta = positive("2136-01-15T09:00:00")
    # cutoff 08:00; the 07:00 bin ends exactly at 08:00 and is kept
    assert cutoff_rows(meta, TaskSpec()) == 11
    # gap 0 still drops the anchor bin (09:00 bin ends at 10:00 > 09:00)
    assert cutoff_rows(meta, TaskSpec(prediction_gap_hours=0.0)) == 12
    assert cutoff_rows(positive("2136-01-15T09:40:00"), TaskSpec(prediction_gap_hours=0.0)) == 12
    assert cutoff_rows(meta, TaskSpec(prediction_gap_hours=2.0)) == 10


def test_cutoff_control_uses_its_own_window_end_not_the_paired_anchor() -> None:
    meta = control("2136-01-15T09:40:00", paired_anchor="2136-01-15T21:40:00")
    assert episode_label(meta) == 0
    assert reference_time(meta) == pd.Timestamp("2136-01-15T09:40:00")
    assert cutoff_rows(meta, TaskSpec()) == 11  # same as the positive with the same window end
    assert is_full_window(meta, TaskSpec())


def test_cutoff_rows_edge_cases() -> None:
    meta = positive("2136-01-15T09:40:00")
    assert cutoff_rows(meta, TaskSpec(prediction_gap_hours=13.0)) == 0  # cutoff before the first bin
    empty = {**meta, "shape": [0, len(FEATURES)], "start_time": None, "end_time": None}
    assert cutoff_rows(empty, TaskSpec()) == 0
    no_shape = {key: value for key, value in meta.items() if key != "shape"}
    assert cutoff_rows(no_shape, TaskSpec()) == 11  # row count derived from start/end time
    with pytest.raises(ValueError):
        cutoff_rows({**meta, "resample_minutes": 30}, TaskSpec())
    with pytest.raises(ValueError):
        reference_time({**meta, "episode_type": "unknown"})
    with pytest.raises(ValueError):
        reference_time({**meta, "anchor_time": None})


def test_expected_rows() -> None:
    assert expected_rows(TaskSpec()) == 11
    assert expected_rows(TaskSpec(prediction_gap_hours=0.0)) == 12
    assert expected_rows(TaskSpec(prediction_gap_hours=2.0)) == 10
    assert expected_rows(TaskSpec(prediction_gap_hours=0.5)) == 11  # minimum over minute offsets
    assert expected_rows(TaskSpec(window_hours=1.0, prediction_gap_hours=2.0)) == 0


def test_non_multiple_gap_trims_full_windows_to_equal_length() -> None:
    spec = TaskSpec(prediction_gap_hours=0.5)
    off_hour = positive("2136-01-15T09:40:00")  # cutoff 09:10 -> 12 complete bins
    on_hour = positive("2136-01-15T09:00:00")  # cutoff 08:30 -> 11 complete bins
    assert cutoff_rows(off_hour, spec) == 12 and cutoff_rows(on_hour, spec) == 11
    assert kept_row_slice(off_hour, spec) == slice(1, 12)
    assert kept_row_slice(on_hour, spec) == slice(0, 11)


def test_truncated_positive_is_short() -> None:
    meta = positive("2136-01-15T09:40:00", window_start="2136-01-15T03:15:00")  # admitted 03:15
    assert meta["shape"][0] == 7  # bins 03:00 ... 09:00
    assert cutoff_rows(meta, TaskSpec()) == 5  # bins 03:00 ... 07:00
    assert kept_row_slice(meta, TaskSpec()) == slice(0, 5)
    assert not is_full_window(meta, TaskSpec())
    assert not is_full_window(positive("2136-01-15T09:40:00"), TaskSpec(window_hours=13.0))


# --------------------------------------------------------------------------- transform


def test_transform_episode_truncates_and_drops_excluded_columns() -> None:
    meta = positive("2136-01-15T09:40:00")
    values = np.arange(13 * len(FEATURES), dtype=float).reshape(13, len(FEATURES))
    mask = np.zeros_like(values, dtype=bool)
    mask[3, 0] = True
    original = values.copy()
    spec = TaskSpec(excluded_features=("vasopressor_on", "SOFA"), reimpute_after_cutoff=False)
    new_values, new_mask, names, info = transform_episode(values, mask, meta, spec)
    assert names == ["HR", "MAP", "Lactate", "urine_output"]
    assert new_values.shape == new_mask.shape == (11, 4)
    np.testing.assert_array_equal(new_values, original[:11][:, [0, 1, 2, 4]])
    assert new_mask[3, 0] and new_mask.sum() == 1
    np.testing.assert_array_equal(values, original)  # input untouched
    assert info["rows_before"] == 13 and info["rows_after"] == 11 and info["rows_trimmed_start"] == 0
    assert info["dropped_features"] == ["vasopressor_on"]
    assert info["excluded_not_found"] == ["SOFA"]
    assert info["cutoff_time"] == "2136-01-15T08:40:00"
    assert info["full_window"] is True and info["reimputed"] is False


def test_transform_episode_short_window_and_shape_checks() -> None:
    meta = positive("2136-01-15T09:40:00", window_start="2136-01-15T03:15:00")
    values = np.ones((7, len(FEATURES)))
    _, _, _, info = transform_episode(values, np.zeros_like(values, dtype=bool), meta, TaskSpec())
    assert info["rows_after"] == 5 and info["full_window"] is False
    with pytest.raises(ValueError):
        transform_episode(values, np.zeros((6, len(FEATURES)), dtype=bool), meta, TaskSpec())
    with pytest.raises(ValueError):
        transform_episode(np.ones((8, len(FEATURES))), np.zeros((8, len(FEATURES)), dtype=bool), meta, TaskSpec())
    with pytest.raises(ValueError):
        transform_episode(values[:, :3], np.zeros((7, 3), dtype=bool), meta, TaskSpec())


def _raw_events() -> pd.DataFrame:
    """Raw events for a positive with anchor 2136-01-15 09:40 (window starts 21:40)."""
    events = [
        ("HR", "2136-01-14T21:50", 80.0),
        ("HR", "2136-01-14T23:10", 85.0),
        ("HR", "2136-01-15T01:30", 90.0),
        ("HR", "2136-01-15T04:00", 88.0),
        ("HR", "2136-01-15T06:20", 95.0),
        ("HR", "2136-01-15T08:50", 120.0),  # after the cutoff
        ("HR", "2136-01-15T09:30", 130.0),  # anchor bin
        ("MAP", "2136-01-14T22:15", 75.0),
        ("MAP", "2136-01-15T03:05", 70.0),
        ("MAP", "2136-01-15T09:35", 55.0),  # anchor bin: v1 interpolates kept rows toward it
        ("Lactate", "2136-01-14T23:30", 1.5),
        ("Lactate", "2136-01-15T09:20", 4.0),  # anchor bin
        ("vasopressor_on", "2136-01-14T23:00", 1.0),
        ("vasopressor_on", "2136-01-15T09:40", 1.0),  # the anchor itself
        ("urine_output", "2136-01-14T22:00", 100.0),
        ("urine_output", "2136-01-14T22:30", 50.0),
        ("urine_output", "2136-01-15T05:00", 80.0),
    ]
    agg = {"vasopressor_on": "max", "urine_output": "sum"}
    return pd.DataFrame(
        {
            "event_time": pd.to_datetime([time for _, time, _ in events]),
            "feature_name": [name for name, _, _ in events],
            "value": [value for _, _, value in events],
            "fill": [V1_FILL_BY_FEATURE[name] for name, _, _ in events],
            "agg": [agg.get(name, "mean") for name, _, _ in events],
        }
    )


def test_refit_matches_v1_materialization_of_the_truncated_window() -> None:
    rows = _raw_events()
    fill_map = {name: V1_FILL_BY_FEATURE[name] for name in FEATURES}
    meta = positive("2136-01-15T09:40:00")
    full, full_meta = build_feature_matrix(
        rows, 60, list(FEATURES), meta["window_start"], meta["window_end"], fill_map, include_missing_mask=True
    )
    full_mask = full_meta.pop("missing_mask")
    assert full.shape == (13, len(FEATURES)) and full_meta["start_time"] == meta["start_time"]
    # what v1 would store if the window had ended at the last kept bin (07:00)
    truncated, truncated_meta = build_feature_matrix(
        rows, 60, list(FEATURES), meta["window_start"], "2136-01-15T07:00:00", fill_map, include_missing_mask=True
    )
    truncated_mask = truncated_meta.pop("missing_mask")

    spec = TaskSpec(excluded_features=())
    refit, refit_mask, names, info = transform_episode(full, full_mask, {**meta, **full_meta}, spec)
    assert info["reimputed"] is True and names == FEATURES
    np.testing.assert_array_equal(refit_mask, truncated_mask)
    np.testing.assert_allclose(refit, truncated, rtol=1e-6, atol=1e-6)

    plain, _, _, _ = transform_episode(full, full_mask, {**meta, **full_meta}, TaskSpec(excluded_features=(), reimpute_after_cutoff=False))
    for name in ("HR", "MAP", "Lactate"):  # v1 rows still carry post-cutoff information
        j = FEATURES.index(name)
        assert not np.allclose(plain[:, j], truncated[:, j])
    for name in ("vasopressor_on", "urine_output"):  # zero-fill columns are bin-local
        j = FEATURES.index(name)
        np.testing.assert_array_equal(plain[:, j], truncated[:, j])


def test_refit_handles_unobserved_and_zero_fill_columns() -> None:
    values = np.array([[5.0, 1.0], [7.0, 0.0], [9.0, 1.0]])
    mask = np.array([[True, False], [True, False], [True, False]])
    out = refit_imputation_and_scaling(values, mask, ["Lactate", "vasopressor_on"])
    np.testing.assert_array_equal(out[:, 0], 0.0)  # never observed -> 0.0, as v1
    np.testing.assert_array_equal(out[:, 1], values[:, 1])  # zero-fill untouched


# --------------------------------------------------------------------------- eligibility


def test_build_stay_anchor_index() -> None:
    metas = [
        positive("2136-01-15T09:40:00", stay=1),
        positive("2136-01-15T07:00:00", stay=1),
        positive("2136-01-15T07:00:00", stay=1),  # duplicate anchor
        control("2136-01-14T19:00:00", paired_anchor="2136-01-15T19:00:00", stay=1),
        control("2136-01-01T00:00:00", paired_anchor="2136-01-02T00:00:00", stay=2),
    ]
    index = build_stay_anchor_index(metas)
    assert index == {1: [pd.Timestamp("2136-01-15T07:00:00"), pd.Timestamp("2136-01-15T09:40:00")]}
    with_controls = build_stay_anchor_index(metas, include_control_anchors=True)
    assert with_controls[1][-1] == pd.Timestamp("2136-01-15T19:00:00")
    assert with_controls[2] == [pd.Timestamp("2136-01-02T00:00:00")]
    assert build_stay_anchor_index([]) == {}


def test_positive_eligibility_reasons() -> None:
    anchor = pd.Timestamp("2136-01-15T09:40:00")
    initiation = positive(anchor.isoformat(), kind="initiation")
    escalation = positive(anchor.isoformat(), kind="escalation")
    assert episode_eligibility(initiation, TaskSpec(), [anchor]) == (True, ELIGIBLE)
    assert episode_eligibility(escalation, TaskSpec(anchor_kinds=("initiation",)), [anchor]) == (False, REASON_ANCHOR_KIND)

    first_only = TaskSpec(first_anchor_per_stay_only=True)
    assert episode_eligibility(initiation, first_only, [anchor - 30 * H, anchor]) == (False, REASON_NOT_FIRST_ANCHOR)
    assert episode_eligibility(initiation, first_only, [anchor, anchor + 5 * H]) == (True, ELIGIBLE)

    washout = TaskSpec(initiation_washout_hours=24.0)
    assert episode_eligibility(initiation, washout, [anchor - 10 * H, anchor]) == (False, REASON_INITIATION_WASHOUT)
    assert episode_eligibility(initiation, washout, [anchor - 24 * H, anchor]) == (False, REASON_INITIATION_WASHOUT)
    assert episode_eligibility(initiation, washout, [anchor - 30 * H, anchor]) == (True, ELIGIBLE)
    assert episode_eligibility(escalation, washout, [anchor - 10 * H, anchor]) == (True, ELIGIBLE)  # initiation only

    short = positive(anchor.isoformat(), window_start="2136-01-15T03:15:00")
    assert episode_eligibility(short, TaskSpec(), [anchor]) == (False, REASON_SHORT_WINDOW)
    assert episode_eligibility(short, TaskSpec(require_full_window=False), [anchor]) == (True, ELIGIBLE)
    both = exclusion_reasons(
        positive(anchor.isoformat(), kind="escalation", window_start="2136-01-15T03:15:00"),
        TaskSpec(anchor_kinds=("initiation",)),
        [],
    )
    assert both == [REASON_ANCHOR_KIND, REASON_SHORT_WINDOW]


def test_control_eligibility_reasons_and_boundaries() -> None:
    end = pd.Timestamp("2136-01-10T12:30:00")
    start = end - 12 * H
    meta = control(end.isoformat(), paired_anchor=(end + 36 * H).isoformat())
    spec = TaskSpec()
    assert episode_eligibility(meta, spec, []) == (True, ELIGIBLE)

    inside = exclusion_reasons(meta, spec, [end - 2 * H])
    assert inside == [REASON_ANCHOR_INSIDE_CONTROL, REASON_CONTROL_OVERLAPS_POSITIVE, REASON_CONTROL_TOO_CLOSE]
    assert episode_eligibility(meta, spec, [start])[1] == REASON_ANCHOR_INSIDE_CONTROL  # closed at start
    assert episode_eligibility(meta, spec, [end])[1] == REASON_ANCHOR_INSIDE_CONTROL  # closed at end
    assert episode_eligibility(meta, spec, [end + 3 * H]) == (False, REASON_CONTROL_OVERLAPS_POSITIVE)
    # back-to-back with a positive window (v1 first control): touching is not overlapping
    assert episode_eligibility(meta, spec, [end + 12 * H]) == (True, ELIGIBLE)
    assert episode_eligibility(meta, TaskSpec(control_min_hours_from_anchor=24.0), [end + 12 * H]) == (
        False,
        REASON_CONTROL_TOO_CLOSE,
    )
    assert episode_eligibility(meta, spec, [start - 0.5 * H]) == (True, ELIGIBLE)  # rules look forward only

    lenient = TaskSpec(
        forbid_anchor_inside_control=False,
        forbid_control_overlap_with_positive=False,
        control_min_hours_from_anchor=None,
    )
    assert episode_eligibility(meta, lenient, [end - 2 * H]) == (True, ELIGIBLE)

    near_own_anchor = control(end.isoformat(), paired_anchor=(end + 3 * H).isoformat())
    assert episode_eligibility(near_own_anchor, spec, []) == (False, REASON_CONTROL_OVERLAPS_POSITIVE)


def test_apply_task_on_metadata_only() -> None:
    t = pd.Timestamp("2136-03-10T12:00:00")
    a1, a2 = t, t + 2.5 * H  # clustered anchors in stay 7
    metas = [
        positive(a1.isoformat(), stay=7, kind="initiation"),
        positive(a2.isoformat(), stay=7, kind="escalation"),
        control((a1 - 12 * H).isoformat(), paired_anchor=a1.isoformat(), stay=7),
        control((a1 - 24 * H).isoformat(), paired_anchor=a1.isoformat(), stay=7, suffix="_2"),
        control((a2 - 12 * H).isoformat(), paired_anchor=a2.isoformat(), stay=7),  # overlaps a1's window
        positive("2136-04-01T09:40:00", stay=8, window_start="2136-04-01T03:15:00"),  # truncated
    ]
    kept, excluded = apply_task(metas, TaskSpec())
    assert [meta["episode_id"] for meta in kept] == [metas[i]["episode_id"] for i in (0, 1, 2, 3)]
    assert all(meta is metas[i] for meta, i in zip(kept, (0, 1, 2, 3)))  # metadata passed through
    reasons = {row["episode_id"]: row["reason"] for row in excluded}
    assert reasons == {
        metas[4]["episode_id"]: REASON_CONTROL_OVERLAPS_POSITIVE,
        metas[5]["episode_id"]: REASON_SHORT_WINDOW,
    }
    summary = summarize_exclusions(excluded)
    assert summary["n_excluded"] == 2
    assert summary["primary_by_type"] == {
        "control": {REASON_CONTROL_OVERLAPS_POSITIVE: 1},
        "positive": {REASON_SHORT_WINDOW: 1},
    }
    assert summary["any"][REASON_CONTROL_TOO_CLOSE] == 1

    # an explicit index replaces the one built from the positives
    kept_override, _ = apply_task(metas, TaskSpec(), stay_anchors={7: []})
    assert metas[4] in kept_override

    initiation_only, dropped = apply_task(metas, TaskSpec(anchor_kinds=("initiation",)))
    assert metas[1] not in initiation_only
    assert {row["reason"] for row in dropped} >= {REASON_ANCHOR_KIND}
    assert summarize_exclusions([]) == {"n_excluded": 0, "primary": {}, "any": {}, "primary_by_type": {}}
