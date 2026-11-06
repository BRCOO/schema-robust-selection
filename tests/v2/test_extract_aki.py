"""AKI-onset task (KDIGO creatinine criterion) in src/icu/v2/extract/common.py."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.icu.v2.extract.common import (
    AKI_FEATURES,
    CANONICAL_FEATURES,
    ExtractionSpec,
    aki_cohort,
    creatinine_table,
    extract_cohort,
    find_positive_anchors,
    kdigo_onsets,
    match_controls,
)
from src.icu.v2.materialize import episode_paths

T0 = pd.Timestamp("2150-01-01")
H = pd.Timedelta(hours=1)
COL = {f: j for j, f in enumerate(AKI_FEATURES)}


def creat(rows):
    """(stay, hours after T0, mg/dL) -> creatinine frame."""
    return pd.DataFrame([{"stay_id": s, "time": T0 + h * H, "value": float(v)} for s, h, v in rows])


def stays_frame(rows):
    """(subject, stay, length of stay in hours), all admitted at T0."""
    return pd.DataFrame([{"subject_id": s, "stay_id": st, "intime": T0, "outtime": T0 + los * H} for s, st, los in rows])


# ---------------------------------------------------------------------------- criterion


def test_absolute_rise_within_48_hours_marks_the_first_onset():
    onsets = kdigo_onsets(creat([(1, 0, 1.0), (1, 20, 1.1), (1, 30, 1.3), (1, 40, 2.0)]), ExtractionSpec())
    assert list(onsets.columns) == ["stay_id", "start", "end", "drug"]
    assert onsets["stay_id"].tolist() == [1] and onsets["start"].tolist() == [T0 + 30 * H]


def test_relative_rise_uses_the_7_day_minimum_and_old_baselines_do_not_count():
    assert kdigo_onsets(creat([(1, 0, 1.0), (1, 60, 1.3)]), ExtractionSpec()).empty  # +0.3 but baseline > 48 h old; 1.3 < 1.5x
    onsets = kdigo_onsets(creat([(2, 0, 0.6), (2, 60, 0.9)]), ExtractionSpec())  # 0.9 >= 1.5 x 0.6 within 7 d
    assert onsets["start"].tolist() == [T0 + 60 * H]
    assert kdigo_onsets(creat([(3, 0, 0.6), (3, 24 * 7 + 1, 0.95)]), ExtractionSpec()).empty


def test_rise_is_measured_from_the_window_minimum():
    onsets = kdigo_onsets(creat([(1, 0, 2.0), (1, 10, 1.5), (1, 20, 1.8)]), ExtractionSpec())  # 1.8 - 1.5 = 0.3
    assert onsets["start"].tolist() == [T0 + 20 * H]


def test_simultaneous_duplicates_and_single_values_are_not_onsets():
    assert kdigo_onsets(creat([(1, 5, 1.0), (1, 5, 1.4), (2, 3, 5.0)]), ExtractionSpec()).empty


def test_spec_validates_the_task_and_criterion():
    with pytest.raises(ValueError):
        ExtractionSpec(task="sepsis")
    with pytest.raises(ValueError):
        ExtractionSpec(aki_rel_rise=1.0)
    with pytest.raises(ValueError):
        ExtractionSpec(aki_min_creatinine_for_control=1)


# ---------------------------------------------------------------------------- cohort


def test_aki_cohort_excludes_high_first_creatinine_and_records_the_at_risk_window():
    stays = stays_frame([(1, 10, 100), (2, 20, 100), (3, 30, 100), (4, 40, 100)])
    c = creat([(10, 2, 1.0), (10, 30, 1.4),  # onset at 30 h
               (20, 1, 4.2), (20, 30, 4.8),  # first value >= 4 mg/dL: neither positive nor control
               (30, 5, 0.9), (30, 60, 0.9),  # control source, at risk on (5 h, 60 h]
               (40, 5, 1.0)])  # one measurement: not a control source
    positive_events, onsets, control_stays, counts = aki_cohort(stays, c, ExtractionSpec())
    assert positive_events["stay_id"].tolist() == [10]
    assert sorted(onsets["stay_id"]) == [10, 20]
    assert set(control_stays["stay_id"]) == {10, 20, 30}  # onset stays stay so their patients are excluded
    row = control_stays.set_index("stay_id").loc[30]
    assert row["at_risk_from"] == 5 * H and row["at_risk_until"] == 60 * H
    assert counts["stays_excluded_high_first_creatinine"] == 1
    assert counts["control_source_stays"] == 1 and counts["aki_onsets_after_exclusion"] == 1


def test_controls_must_be_at_risk_at_the_landmark():
    stays = stays_frame([(1, 10, 100), (2, 20, 100), (3, 30, 100), (4, 40, 100)])
    c = creat([(10, 2, 1.0), (10, 30, 1.4),  # positive, onset (landmark) at 30 h
               (20, 31, 1.0), (20, 50, 1.0),  # first creatinine after the landmark
               (30, 1, 1.0), (30, 29, 1.0),  # no creatinine at or after the landmark
               (40, 10, 1.0), (40, 30, 1.0)])  # before and exactly at the landmark: at risk
    spec = ExtractionSpec(window_hours=12, prediction_gap_hours=6, controls_per_positive=3)
    positive_events, onsets, control_stays, _ = aki_cohort(stays, c, spec)
    positives, _ = find_positive_anchors(stays, positive_events, spec)
    controls, counts = match_controls(control_stays, onsets, positives, spec)
    assert controls["stay_id"].tolist() == [40] and controls["anchor_time"].tolist() == [T0 + 30 * H]
    assert counts["control_shortfall"] == 2


class AKIAdapter:
    """Five stays. Subject 1: stay 10 has AKI at 30 h (vasopressor 20-22 h), stay 11 none.
    Subjects 2 and 3: control sources (stay 30 gets a vasopressor at 15-16 h). Subject 4:
    AKI at 10 h, too early for an 18 h lookback."""

    name = "fake_aki"

    def __init__(self, feature_events: bool = True) -> None:
        if not feature_events:
            self.feature_events = None  # common.creatinine_table falls back to events()
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def stays(self):
        return stays_frame([(1, 10, 100), (1, 11, 100), (2, 20, 100), (3, 30, 100), (4, 40, 100)])

    def vasopressor_intervals(self):
        return pd.DataFrame([{"stay_id": 10, "start": T0 + 20 * H, "end": T0 + 22 * H, "drug": "norepinephrine"},
                             {"stay_id": 30, "start": T0 + 15 * H, "end": T0 + 16 * H, "drug": "norepinephrine"}])

    _creatinine = [(10, 2, 1.0), (10, 20, 1.1), (10, 30, 1.5), (11, 5, 1.0), (11, 50, 1.0),
                   (20, 5, 0.9), (20, 60, 0.9), (30, 3, 0.8), (30, 40, 0.85), (40, 1, 1.0), (40, 10, 1.5)]

    def events(self, stay_ids):
        self.calls.append(("events", tuple(stay_ids)))
        ids = set(int(s) for s in stay_ids)
        rows = [{"stay_id": s, "time": T0 + h * H + pd.Timedelta(minutes=30), "feature": "HR", "value": 80.0 + h, "modality": None}
                for s in sorted(ids) for h in range(100)]
        rows += [{"stay_id": s, "time": T0 + h * H, "feature": "Creatinine", "value": v, "modality": None}
                 for s, h, v in self._creatinine if s in ids]
        return pd.DataFrame(rows, columns=["stay_id", "time", "feature", "value", "modality"])

    def feature_events(self, stay_ids, features):
        ev = self.events(stay_ids)
        self.calls[-1] = ("feature_events", tuple(stay_ids))
        return ev[ev["feature"].isin(set(features))].reset_index(drop=True)

    def interventions(self, stay_ids):
        return pd.DataFrame(columns=["stay_id", "start", "end", "feature"])

    def available_features(self):
        return frozenset(CANONICAL_FEATURES)

    def provenance(self):
        return {"adapter": "AKIAdapter"}


def test_extract_cohort_aki_end_to_end(tmp_path):
    adapter = AKIAdapter()
    spec = ExtractionSpec(window_hours=12, prediction_gap_hours=6, controls_per_positive=2, task="aki_onset")
    manifest = extract_cohort(adapter, spec, tmp_path / "out")
    counts = manifest["counts"]
    assert counts["stays_with_aki_onset"] == 2 and counts["stays_with_anchor_event"] == 2
    assert counts["anchor_too_early"] == 1 and counts["positives_written"] == 1 and counts["controls_written"] == 2
    assert "stays_with_vasopressor" not in counts and "control_stays_excluded_vasopressor_patient" not in counts
    assert counts["control_stays_excluded_aki_patient"] == 1  # stay 11
    assert manifest["available_features"] == sorted(set(CANONICAL_FEATURES) | {"vasopressor_on"})
    assert adapter.calls[0][0] == "feature_events"  # creatinine pass used the fast path
    json.dumps(manifest, default=str)
    index = json.loads((tmp_path / "out" / "shock_episode_index.json").read_text(encoding="utf-8"))
    by_stay = {m["stay_id"]: m for m in index}
    assert set(by_stay) == {10, 20, 30}  # stay 11 belongs to a patient with AKI
    pos = by_stay[10]
    assert pos["anchor_kind"] == "aki_onset" and pos["task"] == "aki_onset"
    assert pos["anchor_time"] == (T0 + 30 * H).isoformat() and pos["window_end"] == (T0 + 24 * H).isoformat()
    assert pos["feature_names"] == list(AKI_FEATURES) and "vasopressor_on" in pos["intervention_features"]
    values = np.load(episode_paths(tmp_path / "out", pos["episode_id"])["values"])
    mask = np.load(episode_paths(tmp_path / "out", pos["episode_id"])["mask"])
    assert values.shape == (12, 18)
    # bin i covers [12 + i, 13 + i) hours after admission
    assert values[:, COL["vasopressor_on"]].tolist() == [0.0] * 8 + [1.0, 1.0] + [0.0] * 2
    assert not mask[:, COL["vasopressor_on"]].any()
    assert values[8, COL["Creatinine"]] == 1.1 and int((~mask[:, COL["Creatinine"]]).sum()) == 1  # 30 h value is after the cutoff
    control = by_stay[30]
    assert control["anchor_kind"] == "landmark_control" and control["anchor_time"] == (T0 + 30 * H).isoformat()
    cvalues = np.load(episode_paths(tmp_path / "out", control["episode_id"])["values"])
    assert cvalues[:, COL["vasopressor_on"]].tolist() == [0.0] * 3 + [1.0] + [0.0] * 8


def test_creatinine_pass_without_feature_events_gives_the_same_cohort(tmp_path):
    spec = ExtractionSpec(window_hours=12, prediction_gap_hours=6, controls_per_positive=2, task="aki_onset")
    fast = extract_cohort(AKIAdapter(), spec, tmp_path / "fast")
    slow_adapter = AKIAdapter(feature_events=False)
    slow = extract_cohort(slow_adapter, spec, tmp_path / "slow")
    assert fast["counts"] == slow["counts"]
    assert all(kind == "events" for kind, _ in slow_adapter.calls)
    fast_index = json.loads((tmp_path / "fast" / "shock_episode_index.json").read_text(encoding="utf-8"))
    slow_index = json.loads((tmp_path / "slow" / "shock_episode_index.json").read_text(encoding="utf-8"))
    assert [m["episode_id"] for m in fast_index] == [m["episode_id"] for m in slow_index]


def test_creatinine_table_keeps_icu_time_only():
    class Adapter(AKIAdapter):
        def stays(self):
            return pd.DataFrame([{"subject_id": 1, "stay_id": 10, "intime": T0 + 3 * H, "outtime": T0 + 25 * H}])

    table = creatinine_table(Adapter(), Adapter().stays())
    assert table["time"].tolist() == [T0 + 20 * H]  # 2 h is before admission, 30 h after discharge
    assert list(table.columns) == ["stay_id", "time", "value"]


def test_vasopressor_task_is_unchanged():
    spec = ExtractionSpec()
    assert spec.task == "vasopressor_initiation"
    assert "vasopressor_on" not in CANONICAL_FEATURES
