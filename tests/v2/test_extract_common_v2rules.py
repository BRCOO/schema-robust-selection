from __future__ import annotations

import numpy as np
import pandas as pd

from src.icu.v2.extract.common import (
    CANONICAL_FEATURES,
    ExtractionSpec,
    episode_matrix,
    extract_cohort,
    find_positive_anchors,
    limit_positives_per_patient,
    match_controls,
    vital_observed_near_cutoff,
)

T0 = pd.Timestamp("2100-01-01 00:00")
H = pd.Timedelta(hours=1)
COL = {f: j for j, f in enumerate(CANONICAL_FEATURES)}
EMPTY_IV = pd.DataFrame(columns=["stay_id", "start", "end", "feature"])


def _stays(rows):
    return pd.DataFrame(rows)


def test_one_positive_per_patient_keeps_earliest_stay():
    stays = _stays(
        [
            {"subject_id": 1, "stay_id": 10, "intime": T0 + 100 * H, "outtime": T0 + 200 * H},
            {"subject_id": 1, "stay_id": 11, "intime": T0, "outtime": T0 + 90 * H},
            {"subject_id": 2, "stay_id": 20, "intime": T0, "outtime": T0 + 90 * H},
        ]
    )
    vaso = pd.DataFrame(
        [{"stay_id": s, "start": stays.set_index("stay_id").loc[s, "intime"] + 20 * H, "end": pd.NaT, "drug": "norepinephrine"} for s in (10, 11, 20)]
    )
    positives, _ = find_positive_anchors(stays, vaso, ExtractionSpec())
    kept, dropped = limit_positives_per_patient(positives, ExtractionSpec())
    assert dropped == 1 and sorted(kept["stay_id"]) == [11, 20]
    all_kept, none_dropped = limit_positives_per_patient(positives, ExtractionSpec(max_positives_per_patient=None))
    assert none_dropped == 0 and len(all_kept) == 3


def test_tied_intime_uses_a_deterministic_hash():
    stays = _stays([{"subject_id": 1, "stay_id": s, "intime": T0, "outtime": T0 + 90 * H} for s in (5, 6, 7)])
    vaso = pd.DataFrame([{"stay_id": s, "start": T0 + 20 * H, "end": pd.NaT, "drug": "x"} for s in (5, 6, 7)])
    positives, _ = find_positive_anchors(stays, vaso, ExtractionSpec())
    a, _ = limit_positives_per_patient(positives, ExtractionSpec(seed=1))
    b, _ = limit_positives_per_patient(positives.iloc[::-1], ExtractionSpec(seed=1))
    assert len(a) == 1 and a["stay_id"].tolist() == b["stay_id"].tolist()


def test_controls_match_within_hospital_and_exclude_vasopressor_patients():
    rows = [
        {"subject_id": 1, "stay_id": 1, "intime": T0, "outtime": T0 + 90 * H, "hospitalid": 7},
        {"subject_id": 2, "stay_id": 2, "intime": T0, "outtime": T0 + 90 * H, "hospitalid": 8},
        # controls: one per hospital, plus a stay of patient 1 (who had vasopressors) in hospital 7
        {"subject_id": 3, "stay_id": 3, "intime": T0, "outtime": T0 + 90 * H, "hospitalid": 7},
        {"subject_id": 4, "stay_id": 4, "intime": T0, "outtime": T0 + 90 * H, "hospitalid": 8},
        {"subject_id": 1, "stay_id": 5, "intime": T0 + 300 * H, "outtime": T0 + 400 * H, "hospitalid": 7},
    ]
    stays = _stays(rows)
    vaso = pd.DataFrame([{"stay_id": s, "start": T0 + 20 * H, "end": pd.NaT, "drug": "x"} for s in (1, 2)])
    spec = ExtractionSpec(match_columns=("hospitalid",))
    positives, _ = find_positive_anchors(stays, vaso, spec)
    controls, counts = match_controls(stays, vaso, positives, spec)
    pairs = dict(zip(controls["matched_stay_id"], controls["stay_id"]))
    assert pairs == {1: 3, 2: 4}
    assert counts["control_stays_excluded_vasopressor_patient"] == 1
    assert counts["control_pool"] == 2
    _, loose_counts = match_controls(stays, vaso, positives, ExtractionSpec(exclude_vasopressor_patients_from_controls=False))
    assert loose_counts["control_pool"] == 3  # stay 5 of patient 1 is eligible again


def test_unavailable_intervention_stays_missing():
    spec = ExtractionSpec(window_hours=4, prediction_gap_hours=0)
    available = set(CANONICAL_FEATURES) - {"ventilation_on"}
    values, mask = episode_matrix(pd.DataFrame(columns=["stay_id", "time", "feature", "value"]), EMPTY_IV, T0 + 10 * H, spec, available=available)
    assert mask[:, COL["ventilation_on"]].all() and np.isnan(values[:, COL["ventilation_on"]]).all()
    assert not mask[:, COL["sedation_on"]].any()


def test_categorical_columns_keep_the_invasive_first_merge():
    spec = ExtractionSpec(window_hours=2, prediction_gap_hours=0)
    cutoff = T0 + 10 * H
    t = cutoff - pd.Timedelta(minutes=30)
    events = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "time": [t, t],
            "feature": pd.Categorical(["MAP", "MAP"]),
            "value": [60.0, 90.0],
            "modality": pd.Categorical(["invasive", "noninvasive"]),
        }
    )
    values, _ = episode_matrix(events, EMPTY_IV, cutoff, spec)
    assert values[-1, COL["MAP"]] == 60.0


def test_vital_rule_drops_episodes_without_recent_hr_for_both_classes(tmp_path):
    class Adapter:
        name = "fake"

        def __init__(self):
            self._stays = _stays(
                [{"subject_id": 100 + i, "stay_id": i, "intime": T0, "outtime": T0 + 72 * H} for i in range(4)]
                + [{"subject_id": 500 + i, "stay_id": 50 + i, "intime": T0, "outtime": T0 + 80 * H} for i in range(4)]
            )
            self._vaso = pd.DataFrame([{"stay_id": i, "start": T0 + 30 * H, "end": pd.NaT, "drug": "x"} for i in range(4)])

        def stays(self):
            return self._stays

        def vasopressor_intervals(self):
            return self._vaso

        def events(self, stay_ids):
            rows = []
            for s in stay_ids:
                last = 25 if s in (0, 50) else 29  # stays 0 and 50 lose monitoring before the cutoff (29 h)
                rows += [{"stay_id": s, "time": T0 + h * H + pd.Timedelta(minutes=5), "feature": "HR", "value": 80.0, "modality": None} for h in range(0, last)]
            return pd.DataFrame(rows)

        def interventions(self, stay_ids):
            return EMPTY_IV

        def available_features(self):
            return frozenset(CANONICAL_FEATURES)

        def provenance(self):
            return {}

    manifest = extract_cohort(Adapter(), ExtractionSpec(seed=0), tmp_path / "out")
    dropped = manifest["counts"]["dropped_no_recent_vital"]
    assert dropped["positive"] == 1
    assert manifest["counts"]["positives_written"] == 3
    assert dropped["control"] + manifest["counts"]["controls_written"] == 4


def test_vital_rule_can_be_disabled():
    spec = ExtractionSpec(required_vital=None)
    mask = np.ones((spec.n_bins, len(CANONICAL_FEATURES)), dtype=bool)
    assert vital_observed_near_cutoff(mask, CANONICAL_FEATURES, spec)
    assert not vital_observed_near_cutoff(mask, CANONICAL_FEATURES, ExtractionSpec())
