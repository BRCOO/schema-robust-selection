from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.icu.v2.extract.common import (
    CANONICAL_FEATURES,
    ExtractionSpec,
    bin_edges,
    episode_matrix,
    episode_records,
    extract_cohort,
    find_positive_anchors,
    fmg_representation,
    match_controls,
    sample_positive_patients,
    write_fmg_representation,
)
from src.icu.v2.materialize import episode_paths

T0 = pd.Timestamp("2100-01-01 00:00")
H = pd.Timedelta(hours=1)
COL = {f: j for j, f in enumerate(CANONICAL_FEATURES)}


class FakeAdapter:
    name = "fake"

    def __init__(self, n_pos: int = 6, n_ctrl: int = 10):
        stays, vaso = [], []
        for i in range(n_pos):
            stays.append({"subject_id": 100 + i, "stay_id": i, "intime": T0, "outtime": T0 + 72 * H})
            vaso.append({"stay_id": i, "start": T0 + (20 + i) * H, "end": T0 + (30 + i) * H, "drug": "norepinephrine"})
            vaso.append({"stay_id": i, "start": T0 + (40 + i) * H, "end": T0 + (45 + i) * H, "drug": "vasopressin"})
        # one positive whose first vasopressor comes too early (5 h after admission)
        stays.append({"subject_id": 900, "stay_id": 900, "intime": T0, "outtime": T0 + 72 * H})
        vaso.append({"stay_id": 900, "start": T0 + 5 * H, "end": T0 + 9 * H, "drug": "norepinephrine"})
        for k in range(n_ctrl):
            los = 10 if k == 0 else 80  # stay 1000 is too short for any match
            stays.append({"subject_id": 500 + k, "stay_id": 1000 + k, "intime": T0, "outtime": T0 + los * H})
        self._stays = pd.DataFrame(stays)
        self._vaso = pd.DataFrame(vaso)

    def stays(self):
        return self._stays

    def vasopressor_intervals(self):
        return self._vaso

    def events(self, stay_ids):
        rows = []
        for s in stay_ids:
            for h in range(0, 60):
                rows.append({"stay_id": s, "time": T0 + h * H + pd.Timedelta(minutes=10), "feature": "HR", "value": 80.0 + h, "modality": None})
            rows.append({"stay_id": s, "time": T0 + 15 * H, "feature": "Lactate", "value": 2.0, "modality": None})
        return pd.DataFrame(rows)

    def interventions(self, stay_ids):
        return pd.DataFrame([{"stay_id": s, "start": T0 + 12 * H, "end": T0 + 14 * H, "feature": "ventilation_on"} for s in stay_ids])

    def available_features(self):
        return frozenset(CANONICAL_FEATURES) - {"Bilirubin_Total"}

    def provenance(self):
        return {"adapter": "fake"}


def test_positive_anchor_is_first_start_with_enough_history():
    adapter = FakeAdapter()
    spec = ExtractionSpec(window_hours=12, prediction_gap_hours=1)
    positives, counts = find_positive_anchors(adapter.stays(), adapter.vasopressor_intervals(), spec)
    assert counts["anchor_too_early"] == 1 and counts["positives"] == 6
    assert (positives["anchor_time"] == T0 + (20 + positives["stay_id"]) * H).all()


def test_controls_have_no_vasopressor_and_cover_the_matched_time():
    adapter = FakeAdapter()
    spec = ExtractionSpec(controls_per_positive=1, seed=3)
    positives, _ = find_positive_anchors(adapter.stays(), adapter.vasopressor_intervals(), spec)
    controls, counts = match_controls(adapter.stays(), adapter.vasopressor_intervals(), positives, spec)
    assert len(controls) == 6 and counts["control_shortfall"] == 0
    assert set(controls["stay_id"]).isdisjoint(set(adapter.vasopressor_intervals()["stay_id"]))
    assert 1000 not in set(controls["stay_id"])  # 10 h stay cannot host a >= 20 h landmark
    assert controls["stay_id"].is_unique
    merged = controls.merge(positives[["stay_id", "t_since_admit"]], left_on="matched_stay_id", right_on="stay_id", suffixes=("", "_pos"))
    assert (merged["t_since_admit"] == merged["t_since_admit_pos"]).all()
    assert (controls["anchor_time"] <= controls["outtime"]).all()


def test_matching_reports_shortfall():
    adapter = FakeAdapter(n_pos=6, n_ctrl=3)
    spec = ExtractionSpec(controls_per_positive=1)
    positives, _ = find_positive_anchors(adapter.stays(), adapter.vasopressor_intervals(), spec)
    _, counts = match_controls(adapter.stays(), adapter.vasopressor_intervals(), positives, spec)
    assert counts["controls"] == 2 and counts["control_shortfall"] == 4


def test_window_ends_at_cutoff_and_excludes_later_data():
    spec = ExtractionSpec(window_hours=12, prediction_gap_hours=1)
    cutoff = T0 + 24 * H
    edges = bin_edges(cutoff, spec)
    assert len(edges) == 13 and edges[-1] == cutoff and edges[0] == cutoff - 12 * H
    events = pd.DataFrame(
        [
            {"stay_id": 1, "time": cutoff - pd.Timedelta(minutes=30), "feature": "HR", "value": 90.0, "modality": None},
            {"stay_id": 1, "time": cutoff, "feature": "HR", "value": 999.0, "modality": None},  # at cutoff: excluded
            {"stay_id": 1, "time": cutoff + H, "feature": "Lactate", "value": 9.0, "modality": None},  # after: excluded
            {"stay_id": 1, "time": cutoff - 13 * H, "feature": "HR", "value": 1.0, "modality": None},  # before window
        ]
    )
    values, mask = episode_matrix(events, pd.DataFrame(columns=["stay_id", "start", "end", "feature"]), cutoff, spec)
    assert values[-1, COL["HR"]] == 90.0 and not mask[-1, COL["HR"]]
    assert np.isnan(values[:, COL["Lactate"]]).all() and mask[:, COL["Lactate"]].all()
    assert np.nanmax(values[:, COL["HR"]]) == 90.0


def test_bp_prefers_invasive_urine_sums_and_interventions_overlap():
    spec = ExtractionSpec(window_hours=4, prediction_gap_hours=0)
    cutoff = T0 + 10 * H
    t = cutoff - 2 * H + pd.Timedelta(minutes=5)  # third bin
    events = pd.DataFrame(
        [
            {"stay_id": 1, "time": t, "feature": "MAP", "value": 60.0, "modality": "invasive"},
            {"stay_id": 1, "time": t, "feature": "MAP", "value": 90.0, "modality": "noninvasive"},
            {"stay_id": 1, "time": t - H, "feature": "MAP", "value": 80.0, "modality": "noninvasive"},
            {"stay_id": 1, "time": t, "feature": "urine_output", "value": 30.0, "modality": None},
            {"stay_id": 1, "time": t + pd.Timedelta(minutes=20), "feature": "urine_output", "value": 20.0, "modality": None},
        ]
    )
    interventions = pd.DataFrame([{"stay_id": 1, "start": cutoff - 90 * pd.Timedelta(minutes=1), "end": pd.NaT, "feature": "sedation_on"}])
    values, mask = episode_matrix(events, interventions, cutoff, spec)
    assert values[2, COL["MAP"]] == 60.0  # invasive wins within the bin
    assert values[1, COL["MAP"]] == 80.0  # non-invasive used when no invasive reading
    assert values[2, COL["urine_output"]] == 50.0
    assert values[:, COL["sedation_on"]].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert not mask[:, COL["sedation_on"]].any() and not mask[:, COL["ventilation_on"]].any()


def test_sampling_is_patient_level_and_order_independent():
    adapter = FakeAdapter()
    positives, _ = find_positive_anchors(adapter.stays(), adapter.vasopressor_intervals(), ExtractionSpec())
    a = sample_positive_patients(positives, 3, seed=1)
    b = sample_positive_patients(positives.iloc[::-1], 3, seed=1)
    assert set(a["subject_id"]) == set(b["subject_id"]) and len(a) == 3


def test_extract_cohort_writes_episodes_index_and_fmg_view(tmp_path):
    adapter = FakeAdapter()
    spec = ExtractionSpec(window_hours=12, prediction_gap_hours=1, controls_per_positive=1)
    manifest = extract_cohort(adapter, spec, tmp_path / "raw")
    assert manifest["counts"]["positives_written"] == 6 and manifest["counts"]["controls_written"] == 6
    assert manifest["unavailable_features"] == ["Bilirubin_Total"]
    index = json.loads((tmp_path / "raw" / "shock_episode_index.json").read_text(encoding="utf-8"))
    pos = next(m for m in index if m["episode_type"] == "positive" and m["stay_id"] == 0)
    assert pos["window_end"] == (T0 + 19 * H).isoformat()  # anchor 20 h - gap 1 h
    values = np.load(episode_paths(tmp_path / "raw", pos["episode_id"])["values"])
    assert values.shape == (12, len(CANONICAL_FEATURES))
    # HR at 10 past each hour: the last bin [18 h, 19 h) holds the 18:10 reading = 98
    assert values[-1, COL["HR"]] == 98.0
    records = episode_records(tmp_path / "raw" / "shock_episode_index.json")
    assert {r.label for r in records} == {0, 1} and len(records) == 12
    out = write_fmg_representation(tmp_path / "raw", tmp_path / "fmg")
    assert out["n_episodes"] == 12
    scaled = np.load(episode_paths(tmp_path / "fmg", pos["episode_id"])["values"])
    assert np.isfinite(scaled).all() and scaled.shape == values.shape
    with pytest.raises(ValueError):
        write_fmg_representation(tmp_path / "raw", tmp_path / "raw")


def test_fmg_representation_ignores_masked_entries():
    values = np.array([[1.0, np.nan], [np.nan, np.nan], [3.0, np.nan]])
    mask = np.array([[False, True], [True, True], [False, True]])
    out = fmg_representation(values, mask, ["HR", "Lactate"])
    assert np.isfinite(out).all()
    assert out[1, 0] == pytest.approx((2.0 - 2.0) / 1.0, abs=1e-6)  # interpolated midpoint at the median
    assert (out[:, 1] == 0.0).all()  # never observed -> 0, as in v1
