from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import pytest

from src.icu.v2.cohort import (
    build_cohort,
    iter_index_records,
    preregistration_record,
    task_spec_from_dict,
    to_episode_record,
    trim_record,
    write_cohort_outputs,
)
from src.icu.v2.splits import assert_patient_disjoint
from src.icu.v2.task import TaskSpec

FEATURES = 18


def _meta(episode_type: str, subject: int, stay: int, anchor: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, kind: str = "initiation") -> dict:
    floor_start = start.floor("60min")
    floor_end = end.floor("60min")
    rows = int((floor_end - floor_start) // pd.Timedelta(minutes=60)) + 1
    tag = f"{episode_type}_{int(start.timestamp())}"
    return {
        "episode_id": f"stay_{stay}_{anchor.strftime('%Y%m%d%H%M%S')}_{tag}",
        "episode_type": episode_type,
        "stay_id": stay,
        "subject_id": subject,
        "hadm_id": stay + 1,
        "anchor_kind": kind,
        "anchor_time": anchor.isoformat(),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "start_time": floor_start.isoformat(),
        "end_time": floor_end.isoformat(),
        "shape": [rows, FEATURES],
        "resample_minutes": 60,
        "feature_names": ["unused"] * FEATURES,
    }


def _synthetic_index(n_patients: int = 12, second_anchor_every: int = 4) -> list[dict]:
    """v1-like layout: positive [T-12h, T], controls back-to-back before it."""
    records: list[dict] = []
    for i in range(n_patients):
        subject, stay = 100 + i, 1000 + i
        anchor = pd.Timestamp("2130-01-01 00:40") + pd.Timedelta(days=3 * i)
        twelve = pd.Timedelta(hours=12)
        records.append(_meta("positive", subject, stay, anchor, anchor - twelve, anchor))
        for k in range(2):
            end = anchor - twelve * (k + 1)
            records.append(_meta("control", subject, stay, anchor, end - twelve, end))
        if i % second_anchor_every == 0:
            # An earlier escalation 18 h before the anchor falls inside the first control.
            early = anchor - pd.Timedelta(hours=18)
            records.append(_meta("positive", subject, stay, early, early - twelve, early, kind="escalation"))
    return records


def _write_array(path, records, *, indent=None) -> None:
    path.write_text(json.dumps(records, indent=indent), encoding="utf-8")


@pytest.mark.parametrize("chunk_size", [7, 64, 1 << 20])
@pytest.mark.parametrize("indent", [None, 2])
def test_iter_index_records_matches_json_load(tmp_path, chunk_size, indent):
    records = _synthetic_index()
    path = tmp_path / "index.json"
    _write_array(path, records, indent=indent)
    assert list(iter_index_records(path, chunk_size=chunk_size)) == json.loads(path.read_text(encoding="utf-8"))


def test_iter_index_records_empty_array(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(" [ ] ", encoding="utf-8")
    assert list(iter_index_records(path, chunk_size=3)) == []


def test_iter_index_records_rejects_truncated_and_non_objects(tmp_path):
    truncated = tmp_path / "truncated.json"
    truncated.write_text('[{"a": 1}, {"b": ', encoding="utf-8")
    with pytest.raises(ValueError, match="truncated"):
        list(iter_index_records(truncated, chunk_size=4))
    scalars = tmp_path / "scalars.json"
    scalars.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="objects"):
        list(iter_index_records(scalars))
    not_array = tmp_path / "object.json"
    not_array.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="array"):
        list(iter_index_records(not_array))


def test_trim_record_drops_heavy_fields():
    record = _synthetic_index(1)[0]
    trimmed = trim_record(record)
    assert "feature_names" not in trimmed
    assert trimmed["episode_id"] == record["episode_id"]


def test_to_episode_record_requires_subject_id():
    meta = trim_record(_synthetic_index(1)[0])
    assert to_episode_record(meta).subject_id == 100
    meta.pop("subject_id")
    with pytest.raises(ValueError, match="subject_id"):
        to_episode_record(meta)
    meta["subject_id"] = float("nan")
    with pytest.raises(ValueError, match="subject_id"):
        to_episode_record(meta)


def test_task_spec_round_trips_through_json():
    spec = TaskSpec(prediction_gap_hours=2.0, anchor_kinds=("initiation",))
    payload = json.loads(json.dumps(spec.to_dict()))
    assert task_spec_from_dict(payload) == spec
    with pytest.raises(ValueError, match="unknown"):
        task_spec_from_dict({**payload, "bogus": 1})


def test_build_cohort_excludes_controls_with_inner_anchor_and_splits_by_patient():
    records = [trim_record(r) for r in _synthetic_index()]
    result = build_cohort(records, TaskSpec(), seed=3)
    reasons = {row["episode_id"]: row["reasons"] for row in result["exclusions"]}
    # Patients 0, 4, 8 have an escalation 18 h before the anchor, inside their first control.
    inner = [eid for eid, rs in reasons.items() if "anchor_inside_control" in rs]
    assert len(inner) == 3
    # The escalation positives overlap the later positive window only through anchors,
    # which do not exclude positives under the default spec.
    assert result["n_eligible"] == len(records) - len(result["exclusions"])
    episodes = result["episodes"]
    assert_patient_disjoint(episodes)
    assert {e.split for e in episodes} == {"train", "val", "test"}
    assert all(e.label in (0, 1) for e in episodes)


def test_build_cohort_initiation_only_drops_escalations():
    records = [trim_record(r) for r in _synthetic_index()]
    result = build_cohort(records, TaskSpec(anchor_kinds=("initiation",)), seed=0)
    kinds = {row["anchor_kind"] for row in result["exclusions"] if row["reason"] == "anchor_kind"}
    assert kinds == {"escalation"}
    assert all(not (e.label == 1 and e.anchor_kind == "escalation") for e in result["episodes"])


def test_preregistration_hash_changes_with_settings(tmp_path):
    a = preregistration_record(TaskSpec(), {"seed": 0}, root=tmp_path)
    b = preregistration_record(TaskSpec(), {"seed": 0}, root=tmp_path)
    c = preregistration_record(TaskSpec(prediction_gap_hours=2.0), {"seed": 0}, root=tmp_path)
    d = preregistration_record(TaskSpec(), {"seed": 0}, root=tmp_path, extra={"primary_gap_hours": 1.0})
    assert a["content_sha256"] == b["content_sha256"]
    assert len({a["content_sha256"], c["content_sha256"], d["content_sha256"]}) == 3


def test_write_outputs_and_cli_end_to_end(tmp_path):
    import importlib.util
    from pathlib import Path

    index = tmp_path / "index.json"
    _write_array(index, _synthetic_index())
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"task_spec": TaskSpec().to_dict()}), encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "build_task_cohort.py"
    module_spec = importlib.util.spec_from_file_location("build_task_cohort", script)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    out_dir = tmp_path / "out"
    assert module.main(["--index", str(index), "--out-dir", str(out_dir), "--task-spec", str(spec_path), "--seed", "1"]) == 0
    episodes = json.loads((out_dir / "episodes_split.json").read_text(encoding="utf-8"))
    assert episodes and {"episode_id", "subject_id", "label", "split"} <= set(episodes[0])
    prereg = json.loads((out_dir / "preregistration.json").read_text(encoding="utf-8"))
    assert prereg["split_params"]["grouping"] == "subject_id"
    assert len(prereg["content_sha256"]) == 64
    summary = json.loads((out_dir / "cohort_summary.json").read_text(encoding="utf-8"))
    assert summary["n_input_records"] == len(_synthetic_index())


def test_write_cohort_outputs_returns_paths(tmp_path):
    records = [trim_record(r) for r in _synthetic_index()]
    spec = TaskSpec()
    result = build_cohort(records, spec, seed=0)
    prereg = preregistration_record(spec, {"seed": 0}, root=tmp_path)
    written = write_cohort_outputs(result, spec, prereg, tmp_path / "o")
    assert set(written) == {
        "episodes_split.json",
        "task_spec.json",
        "split_manifest.json",
        "cohort_summary.json",
        "exclusions.json",
        "preregistration.json",
    }
    rows = json.loads((tmp_path / "o" / "episodes_split.json").read_text(encoding="utf-8"))
    assert rows == [asdict(e) for e in result["episodes"]]


def test_sample_patient_ids_is_order_independent_and_sized():
    from src.icu.v2.cohort import sample_patient_ids

    ids = list(range(100))
    a = sample_patient_ids(ids, 10, seed=1)
    b = sample_patient_ids(reversed(ids), 10, seed=1)
    c = sample_patient_ids(ids, 10, seed=2)
    assert a == b and len(a) == 10 and a != c
    assert a != frozenset(range(10))  # not a prefix
    with pytest.raises(ValueError):
        sample_patient_ids(ids, 0, seed=1)


def test_build_cohort_samples_patients_after_eligibility():
    records = [trim_record(r) for r in _synthetic_index(n_patients=30)]
    full = build_cohort(records, TaskSpec(), seed=0)
    sampled = build_cohort(records, TaskSpec(), seed=0, sample_patients=12, sample_seed=5)
    # exclusions use anchors of ALL records, so they do not change with the sample
    assert sampled["exclusion_summary"] == full["exclusion_summary"]
    assert sampled["n_patients_sampled"] == 12
    assert len({e.subject_id for e in sampled["episodes"]}) == 12
    assert sampled["n_patients_eligible"] == full["n_patients_eligible"] == 30
