"""HiRID 1.1.1 v2 adapter on tiny synthetic raw-stage parts (no PhysioNet data needed)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.icu.v2 import harmonization as hz
from src.icu.v2.extract.common import CANONICAL_FEATURES, NUMERIC_FEATURES, ExtractionSpec, episode_matrix, extract_cohort
from src.icu.v2.extract.hirid import (
    AIRWAY_ID,
    DATA_EXCLUDED_OBS_IDS,
    FEATURE_IDS,
    OBS_FEATURE_ROWS,
    PEEP_SET_ID,
    SEDATION_ROWS,
    URINE_ID,
    VASOPRESSOR_ROWS,
    VENT_IDS,
    VENT_MODE_ID,
    VENT_MODES_INVASIVE,
    VENT_MODES_NIV,
    HiRIDAdapter,
    classify_ventilation_points,
    derive_infusion_intervals,
    drug_name,
    iter_archive_parts,
    urine_volumes,
    ventilation_intervals,
)
from src.icu.v2.materialize import episode_paths

T0 = pd.Timestamp("2150-01-01 00:00")
H = pd.Timedelta(hours=1)
M = pd.Timedelta(minutes=1)

OBS_SCHEMA = pa.schema(
    [
        ("datetime", pa.timestamp("ns")),
        ("entertime", pa.timestamp("ns")),
        ("patientid", pa.int32()),
        ("status", pa.int16()),
        ("stringvalue", pa.string()),
        ("type", pa.string()),
        ("value", pa.float32()),
        ("variableid", pa.int32()),
    ]
)
PHARMA_SCHEMA = pa.schema(
    [
        ("patientid", pa.int32()),
        ("pharmaid", pa.int32()),
        ("givenat", pa.timestamp("ns")),
        ("enteredentryat", pa.timestamp("ns")),
        ("givendose", pa.float32()),
        ("cumulativedose", pa.float32()),
        ("fluidamount_calc", pa.float64()),
        ("cumulfluidamount_calc", pa.float64()),
        ("doseunit", pa.string()),
        ("route", pa.string()),
        ("infusionid", pa.int64()),
        ("typeid", pa.int16()),
        ("subtypeid", pa.float64()),
        ("recordstatus", pa.int16()),
    ]
)


def _obs(pid, t, vid, value, status=8, entered=None, typ=""):
    return {
        "datetime": t, "entertime": entered if entered is not None else t, "patientid": pid, "status": status,
        "stringvalue": "", "type": typ, "value": value, "variableid": vid,
    }


def _rx(pid, pharmaid, infusionid, t, dose, status):
    return {
        "patientid": pid, "pharmaid": pharmaid, "givenat": t, "enteredentryat": t, "givendose": dose,
        "cumulativedose": dose, "fluidamount_calc": 0.0, "cumulfluidamount_calc": 0.0, "doseunit": "µg",
        "route": "cv-inf", "infusionid": infusionid, "typeid": 1, "subtypeid": 8.0, "recordstatus": status,
    }


def _observation_parts():
    part0, part1 = [], []
    # patient 1: positive (norepinephrine infusion from T0+20h)
    for k in range(61):
        part0.append(_obs(1, T0 + k * 30 * M, 200, 80.0))
    part0.append(_obs(1, T0 + 3 * H + 5 * M, 200, 999.0, status=2))  # invalidated
    part0.append(_obs(1, T0 + 7 * H + 7 * M, 200, 0.0))  # implausible HR
    part0.append(_obs(1, T0 + 5 * H, 300, 18.0))  # RR (observation namespace)
    part0.append(_obs(1, T0 + 5 * H, 310, 0.0))  # ventilator spontaneous rate: data-excluded
    part0.append(_obs(1, T0 + 5 * H + M, 5685, 30.0))  # not breaths/min: data-excluded
    part0.append(_obs(1, T0 + 10 * H + 10 * M, 110, 70.0))  # invasive MAP
    part0.append(_obs(1, T0 + 10 * H + 20 * M, 610, 90.0))  # non-invasive MAP
    part0.append(_obs(1, T0 + 9 * H + 30 * M, 610, 95.0))  # non-invasive MAP only bin
    part0.append(_obs(1, T0 + 8 * H, 20000110, 250.0, typ="F"))  # platelets G/l
    part0.append(_obs(1, T0 + 9 * H, 20000110, 200.0, entered=T0 + 9 * H + 30 * M, typ="F"))
    part0.append(_obs(1, T0 + 9 * H, 20000110, 210.0, entered=T0 + 10 * H, typ="C"))  # later correction wins
    part0.append(_obs(1, T0 + 8 * H, 20000600, 88.4, typ="F"))  # creatinine umol/l
    part0.append(_obs(1, T0 + 8 * H, 20005110, 5.0, typ="F"))  # glucose mmol/l
    part0.append(_obs(1, T0 + 8 * H, 20004300, 17.104, typ="F"))  # bilirubin umol/l
    part0.append(_obs(1, T0 + 8 * H, 24000572, 5000.0, typ="F"))  # urine creatinine: excluded
    part0.append(_obs(1, T0 + 8 * H, 4200, 65.0))  # SvO2: excluded
    part0.append(_obs(1, T0 + 2 * H, URINE_ID, 60.0))
    part0.append(_obs(1, T0 + 4 * H, URINE_ID, 60.0))
    part0.append(_obs(1, T0 + 6 * H, 20003200, 80.0, typ="F"))  # urine sodium: excluded
    part0.append(_obs(1, T0 + 6 * H + 30 * M, 2010, 40.0))  # FiO2: not ventilation evidence
    for k in range(13):  # CMV 12:00-14:00 every 10 min, then standby
        part0.append(_obs(1, T0 + 12 * H + k * 10 * M, VENT_MODE_ID, 3.0))
    part0.append(_obs(1, T0 + 14 * H + 10 * M, VENT_MODE_ID, 1.0))
    for k in range(9):
        part0.append(_obs(1, T0 + 12 * H + k * 15 * M, PEEP_SET_ID, 5.0))
    # patient 2: control; NIV 05:00-06:00; PEEP without mode but intubated 20:00-21:00
    for k in range(81):
        part0.append(_obs(2, T0 + k * 30 * M, 200, 90.0))
    part0.append(_obs(2, T0 + 60 * H, 9990002, 3.0))  # administrative entry long after monitoring ended
    for k in range(7):
        part0.append(_obs(2, T0 + 5 * H + k * 10 * M, VENT_MODE_ID, 13.0))
    part0.append(_obs(2, T0 + 19 * H + 50 * M, AIRWAY_ID, 1.0))
    for k in range(5):
        part0.append(_obs(2, T0 + 20 * H + k * 15 * M, PEEP_SET_ID, 8.0))
    # patient 3: minor; patient 4: dobutamine only
    for pid in (3, 4):
        for k in range(61):
            part0.append(_obs(pid, T0 + k * 30 * M, 200, 70.0))
    # patient 5 lives in part 1 (archive only): boluses only
    for k in range(61):
        part1.append(_obs(5, T0 + k * 30 * M, 200, 100.0))
    return part0, part1


def _pharma_parts():
    part0 = [
        _rx(1, 1000656, 1001, T0 + 20 * H, 0.0, 524),
        _rx(1, 1000656, 1001, T0 + 20 * H + 15 * M, 5.0, 8),
        _rx(1, 1000656, 1001, T0 + 20 * H + 30 * M, 0.0, 8),
        _rx(1, 1000656, 1001, T0 + 20 * H + 45 * M, 3.0, 520),
        _rx(1, 1000656, 1001, T0 + 21 * H, 1.0, 776),
        _rx(1, 1000656, 1002, T0 + 19 * H, 5.0, 522),  # invalidated
        _rx(1, 300, 1003, T0 + 5 * H, 200.0, 780),  # metronidazole (pharmaid 300)
        _rx(2, 208, 2001, T0 + 10 * H, 0.0, 524),
        _rx(2, 208, 2001, T0 + 10 * H + 15 * M, 50.0, 8),
        _rx(2, 208, 2001, T0 + 10 * H + 30 * M, 20.0, 776),
        _rx(2, 1000741, 2002, T0 + 11 * H, 0.0, 524),
        _rx(2, 1000741, 2002, T0 + 11 * H + 15 * M, 10.0, 8),
        _rx(2, 1000741, 2002, T0 + 11 * H + 30 * M, 5.0, 776),
        _rx(4, 426, 4001, T0 + 15 * H, 0.0, 524),
        _rx(4, 426, 4001, T0 + 15 * H + 15 * M, 100.0, 8),
        _rx(4, 426, 4001, T0 + 15 * H + 30 * M, 50.0, 776),
    ]
    part1 = [
        _rx(5, 1000658, 5001, T0 + 15 * H, 10.0, 780),  # norepinephrine bolus item
        _rx(5, 1000656, 5002, T0 + 16 * H, 10.0, 780),  # instantaneous record of an infusion item
    ]
    return part0, part1


def _write(rows, schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("hirid")
    raw, subset, staging = root / "raw", root / "subset", root / "staging"
    (raw / "raw_stage").mkdir(parents=True)
    general = pd.DataFrame(
        {
            "patientid": [1, 2, 3, 4, 5],
            "admissiontime": [T0.strftime("%Y-%m-%d %H:%M:%S")] * 5,
            "sex": ["M", "F", "M", "F", "M"],
            "age": [60, 70, 17, 55, 65],
            "discharge_status": ["alive"] * 5,
        }
    )
    general.to_csv(raw / "general_table.csv", index=False)
    for kind, parts, schema, sub, archive in (
        ("observation", _observation_parts(), OBS_SCHEMA, "observation_tables", "observation_tables_parquet.tar.gz"),
        ("pharma", _pharma_parts(), PHARMA_SCHEMA, "pharma_records", "pharma_records_parquet.tar.gz"),
    ):
        for number, rows in enumerate(parts):
            _write(rows, schema, staging / sub / "parquet" / f"part-{number}.parquet")
        # only part 0 is "already extracted"; part 1 must be streamed from the archive
        _write(parts[0], schema, subset / sub / "parquet" / "part-0.parquet")
        with tarfile.open(raw / "raw_stage" / archive, "w:gz") as tar:
            tar.add(staging / sub, arcname=sub)
    cache = root / "cache"
    adapter = HiRIDAdapter(raw, cache, subset_dir=subset, log=lambda message: None)
    manifest = adapter.build_cache()
    return {"raw": raw, "subset": subset, "cache": cache, "root": root, "manifest": manifest}


def _adapter(world, **options) -> HiRIDAdapter:
    return HiRIDAdapter(world["raw"], world["cache"], subset_dir=world["subset"], log=lambda message: None, **options)


# ------------------------------------------------------------------------------ mapping


def test_identifier_sets_follow_the_harmonization_table():
    vent_rows = {int(r.identifier) for r in hz.HIRID_TABLE if r.namespace == "observation" and r.feature == "ventilation_on" and r.decision == "include"}
    assert set(VENT_IDS) == vent_rows
    assert OBS_FEATURE_ROWS[URINE_ID].feature == "urine_output"
    assert {row.feature for row in OBS_FEATURE_ROWS.values()} == set(NUMERIC_FEATURES)
    assert 426 not in VASOPRESSOR_ROWS and 1000441 not in VASOPRESSOR_ROWS and 71 not in VASOPRESSOR_ROWS
    assert {drug_name(r.label) for r in VASOPRESSOR_ROWS.values()} == {"norepinephrine", "epinephrine", "vasopressin"}
    assert {drug_name(r.label) for r in SEDATION_ROWS.values()} == {"propofol", "midazolam", "dexmedetomidine", "fentanyl"}
    assert drug_name("Sufentanyl Perfusor") == "sufentanil" and drug_name("Noradrenalin 20 µg/ml Perfusor") == "norepinephrine"
    assert set(VENT_MODES_NIV) == {12, 13, 14, 17}
    assert set(VENT_MODES_INVASIVE) == {2, 3, 4, 5, 6, 7, 8, 10, 11, 15}
    # numeric ids never include the pharma namespace or excluded look-alikes
    for vid in (4200, 20000800, 24000572, 24000400, 20003200, 320, 24000564):
        assert vid not in FEATURE_IDS


# --------------------------------------------------------------------------- the cache


def test_missing_part_is_streamed_from_the_archive(world):
    manifest = world["manifest"]
    for kind in ("observation", "pharma"):
        info = manifest[kind]
        assert info["complete"] is True
        assert set(info["archive_members"]) == {"0", "1"}
        assert info["parts"]["0"]["source"] == "extracted"
        assert info["parts"]["1"]["source"] == "archive"
    stays = _adapter(world).stays()
    assert 5 in set(stays["stay_id"])  # patient 5 only exists in the streamed part


def test_archive_iteration_reads_only_wanted_members(world):
    archive = world["raw"] / "raw_stage" / "observation_tables_parquet.tar.gz"
    seen = list(iter_archive_parts(archive, lambda number, size: number == 1))
    assert [(n, d is not None) for n, _, d in seen] == [(0, False), (1, True)]
    table = pq.read_table(pa.BufferReader(seen[1][2]))
    assert set(table.column("patientid").to_pylist()) == {5}


def test_second_build_reuses_the_cache(world):
    adapter = _adapter(world)
    assert adapter.table_complete("observation") and adapter.table_complete("pharma")
    before = json.loads((world["cache"] / "manifest.json").read_text(encoding="utf-8"))
    adapter.build_cache()
    after = json.loads((world["cache"] / "manifest.json").read_text(encoding="utf-8"))
    assert before["derived"]["built_utc"] == after["derived"]["built_utc"]


# ------------------------------------------------------------------------------- stays


def test_adult_filter_and_stay_columns(world):
    stays = _adapter(world).stays()
    assert set(stays["stay_id"]) == {1, 2, 4, 5}  # patient 3 is 17
    assert (stays["stay_id"] == stays["subject_id"]).all()
    assert (stays["intime"] == T0).all()
    row = stays.set_index("stay_id").loc[1]
    assert row["outtime"] == T0 + 30 * H  # last vital sign (HR every 30 min to 30 h)
    assert str(stays["intime"].dtype) == "datetime64[ns]" and stays["stay_id"].dtype == np.int64


def test_outtime_rule_last_vital_ignores_late_administrative_entries(world):
    default = _adapter(world).stays().set_index("stay_id")["outtime"]
    literal = _adapter(world, outtime_rule="last_observation").stays().set_index("stay_id")["outtime"]
    assert default.loc[2] == T0 + 40 * H  # last HR
    assert literal.loc[2] == T0 + 60 * H  # APACHE group entry 20 h after the last vital sign
    with pytest.raises(ValueError):
        _adapter(world, outtime_rule="bogus")


# ------------------------------------------------------------------------------ events


def test_namespace_300_is_rr_in_observations_only(world):
    adapter = _adapter(world)
    ev = adapter.events([1])
    rr = ev[ev["feature"] == "RR"]
    assert rr["value"].tolist() == [18.0]  # 310 and 5685 are data-excluded by default
    assert not (ev["value"] == 200.0).any()  # metronidazole dose never becomes a measurement
    iv = adapter.interventions([1])
    assert not ((iv["start"] <= T0 + 5 * H) & (iv["end"] > T0 + 5 * H)).any()
    assert 300 not in set(adapter._derived("infusion_intervals")["pharmaid"])


def test_table_rr_ids_option_restores_310_and_5685(world):
    adapter = _adapter(world, table_rr_ids=True)
    rr = adapter.events([1]).query("feature == 'RR'")
    assert sorted(rr["value"].tolist()) == [18.0, 30.0]  # 310's 0 fails the plausibility screen
    assert set(DATA_EXCLUDED_OBS_IDS) == {310, 5685}
    assert set(DATA_EXCLUDED_OBS_IDS) <= set(FEATURE_IDS)  # still include rows in HIRID_TABLE
    assert not set(DATA_EXCLUDED_OBS_IDS) & set(_adapter(world).feature_ids)


def test_units_duplicates_and_exclusions(world):
    ev = _adapter(world).events([1])
    by = {f: g.sort_values("time") for f, g in ev.groupby("feature")}
    assert by["Platelets"]["value"].tolist() == [250.0, 210.0]  # G/l = 10^9/L; correction kept
    assert by["Creatinine"]["value"].iloc[0] == pytest.approx(1.0, rel=1e-6)
    assert by["Glucose"]["value"].iloc[0] == pytest.approx(5.0 * 18.016, rel=1e-6)
    assert by["Bilirubin_Total"]["value"].iloc[0] == pytest.approx(1.0, rel=1e-6)
    assert "SpO2" not in by  # 4200 (SvO2) excluded
    assert len(by["Creatinine"]) == 1  # urine creatinine excluded
    assert not ((ev["time"] == T0 + 6 * H) & (ev["feature"] == "urine_output")).any()  # urine sodium excluded


def test_status_filter_and_plausibility(world):
    adapter = _adapter(world)
    ev = adapter.events([1])
    hr = ev[ev["feature"] == "HR"]
    assert not (hr["value"] == 999.0).any()  # invalidated
    assert not (hr["value"] == 0.0).any()  # implausible
    assert adapter.provenance()["counters"]["implausible_dropped"]["HR"] == 1


def test_modality_tagging_and_bp_merge(world):
    ev = _adapter(world).events([1])
    bp = ev[ev["feature"] == "MAP"].set_index("value")["modality"].to_dict()
    assert bp == {70.0: "invasive", 90.0: "noninvasive", 95.0: "noninvasive"}
    assert ev.loc[ev["feature"] == "HR", "modality"].isna().all()
    spec = ExtractionSpec()
    values, mask = episode_matrix(ev, ev.iloc[0:0].rename(columns={"time": "start"}).assign(end=pd.NaT), T0 + 11 * H, spec)
    j = list(CANONICAL_FEATURES).index("MAP")
    assert values[-2, j] == 95.0  # bin 09-10: non-invasive only
    assert values[-1, j] == 70.0  # bin 10-11: invasive preferred


def test_urine_rate_becomes_volume_over_the_charting_interval(world):
    ev = _adapter(world).events([1])
    urine = ev[ev["feature"] == "urine_output"].sort_values("time")
    assert urine["time"].tolist() == [T0 + 2 * H, T0 + 4 * H]
    assert urine["value"].tolist() == pytest.approx([120.0, 120.0])  # 60 ml/h x 2 h each


def test_urine_volumes_unit():
    pid = np.array([1, 1, 1, 2])
    t = np.array([T0 + H, T0 + 2 * H, T0 + 20 * H, T0 + 30 * M], dtype="datetime64[ns]")
    intime = np.array([T0, T0, T0, T0], dtype="datetime64[ns]")
    vol, hours = urine_volumes(pid, t, np.array([50.0, 40.0, 10.0, 100.0]), intime, max_interval_hours=12)
    assert hours.tolist() == pytest.approx([1.0, 1.0, 12.0, 0.5])
    assert vol.tolist() == pytest.approx([50.0, 40.0, 120.0, 50.0])


# ------------------------------------------------------------------------- infusions


def test_vasopressor_intervals_from_dose_segments(world):
    vaso = _adapter(world).vasopressor_intervals()
    assert set(vaso["stay_id"]) == {1}  # dobutamine (4) and boluses (5) are not vasopressor infusions
    rows = vaso.sort_values("start")[["start", "end", "drug"]].values.tolist()
    assert rows == [
        [T0 + 20 * H, T0 + 20 * H + 15 * M, "norepinephrine"],
        [T0 + 20 * H + 30 * M, T0 + 21 * H, "norepinephrine"],
    ]


def test_derive_infusion_intervals_unit():
    rec = pd.DataFrame(
        [
            # infusion 1 starts without a start record: first dose back-projected at the next rate
            _rx(1, 1000657, 1, T0 + 2 * H, 10.0, 520),
            _rx(1, 1000657, 1, T0 + 2 * H + 15 * M, 15.0, 8),
            _rx(1, 1000657, 1, T0 + 2 * H + 30 * M, 0.0, 776),
            # bolus and not-administered records never count
            _rx(1, 1000657, 2, T0 + 5 * H, 10.0, 780),
            _rx(1, 1000657, 3, T0 + 6 * H, 0.0, 524),
            _rx(1, 1000657, 3, T0 + 6 * H + 15 * M, 10.0, 544),
            # single usable record: no duration, ignored
            _rx(1, 1000657, 4, T0 + 7 * H, 10.0, 520),
        ]
    )
    intervals, per = derive_infusion_intervals(rec)
    assert len(intervals) == 1
    first = intervals.iloc[0]
    assert first["start"] == T0 + 2 * H - 10 * M  # 10 units at 1 unit/min
    assert first["end"] == T0 + 2 * H + 15 * M
    assert bool(first["backprojected"])
    per = per.set_index("infusionid")
    assert per.loc[2, "n_instant"] == 1 and per.loc[2, "n_intervals"] == 0
    assert per.loc[3, "n_dropped"] == 1 and per.loc[3, "n_intervals"] == 0
    assert per.loc[4, "n_usable"] == 1 and per.loc[4, "n_intervals"] == 0


def test_sedation_excludes_opioids_by_default(world):
    default = _adapter(world).interventions([2])
    sed = default[default["feature"] == "sedation_on"]
    assert list(zip(sed["start"], sed["end"])) == [(T0 + 10 * H, T0 + 10 * H + 30 * M)]
    with_opioids = _adapter(world, include_opioids=True).interventions([2])
    sed = with_opioids[with_opioids["feature"] == "sedation_on"].sort_values("start")
    assert sed["start"].tolist() == [T0 + 10 * H, T0 + 11 * H]


# ----------------------------------------------------------------------- ventilation


def test_ventilation_invasive_by_default_and_niv_option(world):
    adapter = _adapter(world)
    iv = adapter.interventions([1, 2])
    vent = iv[iv["feature"] == "ventilation_on"].sort_values(["stay_id", "start"])
    p1 = vent[vent["stay_id"] == 1]
    assert p1["start"].tolist() == [T0 + 12 * H]
    assert p1["end"].iloc[0] == T0 + 14 * H + 10 * M  # ends at the standby record
    p2 = vent[vent["stay_id"] == 2]
    assert p2["start"].tolist() == [T0 + 20 * H]  # PEEP without mode, airway 'intubiert' -> invasive
    assert p2["end"].iloc[0] == T0 + 21 * H + 15 * M
    niv = _adapter(world, include_niv=True).interventions([2])
    starts = niv.loc[niv["feature"] == "ventilation_on", "start"].sort_values().tolist()
    assert starts == [T0 + 5 * H, T0 + 20 * H]


def test_ventilation_points_and_interval_rules():
    rows = pd.DataFrame(
        [
            {"patientid": 7, "datetime": T0, "variableid": VENT_MODE_ID, "value": 8.0},
            {"patientid": 7, "datetime": T0 + 10 * M, "variableid": PEEP_SET_ID, "value": 5.0},
            {"patientid": 7, "datetime": T0 + 50 * M, "variableid": VENT_MODE_ID, "value": 8.0},
            {"patientid": 7, "datetime": T0 + 3 * H, "variableid": VENT_MODE_ID, "value": 8.0},  # > 60 min gap
            {"patientid": 7, "datetime": T0 + 3 * H + 5 * M, "variableid": VENT_MODE_ID, "value": 1.0},  # standby
            {"patientid": 7, "datetime": T0 + 3 * H + 10 * M, "variableid": VENT_MODE_ID, "value": 8.0},
            {"patientid": 8, "datetime": T0, "variableid": AIRWAY_ID, "value": 3.0},  # mask
            {"patientid": 8, "datetime": T0 + H, "variableid": PEEP_SET_ID, "value": 5.0},
            {"patientid": 9, "datetime": T0, "variableid": PEEP_SET_ID, "value": 5.0},  # no context
        ]
    )
    points = classify_ventilation_points(rows)
    peep = points[points["source"] == "peep"].set_index("patientid")
    assert peep.loc[7, "cls"] == "invasive" and peep.loc[7, "context"] == "mode"
    assert peep.loc[8, "cls"] == "niv" and peep.loc[8, "context"] == "airway"
    assert peep.loc[9, "cls"] == "unknown" and peep.loc[9, "context"] == "none"
    every = ventilation_intervals(points, {"invasive"}, {"off", "niv"}, min_points=1)
    p7 = every[every["patientid"] == 7]
    assert p7["start"].tolist() == [T0, T0 + 3 * H, T0 + 3 * H + 10 * M]
    assert p7["end"].tolist() == [T0 + 50 * M + 15 * M, T0 + 3 * H + 5 * M, T0 + 3 * H + 25 * M]
    assert set(every["patientid"]) == {7}
    default = ventilation_intervals(points, {"invasive"}, {"off", "niv"})  # isolated single points dropped
    assert default["start"].tolist() == [T0] and default["n_points"].tolist() == [3]
    loose = ventilation_intervals(points, {"invasive", "niv", "unknown"}, {"off"}, min_points=1)
    assert set(loose["patientid"]) == {7, 8, 9}


# -------------------------------------------------------------------- contract / e2e


def test_available_features_and_provenance(world):
    adapter = _adapter(world)
    available = adapter.available_features()
    assert {"HR", "RR", "MAP", "Platelets", "urine_output", "ventilation_on", "sedation_on"} <= available
    assert "SpO2" not in available and "Lactate" not in available  # absent from the synthetic data
    prov = adapter.provenance()
    assert prov["stage"].startswith("raw_stage")
    assert prov["tables"]["observation"]["parts_by_source"] == {"extracted": 1, "archive": 1}
    json.dumps(prov, default=str)


def test_extract_cohort_end_to_end(world):
    adapter = _adapter(world)
    out = world["root"] / "episodes"
    # the recent-vital eligibility rule is tested in test_extract_common_v2rules.py
    manifest = extract_cohort(adapter, ExtractionSpec(window_hours=12, prediction_gap_hours=1, required_vital=None), out)
    counts = manifest["counts"]
    assert counts["positives"] == 1 and counts["positives_written"] == 1
    assert counts["controls_written"] == 1
    index = json.loads((out / "shock_episode_index.json").read_text(encoding="utf-8"))
    pos = next(m for m in index if m["episode_type"] == "positive")
    assert pos["stay_id"] == 1 and pd.Timestamp(pos["anchor_time"]) == T0 + 20 * H
    assert pd.Timestamp(pos["window_end"]) == T0 + 19 * H
    paths = episode_paths(out, pos["episode_id"])
    values, mask = np.load(paths["values"]), np.load(paths["mask"])
    assert values.shape == (12, len(CANONICAL_FEATURES))
    hr = list(CANONICAL_FEATURES).index("HR")
    vent = list(CANONICAL_FEATURES).index("ventilation_on")
    assert not mask[:, hr].any() and np.allclose(values[:, hr], 80.0)
    # window 07:00-19:00: ventilation 12:00-14:10 marks bins 12, 13, 14
    assert values[:, vent].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert "unavailable_features" in manifest and "SpO2" in manifest["unavailable_features"]


def test_feature_events_read_only_the_requested_variables(world):
    adapter = _adapter(world)
    creat = adapter.feature_events([1], ["Creatinine"])
    assert adapter._counters["events_calls"] == 0 and adapter._feature_pass_counters["calls"] == 1
    full = adapter.events([1])
    pd.testing.assert_frame_equal(creat, full[full["feature"] == "Creatinine"].reset_index(drop=True))
    assert adapter.feature_events([1], ["no_such_feature"]).empty
