"""MimicIVAdapter on tiny synthetic MIMIC-IV-layout csv.gz files (no real data)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.icu.v2.extract.common import CANONICAL_FEATURES, ExtractionSpec, extract_cohort
from src.icu.v2.extract.mimic_iv import MimicIVAdapter
from src.icu.v2.materialize import episode_paths

T0 = pd.Timestamp("2150-01-01 00:00:00")
COL = {f: j for j, f in enumerate(CANONICAL_FEATURES)}


def at(hours: float) -> str:
    return (T0 + pd.Timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")


def _csv_gz(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, compression="gzip")


def _gz_bytes(rows: list[dict], columns: list[str]) -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame(rows, columns=columns).to_csv(buffer, index=False, compression={"method": "gzip"})
    return buffer.getvalue()


ICU_COLS = ["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime", "los"]
PAT_COLS = ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"]
CHART_COLS = ["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "storetime", "itemid", "value", "valuenum", "valueuom", "warning"]
LAB_COLS = ["labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid", "order_provider_id", "charttime", "storetime", "value",
            "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "priority", "comments"]
INPUT_COLS = ["subject_id", "hadm_id", "stay_id", "caregiver_id", "starttime", "endtime", "storetime", "itemid", "amount", "amountuom",
              "rate", "rateuom", "orderid", "linkorderid", "ordercategoryname", "secondaryordercategoryname",
              "ordercomponenttypedescription", "ordercategorydescription", "patientweight", "totalamount", "totalamountuom",
              "isopenbag", "continueinnextdept", "statusdescription", "originalamount", "originalrate"]
OUTPUT_COLS = ["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "storetime", "itemid", "value", "valueuom"]
PROC_COLS = ["subject_id", "hadm_id", "stay_id", "caregiver_id", "starttime", "endtime", "storetime", "itemid", "value", "valueuom",
             "location", "locationcategory", "orderid", "linkorderid", "ordercategoryname", "ordercategorydescription",
             "patientweight", "isopenbag", "continueinnextdept", "statusdescription", "originalamount", "originalrate"]


def _stay(subject: int, stay: int, start: float, end: float) -> dict:
    return {"subject_id": subject, "hadm_id": subject * 10, "stay_id": stay, "first_careunit": "MICU", "last_careunit": "MICU",
            "intime": at(start), "outtime": at(end), "los": (end - start) / 24}


def _chart(stay: int, hours: float, itemid: int, value: float, uom: str) -> dict:
    return {"subject_id": stay // 10, "hadm_id": stay, "stay_id": stay, "charttime": at(hours), "itemid": itemid, "value": str(value),
            "valuenum": value, "valueuom": uom}


def _lab(subject: int, hours: float, itemid: int, value, uom: str, specimen: int = 0, hadm=None, text=None) -> dict:
    return {"subject_id": subject, "hadm_id": hadm, "specimen_id": specimen, "itemid": itemid, "charttime": at(hours),
            "value": text if text is not None else str(value), "valuenum": value, "valueuom": uom}


def _drug(stay: int, start: float, end: float, itemid: int, rate, category: str, status: str) -> dict:
    return {"subject_id": stay // 10, "stay_id": stay, "starttime": at(start), "endtime": at(end), "itemid": itemid,
            "amount": 1.0, "rate": rate, "rateuom": "mcg/kg/min" if rate == rate and rate is not None else None,
            "ordercategorydescription": category, "statusdescription": status}


def _out(stay: int, hours: float, itemid: int, value: float) -> dict:
    return {"subject_id": stay // 10, "stay_id": stay, "charttime": at(hours), "itemid": itemid, "value": value, "valueuom": "mL"}


def _proc(stay: int, start: float, end: float, itemid: int, status: str = "FinishedRunning") -> dict:
    return {"subject_id": stay // 10, "stay_id": stay, "starttime": at(start), "endtime": at(end), "itemid": itemid,
            "value": (end - start) * 60, "valueuom": "min", "statusdescription": status}


def make_raw(root: Path, patients_in_zip: bool = False) -> Path:
    """Synthetic raw_dir. Stay 11 (subject 1) gets a vasopressor at 20 h, so its window is
    [7 h, 19 h); stay 51 starts angiotensin II at 20 h; 21, 41 and 42 have no vasopressor
    infusion; stay 31 belongs to a 16-year-old."""
    raw = root / "mimic-iv-3.1"
    stays = [_stay(1, 11, 0, 72), _stay(2, 21, 0, 80), _stay(3, 31, 0, 60), _stay(4, 41, 0, 48), _stay(4, 42, 100, 150),
             _stay(5, 51, 0, 60)]
    _csv_gz(raw / "icu" / "icustays.csv.gz", stays, ICU_COLS)
    patients = [
        {"subject_id": s, "gender": "F", "anchor_age": age, "anchor_year": 2150, "anchor_year_group": "2014 - 2016", "dod": None}
        for s, age in ((1, 60), (2, 55), (3, 16), (4, 70), (5, 45))
    ]
    if patients_in_zip:
        with zipfile.ZipFile(root / "mimic-iv-3.1.zip", "w") as zf:
            zf.writestr("mimic-iv-3.1/hosp/patients.csv.gz", _gz_bytes(patients, PAT_COLS))
            zf.writestr("mimic-iv-3.1/icu/d_items.csv.gz", _gz_bytes([{"itemid": 220045, "label": "Heart Rate"}], ["itemid", "label"]))
    else:
        _csv_gz(raw / "hosp" / "patients.csv.gz", patients, PAT_COLS)
    chart = [
        _chart(11, 10.0, 220045, 80.0, "bpm"),
        _chart(11, 10.5, 220045, 1e7, "bpm"),  # implausible
        _chart(11, 19.0, 220045, 95.0, "bpm"),  # at the cutoff: not in the window
        _chart(11, 10.0, 220210, 18.0, "insp/min"),
        _chart(11, 10.0, 224688, 12.0, "insp/min"),  # set rate: not mapped
        _chart(11, 10.0, 220277, 97.0, "%"),
        _chart(11, 11.0, 223761, 98.6, "°F"),
        _chart(11, 11 + 1 / 6, 223762, 37.5, "°C"),
        _chart(11, 12.0, 220050, 110.0, "mmHg"),
        _chart(11, 12 + 1 / 12, 220179, 120.0, "mmHg"),
        _chart(11, 13.0, 220179, 125.0, "mmHg"),
        _chart(11, 13.0, 225312, 70.0, "mmHg"),
        _chart(11, 14.0, 225664, 150.0, ""),  # POC finger stick, empty unit
        _chart(11, 14.0, 226537, 140.0, "mg/dL"),  # whole-blood copy: not mapped
        _chart(31, 10.0, 220045, 70.0, "bpm"),
    ]
    _csv_gz(raw / "icu" / "chartevents.csv.gz", chart, CHART_COLS)
    labs = [
        _lab(1, 14 + 1 / 3, 50931, 110.0, "mg/dL"),
        _lab(1, 15.0, 50813, 2.1, "mmol/L"),
        _lab(1, 15.0, 50912, 1.2, "mg/dL", hadm=10),
        _lab(1, 15.0, 50885, 0.8, "mg/dL"),
        _lab(1, 15.0, 51301, 12.0, "K/uL"),
        _lab(1, 15.0, 51265, 200.0, "K/uL"),
        _lab(1, 15.0, 50820, 7.30, "units", specimen=1001),
        _lab(1, 15.0, 52033, None, "", specimen=1001, text="ART."),
        _lab(1, 15.5, 50820, 7.36, "units", specimen=1002),
        _lab(1, 15.5, 52033, None, "", specimen=1002, text="VEN."),
        _lab(1, 15.0, 51082, 80.0, "mg/dL"),  # urine creatinine: not mapped
        _lab(4, 10.0, 50813, 1.0, "mmol/L"),  # stay 41
        _lab(4, 60.0, 50813, 2.0, "mmol/L"),  # between the two stays: dropped
        _lab(4, 120.0, 50813, 3.0, "mmol/L"),  # stay 42
        _lab(9, 10.0, 50813, 9.0, "mmol/L"),  # subject without ICU stay
    ]
    for i, row in enumerate(labs):
        row["labevent_id"] = i
    _csv_gz(raw / "hosp" / "labevents.csv.gz", labs, LAB_COLS)
    nan = float("nan")
    drugs = [
        _drug(11, 15.0, 15.02, 229617, nan, "Drug Push", "FinishedRunning"),  # 'Epinephrine.' push
        _drug(11, 16.0, 17.0, 221906, 0.0, "Continuous Med", "ChangeDose/Rate"),  # zero rate
        _drug(11, 18.0, 18.05, 221906, 0.5, "Continuous Med", "Bolus"),  # bolus from the drip
        _drug(11, 20.0, 21.0, 221906, 0.1, "Continuous Med", "Paused"),  # delivered, then paused
        _drug(11, 21.5, 30.0, 221906, 0.2, "Continuous Med", "ChangeDose/Rate"),
        _drug(21, 10.0, 20.0, 221653, 5.0, "Continuous Med", "FinishedRunning"),  # dobutamine
        _drug(21, 11.0, 11.02, 229617, nan, "Drug Push", "FinishedRunning"),
        _drug(31, 20.0, 25.0, 221906, 0.1, "Continuous Med", "FinishedRunning"),  # minor
        _drug(51, 20.0, 30.0, 229764, 20.0, "Continuous Med", "FinishedRunning"),  # angiotensin II
        _drug(11, 14.0, 16.0, 222168, 20.0, "Continuous Med", "Stopped"),  # propofol drip
        _drug(11, 9.0, 9.02, 222168, nan, "Drug Push", "FinishedRunning"),  # propofol push
        _drug(11, 10.0, 11.0, 225942, 50.0, "Continuous Med", "FinishedRunning"),  # fentanyl drip (opioid)
        _drug(11, 17.0, 17.5, 221385, 2.0, "Continuous Med", "Rewritten"),  # rewritten lorazepam
    ]
    _csv_gz(raw / "icu" / "inputevents.csv.gz", drugs, INPUT_COLS)
    outputs = [
        _out(11, 16.0, 226559, 100.0),
        _out(11, 16.5, 226560, 200.0),
        _out(11, 17.0, 226627, 300.0),  # OR urine: excluded
        _out(11, 17.0, 227489, 1000.0), _out(11, 17.0, 227488, 900.0),  # net 100
        _out(11, 18.0, 227489, 500.0), _out(11, 18.0, 227488, 800.0),  # negative net: dropped
        _out(11, 18.5, 227488, 400.0),  # unpaired: dropped
        _out(11, 18.75, 226559, -5.0),  # implausible
    ]
    _csv_gz(raw / "icu" / "outputevents.csv.gz", outputs, OUTPUT_COLS)
    procedures = [_proc(11, 12.0, 40.0, 225792), _proc(11, 8.0, 9.0, 225794), _proc(11, 1.0, 1.02, 224385)]
    _csv_gz(raw / "icu" / "procedureevents.csv.gz", procedures, PROC_COLS)
    return raw


@pytest.fixture()
def raw(tmp_path):
    return make_raw(tmp_path)


def adapter(raw: Path, cache: Path, **options) -> MimicIVAdapter:
    return MimicIVAdapter(raw, cache, verbose=False, **options)


def test_stays_are_adults_with_contract_dtypes(raw, tmp_path):
    stays = adapter(raw, tmp_path / "cache").stays()
    assert list(stays.columns) == ["subject_id", "stay_id", "intime", "outtime"]
    assert set(stays["stay_id"]) == {11, 21, 41, 42, 51}  # 31 is 16 years old
    assert stays["stay_id"].dtype == np.int64 and str(stays["intime"].dtype) == "datetime64[ns]"


def test_patients_are_read_from_the_zip_when_not_extracted(tmp_path):
    raw = make_raw(tmp_path, patients_in_zip=True)
    a = adapter(raw, tmp_path / "cache")
    assert not (raw / "hosp" / "patients.csv.gz").exists()
    assert set(a.stays()["stay_id"]) == {11, 21, 41, 42, 51}
    assert a.provenance()["items"]["chartevents"]["220045"]["label"] == "Heart Rate"


def test_vasopressor_intervals_exclude_boluses_zero_rates_and_inotropes(raw, tmp_path):
    a = adapter(raw, tmp_path / "cache")
    vaso = a.vasopressor_intervals()
    assert list(vaso.columns) == ["stay_id", "start", "end", "drug"]
    assert set(vaso["stay_id"]) == {11, 31, 51}  # 21 had only dobutamine and an 'Epinephrine.' push
    first = vaso.groupby("stay_id")["start"].min()
    assert first[11] == T0 + pd.Timedelta(hours=20)  # the Paused row counts; push, zero rate and drip bolus do not
    assert set(vaso.loc[vaso["stay_id"] == 51, "drug"]) == {"angiotensin_ii"}
    no_angiotensin = adapter(raw, tmp_path / "cache", include_angiotensin=False).vasopressor_intervals()
    assert 51 not in set(no_angiotensin["stay_id"])


def test_events_units_modality_and_plausibility(raw, tmp_path):
    ev = adapter(raw, tmp_path / "cache").events([11])
    assert list(ev.columns) == ["stay_id", "time", "feature", "value", "modality"]
    assert str(ev["time"].dtype) == "datetime64[ns]" and ev["feature"].dtype == object
    hr = ev[ev["feature"] == "HR"]
    assert sorted(hr["value"]) == [80.0, 95.0]  # 1e7 dropped
    assert sorted(ev.loc[ev["feature"] == "RR", "value"]) == [18.0]  # 224688 (set rate) is not RR
    temp = sorted(ev.loc[ev["feature"] == "Temp", "value"])
    assert temp == pytest.approx([37.0, 37.5])  # 98.6 degF -> 37 degC
    bp = ev[ev["feature"].isin(["SBP", "MAP"])]
    assert set(zip(bp["feature"], bp["value"], bp["modality"])) == {
        ("SBP", 110.0, "invasive"), ("SBP", 120.0, "noninvasive"), ("SBP", 125.0, "noninvasive"), ("MAP", 70.0, "invasive")}
    assert ev.loc[~ev["feature"].isin(["SBP", "DBP", "MAP"]), "modality"].isna().all()
    assert sorted(ev.loc[ev["feature"] == "Glucose", "value"]) == [110.0, 150.0]  # lab + POC; 226537 copy ignored
    no_poc = adapter(raw, tmp_path / "cache", poc_glucose=False).events([11])
    assert sorted(no_poc.loc[no_poc["feature"] == "Glucose", "value"]) == [110.0]
    labs = dict(zip(ev["feature"], ev["value"]))
    assert labs["Lactate"] == 2.1 and labs["Creatinine"] == 1.2 and labs["WBC"] == 12.0 and labs["Platelets"] == 200.0
    assert labs["Bilirubin_Total"] == 0.8
    assert 80.0 not in set(ev.loc[ev["feature"] == "Creatinine", "value"])  # urine creatinine not mapped
    assert set(ev["stay_id"]) == {11}


def test_urine_items_and_irrigation_netting(raw, tmp_path):
    a = adapter(raw, tmp_path / "cache")
    urine = a.events([11]).query("feature == 'urine_output'")
    got = sorted(zip(urine["time"], urine["value"]))
    assert got == [(T0 + pd.Timedelta(hours=16), 100.0), (T0 + pd.Timedelta(hours=16.5), 200.0), (T0 + pd.Timedelta(hours=17), 100.0)]
    counts = a.provenance()["event_counts"]["urine_output"]
    assert counts["irrigation_negative_net_dropped"] == 1 and counts["irrigation_unpaired_dropped"] == 1 and counts["implausible"] == 1


def test_labs_are_assigned_by_subject_and_icu_time(raw, tmp_path):
    ev = adapter(raw, tmp_path / "cache").events([41, 42])
    lactate = ev[ev["feature"] == "Lactate"]
    assert set(zip(lactate["stay_id"], lactate["value"])) == {(41, 1.0), (42, 3.0)}  # the 60 h draw lies between the stays


def test_ph_specimen_option(raw, tmp_path):
    pooled = adapter(raw, tmp_path / "cache").events([11])
    arterial = adapter(raw, tmp_path / "cache", ph_specimen="arterial").events([11])
    assert sorted(pooled.loc[pooled["feature"] == "pH", "value"]) == [7.30, 7.36]
    assert sorted(arterial.loc[arterial["feature"] == "pH", "value"]) == [7.30]
    with pytest.raises(ValueError):
        adapter(raw, tmp_path / "cache", ph_specimen="venous")


def test_interventions_defaults_and_options(raw, tmp_path):
    iv = adapter(raw, tmp_path / "cache").interventions([11, 21])
    assert list(iv.columns) == ["stay_id", "start", "end", "feature"]
    vent = iv[iv["feature"] == "ventilation_on"]
    assert list(zip(vent["start"], vent["end"])) == [(T0 + pd.Timedelta(hours=12), T0 + pd.Timedelta(hours=40))]  # NIV excluded
    sed = iv[iv["feature"] == "sedation_on"]
    assert list(sed["start"]) == [T0 + pd.Timedelta(hours=14)]  # push, opioid and rewritten rows excluded
    wide = adapter(raw, tmp_path / "cache", include_noninvasive_ventilation=True, include_opioids=True).interventions([11])
    assert sorted(wide.loc[wide["feature"] == "ventilation_on", "start"]) == [T0 + pd.Timedelta(hours=8), T0 + pd.Timedelta(hours=12)]
    assert sorted(wide.loc[wide["feature"] == "sedation_on", "start"]) == [T0 + pd.Timedelta(hours=10), T0 + pd.Timedelta(hours=14)]


def test_cache_is_reused_and_keyed_on_the_source_files(raw, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    manifest = adapter(raw, cache).build_cache()
    assert {"stays", "chartevents", "labevents", "inputevents", "outputevents", "procedureevents"} <= set(manifest)
    assert manifest["labevents"]["ph_specimen_types"] == {"ART.": 1, "VEN.": 1}

    def fail(self):
        raise AssertionError("cache should have been reused")

    monkeypatch.setattr(MimicIVAdapter, "_build_chartevents", fail)
    assert len(adapter(raw, cache).events([11])) > 0
    assert len(adapter(raw.resolve(), cache).events([11])) > 0  # the spelling of raw_dir is not part of the key
    rows = pd.read_csv(raw / "icu" / "chartevents.csv.gz")
    rows.loc[len(rows)] = rows.iloc[0]
    rows.to_csv(raw / "icu" / "chartevents.csv.gz", index=False, compression="gzip")  # source changed -> rebuild
    with pytest.raises(AssertionError, match="reused"):
        adapter(raw, cache).events([11])


def test_unknown_stays_give_empty_frames_with_the_contract_columns(raw, tmp_path):
    a = adapter(raw, tmp_path / "cache")
    ev, iv = a.events([999]), a.interventions([999])
    assert ev.empty and list(ev.columns) == ["stay_id", "time", "feature", "value", "modality"]
    assert str(ev["time"].dtype) == "datetime64[ns]"
    assert iv.empty and list(iv.columns) == ["stay_id", "start", "end", "feature"]
    assert a.available_features() == frozenset(CANONICAL_FEATURES)


def test_extract_cohort_end_to_end(raw, tmp_path):
    a = adapter(raw, tmp_path / "cache")
    # the recent-vital eligibility rule is tested in test_extract_common_v2rules.py
    spec = ExtractionSpec(window_hours=12, prediction_gap_hours=1, controls_per_positive=1, required_vital=None)
    manifest = extract_cohort(a, spec, tmp_path / "out")
    counts = manifest["counts"]
    assert counts["positives"] == 2 and counts["positives_written"] == 2 and counts["controls_written"] == 2
    assert manifest["unavailable_features"] == []
    json.dumps(manifest, default=str)
    index = json.loads((tmp_path / "out" / "shock_episode_index.json").read_text(encoding="utf-8"))
    pos = next(m for m in index if m["stay_id"] == 11)
    assert pos["window_end"] == (T0 + pd.Timedelta(hours=19)).isoformat()
    paths = episode_paths(tmp_path / "out", pos["episode_id"])
    values, mask = np.load(paths["values"]), np.load(paths["mask"])
    assert values.shape == (12, 17) and mask.shape == (12, 17)
    # bin i covers [7 + i, 8 + i) hours after admission
    assert values[3, COL["HR"]] == 80.0 and np.isnan(values[11, COL["HR"]])  # 95 at the cutoff is excluded
    assert values[4, COL["Temp"]] == pytest.approx(37.25)
    assert values[5, COL["SBP"]] == 110.0 and values[6, COL["SBP"]] == 125.0  # invasive preferred within a bin
    assert values[7, COL["Glucose"]] == pytest.approx(130.0)  # mean of POC 150 and lab 110
    assert values[9, COL["urine_output"]] == 300.0 and values[10, COL["urine_output"]] == 100.0
    assert mask[11, COL["urine_output"]]  # only dropped irrigation rows in the last bin
    assert values[:, COL["ventilation_on"]].tolist() == [0.0] * 5 + [1.0] * 7
    assert values[:, COL["sedation_on"]].tolist() == [0.0] * 7 + [1.0, 1.0] + [0.0] * 3
    assert values[8, COL["pH"]] == pytest.approx(7.33)
    controls = [m for m in index if m["episode_type"] == "control"]
    assert {m["stay_id"] for m in controls} <= {21, 41, 42}
    assert manifest["adapter"]["stay_counts"]["below_min_age"] == 1


def test_feature_events_equal_filtered_events_and_keep_their_counts_apart(raw, tmp_path):
    a = adapter(raw, tmp_path / "cache")
    ids = [11, 21, 41, 42]
    creat = a.feature_events(ids, ["Creatinine"])
    assert a.provenance()["event_counts"] == {}
    assert a.provenance()["feature_events_counts"]["Creatinine"]["rows"] == 1
    full = a.events(ids)
    pd.testing.assert_frame_equal(creat, full[full["feature"] == "Creatinine"].reset_index(drop=True))
    for features in (["HR", "SBP"], ["urine_output"], ["Glucose"]):
        got = a.feature_events(ids, features)
        pd.testing.assert_frame_equal(got, full[full["feature"].isin(features)].reset_index(drop=True))
    empty = a.feature_events(ids, ["no_such_feature"])
    assert empty.empty and list(empty.columns) == ["stay_id", "time", "feature", "value", "modality"]
