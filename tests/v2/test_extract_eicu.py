from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.icu.v2.extract.common import CANONICAL_FEATURES, ExtractionSpec, extract_cohort
from src.icu.v2.extract.eicu import (
    BASE_TIME,
    EICUAdapter,
    adult_mask,
    classify_infusion,
    classify_lab,
    derive_intervals,
    offsets_to_time,
    parse_rate,
    subject_id_from_uniquepid,
    urine_cellpath_decision,
)
from src.icu.v2.materialize import episode_paths

OUT = "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|"
MIN = pd.Timedelta(minutes=1)


def _write(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, compression="gzip")


def make_raw(root):
    """Tiny eICU-shaped raw directory (offsets in minutes from unit admission)."""
    root.mkdir(parents=True, exist_ok=True)
    patient_cols = ["patientunitstayid", "patienthealthsystemstayid", "gender", "age", "hospitalid", "unitdischargeoffset", "uniquepid"]
    patients = [
        (101, 1, "Male", "65", 1, 3000, "002-1"),      # positive (norepinephrine at 20 h)
        (102, 2, "Male", "66", 1, 2000, "002-1"),      # same patient, control candidate
        (103, 3, "Female", "> 89", 1, 4000, "002-2"),  # control, nurse-charted temperature
        (104, 4, "Female", "17", 1, 3000, "002-3"),    # minor
        (105, 5, "Male", None, 1, 3000, "002-4"),      # age missing
        (106, 6, "Male", "40", 1, 0, "002-5"),         # no ICU time
        (107, 7, "Male", "50", 2, 3000, "002-6"),      # hospital that never charts vasopressors
        (108, 8, "Male", "70", 1, 3000, "002-7"),      # vasopressor rows without a usable rate
        (109, 9, "Male", "55", 1, 3000, "002-8"),      # treatment says vasopressors, no infusion
        (110, 10, "Male", "60", 1, 3000, "002-9"),     # drip ordered long before the first charted rate
        (111, 11, "Female", "45", 1, 3000, "002-10"),
        (112, 12, "Female", "47", 1, 3000, "002-11"),
        (113, 13, "Male", "77", 1, 3000, "002-12"),    # positive whose run is capped at discharge
    ]
    _write(root / "patient.csv.gz", patients, patient_cols)
    inf_cols = ["infusiondrugid", "patientunitstayid", "infusionoffset", "drugname", "drugrate", "infusionrate", "drugamount", "volumeoffluid", "patientweight"]
    inf = [
        (1, 101, 1200, "Norepinephrine (mcg/min)", "5", "18", "4", "250", None),
        (2, 101, 1260, "Norepinephrine (mcg/min)", "7", "26", "4", "250", None),
        (3, 101, 1320, "Norepinephrine (mcg/min)", "0", "0", "4", "250", None),
        (4, 101, 1500, "Norepinephrine (mcg/min)", "3", "11", "4", "250", None),
        (5, 101, 1900, "Norepinephrine (mcg/min)", "4", "15", "4", "250", None),
        (6, 101, 600, "Propofol (mcg/kg/min)", "20", "12", None, None, None),
        (7, 101, 660, "Propofol (mcg/kg/min)", "OFF", None, None, None, None),
        (8, 101, 700, "Fentanyl (mcg/hr)", "50", "5", None, None, None),
        (9, 101, 800, "Lorazepam (mg/hr)", "2", "2", None, None, None),
        (10, 101, 900, "fentanyl epi (ml/hr)", "5", "5", None, None, None),
        (11, 103, 100, "Dobutamine (mcg/kg/min)", "5", "7", None, None, None),
        (12, 108, 500, "Norepinephrine (ml/hr)", "0", "0", None, None, None),
        (13, 108, 560, "Norepinephrine (ml/hr)", "ERROR", None, None, None, None),
        (14, 110, 2000, "Levophed (mcg/min)", "10", "37", None, None, None),
        (15, 113, 2900, "Vasopressin (units/min)", "0.04", "6", None, None, None),
    ]
    _write(root / "infusionDrug.csv.gz", inf, inf_cols)
    vp_cols = ["vitalperiodicid", "patientunitstayid", "observationoffset", "temperature", "sao2", "heartrate", "respiration", "cvp",
               "etco2", "systemicsystolic", "systemicdiastolic", "systemicmean", "pasystolic", "padiastolic", "pamean", "st1", "st2", "st3", "icp"]
    nan = None
    vp = [
        # deliberately not sorted by stay (the adapter must still produce a sorted store)
        (3, 103, 900, 37.0, 97, 90, 16, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan),
        (1, 101, 600, nan, 95, 80, 18, 12, nan, 110, 60, 70, 30, 15, 22, nan, nan, nan, nan),
        (2, 101, 605, nan, 96, 82, 19, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan),
        (4, 101, 1100, nan, nan, 400, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan),
        (5, 104, 100, nan, 99, 70, 12, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan),
    ]
    _write(root / "vitalPeriodic.csv.gz", vp, vp_cols)
    va_cols = ["vitalaperiodicid", "patientunitstayid", "observationoffset", "noninvasivesystolic", "noninvasivediastolic", "noninvasivemean",
               "paop", "cardiacoutput", "cardiacinput", "svr", "svri", "pvr", "pvri"]
    va = [
        (1, 101, 610, nan, nan, 75, nan, nan, nan, nan, nan, nan, nan),
        (2, 103, 615, 120, 70, nan, nan, nan, nan, nan, nan, nan, nan),
    ]
    _write(root / "vitalAperiodic.csv.gz", va, va_cols)
    lab_cols = ["labid", "patientunitstayid", "labresultoffset", "labtypeid", "labname", "labresult", "labresulttext", "labmeasurenamesystem",
                "labmeasurenameinterface", "labresultrevisedoffset"]
    lab = [
        (1, 101, 500, 1, "creatinine", 1.2, "1.2", "mg/dL", "mg/dL", 520),
        (2, 101, 500, 4, "urinary creatinine", 80.0, "80", "mg/dL", "mg/dL", 520),
        (3, 101, 510, 4, "glucose - CSF", 60.0, "60", "mg/dL", "mg/dL", 520),
        (4, 101, 520, 4, "bedside glucose", 140.0, "140", "mg/dL", "mg/dL", 525),
        (5, 101, 520, 1, "glucose", 130.0, "130", "mg/dL", "mg/dL", 600),
        (6, 101, 530, 7, "lactate", 2.0, "2.0", "mmol/L", "mmol/L", 600),
        (7, 101, 530, 7, "lactate", 3.0, "3.0", "mmol/L", "mmol/L", 700),
        (8, 101, 540, 7, "pH", 7.35, "7.35", None, None, 560),
        (9, 101, 540, 7, "pH", 735.0, "735", None, None, 560),
        (10, 101, 550, 3, "WBC x 1000", 12.0, "12", "K/mcL", "K/uL", 600),
        (11, 101, 550, 4, "WBC's in urine", 5.0, "5", None, None, 600),
        (12, 101, 560, 3, "platelets x 1000", 210.0, "210", "K/mcL", "K/uL", 600),
        (13, 101, 570, 1, "total bilirubin", 0.8, "0.8", "mg/dL", "mg/dL", 600),
        (14, 101, 580, 1, "direct bilirubin", 0.3, "0.3", "mg/dL", "mg/dL", 600),
    ]
    _write(root / "lab.csv.gz", lab, lab_cols)
    io_cols = ["intakeoutputid", "patientunitstayid", "intakeoutputoffset", "intaketotal", "outputtotal", "dialysistotal", "nettotal",
               "intakeoutputentryoffset", "cellpath", "celllabel", "cellvaluenumeric", "cellvaluetext"]
    io = [
        (1, 101, 700, 0, 5000, 0, 0, 700, OUT + "Urine", "Urine", 200.0, "200"),
        (2, 101, 700, 0, 5000, 0, 0, 700, OUT + "Stool", "Stool", 100.0, "100"),
        (3, 101, 760, 0, 5150, 0, 0, 790, OUT + "Indwelling Catheter Output", "Indwelling Catheter Output", 150.0, "150"),
        (4, 101, 770, 0, 5150, 0, 0, 790, OUT + "incontinent urine", "incontinent urine", 1.0, "1"),
        (5, 101, 780, 0, 5150, 0, 0, 790, OUT + "Urine", "Urine", -5.0, "-5"),
        (6, 101, 790, 300, 5150, 0, 0, 790, "flowsheet|Flowsheet Cell Labels|I&O|Intake (ml)|P.O.", "P.O.", 300.0, "300"),
    ]
    _write(root / "intakeOutput.csv.gz", io, io_cols)
    nc_cols = ["nursingchartid", "patientunitstayid", "nursingchartoffset", "nursingchartentryoffset", "nursingchartcelltypecat",
               "nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue"]
    nc = [
        (1, 103, 960, 960, "Vital Signs", "Temperature", "Temperature (C)", "37.5"),
        (2, 103, 960, 960, "Vital Signs", "Temperature", "Temperature (F)", "99.5"),
        (3, 103, 1020, 1020, "Vital Signs", "Temperature", "Temperature (C)", "98.6"),
        (4, 103, 1020, 1020, "Vital Signs", "Temperature", "Temperature Location", "Oral"),
        (5, 103, 1080, 1080, "Vital Signs", "Heart Rate", "Heart Rate", "88"),
    ]
    _write(root / "nurseCharting.csv.gz", nc, nc_cols)
    med_cols = ["medicationid", "patientunitstayid", "drugorderoffset", "drugstartoffset", "drugivadmixture", "drugordercancelled", "drugname",
                "drughiclseqno", "dosage", "routeadmin", "frequency", "loadingdose", "prn", "drugstopoffset", "gtc"]
    med = [
        (1, 110, 90, 100, "Yes", "No", "NOREPINEPHRINE 4 MG/250 ML NS", None, "4 mg", "IV", "Continuous", None, "No", 3000, 0),
        (2, 111, 90, 100, "No", "Yes", "NOREPINEPHRINE 4 MG/250 ML NS", None, "4 mg", "IV", "Continuous", None, "No", 3000, 0),
        (3, 112, 90, 100, "No", "No", "LIDOCAINE-EPINEPHRINE 1%-1:100000 INJ", None, "5 ml", "SubQ", "Once", None, "No", 100, 0),
    ]
    _write(root / "medication.csv.gz", med, med_cols)
    tr_cols = ["treatmentid", "patientunitstayid", "treatmentoffset", "treatmentstring", "activeupondischarge"]
    tr = [
        (1, 109, 300, "cardiovascular|shock|vasopressors|norepinephrine > 0.1 micrograms/kg/min", False),
        (2, 112, 300, "cardiovascular|hypertension|angiotensin II receptor blocker (ARB)|losartan", False),
    ]
    _write(root / "treatment.csv.gz", tr, tr_cols)
    return root


@pytest.fixture()
def raw(tmp_path):
    return make_raw(tmp_path / "raw")


def adapter_for(tmp_path, raw, **options):
    return EICUAdapter(raw, tmp_path / "cache", verbose=False, **options)


def test_offsets_map_to_a_shared_admission_origin(tmp_path, raw):
    adapter = adapter_for(tmp_path, raw)
    stays = adapter.stays().set_index("stay_id")
    assert (stays["intime"] == BASE_TIME).all()
    assert stays.loc[101, "outtime"] == BASE_TIME + 3000 * MIN
    # hospitalid is an extra column used to match controls within hospital
    assert list(adapter.stays().columns) == ["subject_id", "stay_id", "intime", "outtime", "hospitalid"]
    ev = adapter.events([101])
    hr = ev[(ev["feature"] == "HR")].sort_values("time")
    assert list(hr["time"]) == [BASE_TIME + 600 * MIN, BASE_TIME + 605 * MIN]
    assert offsets_to_time([90])[0] == np.datetime64("2000-01-01T01:30:00")


def test_subject_id_is_a_stable_hash_of_uniquepid(tmp_path, raw):
    expected = int(hashlib.sha1(b"002-34851").hexdigest()[:15], 16)
    assert subject_id_from_uniquepid("002-34851") == expected == subject_id_from_uniquepid(" 002-34851 ")
    assert 0 < expected < 2**63 - 1
    stays = adapter_for(tmp_path, raw).stays().set_index("stay_id")
    assert stays["subject_id"].dtype == np.int64
    assert stays.loc[101, "subject_id"] == stays.loc[102, "subject_id"] == subject_id_from_uniquepid("002-1")
    assert stays.loc[103, "subject_id"] != stays.loc[101, "subject_id"]
    assert stays.loc[101, "subject_id"] != 101  # not the stay id (v1 bug)


def test_adult_filter_and_positive_stay_length():
    ages = pd.Series(["18", "17", "> 89", None, "", "0", "89"])
    assert adult_mask(ages).tolist() == [True, False, True, False, False, False, True]


def test_cohort_drops_minors_missing_age_and_zero_length_stays(tmp_path, raw):
    adapter = adapter_for(tmp_path, raw, restrict_hospitals=False, exclude_ambiguous_controls=False, require_washout=False)
    ids = set(adapter.stays()["stay_id"])
    assert {101, 102, 103, 107, 108, 109, 110, 111, 112, 113} <= ids
    assert not ids & {104, 105, 106}


def test_cohort_restrictions_remove_ambiguous_vasopressor_exposure(tmp_path, raw):
    adapter = adapter_for(tmp_path, raw)
    ids = set(adapter.stays()["stay_id"])
    assert ids == {101, 102, 103, 111, 112, 113}
    excluded = adapter.provenance()["cohort"]["excluded"]
    assert excluded == {
        "hospital_without_vasopressor_charting": 1,
        "vasopressor_rows_without_rate": 1,
        "vasopressor_evidence_without_infusion": 1,
        "vasopressor_evidence_before_first_infusion": 1,
    }
    # 111's order was cancelled and 112's lidocaine-epinephrine is local anaesthesia: both stay controls
    vaso = adapter.vasopressor_intervals()
    assert set(vaso["stay_id"]) == {101, 113}
    no_washout = adapter_for(tmp_path, raw, require_washout=False)
    assert set(no_washout.stays()["stay_id"]) == ids | {110}
    assert set(no_washout.vasopressor_intervals()["stay_id"]) == {101, 110, 113}


def test_infusion_provenance_lists_deviations_from_harmonization(tmp_path, raw):
    adapter = adapter_for(tmp_path, raw)
    adapter.stays()
    stats = adapter.provenance()["build"]["infusions"]["stats"]
    assert stats["adapter_excluded_names"] == {"fentanyl epi (ml/hr)": 1, "Fentanyl (mcg/hr)": 1}
    assert stats["adapter_added_names"] == {"Lorazepam (mg/hr)": 1}
    assert stats["vasopressor_on"]["rate_text_read_as_stop"] == 0
    assert stats["sedation_on"]["rate_text_read_as_stop"] == 1  # propofol "OFF"


def test_lab_names_are_filtered_exactly():
    assert classify_lab("urinary creatinine")[0] is None
    assert classify_lab("glucose - CSF")[0] is None
    assert classify_lab("WBC's in urine")[0] is None
    assert classify_lab("creatinine")[0] == "Creatinine"
    assert classify_lab("bedside glucose")[0] == "Glucose"
    assert classify_lab("bedside glucose", include_bedside_glucose=False)[0] is None
    assert classify_lab("total bilirubin")[0] == "Bilirubin_Total"
    assert classify_lab("direct bilirubin")[0] is None


def test_lab_events_use_accepted_names_latest_revision_and_plausible_values(tmp_path, raw):
    ev = adapter_for(tmp_path, raw).events([101])
    by = {f: sorted(g["value"].round(6).tolist()) for f, g in ev.groupby("feature")}
    assert by["Creatinine"] == [1.2]  # urinary creatinine (80) rejected
    assert by["Glucose"] == [130.0, 140.0]  # lab + bedside, CSF glucose rejected
    assert by["Lactate"] == [3.0]  # superseded revision dropped
    assert by["pH"] == [7.35]  # 735 is implausible
    assert by["WBC"] == [12.0] and by["Platelets"] == [210.0] and by["Bilirubin_Total"] == [0.8]
    lab_times = ev[ev["feature"] == "Creatinine"]["time"].tolist()
    assert lab_times == [BASE_TIME + 500 * MIN]  # draw time, not the revision time


def test_rate_parsing():
    assert parse_rate("OFF") == 0.0 and parse_rate("off\\.br\\") == 0.0 and parse_rate("0ff") == 0.0
    assert parse_rate("12.5") == 12.5 and parse_rate("30\\.br\\") == 30.0 and parse_rate(".5") == 0.5
    for bad in ("ERROR", "UD", "-3", "50 mcg/min", "", None):
        assert np.isnan(parse_rate(bad))


def test_interval_derivation_rate_zero_max_gap_and_discharge():
    rows = pd.DataFrame(
        [
            (1, "a", 0, 5.0), (1, "a", 60, 0.0), (1, "a", 120, 2.0), (1, "a", 600, 3.0), (1, "a", 600, 0.0),
            (1, "b", 30, 0.0), (1, "b", 90, 1.0),
            (2, "a", 100, 1.0), (2, "a", 130, 1.0),
        ],
        columns=["stay_id", "key", "offset", "rate"],
    )
    iv = derive_intervals(rows, {1: 10_000, 2: 200}, max_gap_minutes=240)
    got = [tuple(r) for r in iv[["stay_id", "key", "start_offset", "end_offset", "end_reason"]].itertuples(index=False)]
    assert got == [
        (1, "a", 0.0, 60.0, "rate_zero"),
        (1, "b", 90.0, 330.0, "last_row"),
        (1, "a", 120.0, 360.0, "max_gap"),  # next row 480 min later > 4 h
        (1, "a", 600.0, 840.0, "last_row"),  # 0 and 3 at the same offset -> max = on
        (2, "a", 100.0, 200.0, "unit_discharge"),
    ]


def test_adapter_vasopressor_intervals(tmp_path, raw):
    vaso = adapter_for(tmp_path, raw).vasopressor_intervals()
    assert list(vaso.columns) == ["stay_id", "start", "end", "drug"]
    s101 = vaso[vaso["stay_id"] == 101].sort_values("start")
    assert [(int((a - BASE_TIME) / MIN), int((b - BASE_TIME) / MIN)) for a, b in zip(s101["start"], s101["end"])] == [
        (1200, 1320), (1500, 1740), (1900, 2140)
    ]
    assert set(s101["drug"]) == {"norepinephrine"}
    s113 = vaso[vaso["stay_id"] == 113]
    assert int((s113["end"].iat[0] - BASE_TIME) / MIN) == 3000  # capped at unit discharge


def test_drug_classification():
    assert classify_infusion("Levophed (mcg/min)")[:2] == ("vasopressor_on", "norepinephrine")
    assert classify_infusion("NSS w/ levo/vaso (ml/hr)")[:2] == ("vasopressor_on", "norepinephrine+vasopressin")
    assert classify_infusion("Neo-Synephrine (mcg/min)")[:2] == ("vasopressor_on", "phenylephrine")
    assert classify_infusion("EPI (mcg/min)")[:2] == ("vasopressor_on", "epinephrine")
    assert classify_infusion("Dobutamine (mcg/kg/min)")[0] is None
    assert classify_infusion("Milrinone (mcg/kg/min)")[0] is None
    assert classify_infusion("fentanyl epi (ml/hr)")[0] is None
    assert classify_infusion("Fentanyl/Bupivicaine PCEA (ml/hr)")[0] is None
    assert classify_infusion("Fentanyl (mcg/hr)")[0] is None  # not in the default sedation set
    assert classify_infusion("Fentanyl (mcg/hr)", sedation_drugs=("fentanyl",))[:2] == ("sedation_on", "fentanyl")
    assert classify_infusion("Lorazepam (mg/hr)")[:2] == ("sedation_on", "lorazepam")
    assert classify_infusion("Precedex (mcg/kg/hr)")[:2] == ("sedation_on", "dexmedetomidine")
    assert classify_infusion("Propofol (mcg/kg/min)")[:2] == ("sedation_on", "propofol")
    assert classify_infusion("Angiotensin II (Giapreza) (ng/kg/min)")[:2] == ("vasopressor_on", "angiotensin_ii")
    assert classify_infusion("angiotensin II receptor blocker")[0] is None


def test_blood_pressure_modality(tmp_path, raw):
    ev = adapter_for(tmp_path, raw).events([101, 103])
    bp = ev[ev["feature"].isin(["SBP", "DBP", "MAP"])]
    inv = bp[(bp["stay_id"] == 101) & (bp["feature"] == "MAP")].set_index("modality")["value"]
    assert inv.to_dict() == {"invasive": 70.0, "noninvasive": 75.0}
    assert set(bp.loc[bp["stay_id"] == 103, "modality"]) == {"noninvasive"}
    assert ev.loc[~ev["feature"].isin(["SBP", "DBP", "MAP"]), "modality"].isna().all()
    assert not (ev["value"] == 400).any()  # implausible HR dropped
    assert not ev["feature"].isin(["pamean", "cvp"]).any()


def test_urine_cellpaths():
    assert urine_cellpath_decision(OUT + "Urine")[0]
    assert urine_cellpath_decision(OUT + "Indwelling Catheter Output")[0]  # adapter addition
    assert urine_cellpath_decision(OUT + "Output (ml)-Nephrostomy Left")[0]
    assert not urine_cellpath_decision(OUT + "Stool")[0]
    assert not urine_cellpath_decision(OUT + "Urine Count")[0]
    assert not urine_cellpath_decision(OUT + "incontinent urine")[0]  # a count, mean 0.95
    assert not urine_cellpath_decision(OUT + "Ileostomy")[0]
    assert not urine_cellpath_decision("flowsheet|Flowsheet Cell Labels|I&O|Intake (ml)|P.O.")[0]


def test_urine_events_use_cell_values_not_output_total(tmp_path, raw):
    ev = adapter_for(tmp_path, raw).events([101])
    uo = ev[ev["feature"] == "urine_output"].sort_values("time")
    assert uo["value"].tolist() == [200.0, 150.0]  # stool, count label and negative value dropped
    assert not (ev["value"] >= 5000).any()


def test_temperature_from_monitor_and_nurse_charting(tmp_path, raw):
    ev = adapter_for(tmp_path, raw).events([103])
    temp = ev[ev["feature"] == "Temp"].sort_values("time")
    assert temp["value"].tolist() == [37.0, 37.5]  # Fahrenheit duplicate and 98.6 in the Celsius field dropped
    no_nurse = EICUAdapter(raw, tmp_path / "cache_no_nurse", verbose=False, nursecharting_temperature=False).events([103])
    assert no_nurse.loc[no_nurse["feature"] == "Temp", "value"].tolist() == [37.0]


def test_interventions_and_available_features(tmp_path, raw):
    adapter = adapter_for(tmp_path, raw)
    iv = adapter.interventions([101, 103])
    assert list(iv.columns) == ["stay_id", "start", "end", "feature"]
    assert set(iv["feature"]) == {"sedation_on"}
    spans = sorted((int((a - BASE_TIME) / MIN), int((b - BASE_TIME) / MIN)) for a, b in zip(iv["start"], iv["end"]))
    assert spans == [(600, 660), (800, 1040)]  # propofol stopped by "OFF"; lorazepam; fentanyl not counted
    available = adapter.available_features()
    assert "ventilation_on" not in available
    assert available == frozenset(CANONICAL_FEATURES) - {"ventilation_on"}
    prov = adapter.provenance()
    assert prov["unavailable_features"] == {"ventilation_on": "no reliable start/stop signal (R-E4, R-E5)"}
    assert "Fentanyl (mcg/hr)" not in prov["drug_names"]["sedation_on"]
    json.dumps(prov, default=str)


def test_events_contract_and_empty_requests(tmp_path, raw):
    adapter = adapter_for(tmp_path, raw)
    ev = adapter.events([101, 103, 999])
    assert list(ev.columns) == ["stay_id", "time", "feature", "value", "modality"]
    assert ev["stay_id"].dtype == np.int64 and str(ev["time"].dtype) == "datetime64[ns]"
    assert set(ev["feature"]) <= set(CANONICAL_FEATURES)
    assert adapter.events([]).empty and adapter.interventions([]).empty


def test_cache_is_reused_and_rebuilt_on_request(tmp_path, raw):
    adapter = adapter_for(tmp_path, raw)
    adapter.stays()
    adapter.events([101])
    manifest = json.loads((tmp_path / "cache" / "cache_manifest.json").read_text(encoding="utf-8"))
    assert {"base_stays", "infusions", "intervals", "evidence", "store_vital_periodic", "store_lab", "store_urine"} <= set(manifest)
    assert manifest["store_vital_periodic"]["stats"]["raw_sorted_by_stay"] is False
    again = adapter_for(tmp_path, raw)
    again.events([101])
    assert json.loads((tmp_path / "cache" / "cache_manifest.json").read_text(encoding="utf-8")) == manifest
    pd.testing.assert_frame_equal(again.stays(), adapter.stays())
    other_gap = adapter_for(tmp_path, raw, max_gap_hours=1.0)
    s101 = other_gap.vasopressor_intervals().query("stay_id == 101")
    assert int((s101["end"].max() - BASE_TIME) / MIN) == 1960  # 1900 + 1 h
    rebuilt = adapter_for(tmp_path, raw, rebuild=True)
    rebuilt.stays()
    after = json.loads((tmp_path / "cache" / "cache_manifest.json").read_text(encoding="utf-8"))
    assert after["base_stays"]["built_utc"] >= manifest["base_stays"]["built_utc"]


def test_extract_cohort_end_to_end(tmp_path, raw):
    adapter = adapter_for(tmp_path, raw)
    # the recent-vital eligibility rule is tested in test_extract_common_v2rules.py
    spec = ExtractionSpec(window_hours=12, prediction_gap_hours=1, required_vital=None)
    out = tmp_path / "episodes"
    manifest = extract_cohort(adapter, spec, out)
    counts = manifest["counts"]
    assert counts["positives"] == 2 and counts["positives_written"] == 2
    assert counts["controls_written"] == 2 and counts["control_shortfall"] == 0
    assert manifest["unavailable_features"] == ["ventilation_on"]
    metas = json.loads((out / "shock_episode_index.json").read_text(encoding="utf-8"))
    for meta in metas:
        paths = episode_paths(out, meta["episode_id"])
        values, mask = np.load(paths["values"]), np.load(paths["mask"])
        assert values.shape == (12, 17) and mask.shape == (12, 17)
        assert np.array_equal(np.isnan(values), mask)
        assert meta["subject_id"] != meta["stay_id"]
    pos = next(m for m in metas if m["stay_id"] == 101)
    assert pos["anchor_time"] == (BASE_TIME + 1200 * MIN).isoformat()
    values = np.load(episode_paths(out, pos["episode_id"])["values"])
    col = {f: j for j, f in enumerate(CANONICAL_FEATURES)}
    # window [07:00, 19:00): HR at 10:00/10:05, MAP invasive 70 wins over cuff 75 in that bin
    assert values[3, col["HR"]] == 81.0 and values[3, col["MAP"]] == 70.0
    assert values[0, col["sedation_on"]] == 0.0 and values[3, col["sedation_on"]] == 1.0
    # unavailable intervention features stay missing (not "observed: not ventilated")
    pos_mask = np.load(episode_paths(out, pos["episode_id"])["mask"])
    assert np.isnan(values[:, col["ventilation_on"]]).all() and pos_mask[:, col["ventilation_on"]].all()


def test_feature_events_equal_filtered_events(tmp_path, raw):
    adapter = adapter_for(tmp_path, raw)
    full = adapter.events([101, 102, 103])
    for features in (["Creatinine"], ["Creatinine", "Lactate"], ["HR", "Creatinine"]):
        got = adapter.feature_events([101, 102, 103], features)
        pd.testing.assert_frame_equal(got, full[full["feature"].isin(features)].reset_index(drop=True))
    assert adapter.feature_events([], ["Creatinine"]).empty
