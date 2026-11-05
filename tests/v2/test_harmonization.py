from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.icu.mimic_iv_schema import get_feature_names
from src.icu.v2.harmonization import (
    AUXILIARY_INDICATORS,
    BINARIZE,
    BP_FEATURES,
    CANONICAL_FEATURES,
    CANONICAL_SPECS,
    EICU_URINE_CELLPATH_PREFIX,
    HIRID_TABLE,
    MAPPING_TABLES,
    MIMIC_IV_TABLE,
    SOURCES,
    MappingRow,
    bp_modality,
    check_units,
    classify_eicu_drug,
    classify_eicu_lab,
    conversion_for,
    cross_namespace_identifiers,
    declared_units,
    feature_summary,
    from_canonical,
    implausible_mask,
    is_eicu_urine_cellpath,
    is_excluded_eicu_drug,
    is_excluded_eicu_lab,
    is_excluded_hirid_variable,
    is_excluded_mimic_item,
    lookup_identifier,
    map_eicu_lab,
    map_identifier,
    mapping_frame,
    merge_blood_pressure,
    normalize_unit,
    plausibility_report,
    to_canonical,
    to_canonical_frame,
    validate_mapping_tables,
)


def _hirid_row(feature: str, namespace: str, identifier: int, decision: str = "include", **kwargs) -> MappingRow:
    base = dict(source="hirid", feature=feature, namespace=namespace, identifier=identifier, match="id",
                decision=decision, label="test", note="test reason")
    base.update(kwargs)
    return MappingRow(**base)


# --- tables and validators ---------------------------------------------------------------


def test_canonical_features_match_v1_episode_order():
    assert CANONICAL_FEATURES == tuple(get_feature_names("core"))
    assert tuple(CANONICAL_SPECS) == CANONICAL_FEATURES


def test_builtin_tables_validate_clean_and_cover_every_feature():
    assert validate_mapping_tables() == []
    for source in SOURCES:
        included = {row.feature for row in MAPPING_TABLES[source] if row.decision == "include"}
        assert included == set(CANONICAL_FEATURES), source


def test_validator_catches_duplicate_identifier_within_namespace():
    duplicate = _hirid_row("HR", "observation", 200)
    problems = validate_mapping_tables({"hirid": HIRID_TABLE + (duplicate,)}, {"hirid": {}})
    assert any("duplicate identifier 200" in problem for problem in problems)


def test_validator_allows_same_id_in_two_tagged_namespaces_but_not_untagged():
    # The shipped HiRID table has 300 under observation (RR) and pharma (metronidazole): fine.
    assert cross_namespace_identifiers("hirid") == {300: ("observation", "pharma")}
    untagged = [row for row in HIRID_TABLE if row.identifier != 300 or row.namespace != "pharma"]
    untagged.append(_hirid_row("RR", "", 300, decision="exclude"))
    problems = validate_mapping_tables({"hirid": tuple(untagged)}, {"hirid": {}})
    assert any("missing or unknown namespace tag" in problem for problem in problems)
    assert any("identifier 300 is used in namespaces" in problem for problem in problems)


def test_validator_requires_reason_for_unavailable_feature_and_rejects_double_status():
    without_platelets = tuple(row for row in HIRID_TABLE if row.feature != "Platelets")
    problems = validate_mapping_tables({"hirid": without_platelets}, {"hirid": {}})
    assert any("Platelets has no include row" in problem for problem in problems)
    problems = validate_mapping_tables({"hirid": without_platelets}, {"hirid": {"Platelets": "  "}})
    assert any("marked unavailable without a reason" in problem for problem in problems)
    assert validate_mapping_tables({"hirid": without_platelets}, {"hirid": {"Platelets": "not recorded"}}) == []
    problems = validate_mapping_tables({"hirid": HIRID_TABLE}, {"hirid": {"HR": "test"}})
    assert any("HR is both mapped and marked unavailable" in problem for problem in problems)


def test_validator_flags_bad_units_regex_modality_and_missing_reason():
    bad_unit = replace(HIRID_TABLE[0], identifier=999999, feature="Platelets", unit="%")
    bad_regex = MappingRow("eicu", "Lactate", "lab", "lact(", "regex", "exclude", "x", note="reason")
    no_modality = _hirid_row("MAP", "observation", 999998, unit="mmHg")
    stray_modality = _hirid_row("HR", "observation", 999997, unit="/min", modality="invasive")
    no_reason = _hirid_row("HR", "observation", 999996, decision="exclude", note="")
    problems = validate_mapping_tables(
        {
            "hirid": HIRID_TABLE + (bad_unit, no_modality, stray_modality, no_reason),
            "eicu": MAPPING_TABLES["eicu"] + (bad_regex,),
        },
        {"hirid": {}, "eicu": {}},
    )
    assert any("cannot be converted" in problem for problem in problems)
    assert any("invalid regex" in problem for problem in problems)
    assert any("blood-pressure rows need modality" in problem for problem in problems)
    assert any("modality is only defined" in problem for problem in problems)
    assert any("exclusion without a reason" in problem for problem in problems)


def test_mapping_frame_and_feature_summary_are_reviewable():
    frame = mapping_frame("hirid")
    assert len(frame) == len(HIRID_TABLE)
    assert {"namespace", "identifier", "decision", "confidence", "v1", "modality"} <= set(frame.columns)
    summary = feature_summary("mimic_iv")
    assert summary["feature"].tolist() == list(CANONICAL_FEATURES)
    sbp = summary.set_index("feature").loc["SBP", "include"]
    assert "220050 [invasive]" in sbp and "220179 [noninvasive]" in sbp
    with pytest.raises(ValueError):
        mapping_frame("physionet")


# --- lookups and exclusion helpers -------------------------------------------------------


def test_lookup_requires_namespace_and_separates_hirid_300():
    with pytest.raises(ValueError):
        lookup_identifier("hirid", "", 300)
    with pytest.raises(ValueError):
        is_excluded_hirid_variable("pharmacy", 300)
    assert lookup_identifier("hirid", "observation", 300).feature == "RR"
    assert map_identifier("hirid", "observation", "300") == "RR"
    pharma_300 = lookup_identifier("hirid", "pharma", 300)
    assert pharma_300.decision == "exclude" and "Metronidazole" in pharma_300.label
    assert map_identifier("hirid", "pharma", 300) is None
    assert lookup_identifier("hirid", "observation", 123456789) is None


def test_hirid_exclusions_and_fixes():
    for variable_id in (24000572, 24000573, 24000400, 24000564, 24000608, 24000741, 4200, 20000800):
        assert is_excluded_hirid_variable("observation", variable_id), variable_id
    assert map_identifier("hirid", "observation", 20000110) == "Platelets"  # 'G/l' no longer quarantined
    assert map_identifier("hirid", "observation", 20000600) == "Creatinine"
    assert is_excluded_hirid_variable("pharma", 426)  # dobutamine
    assert is_excluded_hirid_variable("pharma", 1000843)  # "ohne Fentanyl" epidural
    assert map_identifier("hirid", "pharma", 1000491) == "sedation_on"  # Disoprivan (propofol)
    assert map_identifier("hirid", "pharma", 252) == "sedation_on"  # Dormicum Perfusor (midazolam)
    assert map_identifier("hirid", "observation", 3845) == "ventilation_on"
    assert not is_excluded_hirid_variable("observation", 200)


def test_mimic_exclusions_and_urine_items():
    assert is_excluded_mimic_item(226627) and is_excluded_mimic_item(226631)  # v1 OR/PACU urine items
    assert is_excluded_mimic_item(51516)  # urine WBC
    assert not is_excluded_mimic_item(51301)
    assert not is_excluded_mimic_item("not an itemid")
    assert map_identifier("mimic_iv", "outputevents", 226559) == "urine_output"  # Foley, missed by v1
    assert lookup_identifier("mimic_iv", "outputevents", 227488).sign == -1.0
    assert map_identifier("mimic_iv", "chartevents", 223835) is None  # FiO2 is not ventilation


def test_eicu_lab_exclusions_reject_non_serum_and_accept_serum_names():
    for name in (
        "urinary creatinine",
        "glucose - CSF",
        "WBC's in urine",
        "WBC's in body fluid",
        "WBC's in cerebrospinal fluid",
        "bedside glucose",
        "WBC's in pericardial fluid",  # not listed exactly: caught by the v1-rule guard
    ):
        assert is_excluded_eicu_lab(name), name
        assert map_eicu_lab(name) is None
    assert classify_eicu_lab("WBC's in pericardial fluid") == (None, "excluded: non-blood specimen")
    expected = {
        "creatinine": "Creatinine",
        "glucose": "Glucose",
        "lactate": "Lactate",
        "pH": "pH",
        "total bilirubin": "Bilirubin_Total",
        "WBC x 1000": "WBC",
        "platelets x 1000": "Platelets",
        "  Creatinine ": "Creatinine",  # case and whitespace insensitive
    }
    for name, feature in expected.items():
        assert classify_eicu_lab(name) == (feature, "accepted"), name
        assert not is_excluded_eicu_lab(name)
    for unrelated in ("sodium", None, float("nan"), ""):
        assert classify_eicu_lab(unrelated) == (None, "unmapped")
        assert not is_excluded_eicu_lab(unrelated)


def test_eicu_drug_classification():
    vasopressors = (
        "Norepinephrine (mcg/min)",
        "Levophed (mcg/kg/min)",
        "NSS w/ levo/vaso (ml/hr)",
        "EPINEPHrine(Adrenalin)MAX 30 mg Sodium Chloride 0.9% 250 ml (mcg/min)",
        "Neo-Synephrine (mcg/min)",
        "Vasopressin (units/min)",
        "Dopamine (mcg/kg/min)",
    )
    for name in vasopressors:
        assert classify_eicu_drug(name) == ("vasopressor_on", "accepted"), name
    for name in ("Propofol (mcg/kg/min)", "Precedex (mcg/kg/hr)", "Versed (mg/hr)", "Fentanyl (mcg/hr)"):
        assert classify_eicu_drug(name) == ("sedation_on", "accepted"), name
    for name in ("Dobutamine (mcg/kg/min)", "Milrinone (mcg/kg/min)", "Fentanyl/bupivacaine epidural (ml/hr)"):
        assert is_excluded_eicu_drug(name), name
    for name in ("Heparin (units/hr)", "Levofloxacin", None):
        assert classify_eicu_drug(name) == (None, "unmapped")
        assert not is_excluded_eicu_drug(name)


def test_eicu_urine_cellpath_rule():
    prefix = EICU_URINE_CELLPATH_PREFIX
    assert is_eicu_urine_cellpath(prefix + "Urine")
    assert is_eicu_urine_cellpath(prefix + "Foley")
    assert is_eicu_urine_cellpath(prefix + "Urine Output (mL)-Urethral Catheter")
    assert not is_eicu_urine_cellpath(prefix + "Foley PACU")
    assert not is_eicu_urine_cellpath(prefix + "incont. of urine count")
    assert not is_eicu_urine_cellpath(prefix + "urinary output 7AM-7PM")
    assert not is_eicu_urine_cellpath(prefix + "Chest Tube")
    assert not is_eicu_urine_cellpath("flowsheet|Flowsheet Cell Labels|I&O|Intake (ml)|Urine")
    assert not is_eicu_urine_cellpath(None)
    assert lookup_identifier("eicu", "intakeOutput", "outputtotal").decision == "exclude"


def test_bp_modality_lookup():
    assert bp_modality("mimic_iv", "chartevents", 220052) == "invasive"
    assert bp_modality("mimic_iv", "chartevents", 220181) == "noninvasive"
    assert bp_modality("eicu", "vitalPeriodic", "systemicmean") == "invasive"
    assert bp_modality("hirid", "observation", 610) == "noninvasive"
    assert bp_modality("mimic_iv", "chartevents", 220045) is None  # HR is not BP
    assert bp_modality("mimic_iv", "chartevents", 220056) is None  # excluded alarm item
    for source in SOURCES:
        rows = [row for row in MAPPING_TABLES[source] if row.feature in BP_FEATURES and row.decision == "include"]
        assert {row.modality for row in rows} == {"invasive", "noninvasive"}, source


# --- units -------------------------------------------------------------------------------


def test_normalize_unit_cases():
    assert normalize_unit("G/l") == "10^9/L"
    assert normalize_unit("g/l") == "g/L"  # grams per litre stays distinct
    assert normalize_unit("K/µL") == "10^9/L"
    assert normalize_unit("K/mcL") == "10^9/L"
    assert normalize_unit("°F") == "degF"
    assert normalize_unit(" mm Hg ") == "mmHg"
    assert normalize_unit("'/min") == "/min"
    assert normalize_unit(None) == "" and normalize_unit(float("nan")) == ""
    assert normalize_unit("mcg/kg/min") == "mcg/kg/min"
    for spec in CANONICAL_SPECS.values():
        for token in (spec.unit, *spec.accepted_units):
            assert normalize_unit(token) == token  # canonical tokens are fixed points


def _non_identity_conversions():
    for feature, spec in CANONICAL_SPECS.items():
        for unit, conversion in spec.accepted_units.items():
            if not conversion.is_identity:
                yield feature, unit


@pytest.mark.parametrize("feature,unit", list(_non_identity_conversions()))
def test_unit_conversion_round_trips(feature, unit):
    spec = CANONICAL_SPECS[feature]
    canonical = np.linspace(spec.low, spec.high, 7)
    source_values = from_canonical(feature, canonical, unit)
    assert np.allclose(to_canonical(feature, source_values, unit), canonical)


def test_unit_conversion_anchors_and_errors():
    assert to_canonical("Temp", 98.6, "°F") == pytest.approx(37.0)
    assert to_canonical("Creatinine", 88.4, "umol/l") == pytest.approx(1.0)
    assert to_canonical("Bilirubin_Total", 17.104, "µmol/l") == pytest.approx(1.0)
    assert to_canonical("Glucose", 5.0, "mmol/l") == pytest.approx(90.08)
    assert to_canonical("Lactate", 9.008, "mg/dL") == pytest.approx(1.0)
    assert to_canonical("Platelets", 250.0, "G/l") == pytest.approx(250.0)
    flags = to_canonical("vasopressor_on", [0.0, 0.05, np.nan], "mcg/kg/min")
    assert flags[0] == 0.0 and flags[1] == 1.0 and np.isnan(flags[2])
    assert conversion_for("sedation_on", "anything") is BINARIZE
    with pytest.raises(ValueError):
        from_canonical("vasopressor_on", 1.0, "mcg/kg/min")
    with pytest.raises(ValueError):
        to_canonical("Platelets", 250.0, "%")
    with pytest.raises(ValueError):
        conversion_for("Potassium", "mmol/L")


def test_declared_units_per_source():
    assert declared_units("hirid", "Creatinine") == ("umol/L",)
    assert declared_units("mimic_iv", "Temp") == ("degC", "degF")
    assert declared_units("eicu", "Platelets") == ("10^9/L",)
    with pytest.raises(ValueError):
        declared_units("physionet", "HR")


def test_check_units_statuses():
    frame = pd.DataFrame(
        {
            "feature": ["Platelets", "Platelets", "Creatinine", "pH", "WBC", "Temp", "Foo", "vasopressor_on"],
            "unit": ["G/l", "%", "umol/l", None, "g/l", "°F", "x", "mcg/kg/min"],
        }
    )
    report = check_units(frame, "hirid").set_index(["feature", "unit"])["status"]
    assert report[("Platelets", "G/l")] == "ok"
    assert report[("Platelets", "%")] == "rejected"
    assert report[("Creatinine", "umol/l")] == "convert"
    assert report[("pH", "")] == "ok"
    assert report[("WBC", "g/l")] == "rejected"  # grams per litre is not a cell count
    assert report[("Temp", "°F")] == "unexpected"  # HiRID only declares degC
    assert report[("Foo", "x")] == "unknown_feature"
    assert report[("vasopressor_on", "mcg/kg/min")] == "ok"
    mimic = check_units(pd.DataFrame({"feature": ["Creatinine"], "unit": ["umol/L"]}), "mimic_iv")
    assert mimic["status"].tolist() == ["unexpected"]
    empty = check_units(pd.DataFrame({"feature": [], "unit": []}), "eicu")
    assert empty.empty and "status" in empty.columns


# --- plausibility ------------------------------------------------------------------------


def test_plausibility_report_flags_out_of_range_values():
    frame = pd.DataFrame(
        {
            "feature": ["HR", "HR", "HR", "HR", "SpO2", "SpO2", "Temp", "Temp", "Temp"],
            "value": [80.0, 400.0, -5.0, np.nan, 97.0, 120.0, 37.0, 98.6, 98.6],
            "unit": [None, None, None, None, None, None, "°C", "°F", None],
        }
    )
    report = plausibility_report(frame, "mimic_iv").set_index("feature")
    assert report.loc["HR", "n_above"] == 1 and report.loc["HR", "n_below"] == 1
    assert report.loc["HR", "n_missing"] == 1 and report.loc["HR", "n_checked"] == 3
    assert report.loc["SpO2", "n_above"] == 1
    # 98.6 degF converts to 37 degC; 98.6 without a unit is read as degC and flagged
    assert report.loc["Temp", "n_above"] == 1 and report.loc["Temp", "status"] == "flagged"
    mask = implausible_mask(frame, "mimic_iv")
    assert mask.tolist() == [False, True, True, False, False, True, False, False, True]


def test_plausibility_uses_declared_source_unit_and_reports_unknown_features():
    frame = pd.DataFrame({"feature": ["Creatinine", "Foo"], "value": [90.0, 1.0]})
    hirid = plausibility_report(frame, "hirid").set_index("feature")
    assert hirid.loc["Creatinine", "status"] == "ok"  # 90 umol/L = 1.02 mg/dL
    assert hirid.loc["Creatinine", "max"] == pytest.approx(90.0 / 88.4)
    assert hirid.loc["Foo", "status"] == "unknown_feature"
    mimic = plausibility_report(frame.iloc[:1], "mimic_iv").set_index("feature")
    assert mimic.loc["Creatinine", "status"] == "flagged"  # 90 mg/dL is implausible
    rejected = plausibility_report(pd.DataFrame({"feature": ["WBC"], "value": [5.0], "unit": ["%"]}), "eicu")
    assert rejected.loc[0, "n_unit_rejected"] == 1 and rejected.loc[0, "status"] == "no_values"


def test_to_canonical_frame_resolves_units_from_identifiers():
    frame = pd.DataFrame(
        {
            "feature": ["Temp", "Temp"],
            "value": [98.6, 37.0],
            "namespace": ["chartevents", "chartevents"],
            "identifier": [223761, 223762],
        }
    )
    out = to_canonical_frame(frame, "mimic_iv")
    assert out["unit_resolved"].tolist() == ["degF", "degC"]
    assert np.allclose(out["value_canonical"], [37.0, 37.0])
    with pytest.raises(ValueError):
        to_canonical_frame(pd.DataFrame({"feature": ["HR"]}), "mimic_iv")


# --- blood pressure merge ----------------------------------------------------------------


def test_merge_blood_pressure_prefers_invasive_and_falls_back():
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 2, 2],
            "bin": [0, 0, 0, 1, 1, 0, 0],
            "feature": ["MAP", "MAP", "MAP", "MAP", "MAP", "SBP", "HR"],
            "namespace": ["chartevents"] * 7,
            "identifier": [220052, 220181, 220181, 220052, 220181, 220179, 220045],
            "value": [70.0, 60.0, 62.0, np.nan, 65.0, 110.0, 80.0],
        }
    )
    merged = merge_blood_pressure(frame, "mimic_iv").set_index(["stay_id", "bin", "feature"])
    assert merged.loc[(1, 0, "MAP"), "value"] == 70.0
    assert merged.loc[(1, 0, "MAP"), "bp_source"] == "invasive"
    assert merged.loc[(1, 1, "MAP"), "bp_source"] == "noninvasive"  # NaN arterial value ignored
    assert merged.loc[(1, 1, "MAP"), "value"] == 65.0
    assert merged.loc[(2, 0, "SBP"), "bp_source"] == "noninvasive"
    assert len(merged) == 3  # HR rows are ignored
    assert "bp_source" in AUXILIARY_INDICATORS


def test_merge_blood_pressure_with_modality_column_and_unresolved_rows():
    frame = pd.DataFrame(
        {"stay_id": [7, 7], "bin": [3, 3], "feature": ["DBP", "DBP"], "value": [40.0, 50.0],
         "modality": ["noninvasive", "noninvasive"]}
    )
    merged = merge_blood_pressure(frame, "eicu")
    assert merged["value"].tolist() == [45.0] and merged["n_noninvasive"].tolist() == [2]
    unknown = pd.DataFrame(
        {"stay_id": [1], "bin": [0], "feature": ["MAP"], "namespace": ["observation"], "identifier": [160], "value": [20.0]}
    )
    with pytest.raises(ValueError):
        merge_blood_pressure(unknown, "hirid")  # 160 is pulmonary artery pressure (excluded)
    with pytest.raises(ValueError):
        merge_blood_pressure(frame.drop(columns="modality"), "eicu")
