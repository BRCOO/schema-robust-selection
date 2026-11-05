"""scripts/v2/export_mapping_table.py: the appendix tables are derived from the adapter code."""

from __future__ import annotations

import csv
import functools
import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

from src.icu.v2 import harmonization as hz
from src.icu.v2.extract import common, eicu, hirid, mimic_iv

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "v2" / "export_mapping_table.py"
SOURCES = ("mimic_iv", "eicu", "hirid")


def _load_script():
    spec = importlib.util.spec_from_file_location("export_mapping_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def module():
    return _load_script()


@pytest.fixture(scope="module")
def exported(module, tmp_path_factory):
    out = tmp_path_factory.mktemp("tables")
    assert module.main(["--out-dir", str(out)]) == 0
    return out, _read(out / "feature_mappings.csv"), _read(out / "task_rules.csv")


def _row(module, rows, feature: str, source: str) -> dict[str, str]:
    label = module.source_version(source)
    matches = [r for r in rows if r["canonical_feature"] == feature and r["source"] == label]
    assert len(matches) == 1, (feature, source)
    return matches[0]


def _ids(row: dict[str, str]) -> list[str]:
    return row["source_ids"].split(" | ") if row["source_ids"] else []


def test_writes_four_files_with_the_default_dir_as_option(module, exported):
    out, rows, rules = exported
    assert sorted(p.name for p in out.iterdir()) == ["feature_mappings.csv", "feature_mappings.tex", "task_rules.csv", "task_rules.tex"]
    assert module.build_parser().parse_args([]).out_dir == str(ROOT / "paper" / "v2_assets" / "tables")
    assert list(rows[0]) == list(module.FEATURE_COLUMNS)
    assert list(rules[0]) == list(module.RULE_COLUMNS)


def test_every_canonical_feature_appears_once_for_each_source(module, exported):
    _, rows, _ = exported
    pairs = [(r["canonical_feature"], r["source"]) for r in rows]
    assert len(pairs) == len(set(pairs)) == len(hz.CANONICAL_FEATURES) * len(SOURCES)
    for source in SOURCES:
        label = module.source_version(source)
        assert [f for f, s in pairs if s == label] == list(hz.CANONICAL_FEATURES)
    assert {r["source"] for r in rows} == {"MIMIC-IV 3.1", "eICU-CRD 2.0", "HiRID 1.1.1"}
    for r in rows:
        assert r["canonical_unit"] == hz.CANONICAL_SPECS[r["canonical_feature"]].unit


def test_eicu_ventilation_is_marked_unavailable_and_nothing_else_is(module, exported):
    out, rows, rules = exported
    vent = _row(module, rows, "ventilation_on", "eicu")
    assert vent["available"] == "no"
    assert vent["source_ids"] == "" and vent["source_tables"] == ""
    assert "not available in this source" in vent["notes"]
    assert [(r["canonical_feature"], r["source"]) for r in rows if r["available"] != "yes"] == [("ventilation_on", "eICU-CRD 2.0")]
    tex = (out / "feature_mappings.tex").read_text(encoding="utf-8")
    assert "not available in this source" in tex
    unavailable_rule = next(r for r in rules if r["rule"] == "Unavailable features")
    assert "eICU: ventilation_on" in unavailable_rule["setting"]


def test_vasopressor_ids_match_the_adapter_constants_exactly(module, exported):
    _, rows, _ = exported
    angiotensin = inspect.signature(mimic_iv.MimicIVAdapter.__init__).parameters["include_angiotensin"].default
    expected_mimic = list(mimic_iv.VASOPRESSOR_ITEMS) + (list(mimic_iv.ANGIOTENSIN_ITEMS) if angiotensin else [])
    assert _ids(_row(module, rows, "vasopressor_on", "mimic_iv")) == [str(i) for i in expected_mimic]
    assert _ids(_row(module, rows, "vasopressor_on", "eicu")) == list(eicu.VASOPRESSOR_DRUGS)
    assert _ids(_row(module, rows, "vasopressor_on", "hirid")) == [str(i) for i in hirid.VASOPRESSOR_ROWS]
    eicu_items = _row(module, rows, "vasopressor_on", "eicu")["source_items"]
    assert eicu.VASOPRESSOR_DRUGS["angiotensin_ii"] in eicu_items  # adapter rule pattern, not in the harmonization table
    for excluded in mimic_iv.EXCLUDED_DRUG_ITEMS:
        assert str(excluded) not in _ids(_row(module, rows, "vasopressor_on", "mimic_iv"))


def test_other_id_lists_match_the_adapters(module, exported):
    _, rows, _ = exported
    assert _ids(_row(module, rows, "sedation_on", "mimic_iv")) == [str(i) for i in mimic_iv.SEDATIVE_ITEMS]  # opioids off by default
    assert _ids(_row(module, rows, "sedation_on", "eicu")) == list(eicu.DEFAULT_SEDATION_DRUGS)
    adapter = hirid.HiRIDAdapter("unused", "unused")
    assert _ids(_row(module, rows, "sedation_on", "hirid")) == [str(i) for i in adapter.sedation_pharmaids()]
    assert _ids(_row(module, rows, "urine_output", "mimic_iv")) == [
        str(i) for i in (*mimic_iv.URINE_ITEMS, mimic_iv.GU_IRRIGANT_OUT, mimic_iv.GU_IRRIGANT_IN)
    ]
    for feature in common.NUMERIC_FEATURES:
        expected = [str(v) for v in adapter.feature_ids if hirid.OBS_FEATURE_ROWS[v].feature == feature]
        assert _ids(_row(module, rows, feature, "hirid")) == expected, feature
    for feature in common.MISSINGNESS_FEATURES:  # numeric features without urine_output: chartevents/labevents items
        mimic = [k for k, v in mimic_iv.LAB_ITEMS.items() if v[0] == feature] + [k for k, v in mimic_iv.CHART_ITEMS.items() if v[0] == feature]
        assert sorted(_ids(_row(module, rows, feature, "mimic_iv"))) == sorted(str(i) for i in mimic), feature
    rr = _ids(_row(module, rows, "RR", "hirid"))
    assert not set(rr) & {str(i) for i in hirid.DATA_EXCLUDED_OBS_IDS}
    glucose = _ids(_row(module, rows, "Glucose", "eicu"))
    assert glucose == ["glucose", "bedside glucose"]  # point-of-care glucose included by the adapter
    assert "225664" in _ids(_row(module, rows, "Glucose", "mimic_iv"))
    vitals = {**eicu._VITAL_PERIODIC, **eicu._VITAL_APERIODIC}
    for column, (feature, _) in vitals.items():
        assert any(i.endswith("." + column) for i in _ids(_row(module, rows, feature, "eicu")))


def test_unit_handling_comes_from_the_harmonization_conversions(module, exported):
    _, rows, _ = exported
    assert hz.conversion_for("Creatinine", "umol/l").name in _row(module, rows, "Creatinine", "hirid")["unit_handling"]
    assert hz.conversion_for("Glucose", "mmol/l").name in _row(module, rows, "Glucose", "hirid")["unit_handling"]
    assert hz.FAHRENHEIT_TO_CELSIUS.name in _row(module, rows, "Temp", "mimic_iv")["unit_handling"]
    urine = _row(module, rows, "urine_output", "hirid")["unit_handling"]
    hours = inspect.signature(hirid.HiRIDAdapter.__init__).parameters["urine_max_interval_hours"].default
    assert "rate" in urine and f"[0, {hours:g}] h" in urine
    for feature in ("SBP", "DBP", "MAP"):
        for source in SOURCES:
            assert "invasive-first" in _row(module, rows, feature, source)["notes"]
    for r in rows:
        spec = hz.CANONICAL_SPECS[r["canonical_feature"]]
        assert (r["plausible_range"] != "") == (spec.kind != "state" and r["available"] == "yes")


def test_tex_tables_are_escaped_booktabs_tabulars(module, exported):
    out, rows, rules = exported
    for name, n_cols, n_rows in (("feature_mappings", 6, len(rows)), ("task_rules", 3, len(rules))):
        lines = (out / f"{name}.tex").read_text(encoding="utf-8").splitlines()
        assert lines[0] == "% generated by scripts/v2/export_mapping_table.py -- do not edit by hand"
        assert lines[3].startswith("\\begin{tabular}") and lines[-1] == "\\end{tabular}"
        start, end = lines.index("\\midrule"), lines.index("\\bottomrule")
        assert lines[start - 2] == "\\toprule"
        body = [line for line in lines[start + 1 : end] if line != "\\addlinespace"]
        assert len(body) == n_rows
        for line in body:
            assert line.endswith(" \\\\")
            assert len(re.findall(r"(?<!\\)&", line)) == n_cols - 1, line
            bare = re.sub(r"\\[_%&#$]", "", line)
            assert not re.search(r"[_%#]", bare), line
            assert line.count("{") == line.count("}"), line


def test_tex_escape_and_reason_cleaning(module):
    assert module.tex_escape("a_b & 50% #1 {x}") == r"a\_b \& 50\% \#1 \{x\}"
    assert module.tex_escape("--gap-hours") == "-{}-gap-hours"
    assert module.tex_escape("10^9/L") == "10$^{9}$/L"
    assert module.tex_escape("degF->degC") == r"degF$\rightarrow$degC"
    assert module.clean_reason("blood-gas analyser glucose; pending team decision (lab vs point of care)") == "blood-gas analyser glucose"
    assert module.clean_reason("inotrope; v1 counted it as a vasopressor (eicu_dataset.py:33)") == "inotrope"


def test_task_rules_follow_extraction_spec_and_cli_defaults(module, exported):
    _, _, rules = exported
    spec = common.ExtractionSpec()
    by_rule = {r["rule"]: r for r in rules}
    assert f"{spec.window_hours} h" in by_rule["Observation window"]["setting"]
    assert f"{spec.n_bins} bins of {spec.resample_minutes} min" in by_rule["Observation window"]["setting"]
    assert f"anchor - {spec.prediction_gap_hours:g} h" in by_rule["Prediction gap"]["setting"]
    assert by_rule["Positives per patient"]["setting"].startswith(f"{spec.max_positives_per_patient}:")
    assert "hospitalid" in by_rule["eICU hospital matching"]["setting"]
    assert spec.required_vital in by_rule["Recent-vital rule"]["setting"]
    assert "landmark_control" in by_rule["Landmark matching"]["setting"]
    assert "initiation" in by_rule["Positive anchor"]["setting"]
    cli = module.cli_options()
    assert cli["--window-hours"]["default"] == spec.window_hours
    assert cli["--gap-hours"]["default"] == spec.prediction_gap_hours


def test_export_reads_no_data_and_builds_no_cache(module, tmp_path, monkeypatch):
    """Every adapter method that touches raw files or caches raises; the export still runs.
    functools.wraps keeps __wrapped__, so the signatures and sources the export parses stay readable."""

    def guard(cls: type, name: str) -> None:
        original = getattr(cls, name)

        @functools.wraps(original)
        def forbidden(*args, **kwargs):
            raise AssertionError(f"{cls.__name__}.{name} must not run during the export")

        monkeypatch.setattr(cls, name, forbidden)

    common_api = ("provenance", "stays", "vasopressor_intervals", "events", "interventions")
    for cls, names in (
        (mimic_iv.MimicIVAdapter, (*common_api, "_load", "_read_csv", "_read_manifest", "_labels", "build_cache")),
        (eicu.EICUAdapter, (*common_api, "__init__", "_ensure", "_base", "_cohort", "_store_parts", "available_features")),
        (hirid.HiRIDAdapter, (*common_api, "ensure_cache", "build_cache", "_read_manifest", "_derived", "_general", "available_features")),
    ):
        for name in names:
            guard(cls, name)
    with pytest.raises(AssertionError):  # the guards are active
        eicu.EICUAdapter(tmp_path / "raw", tmp_path / "cache")
    assert module.main(["--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "feature_mappings.csv").exists()
