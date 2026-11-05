"""Explicit cross-ICU feature harmonization tables for the v2 protocol.

The v1 loaders map the 18 episode features with three unrelated mechanisms: MIMIC-IV by
itemid (``src/icu/mimic_iv_schema.py``), eICU by substring keywords
(``src/icu/eicu_dataset.py``) and HiRID by name aliases plus inference from variable
descriptions (``src/icu/hirid_dataset.py``). For v2 this module replaces them with one
reviewable table per source:

* every :class:`MappingRow` names an exact identifier (MIMIC itemid; eICU column, exact
  ``labname``, ``drugname`` pattern or intake/output rule; HiRID ``variableid`` /
  ``pharmaid``) with a namespace tag, the unit the source documents, a confidence level and
  its status relative to v1;
* exclusion rows document identifiers v1 mapped wrongly or that look alike (urine
  creatinine, CSF glucose, mixed-venous saturation, dobutamine, boluses, ...);
* :data:`CANONICAL_SPECS` fixes the canonical unit, the accepted source units with their
  conversions and a plausible value range per feature.

Data and checks only. Nothing here is wired into the v1 loaders. Rationale, confirmed v1
defects and the raw-data checks that are still pending are in
``docs/reviews/v2_harmonization_spec.md``.

STATUS (2026-10-21): the raw-data checks have been run, and the AUTHORITATIVE mappings used
for extraction are now the source adapters ``src/icu/v2/extract/{mimic_iv,eicu,hirid}.py``
(each documents its decisions with evidence counts). Where they differ from the tables
below, the adapters win; the differences are listed in section 9 of
``docs/reviews/v2_harmonization_spec.md`` (point-of-care glucose included, sedation without
opioids by default, MIMIC ventilation from procedureevents, HiRID RR id 300 only and urine as
rate x interval, eICU ventilation unavailable, and others).
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------------------

#: Feature order of the v1 episode matrices (``get_feature_names("core")``).
CANONICAL_FEATURES: tuple[str, ...] = (
    "HR",
    "RR",
    "SpO2",
    "SBP",
    "DBP",
    "MAP",
    "Temp",
    "ventilation_on",
    "Lactate",
    "pH",
    "Creatinine",
    "Glucose",
    "Bilirubin_Total",
    "WBC",
    "Platelets",
    "vasopressor_on",
    "sedation_on",
    "urine_output",
)

SOURCES: tuple[str, ...] = ("mimic_iv", "eicu", "hirid")

#: Allowed namespace tags per source. An identifier is only meaningful together with its tag:
#: HiRID ``variableid`` 300 (respiratory rate) and ``pharmaid`` 300 (metronidazole) collide.
NAMESPACES: dict[str, tuple[str, ...]] = {
    "mimic_iv": ("chartevents", "labevents", "inputevents", "outputevents"),
    "eicu": (
        "vitalPeriodic",
        "vitalAperiodic",
        "lab",
        "infusionDrug",
        "intakeOutput",
        "respiratoryCare",
        "respiratoryCharting",
        "nurseCharting",
    ),
    "hirid": ("observation", "pharma"),
}

#: ``id``: integer itemid/variableid; ``column``: a column of the source table; ``exact``:
#: exact string (case-insensitive, whitespace collapsed); ``regex``: case-insensitive
#: pattern; ``rule``: a named rule implemented in this module (see :data:`RULES`).
MATCH_KINDS: tuple[str, ...] = ("id", "column", "exact", "regex", "rule")
DECISIONS: tuple[str, ...] = ("include", "exclude")
CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low")

#: Status relative to the v1 code at HEAD. ``kept``: v1 maps it the same way; ``fixed``: v1
#: meant to map it but mishandled it; ``added``: new in v2; ``removed``: v1 maps it, v2
#: excludes it; ``never``: neither maps it (documents a trap or a pending candidate).
V1_STATUSES: tuple[str, ...] = ("kept", "fixed", "added", "removed", "never")

FEATURE_KINDS: tuple[str, ...] = ("measurement", "state", "volume")


# ---------------------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitConversion:
    """Linear map ``canonical = value * factor + offset``; ``binarize`` maps value > 0 to 1."""

    name: str
    factor: float = 1.0
    offset: float = 0.0
    binarize: bool = False

    @property
    def is_identity(self) -> bool:
        """True only for the plain identity (a 1:1 map that rests on an assumption is not)."""
        return self.name == "identity" and not self.binarize and self.factor == 1.0 and self.offset == 0.0

    def to_canonical(self, values: object) -> float | np.ndarray:
        """Convert source-unit values to the canonical unit (NaN stays NaN)."""
        arr = np.asarray(values, dtype=float)
        if self.binarize:
            out = np.where(np.isnan(arr), np.nan, (arr > 0).astype(float))
        else:
            out = arr * self.factor + self.offset
        return float(out) if np.ndim(out) == 0 else out

    def from_canonical(self, values: object) -> float | np.ndarray:
        """Inverse of :meth:`to_canonical`; binarized flags cannot be inverted."""
        if self.binarize:
            raise ValueError(f"{self.name} is not invertible")
        arr = np.asarray(values, dtype=float)
        out = (arr - self.offset) / self.factor
        return float(out) if np.ndim(out) == 0 else out


IDENTITY = UnitConversion("identity")
FAHRENHEIT_TO_CELSIUS = UnitConversion("degF->degC", factor=5.0 / 9.0, offset=-32.0 * 5.0 / 9.0)
CREATININE_UMOL_TO_MGDL = UnitConversion("umol/L->mg/dL (creatinine, /88.4)", factor=1.0 / 88.4)
BILIRUBIN_UMOL_TO_MGDL = UnitConversion("umol/L->mg/dL (bilirubin, /17.104)", factor=1.0 / 17.104)
GLUCOSE_MMOL_TO_MGDL = UnitConversion("mmol/L->mg/dL (glucose, x18.016)", factor=18.016)
LACTATE_MGDL_TO_MMOL = UnitConversion("mg/dL->mmol/L (lactate, /9.008)", factor=1.0 / 9.008)
#: HiRID 10020000 is an hourly urine volume in ml/h; one record per hour means value = mL.
HOURLY_URINE_TO_ML = UnitConversion("mL/h->mL (one record per hour)")
BINARIZE = UnitConversion("binarize(value > 0)", binarize=True)

# Case-sensitive tokens first: HiRID writes Giga-per-litre as "G/l"; lower-case "g/l" is
# grams per litre (fibrinogen, haemoglobin). v1 lower-cased everything and quarantined
# HiRID Platelets because "g/l" was not in its hint list (hirid_dataset.py:107-108).
_UNIT_CASE_SENSITIVE: dict[str, str] = {
    "G/l": "10^9/L",
    "G/L": "10^9/L",
    "Gpt/l": "10^9/L",
    "Gpt/L": "10^9/L",
}
_UNIT_ALIASES: dict[str, str] = {
    "": "",
    "bpm": "bpm",
    "beats/min": "bpm",
    "/min": "/min",
    "1/min": "/min",
    "min-1": "/min",
    "insp/min": "insp/min",
    "breaths/min": "insp/min",
    "%": "%",
    "percent": "%",
    "mmhg": "mmHg",
    "torr": "mmHg",
    "°c": "degC",
    "degc": "degC",
    "c": "degC",
    "celsius": "degC",
    "°f": "degF",
    "degf": "degF",
    "f": "degF",
    "fahrenheit": "degF",
    "mmol/l": "mmol/L",
    "mg/dl": "mg/dL",
    "umol/l": "umol/L",
    "micromol/l": "umol/L",
    "10^9/l": "10^9/L",
    "x10^9/l": "10^9/L",
    "10*9/l": "10^9/L",
    "10e9/l": "10^9/L",
    "k/ul": "10^9/L",
    "k/mcl": "10^9/L",
    "10^3/ul": "10^9/L",
    "x10^3/ul": "10^9/L",
    "10*3/ul": "10^9/L",
    "thou/ul": "10^9/L",
    "g/l": "g/L",
    "g/dl": "g/dL",
    "ml": "mL",
    "cc": "mL",
    "ml/h": "mL/h",
    "ml/hr": "mL/h",
    "ml/hour": "mL/h",
    "ph": "pH",
    "units": "units",
    "unit": "units",
    "flag": "flag",
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_unit(unit: object) -> str:
    """Map a raw unit string to a canonical unit token.

    Matching ignores case, spaces and micro-sign variants, except for ``G/l`` (Giga per
    litre), which must stay distinct from grams per litre. Unknown units are returned
    lower-cased so they show up verbatim in :func:`check_units`.
    """
    if _is_missing(unit):
        return ""
    text = str(unit).strip()
    if text in _UNIT_CASE_SENSITIVE:
        return _UNIT_CASE_SENSITIVE[text]
    key = re.sub(r"\s+", "", text).replace("µ", "u").replace("μ", "u").replace("º", "°")
    key = key.lstrip("'").lower()
    return _UNIT_ALIASES.get(key, key)


# ---------------------------------------------------------------------------------------
# Canonical features
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalFeature:
    """Canonical definition of one feature. ``low``/``high`` bound plausible values (inclusive)."""

    name: str
    kind: str
    unit: str
    low: float
    high: float
    accepted_units: Mapping[str, UnitConversion] = field(default_factory=dict)
    definition: str = ""


def _measurement(name: str, unit: str, low: float, high: float, definition: str, **extra: UnitConversion) -> CanonicalFeature:
    accepted = {unit: IDENTITY}
    accepted.update({_token(key): conv for key, conv in extra.items()})
    return CanonicalFeature(name, "measurement", unit, low, high, accepted, definition)


def _token(keyword: str) -> str:
    return {
        "per_min": "/min",
        "degF": "degF",
        "umol_L": "umol/L",
        "mmol_L": "mmol/L",
        "mg_dL": "mg/dL",
        "empty": "",
        "units": "units",
    }[keyword]


def _state(name: str, definition: str) -> CanonicalFeature:
    return CanonicalFeature(name, "state", "flag", 0.0, 1.0, {}, definition)


# Plausible ranges are artefact screens, not clinical thresholds. Sources: mimic-code
# measurement/vitalsign.sql and chemistry.sql, eicu-code pivoted-vital.sql / pivoted-lab.sql
# / pivoted-bg.sql (see the spec for the per-feature choice).
CANONICAL_SPECS: dict[str, CanonicalFeature] = {
    "HR": _measurement("HR", "bpm", 1.0, 300.0, "Heart rate from the bedside monitor or nurse charting.", per_min=IDENTITY),
    "RR": _measurement(
        "RR", "insp/min", 1.0, 70.0, "Measured (total) respiratory rate; set ventilator rates excluded.", per_min=IDENTITY
    ),
    "SpO2": _measurement(
        "SpO2", "%", 1.0, 100.0, "Peripheral (pulse-oximetry) saturation; blood-gas SaO2, SvO2 and ScvO2 excluded."
    ),
    "SBP": _measurement("SBP", "mmHg", 1.0, 400.0, "Systemic systolic pressure; invasive preferred per bin (BP_MERGE_RULE)."),
    "DBP": _measurement("DBP", "mmHg", 1.0, 300.0, "Systemic diastolic pressure; invasive preferred per bin (BP_MERGE_RULE)."),
    "MAP": _measurement(
        "MAP", "mmHg", 1.0, 300.0, "Systemic mean arterial pressure; invasive preferred per bin (BP_MERGE_RULE); PA and CVP excluded."
    ),
    "Temp": _measurement(
        "Temp", "degC", 25.0, 46.0, "Body temperature (core, rectal, oral, axillary pooled); device temperatures excluded.",
        degF=FAHRENHEIT_TO_CELSIUS,
    ),
    "ventilation_on": _state(
        "ventilation_on", "1 while positive-pressure ventilation (invasive or NIV) is documented; HFNC and conventional O2 are 0."
    ),
    "Lactate": _measurement("Lactate", "mmol/L", 0.1, 30.0, "Blood lactate (arterial or venous).", mg_dL=LACTATE_MGDL_TO_MMOL),
    "pH": _measurement("pH", "pH", 6.5, 8.5, "Blood pH from blood-gas analysis.", empty=IDENTITY, units=IDENTITY),
    "Creatinine": _measurement(
        "Creatinine", "mg/dL", 0.1, 30.0, "Serum/plasma creatinine from the laboratory.", umol_L=CREATININE_UMOL_TO_MGDL
    ),
    "Glucose": _measurement(
        "Glucose", "mg/dL", 10.0, 1500.0, "Serum/plasma glucose from the central laboratory; point-of-care glucose excluded.",
        mmol_L=GLUCOSE_MMOL_TO_MGDL,
    ),
    "Bilirubin_Total": _measurement(
        "Bilirubin_Total", "mg/dL", 0.1, 70.0, "Total bilirubin in serum/plasma.", umol_L=BILIRUBIN_UMOL_TO_MGDL
    ),
    "WBC": _measurement("WBC", "10^9/L", 0.1, 500.0, "White blood cell count in blood."),
    "Platelets": _measurement("Platelets", "10^9/L", 1.0, 2000.0, "Platelet count in blood."),
    "vasopressor_on": _state(
        "vasopressor_on",
        "1 while a continuous infusion of norepinephrine, epinephrine, vasopressin, phenylephrine or dopamine runs at "
        "rate > 0; inotropes (dobutamine, milrinone, levosimendan) and boluses excluded.",
    ),
    "sedation_on": _state(
        "sedation_on",
        "1 while a continuous infusion of propofol, midazolam, dexmedetomidine or fentanyl runs at rate > 0; boluses, "
        "PCA/epidural, patches, oral forms and other drugs excluded.",
    ),
    "urine_output": CanonicalFeature(
        "urine_output",
        "volume",
        "mL",
        0.0,
        5000.0,
        {"mL": IDENTITY, "mL/h": HOURLY_URINE_TO_ML},
        "Urine volume per record (summed per bin); totals, drains and irrigation inflow handled explicitly.",
    ),
}


def conversion_for(feature: str, unit: object) -> UnitConversion | None:
    """Conversion from ``unit`` to the canonical unit of ``feature`` (``None`` if not accepted)."""
    spec = CANONICAL_SPECS.get(feature)
    if spec is None:
        raise ValueError(f"unknown canonical feature: {feature!r}")
    if spec.kind == "state":
        return BINARIZE
    return spec.accepted_units.get(normalize_unit(unit))


def to_canonical(feature: str, values: object, unit: object) -> float | np.ndarray:
    """Convert ``values`` recorded in ``unit`` to the canonical unit of ``feature``."""
    conversion = conversion_for(feature, unit)
    if conversion is None:
        raise ValueError(f"unit {unit!r} is not accepted for {feature}")
    return conversion.to_canonical(values)


def from_canonical(feature: str, values: object, unit: object) -> float | np.ndarray:
    """Convert canonical ``values`` of ``feature`` back to ``unit``."""
    conversion = conversion_for(feature, unit)
    if conversion is None:
        raise ValueError(f"unit {unit!r} is not accepted for {feature}")
    return conversion.from_canonical(values)


# ---------------------------------------------------------------------------------------
# Mapping rows
# ---------------------------------------------------------------------------------------


#: Blood-pressure features and the two measurement modalities they are recorded with.
BP_FEATURES: tuple[str, ...] = ("SBP", "DBP", "MAP")
BP_MODALITIES: tuple[str, ...] = ("invasive", "noninvasive")

#: How invasive and non-invasive values of one BP feature are combined in one time bin.
BP_MERGE_RULE = (
    "per (stay, bin, feature): mean of the invasive (arterial line) values when the bin has at least one, "
    "otherwise mean of the non-invasive (cuff) values; the modality used is recorded in bp_source"
)

#: Columns produced next to the features that are NOT model inputs by default. ``bp_source``
#: says which modality filled a BP bin; arterial-line presence tracks severity, so it is kept
#: for audits and stratified analyses only.
AUXILIARY_INDICATORS: tuple[str, ...] = ("bp_source",)


@dataclass(frozen=True)
class MappingRow:
    """One identifier of one source and what v2 does with it.

    ``note`` carries the reason for exclusion rows and caveats for include rows. ``sign`` is
    -1 for inflow items that are subtracted (MIMIC 227488 GU irrigant in). ``modality`` is
    ``invasive`` or ``noninvasive`` for blood-pressure rows and empty otherwise.
    """

    source: str
    feature: str
    namespace: str
    identifier: int | str
    match: str
    decision: str
    label: str
    unit: str = ""
    confidence: str = "high"
    v1: str = "kept"
    note: str = ""
    sign: float = 1.0
    modality: str = ""


def _row(
    source: str,
    feature: str,
    namespace: str,
    identifier: int | str,
    label: str,
    unit: str = "",
    confidence: str = "high",
    v1: str = "kept",
    note: str = "",
    *,
    match: str = "id",
    decision: str = "include",
    sign: float = 1.0,
    modality: str = "",
) -> MappingRow:
    return MappingRow(
        source, feature, namespace, identifier, match, decision, label, unit, confidence, v1, note, sign, modality
    )


def _mimic(feature: str, table: str, itemid: int, label: str, unit: str = "", confidence: str = "high",
           v1: str = "kept", note: str = "", sign: float = 1.0, modality: str = "") -> MappingRow:
    return _row("mimic_iv", feature, table, itemid, label, unit, confidence, v1, note, sign=sign, modality=modality)


def _mimic_x(feature: str, table: str, itemid: int, label: str, reason: str, v1: str = "never",
             confidence: str = "high") -> MappingRow:
    return _row("mimic_iv", feature, table, itemid, label, "", confidence, v1, reason, decision="exclude")


def _eicu(feature: str, table: str, identifier: str, match: str, label: str, unit: str = "",
          confidence: str = "high", v1: str = "kept", note: str = "", modality: str = "") -> MappingRow:
    return _row("eicu", feature, table, identifier, label, unit, confidence, v1, note, match=match, modality=modality)


def _eicu_x(feature: str, table: str, identifier: str, match: str, label: str, reason: str, v1: str = "never",
            confidence: str = "high") -> MappingRow:
    return _row("eicu", feature, table, identifier, label, "", confidence, v1, reason, match=match, decision="exclude")


def _hirid(feature: str, namespace: str, variable_id: int, label: str, unit: str = "", confidence: str = "high",
           v1: str = "kept", note: str = "", modality: str = "") -> MappingRow:
    return _row("hirid", feature, namespace, variable_id, label, unit, confidence, v1, note, modality=modality)


def _hirid_x(feature: str, namespace: str, variable_id: int, label: str, reason: str, v1: str = "never",
             confidence: str = "high") -> MappingRow:
    return _row("hirid", feature, namespace, variable_id, label, "", confidence, v1, reason, decision="exclude")


_CHART_COPY = "nurse-charted copy of a lab result; v2 reads labevents only to avoid duplicates"
_PENDING = "second itemid for the same analyte; pending raw row-count/unit check before inclusion"
_NONBLOOD = "non-blood specimen"
_OUTSIDE_SET = "outside the canonical drug set"

# ---------------------------------------------------------------------------------------
# MIMIC-IV 3.1 (labels verified against icu/d_items and hosp/d_labitems)
# ---------------------------------------------------------------------------------------

#: Values of 223849 that mean a ventilator is in use (mimic-code treatment/ventilation.sql).
MIMIC_VENT_MODES_ON: tuple[str, ...] = (
    "(S) CMV", "APRV", "APRV/Biphasic+ApnPress", "APRV/Biphasic+ApnVol", "APV (cmv)", "Ambient",
    "Apnea Ventilation", "CMV", "CMV/ASSIST", "CMV/ASSIST/AutoFlow", "CMV/AutoFlow", "CPAP/PPS",
    "CPAP/PSV+Apn TCPL", "CPAP/PSV+ApnPres", "CPAP/PSV+ApnVol", "MMV", "MMV/AutoFlow", "MMV/PSV",
    "MMV/PSV/AutoFlow", "P-CMV", "PCV+", "PCV+/PSV", "PCV+Assist", "PRES/AC", "PRVC/AC", "PRVC/SIMV",
    "PSV/SBT", "SIMV", "SIMV/AutoFlow", "SIMV/PRES", "SIMV/PSV", "SIMV/PSV/AutoFlow", "SIMV/VOL",
    "SYNCHRON MASTER", "SYNCHRON SLAVE", "VOL/AC",
)
#: Values of 229314 (Hamilton): invasive modes plus the NIV modes DuoPaP, NIV, NIV-ST.
MIMIC_HAMILTON_MODES_ON: tuple[str, ...] = (
    "APRV", "APV (cmv)", "Ambient", "(S) CMV", "P-CMV", "SIMV", "APV (simv)", "P-SIMV", "VS", "ASV",
    "DuoPaP", "NIV", "NIV-ST",
)
#: Values of 226732 that imply positive-pressure ventilation (trailing spaces are in the data).
MIMIC_O2_DEVICES_ON: tuple[str, ...] = ("Endotracheal tube", "Bipap mask ", "CPAP mask ")

MIMIC_IV_TABLE: tuple[MappingRow, ...] = (
    # HR / RR / SpO2
    _mimic("HR", "chartevents", 220045, "Heart Rate", "bpm"),
    _mimic("RR", "chartevents", 220210, "Respiratory Rate", "insp/min"),
    _mimic("RR", "chartevents", 224690, "Respiratory Rate (Total)", "insp/min", "medium", "added",
           "ventilator-measured total rate; part of mimic-code vitalsign resp_rate"),
    _mimic_x("RR", "chartevents", 224688, "Respiratory Rate (Set)", "set ventilator rate, not a measurement"),
    _mimic_x("RR", "chartevents", 224689, "Respiratory Rate (spontaneous)", "spontaneous component only"),
    _mimic("SpO2", "chartevents", 220277, "O2 saturation pulseoxymetry", "%"),
    _mimic_x("SpO2", "chartevents", 220227, "Arterial O2 Saturation", "blood-gas SaO2, not pulse oximetry"),
    _mimic_x("SpO2", "chartevents", 223772, "SvO2", "mixed venous saturation"),
    _mimic_x("SpO2", "chartevents", 223769, "O2 Saturation Pulseoxymetry Alarm - High", "alarm threshold"),
    _mimic_x("SpO2", "chartevents", 223770, "O2 Saturation Pulseoxymetry Alarm - Low", "alarm threshold"),
    _mimic_x("SpO2", "chartevents", 226253, "SpO2 Desat Limit", "alarm threshold"),
    _mimic_x("SpO2", "labevents", 50817, "Oxygen Saturation (Blood Gas)", "blood-gas saturation (arterial or venous)"),
    # Blood pressure: v1 used the non-invasive items only (mimic_iv_schema.py:21-23); v2 keeps
    # both modalities explicitly and combines them with BP_MERGE_RULE
    _mimic("SBP", "chartevents", 220179, "Non Invasive Blood Pressure systolic", "mmHg", modality="noninvasive"),
    _mimic("SBP", "chartevents", 220050, "Arterial Blood Pressure systolic", "mmHg", "high", "added",
           "arterial line; v1 dropped it, so arterial-line patients looked like missing BP", modality="invasive"),
    _mimic("SBP", "chartevents", 225309, "ART BP Systolic", "mmHg", "medium", "added", "second arterial line; check counts",
           modality="invasive"),
    _mimic_x("SBP", "chartevents", 220056, "Arterial Blood Pressure Alarm - Low", "alarm threshold"),
    _mimic_x("SBP", "chartevents", 220058, "Arterial Blood Pressure Alarm - High", "alarm threshold"),
    _mimic("DBP", "chartevents", 220180, "Non Invasive Blood Pressure diastolic", "mmHg", modality="noninvasive"),
    _mimic("DBP", "chartevents", 220051, "Arterial Blood Pressure diastolic", "mmHg", "high", "added",
           "arterial line", modality="invasive"),
    _mimic("DBP", "chartevents", 225310, "ART BP Diastolic", "mmHg", "medium", "added", "second arterial line; check counts",
           modality="invasive"),
    _mimic("MAP", "chartevents", 220181, "Non Invasive Blood Pressure mean", "mmHg", modality="noninvasive"),
    _mimic("MAP", "chartevents", 220052, "Arterial Blood Pressure mean", "mmHg", "high", "added", "arterial line",
           modality="invasive"),
    _mimic("MAP", "chartevents", 225312, "ART BP Mean", "mmHg", "medium", "added", "in mimic-code mbp",
           modality="invasive"),
    # Temperature
    _mimic("Temp", "chartevents", 223762, "Temperature Celsius", "°C"),
    _mimic("Temp", "chartevents", 223761, "Temperature Fahrenheit", "°F", note="degF->degC, the only unit conversion in v1"),
    _mimic_x("Temp", "chartevents", 224642, "Temperature Site", "text site descriptor"),
    _mimic_x("Temp", "chartevents", 226329, "Blood Temperature CCO (C)",
             "PA-catheter blood temperature for a small subgroup; excluded as in mimic-code vitalsign"),
    _mimic_x("Temp", "chartevents", 229236, "Cerebral Temperature (C)", "brain probe"),
    _mimic_x("Temp", "chartevents", 228242, "Pt. Temperature (BG) (SOFT)", "temperature entered for blood-gas correction"),
    # Ventilation
    _mimic("ventilation_on", "chartevents", 223849, "Ventilator Mode", "", "medium", "fixed",
           "Text item: v1 listed it but dropped every row (no valuenum, mimic_iv_io.py:117-118); on if value in "
           "MIMIC_VENT_MODES_ON"),
    _mimic("ventilation_on", "chartevents", 229314, "Ventilator Mode (Hamilton)", "", "medium", "added",
           "Text; on if value in MIMIC_HAMILTON_MODES_ON"),
    _mimic("ventilation_on", "chartevents", 226732, "O2 Delivery Device(s)", "", "medium", "added",
           "Text; on only for MIMIC_O2_DEVICES_ON (ETT, BiPAP/CPAP mask)"),
    _mimic("ventilation_on", "chartevents", 220339, "PEEP set", "cmH2O", "medium", "kept",
           "a set PEEP implies a ventilator (invasive or NIV)"),
    _mimic_x("ventilation_on", "chartevents", 223835, "Inspired O2 Fraction",
             "FiO2 is also charted for HFNC and conventional O2 therapy", v1="removed"),
    _mimic_x("ventilation_on", "chartevents", 223848, "Ventilator Type",
             "device type, not an in-use mode (v1 listed it but dropped it: Text item)", v1="removed"),
    # Lactate / pH
    _mimic("Lactate", "labevents", 50813, "Lactate (Blood, Blood Gas)", "mmol/L"),
    _mimic_x("Lactate", "labevents", 52442, "Lactate (Blood, Blood Gas)", _PENDING, confidence="medium"),
    _mimic_x("Lactate", "labevents", 53154, "Lactate (Blood, Chemistry)", _PENDING, confidence="medium"),
    _mimic_x("Lactate", "labevents", 50954, "Lactate Dehydrogenase (LD)", "different analyte (LDH)"),
    _mimic("pH", "labevents", 50820, "pH (Blood, Blood Gas)", "units", note="arterial and venous specimens pooled (52033)"),
    _mimic_x("pH", "labevents", 50831, "pH (Other Body Fluid)", _NONBLOOD),
    _mimic_x("pH", "labevents", 52041, "pH (Fluid)", _NONBLOOD),
    _mimic_x("pH", "labevents", 51491, "pH (Urine, Hematology)", "urine"),
    _mimic_x("pH", "labevents", 51094, "pH (Urine, Chemistry)", "urine"),
    _mimic_x("pH", "labevents", 52730, "pH (Urine, Chemistry)", "urine"),
    _mimic_x("pH", "labevents", 52045, "pH, Urine (Blood Gas)", "urine"),
    # Creatinine
    _mimic("Creatinine", "labevents", 50912, "Creatinine (Blood, Chemistry)", "mg/dL"),
    _mimic_x("Creatinine", "labevents", 52546, "Creatinine (Blood, Chemistry)", _PENDING, confidence="medium"),
    _mimic_x("Creatinine", "labevents", 52024, "Creatinine, Whole Blood (Blood Gas)",
             "point-of-care analyser; pending team decision", confidence="medium"),
    _mimic_x("Creatinine", "labevents", 51082, "Creatinine, Urine", "urine"),
    _mimic_x("Creatinine", "labevents", 51081, "Creatinine, Serum (fluid=Urine)", "clearance panel item filed under urine"),
    _mimic_x("Creatinine", "labevents", 51977, "Creatinine, Blood (fluid=Urine)", "clearance panel item filed under urine"),
    _mimic_x("Creatinine", "labevents", 51080, "Creatinine Clearance", "derived clearance, not a concentration"),
    _mimic_x("Creatinine", "labevents", 50841, "Creatinine, Ascites", _NONBLOOD),
    _mimic_x("Creatinine", "labevents", 51032, "Creatinine, Body Fluid", _NONBLOOD),
    _mimic_x("Creatinine", "labevents", 51787, "Creatinine, CSF", _NONBLOOD),
    _mimic_x("Creatinine", "labevents", 51021, "Creatinine, Joint Fluid", _NONBLOOD),
    _mimic_x("Creatinine", "labevents", 51052, "Creatinine, Pleural", _NONBLOOD),
    _mimic_x("Creatinine", "labevents", 51937, "Creatinine, Stool", _NONBLOOD),
    _mimic_x("Creatinine", "chartevents", 220615, "Creatinine (serum)", _CHART_COPY),
    _mimic_x("Creatinine", "chartevents", 229761, "Creatinine (whole blood)", _CHART_COPY),
    # Glucose
    _mimic("Glucose", "labevents", 50931, "Glucose (Blood, Chemistry)", "mg/dL"),
    _mimic_x("Glucose", "labevents", 50809, "Glucose (Blood, Blood Gas)",
             "blood-gas analyser glucose; pending team decision (lab vs point of care)", confidence="medium"),
    _mimic_x("Glucose", "labevents", 52569, "Glucose (Blood, Chemistry)", _PENDING, confidence="medium"),
    _mimic_x("Glucose", "labevents", 52027, "Glucose, Whole Blood (Blood Gas)", "point of care; pending team decision",
             confidence="medium"),
    _mimic_x("Glucose", "labevents", 51478, "Glucose (Urine, Hematology)", "urine"),
    _mimic_x("Glucose", "labevents", 51981, "Glucose (Urine, Chemistry)", "urine"),
    _mimic_x("Glucose", "labevents", 51084, "Glucose, Urine", "urine"),
    _mimic_x("Glucose", "labevents", 51790, "Glucose, CSF", _NONBLOOD),
    _mimic_x("Glucose", "labevents", 50842, "Glucose, Ascites", _NONBLOOD),
    _mimic_x("Glucose", "labevents", 51034, "Glucose, Body Fluid", _NONBLOOD),
    _mimic_x("Glucose", "labevents", 51022, "Glucose, Joint Fluid", _NONBLOOD),
    _mimic_x("Glucose", "labevents", 51053, "Glucose, Pleural", _NONBLOOD),
    _mimic_x("Glucose", "labevents", 51941, "Glucose, Stool", _NONBLOOD),
    _mimic_x("Glucose", "chartevents", 225664, "Glucose finger stick (range 70-100)",
             "point-of-care capillary glucose (same decision as eICU 'bedside glucose')"),
    _mimic_x("Glucose", "chartevents", 220621, "Glucose (serum)", _CHART_COPY),
    _mimic_x("Glucose", "chartevents", 226537, "Glucose (whole blood)", "nurse-charted whole-blood/POC glucose"),
    # Bilirubin
    _mimic("Bilirubin_Total", "labevents", 50885, "Bilirubin, Total (Blood)", "mg/dL"),
    _mimic_x("Bilirubin_Total", "labevents", 53089, "Bilirubin, Total (Blood)", _PENDING, confidence="medium"),
    _mimic_x("Bilirubin_Total", "labevents", 50883, "Bilirubin, Direct", "different analyte"),
    _mimic_x("Bilirubin_Total", "labevents", 50838, "Bilirubin, Total, Ascites", _NONBLOOD),
    _mimic_x("Bilirubin_Total", "labevents", 51028, "Bilirubin, Total, Body Fluid", _NONBLOOD),
    _mimic_x("Bilirubin_Total", "labevents", 51783, "Bilirubin, Total, CSF", _NONBLOOD),
    _mimic_x("Bilirubin_Total", "labevents", 51812, "Bilirubin, Total, Joint Fluid", _NONBLOOD),
    _mimic_x("Bilirubin_Total", "labevents", 51049, "Bilirubin, Total, Pleural", _NONBLOOD),
    _mimic_x("Bilirubin_Total", "labevents", 51932, "Bilirubin, Total, Stool", _NONBLOOD),
    _mimic_x("Bilirubin_Total", "chartevents", 225690, "Total Bilirubin", _CHART_COPY),
    # WBC / Platelets
    _mimic("WBC", "labevents", 51301, "White Blood Cells (Blood, Hematology)", "K/uL"),
    _mimic("WBC", "labevents", 51300, "WBC Count (Blood, Hematology)", "K/uL"),
    _mimic_x("WBC", "labevents", 51755, "White Blood Cells (Blood, Chemistry)", _PENDING, confidence="medium"),
    _mimic_x("WBC", "labevents", 51756, "White Blood Cells (Blood, Chemistry)", _PENDING, confidence="medium"),
    _mimic_x("WBC", "labevents", 51516, "WBC (Urine)", "urine"),
    _mimic_x("WBC", "labevents", 51517, "WBC Casts (Urine)", "urine sediment"),
    _mimic_x("WBC", "labevents", 51518, "WBC Clumps (Urine)", "urine sediment"),
    _mimic_x("WBC", "labevents", 52407, "WBC (Stool)", _NONBLOOD),
    _mimic_x("WBC", "chartevents", 220546, "WBC", _CHART_COPY),
    _mimic("Platelets", "labevents", 51265, "Platelet Count (Blood, Hematology)", "K/uL"),
    _mimic_x("Platelets", "labevents", 53189, "Platelet Count (Blood, Chemistry)", _PENDING, confidence="medium"),
    _mimic_x("Platelets", "chartevents", 227457, "Platelet Count", _CHART_COPY),
    _mimic_x("Platelets", "chartevents", 225678, "ZPlatelet Count", "retired nurse-charted copy"),
    _mimic_x("Platelets", "inputevents", 225170, "Platelets", "transfused blood product, not a count"),
    _mimic_x("Platelets", "inputevents", 226369, "OR Platelet Intake", "transfused blood product"),
    _mimic_x("Platelets", "inputevents", 227071, "PACU Platelet Intake", "transfused blood product"),
    # Vasopressors (same five itemids as the v1 anchors, shock_anchor_builder.py:8-14)
    _mimic("vasopressor_on", "inputevents", 221906, "Norepinephrine", "mcg/kg/min"),
    _mimic("vasopressor_on", "inputevents", 221289, "Epinephrine", "mcg/kg/min"),
    _mimic("vasopressor_on", "inputevents", 221662, "Dopamine", "mcg/kg/min"),
    _mimic("vasopressor_on", "inputevents", 221749, "Phenylephrine", "mcg/kg/min"),
    _mimic("vasopressor_on", "inputevents", 222315, "Vasopressin", "units/hour"),
    _mimic("vasopressor_on", "inputevents", 229630, "Phenylephrine (50/250)", "", "medium", "added",
           "concentration-specific item new in MIMIC-IV 3.x; verify ordercategorydescription/rateuom"),
    _mimic("vasopressor_on", "inputevents", 229632, "Phenylephrine (200/250)", "", "medium", "added",
           "concentration-specific item new in MIMIC-IV 3.x; verify ordercategorydescription/rateuom"),
    _mimic("vasopressor_on", "inputevents", 229631, "Phenylephrine (200/250)_OLD_1", "", "low", "added",
           "retired item; verify row counts"),
    _mimic_x("vasopressor_on", "inputevents", 229617, "Epinephrine.",
             "possibly a bolus/code item; pending raw check of ordercategorydescription", confidence="medium"),
    _mimic_x("vasopressor_on", "inputevents", 221653, "Dobutamine", "inotrope, not a vasopressor"),
    _mimic_x("vasopressor_on", "inputevents", 221986, "Milrinone", "inodilator"),
    _mimic_x("vasopressor_on", "chartevents", 229789, "Phenylephrine (Intubation)", "intubation bolus"),
    # Sedation
    _mimic("sedation_on", "inputevents", 222168, "Propofol"),
    _mimic("sedation_on", "inputevents", 221668, "Midazolam (Versed)"),
    _mimic("sedation_on", "inputevents", 221744, "Fentanyl"),
    _mimic("sedation_on", "inputevents", 225942, "Fentanyl (Concentrate)", "", "medium", "added"),
    _mimic("sedation_on", "inputevents", 225150, "Dexmedetomidine (Precedex)", "", "high", "added",
           "eICU v1 already counted dexmedetomidine"),
    _mimic("sedation_on", "inputevents", 229420, "Dexmedetomidine (Precedex)", "", "medium", "added",
           "second dexmedetomidine itemid"),
    _mimic_x("sedation_on", "inputevents", 225972, "Fentanyl (Push)", "bolus"),
    _mimic_x("sedation_on", "chartevents", 227210, "Propofol (Intubation)", "intubation bolus"),
    _mimic_x("sedation_on", "inputevents", 221385, "Lorazepam (Ativan)", _OUTSIDE_SET),
    _mimic_x("sedation_on", "inputevents", 221833, "Hydromorphone (Dilaudid)", _OUTSIDE_SET),
    _mimic_x("sedation_on", "inputevents", 225154, "Morphine Sulfate", _OUTSIDE_SET),
    # Urine output (mimic-code measurement/urine_output.sql)
    _mimic("urine_output", "outputevents", 226559, "Foley", "mL", "high", "added", "v1 missed it (mimic_iv_schema.py:35)"),
    _mimic("urine_output", "outputevents", 226560, "Void", "mL", "high", "added", "v1 missed it"),
    _mimic("urine_output", "outputevents", 226561, "Condom Cath", "mL", "high", "added"),
    _mimic("urine_output", "outputevents", 226584, "Ileoconduit", "mL", "high", "added"),
    _mimic("urine_output", "outputevents", 226563, "Suprapubic", "mL", "high", "added"),
    _mimic("urine_output", "outputevents", 226564, "R Nephrostomy", "mL", "high", "added"),
    _mimic("urine_output", "outputevents", 226565, "L Nephrostomy", "mL", "high", "added"),
    _mimic("urine_output", "outputevents", 226567, "Straight Cath", "mL", "high", "added"),
    _mimic("urine_output", "outputevents", 226557, "R Ureteral Stent", "mL", "high", "added"),
    _mimic("urine_output", "outputevents", 226558, "L Ureteral Stent", "mL", "high", "added"),
    _mimic("urine_output", "outputevents", 227489, "GU Irrigant/Urine Volume Out", "mL", "high", "kept",
           "net of 227488 (GU irrigant in)"),
    _mimic("urine_output", "outputevents", 227488, "GU Irrigant Volume In", "mL", "high", "added",
           "subtracted, as in mimic-code urine_output", sign=-1.0),
    _mimic_x("urine_output", "outputevents", 226566, "Urine and GU Irrigant Out",
             "mixes irrigant with urine; not in mimic-code urine_output", v1="removed"),
    _mimic_x("urine_output", "outputevents", 226627, "OR Urine", "intra-operative, charted outside the ICU", v1="removed"),
    _mimic_x("urine_output", "outputevents", 226631, "PACU Urine", "post-anaesthesia unit", v1="removed"),
    _mimic_x("urine_output", "outputevents", 226713, "Incontinent/voids (estimate)", "text estimate, no volume"),
)

# ---------------------------------------------------------------------------------------
# eICU-CRD 2.0 (column names and examples from the eicu-code table docs; lab names and
# urine labels from eicu-code concepts/pivoted/*.sql)
# ---------------------------------------------------------------------------------------

EICU_URINE_CELLPATH_PREFIX = "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|"

#: Urine labels after the prefix, from eicu-code concepts/pivoted/pivoted-uo.sql (125 labels;
#: case/whitespace duplicates collapsed because matching is case-insensitive).
EICU_CODE_URINE_LABELS: tuple[str, ...] = (
    'Urine', '3 way foley', 'Actual Urine', 'Adjusted total UO NOC end shift', 'BRP (urine)',
    'condome cath urine', 'diaper urine', 'inc of urine', 'incontient urine',
    'incontinence of urine', 'Incontinence-urine', 'incontinence/ voids urine',
    'incontinent of urine', 'Incontinent UOP', 'incontinent urine', 'Incontinent (urine)',
    'incontinent urine counts', 'incont of urine', 'incont. of urine', 'incont. of urine count',
    'incont urine', 'incont. urine', 'inc urine', 'inc. urine', 'indwelling foley',
    'Straight Catheter-Foley', 'Straight Catheterization Urine', 'Straight Cath UOP',
    'straight cath urine', 'strait cath Urine', 'Suprapubic Urine Output', 'true urine',
    'True Urine out', 'unmeasured urine', 'unmeasured urine output', 'Urethal Catheter',
    'Urethral Catheter', 'urinary output 7AM - 7 PM', 'urinary output 7AM-7PM', 'URINE CATHETER',
    'Intermittent/Straight Cath (mL)', 'straightcath', 'straight cath', "Straight Cath'd",
    'straight cath daily', 'straight cathed', 'Straight catheterization',
    'Straight Catheter Output', 'Straight Catheter-Straight Catheter', "straight cath ml's",
    'Straight Cath Q6hrs', 'Straight caths', 'Urine Output-straight cath', 'Urine-straight cath',
    'Urine Straight Cath', 'Condom Catheter', 'condom cath', 'CONDOM CATHETER OUTPUT',
    'Urine via condom catheter', 'Urine-foley', 'Urine- foley', 'Urine foley catheter',
    'Urine, L neph:', 'Urine (measured)', 'urine output', 'Urine Output-external catheter',
    'Urine Output-foley', 'Urine Output-Foley cath', 'Urine Output-foley catheter',
    'Urine Output-Foley Output', 'Urine Output-Fpley', 'Urine Output-Ileoconduit',
    'Urine Output-left nephrostomy', 'Urine Output-Left Nephrostomy Tube',
    'Urine Output-LEFT PCN TUBE', 'Urine Output-L Nephrostomy', 'Urine Output-L Nephrostomy Tube',
    'Urine Output-Nephrostomy', 'Urine Output-right nephrostomy',
    'Urine Output-RIGHT Nephrouretero Stent Urine Output', 'Urine Output-R nephrostomy',
    'Urine Output-R. Nephrostomy', 'Urine Output-R Nephrostomy Tube',
    'Urine Output-Rt Nephrectomy', 'Urine Output-stent', 'Urine Output-suprapubic',
    'Urine Output-Texas Cath', 'Urine Output-Urine', 'Urine Output-Urine Output', 'Urine, R neph:',
    'urine (void)', 'Urine- void', 'Urine, void:',
)
#: Prefix rules from the same SQL (applied to the label after the prefix).
EICU_CODE_URINE_LABEL_PATTERNS: tuple[str, ...] = (
    r"^output.*urinary catheter",
    r"^output.*urethral catheter",
    r"^urine output \(ml\)",
    r"^output.*external urethral",
    r"^urinary catheter output",
)
#: v2 additions on top of eicu-code: counts, unmeasured entries and shift totals are not
#: per-record volumes, and irrigation is not urine.
EICU_URINE_LABEL_EXCLUSION = re.compile(r"\bcounts?\b|unmeasured|7\s*am|7\s*pm|end shift|irrigat|\bcbi\b|pacu")
_EICU_FOLEY_EXCLUDED = re.compile(r"pacu|or|ir")  # eicu-code: foley% not like %pacu%/%or%/%ir%

#: Values of respiratoryCharting 'RT Vent On/Off' (unverified on raw data, see the spec).
EICU_VENT_ON_VALUES: tuple[str, ...] = ("Start", "Continued")
EICU_VENT_OFF_VALUES: tuple[str, ...] = ("Off", "Suspended")

#: Non-blood specimen markers; only used to word the reason of an excluded lab name.
EICU_NON_BLOOD_SPECIMEN = re.compile(
    r"urin|csf|cerebrospinal|fluid|pleural|peritoneal|ascit|synovial|pericard|drain|stool|sputum|dialysate|\bbal\b"
)

_V1_LAB_RULE = "v1 substring rule (eicu_dataset.py:223-239); captured names outside the exact allowlist are excluded"
_UNVERIFIED = "exact string not yet verified on raw lab.csv.gz"

EICU_TABLE: tuple[MappingRow, ...] = (
    # Vitals: vitalPeriodic = 5-min medians from monitors (invasive BP), vitalAperiodic = NIBP
    _eicu("HR", "vitalPeriodic", "heartrate", "column", "heartRate", "bpm"),
    _eicu("RR", "vitalPeriodic", "respiration", "column", "respiration", "/min"),
    _eicu("SpO2", "vitalPeriodic", "sao2", "column", "saO2", "%", "medium", "kept",
          "named saO2 but documented as pulse oximetry (eICU vitalPeriodic docs)"),
    _eicu("SBP", "vitalPeriodic", "systemicsystolic", "column", "systemicSystolic (arterial line)", "mmHg",
          modality="invasive"),
    _eicu("SBP", "vitalAperiodic", "noninvasivesystolic", "column", "nonInvasiveSystolic", "mmHg",
          modality="noninvasive"),
    _eicu("DBP", "vitalPeriodic", "systemicdiastolic", "column", "systemicDiastolic (arterial line)", "mmHg",
          modality="invasive"),
    _eicu("DBP", "vitalAperiodic", "noninvasivediastolic", "column", "nonInvasiveDiastolic", "mmHg",
          modality="noninvasive"),
    _eicu("MAP", "vitalPeriodic", "systemicmean", "column", "systemicMean (arterial line)", "mmHg", modality="invasive"),
    _eicu("MAP", "vitalAperiodic", "noninvasivemean", "column", "nonInvasiveMean", "mmHg", modality="noninvasive"),
    _eicu("Temp", "vitalPeriodic", "temperature", "column", "temperature", "°C", "medium", "kept",
          "only present when a probe is interfaced; manual temperatures live in nurseCharting"),
    _eicu_x("SBP", "vitalPeriodic", "pasystolic", "column", "paSystolic", "pulmonary artery pressure"),
    _eicu_x("DBP", "vitalPeriodic", "padiastolic", "column", "paDiastolic", "pulmonary artery pressure"),
    _eicu_x("MAP", "vitalPeriodic", "pamean", "column", "paMean", "pulmonary artery pressure"),
    _eicu_x("MAP", "vitalPeriodic", "cvp", "column", "cvp", "central venous pressure"),
    _eicu_x("Temp", "nurseCharting", "Temperature (C)", "exact", "nursingchartcelltypevalname 'Temperature (C)'",
            "candidate source of manual temperatures; needs a pass over the 1.7 GB nurseCharting table",
            confidence="medium"),
    # Labs: exact allowlist (names from eicu-code pivoted-lab.sql / pivoted-bg.sql)
    _eicu("Lactate", "lab", "lactate", "exact", "lactate", "mmol/L"),
    _eicu("pH", "lab", "pH", "exact", "pH", ""),
    _eicu("Creatinine", "lab", "creatinine", "exact", "creatinine", "mg/dL"),
    _eicu("Glucose", "lab", "glucose", "exact", "glucose (laboratory)", "mg/dL"),
    _eicu("Bilirubin_Total", "lab", "total bilirubin", "exact", "total bilirubin", "mg/dL"),
    _eicu("WBC", "lab", "WBC x 1000", "exact", "WBC x 1000", "K/mcL"),
    _eicu("Platelets", "lab", "platelets x 1000", "exact", "platelets x 1000", "K/mcL"),
    _eicu_x("Glucose", "lab", "bedside glucose", "exact", "bedside glucose",
            "point-of-care capillary glucose (3.18M rows vs 1.32M lab glucose); excluded to match MIMIC lab glucose "
            "(team decision)", v1="removed", confidence="medium"),
    _eicu_x("Glucose", "lab", "glucose - CSF", "exact", "glucose - CSF", f"CSF; {_UNVERIFIED}", v1="removed",
            confidence="medium"),
    _eicu_x("Creatinine", "lab", "urinary creatinine", "exact", "urinary creatinine", f"urine; {_UNVERIFIED}",
            v1="removed", confidence="medium"),
    _eicu_x("WBC", "lab", "WBC's in urine", "exact", "WBC's in urine", f"urine; {_UNVERIFIED}", v1="removed",
            confidence="medium"),
    _eicu_x("WBC", "lab", "WBC's in body fluid", "exact", "WBC's in body fluid", f"body fluid; {_UNVERIFIED}",
            v1="removed", confidence="medium"),
    _eicu_x("WBC", "lab", "WBC's in cerebrospinal fluid", "exact", "WBC's in cerebrospinal fluid",
            f"CSF; {_UNVERIFIED}", v1="removed", confidence="medium"),
    _eicu_x("WBC", "lab", "WBC's in pleural fluid", "exact", "WBC's in pleural fluid", f"pleural; {_UNVERIFIED}",
            v1="removed", confidence="medium"),
    _eicu_x("WBC", "lab", "WBC's in peritoneal fluid", "exact", "WBC's in peritoneal fluid",
            f"peritoneal; {_UNVERIFIED}", v1="removed", confidence="medium"),
    _eicu_x("WBC", "lab", "WBC's in synovial fluid", "exact", "WBC's in synovial fluid", f"synovial; {_UNVERIFIED}",
            v1="removed", confidence="medium"),
    _eicu_x("Bilirubin_Total", "lab", "direct bilirubin", "exact", "direct bilirubin",
            f"different analyte; {_UNVERIFIED}", confidence="medium"),
    _eicu_x("SpO2", "lab", "O2 Sat (%)", "exact", "O2 Sat (%)", f"blood-gas saturation; {_UNVERIFIED}",
            confidence="low"),
    _eicu_x("Lactate", "lab", r"lactate", "regex", "v1 rule: 'lactate' in name", _V1_LAB_RULE, v1="removed"),
    _eicu_x("pH", "lab", r"^ph$|^ph ", "regex", "v1 rule: name == 'ph' or startswith 'ph '", _V1_LAB_RULE, v1="removed"),
    _eicu_x("Creatinine", "lab", r"creatinine", "regex", "v1 rule: 'creatinine' in name", _V1_LAB_RULE, v1="removed"),
    _eicu_x("Glucose", "lab", r"glucose", "regex", "v1 rule: 'glucose' in name", _V1_LAB_RULE, v1="removed"),
    _eicu_x("Bilirubin_Total", "lab", r"bilirubin.*total|total.*bilirubin", "regex",
            "v1 rule: 'bilirubin' and 'total' in name", _V1_LAB_RULE, v1="removed"),
    _eicu_x("WBC", "lab", r"wbc|white blood", "regex", "v1 rule: 'wbc' or 'white blood' in name", _V1_LAB_RULE,
            v1="removed"),
    _eicu_x("Platelets", "lab", r"platelet", "regex", "v1 rule: 'platelet' in name", _V1_LAB_RULE, v1="removed"),
    # Vasopressors: infusionDrug.drugname patterns (derived from eicu-code pivoted-infusion.sql)
    _eicu("vasopressor_on", "infusionDrug", r"\bnor-?epinephrine\b|\bnorepi\b|\blevophed\b|\blevo\b", "regex",
          "norepinephrine incl. Levophed and 'NSS with LEVO'", "", "medium", "fixed",
          "v1 substrings missed the Levophed/LEVO names listed in eicu-code"),
    _eicu("vasopressor_on", "infusionDrug", r"\bepinephrine\b|\bepinepherine\b|\bepi\b|\badrenaline?\b", "regex",
          "epinephrine incl. 'EPI (mcg/min)' and the misspelling 'Epinepherine'", "", "medium", "fixed"),
    _eicu("vasopressor_on", "infusionDrug", r"\bphenylephrine\b|\bneo-? ?synephrine\b|\bneosynsprine\b", "regex",
          "phenylephrine incl. Neo-Synephrine", "", "medium", "fixed", "v1 substrings missed Neo-Synephrine"),
    _eicu("vasopressor_on", "infusionDrug", r"\bvasopressin\b|\bvasostrict\b|\bvaso\b", "regex",
          "vasopressin incl. 'levo/vaso' mixtures", "", "medium", "fixed"),
    _eicu("vasopressor_on", "infusionDrug", r"\bdopamine\b|\bino?tropin\b", "regex", "dopamine", "", "medium", "kept"),
    _eicu_x("vasopressor_on", "infusionDrug", r"dobu", "regex", "dobutamine incl. Dobutrex",
            "inotrope; v1 counted it as a vasopressor (eicu_dataset.py:33)", v1="removed"),
    _eicu_x("vasopressor_on", "infusionDrug", r"milrinone|milronone|primacor", "regex", "milrinone", "inodilator"),
    _eicu_x("vasopressor_on", "infusionDrug", r"isoproterenol|isuprel", "regex", "isoproterenol", "inotrope/chronotrope"),
    _eicu_x("vasopressor_on", "infusionDrug", r"terlipressin|angiotensin|giapreza|methylene blue", "regex",
            "terlipressin, angiotensin II, methylene blue", f"{_OUTSIDE_SET} (no MIMIC itemid)"),
    _eicu_x("vasopressor_on", "infusionDrug", r"lidocaine|racemic", "regex",
            "lidocaine-epinephrine or nebulised racemic epinephrine", "not a systemic vasopressor infusion"),
    # Sedation
    _eicu("sedation_on", "infusionDrug", r"\bpropofol\b|\bdiprivan\b", "regex", "propofol incl. Diprivan", "",
          "medium", "fixed", "brand name added; unverified on raw data"),
    _eicu("sedation_on", "infusionDrug", r"\bmidazolam\b|\bversed\b", "regex", "midazolam incl. Versed", "",
          "medium", "fixed", "brand name added; unverified on raw data"),
    _eicu("sedation_on", "infusionDrug", r"\bdexmedetomidine\b|\bprecedex\b", "regex",
          "dexmedetomidine incl. Precedex", "", "medium", "fixed", "brand name added; unverified on raw data"),
    _eicu("sedation_on", "infusionDrug", r"\bfentanyl\b|\bsublimaze\b", "regex", "fentanyl incl. Sublimaze", "",
          "medium", "kept"),
    _eicu_x("sedation_on", "infusionDrug", r"bupivacaine|ropivacaine|epidural|\bpca\b|patch|transdermal", "regex",
            "epidural, PCA, local-anaesthetic and transdermal preparations",
            "not a continuous IV infusion (also guards epinephrine-containing epidural mixtures)"),
    _eicu_x("sedation_on", "infusionDrug",
            r"lorazepam|ativan|ketamine|ketalar|morphine|hydromorphone|dilaudid|sufentanil|remifentanil|ultiva|"
            r"clonidine|pentobarbital", "regex", "other sedatives and opioids", _OUTSIDE_SET),
    # Urine output
    _eicu("urine_output", "intakeOutput", "eicu_urine_cellpath", "rule",
          "cellPath under Output (ml) with a urine label (eicu-code pivoted-uo.sql + v2 exclusions)", "mL",
          "medium", "fixed", "value = cellValueNumeric of the row, summed per bin"),
    _eicu_x("urine_output", "intakeOutput", "outputtotal", "column", "outputTotal",
            "cumulative total of ALL outputs (drains, blood loss, ...) up to the offset, duplicated across rows at the "
            "same offset (eICU intakeOutput docs); v1 summed it as urine (eicu_dataset.py:421-426)", v1="removed"),
    # Ventilation
    _eicu("ventilation_on", "respiratoryCare", "ventstartoffset", "column",
          "ventStartOffset/ventEndOffset interval", "", "low", "added",
          "documented vent start/end; coverage is known to be sparse, verify"),
    _eicu("ventilation_on", "respiratoryCharting", "RT Vent On/Off", "exact",
          "respChartValueLabel 'RT Vent On/Off' (on: EICU_VENT_ON_VALUES)", "", "low", "added",
          "label and values not yet verified on raw data"),
    _eicu_x("ventilation_on", "respiratoryCharting", r"vent|peep|fio2|ett|endotracheal", "regex",
            "v1 keyword rule over respChartTypeCat + label + value",
            "'ett' matches 'respFlowSettings' and 'vent' matches 'respFlowPtVentData', so any flow-settings row "
            "(including nasal-cannula O2) became ventilation_on (eicu_dataset.py:43-49, 384-391)", v1="removed"),
    _eicu_x("ventilation_on", "respiratoryCharting", "FiO2", "exact", "FiO2",
            "also charted for conventional O2 and HFNC", v1="removed", confidence="medium"),
)

# ---------------------------------------------------------------------------------------
# HiRID 1.1.1 (hirid_variable_reference.csv; codes from ordinal_vars_ref.csv)
# ---------------------------------------------------------------------------------------

#: Ventilator mode (variableid 3845) codes that mean the ventilator is in use.
HIRID_VENT_MODE_CODES_ON: dict[int, str] = {
    2: "DGV", 3: "CMV", 4: "Druckreg. Vol.Kontr.", 5: "Volumenunterstützung", 6: "SIMV(CMV)+DU",
    7: "SIMV(DGV.)+DU", 8: "DU/ST", 10: "SIMV-PregVctrl+Psupp", 11: "Bi-Vent", 12: "NIV DGV",
    13: "NIV DU", 14: "Nasal CPAP", 15: "NAVA", 17: "NIV neur adj",
}
HIRID_VENT_MODE_CODES_OFF: dict[int, str] = {1: "Stand By", 9: "Optional (meaning unclear, verify)"}
#: Ventilator airway code (variableid 15001552): 2 'tracheotomiert' counts only with a mode.
HIRID_AIRWAY_CODES_ON: dict[int, str] = {1: "intubiert", 3: "Maske", 4: "Helm", 5: "Mundstück(Stellar)"}

_V1_INFER = "v1 mapped it by description inference (hirid_dataset.py:141-182)"
_UNIT_GATE = "name-inferred by v1 and blocked only by the unit gate (hirid_dataset.py:195-209)"
_HIRID_BOLUS = "bolus preparation, not a continuous infusion"

HIRID_TABLE: tuple[MappingRow, ...] = (
    _hirid("HR", "observation", 200, "Heart rate", "/min"),
    _hirid("RR", "observation", 300, "Respiratory rate", "/min", note="pharmaid 300 is metronidazole; always pass the namespace"),
    _hirid("RR", "observation", 310, "Respiratory rate", "/min", "medium"),
    _hirid("RR", "observation", 5685, "Respiratory rate", "/min", "medium"),
    _hirid_x("RR", "observation", 320, "Ventilator rate", "set ventilator rate, not a measurement"),
    _hirid_x("RR", "pharma", 300, "Metronidazole tabl 200 mg",
             "namespace collision with observation 300; v1 keeps one ID lookup for variableid and pharmaid "
             "(hirid_dataset.py:312-332, 611-637), so pharma rows with pharmaid 300 would become RR", v1="removed"),
    _hirid("SpO2", "observation", 4000, "Peripheral oxygen saturation", "%"),
    _hirid("SpO2", "observation", 8280, "Peripheral oxygen saturation", "%"),
    _hirid_x("SpO2", "observation", 4200, "Mixed venous oxygen saturation", f"SvO2; {_V1_INFER}", v1="removed"),
    _hirid_x("SpO2", "observation", 20000800, "Oxygen saturation in Arterial blood", f"blood-gas SaO2; {_V1_INFER}",
             v1="removed"),
    _hirid_x("SpO2", "observation", 20001000, "Central venous oxygenation saturation", "ScvO2"),
    _hirid_x("SpO2", "observation", 24000737, "Central venous oxygenation saturation", "ScvO2"),
    _hirid("SBP", "observation", 100, "Invasive systolic arterial pressure", "mmHg", modality="invasive"),
    _hirid("SBP", "observation", 600, "Non-invasive systolic arterial pressure", "mmHg", modality="noninvasive"),
    _hirid_x("SBP", "observation", 150, "Pulmonary artery systolic pressure", "pulmonary circulation"),
    _hirid("DBP", "observation", 120, "Invasive diastolic arterial pressure", "mmHg", modality="invasive"),
    _hirid("DBP", "observation", 620, "Non-invasive diastolic arterial pressure", "mmHg", modality="noninvasive"),
    _hirid_x("DBP", "observation", 170, "Pulmonary artery diastolic pressure", "pulmonary circulation"),
    _hirid("MAP", "observation", 110, "Invasive mean arterial pressure", "mmHg", modality="invasive"),
    _hirid("MAP", "observation", 610, "Non-invasive mean arterial pressure", "mmHg", modality="noninvasive"),
    _hirid_x("MAP", "observation", 160, "Pulmonary artery mean pressure", "pulmonary circulation"),
    _hirid_x("MAP", "observation", 700, "Central venous pressure", "CVP"),
    _hirid_x("MAP", "observation", 960, "Central venous pressure", "CVP"),
    _hirid_x("MAP", "observation", 15001441, "Central venous pressure", "CVP"),
    _hirid("Temp", "observation", 410, "Core body temperature", "°C"),
    _hirid("Temp", "observation", 7100, "Rectal temperature", "°C"),
    _hirid("Temp", "observation", 400, "Axillary temperature", "°C", "medium", note="peripheral site, reads lower"),
    _hirid("ventilation_on", "observation", 3845, "Ventilator mode", "", "medium", "added",
           "on for HIRID_VENT_MODE_CODES_ON; v1 had no ventilation mapping"),
    _hirid("ventilation_on", "observation", 15001552, "Ventilator Airway Code", "", "medium", "added",
           "on for HIRID_AIRWAY_CODES_ON"),
    _hirid("ventilation_on", "observation", 2600, "Positive end expiratory pressure setting", "cmH2O", "medium", "added"),
    _hirid_x("ventilation_on", "observation", 2010, "Inspired oxygen concentration", "FiO2 is also set for O2 therapy"),
    _hirid_x("ventilation_on", "observation", 15001555, "Oxygen administration by mask", "conventional O2 (l/min)"),
    _hirid_x("ventilation_on", "observation", 15001525, "Oxygen administration by nasal cannula", "conventional O2"),
    _hirid("Lactate", "observation", 24000524, "Lactate [Mass/volume] in Arterial blood", "mmol/l",
           note="LOINC name says Mass/volume but the unit is mmol/l; verify the value scale"),
    _hirid("Lactate", "observation", 24000732, "Lactate [Moles/volume] in Venous blood", "mmol/l", "medium"),
    _hirid("Lactate", "observation", 24000485, "Lactate [Moles/volume] in Venous blood", "mmol/l", "medium"),
    _hirid_x("Lactate", "observation", 24000564, "Lactate [Moles/volume] in Cerebral spinal fluid",
             f"CSF; {_V1_INFER}", v1="removed"),
    _hirid("pH", "observation", 20000300, "pH of Arterial blood", ""),
    _hirid_x("pH", "observation", 24000608, "pH of Cerebral spinal fluid", f"CSF; {_V1_INFER}", v1="removed"),
    _hirid_x("pH", "observation", 24000741, "pH of Body fluid", f"drain fluid; {_V1_INFER}", v1="removed"),
    _hirid("Creatinine", "observation", 20000600, "Creatinine [Moles/volume] in Blood", "umol/l",
           note="converted umol/L -> mg/dL"),
    _hirid_x("Creatinine", "observation", 24000572, "Creatinine [Moles/volume] in Urine", f"urine; {_V1_INFER}",
             v1="removed"),
    _hirid_x("Creatinine", "observation", 24000573, "Creatinine [Moles/volume] in Urine", f"urine; {_V1_INFER}",
             v1="removed"),
    _hirid("Glucose", "observation", 20005110, "Glucose [Moles/volume] in Serum or Plasma", "mmol/l",
           note="converted mmol/L -> mg/dL"),
    _hirid("Glucose", "observation", 24000523, "Glucose [Moles/volume] in Serum or Plasma", "mmol/l", "medium",
           note="analyser not documented (lab vs blood gas); verify"),
    _hirid("Glucose", "observation", 24000585, "Glucose [Moles/volume] in Serum or Plasma", "mmol/l", "medium",
           note="analyser not documented (lab vs blood gas); verify"),
    _hirid_x("Glucose", "observation", 24000400, "Glucose [Moles/volume] in Cerebral spinal fluid",
             f"CSF; {_V1_INFER}", v1="removed"),
    *(
        _hirid_x("Glucose", "pharma", pid, name, "dextrose solution (a drug), not a measurement")
        for pid, name in (
            (1000022, "Glucose 10%"), (1000690, "Glucose 10%"), (1000689, "Glucose 20%"),
            (1000544, "Glucose 20% /100ml"), (1000746, "Glucose 20%/100ml Pflege"), (1000835, "Glucose 20% /500ml"),
            (1000060, "Glucose 30%"), (1000545, "Glucose 40%"), (1000567, "Glucose 50%"),
        )
    ),
    _hirid("Bilirubin_Total", "observation", 20004300, "Bilirubin.total [Moles/volume] in Serum or Plasma", "umol/l",
           note="converted umol/L -> mg/dL"),
    _hirid_x("Bilirubin_Total", "observation", 24000560, "Bilirubin.direct [Mass/volume] in Serum or Plasma",
             "different analyte"),
    _hirid("WBC", "observation", 20000700, "Leukocytes [#/volume] in Blood", "G/l"),
    _hirid_x("WBC", "observation", 24000550, "Neutrophils/100 leukocytes in Blood", f"differential %; {_UNIT_GATE}"),
    _hirid_x("WBC", "observation", 24000556, "Segmented neutrophils/100 leukocytes in Blood",
             f"differential %; {_UNIT_GATE}"),
    _hirid_x("WBC", "observation", 24000557, "Band form neutrophils/100 leukocytes in Blood",
             f"differential %; {_UNIT_GATE}"),
    _hirid_x("WBC", "observation", 24000480, "Lymphocytes [#/volume] in Blood", "lymphocyte count, not total WBC"),
    _hirid("Platelets", "observation", 20000110, "Platelets [#/volume] in Blood", "G/l", "high", "fixed",
           "v1 quarantined it: 'G/l' failed the unit hint list (hirid_dataset.py:108, 195-209)"),
    _hirid_x("Platelets", "observation", 24000536,
             "Fibrinogen [Mass/volume] in Platelet poor plasma by Coagulation assay", f"'platelet' is the specimen; {_UNIT_GATE}"),
    _hirid_x("Platelets", "observation", 24000666,
             "Prothrombin activity actual/normal in Platelet poor plasma by Coagulation assay",
             f"'platelet' is the specimen; {_UNIT_GATE}"),
    _hirid_x("Platelets", "observation", 24000637,
             "Coagulation factor V activity actual/normal in Platelet poor plasma by Coagulation assay",
             f"'platelet' is the specimen; {_UNIT_GATE}"),
    _hirid_x("Platelets", "observation", 24000552,
             "Coagulation factor VII activity actual/normal in Platelet poor plasma by Coagulation assay",
             f"'platelet' is the specimen; {_UNIT_GATE}"),
    _hirid_x("Platelets", "observation", 24000667,
             "Coagulation factor X activity actual/normal in Platelet poor plasma by Coagulation assay",
             f"'platelet' is the specimen; {_UNIT_GATE}"),
    _hirid_x("Platelets", "pharma", 1000245, "Platelet transfusion", "transfused blood product"),
    _hirid_x("Platelets", "pharma", 1000201, "Platelet transfusion", "transfused blood product"),
    # Vasopressors (no dopamine or phenylephrine exists in the HiRID reference)
    _hirid("vasopressor_on", "pharma", 1000656, "Noradrenalin 100 µg/ml Perfusor"),
    _hirid("vasopressor_on", "pharma", 1000657, "Noradrenalin 20 µg/ml Perfusor"),
    _hirid("vasopressor_on", "pharma", 1000462, "Noradrenalin 1mg/ml", "", "medium", note="concentrate; verify typeid"),
    _hirid("vasopressor_on", "pharma", 1000649, "Adrenalin 20 µg/ml Perfusor"),
    _hirid("vasopressor_on", "pharma", 1000650, "Adrenalin 100 µg/ml Perfusor"),
    _hirid("vasopressor_on", "pharma", 113, "Vasopressin inf 0.4 U/ml"),
    _hirid("vasopressor_on", "pharma", 112, "Vasopressin inj 20 U/ml", "", "medium", note="injection strength; verify"),
    _hirid_x("vasopressor_on", "pharma", 1000658, "Noradrenalin 10 µg/ml Bolus", _HIRID_BOLUS, v1="removed"),
    _hirid_x("vasopressor_on", "pharma", 1000750, "Adrenalin 100 µg/ml Bolus", _HIRID_BOLUS, v1="removed"),
    _hirid_x("vasopressor_on", "pharma", 1000655, "Adrenalin 10 µg/ml Bolus", _HIRID_BOLUS, v1="removed"),
    _hirid_x("vasopressor_on", "pharma", 71, "Adrenalin  1mg/ml",
             "1 mg/ml ampoule: resuscitation bolus or nebulisation; infusions use the Perfusor items", v1="removed",
             confidence="medium"),
    _hirid_x("vasopressor_on", "pharma", 426, "Dobutrex 250 mg/20ml",
             "dobutamine (inotrope); v1 mapped it through pm41 'Dobutamine' raw ids in the preprocessed reference",
             v1="removed"),
    _hirid_x("vasopressor_on", "pharma", 1000441, "Corotrop Inj Lsg 1mg/ml 10 ml", "milrinone (inodilator)"),
    _hirid_x("vasopressor_on", "pharma", 1000606, "Simdax inf", "levosimendan (inodilator)"),
    _hirid_x("vasopressor_on", "pharma", 1000793, "Glypressin Inj Lsg", f"terlipressin; {_OUTSIDE_SET}"),
    _hirid_x("vasopressor_on", "pharma", 1001061, "Haemopressin TS Inj Lsg", f"terlipressin; {_OUTSIDE_SET}"),
    # Sedation: v1 never matched the brand names Disoprivan, Dormicum, Dexdor
    _hirid("sedation_on", "pharma", 1000491, "Disoprivan 1%", "", "medium", "added", "propofol; bolus items excluded"),
    _hirid("sedation_on", "pharma", 1001053, "Disoprivan 1%", "", "medium", "added", "propofol"),
    _hirid("sedation_on", "pharma", 1001050, "Disoprivan 2%", "", "medium", "added", "propofol"),
    _hirid("sedation_on", "pharma", 208, "Disoprivan 2%", "", "medium", "added", "propofol"),
    _hirid("sedation_on", "pharma", 252, "Dormicum Perfusor 5 mg/ml", "", "high", "added", "midazolam"),
    _hirid("sedation_on", "pharma", 1001051, "Dormicum Perfusor 5mg/ml", "", "high", "added", "midazolam"),
    _hirid("sedation_on", "pharma", 1001049, "Dexdor Inf. Lsg", "", "high", "added", "dexmedetomidine"),
    _hirid("sedation_on", "pharma", 1000847, "Dexdor Inf. Lsg", "", "high", "added", "dexmedetomidine"),
    _hirid("sedation_on", "pharma", 1000741, "Fentanyl Perf", "", "high", "kept"),
    _hirid("sedation_on", "pharma", 1000251, "Fentanyl inj 50 µg/ml", "", "medium", "kept",
           "ampoule used for both perfusor and bolus; verify typeid"),
    _hirid_x("sedation_on", "pharma", 1000692, "Fentanyl Bolus inj 50 µg/ml", _HIRID_BOLUS, v1="removed"),
    _hirid_x("sedation_on", "pharma", 1000621, "Fentanyl-PCA-Lösung", "patient-controlled analgesia", v1="removed"),
    _hirid_x("sedation_on", "pharma", 1000843, "PDA-Bupivacain 0.125% ohne Fentanyl",
             "epidural bupivacaine WITHOUT fentanyl; v1 matched the substring 'fentanyl'", v1="removed"),
    *(
        _hirid_x("sedation_on", "pharma", pid, name, "transdermal patch")
        for pid, name in (
            (1000350, "Durogesic TTS 25 ug"), (1000351, "Durogesic TTS 50 ug"),
            (1000627, "Durogesic TTS Matrix 100 ug"), (1000768, "Durogesic Matrix 12 ug/h"),
        )
    ),
    _hirid_x("sedation_on", "pharma", 1001052, "Disoprivan 2% BOLUS", _HIRID_BOLUS),
    _hirid_x("sedation_on", "pharma", 1000699, "Disoprivan 2% BOLUS", _HIRID_BOLUS),
    _hirid_x("sedation_on", "pharma", 1000691, "Disoprivan  BOLUS 2% 20mg/ml", _HIRID_BOLUS),
    _hirid_x("sedation_on", "pharma", 1000700, "Dormicum 5mg/ml BOLUS", _HIRID_BOLUS),
    _hirid_x("sedation_on", "pharma", 1001054, "Dormicum Bolus 1mg/ml", _HIRID_BOLUS),
    _hirid_x("sedation_on", "pharma", 251, "Dormicum inj 1 mg/ml", "1 mg/ml is the bolus strength; verify",
             confidence="medium"),
    _hirid_x("sedation_on", "pharma", 1000902, "Dormicum Tbl 7.5 mg", "oral"),
    _hirid_x("sedation_on", "pharma", 1000771, "Sufentanyl Perfusor",
             f"sufentanil; {_OUTSIDE_SET}; v1 matched the substring 'fentanyl'", v1="removed"),
    _hirid_x("sedation_on", "pharma", 1000705, "Ultiva Inj Lsg", f"remifentanil; {_OUTSIDE_SET}"),
    _hirid_x("sedation_on", "pharma", 1000862, "Morphine Inj à 10mg", f"morphine; {_OUTSIDE_SET}", v1="removed"),
    _hirid_x("sedation_on", "pharma", 1000659, "Morphin HCL Perfusor", f"morphine; {_OUTSIDE_SET}"),
    _hirid_x("sedation_on", "pharma", 1000239, "Temesta inj Amp 4mg/ml", f"lorazepam; {_OUTSIDE_SET}"),
    _hirid_x("sedation_on", "pharma", 1000400, "Ketalar Inj Lsg 500 mg/10 ml", f"ketamine; {_OUTSIDE_SET}"),
    _hirid_x("sedation_on", "pharma", 1000857, "Ketalar Perfusor", f"ketamine; {_OUTSIDE_SET}"),
    _hirid_x("sedation_on", "pharma", 1001194, "Ketalar 5mg/ml Perfusor Schmerzdienst", f"ketamine; {_OUTSIDE_SET}"),
    _hirid_x("sedation_on", "pharma", 1000728, "Catapresan 25µg /ml Perfusor", f"clonidine; {_OUTSIDE_SET}"),
    _hirid_x("sedation_on", "pharma", 202, "Pentothal 1g Inf Lsg", f"thiopental; {_OUTSIDE_SET}"),
    _hirid_x("sedation_on", "pharma", 1000607, "Pentothal inj 0.5g", f"thiopental; {_OUTSIDE_SET}"),
    _hirid_x("sedation_on", "pharma", 1001215, "Rivotril Perfusor", f"clonazepam; {_OUTSIDE_SET}"),
    # Urine output
    _hirid("urine_output", "observation", 10020000, "Hourly urine volume", "ml/h", "medium", "kept",
           "assumes one record per hour (value = mL in that hour); verify the record interval"),
    _hirid_x("urine_output", "observation", 20003200, "Sodium [Moles/volume] in Urine", f"urine chemistry; {_UNIT_GATE}"),
    _hirid_x("urine_output", "observation", 24000754, "Urea [Moles/volume] in Urine", f"urine chemistry; {_UNIT_GATE}"),
    *(
        _hirid_x("urine_output", "observation", vid, "Measurement of output from drain", "drain (Liquor) output")
        for vid in (10010020, 10010070, 10010071, 10010072)
    ),
    _hirid_x("urine_output", "observation", 30005010, "Fluid balance", "cumulative balance"),
    _hirid_x("urine_output", "observation", 30005110, "Fluid balance", "cumulative balance"),
)

MAPPING_TABLES: dict[str, tuple[MappingRow, ...]] = {
    "mimic_iv": MIMIC_IV_TABLE,
    "eicu": EICU_TABLE,
    "hirid": HIRID_TABLE,
}

#: Features a source cannot provide, with the reason. Every source covers all 18 features,
#: so the maps are empty; the validator still enforces "mapped or explicitly unavailable".
UNAVAILABLE_FEATURES: dict[str, dict[str, str]] = {"mimic_iv": {}, "eicu": {}, "hirid": {}}

#: How values become episode bins, per source, for the non-measurement features.
VALUE_RULES: dict[str, dict[str, str]] = {
    "mimic_iv": {
        "measurement": "valuenum at charttime; labevents joined to the stay by hadm_id within [intime, outtime]",
        "ventilation_on": "on between consecutive 'on' documentations (mode/device/PEEP), new episode after a 14 h gap "
        "(mimic-code ventilation.sql); text items are read from `value`, not `valuenum`",
        "vasopressor_on": "on for bins overlapping [starttime, endtime) of rows with rate > 0; rows without rate "
        "(boluses) do not count",
        "sedation_on": "same as vasopressor_on",
        "urine_output": "value per row (227488 negated), summed per bin",
    },
    "eicu": {
        "measurement": "value at the row offset (minutes from unit admission)",
        "ventilation_on": "respiratoryCare [ventstartoffset, ventendoffset]; respiratoryCharting 'RT Vent On/Off' "
        "Start/Continued until Off/Suspended or a 14 h gap",
        "vasopressor_on": "drugrate (else infusionrate) > 0 switches on, 0 switches off; carry forward to the next row "
        "of the same drug, capped by a gap fixed after the raw check",
        "sedation_on": "same as vasopressor_on",
        "urine_output": "cellValueNumeric of rows accepted by is_eicu_urine_cellpath, summed per bin; never outputTotal",
    },
    "hirid": {
        "measurement": "observation value at datetime; look up (namespace, variableid), never the bare ID",
        "ventilation_on": "on between consecutive 'on' codes of 3845/15001552 or a PEEP setting, 14 h gap rule",
        "vasopressor_on": "pharma rows of included pharmaids with givendose > 0, grouped by infusionid; on while the "
        "infusion runs (recordstatus/typeid semantics to verify)",
        "sedation_on": "same as vasopressor_on",
        "urine_output": "10020000 value (ml/h, one record per hour) summed per bin",
    },
}


# ---------------------------------------------------------------------------------------
# Lookups and exclusion helpers
# ---------------------------------------------------------------------------------------


def _norm_text(value: object) -> str:
    if _is_missing(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _try_int(value: object) -> int | None:
    if isinstance(value, bool) or _is_missing(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if float(value).is_integer() else None
    text = str(value).strip()
    return int(text) if re.fullmatch(r"[+-]?\d+", text) else None


def _identifier_key(match: str, identifier: object) -> object:
    if match == "id":
        as_int = _try_int(identifier)
        return as_int if as_int is not None else _norm_text(identifier)
    if match in ("exact", "column"):
        return _norm_text(identifier)
    return str(identifier)


@lru_cache(maxsize=None)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _check_source(source: str) -> None:
    if source not in MAPPING_TABLES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")


def _check_namespace(source: str, namespace: object) -> str:
    _check_source(source)
    if not isinstance(namespace, str) or namespace not in NAMESPACES[source]:
        raise ValueError(
            f"{source}: a namespace tag from {NAMESPACES[source]} is required, got {namespace!r} "
            "(identifiers are only unique within a namespace)"
        )
    return namespace


@lru_cache(maxsize=None)
def _lookup_index(source: str) -> dict[tuple[str, object], MappingRow]:
    index: dict[tuple[str, object], MappingRow] = {}
    for row in MAPPING_TABLES[source]:
        if row.match in ("id", "exact", "column"):
            index.setdefault((row.namespace, _identifier_key(row.match, row.identifier)), row)
    return index


def lookup_identifier(source: str, namespace: str, identifier: object) -> MappingRow | None:
    """Return the row for an exact identifier (id, column or exact name) in ``namespace``.

    The namespace is mandatory: a bare ID such as HiRID 300 is ambiguous. Pattern and rule
    rows are not returned here; use the classify helpers for those namespaces.
    """
    namespace = _check_namespace(source, namespace)
    index = _lookup_index(source)
    as_int = _try_int(identifier)
    if as_int is not None and (namespace, as_int) in index:
        return index[(namespace, as_int)]
    return index.get((namespace, _norm_text(identifier)))


def map_identifier(source: str, namespace: str, identifier: object) -> str | None:
    """Canonical feature of an included identifier, else ``None`` (excluded or unknown)."""
    row = lookup_identifier(source, namespace, identifier)
    return row.feature if row is not None and row.decision == "include" else None


def is_excluded_mimic_item(itemid: object) -> bool:
    """True if a MIMIC-IV itemid is explicitly excluded (itemids are unique across tables)."""
    key = _try_int(itemid)
    return any(row.decision == "exclude" and row.identifier == key for row in MIMIC_IV_TABLE)


def is_excluded_hirid_variable(namespace: str, variable_id: object) -> bool:
    """True if a HiRID (namespace, id) pair is explicitly excluded."""
    row = lookup_identifier("hirid", namespace, variable_id)
    return row is not None and row.decision == "exclude"


def bp_modality(source: str, namespace: str, identifier: object) -> str | None:
    """``invasive`` or ``noninvasive`` for an included blood-pressure identifier, else ``None``."""
    row = lookup_identifier(source, namespace, identifier)
    if row is None or row.decision != "include" or row.feature not in BP_FEATURES:
        return None
    return row.modality


@lru_cache(maxsize=None)
def _text_rules(source: str, namespace: str) -> tuple[tuple[MappingRow, ...], ...]:
    rows = [row for row in MAPPING_TABLES[source] if row.namespace == namespace]
    exact_include = tuple(r for r in rows if r.match in ("exact", "column") and r.decision == "include")
    exact_exclude = tuple(r for r in rows if r.match in ("exact", "column") and r.decision == "exclude")
    regex_exclude = tuple(r for r in rows if r.match == "regex" and r.decision == "exclude")
    regex_include = tuple(r for r in rows if r.match == "regex" and r.decision == "include")
    return exact_include, exact_exclude, regex_exclude, regex_include


def _classify_text(source: str, namespace: str, text: object) -> tuple[str, MappingRow | None]:
    """Precedence: exact include > exact exclude > pattern exclude > pattern include."""
    key = _norm_text(text)
    if not key:
        return "unmapped", None
    exact_include, exact_exclude, regex_exclude, regex_include = _text_rules(source, namespace)
    for row in exact_include:
        if _norm_text(row.identifier) == key:
            return "accepted", row
    for row in exact_exclude:
        if _norm_text(row.identifier) == key:
            return "excluded", row
    for row in regex_exclude:
        if _compiled(str(row.identifier)).search(key):
            return "excluded", row
    for row in regex_include:
        if _compiled(str(row.identifier)).search(key):
            return "accepted", row
    return "unmapped", None


def classify_eicu_lab(labname: object) -> tuple[str | None, str]:
    """Classify an eICU ``lab.labname``: ``(feature, "accepted")``, ``(None, "excluded: ...")``
    or ``(None, "unmapped")``. Only names in the exact allowlist are accepted."""
    decision, row = _classify_text("eicu", "lab", labname)
    if decision == "accepted" and row is not None:
        return row.feature, "accepted"
    if decision == "excluded" and row is not None:
        if row.match == "regex" and EICU_NON_BLOOD_SPECIMEN.search(_norm_text(labname)):
            return None, "excluded: non-blood specimen"
        return None, f"excluded: {row.note}"
    return None, "unmapped"


def map_eicu_lab(labname: object) -> str | None:
    """Canonical feature of an accepted eICU lab name, else ``None``."""
    return classify_eicu_lab(labname)[0]


def is_excluded_eicu_lab(labname: object) -> bool:
    """True for lab names v2 rejects although they resemble a feature (urine creatinine,
    CSF glucose, body-fluid WBC, bedside glucose, ...). Unrelated names return False."""
    return classify_eicu_lab(labname)[1].startswith("excluded")


def classify_eicu_drug(drugname: object) -> tuple[str | None, str]:
    """Classify an eICU ``infusionDrug.drugname`` into vasopressor_on / sedation_on."""
    decision, row = _classify_text("eicu", "infusionDrug", drugname)
    if decision == "accepted" and row is not None:
        return row.feature, "accepted"
    if decision == "excluded" and row is not None:
        return None, f"excluded: {row.note}"
    return None, "unmapped"


def is_excluded_eicu_drug(drugname: object) -> bool:
    """True for drug names v2 rejects explicitly (dobutamine, epidural mixtures, ...)."""
    return classify_eicu_drug(drugname)[1].startswith("excluded")


_EICU_URINE_LABEL_KEYS = frozenset(_norm_text(label) for label in EICU_CODE_URINE_LABELS)


def is_eicu_urine_cellpath(cellpath: object) -> bool:
    """True if an ``intakeOutput.cellPath`` row is a urine output volume (v2 rule)."""
    if _is_missing(cellpath):
        return False
    text = str(cellpath)
    if not text.startswith(EICU_URINE_CELLPATH_PREFIX):
        return False
    label = _norm_text(text[len(EICU_URINE_CELLPATH_PREFIX):])
    if not label or EICU_URINE_LABEL_EXCLUSION.search(label):
        return False
    if label in _EICU_URINE_LABEL_KEYS:
        return True
    if label.startswith("foley"):
        return _EICU_FOLEY_EXCLUDED.search(label) is None
    return any(re.search(pattern, label) for pattern in EICU_CODE_URINE_LABEL_PATTERNS)


#: Named rules referenced by ``match="rule"`` rows.
RULES: dict[str, Callable[[object], bool]] = {"eicu_urine_cellpath": is_eicu_urine_cellpath}


def cross_namespace_identifiers(
    source: str, rows: Iterable[MappingRow] | None = None
) -> dict[object, tuple[str, ...]]:
    """Integer identifiers that appear under more than one namespace (e.g. HiRID 300)."""
    seen: dict[object, set[str]] = {}
    for row in MAPPING_TABLES[source] if rows is None else rows:
        if row.match == "id":
            seen.setdefault(_identifier_key("id", row.identifier), set()).add(row.namespace)
    return {key: tuple(sorted(spaces)) for key, spaces in seen.items() if len(spaces) > 1}


# ---------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------


def _match_group(match: str) -> str:
    return {"id": "id", "exact": "str", "column": "str"}.get(match, match)


def _validate_specs() -> list[str]:
    problems: list[str] = []
    if tuple(CANONICAL_SPECS) != CANONICAL_FEATURES:
        problems.append("CANONICAL_SPECS keys differ from CANONICAL_FEATURES")
    for name, spec in CANONICAL_SPECS.items():
        if spec.kind not in FEATURE_KINDS:
            problems.append(f"{name}: unknown kind {spec.kind!r}")
        if not spec.low < spec.high:
            problems.append(f"{name}: plausible range low >= high")
        if spec.kind != "state" and spec.accepted_units.get(spec.unit) != IDENTITY:
            problems.append(f"{name}: canonical unit {spec.unit!r} must map with the identity conversion")
    return problems


def _validate_row(source: str, row: MappingRow) -> list[str]:
    where = f"{source}: {row.namespace or '<no namespace>'}/{row.identifier!r} ({row.feature})"
    problems: list[str] = []
    if row.source != source:
        problems.append(f"{where}: row.source is {row.source!r}")
    if row.feature not in CANONICAL_SPECS:
        problems.append(f"{where}: unknown feature")
    if row.namespace not in NAMESPACES.get(source, ()):
        problems.append(f"{where}: missing or unknown namespace tag (allowed: {NAMESPACES.get(source, ())})")
    if row.match not in MATCH_KINDS:
        problems.append(f"{where}: unknown match kind {row.match!r}")
    if row.decision not in DECISIONS:
        problems.append(f"{where}: unknown decision {row.decision!r}")
    if row.confidence not in CONFIDENCE_LEVELS:
        problems.append(f"{where}: unknown confidence {row.confidence!r}")
    if row.v1 not in V1_STATUSES:
        problems.append(f"{where}: unknown v1 status {row.v1!r}")
    if row.sign not in (1.0, -1.0):
        problems.append(f"{where}: sign must be +1 or -1")
    if row.match == "id" and _try_int(row.identifier) is None:
        problems.append(f"{where}: id rows need an integer identifier")
    if row.match in ("exact", "column", "regex", "rule") and not str(row.identifier).strip():
        problems.append(f"{where}: empty identifier")
    if row.match == "regex":
        try:
            re.compile(str(row.identifier))
        except re.error as exc:
            problems.append(f"{where}: invalid regex ({exc})")
    if row.match == "rule" and row.identifier not in RULES:
        problems.append(f"{where}: unknown rule")
    if row.decision == "exclude" and not row.note.strip():
        problems.append(f"{where}: exclusion without a reason")
    if row.decision == "include" and row.feature in BP_FEATURES and row.modality not in BP_MODALITIES:
        problems.append(f"{where}: blood-pressure rows need modality in {BP_MODALITIES}")
    if row.feature not in BP_FEATURES and row.modality:
        problems.append(f"{where}: modality is only defined for {BP_FEATURES}")
    if row.decision == "include" and row.feature in CANONICAL_SPECS and conversion_for(row.feature, row.unit) is None:
        problems.append(f"{where}: unit {row.unit!r} cannot be converted to {CANONICAL_SPECS[row.feature].unit}")
    return problems


def validate_mapping_tables(
    tables: Mapping[str, Sequence[MappingRow]] | None = None,
    unavailable: Mapping[str, Mapping[str, str]] | None = None,
) -> list[str]:
    """Check the mapping tables and return a list of problems (empty means valid).

    Checks: field vocabularies; every row carries a valid namespace tag; no identifier is
    listed twice within a namespace; an ID used in several namespaces is tagged everywhere;
    every canonical feature has an include row or an explicit unavailable reason (not both);
    include units convert to the canonical unit; regexes compile; exclusions give a reason.
    """
    tables = MAPPING_TABLES if tables is None else tables
    unavailable = UNAVAILABLE_FEATURES if unavailable is None else unavailable
    problems = _validate_specs()
    for source, rows in tables.items():
        if source not in NAMESPACES:
            problems.append(f"unknown source {source!r}")
            continue
        seen: dict[tuple[str, str, object], MappingRow] = {}
        for row in rows:
            problems.extend(_validate_row(source, row))
            key = (row.namespace, _match_group(row.match), _identifier_key(row.match, row.identifier))
            if key in seen:
                problems.append(
                    f"{source}: duplicate identifier {row.identifier!r} in namespace {row.namespace!r} "
                    f"({seen[key].feature}/{seen[key].decision} vs {row.feature}/{row.decision})"
                )
            else:
                seen[key] = row
        for identifier, spaces in cross_namespace_identifiers(source, rows).items():
            untagged = [space for space in spaces if space not in NAMESPACES[source]]
            if untagged:
                problems.append(
                    f"{source}: identifier {identifier!r} is used in namespaces {spaces} without a valid tag on every row"
                )
        declared = dict(unavailable.get(source, {}))
        for feature, reason in declared.items():
            if feature not in CANONICAL_SPECS:
                problems.append(f"{source}: unavailable feature {feature!r} is not canonical")
            elif not str(reason).strip():
                problems.append(f"{source}: {feature} is marked unavailable without a reason")
        included = {row.feature for row in rows if row.decision == "include"}
        for feature in CANONICAL_FEATURES:
            if feature in included and feature in declared:
                problems.append(f"{source}: {feature} is both mapped and marked unavailable")
            elif feature not in included and feature not in declared:
                problems.append(f"{source}: {feature} has no include row and no unavailable reason")
    return problems


# ---------------------------------------------------------------------------------------
# Checks on small frames
# ---------------------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")


def declared_units(source: str, feature: str) -> tuple[str, ...]:
    """Normalized units that ``source``'s include rows declare for ``feature``."""
    _check_source(source)
    units = {normalize_unit(row.unit) for row in MAPPING_TABLES[source] if row.feature == feature and row.decision == "include"}
    return tuple(sorted(units))


def _default_unit(source: str, feature: object) -> str:
    spec = CANONICAL_SPECS.get(feature) if isinstance(feature, str) else None
    if spec is None:
        return ""
    declared = declared_units(source, spec.name)
    if spec.kind != "state" and len(declared) == 1:
        return declared[0]
    return spec.unit


def _resolve_units(frame: pd.DataFrame, source: str) -> list[str]:
    """Unit per row: the ``unit`` column, else the row's (namespace, identifier), else the
    single unit the source declares for the feature, else the canonical unit."""
    has_ids = {"namespace", "identifier"}.issubset(frame.columns)
    units = frame["unit"].tolist() if "unit" in frame.columns else [None] * len(frame)
    namespaces = frame["namespace"].tolist() if has_ids else [None] * len(frame)
    identifiers = frame["identifier"].tolist() if has_ids else [None] * len(frame)
    resolved: list[str] = []
    for feature, unit, namespace, identifier in zip(frame["feature"].tolist(), units, namespaces, identifiers):
        if not _is_missing(unit):
            resolved.append(normalize_unit(unit))
            continue
        row = None
        if has_ids and isinstance(namespace, str) and namespace in NAMESPACES[source]:
            row = lookup_identifier(source, namespace, identifier)
        if row is not None and row.feature == feature:
            resolved.append(normalize_unit(row.unit))
        else:
            resolved.append(_default_unit(source, feature))
    return resolved


def to_canonical_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Copy of ``frame`` (columns ``feature``, ``value``, optional ``unit`` / ``namespace`` /
    ``identifier``) with ``unit_resolved``, ``unit_status`` and ``value_canonical`` added."""
    _check_source(source)
    _require_columns(frame, ("feature", "value"))
    resolved = _resolve_units(frame, source)
    values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)
    canonical = np.full(len(frame), np.nan)
    status: list[str] = []
    for position, (feature, unit) in enumerate(zip(frame["feature"].tolist(), resolved)):
        if feature not in CANONICAL_SPECS:
            status.append("unknown_feature")
            continue
        conversion = conversion_for(feature, unit)
        if conversion is None:
            status.append("rejected")
            continue
        canonical[position] = conversion.to_canonical(values[position])
        status.append("ok")
    out = frame.copy()
    out["unit_resolved"] = resolved
    out["unit_status"] = status
    out["value_canonical"] = canonical
    return out


_UNIT_REPORT_COLUMNS = [
    "feature", "unit", "normalized_unit", "canonical_unit", "n_rows", "expected_for_source", "conversion", "status",
]


def check_units(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Summarise the unit strings per feature in a small frame (columns ``feature``, ``unit``).

    ``status`` per (feature, unit): ``ok`` (canonical, or any unit of a state feature),
    ``convert`` (declared for this source, converted), ``unexpected`` (convertible but not
    declared for this source), ``rejected`` (not convertible) or ``unknown_feature``.
    """
    _check_source(source)
    _require_columns(frame, ("feature", "unit"))
    work = pd.DataFrame(
        {
            "feature": frame["feature"].astype(object).tolist(),
            "unit": ["" if _is_missing(unit) else str(unit) for unit in frame["unit"].tolist()],
        }
    )
    records: list[dict[str, object]] = []
    for (feature, unit), group in work.groupby(["feature", "unit"], sort=True, dropna=False):
        normalized = normalize_unit(unit)
        spec = CANONICAL_SPECS.get(feature)
        if spec is None:
            records.append(dict(zip(_UNIT_REPORT_COLUMNS, (feature, unit, normalized, "", len(group), False, "", "unknown_feature"))))
            continue
        conversion = conversion_for(feature, normalized)
        expected = spec.kind == "state" or normalized in declared_units(source, feature)
        if conversion is None:
            status = "rejected"
        elif spec.kind == "state" or (conversion.is_identity and expected):
            status = "ok"
        elif not expected:
            status = "unexpected"
        else:
            status = "convert"
        records.append(
            dict(
                zip(
                    _UNIT_REPORT_COLUMNS,
                    (feature, unit, normalized, spec.unit, len(group), expected, conversion.name if conversion else "", status),
                )
            )
        )
    return pd.DataFrame.from_records(records, columns=_UNIT_REPORT_COLUMNS)


_PLAUSIBILITY_COLUMNS = [
    "feature", "canonical_unit", "low", "high", "n_rows", "n_missing", "n_unit_rejected", "n_checked",
    "n_below", "n_above", "frac_implausible", "min", "max", "status",
]


def implausible_mask(frame: pd.DataFrame, source: str) -> pd.Series:
    """Boolean per row: True when the converted value lies outside the plausible range.

    Rows with missing values, rejected units or unknown features are False (see
    :func:`check_units` and :func:`plausibility_report` for those).
    """
    converted = to_canonical_frame(frame, source)
    flags: list[bool] = []
    for feature, value, status in zip(
        converted["feature"].tolist(), converted["value_canonical"].tolist(), converted["unit_status"].tolist()
    ):
        spec = CANONICAL_SPECS.get(feature)
        if spec is None or status != "ok" or math.isnan(value):
            flags.append(False)
        else:
            flags.append(bool(value < spec.low or value > spec.high))
    return pd.Series(flags, index=frame.index, dtype=bool)


def plausibility_report(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Per-feature count of values outside the canonical plausible range (after unit
    conversion; see :func:`to_canonical_frame` for how units are resolved)."""
    converted = to_canonical_frame(frame, source)
    converted["_raw"] = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)
    records: list[dict[str, object]] = []
    for feature, group in converted.groupby("feature", sort=True, dropna=False):
        spec = CANONICAL_SPECS.get(feature)
        missing = group["_raw"].isna()
        rejected = group["unit_status"] != "ok"
        checked = ~missing & ~rejected
        values = group.loc[checked, "value_canonical"].to_numpy(dtype=float)
        n_checked = int(checked.sum())
        if spec is None:
            low = high = math.nan
            n_below = n_above = 0
            status = "unknown_feature"
        else:
            low, high = spec.low, spec.high
            n_below = int((values < low).sum())
            n_above = int((values > high).sum())
            status = "no_values" if n_checked == 0 else ("flagged" if n_below + n_above else "ok")
        records.append(
            dict(
                zip(
                    _PLAUSIBILITY_COLUMNS,
                    (
                        feature,
                        spec.unit if spec else "",
                        low,
                        high,
                        len(group),
                        int(missing.sum()),
                        int((~missing & rejected).sum()),
                        n_checked,
                        n_below,
                        n_above,
                        (n_below + n_above) / n_checked if n_checked else math.nan,
                        float(values.min()) if values.size else math.nan,
                        float(values.max()) if values.size else math.nan,
                        status,
                    ),
                )
            )
        )
    return pd.DataFrame.from_records(records, columns=_PLAUSIBILITY_COLUMNS)


def merge_blood_pressure(
    frame: pd.DataFrame, source: str, keys: Sequence[str] = ("stay_id", "bin")
) -> pd.DataFrame:
    """Combine invasive and non-invasive SBP/DBP/MAP values with :data:`BP_MERGE_RULE`.

    ``frame`` is long format with ``keys``, ``feature``, ``value`` and either a ``modality``
    column or ``namespace`` + ``identifier`` (looked up in the source table). Other features
    are ignored; a BP row whose modality cannot be resolved raises ``ValueError``. Returns one
    row per (keys, feature) with ``value``, ``bp_source`` (auxiliary, not a model input by
    default), ``n_invasive`` and ``n_noninvasive``.
    """
    _check_source(source)
    keys = list(keys)
    _require_columns(frame, (*keys, "feature", "value"))
    columns = [*keys, "feature", "value", "bp_source", "n_invasive", "n_noninvasive"]
    bp = frame[frame["feature"].isin(BP_FEATURES)].copy()
    if "modality" in bp.columns:
        modality = bp["modality"].tolist()
    elif {"namespace", "identifier"}.issubset(bp.columns):
        modality = [
            bp_modality(source, namespace, identifier)
            if isinstance(namespace, str) and namespace in NAMESPACES[source]
            else None
            for namespace, identifier in zip(bp["namespace"].tolist(), bp["identifier"].tolist())
        ]
    else:
        raise ValueError("merge_blood_pressure needs a 'modality' column or 'namespace' + 'identifier' columns")
    bp["_modality"] = modality
    unresolved = ~bp["_modality"].isin(BP_MODALITIES)
    if unresolved.any():
        raise ValueError(f"cannot resolve the blood-pressure modality of {int(unresolved.sum())} row(s)")
    bp["_value"] = pd.to_numeric(bp["value"], errors="coerce")
    bp = bp.dropna(subset=["_value"])
    records: list[dict[str, object]] = []
    for group_key, group in bp.groupby([*keys, "feature"], sort=True, dropna=False):
        invasive = group.loc[group["_modality"] == "invasive", "_value"]
        noninvasive = group.loc[group["_modality"] == "noninvasive", "_value"]
        chosen, chosen_source = (invasive, "invasive") if len(invasive) else (noninvasive, "noninvasive")
        values = (*group_key, float(chosen.mean()), chosen_source, int(len(invasive)), int(len(noninvasive)))
        records.append(dict(zip(columns, values)))
    return pd.DataFrame.from_records(records, columns=columns)


# ---------------------------------------------------------------------------------------
# Review helpers
# ---------------------------------------------------------------------------------------


def mapping_frame(source: str) -> pd.DataFrame:
    """The mapping table of ``source`` as a DataFrame (one row per identifier)."""
    _check_source(source)
    return pd.DataFrame([asdict(row) for row in MAPPING_TABLES[source]])


def feature_summary(source: str) -> pd.DataFrame:
    """One row per canonical feature: included identifiers, declared units, conversions,
    excluded identifiers and the best/worst confidence of the include rows."""
    _check_source(source)
    order = {level: rank for rank, level in enumerate(CONFIDENCE_LEVELS)}
    records: list[dict[str, object]] = []
    for feature in CANONICAL_FEATURES:
        rows = [row for row in MAPPING_TABLES[source] if row.feature == feature]
        include = [row for row in rows if row.decision == "include"]
        exclude = [row for row in rows if row.decision == "exclude"]
        conversions = sorted(
            {
                conversion.name
                for row in include
                if (conversion := conversion_for(feature, row.unit)) is not None and not conversion.is_identity
            }
        )
        ranks = sorted(order[row.confidence] for row in include)
        confidence = (
            "unavailable"
            if not include
            else CONFIDENCE_LEVELS[ranks[0]]
            if ranks[0] == ranks[-1]
            else f"{CONFIDENCE_LEVELS[ranks[0]]}/{CONFIDENCE_LEVELS[ranks[-1]]}"
        )
        records.append(
            {
                "feature": feature,
                "include": "; ".join(
                    f"{row.namespace}:{row.identifier}" + (f" [{row.modality}]" if row.modality else "") for row in include
                ),
                "units": ", ".join(sorted({row.unit or "-" for row in include})),
                "conversion": "; ".join(conversions) or "identity",
                "exclude": "; ".join(f"{row.namespace}:{row.identifier}" for row in exclude),
                "confidence": confidence,
                "unavailable_reason": UNAVAILABLE_FEATURES.get(source, {}).get(feature, ""),
            }
        )
    return pd.DataFrame.from_records(records)
