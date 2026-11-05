"""eICU-CRD 2.0 source adapter for the v2 raw-data extraction (external cohort).

Converts the raw eICU tables into the four frames of :mod:`src.icu.v2.extract.common`
(``stays``, ``vasopressors``, ``events``, ``interventions``). The task itself (anchors,
windows, landmark-matched controls) is defined in ``common.py``; this module only maps
eICU onto it. Mapping tables and units come from :mod:`src.icu.v2.harmonization`
(read-only); every place where this adapter deviates from or extends those tables is
listed below with the raw-data evidence (checks R-E1..R-E7 of
``docs/reviews/v2_harmonization_spec.md`` section 8, run on 2026-10-21; outputs in
``artifacts/v2/harmonization/eicu_*`` and logs in ``artifacts/v2/logs/extract_eicu_*.log``).

Time and identifiers
--------------------
* eICU stores minutes relative to unit admission (``*offset`` columns). Every stay gets
  ``intime = 2000-01-01 00:00`` (offset 0) and ``outtime = intime + unitdischargeoffset``;
  every event is placed at ``intime + offset`` minutes. Only times relative to the stay
  matter for the task, so the shared calendar origin is harmless (all stays overlap in
  calendar time, nothing in the pipeline compares calendar times across stays).
* ``stay_id`` = ``patientunitstayid``. ``subject_id`` = ``int(sha1(uniquepid)[:15], 16)``
  (first 15 hex digits, < 2**60, so a positive int64; stable across runs and machines;
  collisions are checked at build time). v1 converted the text ``uniquepid`` ("002-34851")
  with ``to_numeric``, got NaN for every row and fell back to the stay id, so its
  "patient" splits were stay splits. R-E6: 200,859 stays, 139,367 patients, 38,483
  patients with more than one stay (99,975 stays).
* Adults only: ``age`` >= 18, the string ``"> 89"`` counts as adult (7,081 stays); 95
  stays with missing age and 530 minors are dropped; 1,213 stays with
  ``unitdischargeoffset <= 0`` are dropped (outtime must follow intime). 199,028 adult
  stays remain (the "base" population of every cache file).

Vasopressors (anchors) and sedation (``sedation_on``)
-------------------------------------------------------
* Source: ``infusionDrug`` (continuous infusions charted on the nursing flowsheet).
  Drug names are classified per distinct ``drugname`` with
  :func:`harmonization.classify_eicu_drug` plus the adapter rules in
  :func:`classify_infusion`. R-E2 (``eicu_drugnames.csv``): 108 names / 1,004,292 rows are
  vasopressors, including every eicu-code ``pivoted-infusion.sql`` name (Levophed,
  'NSS with LEVO', 'NSS w/ levo/vaso', Neo-Synephrine variants, 'EPI (mcg/min)',
  'Epinepherine (mcg/min)'); dobutamine, milrinone, isoproterenol and epidural mixtures
  are excluded. Angiotensin II is in the vasopressor set as requested, but eICU 2.0 has no
  drugname matching angiotensin/Giapreza (0 rows).
* Adapter exclusions on top of harmonization (R-E2 evidence): 'fentanyl epi (ml/hr)' and
  'fent epi (ml/hr)' (6 rows) are epidural fentanyl, but ``\\bepi\\b`` makes harmonization
  call them epinephrine; 'fentanyl/bupivicaine (ml/hr)' (64 rows) and
  'Fentanyl/Bupivicaine PCEA (ml/hr)' (16 rows) are epidural mixtures that escape the
  harmonization pattern because of the misspelling "bupivicaine" and the token "PCEA".
  Rule: names containing bupi/ropi/pcea/pca/epidural/duramorph, or "epi" together with an
  opioid or local anaesthetic, are never vasopressors or sedatives.
* Sedation set (task instruction): propofol, midazolam, dexmedetomidine, lorazepam.
  DEVIATION from ``harmonization.CANONICAL_SPECS['sedation_on']`` / ``EICU_TABLE``, which
  include fentanyl and exclude lorazepam. Evidence for the switch (R-E2): fentanyl
  infusions 597,028 rows / 12,323 stays (no longer counted), lorazepam infusions
  12,657 rows (14 names, all "(mg/hr)"/"(ml/hr)" drips, now counted). The set is the
  ``sedation_drugs`` option; the other sources must use the same set.
* Rate = ``drugrate`` parsed as a number, else ``infusionrate`` (never needed: 0 rows have
  a missing drugrate and a present infusionrate). Text rates: "OFF"/"off"/"0ff" (and the
  same with the "\\.br\\" markup) mean a documented stop and are read as 0; everything
  else that is not a plain number ("ERROR" 2,831 rows, "UD", "Documentation undone",
  "50 mcg/min", ...) and negative numbers are unparseable and the row is dropped.
  Vasopressor rows: 96,433 rate == 0, 3,369 without any rate (all from the unit-less
  names 'Norepinephrine', 'Phenylephrine', ...), 54 with text (8 read as stops).
* Intervals, per (stay, drugname) because each drugname is one flowsheet line: duplicate
  rows at one offset are merged with the max rate (1,227 vasopressor and 544 sedation
  offsets carry both 0 and > 0); a run starts at the first row with rate > 0; it ends at
  the next row with rate == 0, or ``last on-row + max_gap`` when no row follows within
  ``max_gap`` (default 4 h), or at unit discharge. Rows with rate 0 are never "on".
  ``max_gap`` evidence (R-E2): gaps between consecutive charted rows of one line have
  median 1 h, p90 2 h, p95 4 h (vasopressors 4.9 h incl. off rows), p99 12-14 h; 4.8 % of
  on->on gaps exceed 4 h. Consequence (2026-10-21 run): 87,220 vasopressor runs in
  24,607 stays end by gap 45,173 / rate 0 18,902 / last row 17,018 / discharge 6,127;
  85,222 sedation runs end by gap 48,950. Most > 4 h gaps are followed by another "on"
  row, so ``sedation_on`` has short artificial "off" stretches; the anchor (first start)
  does not depend on ``max_gap``. 8-12 h is the natural sensitivity setting.

Cohort restrictions (eICU-specific data-quality exclusions in ``stays()``)
-------------------------------------------------------------------------
eICU is multi-centre and not every hospital charts infusions in ``infusionDrug``: 56 of
208 hospitals (44,014 adult stays) have no infusionDrug row at all, 6 more chart other
infusions but never a vasopressor, and 13 more chart a vasopressor infusion for < 2 % of
their stays (e.g. 1 of 1,812) against a median of 13.7 %. In those hospitals "no
vasopressor row" does not mean "no vasopressor", so their stays cannot serve as controls,
and keeping only their positives would let hospital identity predict the label.
Separately, 16,164 adult stays without a vasopressor infusion row have vasopressor
evidence in ``medication`` (non-cancelled orders) or ``treatment``. ``stays()`` therefore
drops, in this order (counts are in ``provenance()['cohort']``):

1. hospitals where < ``min_hospital_vasopressor_share`` (default 0.02) of adult stays have
   a vasopressor interval;
2. stays with vasopressor rows in infusionDrug but no derivable interval (only rate 0,
   missing or text rates, or "on" rows only after unit discharge);
3. stays without a vasopressor interval but with vasopressor evidence elsewhere (treatment
   strings naming '|vasopressors' or one of the five drugs; medication orders of the five
   drugs that are not cancelled and look like drips: concentration/diluent in the name or
   frequency continuous/titrate/drip). Such stays are neither clean controls (MIMIC
   controls have no vasopressor at all) nor positives with a known start;
4. stays whose first vasopressor interval starts more than
   ``prior_evidence_tolerance_minutes`` (default 60 = the default prediction gap) after
   such medication/treatment evidence: the infusion charting started late, so the anchor
   is not the initiation and the pre-anchor window may already be on vasopressors. The
   decision memo asks for a vasopressor-free washout before the anchor.

Each is an option (1: ``restrict_hospitals``; 2-3: ``exclude_ambiguous_controls``; 4:
``require_washout``) and the counts are reported; 1 applies to positives and controls
alike. Effect in the 2026-10-21 run: 130,484 of 199,028 adult stays kept; dropped 54,608
(75 hospitals: 62 without any vasopressor interval, 13 below 2 %), 470, 6,831 and 6,635
stays for reasons 1-4. Of the 5,232 stays that would otherwise be positives (first start
inside the stay and >= 13 h after admission), 1,516 fall under rule 4 (median lag from
the first medication/treatment evidence to the first charted rate 4.0 h; many have
evidence before unit admission, i.e. arrived on vasopressors) and 15 under rule 1.
Caveat: positives and controls are not matched within hospital (common.py), and the
leave-one-out hospital positive rate alone separates them with AUROC 0.62 in that run.

Numeric features (``events``), canonical units, implausible values dropped
------------------------------------------------------------------------
* ``vitalPeriodic`` (5-min medians from monitors): heartrate -> HR, respiration -> RR,
  sao2 -> SpO2 (pulse oximetry per the eICU docs), temperature -> Temp,
  systemicsystolic/diastolic/mean -> SBP/DBP/MAP with ``modality="invasive"``.
  ``vitalAperiodic`` noninvasivesystolic/diastolic/mean -> SBP/DBP/MAP with
  ``modality="noninvasive"``. pa*/cvp are never used. R-E7: vitalPeriodic 146,671,642
  rows (sorted by stay), HR in 192,277 stays, invasive MAP in 47,682 stays; NIBP in
  189,663 stays. Stay-hours: invasive only 1,241,248, non-invasive only 9,081,252, both
  1,162,670, i.e. 52 % of hours with an arterial line have no cuff pressure (the bias of
  harmonization spec section 3 exists in eICU too; common.py merges per bin preferring
  invasive values).
* Temp also from ``nurseCharting`` (R-E7 extension): vitalPeriodic temperature exists in
  only 19,419 stays (probe-interfaced), while nurseCharting 'Temperature' /
  'Temperature (C)' has plausible values in 187,279 stays (6,267,541 rows, all numeric,
  median 36.9, p1 34.3, p99 39.1; 0.38 % look like Fahrenheit typed into the Celsius
  field and fall outside 25-46 degC, so they are dropped). 'Temperature (F)' rows
  (6,267,330) duplicate the Celsius rows of the same measurement and are not used.
  'Temperature Location' is not used (core/oral/axillary/... are pooled per the
  canonical definition; 'PA CATHETER' is 0.35 % of locations).
* Labs (``lab``; R-E1): exact names only - 'lactate' (mmol/L, 205,591 rows), 'pH' (no
  unit, 413,071), 'creatinine' (mg/dL, 1,277,760), 'glucose' (mg/dL, 1,319,496),
  'bedside glucose' (mg/dL, 3,175,835; INCLUDED by the team decision to keep
  point-of-care glucose in all three databases - DEVIATION from EICU_TABLE, which
  excludes it), 'total bilirubin' (mg/dL, 427,307), 'WBC x 1000' (K/mcL, 1,134,952),
  'platelets x 1000' (K/mcL, 1,149,488). ``labmeasurenamesystem`` has exactly one unit
  per name (the interface unit varies by hospital; the system unit is eICU's
  harmonized one) and every unit is accepted by ``conversion_for``. Rejected
  look-alikes found in the data with these exact spellings: 'urinary creatinine'
  (13,282), 'glucose - CSF' (4,071), "WBC's in urine" (24,453), "WBC's in cerebrospinal
  fluid" (3,331), "WBC's in body fluid" (2,022), "WBC's in pleural fluid" (456),
  "WBC's in peritoneal fluid" (341), "WBC's in synovial fluid" (146), "WBC's in
  pericardial fluid" (47), 'direct bilirubin' (77,732), 'O2 Sat (%)' (340,062). No
  unmapped name looks like a target (paO2, paCO2, FiO2, Total CO2, O2 Content, LPM O2,
  prolactin). Time = ``labresultoffset`` (specimen draw time, as MIMIC ``charttime``);
  revised results: per (stay, labname, labresultoffset) only the rows with the largest
  ``labresultrevisedoffset`` are kept (eicu-code pivoted-lab.sql); only the numeric
  ``labresult`` is used.
* urine_output (``intakeOutput``; R-E3): ``cellvaluenumeric`` (never ``outputtotal``) of
  rows accepted by :func:`harmonization.is_eicu_urine_cellpath` (168 paths, 4,092,561
  rows, per-label means 110-520 mL) plus adapter additions, minus one adapter exclusion
  (see :func:`urine_cellpath_decision`). Added (volumes, urinary): 'Indwelling Catheter
  Output' (143,217 rows, mean 159 mL), 'Voided Amount' (34,967, 336 mL), 'SN Urine
  Output(ml)' (1,233, 305 mL), straight-catheter outputs (~1,640, ~500 mL), nephrostomy
  outputs (~4,900, 90-250 mL), 'Ileal Conduit'/urostomy (~935), 'URETERAL STENT' (320),
  suprapubic (149) and 'Texas Cath' (135); MIMIC counts the same devices. Without them
  hospitals that use these labels would have no urine at all. Excluded on top: 'incontinent
  urine' (100 rows, mean 0.95: a count). Counts, occurrences, ileostomy (stool), mixed
  urine/stool, CBI irrigation, OR urine and ventriculostomy stay excluded. Values outside
  0-5000 mL per record are dropped (46 negative and 1,134 > 5000 mL in the main 'Urine'
  path); 16 exact duplicate rows are dropped. Time = ``intakeoutputoffset`` (chart time;
  entry is up to hours later, p99 592 min).

Ventilation: NOT available
--------------------------
R-E5: ``respiratoryCare.ventendoffset`` is 0 (= not recorded) in 865,224 of 865,381 rows;
the 157 non-zero rows come from a single stay. ``ventstartoffset`` is non-zero in 36,603
stays but without an end it cannot give intervals. R-E4: 'RT Vent On/Off' exists (values
Continued 55,981, Start 2,734, Off 1,016, Suspended 368, off 2) but in only 2,741 stays,
whereas ventilator settings ('PEEP' 45,541 stays, 'Tidal Volume (set)' 37,448,
'Vent Rate' 39,057) show that ventilation is far more common; absence of the label
therefore does not mean "not ventilated". v1's keyword rule hits 18.4 M rows (every
respFlowSettings / respFlowPtVentData row). Per the task rule, ``interventions()``
returns no ventilation rows and ``ventilation_on`` is not in ``available_features()``.
(A rule based on ventilator-settings documentation with a gap rule, like mimic-code's,
would be possible but is not implemented.)

Caching
-------
All raw tables are read once, by streaming (pyarrow CSV reader, about 45 s for the
1.8 GB vitalPeriodic file), for the whole adult base population, into compact parquet
files under ``cache_dir`` (stays, classified infusion rows, intervals, medication/
treatment evidence, and an "event store": canonical, plausibility-filtered values sorted
by stay). The adapter interface does not reveal which stays ``extract_cohort`` will ask
for, so the store covers every adult stay (a superset of any selection); ``events()``
and ``interventions()`` then read only the requested stays' row groups. Re-running with
another gap, seed or sample size costs no raw pass. Each cache part has a key (adapter
version, options it depends on, raw file size and mtime) in ``cache_manifest.json`` and is
rebuilt when the key changes. Measured on 2026-10-21: cold build 2.5-4.5 min (peak working
set 3.9 GiB, in the vitalPeriodic pass), cache 0.84 GB; ``events()`` for 2,000 stays about
2 s and 1.4 GiB; full CLI run 4 min 40 s cold, under 2 min warm.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from src.icu.v2.extract.common import CANONICAL_FEATURES, NUMERIC_FEATURES
from src.icu.v2.harmonization import (
    CANONICAL_SPECS,
    EICU_URINE_CELLPATH_PREFIX,
    EICU_URINE_LABEL_EXCLUSION,
    classify_eicu_drug,
    classify_eicu_lab,
    conversion_for,
    is_eicu_urine_cellpath,
)

#: Calendar origin shared by every stay (offset 0 = unit admission).
BASE_TIME = pd.Timestamp("2000-01-01 00:00:00").as_unit("ns")
#: Bump when a cache file's content or layout changes.
CACHE_VERSION = "eicu-adapter-2026-10-21.2"

#: Headline numbers of the raw-data checks (2026-10-21) behind the decisions above.
RCHECK_EVIDENCE: dict[str, Any] = {
    "R-E1_lab": "7 allowlisted names present with one system unit each (lactate mmol/L 205,591; pH '' 413,071; creatinine "
                "mg/dL 1,277,760; glucose mg/dL 1,319,496; total bilirubin mg/dL 427,307; WBC x 1000 K/mcL 1,134,952; "
                "platelets x 1000 K/mcL 1,149,488); bedside glucose mg/dL 3,175,835; look-alikes rejected: urinary "
                "creatinine 13,282, glucose - CSF 4,071, WBC's in urine/CSF/body/pleural/peritoneal/synovial/pericardial "
                "fluid 30,796, direct bilirubin 77,732, O2 Sat (%) 340,062",
    "R-E2_infusion": "4,803,719 rows; vasopressor names 108 (1,004,292 rows: 904,489 rate>0, 96,433 rate 0, 3,369 no rate); "
                     "drugrate missing => infusionrate missing (0 fallback rows); text rates mostly 'ERROR' (2,831); "
                     "charting gaps per line median 1 h, p90 2 h, p95 4 h, p99 12-14 h; no angiotensin II names",
    "R-E3_urine": "harmonization rule accepts 168 output paths / 4,092,561 rows (label means 110-520 mL); urinary labels it "
                  "misses: 'Indwelling Catheter Output' 143,217 rows (mean 159 mL), 'Voided Amount' 34,967 (336 mL), "
                  "nephrostomy/straight-catheter/conduit/stent/suprapubic/Texas ~8,000; 'incontinent urine' mean 0.95 (count); "
                  "per stay-hour v1 outputtotal sum mean 327 mL (max 55,500) vs v2 urine 247 mL",
    "R-E4_respiratoryCharting": "'RT Vent On/Off' in 2,741 stays (Continued 55,981, Start 2,734, Off 1,016, Suspended 368); "
                                "PEEP charted in 45,541 stays, Tidal Volume (set) 37,448, Vent Rate 39,057; v1 keyword rule "
                                "hits 18.4 M rows (all respFlowSettings/respFlowPtVentData)",
    "R-E5_respiratoryCare": "865,381 rows / 44,772 stays; ventstartoffset non-zero in 36,603 stays; ventendoffset non-zero in "
                            "157 rows (1 stay)",
    "R-E6_patient": "200,859 stays, 139,367 patients, 38,483 patients with > 1 stay (99,975 stays); uniquepid all 'ddd-d+'",
    "R-E7_vitals": "vitalPeriodic 146,671,642 rows sorted by stay; HR in 192,277 stays, temperature in 19,419, invasive MAP "
                   "in 47,682; NIBP in 189,663 stays; stay-hours invasive-only 1,241,248 / NIBP-only 9,081,252 / both "
                   "1,162,670; nurseCharting 'Temperature (C)' 6,267,541 rows, plausible in 187,279 stays",
    "infusion_charting_by_hospital": "56 of 208 hospitals (44,014 adult stays) have no infusionDrug row; per-hospital share "
                                     "of stays with a vasopressor infusion: median 13.7 %, 13 hospitals < 2 %",
    "vasopressor_evidence_elsewhere": "16,164 adult stays without a vasopressor infusion row have vasopressor medication "
                                      "orders or treatment entries",
}

RAW_FILES: dict[str, str] = {
    "patient": "patient.csv.gz",
    "infusion": "infusionDrug.csv.gz",
    "vital_periodic": "vitalPeriodic.csv.gz",
    "vital_aperiodic": "vitalAperiodic.csv.gz",
    "lab": "lab.csv.gz",
    "intake_output": "intakeOutput.csv.gz",
    "nurse_charting": "nurseCharting.csv.gz",
    "medication": "medication.csv.gz",
    "treatment": "treatment.csv.gz",
}

# --------------------------------------------------------------------------------------
# Drug classification
# --------------------------------------------------------------------------------------

#: Canonical vasopressor drugs and the name patterns that identify them (the patterns of
#: the harmonization include rows, plus angiotensin II as requested for the task).
VASOPRESSOR_DRUGS: dict[str, str] = {
    "norepinephrine": r"\bnor-?epinephrine\b|\bnorepi\b|\blevophed\b|\blevo\b|\bnoradrenaline\b",
    "epinephrine": r"\bepinephrine\b|\bepinepherine\b|\bepi\b|\badrenaline?\b",
    "phenylephrine": r"\bphenylephrine\b|\bneo-? ?synephrine\b|\bneosynsprine\b",
    "vasopressin": r"\bvasopressin\b|\bvasostrict\b|\bvaso\b",
    "dopamine": r"\bdopamine\b|\bino?tropin\b",
    "angiotensin_ii": r"\bangiotensin[\s-]*(?:ii|2)\b|\bgiapreza\b",
}
#: Sedative drugs that can be selected for ``sedation_on``.
SEDATIVE_DRUGS: dict[str, str] = {
    "propofol": r"\bpropofol\b|\bdiprivan\b",
    "midazolam": r"\bmidazolam\b|\bversed\b",
    "dexmedetomidine": r"\bdexmedetomidine\b|\bdexmetetomidine\b|\bprecedex\b",
    "lorazepam": r"\blorazepam\b|\bativan\b",
    "fentanyl": r"\bfentanyl\b|\bsublimaze\b",
}
#: Task instruction for the v2 extraction (differs from harmonization: see module doc).
DEFAULT_SEDATION_DRUGS: tuple[str, ...] = ("propofol", "midazolam", "dexmedetomidine", "lorazepam")

#: Epidural / regional preparations the harmonization patterns miss ("fentanyl epi",
#: misspelled "bupivicaine", "PCEA").
_REGIONAL = re.compile(r"\bbupi|\bropiv|pcea|\bpca\b|epidural|duramorph", re.IGNORECASE)
_EPI_ABBREV = re.compile(r"\bepi\b", re.IGNORECASE)
_OPIOID_OR_LOCAL = re.compile(r"fent|bup|ropi|morph|dilaud|sufent|lido", re.IGNORECASE)


def _match_drugs(text: str, table: Mapping[str, str]) -> list[str]:
    return [drug for drug, pattern in table.items() if re.search(pattern, text, re.IGNORECASE)]


def classify_infusion(drugname: object, sedation_drugs: Sequence[str] = DEFAULT_SEDATION_DRUGS) -> tuple[str | None, str, str]:
    """Classify an ``infusionDrug.drugname``.

    Returns ``(feature, drug, reason)`` with feature ``vasopressor_on``, ``sedation_on`` or
    ``None``; ``drug`` is the canonical drug (``a+b`` for mixtures, ``""`` when unmapped).
    """
    if drugname is None or (isinstance(drugname, float) and np.isnan(drugname)):
        return None, "", "unmapped"
    text = re.sub(r"\s+", " ", str(drugname)).strip()
    if not text:
        return None, "", "unmapped"
    if _REGIONAL.search(text) or (_EPI_ABBREV.search(text) and _OPIOID_OR_LOCAL.search(text)):
        return None, "", "excluded: epidural/regional preparation (adapter rule)"
    angiotensin = re.search(VASOPRESSOR_DRUGS["angiotensin_ii"], text, re.IGNORECASE)
    if angiotensin and not re.search(r"receptor|blocker|\barb\b|\bace\b", text, re.IGNORECASE):
        return "vasopressor_on", "angiotensin_ii", "accepted: angiotensin II (adapter rule, task instruction)"
    feature, reason = classify_eicu_drug(text)
    if feature == "vasopressor_on":
        drugs = [d for d in _match_drugs(text, VASOPRESSOR_DRUGS) if d != "angiotensin_ii"]
        return feature, "+".join(drugs) or "unspecified", reason
    sedatives = [d for d in _match_drugs(text, SEDATIVE_DRUGS) if d in set(sedation_drugs)]
    if feature == "sedation_on":
        if sedatives:
            return feature, "+".join(sedatives), reason
        return None, "", "excluded: sedative outside the configured sedation set"
    if sedatives and reason == "excluded: outside the canonical drug set":
        # e.g. lorazepam: harmonization lists it as "other sedatives", the task set includes it
        return "sedation_on", "+".join(sedatives), "accepted: configured sedation set (adapter rule)"
    return None, "", reason


_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_OFF = re.compile(r"^(?:off|0ff)\b", re.IGNORECASE)
_BR = re.compile(r"(?:\\\.br\\)+", re.IGNORECASE)


def parse_rate(value: object) -> float:
    """Infusion rate from a raw ``drugrate``/``infusionrate`` string.

    Plain numbers are returned (negative ones are invalid -> NaN); "OFF"-like text means a
    documented stop (0.0); anything else is unparseable (NaN, the row is dropped).
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) and number >= 0 else np.nan
    text = _BR.sub("", str(value)).strip()
    if not text:
        return np.nan
    if _NUMBER.match(text):
        number = float(text)
        return number if number >= 0 else np.nan
    if _OFF.match(text):
        return 0.0
    return np.nan


def parse_rates(values: pd.Series) -> pd.Series:
    """Vectorized :func:`parse_rate` (numbers fast-pathed, text parsed per distinct value)."""
    numeric = pd.to_numeric(values, errors="coerce")
    out = numeric.where(numeric >= 0)
    text_mask = values.notna() & numeric.isna()
    if text_mask.any():
        distinct = {v: parse_rate(v) for v in values[text_mask].unique()}
        out[text_mask] = values[text_mask].map(distinct)
    return out.astype(float)


def derive_intervals(rows: pd.DataFrame, discharge_offsets: Mapping[int, float], max_gap_minutes: float) -> pd.DataFrame:
    """[start, end) runs of an infusion line from its charted rates.

    ``rows`` has stay_id, key (the flowsheet line, i.e. drugname), offset (minutes) and a
    parsed, non-missing ``rate``. Rows at the same offset are merged with the max rate. A
    run starts at a row with rate > 0 and ends at the next row with rate == 0, at
    ``last on-row + max_gap`` when the next row is more than ``max_gap`` later (or there is
    none), and never after unit discharge. Returns stay_id, key, start_offset, end_offset,
    end_reason (``rate_zero``, ``max_gap``, ``last_row``, ``unit_discharge``).
    """
    columns = ["stay_id", "key", "start_offset", "end_offset", "end_reason"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    agg = (
        rows.groupby(["stay_id", "key", "offset"], as_index=False, sort=True)["rate"].max()
    )
    stay = agg["stay_id"].to_numpy(np.int64)
    key = agg["key"].to_numpy(object)
    t = agg["offset"].to_numpy(float)
    r = agg["rate"].to_numpy(float)
    n = len(agg)
    new = np.ones(n, dtype=bool)
    new[1:] = (stay[1:] != stay[:-1]) | (key[1:] != key[:-1])
    bounds = np.r_[np.flatnonzero(new), n]
    gap = float(max_gap_minutes)
    out: list[tuple[int, object, float, float, str]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        on = False
        start = last_on = 0.0
        for j in range(a, b):
            tj, rj = t[j], r[j]
            if on:
                if tj - last_on > gap:
                    out.append((stay[a], key[a], start, last_on + gap, "max_gap"))
                    on = False
                elif rj <= 0:
                    out.append((stay[a], key[a], start, tj, "rate_zero"))
                    on = False
                    continue
                else:
                    last_on = tj
                    continue
            if rj > 0:
                on = True
                start = last_on = tj
        if on:
            out.append((stay[a], key[a], start, last_on + gap, "last_row"))
    iv = pd.DataFrame(out, columns=columns)
    if iv.empty:
        return iv
    discharge = iv["stay_id"].map(discharge_offsets).astype(float)
    capped = discharge.notna() & (iv["end_offset"] > discharge)
    iv.loc[capped, "end_offset"] = discharge[capped]
    iv.loc[capped, "end_reason"] = "unit_discharge"
    iv = iv[iv["end_offset"] > iv["start_offset"]]
    return iv.sort_values(["stay_id", "start_offset", "key"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Labs, urine, identifiers, time
# --------------------------------------------------------------------------------------


def classify_lab(labname: object, include_bedside_glucose: bool = True) -> tuple[str | None, str]:
    """:func:`harmonization.classify_eicu_lab` plus the bedside-glucose decision."""
    feature, reason = classify_eicu_lab(labname)
    if feature is None and include_bedside_glucose and re.sub(r"\s+", " ", str(labname or "")).strip().casefold() == "bedside glucose":
        return "Glucose", "accepted: point-of-care glucose (team decision: included in all databases)"
    return feature, reason


#: Urinary output labels missing from the eicu-code list (R-E3, see module doc).
URINE_LABEL_ADDITIONS = re.compile(
    r"^indwelling catheter output$|^voided amount$|^sn urine output\(ml\)$|straight cath|nephrostomy|"
    r"ileal conduit|urostomy|ureteral stent|suprapubic|texas cath"
)
#: Labels whose values are counts although harmonization accepts them (R-E3).
URINE_LABEL_EXCLUSIONS = re.compile(r"incontinen|occurrence|\bcount|ileostomy|stool|irrigat|\bcbi\b|^or urine|ventric")


def urine_cellpath_decision(cellpath: object) -> tuple[bool, str]:
    """``(is_urine, reason)`` for an ``intakeOutput.cellpath``."""
    if cellpath is None or (isinstance(cellpath, float) and np.isnan(cellpath)):
        return False, "missing"
    text = str(cellpath)
    if not text.startswith(EICU_URINE_CELLPATH_PREFIX):
        return False, "not an output (ml) path"
    label = re.sub(r"\s+", " ", text[len(EICU_URINE_CELLPATH_PREFIX):]).strip().casefold()
    if URINE_LABEL_EXCLUSIONS.search(label):
        return False, "excluded: count/stool/irrigation/intra-operative label (adapter rule)"
    if is_eicu_urine_cellpath(text):
        return True, "accepted: harmonization rule"
    if EICU_URINE_LABEL_EXCLUSION.search(label):
        return False, "excluded: harmonization exclusion"
    if URINE_LABEL_ADDITIONS.search(label):
        return True, "accepted: urinary device label (adapter rule)"
    return False, "not urine"


def subject_id_from_uniquepid(uniquepid: object) -> int:
    """Stable positive int64 patient id: first 15 hex digits of sha1(uniquepid)."""
    text = str(uniquepid).strip()
    if not text or text.lower() == "nan":
        raise ValueError("uniquepid is missing")
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:15], 16)


def adult_mask(age: pd.Series) -> pd.Series:
    """True for age >= 18 or the string '> 89'; missing or non-numeric ages are False."""
    text = age.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    return ((numeric >= 18) | (text == "> 89")).fillna(False).astype(bool)


def offsets_to_time(offsets: object) -> np.ndarray:
    """Minutes from unit admission -> datetime64[ns] on the shared calendar origin."""
    minutes = np.asarray(offsets, dtype=float)
    nanos = np.round(minutes * 60_000_000_000).astype(np.int64).view("timedelta64[ns]")
    return BASE_TIME.to_datetime64() + nanos


def _plausible(feature: str, values: np.ndarray) -> np.ndarray:
    spec = CANONICAL_SPECS[feature]
    return np.isfinite(values) & (values >= spec.low) & (values <= spec.high)


# --------------------------------------------------------------------------------------
# Raw reading helpers
# --------------------------------------------------------------------------------------


def _stream_csv(path: Path, columns: Mapping[str, pa.DataType], block_size: int) -> Iterator[pa.Table]:
    convert = pcsv.ConvertOptions(include_columns=list(columns), column_types=dict(columns), strings_can_be_null=True)
    reader = pcsv.open_csv(
        pa.input_stream(str(path), compression="gzip"),
        read_options=pcsv.ReadOptions(block_size=block_size),
        convert_options=convert,
    )
    for batch in reader:
        yield pa.Table.from_batches([batch])


def _write_parquet(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=1_000_000)
    os.replace(tmp, path)


def _is_sorted(values: np.ndarray) -> bool:
    return bool(values.size < 2 or np.all(values[1:] >= values[:-1]))


#: Store layout: name -> (columns that become features, modality)
_VITAL_PERIODIC = {
    "heartrate": ("HR", ""),
    "respiration": ("RR", ""),
    "sao2": ("SpO2", ""),
    "temperature": ("Temp", ""),
    "systemicsystolic": ("SBP", "invasive"),
    "systemicdiastolic": ("DBP", "invasive"),
    "systemicmean": ("MAP", "invasive"),
}
_VITAL_APERIODIC = {
    "noninvasivesystolic": ("SBP", "noninvasive"),
    "noninvasivediastolic": ("DBP", "noninvasive"),
    "noninvasivemean": ("MAP", "noninvasive"),
}
_MODALITIES = (None, "invasive", "noninvasive")
_FEATURE_CODES = {feature: i for i, feature in enumerate(NUMERIC_FEATURES)}

#: Vasopressor evidence outside infusionDrug (for the cohort restrictions).
_EVIDENCE_DRUGS = r"norepinephrine|levophed|vasopressin|pitressin|phenylephrine|neo-?synephrine|epinephrine|adrenalin|dopamine"
_MEDICATION_NOT_SYSTEMIC = (
    r"lidocaine|xylocaine|bupivacaine|marcaine|sensorcaine|ropivacaine|articaine|septocaine|carbocaine|racemic|inhal|"
    r"nebul|ophth|nasal|epipen|auto-?inj"
)
_MEDICATION_DRIP_NAME = (
    r"\bin\b.*\d+\s*ml|/\s*\d+\s*ml|\bns\b|d5w|dextrose|sodium chloride|infusion|drip|premix|gtt|mcg/min|mcg/kg/min|units/min"
)
_MEDICATION_DRIP_FREQUENCY = r"contin|titrat|drip|infus|gtt"
_TREATMENT_VASOPRESSOR = r"\|vasopressors|norepinephrine|epinephrine|vasopressin|phenylephrine|dopamine"


class EICUAdapter:
    """eICU-CRD 2.0 adapter for :func:`src.icu.v2.extract.common.extract_cohort`."""

    name = "eicu"

    def __init__(
        self,
        raw_dir: str | Path,
        cache_dir: str | Path,
        *,
        max_gap_hours: float = 4.0,
        sedation_drugs: Sequence[str] = DEFAULT_SEDATION_DRUGS,
        include_bedside_glucose: bool = True,
        nursecharting_temperature: bool = True,
        restrict_hospitals: bool = True,
        min_hospital_vasopressor_share: float = 0.02,
        exclude_ambiguous_controls: bool = True,
        require_washout: bool = True,
        prior_evidence_tolerance_minutes: float = 60.0,
        rebuild: bool = False,
        block_size: int = 64 << 20,
        verbose: bool = True,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir)
        unknown = [d for d in sedation_drugs if d not in SEDATIVE_DRUGS]
        if unknown:
            raise ValueError(f"unknown sedation drugs {unknown}; choose from {sorted(SEDATIVE_DRUGS)}")
        if max_gap_hours <= 0:
            raise ValueError("max_gap_hours must be positive")
        self.options: dict[str, Any] = {
            "max_gap_hours": float(max_gap_hours),
            "sedation_drugs": list(sedation_drugs),
            "include_bedside_glucose": bool(include_bedside_glucose),
            "nursecharting_temperature": bool(nursecharting_temperature),
            "restrict_hospitals": bool(restrict_hospitals),
            "min_hospital_vasopressor_share": float(min_hospital_vasopressor_share),
            "exclude_ambiguous_controls": bool(exclude_ambiguous_controls),
            "require_washout": bool(require_washout),
            "prior_evidence_tolerance_minutes": float(prior_evidence_tolerance_minutes),
        }
        self.rebuild = bool(rebuild)
        self.block_size = int(block_size)
        self.verbose = bool(verbose)
        self._rebuilt: set[str] = set()
        self._stays: pd.DataFrame | None = None
        self._cohort_counts: dict[str, Any] | None = None
        self._vaso: pd.DataFrame | None = None
        self._sed: pd.DataFrame | None = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ cache plumbing

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[eicu {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

    def _raw(self, key: str) -> Path:
        return self.raw_dir / RAW_FILES[key]

    def _cache(self, name: str) -> Path:
        return self.cache_dir / name

    def _manifest_path(self) -> Path:
        return self._cache("cache_manifest.json")

    def _load_manifest(self) -> dict[str, Any]:
        path = self._manifest_path()
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _part_key(self, part: str, raw_keys: Sequence[str], options: Mapping[str, Any]) -> str:
        files = []
        for key in raw_keys:
            path = self._raw(key)
            stat = path.stat() if path.exists() else None
            files.append([RAW_FILES[key], stat.st_size if stat else None, stat.st_mtime_ns if stat else None])
        payload = json.dumps([CACHE_VERSION, part, files, dict(options)], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ensure(self, part: str, outputs: Sequence[str], raw_keys: Sequence[str], options: Mapping[str, Any],
                builder: Callable[[], dict[str, Any]], depends: Sequence[str] = ()) -> dict[str, Any]:
        """Build ``part`` unless its outputs exist with a matching key; returns its stats."""
        dep_keys = {}
        manifest = self._load_manifest()
        for dep in depends:
            dep_keys[dep] = manifest.get(dep, {}).get("key")
        key = self._part_key(part, raw_keys, {**options, "_depends": dep_keys})
        entry = manifest.get(part)
        fresh = entry is not None and entry.get("key") == key and all(self._cache(o).exists() for o in outputs)
        if fresh and not (self.rebuild and part not in self._rebuilt):
            return entry.get("stats", {})
        self._log(f"building cache part '{part}' ...")
        started = time.perf_counter()
        stats = builder()
        manifest = self._load_manifest()
        manifest[part] = {
            "key": key,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.perf_counter() - started, 1),
            "outputs": list(outputs),
            "stats": stats,
        }
        tmp = self._manifest_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, self._manifest_path())
        self._rebuilt.add(part)
        self._log(f"built '{part}' in {manifest[part]['seconds']} s")
        return stats

    # ------------------------------------------------------------------ base stays

    def _base_part(self) -> dict[str, Any]:
        return self._ensure("base_stays", ["base_stays.parquet"], ["patient"], {}, self._build_base_stays)

    def _build_base_stays(self) -> dict[str, Any]:
        cols = ["patientunitstayid", "uniquepid", "age", "unitdischargeoffset", "hospitalid"]
        p = pd.read_csv(self._raw("patient"), usecols=cols, dtype={"uniquepid": str, "age": str})
        n_all = len(p)
        if p["patientunitstayid"].duplicated().any():
            raise ValueError("patientunitstayid is not unique in patient.csv.gz")
        adult = adult_mask(p["age"])
        age_missing = p["age"].isna() | (p["age"].astype("string").str.strip() == "")
        los_ok = pd.to_numeric(p["unitdischargeoffset"], errors="coerce") > 0
        pid_ok = p["uniquepid"].notna() & (p["uniquepid"].astype("string").str.strip() != "")
        keep = adult & los_ok & pid_ok
        base = p[keep].copy()
        ids = {u: subject_id_from_uniquepid(u) for u in base["uniquepid"].unique()}
        if len(set(ids.values())) != len(ids):
            raise ValueError("subject_id hash collision; use more hex digits")
        base = pd.DataFrame(
            {
                "stay_id": base["patientunitstayid"].astype(np.int64),
                "subject_id": base["uniquepid"].map(ids).astype(np.int64),
                "hospitalid": pd.to_numeric(base["hospitalid"], errors="coerce").fillna(-1).astype(np.int64),
                "unitdischargeoffset": base["unitdischargeoffset"].astype(np.int64),
            }
        ).sort_values("stay_id").reset_index(drop=True)
        _write_parquet(pa.Table.from_pandas(base, preserve_index=False), self._cache("base_stays.parquet"))
        per = p.groupby("uniquepid")["patientunitstayid"].nunique()
        return {
            "patient_rows": int(n_all),
            "patients": int(per.size),
            "patients_with_multiple_stays": int((per > 1).sum()),
            "minors": int((~adult & ~age_missing).sum()),
            "age_missing": int(age_missing.sum()),
            "age_gt_89": int((p["age"].astype("string").str.strip() == "> 89").sum()),
            "los_not_positive": int((~los_ok).sum()),
            "uniquepid_missing": int((~pid_ok).sum()),
            "base_stays": int(len(base)),
            "base_patients": int(base["subject_id"].nunique()),
            "hospitals": int(base["hospitalid"].nunique()),
        }

    def _base(self) -> pd.DataFrame:
        self._base_part()
        return pd.read_parquet(self._cache("base_stays.parquet"))

    # ------------------------------------------------------------------ infusions

    def _infusion_options(self) -> dict[str, Any]:
        return {"sedation_drugs": self.options["sedation_drugs"]}

    def _infusion_part(self) -> dict[str, Any]:
        self._base_part()
        return self._ensure(
            "infusions", ["infusion_rows.parquet", "infusion_names.parquet"], ["infusion"], self._infusion_options(),
            self._build_infusions, depends=["base_stays"],
        )

    def _build_infusions(self) -> dict[str, Any]:
        base_ids = set(self._base()["stay_id"].tolist())
        cols = ["patientunitstayid", "infusionoffset", "drugname", "drugrate", "infusionrate"]
        df = pd.read_csv(self._raw("infusion"), usecols=cols, dtype={"drugname": str, "drugrate": str, "infusionrate": str})
        names = pd.Series(df["drugname"].dropna().unique())
        verdicts = {n: classify_infusion(n, self.options["sedation_drugs"]) for n in names}
        table = pd.DataFrame(
            [(n, *verdicts[n]) for n in names], columns=["drugname", "feature", "drug", "reason"]
        )
        counts = df["drugname"].value_counts()
        table["n_rows"] = table["drugname"].map(counts).fillna(0).astype(np.int64)
        df["feature"] = df["drugname"].map({n: v[0] for n, v in verdicts.items()})
        df["drug"] = df["drugname"].map({n: v[1] for n, v in verdicts.items()})
        rows = df[df["feature"].notna()].copy()
        drug_rate = parse_rates(rows["drugrate"])
        infusion_rate = parse_rates(rows["infusionrate"])
        rows["rate"] = drug_rate.where(drug_rate.notna(), infusion_rate)
        rows["rate_source"] = np.where(drug_rate.notna(), "drugrate", np.where(infusion_rate.notna(), "infusionrate", "none"))
        rows["rate_text"] = rows["drugrate"].notna() & pd.to_numeric(rows["drugrate"], errors="coerce").isna()
        rows = rows[rows["patientunitstayid"].isin(base_ids)]
        out = pd.DataFrame(
            {
                "stay_id": rows["patientunitstayid"].astype(np.int64),
                "offset": rows["infusionoffset"].astype(np.int64),
                "drugname": rows["drugname"].astype(str),
                "feature": rows["feature"].astype(str),
                "drug": rows["drug"].astype(str),
                "rate": rows["rate"].astype(float),
                "rate_source": rows["rate_source"].astype(str),
                "rate_text": rows["rate_text"].astype(bool),
            }
        ).sort_values(["stay_id", "offset"]).reset_index(drop=True)
        _write_parquet(pa.Table.from_pandas(out, preserve_index=False), self._cache("infusion_rows.parquet"))
        _write_parquet(pa.Table.from_pandas(table.sort_values("n_rows", ascending=False), preserve_index=False),
                       self._cache("infusion_names.parquet"))
        stats: dict[str, Any] = {"raw_rows": int(len(df)), "rows_in_base": int(len(out))}
        for feature in ("vasopressor_on", "sedation_on"):
            sub = out[out["feature"] == feature]
            stats[feature] = {
                "names": int((table["feature"] == feature).sum()),
                "rows": int(len(sub)),
                "rate_positive": int((sub["rate"] > 0).sum()),
                "rate_zero": int((sub["rate"] == 0).sum()),
                "rate_missing_or_unparseable": int(sub["rate"].isna().sum()),
                "rate_text_rows": int(sub["rate_text"].sum()),
                "rate_text_read_as_stop": int((sub["rate_text"] & (sub["rate"] == 0)).sum()),
                "rate_from_infusionrate": int((sub["rate_source"] == "infusionrate").sum()),
                "rows_by_drug": {k: int(v) for k, v in sub["drug"].value_counts().items()},
            }
        harmonization_accepts = table["drugname"].map(lambda n: classify_eicu_drug(n)[0] is not None)
        stats["adapter_excluded_names"] = {
            n: int(c)
            for n, c in table.loc[table["feature"].isna() & harmonization_accepts, ["drugname", "n_rows"]].itertuples(index=False)
        }
        stats["adapter_added_names"] = {
            n: int(c)
            for n, c in table.loc[table["feature"].notna() & ~harmonization_accepts, ["drugname", "n_rows"]].itertuples(index=False)
        }
        return stats

    # ------------------------------------------------------------------ intervals

    def _interval_part(self) -> dict[str, Any]:
        self._infusion_part()
        return self._ensure(
            "intervals", ["vasopressor_intervals.parquet", "sedation_intervals.parquet"], [],
            {"max_gap_hours": self.options["max_gap_hours"]}, self._build_intervals, depends=["base_stays", "infusions"],
        )

    def _build_intervals(self) -> dict[str, Any]:
        base = self._base()
        discharge = dict(zip(base["stay_id"].tolist(), base["unitdischargeoffset"].astype(float).tolist()))
        rows = pd.read_parquet(self._cache("infusion_rows.parquet"))
        stats: dict[str, Any] = {}
        for feature, file in (("vasopressor_on", "vasopressor_intervals.parquet"), ("sedation_on", "sedation_intervals.parquet")):
            sub = rows[(rows["feature"] == feature) & rows["rate"].notna()]
            drug_of = dict(zip(sub["drugname"], sub["drug"]))
            iv = derive_intervals(
                sub.rename(columns={"drugname": "key"})[["stay_id", "key", "offset", "rate"]],
                discharge,
                self.options["max_gap_hours"] * 60.0,
            )
            iv["drug"] = iv["key"].map(drug_of).astype(str)
            iv = iv.rename(columns={"key": "drugname"})
            _write_parquet(pa.Table.from_pandas(iv, preserve_index=False), self._cache(file))
            dur = (iv["end_offset"] - iv["start_offset"]) / 60.0
            stats[feature] = {
                "intervals": int(len(iv)),
                "stays": int(iv["stay_id"].nunique()),
                "end_reason": {k: int(v) for k, v in iv["end_reason"].value_counts().items()},
                "duration_hours_quantiles": {str(q): round(float(dur.quantile(q)), 2) for q in (0.1, 0.5, 0.9)} if len(iv) else {},
            }
        return stats

    def _intervals(self, feature: str) -> pd.DataFrame:
        self._interval_part()
        file = "vasopressor_intervals.parquet" if feature == "vasopressor_on" else "sedation_intervals.parquet"
        return pd.read_parquet(self._cache(file))

    # ------------------------------------------------------------------ other vasopressor evidence

    def _evidence_part(self) -> dict[str, Any]:
        self._base_part()
        return self._ensure(
            "evidence", ["vasopressor_evidence.parquet"], ["medication", "treatment"], {}, self._build_evidence,
            depends=["base_stays"],
        )

    def _build_evidence(self) -> dict[str, Any]:
        base_ids = pa.array(self._base()["stay_id"].to_numpy(np.int32))
        frames = []
        stats: dict[str, Any] = {}
        med_path = self._raw("medication")
        if med_path.exists():
            cols = {"patientunitstayid": pa.int32(), "drugorderoffset": pa.float64(), "drugstartoffset": pa.float64(),
                    "drugordercancelled": pa.string(), "drugname": pa.string(), "frequency": pa.string()}
            parts = []
            for t in _stream_csv(med_path, cols, self.block_size):
                name = pc.fill_null(t["drugname"], "")
                m = pc.and_(pc.match_substring_regex(name, _EVIDENCE_DRUGS, ignore_case=True),
                            pc.invert(pc.match_substring_regex(name, _MEDICATION_NOT_SYSTEMIC, ignore_case=True)))
                m = pc.and_(m, pc.is_in(t["patientunitstayid"], value_set=base_ids))
                parts.append(t.filter(m).to_pandas())
            med = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(cols))
            cancelled = med["drugordercancelled"].fillna("No").str.strip().str.lower() == "yes"
            drip = med["drugname"].fillna("").str.contains(_MEDICATION_DRIP_NAME, case=False, regex=True) | med["frequency"].fillna(
                "").str.contains(_MEDICATION_DRIP_FREQUENCY, case=False, regex=True)
            use = med[~cancelled & drip]
            offset = use["drugstartoffset"].where(use["drugstartoffset"].notna(), use["drugorderoffset"])
            frames.append(pd.DataFrame({"stay_id": use["patientunitstayid"].astype(np.int64), "offset": offset, "source": "medication"}))
            stats["medication"] = {"vasopressor_name_rows": int(len(med)), "cancelled": int(cancelled.sum()),
                                   "drip_like_used": int(len(use)), "stays": int(use["patientunitstayid"].nunique())}
        else:
            stats["medication"] = "file missing: no medication evidence"
        tr_path = self._raw("treatment")
        if tr_path.exists():
            cols = {"patientunitstayid": pa.int32(), "treatmentoffset": pa.float64(), "treatmentstring": pa.string()}
            parts = []
            for t in _stream_csv(tr_path, cols, self.block_size):
                m = pc.match_substring_regex(pc.fill_null(t["treatmentstring"], ""), _TREATMENT_VASOPRESSOR, ignore_case=True)
                m = pc.and_(m, pc.is_in(t["patientunitstayid"], value_set=base_ids))
                parts.append(t.filter(m).to_pandas())
            tr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(cols))
            frames.append(pd.DataFrame({"stay_id": tr["patientunitstayid"].astype(np.int64), "offset": tr["treatmentoffset"], "source": "treatment"}))
            stats["treatment"] = {"rows": int(len(tr)), "stays": int(tr["patientunitstayid"].nunique())}
        else:
            stats["treatment"] = "file missing: no treatment evidence"
        ev = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["stay_id", "offset", "source"])
        ev = ev.dropna(subset=["offset"])
        if len(ev):
            out = ev.groupby("stay_id").agg(
                first_evidence_offset=("offset", "min"),
                n_medication=("source", lambda s: int((s == "medication").sum())),
                n_treatment=("source", lambda s: int((s == "treatment").sum())),
            ).reset_index()
        else:
            out = pd.DataFrame(columns=["stay_id", "first_evidence_offset", "n_medication", "n_treatment"])
        out["stay_id"] = out["stay_id"].astype(np.int64)
        out["first_evidence_offset"] = out["first_evidence_offset"].astype(float)
        _write_parquet(pa.Table.from_pandas(out, preserve_index=False), self._cache("vasopressor_evidence.parquet"))
        stats["stays_with_evidence"] = int(len(out))
        return stats

    # ------------------------------------------------------------------ cohort

    def _cohort(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        if self._stays is not None and self._cohort_counts is not None:
            return self._stays, self._cohort_counts
        base = self._base()
        vaso = self._intervals("vasopressor_on")  # also builds the infusion rows
        rows = pd.read_parquet(self._cache("infusion_rows.parquet"), columns=["stay_id", "feature"])
        first = vaso.groupby("stay_id")["start_offset"].min()
        audit = base.copy()
        audit["first_vaso_offset"] = audit["stay_id"].map(first)
        audit["has_interval"] = audit["first_vaso_offset"].notna()
        vaso_row_stays = set(rows.loc[rows["feature"] == "vasopressor_on", "stay_id"].unique())
        audit["vaso_rows_without_interval"] = audit["stay_id"].isin(vaso_row_stays) & ~audit["has_interval"]
        if self.options["exclude_ambiguous_controls"] or self.options["require_washout"]:
            self._evidence_part()
            ev = pd.read_parquet(self._cache("vasopressor_evidence.parquet"))
            audit["first_evidence_offset"] = audit["stay_id"].map(dict(zip(ev["stay_id"], ev["first_evidence_offset"])))
        else:
            audit["first_evidence_offset"] = np.nan
        share = audit.groupby("hospitalid")["has_interval"].mean()
        audit["hospital_vaso_share"] = audit["hospitalid"].map(share)
        reason = pd.Series("", index=audit.index, dtype=object)
        if self.options["restrict_hospitals"]:
            low = audit["hospital_vaso_share"] < self.options["min_hospital_vasopressor_share"]
            reason[low] = "hospital_without_vasopressor_charting"
        if self.options["exclude_ambiguous_controls"]:
            free = reason == ""
            r2 = free & audit["vaso_rows_without_interval"]
            reason[r2] = "vasopressor_rows_without_rate"
            free = reason == ""
            r3 = free & ~audit["has_interval"] & audit["first_evidence_offset"].notna()
            reason[r3] = "vasopressor_evidence_without_infusion"
        if self.options["require_washout"]:
            tol = self.options["prior_evidence_tolerance_minutes"]
            free = reason == ""
            r4 = free & audit["has_interval"] & (audit["first_evidence_offset"] < audit["first_vaso_offset"] - tol)
            reason[r4] = "vasopressor_evidence_before_first_infusion"
        audit["excluded_reason"] = reason
        _write_parquet(pa.Table.from_pandas(audit, preserve_index=False), self._cache("cohort_audit.parquet"))
        kept = audit[audit["excluded_reason"] == ""]
        stays = pd.DataFrame(
            {
                "subject_id": kept["subject_id"].astype(np.int64).to_numpy(),
                "stay_id": kept["stay_id"].astype(np.int64).to_numpy(),
                "intime": np.full(len(kept), BASE_TIME.to_datetime64()).astype("datetime64[ns]"),
                "outtime": offsets_to_time(kept["unitdischargeoffset"].to_numpy()),
                # extra column: lets common.match_controls match controls within hospital
                "hospitalid": kept["hospitalid"].astype(np.int64).to_numpy(),
            }
        ).reset_index(drop=True)
        hospitals = audit.groupby("hospitalid")["hospital_vaso_share"].first()
        counts: dict[str, Any] = {
            "base_adult_stays": int(len(audit)),
            "base_patients": int(audit["subject_id"].nunique()),
            "base_stays_with_vasopressor_interval": int(audit["has_interval"].sum()),
            "excluded": {k: int(v) for k, v in audit.loc[audit["excluded_reason"] != "", "excluded_reason"].value_counts().items()},
            "excluded_with_interval": {
                k: int(v) for k, v in audit.loc[(audit["excluded_reason"] != "") & audit["has_interval"], "excluded_reason"].value_counts().items()
            },
            "hospitals_total": int(len(hospitals)),
            "hospitals_kept": int(kept["hospitalid"].nunique()),
            "hospitals_dropped_no_infusion_charting": int((hospitals < self.options["min_hospital_vasopressor_share"]).sum())
            if self.options["restrict_hospitals"] else 0,
            "cohort_stays": int(len(stays)),
            "cohort_patients": int(stays["subject_id"].nunique()),
            "cohort_stays_with_vasopressor_interval": int(kept["has_interval"].sum()),
        }
        self._stays, self._cohort_counts = stays, counts
        return stays, counts

    # ------------------------------------------------------------------ event store

    def _store_parts(self) -> dict[str, dict[str, Any]]:
        self._base_part()
        stats = {
            "vital_periodic": self._ensure("store_vital_periodic", ["store_vital_periodic.parquet"], ["vital_periodic"], {},
                                           self._build_vital_periodic, depends=["base_stays"]),
            "vital_aperiodic": self._ensure("store_vital_aperiodic", ["store_vital_aperiodic.parquet"], ["vital_aperiodic"], {},
                                            self._build_vital_aperiodic, depends=["base_stays"]),
            "lab": self._ensure("store_lab", ["store_lab.parquet"], ["lab"],
                                {"include_bedside_glucose": self.options["include_bedside_glucose"]}, self._build_lab,
                                depends=["base_stays"]),
            "urine": self._ensure("store_urine", ["store_urine.parquet"], ["intake_output"], {}, self._build_urine,
                                  depends=["base_stays"]),
        }
        if self.options["nursecharting_temperature"]:
            stats["nurse_temperature"] = self._ensure(
                "store_nurse_temperature", ["store_nurse_temperature.parquet"], ["nurse_charting"], {},
                self._build_nurse_temperature, depends=["base_stays"],
            )
        return stats

    def _base_id_array(self) -> pa.Array:
        return pa.array(self._base()["stay_id"].to_numpy(np.int32))

    def _build_wide_vitals(self, raw_key: str, mapping: Mapping[str, tuple[str, str]], out_name: str) -> dict[str, Any]:
        base_ids = self._base_id_array()
        cols: dict[str, pa.DataType] = {"patientunitstayid": pa.int32(), "observationoffset": pa.int32()}
        cols.update({c: pa.float32() for c in mapping})
        out_cols = {c: mapping[c][0] + ("_" + mapping[c][1] if mapping[c][1] else "") for c in mapping}
        schema = pa.schema([("stay", pa.int32()), ("offset", pa.int32())] + [(out_cols[c], pa.float32()) for c in mapping])
        path = self._cache(out_name)
        tmp = path.with_suffix(".parquet.tmp")
        writer = pq.ParquetWriter(tmp, schema, compression="zstd")
        stats: dict[str, Any] = {"raw_rows": 0, "rows_written": 0, "non_null": {}, "implausible_dropped": {}}
        stays_with: dict[str, set] = {out_cols[c]: set() for c in mapping}
        sorted_ok, prev = True, -1
        try:
            for t in _stream_csv(self._raw(raw_key), cols, self.block_size):
                stats["raw_rows"] += t.num_rows
                t = t.filter(pc.is_in(t["patientunitstayid"], value_set=base_ids))
                arrays = [t["patientunitstayid"], t["observationoffset"]]
                any_valid = None
                stay_np = t["patientunitstayid"].to_numpy()
                for c in mapping:
                    feature = mapping[c][0]
                    v = t[c].to_numpy(zero_copy_only=False).astype(float)
                    ok = _plausible(feature, v)
                    present = np.isfinite(v)
                    stats["implausible_dropped"][out_cols[c]] = stats["implausible_dropped"].get(out_cols[c], 0) + int((present & ~ok).sum())
                    v32 = np.where(ok, v, np.nan).astype(np.float32)
                    arrays.append(pa.array(v32, mask=~ok))
                    stats["non_null"][out_cols[c]] = stats["non_null"].get(out_cols[c], 0) + int(ok.sum())
                    stays_with[out_cols[c]].update(np.unique(stay_np[ok]).tolist())
                    any_valid = ok if any_valid is None else (any_valid | ok)
                table = pa.Table.from_arrays(arrays, schema=schema)
                if any_valid is not None:
                    table = table.filter(pa.array(any_valid))
                if table.num_rows:
                    s = table["stay"].to_numpy()
                    if s[0] < prev or not _is_sorted(s):
                        sorted_ok = False
                    prev = max(prev, int(s.max()))
                    writer.write_table(table, row_group_size=1_000_000)
                    stats["rows_written"] += table.num_rows
        finally:
            writer.close()
        if not sorted_ok:
            table = pq.read_table(tmp)
            table = table.sort_by([("stay", "ascending"), ("offset", "ascending")])
            pq.write_table(table, tmp, compression="zstd", row_group_size=1_000_000)
        os.replace(tmp, path)
        stats["raw_sorted_by_stay"] = sorted_ok
        stats["stays_with_value"] = {k: len(v) for k, v in stays_with.items()}
        return stats

    def _build_vital_periodic(self) -> dict[str, Any]:
        return self._build_wide_vitals("vital_periodic", _VITAL_PERIODIC, "store_vital_periodic.parquet")

    def _build_vital_aperiodic(self) -> dict[str, Any]:
        return self._build_wide_vitals("vital_aperiodic", _VITAL_APERIODIC, "store_vital_aperiodic.parquet")

    def _write_long(self, frame: pd.DataFrame, name: str) -> None:
        frame = frame.sort_values(["stay", "offset"]).reset_index(drop=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        _write_parquet(table, self._cache(name))

    def _build_lab(self) -> dict[str, Any]:
        base_ids = self._base_id_array()
        cols = {"patientunitstayid": pa.int32(), "labresultoffset": pa.int32(), "labname": pa.string(),
                "labresult": pa.float64(), "labmeasurenamesystem": pa.string(), "labresultrevisedoffset": pa.int32()}
        verdicts: dict[str, tuple[str | None, str]] = {}
        parts = []
        raw_rows = 0
        for t in _stream_csv(self._raw("lab"), cols, self.block_size):
            raw_rows += t.num_rows
            names = [n for n in pc.unique(t["labname"]).to_pylist() if n is not None]
            for n in names:
                if n not in verdicts:
                    verdicts[n] = classify_lab(n, self.options["include_bedside_glucose"])
            accepted = pa.array([n for n in names if verdicts[n][0] is not None], type=pa.string())
            m = pc.and_(pc.is_in(t["labname"], value_set=accepted), pc.is_in(t["patientunitstayid"], value_set=base_ids))
            parts.append(t.filter(pc.fill_null(m, False)).to_pandas())
        lab = pd.concat(parts, ignore_index=True)
        stats: dict[str, Any] = {"raw_rows": int(raw_rows), "accepted_name_rows": int(len(lab))}
        lab["feature"] = lab["labname"].map({n: v[0] for n, v in verdicts.items()})
        # revised results: keep the latest revision of each (stay, labname, draw offset)
        latest = lab.groupby(["patientunitstayid", "labname", "labresultoffset"])["labresultrevisedoffset"].transform("max")
        stale = lab["labresultrevisedoffset"].notna() & (lab["labresultrevisedoffset"] < latest)
        stats["superseded_revisions_dropped"] = int(stale.sum())
        lab = lab[~stale]
        missing = lab["labresult"].isna()
        stats["non_numeric_result_dropped"] = int(missing.sum())
        lab = lab[~missing]
        values = np.full(len(lab), np.nan)
        unit_rejected: dict[str, int] = {}
        units_seen: dict[str, dict[str, int]] = {}
        for (feature, unit), idx in lab.groupby(["feature", lab["labmeasurenamesystem"].fillna("")]).groups.items():
            units_seen.setdefault(feature, {})[unit] = int(len(idx))
            conv = conversion_for(feature, unit)
            pos = lab.index.get_indexer(idx)
            if conv is None:
                unit_rejected[f"{feature} [{unit}]"] = int(len(idx))
                continue
            values[pos] = conv.to_canonical(lab.loc[idx, "labresult"].to_numpy(float))
        lab = lab.assign(value=values)
        lab = lab[np.isfinite(lab["value"].to_numpy())]
        ok = np.zeros(len(lab), dtype=bool)
        implausible: dict[str, int] = {}
        for feature, idx in lab.groupby("feature").groups.items():
            pos = lab.index.get_indexer(idx)
            good = _plausible(feature, lab["value"].to_numpy()[pos])
            ok[pos] = good
            implausible[feature] = int((~good).sum())
        lab = lab[ok]
        out = pd.DataFrame(
            {
                "stay": lab["patientunitstayid"].astype(np.int32),
                "offset": lab["labresultoffset"].astype(np.int32),
                "feature": lab["feature"].astype(str),
                "labname": lab["labname"].astype(str),
                "value": lab["value"].astype(np.float64),
            }
        )
        self._write_long(out, "store_lab.parquet")
        stats.update({
            "accepted_names": sorted(n for n, v in verdicts.items() if v[0] is not None),
            "units_seen": units_seen,
            "unit_rejected": unit_rejected,
            "implausible_dropped": implausible,
            "rows_written_by_name": {k: int(v) for k, v in out["labname"].value_counts().items()},
            "stays_with_value": {k: int(v) for k, v in out.groupby("feature")["stay"].nunique().items()},
            "rejected_lookalikes": {
                n: v[1] for n, v in verdicts.items() if v[0] is None and v[1].startswith("excluded")
            },
        })
        return stats

    def _build_urine(self) -> dict[str, Any]:
        base_ids = self._base_id_array()
        cols = {"patientunitstayid": pa.int32(), "intakeoutputoffset": pa.int32(), "intakeoutputentryoffset": pa.int32(),
                "cellpath": pa.string(), "cellvaluenumeric": pa.float64()}
        verdicts: dict[str, tuple[bool, str]] = {}
        parts = []
        raw_rows = 0
        for t in _stream_csv(self._raw("intake_output"), cols, self.block_size):
            raw_rows += t.num_rows
            paths = [p for p in pc.unique(t["cellpath"]).to_pylist() if p is not None]
            for p in paths:
                if p not in verdicts:
                    verdicts[p] = urine_cellpath_decision(p)
            accepted = pa.array([p for p in paths if verdicts[p][0]], type=pa.string())
            m = pc.and_(pc.is_in(t["cellpath"], value_set=accepted), pc.is_in(t["patientunitstayid"], value_set=base_ids))
            parts.append(t.filter(pc.fill_null(m, False)).to_pandas())
        uo = pd.concat(parts, ignore_index=True)
        stats: dict[str, Any] = {"raw_rows": int(raw_rows), "accepted_path_rows": int(len(uo))}
        dup = uo.duplicated(["patientunitstayid", "intakeoutputoffset", "intakeoutputentryoffset", "cellpath", "cellvaluenumeric"])
        stats["exact_duplicates_dropped"] = int(dup.sum())
        uo = uo[~dup]
        v = uo["cellvaluenumeric"].to_numpy(float)
        stats["missing_value_dropped"] = int((~np.isfinite(v)).sum())
        ok = _plausible("urine_output", v)
        stats["implausible_dropped"] = int((np.isfinite(v) & ~ok).sum())
        uo = uo[ok]
        out = pd.DataFrame({"stay": uo["patientunitstayid"].astype(np.int32), "offset": uo["intakeoutputoffset"].astype(np.int32),
                            "value": uo["cellvaluenumeric"].astype(np.float64)})
        self._write_long(out, "store_urine.parquet")
        accepted = [p for p, v in verdicts.items() if v[0]]
        added = {p[len(EICU_URINE_CELLPATH_PREFIX):]: v[1] for p, v in verdicts.items() if v[1] == "accepted: urinary device label (adapter rule)"}
        removed = sorted(p[len(EICU_URINE_CELLPATH_PREFIX):] for p, v in verdicts.items()
                         if v[1].startswith("excluded: count") and is_eicu_urine_cellpath(p))
        stats.update({
            "accepted_paths": len(accepted),
            "adapter_added_labels": len(added),
            "adapter_removed_labels": removed,
            "rows_written": int(len(out)),
            "stays_with_value": int(out["stay"].nunique()),
        })
        return stats

    def _build_nurse_temperature(self) -> dict[str, Any]:
        base_ids = self._base_id_array()
        cols = {"patientunitstayid": pa.int32(), "nursingchartoffset": pa.int32(), "nursingchartcelltypevallabel": pa.string(),
                "nursingchartcelltypevalname": pa.string(), "nursingchartvalue": pa.string()}
        parts = []
        raw_rows = 0
        for t in _stream_csv(self._raw("nurse_charting"), cols, self.block_size):
            raw_rows += t.num_rows
            m = pc.and_(pc.equal(t["nursingchartcelltypevallabel"], "Temperature"),
                        pc.equal(t["nursingchartcelltypevalname"], "Temperature (C)"))
            m = pc.and_(pc.fill_null(m, False), pc.is_in(t["patientunitstayid"], value_set=base_ids))
            parts.append(t.filter(m).select(["patientunitstayid", "nursingchartoffset", "nursingchartvalue"]).to_pandas())
        temp = pd.concat(parts, ignore_index=True)
        value = pd.to_numeric(temp["nursingchartvalue"].str.strip(), errors="coerce").to_numpy(float)
        unit_ok = conversion_for("Temp", "degC")
        value = unit_ok.to_canonical(value)
        ok = _plausible("Temp", value)
        out = pd.DataFrame({"stay": temp["patientunitstayid"].astype(np.int32), "offset": temp["nursingchartoffset"].astype(np.int32),
                            "value": value})[ok]
        self._write_long(out, "store_nurse_temperature.parquet")
        return {
            "raw_rows": int(raw_rows),
            "temperature_c_rows": int(len(temp)),
            "non_numeric_dropped": int((~np.isfinite(value)).sum()),
            "implausible_dropped": int((np.isfinite(value) & ~ok).sum()),
            "rows_written": int(len(out)),
            "stays_with_value": int(out["stay"].nunique()),
        }

    def _read_store(self, name: str, ids: np.ndarray, columns: Sequence[str] | None = None) -> pa.Table:
        """Rows of the requested stays. The range predicate prunes row groups (files are
        sorted by stay); the membership predicate is applied inside the scan so only the
        requested rows are materialised."""
        path = self._cache(name)
        lo, hi = int(ids.min()), int(ids.max())
        wanted = [int(i) for i in ids]
        table = pq.read_table(
            path, columns=list(columns) if columns else None,
            filters=[("stay", ">=", lo), ("stay", "<=", hi), ("stay", "in", wanted)],
        )
        return table

    # ------------------------------------------------------------------ SourceAdapter API

    def stays(self) -> pd.DataFrame:
        """Cohort-eligible adult stays: subject_id, stay_id, intime, outtime."""
        stays, counts = self._cohort()
        self._log(f"stays: {counts['cohort_stays']} of {counts['base_adult_stays']} adult stays kept; excluded {counts['excluded']}")
        return stays.copy()

    def vasopressor_intervals(self) -> pd.DataFrame:
        """Vasopressor infusion runs (rate > 0) of the cohort stays: stay_id, start, end, drug."""
        stays, _ = self._cohort()
        iv = self._intervals("vasopressor_on")
        iv = iv[iv["stay_id"].isin(set(stays["stay_id"].tolist()))]
        return pd.DataFrame(
            {
                "stay_id": iv["stay_id"].astype(np.int64).to_numpy(),
                "start": offsets_to_time(iv["start_offset"].to_numpy()),
                "end": offsets_to_time(iv["end_offset"].to_numpy()),
                "drug": iv["drug"].astype(str).to_numpy(),
            }
        )

    def events(self, stay_ids: Sequence[int]) -> pd.DataFrame:
        """Canonical numeric observations of ``stay_ids``: stay_id, time, feature, value, modality."""
        columns = ["stay_id", "time", "feature", "value", "modality"]
        ids = np.unique(np.asarray(list(stay_ids), dtype=np.int64))
        if ids.size == 0:
            return pd.DataFrame(columns=columns)
        self._store_parts()
        stay_parts, offset_parts, code_parts, value_parts, modality_parts = [], [], [], [], []

        def add(stay: np.ndarray, offset: np.ndarray, feature: str, value: np.ndarray, modality: str | None) -> None:
            keep = np.isfinite(value)
            if not keep.any():
                return
            stay_parts.append(stay[keep].astype(np.int64))
            offset_parts.append(offset[keep].astype(np.int64))
            code_parts.append(np.full(int(keep.sum()), _FEATURE_CODES[feature], dtype=np.int16))
            value_parts.append(value[keep].astype(np.float64))
            modality_parts.append(np.full(int(keep.sum()), _MODALITIES.index(modality), dtype=np.int8))

        for name, mapping in (("store_vital_periodic.parquet", _VITAL_PERIODIC), ("store_vital_aperiodic.parquet", _VITAL_APERIODIC)):
            table = self._read_store(name, ids)
            stay = table["stay"].to_numpy()
            offset = table["offset"].to_numpy()
            for feature, modality in mapping.values():
                col = feature + ("_" + modality if modality else "")
                add(stay, offset, feature, table[col].to_numpy(zero_copy_only=False).astype(np.float64), modality or None)
        if self.options["nursecharting_temperature"]:
            table = self._read_store("store_nurse_temperature.parquet", ids)
            add(table["stay"].to_numpy(), table["offset"].to_numpy(), "Temp", table["value"].to_numpy(), None)
        table = self._read_store("store_lab.parquet", ids, ["stay", "offset", "feature", "value"])
        lab = table.to_pandas()
        for feature, group in lab.groupby("feature"):
            add(group["stay"].to_numpy(), group["offset"].to_numpy(), str(feature), group["value"].to_numpy(), None)
        table = self._read_store("store_urine.parquet", ids)
        add(table["stay"].to_numpy(), table["offset"].to_numpy(), "urine_output", table["value"].to_numpy(), None)
        if not stay_parts:
            return pd.DataFrame(columns=columns)
        codes = np.concatenate(code_parts)
        frame = pd.DataFrame(
            {
                "stay_id": np.concatenate(stay_parts),
                "time": offsets_to_time(np.concatenate(offset_parts)),
                "feature": np.asarray(NUMERIC_FEATURES, dtype=object)[codes],
                "value": np.concatenate(value_parts),
                "modality": np.asarray(_MODALITIES, dtype=object)[np.concatenate(modality_parts)],
            }
        )
        return frame.sort_values(["stay_id", "time"], kind="stable").reset_index(drop=True)

    def feature_events(self, stay_ids: Sequence[int], features: Iterable[str]) -> pd.DataFrame:
        """:meth:`events` restricted to ``features``; laboratory features read only the lab store."""
        wanted = frozenset(str(f) for f in features)
        non_lab = {f for f, _ in (*_VITAL_PERIODIC.values(), *_VITAL_APERIODIC.values())} | {"Temp", "urine_output"}
        if wanted & non_lab:
            ev = self.events(stay_ids)
            return ev[ev["feature"].isin(wanted)].reset_index(drop=True)
        columns = ["stay_id", "time", "feature", "value", "modality"]
        ids = np.unique(np.asarray(list(stay_ids), dtype=np.int64))
        if ids.size == 0:
            return pd.DataFrame(columns=columns)
        self._store_parts()
        lab = self._read_store("store_lab.parquet", ids, ["stay", "offset", "feature", "value"]).to_pandas()
        lab = lab[lab["feature"].astype(str).isin(wanted).to_numpy() & np.isfinite(lab["value"].to_numpy(dtype=float))]
        frame = pd.DataFrame(
            {
                "stay_id": lab["stay"].to_numpy(dtype=np.int64),
                "time": offsets_to_time(lab["offset"].to_numpy()),
                "feature": lab["feature"].astype(str).to_numpy(dtype=object),
                "value": lab["value"].to_numpy(dtype=np.float64),
                "modality": np.full(len(lab), None, dtype=object),
            }
        )
        return frame.sort_values(["stay_id", "time"], kind="stable").reset_index(drop=True)

    def interventions(self, stay_ids: Sequence[int]) -> pd.DataFrame:
        """sedation_on runs of ``stay_ids`` (ventilation_on is not available in eICU)."""
        ids = set(int(s) for s in stay_ids)
        iv = self._intervals("sedation_on")
        iv = iv[iv["stay_id"].isin(ids)]
        return pd.DataFrame(
            {
                "stay_id": iv["stay_id"].astype(np.int64).to_numpy(),
                "start": offsets_to_time(iv["start_offset"].to_numpy()),
                "end": offsets_to_time(iv["end_offset"].to_numpy()),
                "feature": np.full(len(iv), "sedation_on", dtype=object),
            }
        )

    def available_features(self) -> frozenset[str]:
        """Canonical features with data in eICU; ``ventilation_on`` is excluded by decision."""
        stats = self._store_parts()
        have: set[str] = set()
        for part in ("vital_periodic", "vital_aperiodic"):
            for col, n in stats[part].get("non_null", {}).items():
                if n > 0:
                    have.add(col.split("_")[0])
        if stats.get("nurse_temperature", {}).get("rows_written", 0) > 0:
            have.add("Temp")
        have.update(f for f, n in stats["lab"].get("stays_with_value", {}).items() if n > 0)
        if stats["urine"].get("rows_written", 0) > 0:
            have.add("urine_output")
        if self._interval_part().get("sedation_on", {}).get("intervals", 0) > 0:
            have.add("sedation_on")
        return frozenset(f for f in CANONICAL_FEATURES if f in have)

    def provenance(self) -> dict[str, Any]:
        """Files, name lists, options, decisions and build statistics."""
        manifest = self._load_manifest()
        _, counts = self._cohort()
        sums = {}
        sums_path = self.raw_dir / "SHA256SUMS.txt"
        if sums_path.exists():
            for line in sums_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) == 2:
                    sums[parts[1].lstrip("*")] = parts[0]
        files = {}
        for key, name in RAW_FILES.items():
            path = self._raw(key)
            evidence_used = self.options["exclude_ambiguous_controls"] or self.options["require_washout"]
            used = {"medication": evidence_used, "treatment": evidence_used,
                    "nurse_charting": self.options["nursecharting_temperature"]}.get(key, True)
            files[name] = {"used": used, "bytes": path.stat().st_size if path.exists() else None, "sha256_listed": sums.get(name)}
        names = pd.read_parquet(self._cache("infusion_names.parquet")) if self._cache("infusion_names.parquet").exists() else pd.DataFrame()
        drug_lists = {}
        if len(names):
            for feature in ("vasopressor_on", "sedation_on"):
                sub = names[names["feature"] == feature]
                drug_lists[feature] = {str(r.drugname): {"drug": str(r.drug), "rows": int(r.n_rows)} for r in sub.itertuples(index=False)}
        return {
            "adapter": "src.icu.v2.extract.eicu.EICUAdapter",
            "cache_version": CACHE_VERSION,
            "raw_dir": str(self.raw_dir),
            "cache_dir": str(self.cache_dir),
            "raw_files": files,
            "time_convention": "intime = 2000-01-01 00:00 for every stay; event time = intime + offset minutes "
                               "(offsets are minutes from unit admission); outtime = intime + unitdischargeoffset",
            "subject_id": "int(sha1(uniquepid)[:15], 16), positive int64, collision-checked",
            "options": self.options,
            "decisions": {
                "vasopressors": "infusionDrug runs per drugname: start at rate > 0, end at rate == 0, after max_gap without "
                                "rows, or at unit discharge; text OFF = stop, other text/missing rates dropped; dobutamine, "
                                "milrinone, epidural mixtures excluded; angiotensin II included (0 rows in eICU 2.0)",
                "sedation_on": f"same run rules for {self.options['sedation_drugs']} (task set; harmonization lists fentanyl "
                               "instead of lorazepam)",
                "ventilation_on": "not available: respiratoryCare.ventendoffset is 0 in 865,224/865,381 rows (157 non-zero rows, "
                                  "1 stay); 'RT Vent On/Off' only in 2,741 stays vs PEEP charted in 45,541",
                "blood_pressure": "vitalPeriodic systemic* = invasive, vitalAperiodic noninvasive* = noninvasive; pa*/cvp unused",
                "temperature": "vitalPeriodic temperature + nurseCharting 'Temperature (C)' (degC); 'Temperature (F)' duplicates",
                "labs": "exact labnames incl. 'bedside glucose' (team decision); system unit checked with conversion_for; "
                        "latest revision per draw offset; labresultoffset = draw time",
                "urine_output": "cellvaluenumeric of urine cellpaths (harmonization rule + adapter additions - count labels); "
                                "never outputtotal; 0-5000 mL per record",
                "cohort": "adults (age >= 18 or '> 89'), unitdischargeoffset > 0, then the eICU data-quality exclusions "
                          "(hospital infusion charting, ambiguous vasopressor exposure) - see module docstring",
                "plausibility": "values outside harmonization.CANONICAL_SPECS [low, high] are dropped",
            },
            "cohort": counts,
            "rcheck_evidence": RCHECK_EVIDENCE,
            "drug_names": drug_lists,
            "build": {k: {kk: vv for kk, vv in v.items() if kk != "key"} for k, v in manifest.items()},
            "available_features": sorted(self.available_features()),
            "unavailable_features": {"ventilation_on": "no reliable start/stop signal (R-E4, R-E5)"},
        }
