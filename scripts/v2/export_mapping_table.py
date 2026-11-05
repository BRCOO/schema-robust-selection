#!/usr/bin/env python3
"""Export the v2 source-to-canonical feature mappings and the shared task rules as paper tables.

The tables are built from the code that performs the extraction, so the appendix cannot drift
from it:

* identifiers, labels, documented units and options come from the three adapters
  ``src/icu/v2/extract/{mimic_iv,eicu,hirid}.py``: module constants, constructor defaults
  (``scripts/v2/extract_v2_cohort.py`` builds every adapter as ``cls(raw_dir, cache_dir)``,
  which is checked here, so the defaults are the options of every run), side-effect-free
  helpers (``eicu.classify_lab``, ``eicu.classify_infusion``, ``eicu.urine_cellpath_decision``,
  ``hirid.drug_name``, ``MimicIVAdapter._drug_items``, ...) and literal entries of the
  adapters' ``provenance()`` dicts (parsed with ``ast``; ``provenance()`` is never called).
  Where an adapter applies a :mod:`src.icu.v2.harmonization` table (eICU lab and drug names,
  HiRID ids), the table rows pass through the adapter's own filters, so overrides such as
  point-of-care glucose or the HiRID RR ids are reflected;
* canonical units, conversions and plausible ranges come from ``harmonization.CANONICAL_SPECS``
  and ``harmonization.conversion_for``;
* the task rules come from the ``common.ExtractionSpec`` defaults and the argparse defaults of
  ``scripts/v2/extract_v2_cohort.py`` (parsed with ``ast``; the script is not run); the
  window, aggregation and missing-feature semantics are checked on a synthetic episode with
  ``common.episode_matrix`` before anything is written.

Nothing under ``data/`` is opened, no adapter cache is read or built and no adapter method
that reads raw data is called. Look-alike exclusion reasons are the notes of the
harmonization rows with clauses about pending checks or v1 code removed (:func:`clean_reason`).

Outputs (``--out-dir``, default ``paper/v2_assets/tables``):

* ``feature_mappings.csv`` / ``.tex``: one row per (canonical feature, source), 18 x 3. The
  CSV carries the full identifier lists (list cells are joined with ``" | "``); the .tex
  (booktabs + array; meant for a ``table*`` in ``\\scriptsize``) abbreviates long lists and
  keeps the key notes only;
* ``task_rules.csv`` / ``.tex``: the shared task rules.

SOURCES.md: ``scripts/v2/make_paper_assets.py`` rewrites ``paper/v2_assets/SOURCES.md`` on
every run, so this script never touches that file. It prints its provenance lines (same format
as make_paper_assets) to stdout; append them to SOURCES.md after the last make_paper_assets run.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/export_mapping_table.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.icu.v2 import harmonization as hz  # noqa: E402
from src.icu.v2.extract import common, eicu, hirid, mimic_iv  # noqa: E402

ASSETS_DIR = ROOT / "paper" / "v2_assets"
DEFAULT_OUT = ASSETS_DIR / "tables"
CLI_SCRIPT = ROOT / "scripts" / "v2" / "extract_v2_cohort.py"
ASSETS_SCRIPT = ROOT / "scripts" / "v2" / "make_paper_assets.py"
SOURCES = ("mimic_iv", "eicu", "hirid")
SOURCE_LABEL = {"mimic_iv": "MIMIC-IV", "eicu": "eICU", "hirid": "HiRID"}
MODULES = {"mimic_iv": mimic_iv, "eicu": eicu, "hirid": hirid}
ADAPTER_CLASSES = {"mimic_iv": mimic_iv.MimicIVAdapter, "eicu": eicu.EICUAdapter, "hirid": hirid.HiRIDAdapter}
FEATURES = hz.CANONICAL_FEATURES
LIST_SEP = " | "
#: Placeholder paths: the MIMIC-IV and HiRID constructors only store paths and options. The
#: eICU constructor creates its cache directory, so EICUAdapter is never instantiated here.
UNUSED = "unused"
#: d_items labels of the two procedureevents items (quoted in the mimic_iv docstring; the
#: table is not part of harmonization.MIMIC_IV_TABLE). Checked against the docstring.
MIMIC_VENT_LABELS = {mimic_iv.VENT_INVASIVE_ITEM: "Invasive Ventilation", mimic_iv.VENT_NONINVASIVE_ITEM: "Non-invasive Ventilation"}
CLI_TO_SPEC = {
    "--window-hours": "window_hours",
    "--gap-hours": "prediction_gap_hours",
    "--controls-per-positive": "controls_per_positive",
    "--seed": "seed",
    "--max-positives-per-patient": "max_positives_per_patient",
    "--match-columns": "match_columns",
    "--required-vital": "required_vital",
    "--required-vital-last-bins": "required_vital_last_bins",
}
FEATURE_COLUMNS = (
    "canonical_feature", "canonical_unit", "kind", "plausible_range", "source", "available", "source_tables",
    "source_ids", "source_items", "unit_handling", "excluded_lookalikes", "notes", "code_reference",
)
RULE_COLUMNS = ("rule", "setting", "encoded_in")


# ------------------------------------------------------------------------------- records


@dataclass
class Item:
    """One mapped source identifier. ``unit`` None = no documented unit (state items)."""

    ident: str
    label: str = ""
    unit: str | None = None
    tag: str = ""
    group: str = ""  # .tex group ("inv.", "POC", ...)
    short: str = ""  # .tex display (default: ident)

    def text(self) -> str:
        out = self.ident if not self.label or self.label == self.ident else f"{self.ident} {self.label}"
        return f"{out} [{self.tag}]" if self.tag else out


@dataclass
class Lookalike:
    """An identifier the adapter does not map although it resembles the feature."""

    where: str
    ident: str
    label: str
    reason: str

    def text(self) -> str:
        head = f"{self.where}:{self.ident}" if self.where else self.ident
        if self.label and self.label != self.ident:
            head += f" {self.label}"
        return f"{head} ({self.reason})" if self.reason else head


@dataclass
class Row:
    feature: str
    source: str
    tables: list[str] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    units: str = ""
    tex_units: str = ""
    notes: list[str] = field(default_factory=list)
    tex_notes: list[str] = field(default_factory=list)
    lookalikes: list[Lookalike] = field(default_factory=list)
    code: list[str] = field(default_factory=list)
    unavailable: str = ""
    tex_items: str = ""

    def add(self, table: str, item: Item, code: str) -> None:
        if table not in self.tables:
            self.tables.append(table)
        self.items.append(item)
        self.cite(code)

    def cite(self, *codes: str) -> None:
        for code in codes:
            if code and code not in self.code:
                self.code.append(code)

    def note(self, text: str, tex: str | None = None) -> None:
        self.notes.append(text)
        if tex:
            self.tex_notes.append(tex)

    def exclude(self, where: str, ident: object, label: str, reason: str) -> None:
        """Record a look-alike once per (table/namespace, identifier); callers skip mapped ids."""
        ident = str(ident)
        if not any(x.ident == ident and x.where == where for x in self.lookalikes):
            self.lookalikes.append(Lookalike(where, ident, label, reason))


# ------------------------------------------------------------------------------- helpers


def constructor_defaults(cls: type) -> dict[str, Any]:
    """Keyword defaults of an adapter constructor (= the options of every CLI run)."""
    params = inspect.signature(cls.__init__).parameters
    return {name: p.default for name, p in params.items() if p.default is not inspect.Parameter.empty}


def source_version(source: str) -> str:
    """"MIMIC-IV 3.1", "eICU-CRD 2.0", "HiRID 1.1.1" (SOURCE_VERSION or the adapter docstring)."""
    module = MODULES[source]
    version = getattr(module, "SOURCE_VERSION", None)
    if version:
        return str(version)
    first = (module.__doc__ or "").strip().splitlines()[0]
    match = re.match(r"(\S+ \d+(?:\.\d+)+)\b", first)
    if not match:
        raise ValueError(f"cannot read the source version from the {source} adapter docstring")
    return match.group(1)


_SKIP = object()


def _literal_entries(node: ast.AST) -> Any:
    """Literal value of ``node``; for dicts and lists the literal entries only."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        pass
    if isinstance(node, ast.Dict):
        out = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant):
                entry = _literal_entries(value)
                if entry is not _SKIP:
                    out[key.value] = entry
        return out
    if isinstance(node, ast.List):
        return [e for e in (_literal_entries(v) for v in node.elts) if e is not _SKIP]
    return _SKIP


def provenance_literals(cls: type) -> dict[str, Any]:
    """Literal entries of the dict that ``cls.provenance`` returns (parsed, never called)."""
    function = ast.parse(textwrap.dedent(inspect.getsource(cls.provenance))).body[0]
    for node in reversed(function.body):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return _literal_entries(node.value)
    raise ValueError(f"{cls.__name__}.provenance does not return a dict literal")


def module_literal(path: Path, name: str) -> Any:
    """Value of a module-level literal assignment ``name = ...`` (parsed, not imported)."""
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} is not a module-level literal of {path.name}")


def cli_options(path: Path = CLI_SCRIPT) -> dict[str, dict[str, Any]]:
    """``parser.add_argument`` calls of a script: first flag -> default/help/type/action/required."""
    options: dict[str, dict[str, Any]] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not flags:
            continue
        entry: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg in ("default", "help", "type", "action", "required"):
                try:
                    entry[keyword.arg] = ast.literal_eval(keyword.value)
                except (ValueError, TypeError, SyntaxError):
                    entry[keyword.arg] = ast.unparse(keyword.value)
        options[flags[0]] = entry
    return options


def check_cli_builds_adapters_with_defaults(path: Path = CLI_SCRIPT) -> None:
    """The CLI must construct the adapters listed here with (raw_dir, cache_dir) only."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {s: f"{c.__module__}:{c.__qualname__}" for s, c in ADAPTER_CLASSES.items()}
    if module_literal(path, "ADAPTERS") != wanted:
        raise RuntimeError(f"{path.name} ADAPTERS differ from {wanted}")
    loader = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_adapter"), None)
    calls = [n for n in ast.walk(loader) if isinstance(n, ast.Call) and isinstance(n.func, ast.Call)] if loader else []
    if not calls or any(len(c.args) != 2 or c.keywords for c in calls):
        raise RuntimeError(f"{path.name}: load_adapter no longer builds adapters with the constructor defaults")


def require_literal_in_source(function: Any, *literals: str) -> None:
    """Guard for values that are literals inside adapter functions, not module constants."""
    source = inspect.getsource(function)
    missing = [lit for lit in literals if lit not in source]
    if missing:
        raise RuntimeError(f"{function.__qualname__} no longer contains {missing}; update export_mapping_table.py")


_STALE = re.compile(r"pending|not yet verified|unverified|\bverify\b|needs a pass|\bv1\b|_dataset\.py|no MIMIC itemid", re.IGNORECASE)


def clean_reason(note: str) -> str:
    """A harmonization note without clauses about pending checks or the v1 code."""
    text = re.sub(r"\s*\([^()]*\)", lambda m: "" if _STALE.search(m.group()) else m.group(), str(note))
    clauses = [c.strip() for c in text.split(";")]
    return "; ".join(c for c in clauses if c and not _STALE.search(c))


def drug_label(name: str) -> str:
    """Canonical drug name for display: angiotensin_ii -> angiotensin II."""
    return re.sub(r"_ii\b", " II", str(name)).replace("_", " ")


def unique(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def strip_table(name: str) -> str:
    return re.sub(r"\.(csv|parquet)(\.gz)?$", "", str(name))


def short_table(name: str) -> str:
    """.tex table name: icu/chartevents -> chartevents, observation_tables -> observation."""
    return re.sub(r"_(tables|records)$", "", str(name).split("/")[-1])


def hours(delta: pd.Timedelta) -> str:
    minutes = delta.total_seconds() / 60.0
    return f"{minutes / 60:g} h" if minutes >= 120 and minutes % 60 == 0 else f"{minutes:g} min"


def plausible_range(feature: str) -> str:
    spec = hz.CANONICAL_SPECS[feature]
    if spec.kind == "state":
        return ""
    return f"{spec.low:g}-{spec.high:g} {spec.unit}" + (" per record" if spec.kind == "volume" else "")


def measurement_units(feature: str, items: Sequence[Item]) -> tuple[str, str]:
    """('<unit> -> <canonical>: <conversion>' per documented unit, short .tex form)."""
    canonical = hz.CANONICAL_SPECS[feature].unit
    groups: dict[tuple[str, str], list[str]] = {}
    for item in items:
        if item.unit is None:
            continue
        conversion = hz.conversion_for(feature, item.unit)
        if conversion is None:
            raise ValueError(f"{feature}: documented unit {item.unit!r} of {item.ident} is not accepted by conversion_for")
        groups.setdefault((item.unit, conversion.name), []).append(item.ident)
    parts, short = [], []
    for (unit, name), idents in groups.items():
        shown = unit or "no unit"
        token = hz.normalize_unit(unit)
        if unit and token == canonical and token.lower() != unit.lower():
            shown += f" (= {canonical})"
        parts.append(f"{shown} -> {canonical}: {name}" + (f" [{', '.join(idents)}]" if len(groups) > 1 else ""))
        brief = re.sub(r"\(\s*[A-Za-z_ ]+,\s*", "(", name)
        if brief not in short:
            short.append(brief)
    short.sort(key=lambda s: s != "identity")
    return "; ".join(parts), "; ".join(short)


def abbreviate(values: Sequence[str], keep: int = 3) -> str:
    values = list(values)
    if len(values) <= keep + 1:
        return ", ".join(values)
    return ", ".join(values[:keep]) + f", ... (n={len(values)})"


# ------------------------------------------------------------------------------- MIMIC-IV


def mimic_rows() -> dict[str, Row]:
    m = mimic_iv
    adapter = m.MimicIVAdapter(UNUSED, UNUSED)  # stores paths and options; reads nothing
    options = adapter.options
    rows = {f: Row(f, "mimic_iv") for f in FEATURES}
    table = {key: strip_table(files[0]) for key, files in m.TABLE_FILES.items()}
    doc = " ".join((m.__doc__ or "").split())
    for itemid, label in MIMIC_VENT_LABELS.items():
        if f"{itemid} {label}" not in doc:
            raise RuntimeError(f"mimic_iv docstring no longer quotes '{itemid} {label}'")

    def label(namespace: str, itemid: int, fallback: str = "") -> str:
        row = hz.lookup_identifier("mimic_iv", namespace, itemid) if namespace in hz.NAMESPACES["mimic_iv"] else None
        return row.label if row is not None else fallback

    lab_features = unique(f for f, _ in m.LAB_ITEMS.values())
    for itemid, (feature, unit) in m.LAB_ITEMS.items():
        rows[feature].add(table["labevents"], Item(str(itemid), label("labevents", itemid), unit), "mimic_iv.LAB_ITEMS")
    chart = {k: v for k, v in m.CHART_ITEMS.items() if options["poc_glucose"] or k not in m.POC_GLUCOSE_ITEMS}
    for itemid, (feature, unit, modality) in chart.items():
        poc = itemid in m.POC_GLUCOSE_ITEMS
        tag = modality or ("point of care" if poc else "")
        group = {"invasive": "inv.", "noninvasive": "non-inv."}.get(modality or "", "POC" if poc else "")
        rows[feature].add(table["chartevents"], Item(str(itemid), label("chartevents", itemid), unit, tag, group), "mimic_iv.CHART_ITEMS")
        if poc:
            rows[feature].cite("mimic_iv.POC_GLUCOSE_ITEMS")
    urine = rows["urine_output"]
    for itemid, name in m.URINE_ITEMS.items():
        urine.add(table["outputevents"], Item(str(itemid), label("outputevents", itemid, name), "mL"), "mimic_iv.URINE_ITEMS")
    urine.add(table["outputevents"], Item(str(m.GU_IRRIGANT_OUT), label("outputevents", m.GU_IRRIGANT_OUT), "mL",
                                          f"irrigant + urine out; minus {m.GU_IRRIGANT_IN}", "GU net"), "mimic_iv.GU_IRRIGANT_OUT")
    urine.add(table["outputevents"], Item(str(m.GU_IRRIGANT_IN), label("outputevents", m.GU_IRRIGANT_IN), "mL",
                                          "irrigant in; subtracted", "GU net"), "mimic_iv.GU_IRRIGANT_IN")
    drugs: dict[str, list[str]] = {}
    for kind, feature, code in (("vasopressor", "vasopressor_on", "mimic_iv.VASOPRESSOR_ITEMS + ANGIOTENSIN_ITEMS"),
                                ("sedation", "sedation_on", "mimic_iv.SEDATIVE_ITEMS (+ OPIOID_ITEMS)")):
        items = adapter._drug_items(kind)
        drugs[feature] = unique(items.values())
        for itemid, drug in items.items():
            rows[feature].add(table["inputevents"], Item(str(itemid), label("inputevents", itemid, drug_label(drug))), code)
        rows[feature].cite("MimicIVAdapter._drug_items")
    vent_ids = [m.VENT_INVASIVE_ITEM] + ([m.VENT_NONINVASIVE_ITEM] if options["include_noninvasive_ventilation"] else [])
    for itemid in vent_ids:
        rows["ventilation_on"].add(table["procedureevents"], Item(str(itemid), MIMIC_VENT_LABELS[itemid]), "mimic_iv.VENT_INVASIVE_ITEM")

    # adapter-level exclusions first (their reason wins), then the harmonization look-alikes
    if not options["include_noninvasive_ventilation"]:
        rows["ventilation_on"].exclude("procedureevents", m.VENT_NONINVASIVE_ITEM, MIMIC_VENT_LABELS[m.VENT_NONINVASIVE_ITEM],
                                       "only with include_noninvasive_ventilation=True")
    if not options["include_opioids"]:
        for itemid, drug in m.OPIOID_ITEMS.items():
            rows["sedation_on"].exclude("inputevents", itemid, label("inputevents", itemid, drug_label(drug)),
                                        "opioid; only with include_opioids=True")
    if not options["include_angiotensin"]:
        for itemid, drug in m.ANGIOTENSIN_ITEMS.items():
            rows["vasopressor_on"].exclude("inputevents", itemid, drug_label(drug), "only with include_angiotensin=True")
    if not options["poc_glucose"]:
        for itemid in m.POC_GLUCOSE_ITEMS:
            rows["Glucose"].exclude("chartevents", itemid, label("chartevents", itemid), "only with poc_glucose=True")
    mapped_ids = {int(i.ident) for r in rows.values() for i in r.items}  # MIMIC itemids are unique across tables
    for itemid, text in m.EXCLUDED_DRUG_ITEMS.items():
        if itemid in mapped_ids:
            raise RuntimeError(f"mimic_iv.EXCLUDED_DRUG_ITEMS[{itemid}] is also mapped")
        match = re.match(r"^(.*?)\s*\((.*)\)$", text)
        name, reason = (match.group(1), match.group(2)) if match else (text, "")
        trow = hz.lookup_identifier("mimic_iv", "inputevents", itemid)
        feature = trow.feature if trow is not None else ("sedation_on" if "sedation" in text.lower() else None)
        if feature is None:
            raise RuntimeError(f"cannot assign mimic_iv.EXCLUDED_DRUG_ITEMS[{itemid}] to a feature")
        shown = label("inputevents", itemid, name)
        if not reason or f"({reason})" in shown:  # the label already says it (e.g. "Fentanyl (Push)")
            reason = clean_reason(trow.note) if trow is not None else "adapter exclusion"
        rows[feature].exclude("inputevents", itemid, shown, reason)
        rows[feature].cite("mimic_iv.EXCLUDED_DRUG_ITEMS")
    for namespace, checks, code in (("chartevents", m.CHART_CHECK_ITEMS, "mimic_iv.CHART_CHECK_ITEMS"),
                                    ("labevents", m.LAB_CHECK_ITEMS, "mimic_iv.LAB_CHECK_ITEMS")):
        for itemid, name in checks.items():
            row = hz.lookup_identifier("mimic_iv", namespace, itemid)
            if row is None:
                raise RuntimeError(f"{code}[{itemid}] is not in harmonization.MIMIC_IV_TABLE")
            rows[row.feature].exclude(namespace, itemid, name, clean_reason(row.note))
            rows[row.feature].cite(code)
    for trow in hz.MIMIC_IV_TABLE:
        if int(trow.identifier) in mapped_ids:
            continue
        reason = clean_reason(trow.note) if trow.decision == "exclude" else "harmonization include row not used by the adapter"
        rows[trow.feature].exclude(trow.namespace, trow.identifier, trow.label, reason)

    # unit handling
    per_row_unit = ("per row: valueuom -> conversion_for (an empty valueuom is read as the documented unit; "
                    "unconvertible rows are dropped)")
    for feature, row in rows.items():
        if hz.CANONICAL_SPECS[feature].kind == "measurement" and row.items:
            full, short = measurement_units(feature, row.items)
            row.units, row.tex_units = f"{full}; {per_row_unit}", short
    kept_statuses = [s for s in ("Paused", "Stopped") if s not in m.EXCLUDED_STATUSES]  # statuses that end a delivery period
    drug_rule = (f"1 in bins overlapping [starttime, endtime) of inputevents rows with ordercategorydescription in "
                 f"({', '.join(m.CONTINUOUS_CATEGORIES)}), rate > 0 and statusdescription not in ({', '.join(m.EXCLUDED_STATUSES)})"
                 + (f"; {'/'.join(kept_statuses)} rows kept" if kept_statuses else ""))
    for feature in ("vasopressor_on", "sedation_on"):
        rows[feature].units, rows[feature].tex_units = drug_rule, "rate > 0 intervals"
        rows[feature].cite("mimic_iv.CONTINUOUS_CATEGORIES", "mimic_iv.EXCLUDED_STATUSES")
    rows["ventilation_on"].units = (f"1 in bins overlapping [starttime, endtime) of procedureevents rows "
                                    f"(statusdescription not in ({', '.join(m.EXCLUDED_STATUSES)}))")
    rows["ventilation_on"].tex_units = "intervals"
    require_literal_in_source(m.MimicIVAdapter._urine_events, 'isin(["mL", ""])')
    rows["urine_output"].units = (f"valueuom mL (or empty) -> mL per record; {m.GU_IRRIGANT_OUT} - {m.GU_IRRIGANT_IN} netted per "
                                  "(stay, charttime), negative nets and unpaired irrigation rows dropped")
    rows["urine_output"].tex_units = "mL; irrigant netted"

    # notes
    for feature in lab_features:
        rows[feature].note("labevents has no stay_id: a result belongs to every ICU stay of the subject whose [intime, outtime] "
                           "contains charttime (specimen collection time)")
    pooled = options["ph_specimen"] == "blood_gas"
    rows["pH"].note(f"blood-gas pH; specimen type (labevents {m.SPECIMEN_TYPE_ITEM}) "
                    + ("pooled" if pooled else f"restricted to {m.ARTERIAL_SPECIMEN!r}")
                    + f" (ph_specimen={options['ph_specimen']!r}; 'arterial' keeps {m.ARTERIAL_SPECIMEN!r} only)",
                    "arterial + venous pooled" if pooled else "arterial only")
    rows["pH"].cite("mimic_iv.SPECIMEN_TYPE_ITEM", "mimic_iv.ARTERIAL_SPECIMEN")
    poc = [i for i in rows["Glucose"].items if i.tag == "point of care"]
    if poc:
        rows["Glucose"].note(f"point-of-care finger stick {', '.join(i.ident for i in poc)} included (poc_glucose=True; "
                             f"cross-database decision); its empty valueuom is read as {poc[0].unit}",
                             f"lab + POC finger stick {', '.join(i.ident for i in poc)}")
    vaso = rows["vasopressor_on"]
    vaso.note(f"drugs: {', '.join(drug_label(d) for d in drugs['vasopressor_on'])}"
              + (" (angiotensin II with include_angiotensin=True)" if options["include_angiotensin"] else ""),
              ", ".join(drug_label(d) for d in drugs["vasopressor_on"]))
    sedation = rows["sedation_on"]
    opioids = unique(m.OPIOID_ITEMS.values())
    sedation.note(f"continuous infusions of {', '.join(drug_label(d) for d in drugs['sedation_on'])}"
                  + ("" if options["include_opioids"] else f"; opioids ({', '.join(opioids)}) only with include_opioids=True"),
                  ", ".join(drug_label(d) for d in drugs["sedation_on"]) + ("" if options["include_opioids"] else "; no opioids"))
    vent = rows["ventilation_on"]
    vent.note(f"invasive ventilation intervals (procedureevents {m.VENT_INVASIVE_ITEM})"
              + ("" if options["include_noninvasive_ventilation"] else
                 f"; non-invasive {m.VENT_NONINVASIVE_ITEM} only with include_noninvasive_ventilation=True"),
              "invasive (procedureevents)" + ("" if options["include_noninvasive_ventilation"] else "; NIV off"))
    urine.note(f"{len(m.URINE_ITEMS)} urine output items + GU irrigation net ({m.GU_IRRIGANT_OUT} - {m.GU_IRRIGANT_IN}); summed per bin",
               "sum per bin; GU irrigant netted")
    rows["Temp"].note("values outside the plausible range are dropped after conversion (e.g. degF typed into the degC item)")
    return rows


# ------------------------------------------------------------------------------- eICU


def _eicu_include_pattern(feature: str, drug: str) -> str | None:
    """The harmonization infusionDrug include pattern that accepts the drug name itself."""
    for row in hz.EICU_TABLE:
        if row.namespace == "infusionDrug" and row.feature == feature and row.decision == "include" and row.match == "regex":
            if re.search(str(row.identifier), drug_label(drug), re.IGNORECASE):
                return str(row.identifier)
    return None


def eicu_rows() -> dict[str, Row]:
    e = eicu
    options = constructor_defaults(e.EICUAdapter)
    literals = provenance_literals(e.EICUAdapter)
    unavailable = dict(literals.get("unavailable_features", {}))
    decisions = dict(literals.get("decisions", {}))
    rows = {f: Row(f, "eicu") for f in FEATURES}
    files = {key: strip_table(name) for key, name in e.RAW_FILES.items()}

    vital_features = []
    for raw_key, namespace, mapping, code in (("vital_periodic", "vitalPeriodic", e._VITAL_PERIODIC, "eicu._VITAL_PERIODIC"),
                                              ("vital_aperiodic", "vitalAperiodic", e._VITAL_APERIODIC, "eicu._VITAL_APERIODIC")):
        for column, (feature, modality) in mapping.items():
            trow = hz.lookup_identifier("eicu", namespace, column)
            if trow is None or trow.feature != feature or trow.decision != "include":
                raise RuntimeError(f"eICU {namespace}.{column} is not an include row for {feature} in harmonization.EICU_TABLE")
            group = {"invasive": "inv.", "noninvasive": "non-inv."}.get(modality, "")
            rows[feature].add(files[raw_key], Item(f"{files[raw_key]}.{column}", "", trow.unit, modality, group, column), code)
            vital_features.append(feature)
    for trow in hz.EICU_TABLE:  # pulmonary-artery pressures and CVP
        if trow.namespace in ("vitalPeriodic", "vitalAperiodic") and trow.decision == "exclude":
            rows[trow.feature].exclude(trow.namespace, trow.identifier, trow.label, clean_reason(trow.note))
    if options["nursecharting_temperature"]:
        # the adapter matches this harmonization candidate row (listed there as excluded) literally
        trow = next(r for r in hz.EICU_TABLE if r.namespace == "nurseCharting" and r.feature == "Temp")
        require_literal_in_source(e.EICUAdapter._build_nurse_temperature, '"nursingchartcelltypevalname"', f'"{trow.identifier}"',
                                  'conversion_for("Temp", "degC")')
        rows["Temp"].add(files["nurse_charting"], Item(f"{files['nurse_charting']}.nursingchartcelltypevalname='{trow.identifier}'", "",
                                                       "degC", "nurse-charted", "nurse", str(trow.identifier)),
                         "EICUAdapter._build_nurse_temperature")

    lab_features = []
    for trow in (r for r in hz.EICU_TABLE if r.namespace == "lab" and r.match == "exact"):
        feature, reason = e.classify_lab(trow.identifier, options["include_bedside_glucose"])
        if feature is None:
            rows[trow.feature].exclude("lab", trow.identifier, "", clean_reason(trow.note))
            continue
        extra = "" if reason == "accepted" else reason.removeprefix("accepted: ")
        poc = "point-of-care" in extra
        unit = trow.unit if trow.decision == "include" else None
        rows[feature].add(files["lab"], Item(str(trow.identifier), "", unit, extra, "POC" if poc else ""), "eicu.classify_lab")
        lab_features.append(feature)

    drugs: dict[str, list[str]] = {}
    sedation_set = list(options["sedation_drugs"])
    for feature, names, patterns, code in (("vasopressor_on", list(e.VASOPRESSOR_DRUGS), e.VASOPRESSOR_DRUGS, "eicu.VASOPRESSOR_DRUGS"),
                                           ("sedation_on", sedation_set, e.SEDATIVE_DRUGS, "eicu.SEDATIVE_DRUGS, sedation_drugs default")):
        drugs[feature] = names
        for drug in names:
            got, canonical, _ = e.classify_infusion(drug_label(drug), sedation_set)
            if got != feature or canonical != drug:
                raise RuntimeError(f"eicu.classify_infusion({drug_label(drug)!r}) -> {got}/{canonical}, expected {feature}/{drug}")
            table_pattern = _eicu_include_pattern(feature, drug)
            pattern, origin = (table_pattern, "harmonization pattern") if table_pattern else (patterns[drug], "adapter rule")
            rows[feature].add(files["infusion"], Item(drug, f"/{pattern}/", None, origin, short=drug_label(drug)), code)
        rows[feature].cite("eicu.classify_infusion")
        rows[feature].tex_items = f"drugname patterns ({len(names)} drugs)"
    for drug, pattern in e.SEDATIVE_DRUGS.items():
        if drug not in sedation_set:
            rows["sedation_on"].exclude("infusionDrug", drug, f"/{pattern}/", "not in the configured sedation set (sedation_drugs)")
    for trow in hz.EICU_TABLE:
        if trow.namespace != "infusionDrug" or trow.decision != "exclude":
            continue
        rescued = [drug_label(d) for d in drugs[trow.feature] if re.search(str(trow.identifier), drug_label(d), re.IGNORECASE)]
        reason = clean_reason(trow.note) + (f"; {', '.join(rescued)} accepted by the adapter rule" if rescued else "")
        rows[trow.feature].exclude("infusionDrug", f"/{trow.identifier}/", trow.label, reason)
    regional = f"/{e._REGIONAL.pattern}/, or /{e._EPI_ABBREV.pattern}/ together with /{e._OPIOID_OR_LOCAL.pattern}/"
    for feature in ("vasopressor_on", "sedation_on"):
        rows[feature].exclude("infusionDrug", regional, "epidural/regional preparations", "adapter rule (never vasopressor or sedative)")

    prefix = hz.EICU_URINE_CELLPATH_PREFIX
    urine = rows["urine_output"]
    accepted, rejected = [], []
    for label in hz.EICU_CODE_URINE_LABELS:
        ok, reason = e.urine_cellpath_decision(prefix + label)
        (accepted if ok else rejected).append((label, reason))
    for label, _ in accepted:
        urine.add(files["intake_output"], Item(label, "", "mL", "eicu-code label"), "harmonization.EICU_CODE_URINE_LABELS")
    for pattern in hz.EICU_CODE_URINE_LABEL_PATTERNS:
        urine.add(files["intake_output"], Item(f"/{pattern}/", "", "mL", "eicu-code pattern"), "harmonization.EICU_CODE_URINE_LABEL_PATTERNS")
    urine.add(files["intake_output"], Item(f"foley* (not /{hz._EICU_FOLEY_EXCLUDED.pattern}/)", "", "mL", "eicu-code rule"),
              "harmonization.is_eicu_urine_cellpath")
    urine.add(files["intake_output"], Item(f"/{e.URINE_LABEL_ADDITIONS.pattern}/", "", "mL", "adapter device labels"),
              "eicu.URINE_LABEL_ADDITIONS")
    urine.cite("eicu.urine_cellpath_decision")
    urine.tex_items = f"Output (ml) urine labels: {len(accepted)} eicu-code labels + {len(hz.EICU_CODE_URINE_LABEL_PATTERNS)} patterns, foley*, device labels"
    urine.exclude("intakeOutput", f"/{e.URINE_LABEL_EXCLUSIONS.pattern}/", "", "adapter rule: counts, stool, irrigation, OR, ventriculostomy")
    urine.exclude("intakeOutput", f"/{hz.EICU_URINE_LABEL_EXCLUSION.pattern}/", "", "harmonization rule: counts, unmeasured, shift totals, irrigation")
    for label, reason in rejected:
        urine.exclude("intakeOutput", label, "", reason.removeprefix("excluded: "))
    for trow in hz.EICU_TABLE:
        if trow.feature == "urine_output" and trow.decision == "exclude":
            urine.exclude(trow.namespace, trow.identifier, trow.label, clean_reason(trow.note))
    urine.cite("eicu.URINE_LABEL_EXCLUSIONS", "harmonization.EICU_URINE_LABEL_EXCLUSION")

    # unit handling
    for feature in unique(vital_features):
        row = rows[feature]
        for item in row.items:
            conversion = hz.conversion_for(feature, item.unit)
            if conversion is None or not conversion.is_identity:
                raise RuntimeError(f"eICU {item.ident}: the adapter takes values as canonical but {item.unit!r} is not identity")
        full, short = measurement_units(feature, row.items)
        row.units, row.tex_units = f"{full}; values used as documented (no per-row unit)", short
    for feature in unique(lab_features):
        row = rows[feature]
        documented = [i for i in row.items if i.unit is not None]
        full, short = measurement_units(feature, documented)
        row.units = f"per row: labmeasurenamesystem -> conversion_for (unconvertible rows rejected); documented {full}"
        row.tex_units = short
    gap = float(options["max_gap_hours"])
    run_rule = (f"runs per (stay, drugname): on from the first row with drugrate (else infusionrate) > 0 until the next row with rate 0, "
                f"{gap:g} h after the last on-row when no row follows (max_gap_hours={gap:g}), or unit discharge; text 'OFF' = 0, "
                "other text or negative rates dropped")
    for feature in ("vasopressor_on", "sedation_on"):
        rows[feature].units, rows[feature].tex_units = run_rule, f"charted-rate runs (gap {gap:g} h)"
        rows[feature].cite("eicu.derive_intervals", "eicu.parse_rate")
    urine.units = "cellvaluenumeric (mL per record) of accepted cellpaths; never outputtotal; exact duplicate rows dropped"
    urine.tex_units = "mL per record"

    # notes
    for feature in unique(vital_features):
        if feature in common.BP_FEATURES:
            rows[feature].note("invasive: vitalPeriodic (5-minute monitor medians); non-invasive: vitalAperiodic (cuff)")
        elif any(i.ident.startswith(files["vital_periodic"] + ".") for i in rows[feature].items):
            rows[feature].note("vitalPeriodic: 5-minute monitor medians")
    rows["SpO2"].note("vitalPeriodic.sao2 is pulse oximetry (eICU documentation)")
    if options["nursecharting_temperature"]:
        rows["Temp"].note("nurse-charted Temperature (C) added (nursecharting_temperature=True): monitor temperature exists only "
                          "when a probe is interfaced", "monitor + nurse-charted")
    for feature in unique(lab_features):
        rows[feature].note("exact labname allowlist (harmonization.classify_eicu_lab); latest revision per (stay, labname, "
                           "labresultoffset); time = labresultoffset (draw)")
    poc = [i.ident for i in rows["Glucose"].items if i.group == "POC"]
    if poc:
        rows["Glucose"].note(f"{', '.join(repr(p) for p in poc)} (point of care) included (include_bedside_glucose=True; "
                             "cross-database decision)", "lab + bedside (POC) glucose")
    vaso = rows["vasopressor_on"]
    vaso.note(f"drugs: {', '.join(drug_label(d) for d in drugs['vasopressor_on'])}",
              ", ".join(drug_label(d) for d in drugs["vasopressor_on"]))
    cohort = []
    if options["restrict_hospitals"]:
        cohort.append(f"hospitals where < {options['min_hospital_vasopressor_share']:.0%} of adult stays have a vasopressor interval")
    if options["exclude_ambiguous_controls"]:
        cohort.append("stays with vasopressor rows but no derivable interval, or vasopressor evidence in medication/treatment "
                      "without an infusion")
    if options["require_washout"]:
        cohort.append(f"stays whose first infusion starts > {options['prior_evidence_tolerance_minutes']:g} min after other "
                      "vasopressor evidence")
    if cohort:
        vaso.note("cohort restrictions (EICUAdapter.stays): dropped " + "; ".join(cohort))
        vaso.note("raw-data evidence: " + "; ".join(e.RCHECK_EVIDENCE[k] for k in ("infusion_charting_by_hospital", "vasopressor_evidence_elsewhere")))
        vaso.cite("EICUAdapter._cohort")
    vaso.note(f"raw check R-E2: {e.RCHECK_EVIDENCE['R-E2_infusion']}")
    vaso.cite("eicu.RCHECK_EVIDENCE")
    rows["sedation_on"].note(f"continuous infusions of {', '.join(drug_label(d) for d in sedation_set)} (sedation_drugs default)",
                             ", ".join(drug_label(d) for d in sedation_set) + ("" if "fentanyl" in sedation_set else "; no fentanyl"))
    if "urine_output" in decisions:
        urine.note(f"adapter: {decisions['urine_output']}", "urinary 'Output (ml)' labels; counts excluded")
    for feature, reason in unavailable.items():
        row = rows[feature]
        row.unavailable = reason
        detail = re.sub(r"^not available:\s*", "", decisions.get(feature, ""))
        row.note(f"not available in this source: {reason}" + (f"; {detail}" if detail else "")
                 + "; NaN and masked in every bin (common.episode_matrix), never read as 0", "not available in this source")
        row.cite("EICUAdapter.provenance()['unavailable_features']", "EICUAdapter.interventions")
        for trow in hz.EICU_TABLE:
            if trow.feature == feature and trow.decision == "include":
                row.exclude(trow.namespace, trow.identifier, trow.label, "harmonization include row not used: no reliable start/stop signal")
    return rows


# ------------------------------------------------------------------------------- HiRID


def hirid_rows() -> dict[str, Row]:
    h = hirid
    options = constructor_defaults(h.HiRIDAdapter)
    adapter = h.HiRIDAdapter(UNUSED, UNUSED)  # stores paths and options; reads nothing
    literals = provenance_literals(h.HiRIDAdapter)
    rows = {f: Row(f, "hirid") for f in FEATURES}
    obs = h.TABLE_LAYOUT["observation"]["subdir"].split("/")[0]
    pharma = h.TABLE_LAYOUT["pharma"]["subdir"].split("/")[0]

    for vid in adapter.feature_ids:
        trow = h.OBS_FEATURE_ROWS[vid]
        group = {"invasive": "inv.", "noninvasive": "non-inv."}.get(trow.modality, "")
        rows[trow.feature].add(obs, Item(str(vid), trow.label, trow.unit, trow.modality, group), "hirid.OBS_FEATURE_ROWS (HIRID_TABLE)")
    kind = adapter._ventilation_kind()
    on_classes, off_classes = h.HiRIDAdapter._ventilation_rule_sets()[kind]
    mode_codes = dict(h.VENT_MODES_INVASIVE)
    if "niv" in on_classes:
        mode_codes.update(h.VENT_MODES_NIV)
    airway_codes = dict(h.AIRWAY_INVASIVE)
    vent = rows["ventilation_on"]
    for vid in h.VENT_IDS:
        trow = hz.lookup_identifier("hirid", "observation", vid)
        if vid == h.VENT_MODE_ID:
            tag = "on codes " + ", ".join(f"{c} {n}" for c, n in sorted(mode_codes.items()))
        elif vid == h.AIRWAY_ID:
            tag = "PEEP context: invasive codes " + ", ".join(f"{c} {n}" for c, n in sorted(airway_codes.items()))
        else:
            tag = "PEEP evidence points"
        vent.add(obs, Item(str(vid), trow.label if trow else "", None, tag), "hirid.VENT_IDS")
    vent.cite("hirid.VENT_MODES_INVASIVE", "hirid.AIRWAY_INVASIVE", "HiRIDAdapter._ventilation_kind")
    drugs: dict[str, list[str]] = {}
    for feature, ids, code in (("vasopressor_on", list(h.VASOPRESSOR_ROWS), "hirid.VASOPRESSOR_ROWS (HIRID_TABLE)"),
                               ("sedation_on", list(adapter.sedation_pharmaids()), "HiRIDAdapter.sedation_pharmaids")):
        table_rows = h.VASOPRESSOR_ROWS if feature == "vasopressor_on" else h.SEDATION_ROWS
        drugs[feature] = unique(h.drug_name(table_rows[pid].label) for pid in ids)
        for pid in ids:
            rows[feature].add(pharma, Item(str(pid), table_rows[pid].label, None, h.drug_name(table_rows[pid].label)), code)

    # exclusions: data-rejected ids, option-disabled items and the harmonization look-alikes;
    # ids are namespace-specific (observation 300 = RR, pharma 300 = metronidazole)
    mapped = {("observation", vid) for vid in (*adapter.feature_ids, *h.VENT_IDS)}
    mapped |= {("pharma", pid) for pid in (*h.VASOPRESSOR_ROWS, *adapter.sedation_pharmaids())}
    for trow in hz.HIRID_TABLE:
        ident = int(trow.identifier)
        if (trow.namespace, ident) in mapped:
            continue
        if trow.decision == "exclude":
            reason = clean_reason(trow.note)
        elif trow.namespace == "observation" and ident in h.DATA_EXCLUDED_OBS_IDS:
            reason = "raw-data check: " + re.sub(r"\s*\(R-H\d.*?\)", "", h.DATA_EXCLUDED_OBS_IDS[ident]) + " (table_rr_ids=False)"
            rows[trow.feature].cite("hirid.DATA_EXCLUDED_OBS_IDS")
        elif trow.feature == "sedation_on" and h.drug_name(trow.label) in h.OPIOID_DRUGS:
            reason = "opioid; only with include_opioids=True"
        else:
            reason = "harmonization include row not used by the adapter"
        label, generic = trow.label, h.drug_name(trow.label)
        if trow.namespace == "pharma" and generic not in label.casefold() and generic not in reason.casefold():
            label = f"{label} ({generic})"  # brand name -> generic drug
        rows[trow.feature].exclude(trow.namespace, ident, label, reason)
    niv_modes = {c: n for c, n in h.VENT_MODES_NIV.items() if c not in mode_codes}
    if niv_modes:
        vent.exclude("observation", f"{h.VENT_MODE_ID} codes " + ", ".join(f"{c} {n}" for c, n in sorted(niv_modes.items())), "",
                     "NIV modes; only with include_niv=True")
    vent.exclude("observation", f"{h.VENT_MODE_ID} codes " + ", ".join(f"{c} {clean_reason(n)}" for c, n in sorted(h.VENT_MODES_OFF.items())),
                 "", "ventilator off / stand-by")
    if "niv" not in on_classes:
        vent.exclude("observation", f"{h.AIRWAY_ID} codes " + ", ".join(f"{c} {n}" for c, n in sorted(h.AIRWAY_NIV.items())), "",
                     "NIV airway: PEEP points in this context count as NIV")

    # unit handling
    record_filters = literals.get("record_filters", {})
    obs_bits = record_filters.get("observation_status_dropped_bits", {})
    duplicates = record_filters.get("observation_duplicates", "")
    bits = [b for b in (1 << k for k in range(16)) if h.OBS_DROP_BITS & b]
    if sorted(int(b) for b in obs_bits) != bits:
        raise RuntimeError(f"HiRIDAdapter.provenance record_filters {obs_bits} disagree with OBS_DROP_BITS={h.OBS_DROP_BITS}")
    obs_filter = ("observation status bits " + ", ".join(f"{b} ({obs_bits[str(b)]})" for b in bits) + " dropped; " + duplicates)
    for feature, row in rows.items():
        if hz.CANONICAL_SPECS[feature].kind == "measurement" and row.items:
            row.units, row.tex_units = measurement_units(feature, row.items)
            row.note(obs_filter)
            row.cite("HiRIDAdapter.canonical_events")
    urine = rows["urine_output"]
    max_interval = float(options["urine_max_interval_hours"])
    urine_unit = h.OBS_FEATURE_ROWS[h.URINE_ID].unit
    urine.units = (f"{urine_unit} is a rate: mL = rate x hours since the previous urine record of the patient (first record: since "
                   f"admission), interval clipped to [0, {max_interval:g}] h (urine_max_interval_hours); harmonization's "
                   f"'{hz.conversion_for('urine_output', urine_unit).name}' is not used")
    urine.tex_units = f"rate x interval -> mL (<= {max_interval:g} h)"
    urine.note(obs_filter)
    urine.note(f"{h.URINE_ID} values are rates, not per-record volumes, so each record contributes rate x elapsed time; "
               "the event keeps the record time and volumes are summed per bin", "rate (mL/h) x interval -> mL")
    urine.cite("hirid.urine_volumes", "hirid.URINE_ID")
    back = inspect.signature(h.derive_infusion_intervals).parameters["max_backprojection"].default
    pharma_bits = [b for b in (1 << k for k in range(16)) if h.PHARMA_DROP_BITS & b]
    drug_rule = (f"within an infusionid, givendose at givenat[k] covers (givenat[k-1], givenat[k]]; on while dose > 0; recordstatus "
                 f"bits {', '.join(map(str, pharma_bits))} dropped; records with the start ({h.PHARMA_START}) and stop "
                 f"({h.PHARMA_STOP}) bits together are boluses and never count; single-record infusions ignored; a dosed first "
                 f"record is back-projected (<= {hours(back)})")
    for feature in ("vasopressor_on", "sedation_on"):
        rows[feature].units, rows[feature].tex_units = drug_rule, "dose > 0 segments"
        rows[feature].cite("hirid.derive_infusion_intervals")
    points = inspect.signature(h.classify_ventilation_points).parameters
    merge = inspect.signature(h.ventilation_intervals).parameters
    vent.units = (f"evidence points -> intervals: a PEEP point takes the class of the last {h.VENT_MODE_ID} mode within "
                  f"{hours(points['mode_window'].default)}, else of the last {h.AIRWAY_ID} airway code within "
                  f"{hours(points['airway_window'].default)}, else 'unknown'; points of the classes {sorted(on_classes)} at most "
                  f"{hours(merge['max_gap'].default)} apart with no {sorted(off_classes)} point between form one interval ending "
                  f"min(last point + {hours(merge['carry'].default)}, next point); runs of < {merge['min_points'].default} points dropped")
    vent.tex_units = "mode/PEEP point runs"
    vent.cite("hirid.classify_ventilation_points", "hirid.ventilation_intervals")

    # notes
    rows["RR"].note(f"id {', '.join(i.ident for i in rows['RR'].items)} only: "
                    f"{', '.join(str(i) for i in h.DATA_EXCLUDED_OBS_IDS)} rejected by the raw-data check (table_rr_ids=False)",
                    f"id {', '.join(i.ident for i in rows['RR'].items)} only")
    rows["Temp"].note("core, rectal and axillary sites pooled", "core, rectal, axillary")
    rows["Lactate"].note("arterial and venous lactate pooled", "arterial + venous")
    rows["Glucose"].note(f"the {len(rows['Glucose'].items)} serum/plasma glucose ids HiRID pools; point-of-care/blood-gas glucose "
                         "included by the cross-database decision", f"{len(rows['Glucose'].items)} ids pooled (incl. POC)")
    vaso = rows["vasopressor_on"]
    vaso.note(f"drugs: {', '.join(drug_label(d) for d in drugs['vasopressor_on'])}",
              ", ".join(drug_label(d) for d in drugs["vasopressor_on"]))
    sedation = rows["sedation_on"]
    sedation.note(f"continuous infusions of {', '.join(drugs['sedation_on'])}"
                  + ("" if options["include_opioids"] else "; fentanyl only with include_opioids=True"),
                  ", ".join(drugs["sedation_on"]) + ("" if options["include_opioids"] else "; no fentanyl"))
    vent.note(f"ventilation kind {kind!r} (include_niv={options['include_niv']}, peep_without_context="
              f"{options['peep_without_context']!r}): invasive modes and PEEP without mode/airway context count as on",
              "invasive (mode/PEEP evidence)" + ("" if options["include_niv"] else "; NIV off"))
    return rows


# ------------------------------------------------------------------------------- assembly


def build_feature_rows() -> list[Row]:
    """All (feature, source) rows in harmonization feature order, sources MIMIC-IV, eICU, HiRID."""
    check_cli_builds_adapters_with_defaults()
    by_source = {"mimic_iv": mimic_rows(), "eicu": eicu_rows(), "hirid": hirid_rows()}
    mimic_available = mimic_iv.MimicIVAdapter(UNUSED, UNUSED).available_features()  # constants only
    mapped = {f for f, r in by_source["mimic_iv"].items() if r.items and f in common.CANONICAL_FEATURES}
    if mapped != set(mimic_available):
        raise RuntimeError(f"MIMIC-IV mapped features {sorted(mapped)} differ from available_features() {sorted(mimic_available)}")
    # drugs a source has no item for, against the union over the sources
    for feature in ("vasopressor_on", "sedation_on"):
        canonical_names = {s: {_canonical_drug(s, i) for i in by_source[s][feature].items} for s in SOURCES}
        union = set().union(*canonical_names.values())
        for source in SOURCES:
            missing = ", ".join(drug_label(d) for d in sorted(union - canonical_names[source]))
            if missing:
                by_source[source][feature].note(f"no {SOURCE_LABEL[source]} item for: {missing}", f"no {missing}")
    out: list[Row] = []
    for feature in FEATURES:
        for source in SOURCES:
            row = by_source[source][feature]
            if feature in common.BP_FEATURES and {i.tag for i in row.items} >= set(hz.BP_MODALITIES):
                row.notes.insert(0, "invasive-first per bin: mean of the invasive readings when the bin has any, else of the "
                                    "non-invasive ones (common.episode_matrix)")
                row.tex_notes.insert(0, "invasive-first per bin")
                row.cite("common.episode_matrix")
            if feature == "vasopressor_on":
                row.notes.insert(0, "defines the positive anchor (first infusion start, common.find_positive_anchors); never a "
                                    f"model input (not among the {len(common.CANONICAL_FEATURES)} episode features)")
            if not row.items and not row.unavailable:
                raise RuntimeError(f"{source} maps nothing to {feature} and does not declare it unavailable")
            if row.items and row.unavailable:
                raise RuntimeError(f"{source} declares {feature} unavailable but maps {[i.ident for i in row.items]}")
            out.append(row)
    return out


def _canonical_drug(source: str, item: Item) -> str:
    if source == "eicu":
        return item.ident
    if source == "hirid":
        return item.tag
    return {**mimic_iv.VASOPRESSOR_ITEMS, **mimic_iv.ANGIOTENSIN_ITEMS, **mimic_iv.SEDATIVE_ITEMS, **mimic_iv.OPIOID_ITEMS}[int(item.ident)]


def feature_csv_row(row: Row) -> dict[str, str]:
    spec = hz.CANONICAL_SPECS[row.feature]
    return {
        "canonical_feature": row.feature,
        "canonical_unit": spec.unit,
        "kind": spec.kind,
        "plausible_range": plausible_range(row.feature) if row.items else "",
        "source": source_version(row.source),
        "available": "no" if row.unavailable else "yes",
        "source_tables": LIST_SEP.join(row.tables),
        "source_ids": LIST_SEP.join(i.ident for i in row.items),
        "source_items": LIST_SEP.join(i.text() for i in row.items),
        "unit_handling": row.units,
        "excluded_lookalikes": LIST_SEP.join(x.text() for x in row.lookalikes),
        "notes": LIST_SEP.join(row.notes),
        "code_reference": LIST_SEP.join(row.code),
    }


def tex_items(row: Row) -> str:
    if row.tex_items:
        return row.tex_items
    groups: dict[str, list[str]] = {}
    for item in row.items:
        groups.setdefault(item.group, []).append(item.short or item.ident)
    order = {"inv.": 0, "non-inv.": 1}
    ordered = sorted(groups.items(), key=lambda kv: order.get(kv[0], 2))  # stable: other groups keep their order
    return "; ".join(f"{g}: {abbreviate(v)}" if g else abbreviate(v) for g, v in ordered)


# ------------------------------------------------------------------------------- task rules


def check_episode_semantics(spec: common.ExtractionSpec) -> None:
    """Run one synthetic episode through common.episode_matrix; fail if the rules below no
    longer describe it (window bounds, invasive-first BP, summed urine, unavailable features)."""
    cutoff = pd.Timestamp("2000-01-02 00:00")
    start = cutoff - pd.Timedelta(hours=spec.window_hours)
    minute = pd.Timedelta(minutes=1)
    events = pd.DataFrame(
        [
            (1, start - minute, "HR", 50.0, None), (1, start, "HR", 60.0, None), (1, cutoff - minute, "HR", 70.0, None),
            (1, cutoff, "HR", 999.0, None), (1, start, "MAP", 80.0, "invasive"), (1, start, "MAP", 60.0, "noninvasive"),
            (1, cutoff - minute, "MAP", 65.0, "noninvasive"), (1, start, "urine_output", 100.0, None),
            (1, start + minute, "urine_output", 50.0, None),
        ],
        columns=["stay_id", "time", "feature", "value", "modality"],
    )
    no_interventions = pd.DataFrame(columns=list(common.INTERVENTION_COLUMNS))
    features = ("HR", "MAP", "urine_output", "ventilation_on", "sedation_on")
    values, mask = common.episode_matrix(events, no_interventions, cutoff, spec, features, available={"HR", "MAP", "urine_output", "sedation_on"})
    col = {f: j for j, f in enumerate(features)}
    checks = {
        "n_bins": values.shape[0] == spec.n_bins,
        "window start included, earlier data excluded": values[0, col["HR"]] == 60.0,
        "cutoff excluded": values[-1, col["HR"]] == 70.0 and not (values == 999.0).any(),
        "invasive-first BP": values[0, col["MAP"]] == 80.0 and values[-1, col["MAP"]] == 65.0,
        "urine summed": values[0, col["urine_output"]] == 150.0,
        "unavailable feature NaN and masked": bool(np.isnan(values[:, col["ventilation_on"]]).all() and mask[:, col["ventilation_on"]].all()),
        "available intervention without interval is 0": bool((values[:, col["sedation_on"]] == 0).all() and not mask[:, col["sedation_on"]].any()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"common.episode_matrix no longer matches the task-rules table: {failed}")


def _example_meta(spec: common.ExtractionSpec, kind: str) -> dict[str, Any]:
    row = {"anchor_time": pd.Timestamp("2000-01-03 00:00"), "stay_id": 1, "subject_id": 1,
           "t_since_admit": pd.Timedelta(hours=30), "matched_stay_id": 2}
    return common.build_episode_meta("example", kind, row, spec, common.CANONICAL_FEATURES, frozenset(common.CANONICAL_FEATURES))


def task_rule_rows(feature_rows: Sequence[Row] | None = None) -> list[tuple[str, str, str]]:
    spec = common.ExtractionSpec()
    cli = cli_options()
    for flag, attr in CLI_TO_SPEC.items():
        if flag not in cli or "default" not in cli[flag]:
            raise RuntimeError(f"{CLI_SCRIPT.name} has no default for {flag}")
        default = cli[flag]["default"]
        as_spec = {
            "match_columns": lambda v: tuple(c for c in str(v).split(",") if c),
            "required_vital": lambda v: v or None,
            "max_positives_per_patient": lambda v: None if v == 0 else v,
        }.get(attr, lambda v: v)(default)
        if as_spec != getattr(spec, attr):
            raise RuntimeError(f"CLI {flag} default {default!r} differs from ExtractionSpec.{attr}={getattr(spec, attr)!r}")
    check_episode_semantics(spec)
    positive, control = _example_meta(spec, "positive"), _example_meta(spec, "control")
    anchor = pd.Timestamp(positive["anchor_time"])
    if pd.Timestamp(positive["window_end"]) != anchor - pd.Timedelta(hours=spec.prediction_gap_hours):
        raise RuntimeError("build_episode_meta: window_end is no longer anchor - gap")
    if "vasopressor_on" in common.CANONICAL_FEATURES:
        raise RuntimeError("vasopressor_on became an episode feature; the anchor rule below says otherwise")
    w, gap, r, n = spec.window_hours, spec.prediction_gap_hours, spec.resample_minutes, spec.n_bins
    lookback = spec.lookback.total_seconds() / 3600.0
    try:
        gaps = [g for g in module_literal(ASSETS_SCRIPT, "GAPS") if g != gap]
    except (KeyError, OSError, ValueError):
        gaps = []
    match_help = cli["--match-columns"].get("help", "")
    found = re.search(r"eICU:\s*([A-Za-z_]\w*)", match_help)
    if not found:
        raise RuntimeError("extract_v2_cohort.py --match-columns help no longer names the eICU matching column")
    eicu_column = found.group(1)
    require_literal_in_source(eicu.EICUAdapter._cohort, f'"{eicu_column}"')
    hirid_defaults = constructor_defaults(hirid.HiRIDAdapter)
    unavailable = [f"{SOURCE_LABEL[r.source]}: {r.feature}" for r in feature_rows or () if r.unavailable]
    sums = ", ".join(common.SUM_FEATURES)
    flags = ", ".join(common.INTERVENTION_FEATURES)
    k = spec.required_vital_last_bins
    last_bins = "the last bin" if k == 1 else f"one of the last {k} bins"
    return [
        ("Observation window",
         f"{w} h ending at the cutoff in {n} bins of {r} min, half-open [cutoff - {w} h, cutoff): nothing at or after the cutoff is used",
         f"ExtractionSpec.window_hours={w}, resample_minutes={r}; CLI --window-hours (default {cli['--window-hours']['default']}); "
         "common.bin_edges, common.episode_matrix"),
        ("Prediction gap",
         f"cutoff = anchor - {gap:g} h" + (f"; sensitivity runs: {', '.join(f'{g:g} h' for g in gaps)}" if gaps else ""),
         f"ExtractionSpec.prediction_gap_hours={gap:g}; CLI --gap-hours (default {cli['--gap-hours']['default']:g})"
         + ("; make_paper_assets.GAPS" if gaps else "")),
        ("Positive anchor",
         f"first start of a continuous vasopressor infusion (rate > 0) in the ICU stay (anchor_kind={positive['anchor_kind']!r}); "
         f"vasopressor_on only defines anchors and is not one of the {len(common.CANONICAL_FEATURES)} episode features",
         "common.find_positive_anchors; common.CANONICAL_FEATURES"),
        ("Anchor eligibility",
         f"anchor inside [intime, outtime] and at least window + gap = {lookback:g} h after ICU admission",
         "ExtractionSpec.lookback; common.find_positive_anchors"),
        ("Positives per patient",
         f"{spec.max_positives_per_patient}: the patient's earliest eligible positive stay (ties broken by a seeded hash)",
         f"ExtractionSpec.max_positives_per_patient={spec.max_positives_per_patient}; CLI --max-positives-per-patient "
         f"(default {cli['--max-positives-per-patient']['default']}; 0 keeps all); common.limit_positives_per_patient"),
        ("Control pool",
         "ICU stays without any vasopressor infusion"
         + (", from patients without a vasopressor infusion in any stay" if spec.exclude_vasopressor_patients_from_controls else ""),
         f"ExtractionSpec.exclude_vasopressor_patients_from_controls={spec.exclude_vasopressor_patients_from_controls}; "
         "common.match_controls"),
        ("Landmark matching",
         f"{spec.controls_per_positive} unused control per positive that is still in the ICU at the positive's time since admission; "
         f"pseudo-anchor = control intime + that time (anchor_kind={control['anchor_kind']!r}); seeded hash order (seed {spec.seed})",
         f"ExtractionSpec.controls_per_positive={spec.controls_per_positive}, seed={spec.seed}; CLI --controls-per-positive, --seed; "
         "common.match_controls"),
        ("eICU hospital matching",
         f"eICU controls share the positive's {eicu_column} (--match-columns {eicu_column}); other sources: no matching column",
         f"ExtractionSpec.match_columns={spec.match_columns!r} by default; CLI --match-columns; EICUAdapter.stays() column {eicu_column}"),
        ("Recent-vital rule",
         f"{spec.required_vital} observed in {last_bins} before the cutoff, for positives and controls alike; other episodes are dropped",
         f"ExtractionSpec.required_vital={spec.required_vital!r}, required_vital_last_bins={spec.required_vital_last_bins}; "
         "CLI --required-vital, --required-vital-last-bins; common.vital_observed_near_cutoff"),
        ("Unavailable features",
         "a feature the source does not provide is NaN and masked in every bin, never 0 or 'off'"
         + (f" ({'; '.join(unavailable)})" if unavailable else ""),
         "common.episode_matrix(available=adapter.available_features())"),
        ("Values",
         f"episode values: {positive['values_units']}; per-bin mean, sum for {sums}; SBP/DBP/MAP invasive-first; {flags} = 1 when "
         "an interval overlaps the bin, else 0; values outside the canonical plausible range are dropped at extraction, not repaired",
         "common.episode_matrix; common.build_episode_meta; harmonization.CANONICAL_SPECS"),
        ("ICU stay end",
         f"for landmark matching: MIMIC-IV icustays.outtime; eICU unit discharge offset; HiRID last HR/BP/SpO2/RR record "
         f"(outtime_rule={hirid_defaults['outtime_rule']!r})",
         "MimicIVAdapter.stays, EICUAdapter.stays, HiRIDAdapter.stays"),
    ]


# ------------------------------------------------------------------------------- writers


_TEX_TOKENS = {
    "->": r"$\rightarrow$", "<=": r"$\leq$", ">=": r"$\geq$", "--": "-{}-", "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "|": r"\textbar{}", "<": r"\textless{}", ">": r"\textgreater{}", "×": r"$\times$", "−": r"$-$", "–": "--",
    "°": r"\textdegree{}", "µ": r"\textmu{}", "μ": r"$\mu$", "≤": r"$\leq$", "≥": r"$\geq$", "…": r"\ldots{}", '"': "''",
}
_TEX_PATTERN = re.compile(r"\^\d+|" + "|".join(re.escape(k) for k in sorted(_TEX_TOKENS, key=len, reverse=True)))
#: .tex display of canonical unit tokens (the CSV keeps the tokens)
_TEX_DISPLAY = {"degC": "°C", "degF": "°F", "flag": "0/1"}


def tex_escape(value: Any) -> str:
    """Escape LaTeX specials in one pass (a superset of make_paper_assets' _ % & escaping);
    ``--`` stays two hyphens (CLI flags), ``^9`` becomes a superscript, "–" an en dash."""
    def replace(match: re.Match[str]) -> str:
        token = match.group()
        return f"$^{{{token[1:]}}}$" if token[0] == "^" and token[1:].isdigit() else _TEX_TOKENS[token]

    return _TEX_PATTERN.sub(replace, str(value))


def tex_display(value: str) -> str:
    """Feature-table cell text with readable unit tokens (degC -> °C)."""
    return re.sub(r"\b(" + "|".join(_TEX_DISPLAY) + r")\b", lambda m: _TEX_DISPLAY[m.group()], value)


def tex_code(value: Any) -> str:
    """Escaped code references that may break after '.', '_', '=' and ',' in narrow columns."""
    text = tex_escape(value)
    return re.sub(r"(\\_|\.(?=[A-Za-z_])|=|,)", lambda m: m.group() + r"\allowbreak{}", text)


def _p(width: str) -> str:
    # \hspace{0pt} lets TeX hyphenate the first word of a cell (long table names)
    return rf">{{\raggedright\arraybackslash\hspace{{0pt}}}}p{{{width}\linewidth}}"


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any] | dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[c] for c in header] if isinstance(row, dict) else row)


def write_tex(path: Path, caption: str, colspec: str, header: Sequence[str], body: Sequence[str], packages: str) -> None:
    lines = [
        "% generated by scripts/v2/export_mapping_table.py -- do not edit by hand",
        f"% {caption}",
        f"% needs {packages}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(tex_escape(h) for h in header) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feature_tables(out: Path, rows: Sequence[Row]) -> None:
    write_csv(out / "feature_mappings.csv", FEATURE_COLUMNS, [feature_csv_row(r) for r in rows])
    body: list[str] = []
    for k, feature in enumerate(FEATURES):
        if k:
            body.append("\\addlinespace")
        for j, row in enumerate(r for r in rows if r.feature == feature):
            unit = hz.CANONICAL_SPECS[feature].unit
            label = unit if feature in common.CANONICAL_FEATURES else f"{unit}; anchor only"  # vasopressor_on is no input
            name = (feature if label == feature else f"{feature} ({label})") if j == 0 else ""
            if row.unavailable:
                cells = [name, SOURCE_LABEL[row.source], "–", "–", "–", "; ".join(row.tex_notes) + " (NaN, masked)"]
            else:
                cells = [name, SOURCE_LABEL[row.source], ", ".join(unique(short_table(t) for t in row.tables)), tex_items(row),
                         row.tex_units, "; ".join(row.tex_notes)]
            body.append(" & ".join(tex_escape(tex_display(c)) for c in cells) + " \\\\")
    versions = ", ".join(source_version(s) for s in SOURCES)
    colspec = "@{}" + _p("0.12") + "l" + _p("0.10") + _p("0.24") + _p("0.13") + _p("0.25") + "@{}"
    write_tex(out / "feature_mappings.tex",
              f"Source-to-canonical feature mappings of the v2 extraction ({versions}); abbreviated, full lists in feature_mappings.csv.",
              colspec, ["Feature (unit)", "Source", "Table(s)", "Items", "Conversion", "Notes"], body,
              "\\usepackage{booktabs,array}; set inside table* with \\scriptsize")


def write_rule_tables(out: Path, rules: Sequence[tuple[str, str, str]]) -> None:
    write_csv(out / "task_rules.csv", RULE_COLUMNS, rules)
    body = [" & ".join((tex_escape(rule), tex_escape(setting), tex_code(code))) + " \\\\" for rule, setting, code in rules]
    colspec = "@{}" + _p("0.15") + _p("0.50") + _p("0.29") + "@{}"
    write_tex(out / "task_rules.tex", "Shared task rules of the v2 extraction (common.ExtractionSpec defaults and extract_v2_cohort.py).",
              colspec, ["Rule", "Setting", "Encoded in"], body, "\\usepackage{booktabs,array}; set inside table* with \\scriptsize")


def provenance_lines(out: Path) -> list[str]:
    def asset(name: str) -> str:
        path = out / name
        try:
            return path.relative_to(ASSETS_DIR).as_posix()
        except ValueError:
            return path.as_posix()

    def rel(path: Path) -> str:
        return path.relative_to(ROOT).as_posix()

    adapters = [Path(inspect.getfile(m)) for m in MODULES.values()]
    mapping_inputs = [*adapters, Path(inspect.getfile(hz)), Path(inspect.getfile(common))]
    rule_inputs = [Path(inspect.getfile(common)), CLI_SCRIPT, ASSETS_SCRIPT, *adapters]
    return [
        f"- `{asset('feature_mappings')}.csv|.tex` <- " + ", ".join(f"`{rel(p)}`" for p in mapping_inputs),
        f"- `{asset('task_rules')}.csv|.tex` <- " + ", ".join(f"`{rel(p)}`" for p in rule_inputs),
    ]


def _say(text: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="directory for the four files (default: paper/v2_assets/tables)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    feature_rows = build_feature_rows()
    rules = task_rule_rows(feature_rows)
    out.mkdir(parents=True, exist_ok=True)
    write_feature_tables(out, feature_rows)
    write_rule_tables(out, rules)
    n_unavailable = sum(1 for r in feature_rows if r.unavailable)
    _say(f"feature_mappings: {len(feature_rows)} rows ({len(FEATURES)} features x {len(SOURCES)} sources, {n_unavailable} unavailable); "
         f"task_rules: {len(rules)} rows -> {out.as_posix()}")
    _say("Provenance lines for paper/v2_assets/SOURCES.md (make_paper_assets.py rewrites that file, so add them by hand):")
    for line in provenance_lines(out):
        _say(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
