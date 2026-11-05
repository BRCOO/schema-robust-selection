#!/usr/bin/env python3
"""Subgroup analysis (sex, age) of the final v2 track: discovery registries and supervised references.

A reviewer asked whether "performance and program survival differ across demographic or
clinical subgroups". For the internal MIMIC-IV TEST split and the external eICU and HiRID
cohorts this script reports, per subgroup (sex F / M; age at admission < 65 / >= 65):

* n episodes, n patients, prevalence, AUPRC and AUROC with patient-clustered bootstrap 95% CIs
  (``src.icu.v2.stats.patient_cluster_bootstrap``; subject_id resampled within the subgroup);
* for every discovery method x {native, v2_selector}: firing coverage (share of the subgroup's
  episodes on which at least one selected, schema-survivable program fires) and firing
  survival (share of the registry that is survivable and fires on at least one of them).
  Schema survival depends only on the target schema, so it is the same in every subgroup;
* for every supervised reference: seed-0 point estimates with CIs, plus mean and SD over seeds.

Nothing is mined, selected or refit anew. Registries are rebuilt from the cached pools exactly as
``compare_methods`` builds them (native top-K; the v2 selector on TRAIN firings with the
comparison's selector config and selector target schemas), and :func:`bridge_scores` repeats the
calls of ``evaluate_registry`` (``fit_frozen_bridge_v2`` with the default C grid, then
``refire_external``) to expose the external per-episode probabilities that ``evaluate_registry``
only summarizes. The all-episode AUPRC/AUROC must reproduce comparison_rows.csv (and the
supervised summary) to 1e-9; otherwise the script stops before writing any output.

Demographics are read-only; raw and cache paths come from
``data/processed/v2/extract/<source>_gap<gap>h_full/extraction_manifest.json``:

* MIMIC-IV: sex = ``hosp/patients.csv.gz`` gender (read from the MIMIC zip by the adapter's own
  reader); age = anchor_age + (year(intime) - anchor_year) from the adapter's cached stays table.
  This is approximate: year resolution only, and anchor_age is top-coded (91 for patients >= 89).
* eICU: ``patient.csv.gz`` gender and age per unit stay; the text "> 89" becomes 90; "Unknown",
  "Other" and blank gender count as unknown sex.
* HiRID: ``general_table.csv`` sex and age. HiRID releases age in 5-year bins (labels 20..90,
  90 = 90 and older); the label is taken as the bin's lower bound, so ">= 65" = labels 65..90.

Episodes with unknown sex (or age) are left out of that variable's subgroups and counted in
subgroup_demographics.csv.

  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" /d/BRCoo/miniconda/python scripts/v2/run_subgroup_analysis.py \\
      --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl \\
      --supervised-dir artifacts/v2/supervised/f_mimic_iv_gap1h \\
      --out-dir artifacts/v2/subgroups/f_mimic_iv_gap1h --gap 1

Outputs in --out-dir: subgroups.csv, subgroup_demographics.csv, summary.md, manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.bridge import DEFAULT_C_GRID, compile_bridge, fit_frozen_bridge_v2, refire_external  # noqa: E402
from src.icu.v2.pipeline import (  # noqa: E402
    SELECTIONS,
    SPLITS,
    MethodPool,
    TargetCohort,
    episodes_by_split,
    native_registry,
    restrict_firings,
    selector_registry,
)
from src.icu.v2.selector import SelectorConfig  # noqa: E402
from src.icu.v2.stats import metric_value, patient_cluster_bootstrap  # noqa: E402
from src.icu.v2.types import EpisodeRecord, ProgramRecord  # noqa: E402

INTERNAL_SOURCE = "mimic_iv"
INTERNAL_COHORT = "mimic_iv_test"
AGE_CUT = 65.0
UNKNOWN = "unknown"
SUBGROUPS: tuple[tuple[str, str], ...] = (("all", "all"), ("sex", "F"), ("sex", "M"), ("age", "<65"), ("age", ">=65"))
#: (variable, first level, second level); differences are first minus second
CONTRASTS: tuple[tuple[str, str, str], ...] = (("sex", "F", "M"), ("age", ">=65", "<65"))
METRICS: tuple[str, ...] = ("auprc", "auroc")
SANITY_TOLERANCE = 1e-9
#: v2 main configuration, used when the comparison config.json has no "selector" entry
DEFAULT_SELECTOR: dict[str, Any] = {"care_process_cap": None}
DEFAULT_TARGETS = "eicu,hirid"
DEFAULT_HEADLINE = "fmg_bridge:native,shapelet:v2_selector"
EXTRACT_SUBDIR = "{source}_gap{gap}h_full"
TEST_PREDICTIONS = "supervised_v2_test_predictions.csv"
SUPERVISED_SUMMARY = "supervised_v2_summary.json"
DISPLAY = {
    "fmg_bridge": "FMG-Bridge",
    "prefixspan": "PrefixSpan",
    "erminer": "ERMiner",
    "matrix_profile": "Matrix profile",
    "shapelet": "Shapelet",
    "native": "native",
    "v2_selector": "v2 selector",
}
COLUMNS = [
    "cohort", "family", "method", "selection", "subgroup_var", "subgroup",
    "n_episodes", "n_patients", "n_positive", "prevalence",
    "auprc", "auprc_low", "auprc_high", "auroc", "auroc_low", "auroc_high", "n_boot_used",
    "n_programs", "firing_coverage", "n_programs_firing", "firing_survival", "schema_survival",
    "auprc_seed_mean", "auprc_seed_sd", "auroc_seed_mean", "auroc_seed_sd", "n_seeds", "ci_seed", "error",
]
DEMO_COLUMNS = ["cohort", "subgroup_var", "subgroup", "label", "n_episodes", "n_patients", "age_median", "age_q1", "age_q3"]


# ---------------------------------------------------------------------------------- demographics


def normalize_sex(value: object) -> str | None:
    """'F' / 'Female' -> 'F', 'M' / 'Male' -> 'M' (any case); Unknown, Other, blank, missing -> None."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("f", "female"):
        return "F"
    if text in ("m", "male"):
        return "M"
    return None


def _finite_or_nan(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def eicu_age(value: object) -> float:
    """eICU ``patient.age`` is text: "> 89" -> 90; a number -> float; blank or other text -> NaN."""
    if value is None:
        return math.nan
    text = str(value).strip()
    if text == "> 89":
        return 90.0
    return _finite_or_nan(text)


def hirid_age(value: object) -> float:
    """HiRID ``general_table.age``: a 5-year bin label (90 = 90+), used as the bin's lower bound."""
    return _finite_or_nan(value)


def mimic_ages(anchor_age: Any, anchor_year: Any, intime: Any) -> np.ndarray:
    """MIMIC-IV age at ICU admission, approximate: anchor_age + (year(intime) - anchor_year).

    Same formula as the MIMIC-IV adapter's cached ``age_at_intime``. Year resolution only, and
    anchor_age is top-coded in MIMIC-IV (91 for patients aged 89 or more).
    """
    years = pd.to_datetime(pd.Series(np.asarray(intime)), errors="coerce").dt.year.to_numpy(dtype=float)
    return np.asarray(anchor_age, dtype=float) + (years - np.asarray(anchor_year, dtype=float))


def age_group(age: object, cut: float = AGE_CUT) -> str | None:
    """'<65' / '>=65' for a finite age; None for a missing age."""
    value = _finite_or_nan(age)
    if not math.isfinite(value):
        return None
    return "<65" if value < cut else ">=65"


def mimic_demographics(stays: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Per ICU stay: stay_id, subject_id, sex (patients.gender), age (:func:`mimic_ages`)."""
    gender = patients.drop_duplicates("subject_id").set_index("subject_id")["gender"]
    frame = pd.DataFrame(
        {
            "stay_id": stays["stay_id"].to_numpy(dtype=np.int64),
            "subject_id": stays["subject_id"].to_numpy(dtype=np.int64),
            "age": mimic_ages(stays["anchor_age"], stays["anchor_year"], stays["intime"]),
        }
    )
    frame["sex"] = [normalize_sex(value) for value in frame["subject_id"].map(gender)]
    return frame


def eicu_demographics(patient: pd.DataFrame) -> pd.DataFrame:
    """Per unit stay (patientunitstayid): sex and age as read from ``patient.csv.gz``."""
    return pd.DataFrame(
        {
            "stay_id": patient["patientunitstayid"].to_numpy(dtype=np.int64),
            "sex": [normalize_sex(value) for value in patient["gender"]],
            "age": np.array([eicu_age(value) for value in patient["age"]], dtype=float),
        }
    )


def hirid_demographics(general: pd.DataFrame) -> pd.DataFrame:
    """Per patient (HiRID stay_id = patientid): sex and binned age from ``general_table.csv``."""
    return pd.DataFrame(
        {
            "stay_id": general["patientid"].to_numpy(dtype=np.int64),
            "sex": [normalize_sex(value) for value in general["sex"]],
            "age": np.array([hirid_age(value) for value in general["age"]], dtype=float),
        }
    )


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def adapter_paths(extract_root: Path, source: str, gap: int) -> dict[str, Any]:
    """Raw, cache and zip paths of the adapter that produced the gap's extraction."""
    manifest = _resolve(extract_root) / EXTRACT_SUBDIR.format(source=source, gap=gap) / "extraction_manifest.json"
    adapter = json.loads(manifest.read_text(encoding="utf-8"))["adapter"]
    return {"manifest": manifest, **{key: _resolve(adapter[key]) for key in ("raw_dir", "cache_dir", "zip_path") if adapter.get(key)}}


def load_mimic_demographics(paths: Mapping[str, Any]) -> pd.DataFrame:
    from src.icu.v2.extract.mimic_iv import MimicIVAdapter

    adapter = MimicIVAdapter(paths["raw_dir"], paths["cache_dir"], zip_path=paths.get("zip_path"), verbose=False)
    # Read-only use of the adapter: refuse a missing or stale stays cache instead of letting
    # _load() rebuild it (rebuilding is the extraction's job, not this analysis').
    entry = adapter._read_manifest().get("stays", {})
    if not adapter._cache_path("stays").exists() or entry.get("key") != adapter._cache_key("stays"):
        raise RuntimeError(f"MIMIC-IV stays cache under {adapter.cache_dir} is missing or stale; rebuild it with the extraction first")
    stays = adapter._load("stays")
    patients = pd.concat(adapter._read_csv("hosp/patients.csv.gz", usecols=["subject_id", "gender"]), ignore_index=True)
    frame = mimic_demographics(stays, patients)
    if not np.array_equal(frame["age"].to_numpy(dtype=float), stays["age_at_intime"].to_numpy(dtype=float), equal_nan=True):
        raise RuntimeError("recomputed MIMIC-IV age differs from the adapter's cached age_at_intime")
    return frame


def load_eicu_demographics(paths: Mapping[str, Any]) -> pd.DataFrame:
    from src.icu.v2.extract.eicu import RAW_FILES

    patient = pd.read_csv(
        Path(paths["raw_dir"]) / RAW_FILES["patient"],
        usecols=["patientunitstayid", "gender", "age"],
        dtype={"gender": str, "age": str},
    )
    return eicu_demographics(patient)


def load_hirid_demographics(paths: Mapping[str, Any]) -> pd.DataFrame:
    general = pd.read_csv(Path(paths["raw_dir"]) / "general_table.csv", usecols=["patientid", "sex", "age"])
    return hirid_demographics(general)


LOADERS = {"mimic_iv": load_mimic_demographics, "eicu": load_eicu_demographics, "hirid": load_hirid_demographics}


def load_demographics(source: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    """Per-stay demographics of one source: from --demographics if given, else from the raw tables."""
    if args.demographics:
        frame = pd.read_csv(args.demographics)
        frame = frame[frame["source"].astype(str) == source].drop(columns="source").reset_index(drop=True)
        return frame, {"from": str(args.demographics)}
    if source not in LOADERS:
        raise SystemExit(f"no demographics loader for source {source!r}; --targets must name real sources ({sorted(LOADERS)})")
    paths = adapter_paths(Path(args.extract_root), source, args.gap)
    return LOADERS[source](paths), {key: str(value) for key, value in paths.items()}


# ---------------------------------------------------------------------------------- cohorts


def subgroup_levels(sex: Sequence[object], age: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    """Per episode: sex level ('F' / 'M' / 'unknown') and age level ('<65' / '>=65' / 'unknown')."""
    sex_level = np.array([normalize_sex(value) or UNKNOWN for value in sex], dtype=object)
    age_level = np.array([age_group(value) or UNKNOWN for value in age], dtype=object)
    return sex_level, age_level


def subgroup_masks(sex: Sequence[object], age: Sequence[object]) -> dict[tuple[str, str], np.ndarray]:
    """Boolean episode masks for every entry of :data:`SUBGROUPS` (unknowns are in no subgroup)."""
    sex_level, age_level = subgroup_levels(sex, age)
    masks = {("all", "all"): np.ones(len(sex_level), dtype=bool)}
    for variable, level in SUBGROUPS[1:]:
        masks[(variable, level)] = (sex_level if variable == "sex" else age_level) == level
    return masks


@dataclass
class Cohort:
    """Episodes of one evaluation cohort (in pipeline order) with per-episode sex and age."""

    name: str
    episodes: tuple[EpisodeRecord, ...]
    sex: np.ndarray
    age: np.ndarray
    audit: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.episodes = tuple(self.episodes)
        self.ids = [episode.episode_id for episode in self.episodes]
        self.y = np.array([episode.label for episode in self.episodes], dtype=int)
        self.groups = np.array([episode.subject_id for episode in self.episodes], dtype=np.int64)
        self.sex_level, self.age_level = subgroup_levels(self.sex, self.age)
        self.masks = subgroup_masks(self.sex, self.age)


def attach_demographics(name: str, episodes: Sequence[EpisodeRecord], demographics: pd.DataFrame) -> Cohort:
    """Join per-stay demographics onto episodes by stay_id (and check subject_id when present)."""
    if demographics["stay_id"].duplicated().any():
        raise ValueError(f"{name}: demographics have duplicate stay_id values")
    table = demographics.set_index(demographics["stay_id"].to_numpy(dtype=np.int64))
    stay_ids = np.array([episode.stay_id for episode in episodes], dtype=np.int64)
    found = np.isin(stay_ids, table.index.to_numpy())
    sex = table["sex"].reindex(stay_ids).to_numpy(dtype=object)
    age = table["age"].reindex(stay_ids).to_numpy(dtype=float)
    audit: dict[str, Any] = {
        "episodes": len(stay_ids),
        "patients": int(np.unique([episode.subject_id for episode in episodes]).size),
        "stays": int(np.unique(stay_ids).size),
        "stays_without_demographics": int(np.unique(stay_ids[~found]).size),
    }
    if "subject_id" in table.columns:
        known = table["subject_id"].reindex(stay_ids[found]).to_numpy(dtype=np.int64)
        episode_subjects = np.array([episode.subject_id for episode in episodes], dtype=np.int64)[found]
        mismatches = int((known != episode_subjects).sum())
        audit["subject_id_mismatches"] = mismatches
        if mismatches:
            raise ValueError(f"{name}: {mismatches} episodes whose stay belongs to another subject_id in the demographics")
    cohort = Cohort(name=name, episodes=tuple(episodes), sex=sex, age=age, audit=audit)
    audit["unknown_sex_episodes"] = int((cohort.sex_level == UNKNOWN).sum())
    audit["unknown_age_episodes"] = int((cohort.age_level == UNKNOWN).sum())
    return cohort


# ---------------------------------------------------------------------------------- scoring


def score_subgroup(y: Any, p: Any, groups: Any, *, n_boot: int, seed: int) -> dict[str, Any]:
    """Counts, prevalence, and AUPRC / AUROC with patient-clustered bootstrap percentile CIs."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    out: dict[str, Any] = {
        "n_episodes": int(y.size),
        "n_patients": int(np.unique(groups).size),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()) if y.size else math.nan,
    }
    if np.unique(y).size < 2:
        out["error"] = "empty subgroup" if y.size == 0 else "single-class subgroup"
        return out
    for metric in METRICS:
        result = patient_cluster_bootstrap(y, p, groups, metric=metric, n_boot=n_boot, seed=seed)
        out.update({metric: result["point"], f"{metric}_low": result["low"], f"{metric}_high": result["high"]})
        out["n_boot_used"] = result["n_boot_used"]
    return out


def firing_stats(active: Any) -> dict[str, Any]:
    """Firing coverage and survival of a registry's activations (rows = episodes, columns = programs)."""
    active = np.asarray(active, dtype=bool)
    n_programs = int(active.shape[1])
    fired = int(active.any(axis=0).sum()) if active.shape[0] else 0
    return {
        "n_programs": n_programs,
        "firing_coverage": float(active.any(axis=1).mean()) if active.shape[0] else math.nan,
        "n_programs_firing": fired,
        "firing_survival": fired / n_programs if n_programs else 0.0,
    }


def subgroup_rows(cohort: Cohort, probabilities: np.ndarray, *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for (variable, level), mask in cohort.masks.items():
        row: dict[str, Any] = {"cohort": cohort.name, "subgroup_var": variable, "subgroup": level}
        row.update(score_subgroup(cohort.y[mask], probabilities[mask], cohort.groups[mask], n_boot=n_boot, seed=seed))
        rows.append(row)
    return rows


def bridge_scores(
    registry: Sequence[ProgramRecord],
    pool: MethodPool,
    episodes: Sequence[EpisodeRecord],
    targets: Sequence[TargetCohort],
    *,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    metric: str = "auprc",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Per-episode probabilities and survival-masked activations: internal TEST and every target.

    Repeats ``evaluate_registry`` (src/icu/v2/pipeline.py) call for call: the same split grouping,
    the same per-split restriction of the registry's firings, ``fit_frozen_bridge_v2`` with the
    same grid and metric, and ``refire_external`` on ``pool.external_firings[target]``. It also
    returns the external probabilities and the bridge activations, which ``evaluate_registry``
    does not expose. Raises ``ValueError`` like the bridge (empty registry, single-class split).
    """
    registry = list(registry)
    if not registry:
        raise ValueError("empty registry")
    grouped = episodes_by_split(episodes)
    ids = {name: [episode.episode_id for episode in rows] for name, rows in grouped.items()}
    id_sets = {name: frozenset(values) for name, values in ids.items()}
    selected_firings = {program.program_id: frozenset(pool.firings.get(program.program_id, frozenset())) for program in registry}
    bridge = fit_frozen_bridge_v2(
        registry,
        {name: restrict_firings(selected_firings, id_sets[name]) for name in SPLITS},
        ids,
        {name: [episode.label for episode in rows] for name, rows in grouped.items()},
        c_grid=c_grid,
        metric=metric,
    )
    matrix, _ = compile_bridge(registry, restrict_firings(selected_firings, id_sets["test"]), ids["test"])
    scores: dict[str, dict[str, Any]] = {
        INTERNAL_COHORT: {
            "episode_ids": ids["test"],
            "probabilities": np.array([bridge["test_probabilities"][eid] for eid in ids["test"]], dtype=float),
            "active": matrix > 0,
            "schema_survival": math.nan,  # the internal schema is the mining schema
        }
    }
    external = dict(pool.external_firings or {})
    for target in targets:
        firings = external.get(target.name, {})
        episode_ids = [episode.episode_id for episode in target.episodes]
        refired = refire_external(bridge["model_bundle"], registry, target.features, firings, episode_ids)
        matrix, _ = compile_bridge(registry, firings, episode_ids)
        matrix[:, ~np.asarray(refired["survival_mask"], dtype=bool)] = 0.0  # as refire_external does
        scores[target.name] = {
            "episode_ids": episode_ids,
            "probabilities": np.array([refired["probabilities"][eid] for eid in episode_ids], dtype=float),
            "active": matrix > 0,
            "schema_survival": refired["schema_survival"],
            "suppressed_firings": refired["suppressed_firings"],
        }
    info = {"selected_c": bridge["selected_c"], "validation_score": bridge["validation_score"]}
    return scores, info


def discovery_rows(
    state: Mapping[str, Any],
    cohorts: Mapping[str, Cohort],
    targets: Sequence[TargetCohort],
    *,
    k: int,
    selector_config: SelectorConfig,
    selector_targets: Sequence[str],
    n_boot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """subgroups.csv rows for every (method, selection), plus a description of each registry."""
    episodes = state["episodes"]
    train_ids = frozenset(episode.episode_id for episode in episodes_by_split(episodes)["train"])
    allowed = set(selector_targets)
    schemas = {target.name: target.features for target in state["targets"] if target.name in allowed}
    rows: list[dict[str, Any]] = []
    registries: list[dict[str, Any]] = []
    for pool in state["pools"]:
        for selection in SELECTIONS:
            started = time.perf_counter()
            if selection == "native":
                registry = native_registry(pool, k)
            else:
                registry, _ = selector_registry(pool, selector_config, train_ids, schemas)
            tags = {"family": "discovery", "method": pool.method, "selection": selection}
            entry = {**tags, "n_programs": len(registry), "program_ids": [program.program_id for program in registry]}
            registries.append(entry)
            try:
                scores, info = bridge_scores(registry, pool, episodes, targets)
            except ValueError as exc:
                entry["error"] = f"bridge: {exc}"
                rows += [{**tags, "cohort": name, "subgroup_var": "all", "subgroup": "all", "n_programs": len(registry), "error": entry["error"]} for name in cohorts]
                continue
            entry.update(info)
            entry["suppressed_firings"] = {name: block.get("suppressed_firings", {}) for name, block in scores.items() if block.get("suppressed_firings")}
            for name, cohort in cohorts.items():
                block = scores[name]
                if block["episode_ids"] != cohort.ids:
                    raise ValueError(f"{pool.method}/{selection}: episode order of {name} differs from the cohort")
                for row in subgroup_rows(cohort, block["probabilities"], n_boot=n_boot, seed=seed):
                    row.update(firing_stats(block["active"][cohort.masks[(row["subgroup_var"], row["subgroup"])]]))
                    row["schema_survival"] = block["schema_survival"]
                    rows.append({**tags, **row})
            print(f"[subgroups] {pool.method}/{selection}: {len(registry)} programs, C={info['selected_c']}, {time.perf_counter() - started:.0f}s", flush=True)
    return rows, registries


# ---------------------------------------------------------------------------------- supervised


def load_supervised(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any] | None, dict[str, str]]:
    """Internal test predictions, external predictions (one CSV with a 'cohort' column) and summary."""
    test_path = directory / TEST_PREDICTIONS
    external_paths = sorted(directory.glob("*external*predictions*.csv"))
    if len(external_paths) > 1:
        raise ValueError(f"several external prediction files in {directory}: {[p.name for p in external_paths]}")
    test = pd.read_csv(test_path, dtype={"episode_id": str, "model": str})
    external = pd.read_csv(external_paths[0], dtype={"episode_id": str, "model": str, "cohort": str}) if external_paths else None
    for frame, required, path in ((test, {"model", "seed", "episode_id", "label", "prob"}, test_path), (external, {"model", "seed", "cohort", "episode_id", "label", "prob"}, external_paths[0] if external_paths else None)):
        if frame is not None and not required <= set(frame.columns):
            raise ValueError(f"{path} lacks columns {sorted(required - set(frame.columns))}")
    summary_path = directory / SUPERVISED_SUMMARY
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    files = {"test_predictions": str(test_path), "external_predictions": str(external_paths[0]) if external_paths else "", "summary": str(summary_path) if summary else ""}
    return test, external, summary, files


def align_predictions(frame: pd.DataFrame, cohort: Cohort) -> np.ndarray:
    """Probabilities in cohort order; the predictions must cover exactly the cohort's episodes."""
    ids = frame["episode_id"].astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate episode ids in the predictions for {cohort.name}")
    expected = set(cohort.ids)
    missing, extra = len(expected - set(ids)), len(set(ids) - expected)
    if missing or extra:
        raise ValueError(f"predictions for {cohort.name}: {missing} cohort episodes without a prediction, {extra} predictions outside the cohort")
    order = pd.Series(np.arange(len(ids)), index=ids).reindex(cohort.ids).to_numpy()
    if not np.array_equal(frame["label"].to_numpy(dtype=int)[order], cohort.y):
        raise ValueError(f"labels in the predictions differ from the {cohort.name} labels")
    if "subject_id" in frame.columns and not np.array_equal(frame["subject_id"].to_numpy(dtype=np.int64)[order], cohort.groups):
        raise ValueError(f"subject ids in the predictions differ from the {cohort.name} subject ids")
    return frame["prob"].to_numpy(dtype=float)[order]


def supervised_rows(
    test: pd.DataFrame,
    external: pd.DataFrame | None,
    cohorts: Mapping[str, Cohort],
    *,
    n_boot: int,
    seed: int,
    ci_seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per (model, cohort, subgroup): seed-``ci_seed`` point and CIs plus seed mean / SD; and the
    all-episode metrics of every (model, cohort, seed) for the sanity check."""
    frames: dict[str, pd.DataFrame] = {INTERNAL_COHORT: test}
    if external is not None:
        frames.update({name: external[external["cohort"] == name] for name in cohorts if name != INTERNAL_COHORT})
    rows: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []
    for model in dict.fromkeys(test["model"].astype(str)):
        started = time.perf_counter()
        for name, cohort in cohorts.items():
            frame = frames.get(name)
            part = None if frame is None else frame[frame["model"].astype(str) == model]
            if part is None or part.empty:
                continue
            per_seed = {int(value): align_predictions(group, cohort) for value, group in part.groupby("seed", sort=True)}
            for value, probabilities in per_seed.items():
                overall.append({"model": model, "cohort": name, "seed": value, **{m: metric_value(cohort.y, probabilities, m) for m in METRICS}})
            used = ci_seed if ci_seed in per_seed else min(per_seed)
            for row in subgroup_rows(cohort, per_seed[used], n_boot=n_boot, seed=seed):
                mask = cohort.masks[(row["subgroup_var"], row["subgroup"])]
                y = cohort.y[mask]
                for metric in METRICS:
                    values = [metric_value(y, probabilities[mask], metric) for probabilities in per_seed.values()] if np.unique(y).size == 2 else []
                    row[f"{metric}_seed_mean"] = float(np.mean(values)) if values else math.nan
                    row[f"{metric}_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
                row.update({"family": "supervised", "method": model, "selection": "supervised", "n_seeds": len(per_seed), "ci_seed": used})
                rows.append(row)
        print(f"[subgroups] supervised {model}: {time.perf_counter() - started:.0f}s", flush=True)
    return rows, overall


def best_supervised(summary: Mapping[str, Any] | None, overall: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Best supervised model by seed-mean VALIDATION AUPRC (falls back to internal test AUPRC)."""
    scores = {name: (block.get("over_seeds") or {}).get("val_auprc_mean") for name, block in ((summary or {}).get("models") or {}).items()}
    scores = {name: float(value) for name, value in scores.items() if value is not None}
    if scores:
        model = max(scores, key=lambda name: scores[name])
        return {"model": model, "criterion": "highest seed-mean validation AUPRC (supervised summary)", "value": scores[model]}
    means: dict[str, list[float]] = {}
    for entry in overall:
        if entry["cohort"] == INTERNAL_COHORT:
            means.setdefault(entry["model"], []).append(entry["auprc"])
    if not means:
        return None
    model = max(means, key=lambda name: float(np.mean(means[name])))
    return {"model": model, "criterion": "highest seed-mean internal TEST AUPRC (no validation summary found)", "value": float(np.mean(means[model]))}


# ---------------------------------------------------------------------------------- sanity checks


def _num(value: object) -> float:
    if value is None or value == "":
        return math.nan
    return _finite_or_nan(value)


def check_discovery(rows: Sequence[Mapping[str, Any]], comparison_csv: Path) -> list[dict[str, Any]]:
    """All-episode metrics and survival vs comparison_rows.csv, per (method, selection, cohort)."""
    with open(comparison_csv, newline="", encoding="utf-8") as handle:
        reference = {(row["method"], row["selection"]): row for row in csv.DictReader(handle)}
    checks = []
    for row in rows:
        if row["family"] != "discovery" or row["subgroup_var"] != "all":
            continue
        ref = reference.get((row["method"], row["selection"]), {})
        prefix = "" if row["cohort"] == INTERNAL_COHORT else f"{row['cohort']}_"
        quantities = [(metric, prefix + metric) for metric in METRICS] + [("n_programs", "n_programs")]
        if row["cohort"] != INTERNAL_COHORT:
            quantities += [("firing_survival", prefix + "firing_survival"), ("schema_survival", prefix + "schema_survival")]
        for quantity, column in quantities:
            checks.append(
                {
                    "reference": "comparison_rows.csv",
                    "method": row["method"],
                    "selection": row["selection"],
                    "cohort": row["cohort"],
                    "quantity": quantity,
                    "expected": _num(ref.get(column)),
                    "recomputed": _num(row.get(quantity)),
                }
            )
    return checks


def check_supervised(overall: Sequence[Mapping[str, Any]], summary: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """All-episode AUPRC / AUROC per (model, seed, cohort) vs the supervised summary JSON."""
    checks: list[dict[str, Any]] = []
    models = (summary or {}).get("models") or {}
    for entry in overall:
        seeds = {int(record["seed"]): record for record in (models.get(entry["model"]) or {}).get("seeds", [])}
        record = seeds.get(entry["seed"])
        if record is None:
            continue
        block = record.get("test_metrics") if entry["cohort"] == INTERNAL_COHORT else (record.get("external_metrics") or {}).get(entry["cohort"])
        for metric in METRICS:
            if block and block.get(metric) is not None:
                checks.append(
                    {
                        "reference": SUPERVISED_SUMMARY,
                        "method": entry["model"],
                        "selection": f"seed{entry['seed']}",
                        "cohort": entry["cohort"],
                        "quantity": metric,
                        "expected": float(block[metric]),
                        "recomputed": float(entry[metric]),
                    }
                )
    return checks


def finalize_checks(checks: list[dict[str, Any]], tolerance: float = SANITY_TOLERANCE) -> dict[str, Any]:
    for check in checks:
        diff = abs(check["expected"] - check["recomputed"])
        check["abs_diff"] = diff
        check["passed"] = bool(math.isfinite(diff) and diff <= tolerance)
    by_reference: dict[str, dict[str, Any]] = {}
    for check in checks:
        block = by_reference.setdefault(check["reference"], {"n_checks": 0, "n_failed": 0, "max_abs_diff": 0.0})
        block["n_checks"] += 1
        block["n_failed"] += int(not check["passed"])
        if math.isfinite(check["abs_diff"]):
            block["max_abs_diff"] = max(block["max_abs_diff"], check["abs_diff"])
    return {
        "tolerance": tolerance,
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "by_reference": by_reference,
        "failed": [check for check in checks if not check["passed"]],
        "checks": checks,
    }


# ---------------------------------------------------------------------------------- outputs


def demographics_rows(cohort: Cohort) -> list[dict[str, Any]]:
    """Counts per cohort x subgroup (including 'unknown') x label, with age quartiles."""
    levels = [("all", "all", np.ones(len(cohort.ids), dtype=bool))]
    levels += [("sex", level, cohort.sex_level == level) for level in ("F", "M", UNKNOWN)]
    levels += [("age", level, cohort.age_level == level) for level in ("<65", ">=65", UNKNOWN)]
    rows = []
    for variable, level, mask in levels:
        for label in ("all", 1, 0):
            selected = mask if label == "all" else mask & (cohort.y == label)
            ages = cohort.age[selected]
            ages = ages[np.isfinite(ages)]
            q1, median, q3 = np.percentile(ages, [25, 50, 75]) if ages.size else (math.nan, math.nan, math.nan)
            rows.append(
                {
                    "cohort": cohort.name,
                    "subgroup_var": variable,
                    "subgroup": level,
                    "label": label,
                    "n_episodes": int(selected.sum()),
                    "n_patients": int(np.unique(cohort.groups[selected]).size),
                    "age_median": float(median),
                    "age_q1": float(q1),
                    "age_q3": float(q3),
                }
            )
    return rows


def _csv_value(value: object) -> object:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> None:
    extra = sorted({key for row in rows for key in row} - set(columns))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns) + extra)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def sort_rows(rows: list[dict[str, Any]], cohort_order: Sequence[str]) -> list[dict[str, Any]]:
    configs = {key: i for i, key in enumerate(dict.fromkeys((r["family"], r["method"], r["selection"]) for r in rows))}
    subgroups = {key: i for i, key in enumerate(SUBGROUPS)}
    cohorts = {name: i for i, name in enumerate(cohort_order)}
    return sorted(rows, key=lambda r: (cohorts.get(r["cohort"], len(cohorts)), configs[(r["family"], r["method"], r["selection"])], subgroups.get((r["subgroup_var"], r["subgroup"]), len(subgroups))))


def config_label(family: str, method: str, selection: str) -> str:
    if family == "supervised":
        return f"{method} (supervised)"
    return f"{DISPLAY.get(method, method)} ({DISPLAY.get(selection, selection)})"


def _fmt(value: object, digits: int = 3) -> str:
    number = _num(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def _pct(value: object) -> str:
    number = _num(value)
    return "n/a" if not math.isfinite(number) else f"{100 * number:.1f}%"


def _ci(row: Mapping[str, Any] | None, metric: str) -> str:
    if row is None or not math.isfinite(_num(row.get(metric))):
        return "n/a"
    return f"{_fmt(row[metric])} [{_fmt(row.get(metric + '_low'))}, {_fmt(row.get(metric + '_high'))}]"


def intervals_overlap(a_low: object, a_high: object, b_low: object, b_high: object) -> bool | None:
    values = [_num(value) for value in (a_low, a_high, b_low, b_high)]
    if not all(math.isfinite(value) for value in values):
        return None
    return values[0] <= values[3] and values[2] <= values[1]


def contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Both levels of every :data:`CONTRASTS` variable side by side, per cohort and registry/model."""
    index = {(r["cohort"], r["family"], r["method"], r["selection"], r["subgroup_var"], r["subgroup"]): r for r in rows}
    out = []
    for cohort, family, method, selection in dict.fromkeys((r["cohort"], r["family"], r["method"], r["selection"]) for r in rows):
        for variable, first, second in CONTRASTS:
            a = index.get((cohort, family, method, selection, variable, first))
            b = index.get((cohort, family, method, selection, variable, second))
            if a is None or b is None:
                continue
            entry: dict[str, Any] = {
                "cohort": cohort, "family": family, "method": method, "selection": selection,
                "variable": variable, "first": first, "second": second, "row_first": a, "row_second": b,
            }
            for metric in METRICS:
                entry[f"{metric}_diff"] = _num(a.get(metric)) - _num(b.get(metric))
                entry[f"{metric}_overlap"] = intervals_overlap(a.get(f"{metric}_low"), a.get(f"{metric}_high"), b.get(f"{metric}_low"), b.get(f"{metric}_high"))
            entry["coverage_diff"] = _num(a.get("firing_coverage")) - _num(b.get("firing_coverage"))
            out.append(entry)
    return out


def _describe(contrast: Mapping[str, Any], metric: str) -> str:
    a, b = contrast["row_first"], contrast["row_second"]
    label = config_label(contrast["family"], contrast["method"], contrast["selection"])
    return f"{label} on {contrast['cohort']}: {contrast['first']} {_ci(a, metric)} vs {contrast['second']} {_ci(b, metric)}"


def build_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    demo_rows: Sequence[Mapping[str, Any]],
    cohorts: Mapping[str, Cohort],
    sanity: Mapping[str, Any],
    best: Mapping[str, Any] | None,
    headline: Sequence[tuple[str, str]],
    settings: Mapping[str, Any],
) -> str:
    index = {(r["cohort"], r["family"], r["method"], r["selection"], r["subgroup_var"], r["subgroup"]): r for r in rows}
    configs = list(dict.fromkeys((r["family"], r["method"], r["selection"]) for r in rows))
    discovery = [c for c in configs if c[0] == "discovery"]
    supervised = [c for c in configs if c[0] == "supervised"]
    pairs = contrasts(rows)
    cells = [(cohort, variable, level) for cohort in cohorts for variable, level in SUBGROUPS]
    lines: list[str] = []
    add = lines.append

    add(f"# Subgroup analysis: final v2 track, prediction gap {settings['gap']} h")
    add("")
    add(f"Generated {settings['created_utc']} by `scripts/v2/run_subgroup_analysis.py` (see `manifest.json` for inputs and settings).")
    add("Full numbers: `subgroups.csv` (one row per cohort x registry/model x subgroup) and `subgroup_demographics.csv`.")
    add("")
    add('Reviewer question: do "performance and program survival differ across demographic or clinical subgroups"? '
        "This analysis covers two demographic variables, sex (F/M) and age at admission (<65 / >=65), "
        "in the internal MIMIC-IV test split and the two external cohorts.")
    add("")

    # ---- short answer
    add("## Main observations")
    add("")
    n_configs = len(discovery) + len(supervised)
    for metric in METRICS:
        usable = [c for c in pairs if c[f"{metric}_overlap"] is not None]
        separated = [c for c in usable if c[f"{metric}_overlap"] is False]
        text = (f"- **{metric.upper()}**: the two subgroups' 95% CIs overlap in {len(usable) - len(separated)} of {len(usable)} contrasts "
                f"({n_configs} registries/models x {len(cohorts)} cohorts x 2 variables).")
        if separated:
            shown = "; ".join(_describe(c, metric) for c in separated[:12])
            more = f" (+{len(separated) - 12} more, see the tables below)" if len(separated) > 12 else ""
            text += f" Non-overlapping: {shown}{more}."
        else:
            text += " No contrast has non-overlapping CIs, i.e. no evidence of a subgroup difference at these sample sizes."
        add(text)
    disc_pairs = [c for c in pairs if c["family"] == "discovery" and math.isfinite(c["coverage_diff"])]
    if disc_pairs:
        worst = max(disc_pairs, key=lambda c: abs(c["coverage_diff"]))
        median = float(np.median([abs(c["coverage_diff"]) for c in disc_pairs]))
        changed = [c for c in disc_pairs if c["row_first"].get("n_programs_firing") != c["row_second"].get("n_programs_firing")]
        text = (f"- **Program firing and survival**: firing coverage (episodes on which at least one selected program fires) differs between "
                f"the two subgroups by at most {100 * abs(worst['coverage_diff']):.1f} percentage points (median {100 * median:.1f}) over the "
                f"{len(disc_pairs)} discovery contrasts; the largest gap is {config_label(worst['family'], worst['method'], worst['selection'])} "
                f"on {worst['cohort']} ({worst['first']} {_pct(worst['row_first'].get('firing_coverage'))} vs {worst['second']} "
                f"{_pct(worst['row_second'].get('firing_coverage'))}). Firing survival (selected programs that survive the schema and fire at "
                f"least once) is the same in both subgroups in {len(disc_pairs) - len(changed)} of {len(disc_pairs)} contrasts")
        if changed:
            text += "; it differs for " + "; ".join(
                f"{config_label(c['family'], c['method'], c['selection'])} on {c['cohort']} ({c['first']} {c['row_first'].get('n_programs_firing')}"
                f" vs {c['second']} {c['row_second'].get('n_programs_firing')} of {c['row_first'].get('n_programs')} programs)"
                for c in changed[:8]
            ) + (f" (+{len(changed) - 8} more)" if len(changed) > 8 else "")
        add(text + ". Schema survival depends only on the target schema, so it cannot differ between subgroups.")
    prevalence_bits = []
    for cohort in cohorts:
        values = {level: next((r.get("prevalence") for r in rows if r["cohort"] == cohort and r["subgroup_var"] == variable and r["subgroup"] == level), None) for variable, level in SUBGROUPS}
        prevalence_bits.append(f"{cohort} F {_pct(values['F'])} / M {_pct(values['M'])}, <65 {_pct(values['<65'])} / >=65 {_pct(values['>=65'])}")
    add("- **Prevalence by subgroup**: " + "; ".join(prevalence_bits) + ". The AUPRC of an uninformative score equals the "
        "prevalence, so AUPRC gaps between subgroups with different prevalence partly reflect prevalence; AUROC does not depend on it.")
    if supervised and discovery:
        above, exceptions = 0, []
        for cohort, variable, level in cells:
            sup = [_num(index.get((cohort, *c, variable, level), {}).get("auprc")) for c in supervised]
            dis = [_num(index.get((cohort, *c, variable, level), {}).get("auprc")) for c in discovery]
            sup, dis = [v for v in sup if math.isfinite(v)], [v for v in dis if math.isfinite(v)]
            if sup and dis and min(sup) > max(dis):
                above += 1
            elif sup and dis:
                exceptions.append(f"{cohort} {level}")
        text = f"- **Ranking**: every supervised reference has a higher AUPRC than every discovery registry in {above} of {len(cells)} cohort x subgroup cells"
        add(text + (f" (not in: {', '.join(exceptions)})." if exceptions else "."))
    if discovery:
        tops = []
        for cohort, variable, level in cells:
            scored = [(c, _num(index.get((cohort, *c, variable, level), {}).get("auprc"))) for c in discovery]
            scored = [item for item in scored if math.isfinite(item[1])]
            if scored:
                tops.append((cohort, level, max(scored, key=lambda item: item[1])[0]))
        counts = Counter(config_label(*c) for _, _, c in tops)
        if counts:
            leader, n_lead = counts.most_common(1)[0]
            others = [f"{cohort} {level}: {config_label(*c)}" for cohort, level, c in tops if config_label(*c) != leader]
            add(f"- The highest-AUPRC discovery registry is {leader} in {n_lead} of {len(tops)} cohort x subgroup cells"
                + (f" (otherwise {'; '.join(others)})." if others else "."))
    add("- How to read this: overlapping 95% CIs are taken as no evidence of a difference at this sample size. The criterion is "
        "conservative (it can miss a real difference) and is not evidence of equivalence; no multiplicity adjustment was made.")
    add("")

    # ---- setup
    add("## Setup")
    add("")
    add(f"- Cohorts: internal MIMIC-IV TEST split ({INTERNAL_COHORT}); external " + ", ".join(c for c in cohorts if c != INTERNAL_COHORT)
        + " (all episodes, scored with the frozen bridge: no refit, non-survivable programs zeroed).")
    add(f"- Discovery registries: {len(discovery)} = each method x {{native top-{settings['k']}, v2 selector}}; v2 selector config "
        f"`{json.dumps(settings['selector_overrides'])}` with target schemas {', '.join(settings['selector_targets'])} "
        "(the comparison's main configuration), frozen L2 logistic bridge with C chosen on validation.")
    if supervised:
        add(f"- Supervised references: {len(supervised)} models; point estimates and CIs on seed {settings.get('ci_seed', 0)}, "
            "seed mean and SD over all seeds in `subgroups.csv` (`*_seed_mean`, `*_seed_sd`).")
    add(f"- CIs: percentile intervals from {settings['n_boot']} patient-clustered bootstrap resamples (seed {settings['seed']}), "
        "resampling subject_id within each subgroup separately (unpaired), via `src.icu.v2.stats.patient_cluster_bootstrap`.")
    add("- Firing coverage: share of a subgroup's episodes with at least one active bridge coordinate (a selected program that fires "
        "and survives the target schema). Firing survival: share of the registry's programs that survive the schema and fire on at "
        "least one of the subgroup's episodes.")
    checks = sanity.get("by_reference", {})
    parts = [f"{name}: {block['n_checks']} checks, max |diff| {block['max_abs_diff']:.1e}" for name, block in checks.items()]
    add(f"- Sanity check (tolerance {sanity['tolerance']:g}): all-episode values recomputed here reproduce the stored results ("
        + "; ".join(parts) + "). The overall CIs here use "
        f"{settings['n_boot']} resamples, so they differ slightly from comparison_rows.csv (2000 resamples); the points are identical.")
    add("")

    # ---- demographics
    add("## Cohorts and subgroups")
    add("")
    add("| Cohort | Subgroup | Episodes | Patients | Positives | Prevalence | Age median [IQR] |")
    add("|---|---|---:|---:|---:|---:|---|")
    demo = {(r["cohort"], r["subgroup_var"], r["subgroup"], r["label"]): r for r in demo_rows}
    for cohort in cohorts:
        for variable, level in (("all", "all"), ("sex", "F"), ("sex", "M"), ("sex", UNKNOWN), ("age", "<65"), ("age", ">=65"), ("age", UNKNOWN)):
            total, positive = demo.get((cohort, variable, level, "all")), demo.get((cohort, variable, level, 1))
            if total is None or (level == UNKNOWN and total["n_episodes"] == 0):
                continue
            prevalence = positive["n_episodes"] / total["n_episodes"] if total["n_episodes"] else math.nan
            name = level if variable == "all" else f"{variable} {level}"
            age = f"{_fmt(total['age_median'], 0)} [{_fmt(total['age_q1'], 0)}, {_fmt(total['age_q3'], 0)}]"
            add(f"| {cohort} | {name} | {total['n_episodes']} | {total['n_patients']} | {positive['n_episodes']} | {_pct(prevalence)} | {age} |")
    add("")

    # ---- headline
    head = [("discovery", method, selection) for method, selection in headline if ("discovery", method, selection) in configs]
    if best and ("supervised", best["model"], "supervised") in configs:
        head.append(("supervised", best["model"], "supervised"))
    if head:
        add("## Headline registries and best supervised model")
        add("")
        if best:
            add(f"Best supervised model: {best['model']} ({best['criterion']}, {_fmt(best['value'], 4)}).")
            add("")
        for metric in METRICS:
            add(f"{metric.upper()} [95% CI]:")
            add("")
            add("| Cohort | Subgroup | " + " | ".join(config_label(*c) for c in head) + " |")
            add("|---|---|" + "---|" * len(head))
            for cohort, variable, level in cells:
                add(f"| {cohort} | {level} | " + " | ".join(_ci(index.get((cohort, *c, variable, level)), metric) for c in head) + " |")
            add("")
        head_disc = [c for c in head if c[0] == "discovery"]
        if head_disc:
            add("Firing coverage (firing survival, programs firing / registry size):")
            add("")
            add("| Cohort | Subgroup | " + " | ".join(config_label(*c) for c in head_disc) + " |")
            add("|---|---|" + "---|" * len(head_disc))
            for cohort, variable, level in cells:
                cells_text = []
                for c in head_disc:
                    row = index.get((cohort, *c, variable, level)) or {}
                    cells_text.append(f"{_pct(row.get('firing_coverage'))} ({row.get('n_programs_firing', 'n/a')}/{row.get('n_programs', 'n/a')})")
                add(f"| {cohort} | {level} | " + " | ".join(cells_text) + " |")
            add("")

    # ---- full side-by-side tables
    for metric in METRICS:
        add(f"## All registries and models: {metric.upper()} by subgroup")
        add("")
        add("Point [95% CI]; diff = first minus second level (F - M, >=65 - <65); '**no**' marks non-overlapping CIs.")
        add("")
        add("| Registry / model | Cohort | F | M | F - M | overlap | >=65 | <65 | >=65 - <65 | overlap |")
        add("|---|---|---|---|---:|---|---|---|---:|---|")
        by_key = {(c["cohort"], c["family"], c["method"], c["selection"], c["variable"]): c for c in pairs}
        for cohort in cohorts:
            for config in configs:
                sex, age = by_key.get((cohort, *config, "sex")), by_key.get((cohort, *config, "age"))
                if sex is None and age is None:
                    continue
                fields = []
                for c in (sex, age):
                    if c is None:
                        fields += ["n/a"] * 4
                        continue
                    flag = {True: "yes", False: "**no**", None: "n/a"}[c[f"{metric}_overlap"]]
                    fields += [_ci(c["row_first"], metric), _ci(c["row_second"], metric), _fmt(c[f"{metric}_diff"]), flag]
                add(f"| {config_label(*config)} | {cohort} | " + " | ".join(fields) + " |")
        add("")

    if discovery:
        add("## Firing coverage by subgroup (discovery registries)")
        add("")
        add("Share of episodes with at least one firing selected program; in parentheses the programs firing at least once / registry size. "
            "Schema survival is per cohort (external cohorts only).")
        add("")
        add("| Registry | Cohort | all | F | M | <65 | >=65 | schema survival |")
        add("|---|---|---|---|---|---|---|---:|")
        for cohort in cohorts:
            for config in discovery:
                rows_here = [index.get((cohort, *config, variable, level)) for variable, level in SUBGROUPS]
                if all(r is None for r in rows_here):
                    continue
                fields = [f"{_pct((r or {}).get('firing_coverage'))} ({(r or {}).get('n_programs_firing', 'n/a')}/{(r or {}).get('n_programs', 'n/a')})" for r in rows_here]
                schema = (rows_here[0] or {}).get("schema_survival")
                add(f"| {config_label(*config)} | {cohort} | " + " | ".join(fields) + f" | {_fmt(schema, 2) if math.isfinite(_num(schema)) else '-'} |")
        add("")

    # ---- caveats
    add("## Definitions and caveats")
    add("")
    add("- MIMIC-IV age = anchor_age + (year of ICU intime - anchor_year), the adapter's `age_at_intime`. This is approximate (year "
        "resolution; anchor_age is top-coded at 91 for patients aged 89 or more). Sex = `patients.gender`.")
    add("- eICU age is the per-unit-stay `patient.age`; the text \"> 89\" is set to 90. Sex = `patient.gender`; \"Unknown\", \"Other\" "
        "and blank are unknown sex and are left out of the sex subgroups.")
    add("- HiRID age is released in 5-year bins (labels 20..90; 90 = 90 and older). The label is taken as the lower bound of its bin, "
        "so >=65 means labels 65..90; if the labels were midpoints or upper bounds the cut would move by up to 5 years. Sex = "
        "`general_table.sex`.")
    unknown = [f"{name}: sex unknown for {c.audit.get('unknown_sex_episodes', 0)}, age unknown for {c.audit.get('unknown_age_episodes', 0)} "
               f"of {c.audit.get('episodes', 0)} episodes ({c.audit.get('stays_without_demographics', 0)} stays without a demographics row)"
               for name, c in cohorts.items()]
    add("- Missing demographics: " + "; ".join(unknown) + ".")
    add("- Subgroups are post hoc (not pre-registered) and few; clinical subgroups (care unit, severity, admission type) were not analysed.")
    add("- Supervised CIs come from seed 0 only; seed variability (`*_seed_sd`) is reported separately and is not part of the CI.")
    add("")
    return "\n".join(lines) + "\n"


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True, help="pools.pkl written by run_discovery_comparison.py --save-pools")
    parser.add_argument("--supervised-dir", default=None, help="folder with supervised_v2_test_predictions.csv and the external predictions CSV")
    parser.add_argument("--out-dir", default=None, help="default: artifacts/v2/subgroups/f_mimic_iv_gap<gap>h")
    parser.add_argument("--gap", type=int, default=1, help="prediction gap in hours (selects the extraction manifests)")
    parser.add_argument("--targets", default=DEFAULT_TARGETS, help="external cohorts to analyse; names of pickled targets that are real sources")
    parser.add_argument("--comparison-rows", default=None, help="default: comparison_rows.csv next to --pools (sanity check)")
    parser.add_argument("--comparison-config", default=None, help="default: config.json next to --pools (k, selector, selector_targets)")
    parser.add_argument("--extract-root", default="data/processed/v2/extract", help="parent of <source>_gap<gap>h_full/extraction_manifest.json")
    parser.add_argument("--demographics", default=None, help="CSV source,stay_id,sex,age[,subject_id] used instead of the raw tables")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headline", default=DEFAULT_HEADLINE, help="comma list of method:selection shown in the summary's headline tables")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    pools_path = Path(args.pools)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "artifacts" / "v2" / "subgroups" / f"f_mimic_iv_gap{args.gap}h"
    config_path = Path(args.comparison_config) if args.comparison_config else pools_path.parent / "config.json"
    comparison_path = Path(args.comparison_rows) if args.comparison_rows else pools_path.parent / "comparison_rows.csv"
    if not comparison_path.exists():
        raise SystemExit(f"{comparison_path} not found; it is needed for the sanity check")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    k = int(config.get("k", 10))
    overrides = config["selector"] if "selector" in config else DEFAULT_SELECTOR
    overrides = {key: tuple(value) if isinstance(value, list) else value for key, value in overrides.items()}
    selector_config = SelectorConfig(k=k, **overrides)
    target_names = [name for name in args.targets.split(",") if name]
    selector_targets = list(config.get("selector_targets") or target_names)
    headline = [tuple(item.split(":", 1)) for item in args.headline.split(",") if ":" in item]

    with open(pools_path, "rb") as handle:
        state = pickle.load(handle)
    by_name = {target.name: target for target in state["targets"]}
    unknown = [name for name in target_names + selector_targets if name not in by_name]
    if unknown:
        raise SystemExit(f"targets {unknown} are not in {pools_path} (available: {sorted(by_name)})")
    targets = [by_name[name] for name in target_names]

    sources = {INTERNAL_COHORT: (INTERNAL_SOURCE, tuple(e for e in state["episodes"] if e.split == "test"))}
    sources.update({target.name: (target.name, target.episodes) for target in targets})
    cohorts: dict[str, Cohort] = {}
    demographics_inputs: dict[str, dict[str, str]] = {}
    for name, (source, episodes) in sources.items():
        frame, info = load_demographics(source, args)
        cohorts[name] = attach_demographics(name, episodes, frame)
        demographics_inputs[name] = info
        audit = cohorts[name].audit
        print(f"[subgroups] {name}: {audit['episodes']} episodes, {audit['patients']} patients; unknown sex {audit['unknown_sex_episodes']}, "
              f"unknown age {audit['unknown_age_episodes']}, stays without demographics {audit['stays_without_demographics']}", flush=True)

    rows, registries = discovery_rows(
        state, cohorts, targets, k=k, selector_config=selector_config, selector_targets=selector_targets, n_boot=args.n_boot, seed=args.seed
    )
    checks = check_discovery(rows, comparison_path)
    supervised_files: dict[str, str] = {}
    best = None
    if args.supervised_dir:
        test, external, summary, supervised_files = load_supervised(Path(args.supervised_dir))
        sup_rows, overall = supervised_rows(test, external, cohorts, n_boot=args.n_boot, seed=args.seed)
        rows += sup_rows
        checks += check_supervised(overall, summary)
        best = best_supervised(summary, overall)
    sanity = finalize_checks(checks)
    for reference, block in sanity["by_reference"].items():
        print(f"[subgroups] sanity vs {reference}: {block['n_checks']} checks, {block['n_failed']} failed, max |diff| {block['max_abs_diff']:.2e}", flush=True)
    if not sanity["passed"]:
        for check in sanity["failed"][:20]:
            print(f"[subgroups] FAILED {check}", flush=True)
        print("[subgroups] sanity check failed; no outputs written", flush=True)
        return 3

    rows = sort_rows(rows, list(cohorts))
    demo_rows = [row for cohort in cohorts.values() for row in demographics_rows(cohort)]
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    settings = {
        "gap": args.gap, "k": k, "n_boot": args.n_boot, "seed": args.seed, "ci_seed": 0, "created_utc": created,
        "selector_overrides": {key: list(value) if isinstance(value, tuple) else value for key, value in overrides.items()},
        "selector_targets": selector_targets,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "subgroups.csv", COLUMNS)
    write_csv(demo_rows, out_dir / "subgroup_demographics.csv", DEMO_COLUMNS)
    summary_text = build_summary(rows=rows, demo_rows=demo_rows, cohorts=cohorts, sanity=sanity, best=best, headline=headline, settings=settings)
    (out_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    manifest = {
        "created_utc": created,
        "git_head": _git_head(),
        "script": "scripts/v2/run_subgroup_analysis.py",
        "args": vars(args),
        "inputs": {
            "pools": str(pools_path),
            "comparison_rows": str(comparison_path),
            "comparison_config": str(config_path) if config_path.exists() else None,
            "supervised": supervised_files,
            "demographics": demographics_inputs,
        },
        "k": k,
        "selector_config": asdict(selector_config),
        "selector_targets": selector_targets,
        "c_grid": list(DEFAULT_C_GRID),
        "subgroups": [list(item) for item in SUBGROUPS],
        "age_cut": AGE_CUT,
        "cohorts": {name: cohort.audit for name, cohort in cohorts.items()},
        "registries": registries,
        "best_supervised": best,
        "sanity_check": {key: value for key, value in sanity.items() if key != "checks"},
        "sanity_checks": sanity["checks"],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[subgroups] wrote {len(rows)} subgroup rows, {len(demo_rows)} demographic rows and summary.md to {out_dir} "
          f"in {time.perf_counter() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
