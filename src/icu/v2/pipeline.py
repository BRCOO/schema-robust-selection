"""Method-agnostic evaluation for the v2 protocol.

Every discovery method (FMG-Bridge, PrefixSpan, CM-SPADE, ERMiner/RuleGrowth, motif,
shapelet, ...) enters as a :class:`MethodPool`: TRAIN-only candidates ranked best-first,
plus each candidate's firing set over every internal episode (and, optionally, over
external target episodes). From there one code path does everything:

* selection: the method's own top-K (``"native"``) and the generator-agnostic v2
  selector (``"v2_selector"``, :func:`src.icu.v2.selector.select`), so the
  candidate-source x selector grid of the paper's fairness table is built with the REAL
  selector for every source (v1 used a one-line stand-in for classical pools);
* the frozen bridge with validation-only model selection
  (:func:`src.icu.v2.bridge.fit_frozen_bridge_v2`);
* structural metrics with one definition for all methods
  (:func:`src.icu.v2.program_metrics.summarize_program_set`);
* patient-clustered bootstrap intervals and paired differences
  (:mod:`src.icu.v2.stats`);
* external refiring without repair (:func:`src.icu.v2.bridge.refire_external`).

Nothing here mines, loads files, or touches labels outside TRAIN for selection; callers
build the pools (see ``scripts/v2/``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import AbstractSet, Any, Mapping, Sequence

import numpy as np

from src.icu.v2.bridge import DEFAULT_C_GRID, fit_frozen_bridge_v2, refire_external
from src.icu.v2.patterns import dedup_programs
from src.icu.v2.program_metrics import summarize_program_set
from src.icu.v2.selector import SelectorConfig, select
from src.icu.v2.stats import paired_cluster_bootstrap_diff, patient_cluster_bootstrap
from src.icu.v2.types import EpisodeRecord, ProgramRecord

SPLITS: tuple[str, ...] = ("train", "val", "test")
SELECTIONS: tuple[str, ...] = ("native", "v2_selector")
FiringSets = Mapping[str, AbstractSet[str]]


@dataclass(frozen=True)
class TargetCohort:
    """An external source: its available canonical features and labeled episodes."""

    name: str
    features: frozenset[str]
    episodes: tuple[EpisodeRecord, ...]


@dataclass
class MethodPool:
    """TRAIN-only candidates of one method, ranked best-first, with their firing sets.

    ``firings`` covers every internal episode (all splits); ``external_firings`` maps a
    target name to program_id -> target episode ids. ``native_registry`` is the method's
    own selection when it has one (e.g. FMG-Bridge's v1 selector); otherwise the native
    registry is the de-duplicated top-K of ``candidates``.
    """

    method: str
    candidates: list[ProgramRecord]
    firings: dict[str, frozenset[str]]
    external_firings: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)
    native_registry: list[ProgramRecord] | None = None


def episodes_by_split(episodes: Sequence[EpisodeRecord]) -> dict[str, list[EpisodeRecord]]:
    """Group episodes by split; raise if a split is missing or unknown."""
    grouped: dict[str, list[EpisodeRecord]] = {name: [] for name in SPLITS}
    for episode in episodes:
        if episode.split not in grouped:
            raise ValueError(f"episode {episode.episode_id!r} has split {episode.split!r}")
        grouped[episode.split].append(episode)
    empty = [name for name, rows in grouped.items() if not rows]
    if empty:
        raise ValueError(f"empty splits: {empty}")
    return grouped


def restrict_firings(firings: FiringSets, episode_ids: AbstractSet[str]) -> dict[str, frozenset[str]]:
    return {pid: frozenset(ids) & episode_ids for pid, ids in firings.items()}


def native_registry(pool: MethodPool, k: int) -> list[ProgramRecord]:
    if pool.native_registry is not None:
        return list(pool.native_registry)[:k]
    return dedup_programs(pool.candidates)[:k]


def selector_registry(
    pool: MethodPool,
    config: SelectorConfig,
    train_ids: AbstractSet[str],
    target_schemas: Mapping[str, AbstractSet[str]],
) -> tuple[list[ProgramRecord], list[dict]]:
    """Apply the v2 selector to the method's candidates using TRAIN firings only."""
    train_firings = restrict_firings({p.program_id: pool.firings.get(p.program_id, frozenset()) for p in pool.candidates}, train_ids)
    return select(pool.candidates, config, train_firings, target_schemas)


def _ci_fields(prefix: str, result: Mapping[str, Any]) -> dict[str, float]:
    return {prefix: result["point"], f"{prefix}_low": result["low"], f"{prefix}_high": result["high"]}


def _scored(episodes: Sequence[EpisodeRecord], probabilities: Mapping[str, float], *, n_boot: int, seed: int) -> dict[str, Any]:
    y = np.array([episode.label for episode in episodes], dtype=int)
    p = np.array([probabilities[episode.episode_id] for episode in episodes], dtype=float)
    groups = np.array([episode.subject_id for episode in episodes])
    out: dict[str, Any] = {
        "n_episodes": int(y.size),
        "n_patients": int(np.unique(groups).size),
        "prevalence": float(y.mean()) if y.size else float("nan"),
    }
    if np.unique(y).size < 2:
        out["error"] = "single-class evaluation set"
        return out
    for metric in ("auprc", "auroc"):
        out.update(_ci_fields(metric, patient_cluster_bootstrap(y, p, groups, metric=metric, n_boot=n_boot, seed=seed)))
    return out


def evaluate_registry(
    registry: Sequence[ProgramRecord],
    firings: FiringSets,
    episodes: Sequence[EpisodeRecord],
    *,
    targets: Sequence[TargetCohort] = (),
    external_firings: Mapping[str, FiringSets] | None = None,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    metric: str = "auprc",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Frozen bridge + structural metrics + external refiring for one registry.

    Returns ``{"error": ...}`` instead of raising when the registry is empty or the bridge
    cannot be fit (for example a single-class validation split), so one failed method does
    not stop a comparison.
    """
    registry = list(registry)
    if not registry:
        return {"error": "empty registry", "n_programs": 0}
    grouped = episodes_by_split(episodes)
    ids = {name: [episode.episode_id for episode in rows] for name, rows in grouped.items()}
    id_sets = {name: frozenset(values) for name, values in ids.items()}
    selected_firings = {program.program_id: frozenset(firings.get(program.program_id, frozenset())) for program in registry}
    try:
        bridge = fit_frozen_bridge_v2(
            registry,
            {name: restrict_firings(selected_firings, id_sets[name]) for name in SPLITS},
            ids,
            {name: [episode.label for episode in rows] for name, rows in grouped.items()},
            c_grid=c_grid,
            metric=metric,
        )
    except ValueError as exc:
        return {"error": f"bridge: {exc}", "n_programs": len(registry)}

    external = dict(external_firings or {})
    structure = summarize_program_set(
        registry,
        restrict_firings(selected_firings, id_sets["train"]),
        target_schemas={target.name: target.features for target in targets},
        external_firings={target.name: external.get(target.name, {}) for target in targets},
    )
    result: dict[str, Any] = {
        "n_programs": len(registry),
        "program_ids": [program.program_id for program in registry],
        "bridge": {
            "selected_c": bridge["selected_c"],
            "validation_score": bridge["validation_score"],
            "grid_scores": bridge["grid_scores"],
            "coordinate_manifest": bridge["coordinate_manifest"],
        },
        "internal": _scored(grouped["test"], bridge["test_probabilities"], n_boot=n_boot, seed=seed),
        "structure": structure,
        "external": {},
        "test_probabilities": bridge["test_probabilities"],
    }
    for target in targets:
        refired = refire_external(
            bridge["model_bundle"],
            registry,
            target.features,
            external.get(target.name, {}),
            [episode.episode_id for episode in target.episodes],
        )
        scored = _scored(target.episodes, refired["probabilities"], n_boot=n_boot, seed=seed)
        scored.update(
            {
                "schema_survival": refired["schema_survival"],
                "firing_survival": refired["firing_survival"],
                "suppressed_firings": refired["suppressed_firings"],
            }
        )
        result["external"][target.name] = scored
    return result


def flatten_row(method: str, selection: str, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """One table row per (method, selection); nested details stay in the full result."""
    row: dict[str, Any] = {"method": method, "selection": selection, "n_programs": evaluation.get("n_programs", 0)}
    if "error" in evaluation:
        row["error"] = evaluation["error"]
        return row
    internal = evaluation["internal"]
    row.update({key: value for key, value in internal.items() if not isinstance(value, dict)})
    structure = evaluation["structure"]
    row.update(
        {
            "redundancy": structure["redundancy"],
            "exact_duplicate_ratio": structure["exact_duplicate_ratio"],
            "anchor_group_coverage": structure["anchor_group_coverage"],
            "feature_coverage": structure["feature_coverage"],
            "n_never_fire_train": structure["n_never_fire_train"],
            "selected_c": evaluation["bridge"]["selected_c"],
            "validation_score": evaluation["bridge"]["validation_score"],
        }
    )
    for name, survival in structure["schema_survival"].items():
        row[f"{name}_schema_survival"] = survival["value"]
    for name, scored in evaluation["external"].items():
        for key in ("auprc", "auprc_low", "auprc_high", "auroc", "firing_survival", "prevalence", "error"):
            if key in scored:
                row[f"{name}_{key}"] = scored[key]
    return row


def compare_methods(
    pools: Sequence[MethodPool],
    episodes: Sequence[EpisodeRecord],
    *,
    targets: Sequence[TargetCohort] = (),
    selector_config: SelectorConfig | None = None,
    k: int = 10,
    selections: Sequence[str] = SELECTIONS,
    reference: tuple[str, str] | None = None,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    n_boot: int = 2000,
    seed: int = 0,
    selector_target_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate every (method, selection) pair and paired AUPRC differences vs ``reference``.

    ``reference`` is ``(method, selection)``; when given, every other pair gets a paired
    patient-clustered bootstrap of test AUPRC(other) - AUPRC(reference).
    ``selector_target_names`` limits which target schemas the v2 selector may use (default:
    all targets). Pass only the real external sources to keep controlled schema-shift
    targets unanticipated by selection.
    """
    unknown = sorted(set(selections) - set(SELECTIONS))
    if unknown:
        raise ValueError(f"unknown selections {unknown}")
    config = selector_config or SelectorConfig(k=k)
    if config.k != k:
        raise ValueError(f"selector_config.k={config.k} differs from k={k}")
    grouped = episodes_by_split(episodes)
    train_ids = frozenset(episode.episode_id for episode in grouped["train"])
    allowed = None if selector_target_names is None else set(selector_target_names)
    unknown_targets = sorted((allowed or set()) - {target.name for target in targets})
    if unknown_targets:
        raise ValueError(f"selector_target_names not among targets: {unknown_targets}")
    target_schemas = {target.name: target.features for target in targets if allowed is None or target.name in allowed}
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for pool in pools:
        for selection in selections:
            trace: list[dict] = []
            if selection == "native":
                registry = native_registry(pool, k)
            else:
                registry, trace = selector_registry(pool, config, train_ids, target_schemas)
            evaluation = evaluate_registry(
                registry,
                pool.firings,
                episodes,
                targets=targets,
                external_firings=pool.external_firings,
                c_grid=c_grid,
                n_boot=n_boot,
                seed=seed,
            )
            evaluation["selector_trace"] = trace
            key = f"{pool.method}|{selection}"
            details[key] = evaluation
            rows.append(flatten_row(pool.method, selection, evaluation))

    paired: list[dict[str, Any]] = []
    if reference is not None:
        ref_key = f"{reference[0]}|{reference[1]}"
        ref = details.get(ref_key)
        if ref is None or "error" in ref:
            raise ValueError(f"reference {ref_key!r} has no usable evaluation")
        test = grouped["test"]
        y = np.array([episode.label for episode in test], dtype=int)
        groups = np.array([episode.subject_id for episode in test])
        ref_p = np.array([ref["test_probabilities"][episode.episode_id] for episode in test])
        for key, evaluation in details.items():
            if key == ref_key or "error" in evaluation:
                continue
            other = np.array([evaluation["test_probabilities"][episode.episode_id] for episode in test])
            diff = paired_cluster_bootstrap_diff(y, other, ref_p, groups, metric="auprc", n_boot=n_boot, seed=seed)
            paired.append({"comparison": f"{key} - {ref_key}", **diff})
    return {"rows": rows, "details": details, "paired_vs_reference": paired, "k": k, "selector_config": asdict(config), "selector_target_schemas": sorted(target_schemas)}
