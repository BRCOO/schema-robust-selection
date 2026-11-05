"""Frozen bridge v2: registry-declared coordinates, validation-only probe selection,
and survival-masked external refiring.

Every discovery method (FMG-Bridge and all baselines) goes through this one bridge, so
method differences cannot come from the decoder. It replaces the v1 protocol in
``src/icu/comparators/bridge.py``, which is left unchanged so the v1 numbers stay auditable.

Fixes relative to v1 (see ``docs/reviews/ICDE2027_558_audit_and_revision_plan.md``):

1. One coordinate per registry program. v1 ``build_bridge_features`` (lines 27-45) created
   a column only for programs that fired on some TRAINING episode, ordered by first
   firing, so a selected program without a training firing silently lost its coordinate
   (10 selected programs -> 9 dimensions; reviewer 3 D8). :func:`compile_bridge` always
   returns exactly ``len(registry)`` columns in registry order. A program that never fires
   keeps an all-zero column and is listed in ``coordinate_manifest(...)["zero_train_columns"]``.
2. C is chosen on validation only. v1 ``select_best_logistic_probe`` (lines 97-103) scored
   the C grid on TRAINING data when the validation split was empty, which picked the grid
   maximum (C=10) for every method although the paper says "validation rule".
   :func:`select_probe_on_validation` raises ``ValueError`` for an empty or single-class
   validation split; there is no fallback.
3. Deterministic tie-break toward the smallest C. v1 kept the LAST grid value with a
   maximal score (``score >= best_score``, line 107), i.e. the largest C of an ascending
   grid. v2 keeps the smallest C whose validation score is within ``TIE_TOLERANCE`` of the
   best (strongest regularization), independent of grid order.
4. No per-episode truncation. v1 encoded at most ``top_k`` firing programs per episode,
   ranked by score (line 32). The registry already holds at most K programs, so v2 encodes
   every firing of every registry program.
5. Frozen external refiring. :func:`refire_external` reuses the fitted probe, C, and
   coordinate order unchanged, and forces the coordinate of every non-survivable program
   (sigma_f not a subset of the target schema) to 0 even when a firing is reported.
   Nothing is repaired or refit (paper Definition 4). v1 fired programs on the external
   episodes without an explicit survival mask.

Unchanged from v1: L2 logistic regression (scikit-learn's default penalty; scikit-learn 1.8
deprecates the ``penalty`` argument, so it is not passed), ``class_weight="balanced"``,
``max_iter=1000``, ``random_state=42``, lbfgs solver, raw (unscaled) features, and the probe
is fit on TRAIN only (validation only chooses C; there is no train+val refit). The default
grid ``(0.01, 0.1, 1.0, 10.0)`` is ``logistic_probe_grid`` from
``docs/experiments/icde_comparator_execution_manifest.json``.

A program whose column is all zero on the training split gets a coefficient of exactly 0
under the L2 probe (its data gradient is identically 0, so lbfgs never moves it), so a
zero-train coordinate cannot change any validation, test, or external probability. It is
kept so that ``dim == len(registry)`` and so its lack of training support is reported
instead of hidden.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.icu.v2.types import ProgramRecord

FiringSets = Mapping[str, AbstractSet[str]]

SPLITS = ("train", "val", "test")
BRIDGE_ENCODINGS = ("binary", "score")
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)
SELECTION_METRICS = ("auprc", "auroc")
TIE_TOLERANCE = 1e-12
PROBE_PARAMS: dict[str, object] = {"max_iter": 1000, "random_state": 42, "class_weight": "balanced"}
TIE_BREAK_RULE = (
    "smallest C among grid values whose validation score is within 1e-12 of the best "
    "(v1 kept the last maximal value in grid order)"
)

_SCORERS = {"auprc": average_precision_score, "auroc": roc_auc_score}
_EMPTY: frozenset[str] = frozenset()


def _registry_ids(registry: Sequence[ProgramRecord]) -> list[str]:
    ids = [program.program_id for program in registry]
    if len(set(ids)) != len(ids):
        duplicates = sorted({pid for pid in ids if ids.count(pid) > 1})
        raise ValueError(f"duplicate program_id in registry: {duplicates}; each coordinate needs a unique id")
    return ids


def _row_index(episode_ids: Sequence[str]) -> dict[str, int]:
    rows = {episode_id: index for index, episode_id in enumerate(episode_ids)}
    if len(rows) != len(episode_ids):
        raise ValueError("duplicate episode ids; each bridge row must be one episode")
    return rows


def _check_encoding(encoding: str) -> None:
    if encoding not in BRIDGE_ENCODINGS:
        raise ValueError(f"encoding must be one of {BRIDGE_ENCODINGS}, got {encoding!r}")


def _check_grid(c_grid: Sequence[float]) -> tuple[float, ...]:
    values = []
    for raw in c_grid:
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"every C must be finite and > 0, got {raw!r}")
        values.append(value)
    if not values:
        raise ValueError("c_grid is empty")
    return tuple(sorted(set(values)))


def _as_matrix(x: object, name: str) -> np.ndarray:
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _as_labels(y: object, name: str) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be a 1-D label vector, got shape {labels.shape}")
    if labels.size and not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 labels")
    return labels.astype(int)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """AUPRC/AUROC with NaN when a split has a single class (reported, not hidden)."""
    n = int(y_true.shape[0])
    n_positive = int(np.sum(y_true))
    result = {
        "n": n,
        "n_positive": n_positive,
        "prevalence": n_positive / n if n else float("nan"),
        "auprc": float("nan"),
        "auroc": float("nan"),
    }
    if 0 < n_positive < n:
        result["auprc"] = float(average_precision_score(y_true, y_prob))
        result["auroc"] = float(roc_auc_score(y_true, y_prob))
    return result


def compile_bridge(
    registry: Sequence[ProgramRecord],
    firings: FiringSets,
    episode_ids: Sequence[str],
    encoding: str = "binary",
) -> tuple[np.ndarray, list[str]]:
    """Return ``(X, names)`` with exactly one column per registry program, in registry order.

    ``X[i, j]`` is the activation of ``registry[j]`` on ``episode_ids[i]``: 1.0 for
    ``"binary"`` or the program's frozen ``score`` for ``"score"`` when the episode is in
    ``firings[program_id]``, else 0.0. A program missing from ``firings`` or never firing on
    these episodes keeps an all-zero column. Firing entries for ids outside the registry or
    episodes outside ``episode_ids`` are ignored. An empty registry gives shape ``(n, 0)``.
    """
    _check_encoding(encoding)
    names = _registry_ids(registry)
    rows = _row_index(list(episode_ids))
    matrix = np.zeros((len(rows), len(names)), dtype=float)
    for column, program in enumerate(registry):
        value = 1.0 if encoding == "binary" else float(program.score)
        if not math.isfinite(value):
            raise ValueError(f"program {program.program_id!r} has a non-finite score; cannot use score encoding")
        for episode_id in firings.get(program.program_id, _EMPTY):
            row = rows.get(episode_id)
            if row is not None:
                matrix[row, column] = value
    return matrix, names


def coordinate_manifest(
    registry: Sequence[ProgramRecord],
    train_firings: FiringSets,
    train_episode_ids: Sequence[str],
) -> dict:
    """Declare the bridge coordinates (names and order) before any probe is fit.

    ``zero_train_columns`` lists registry programs that fire on none of
    ``train_episode_ids``; firings on other episodes are ignored. The result is
    JSON-serializable.
    """
    names = _registry_ids(registry)
    train = frozenset(train_episode_ids)
    counts = {
        program.program_id: len(frozenset(train_firings.get(program.program_id, _EMPTY)) & train)
        for program in registry
    }
    zero = [pid for pid in names if counts[pid] == 0]
    notes = [
        "one coordinate per registry program in registry (selection-rank) order; dim == len(registry)",
        "coordinates never depend on which programs fire on any split (v1 derived them from training firings)",
        "a zero_train_columns program keeps an all-zero training column; its L2 probe coefficient is 0, so it "
        "cannot change any probability, and it is reported here instead of being dropped",
    ]
    if zero:
        notes.append(
            f"{len(zero)} of {len(names)} registry programs never fire on the {len(train)} training episodes: "
            f"{zero}; selection should not have retained them without training support"
        )
    return {
        "names": names,
        "order": {pid: index for index, pid in enumerate(names)},
        "n_coordinates": len(names),
        "coordinates": [
            {
                "index": index,
                "program_id": program.program_id,
                "method": program.method,
                "kind": program.kind,
                "features": sorted(program.features),
                "anchor_signature": sorted(program.anchor_signature),
                "train_firing_count": counts[program.program_id],
            }
            for index, program in enumerate(registry)
        ],
        "zero_train_columns": zero,
        "train_firing_counts": counts,
        "n_train_episodes": len(train),
        "notes": notes,
    }


def select_probe_on_validation(
    x_tr: np.ndarray,
    y_tr: Sequence[int],
    x_val: np.ndarray,
    y_val: Sequence[int],
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    metric: str = "auprc",
) -> dict:
    """Fit one L2 logistic probe per C on TRAIN and keep the best C by VALIDATION score.

    Raises ``ValueError`` when the validation split is empty or has a single class (no
    training-score fallback), when training has a single class, when there are no
    coordinates, or on shape/metric/grid errors. Ties: :data:`TIE_BREAK_RULE`.
    """
    if metric not in _SCORERS:
        raise ValueError(f"metric must be one of {SELECTION_METRICS}, got {metric!r}")
    grid = _check_grid(c_grid)
    y_train = _as_labels(y_tr, "y_tr")
    y_valid = _as_labels(y_val, "y_val")
    if y_valid.size == 0:
        raise ValueError("validation split is empty: v2 selects C on validation only and never falls back to training scores")
    if np.unique(y_valid).size < 2:
        raise ValueError("validation split has a single class: the validation metric is undefined, so C cannot be selected")
    x_train = _as_matrix(x_tr, "x_tr")
    x_valid = _as_matrix(x_val, "x_val")
    if x_train.shape[0] != y_train.shape[0] or x_valid.shape[0] != y_valid.shape[0]:
        raise ValueError("feature matrices and label vectors have different numbers of rows")
    if x_train.shape[1] != x_valid.shape[1]:
        raise ValueError(f"train has {x_train.shape[1]} coordinates but validation has {x_valid.shape[1]}")
    if x_train.shape[1] == 0:
        raise ValueError("no bridge coordinates (empty registry); there is nothing to fit")
    if np.unique(y_train).size < 2:
        raise ValueError("training split must contain both classes to fit the probe")

    scorer = _SCORERS[metric]
    models: dict[float, LogisticRegression] = {}
    grid_scores = []
    for c in grid:
        model = LogisticRegression(C=c, **PROBE_PARAMS).fit(x_train, y_train)
        models[c] = model
        grid_scores.append({"c": c, "validation_score": float(scorer(y_valid, model.predict_proba(x_valid)[:, 1]))})
    best = max(row["validation_score"] for row in grid_scores)
    tied = [row["c"] for row in grid_scores if row["validation_score"] >= best - TIE_TOLERANCE]
    selected = min(tied)
    return {
        "model": models[selected],
        "selected_c": selected,
        "validation_score": next(row["validation_score"] for row in grid_scores if row["c"] == selected),
        "metric": metric,
        "grid_scores": grid_scores,
        "tied_c": sorted(tied),
        "tie_break": TIE_BREAK_RULE,
        "n_train": int(y_train.shape[0]),
        "n_val": int(y_valid.shape[0]),
        "n_coordinates": int(x_train.shape[1]),
    }


def _check_split_mappings(**mappings: Mapping[str, object]) -> None:
    for name, mapping in mappings.items():
        missing = [split for split in SPLITS if split not in mapping]
        if missing:
            raise ValueError(f"{name} is missing splits {missing}; expected {SPLITS}")


def _check_disjoint(episodes: Mapping[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for split in SPLITS:
        for episode_id in episodes[split]:
            first = seen.setdefault(episode_id, split)
            if first != split:
                raise ValueError(f"episode {episode_id!r} appears in both {first!r} and {split!r}")


def fit_frozen_bridge_v2(
    registry: Sequence[ProgramRecord],
    firings_by_split: Mapping[str, FiringSets],
    episodes_by_split: Mapping[str, Sequence[str]],
    labels_by_split: Mapping[str, Sequence[int]],
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    metric: str = "auprc",
    encoding: str = "binary",
) -> dict:
    """Compile the declared coordinates, select C on validation, and score the test split.

    All three mappings need ``"train"``, ``"val"``, and ``"test"``; episode ids must be
    unique within and across splits and labels must align with them. Returns the selected
    C, validation score, per-C grid scores, test probabilities keyed by episode id, test
    metrics (AUPRC/AUROC; NaN if the test split has one class), the coordinate manifest,
    and ``model_bundle`` for :func:`refire_external`.
    """
    _check_split_mappings(
        firings_by_split=firings_by_split,
        episodes_by_split=episodes_by_split,
        labels_by_split=labels_by_split,
    )
    episodes = {split: list(episodes_by_split[split]) for split in SPLITS}
    _check_disjoint(episodes)
    matrices: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    names: list[str] = []
    for split in SPLITS:
        matrices[split], names = compile_bridge(registry, firings_by_split[split], episodes[split], encoding)
        labels[split] = _as_labels(labels_by_split[split], f"labels_by_split[{split!r}]")
        if labels[split].shape[0] != len(episodes[split]):
            raise ValueError(
                f"split {split!r} has {len(episodes[split])} episodes but {labels[split].shape[0]} labels"
            )
    manifest = coordinate_manifest(registry, firings_by_split["train"], episodes["train"])
    probe = select_probe_on_validation(
        matrices["train"], labels["train"], matrices["val"], labels["val"], c_grid=c_grid, metric=metric
    )
    model = probe["model"]
    test_prob = model.predict_proba(matrices["test"])[:, 1] if episodes["test"] else np.zeros(0)
    return {
        "selected_c": probe["selected_c"],
        "validation_score": probe["validation_score"],
        "validation_metric": metric,
        "grid_scores": probe["grid_scores"],
        "tie_break": probe["tie_break"],
        "test_probabilities": {episode_id: float(p) for episode_id, p in zip(episodes["test"], test_prob)},
        "test_metrics": _binary_metrics(labels["test"], test_prob),
        "coordinate_manifest": manifest,
        "model_bundle": {
            "model": model,
            "coordinate_names": list(names),
            "encoding": encoding,
            "selected_c": probe["selected_c"],
            "coef": [float(value) for value in model.coef_[0]],
            "intercept": float(model.intercept_[0]),
        },
        "n_episodes": {split: len(episodes[split]) for split in SPLITS},
        "n_coordinates": len(names),
    }


def refire_external(
    model_bundle: Mapping[str, object],
    registry: Sequence[ProgramRecord],
    target_features: AbstractSet[str],
    external_firings: FiringSets,
    external_episode_ids: Sequence[str],
    labels: Sequence[int] | None = None,
) -> dict:
    """Score a target source with the frozen probe: no repair, no refit.

    ``model_bundle`` is ``fit_frozen_bridge_v2(...)["model_bundle"]`` (the full result dict
    is accepted too). ``registry`` must list the same program ids in the same order as the
    frozen coordinates. A program is survivable when its ``features`` are a subset of
    ``target_features``; non-survivable coordinates are forced to 0 even if
    ``external_firings`` reports firings for them (those are counted in
    ``suppressed_firings``, which should be empty). Fractions use all registry programs as
    the denominator. ``labels`` (optional) adds AUPRC/AUROC.
    """
    bundle = model_bundle.get("model_bundle", model_bundle)
    model = bundle["model"]
    names = list(bundle["coordinate_names"])
    encoding = str(bundle.get("encoding", "binary"))
    registry_ids = _registry_ids(registry)
    if registry_ids != names:
        raise ValueError(
            "registry does not match the frozen bridge coordinates (ids or order differ); "
            "refiring a different registry would be a refit"
        )
    n_features_in = getattr(model, "n_features_in_", len(names))
    if n_features_in != len(names):
        raise ValueError(f"frozen model expects {n_features_in} coordinates but the registry has {len(names)}")

    target = frozenset(target_features)
    episode_ids = list(external_episode_ids)
    matrix, _ = compile_bridge(registry, external_firings, episode_ids, encoding)
    episode_set = frozenset(episode_ids)
    mask = np.array([frozenset(program.features) <= target for program in registry], dtype=bool)
    fired = {
        program.program_id: len(frozenset(external_firings.get(program.program_id, _EMPTY)) & episode_set)
        for program in registry
    }
    matrix[:, ~mask] = 0.0
    probabilities = model.predict_proba(matrix)[:, 1] if episode_ids else np.zeros(0)

    surviving = [pid for pid, keep in zip(names, mask) if keep]
    nonsurvivable = [pid for pid, keep in zip(names, mask) if not keep]
    firing_ids = [pid for pid in surviving if fired[pid] > 0]
    suppressed = {pid: fired[pid] for pid in nonsurvivable if fired[pid] > 0}
    n = len(names)
    notes = [
        "frozen probe, C, and coordinate order reused; nothing refit or repaired",
        "non-survivable coordinates are 0 for every target episode",
    ]
    if suppressed:
        notes.append(
            f"firings were reported for non-survivable programs {sorted(suppressed)} and were zeroed; "
            "this indicates target-specific repair or a schema mismatch upstream"
        )
    result = {
        "probabilities": {episode_id: float(p) for episode_id, p in zip(episode_ids, probabilities)},
        "survival_mask": [bool(keep) for keep in mask],
        "coordinate_names": names,
        "surviving_ids": surviving,
        "nonsurvivable_ids": nonsurvivable,
        "schema_survival": len(surviving) / n if n else 0.0,
        "firing_ids": firing_ids,
        "firing_survival": len(firing_ids) / n if n else 0.0,
        "suppressed_firings": suppressed,
        "n_episodes": len(episode_ids),
        "target_features": sorted(target),
        "notes": notes,
    }
    if labels is not None:
        y = _as_labels(labels, "labels")
        if y.shape[0] != len(episode_ids):
            raise ValueError(f"{len(episode_ids)} external episodes but {y.shape[0]} labels")
        result["metrics"] = _binary_metrics(y, probabilities)
    return result
