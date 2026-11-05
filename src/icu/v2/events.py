"""Shared, miner-independent event representation for the v2 protocol.

Every symbolic discovery method (FMG-Bridge and the SPMF baselines) mines the same
``EventSequence`` objects produced here, so differences between methods are not
confounded with differences in how episodes were turned into events.

Representation
--------------
* One itemset per time bin (one row of the resampled matrix; 60 min in MIMIC-IV).
  Tokens of the same bin stay together as an itemset, so there is no arbitrary
  within-bin order. Bins without tokens are kept as empty itemsets, so position ``t``
  is always bin ``t``. There is no length cap.
* ``"<f>:q<k>"`` (value tokens): the observed value of ``f`` falls in global quantile
  bin ``k``. Edges are fitted once on TRAIN episodes by :func:`fit_feature_bins`
  (pooled over all observed train bins) and then frozen for validation, test, and
  external episodes. Edges are never fitted per episode.
* ``"<f>:on"`` for the binary intervention features (``ventilation_on``,
  ``vasopressor_on``, ``sedation_on`` when present) when the value is > 0.5. The off
  state emits no token.
* ``observation_tokens=True`` adds ``"<f>:missing"`` and ``"<f>:sparse_measurement"``
  for the observation-density features (default: the 14 vitals and labs in
  ``icu_eventizer.DEFAULT_MISSINGNESS_FEATURES``).
* ``trend_tokens=True`` adds ``"<f>:up"`` / ``"<f>:down"`` for value features and
  ``"<f>:onset"`` / ``"<f>:offset"`` for intervention features.

Relation to FMG's v1 eventizer
------------------------------
``src.icu.icu_eventizer.ICUEventizer`` with the MIMIC configuration of
``scripts/run_mimic_experiment.py`` (n_bins=4, window_size=6, quantile_stride=1,
change_threshold=0.25, no support filter). Its feature sets and sign table are imported
from that module; the token rules are re-implemented here because ``ICUEventizer``
fits its level edges per episode (on a trailing window), which v2 must avoid, and it
returns integer ids rather than tokens.

=====================  ==========================  ====================================
v2 token               ICUEventizer token          relation
=====================  ==========================  ====================================
f:q<k>                 f:level_<k>                 Same assignment rule
                                                   (``searchsorted(inner_edges, x,
                                                   side="right")``). v2 edges are global
                                                   train quantiles (observed values,
                                                   ``inverted_cdf``). v1 edges are
                                                   linearly interpolated quantiles of
                                                   the episode's own trailing 6-bin
                                                   window of interpolated values. On
                                                   ties, v2 drops inner edges equal to
                                                   the train minimum; v1 de-duplicates
                                                   all edges, which collapses a 0/1
                                                   series into a single level.
f:on (interventions)   (none)                      v1 emits level_0/up/down from window
                                                   quantiles of the 0/1 series plus
                                                   onset/offset. v2 states the binary
                                                   state directly.
f:missing              f:missing                   Identical rule: emitted when ``f``
                                                   becomes unobserved and then on every
                                                   ``missing_repeat_stride`` (=3)-th
                                                   unobserved bin of the run (run
                                                   lengths 1, 3, 6, 9, ...).
f:sparse_measurement   f:sparse_measurement        Identical rule: trailing window of
                                                   ``observation_window`` (=6) bins with
                                                   at least 3 bins and
                                                   0 < observed < window length, and
                                                   observed fraction <= 0.5.
f:up / f:down          f:up / f:down               v2: the global quantile bin differs
                                                   from the bin at the previous
                                                   observation of ``f`` in the episode.
                                                   v1: |delta| between consecutive
                                                   interpolated values > 0.25 * window
                                                   std.
f:onset / f:offset     f:onset / f:offset          Identical rule: state change between
                                                   consecutive observed bins
                                                   (threshold 0.5).
(not emitted)          f:deterioration/recovery    With bin-change trends these would
                                                   be exact aliases of up/down under
                                                   ``DEFAULT_DETERIORATION_SIGN``, and
                                                   aliased tokens inflate pattern
                                                   redundancy.
                                                   :func:`translate_icu_eventizer_token`
                                                   maps them.
=====================  ==========================  ====================================

Input scale caveat
------------------
Global bins are meaningful only when values share one scale across episodes (raw
clinical units). The v1 episode matrices under ``data/processed/...`` are robust-scaled
per episode by ``src/icu/mimic_iv_features.py::_robust_scale_columns``: the
interpolate-filled features (vitals and labs) are stored as
``(x - episode median) / episode IQR``. Binning those values with global edges still
yields per-episode relative levels, so raw values must be re-materialized before
fitting bins. Zero-filled features (interventions, ``urine_output``) are unscaled.
"""

from __future__ import annotations

from typing import Collection, Mapping, Sequence

import numpy as np

from src.icu.icu_eventizer import (
    DEFAULT_DETERIORATION_SIGN,
    DEFAULT_INTERVENTION_FEATURES,
    DEFAULT_MISSINGNESS_FEATURES,
)
from src.icu.v2.types import EventSequence, feature_of

INTERVENTION_FEATURES: frozenset[str] = frozenset(DEFAULT_INTERVENTION_FEATURES)
OBSERVATION_FEATURES: frozenset[str] = frozenset(DEFAULT_MISSINGNESS_FEATURES)
INTERVENTION_ON_THRESHOLD = 0.5  # ICUEventizer.intervention_on_threshold
OBSERVATION_WINDOW = 6  # ICUEventizer window_size in the MIMIC configuration
SPARSE_THRESHOLD = 0.5  # ICUEventizer.sparse_observation_threshold
MISSING_REPEAT_STRIDE = 3  # ICUEventizer.missing_repeat_stride
_MIN_SPARSE_WINDOW = 3  # ICUEventizer requires window_length >= 3


def _as_matrix(values: np.ndarray, width: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise ValueError(f"expected a (T, {width}) matrix, got shape {matrix.shape}")
    return matrix


def _observed(matrix: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """True where a value is observed: finite and, if a mask is given, not masked.

    ``mask`` follows the repository convention of ``*_missing_mask.npy``
    (True = missing).
    """
    observed = np.isfinite(matrix)
    if mask is not None:
        missing = np.asarray(mask, dtype=bool)
        if missing.shape != matrix.shape:
            raise ValueError(f"mask shape {missing.shape} != values shape {matrix.shape}")
        observed &= ~missing
    return observed


def _check_unique(names: Sequence[str]) -> list[str]:
    names = [str(name) for name in names]
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    return names


def fit_feature_bins(
    train_arrays: Sequence[np.ndarray],
    feature_names: Sequence[str],
    n_bins: int = 4,
    *,
    missing_masks: Sequence[np.ndarray] | None = None,
    intervention_features: Collection[str] = INTERVENTION_FEATURES,
) -> dict[str, np.ndarray]:
    """Fit global quantile edges per value feature. Call this on TRAIN episodes only.

    Observed values (finite and not masked; ``missing_masks`` uses True = missing) are
    pooled over all bins of all training episodes. For every feature that is not a
    binary intervention feature the result holds ``[train_min, *cuts, train_max]``.
    ``cuts`` are the distinct inner quantiles ``q(1/n), ..., q((n-1)/n)`` that lie
    strictly above the train minimum. The quantiles are taken with
    ``method="inverted_cdf"``, so every cut is an observed train value and every bin
    holds training data. A feature therefore has ``len(edges) - 1`` bins, fewer than
    ``n_bins`` when quantiles tie (e.g. zero-inflated ``urine_output``). A feature never
    observed in training maps to an empty array and emits no value token.
    """
    if int(n_bins) < 1:
        raise ValueError("n_bins must be >= 1")
    names = _check_unique(feature_names)
    if missing_masks is not None and len(missing_masks) != len(train_arrays):
        raise ValueError("missing_masks must align with train_arrays")
    value_columns = [(j, name) for j, name in enumerate(names) if name not in intervention_features]
    pooled: dict[str, list[np.ndarray]] = {name: [] for _, name in value_columns}
    for idx, array in enumerate(train_arrays):
        matrix = _as_matrix(array, len(names))
        observed = _observed(matrix, None if missing_masks is None else missing_masks[idx])
        for j, name in value_columns:
            pooled[name].append(matrix[observed[:, j], j])

    probs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges: dict[str, np.ndarray] = {}
    for _, name in value_columns:
        column = np.concatenate(pooled[name]) if pooled[name] else np.empty(0)
        if column.size == 0:
            edges[name] = np.empty(0, dtype=float)
            continue
        quantiles = np.quantile(column, probs, method="inverted_cdf")
        low, high = float(quantiles[0]), float(quantiles[-1])
        cuts = np.unique(quantiles[1:-1])
        cuts = cuts[cuts > low]
        edges[name] = np.concatenate(([low], cuts, [high])).astype(float)
    return edges


def _add_value_tokens(
    itemsets: list[set[str]],
    name: str,
    column: np.ndarray,
    observed: np.ndarray,
    edges: np.ndarray,
    trend_tokens: bool,
) -> None:
    if edges.size == 0:
        return
    positions = np.flatnonzero(observed)
    levels = np.searchsorted(edges[1:-1], column[positions], side="right")
    previous: int | None = None
    for t, level in zip(positions.tolist(), levels.tolist()):
        itemsets[t].add(f"{name}:q{level}")
        if trend_tokens and previous is not None and level != previous:
            itemsets[t].add(f"{name}:up" if level > previous else f"{name}:down")
        previous = level


def _add_intervention_tokens(
    itemsets: list[set[str]],
    name: str,
    column: np.ndarray,
    observed: np.ndarray,
    trend_tokens: bool,
) -> None:
    previous: bool | None = None
    for t in np.flatnonzero(observed).tolist():
        state = bool(column[t] > INTERVENTION_ON_THRESHOLD)
        if state:
            itemsets[t].add(f"{name}:on")
        if trend_tokens and previous is not None and state != previous:
            itemsets[t].add(f"{name}:onset" if state else f"{name}:offset")
        previous = state


def _add_observation_tokens(
    itemsets: list[set[str]],
    name: str,
    observed: np.ndarray,
    window: int,
    sparse_threshold: float,
    repeat_stride: int,
) -> None:
    observed_before = np.concatenate(([0], np.cumsum(observed, dtype=int)))
    streak = 0
    for t in range(observed.shape[0]):
        if observed[t]:
            streak = 0
        else:
            streak += 1
            if streak == 1 or streak % repeat_stride == 0:
                itemsets[t].add(f"{name}:missing")
        start = max(0, t - window + 1)
        length = t + 1 - start
        count = int(observed_before[t + 1] - observed_before[start])
        if length >= _MIN_SPARSE_WINDOW and 0 < count < length and count / length <= sparse_threshold:
            itemsets[t].add(f"{name}:sparse_measurement")


def eventize_episode(
    values: np.ndarray,
    mask: np.ndarray | None,
    feature_names: Sequence[str],
    bins: Mapping[str, np.ndarray],
    *,
    observation_tokens: bool = True,
    trend_tokens: bool = False,
    intervention_features: Collection[str] = INTERVENTION_FEATURES,
    observation_features: Collection[str] = OBSERVATION_FEATURES,
    observation_window: int = OBSERVATION_WINDOW,
    sparse_threshold: float = SPARSE_THRESHOLD,
    missing_repeat_stride: int = MISSING_REPEAT_STRIDE,
) -> EventSequence:
    """Turn one ``(T, D)`` episode into ``T`` itemsets with the shared vocabulary.

    ``mask`` uses True = missing (``None``: missing = non-finite). ``bins`` comes from
    :func:`fit_feature_bins` on TRAIN episodes and must cover every non-intervention
    feature. See the module docstring for the token rules.
    """
    if int(observation_window) < 1 or int(missing_repeat_stride) < 1:
        raise ValueError("observation_window and missing_repeat_stride must be >= 1")
    if not 0.0 <= float(sparse_threshold) <= 1.0:
        raise ValueError("sparse_threshold must be in [0, 1]")
    names = _check_unique(feature_names)
    matrix = _as_matrix(values, len(names))
    observed = _observed(matrix, mask)
    itemsets: list[set[str]] = [set() for _ in range(matrix.shape[0])]
    for j, name in enumerate(names):
        if name in intervention_features:
            _add_intervention_tokens(itemsets, name, matrix[:, j], observed[:, j], trend_tokens)
        else:
            if name not in bins:
                raise KeyError(f"no bins fitted for feature {name!r}")
            edges = np.asarray(bins[name], dtype=float)
            _add_value_tokens(itemsets, name, matrix[:, j], observed[:, j], edges, trend_tokens)
        if observation_tokens and name in observation_features:
            _add_observation_tokens(
                itemsets,
                name,
                observed[:, j],
                int(observation_window),
                float(sparse_threshold),
                int(missing_repeat_stride),
            )
    return tuple(frozenset(itemset) for itemset in itemsets)


def _bin_range_text(edges: np.ndarray, level: int) -> str:
    cuts = edges[1:-1]
    if cuts.size == 0:
        return "any observed value"
    if level == 0:
        return f"value < {cuts[0]:.4g}"
    if level == cuts.size:
        return f"value >= {cuts[-1]:.4g}"
    return f"{cuts[level - 1]:.4g} <= value < {cuts[level]:.4g}"


def describe_vocabulary(
    feature_names: Sequence[str],
    bins: Mapping[str, np.ndarray],
    *,
    observation_tokens: bool = True,
    trend_tokens: bool = False,
    intervention_features: Collection[str] = INTERVENTION_FEATURES,
    observation_features: Collection[str] = OBSERVATION_FEATURES,
) -> list[dict[str, str]]:
    """List every token :func:`eventize_episode` can emit with these settings.

    One row per token with ``token``, ``feature``, ``family``, ``definition``, and
    ``icu_eventizer`` (the v1 counterpart), intended for the paper's vocabulary table.
    """
    rows: list[dict[str, str]] = []

    def add(feature: str, state: str, family: str, definition: str, counterpart: str) -> None:
        rows.append(
            {
                "token": f"{feature}:{state}",
                "feature": feature,
                "family": family,
                "definition": definition,
                "icu_eventizer": counterpart,
            }
        )

    for name in _check_unique(feature_names):
        if name in intervention_features:
            add(name, "on", "intervention", f"value > {INTERVENTION_ON_THRESHOLD} in the bin",
                "none (v1 emits level/up/down from window quantiles of the 0/1 series)")
            if trend_tokens:
                add(name, "onset", "trend", "off -> on between consecutive observed bins",
                    f"{name}:onset (same rule)")
                add(name, "offset", "trend", "on -> off between consecutive observed bins",
                    f"{name}:offset (same rule)")
        else:
            if name not in bins:
                raise KeyError(f"no bins fitted for feature {name!r}")
            edges = np.asarray(bins[name], dtype=float)
            n_levels = max(1, edges.size - 1) if edges.size else 0
            for level in range(n_levels):
                add(name, f"q{level}", "value",
                    f"observed, global train quantile bin {level + 1}/{n_levels}: "
                    f"{_bin_range_text(edges, level)}",
                    f"{name}:level_{level} (per-episode trailing-window quantile bin)")
            if trend_tokens and n_levels > 1:
                add(name, "up", "trend", "quantile bin above the bin at the previous observation",
                    f"{name}:up (v1: |delta| > 0.25 * window std)")
                add(name, "down", "trend", "quantile bin below the bin at the previous observation",
                    f"{name}:down (v1: |delta| > 0.25 * window std)")
        if observation_tokens and name in observation_features:
            add(name, "missing", "observation",
                f"unobserved; emitted at run lengths 1, {MISSING_REPEAT_STRIDE}, "
                f"{2 * MISSING_REPEAT_STRIDE}, ... of an unobserved run",
                f"{name}:missing (same rule)")
            add(name, "sparse_measurement", "observation",
                f"trailing {OBSERVATION_WINDOW}-bin window (>= {_MIN_SPARSE_WINDOW} bins) "
                f"partly observed with observed fraction <= {SPARSE_THRESHOLD}",
                f"{name}:sparse_measurement (same rule)")
    return rows


def translate_icu_eventizer_token(
    token: str,
    *,
    intervention_features: Collection[str] = INTERVENTION_FEATURES,
) -> str | None:
    """Map an ``ICUEventizer`` label to the closest v2 token (``None`` if none exists).

    The mapping is nominal for ``level_<k> -> q<k>`` (v1 edges are per-episode window
    quantiles, v2 edges are global train quantiles); ``deterioration``/``recovery``
    become ``up``/``down`` through ``DEFAULT_DETERIORATION_SIGN``.
    """
    feature = feature_of(token)
    state = token[len(feature) + 1 :] if ":" in token else ""
    if state in ("missing", "sparse_measurement", "onset", "offset"):
        return token
    if feature in intervention_features:
        return {"up": f"{feature}:onset", "down": f"{feature}:offset"}.get(state)
    if state in ("up", "down"):
        return token
    if state.startswith("level_") and state[len("level_"):].isdigit():
        return f"{feature}:q{int(state[len('level_'):])}"
    if state in ("deterioration", "recovery"):
        sign = DEFAULT_DETERIORATION_SIGN.get(feature)
        if sign is None:
            return None
        worse = "up" if sign > 0 else "down"
        better = "down" if sign > 0 else "up"
        return f"{feature}:{worse if state == 'deterioration' else better}"
    return None
