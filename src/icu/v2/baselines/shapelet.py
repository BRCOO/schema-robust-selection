"""Exact information-gain shapelets (Ye & Keogh, KDD 2009), brute force (v2 numeric baseline).

The v1 comparator ``src/icu/comparators/run_shapelet.py`` did not implement shapelets. It
used raw distances scaled by ``max(std, 1)``, quantile thresholds with fallback rules, the
first 24 positives, the first 10 features, two windows per episode, and no
information-gain search, although the paper cites learned shapelets. This module
implements the original shapelet definition exactly. Every valid subsequence of every
training series is a candidate. Its distance to every training series is computed, and
the split point with the largest information gain is found by exhaustive search. Series
access, window validity, and distances follow the ``motif.py`` conventions (mask True =
missing, strict or ffill missing policy, constant windows excluded under
z-normalization, length-normalized distances).

Definitions
-----------
* ``subsequence_distance(T, S)`` is the minimum of ``||w' - S'|| / sqrt(m)`` over the valid
  windows ``w`` of ``T``. Here ``'`` is z-normalization when ``normalize=True`` (default)
  and the identity otherwise. Dividing by ``sqrt(m)`` makes lengths comparable; for
  z-normalized windows the distance is ``sqrt(2 * (1 - rho))`` in ``[0, 2]``. A series
  without a valid window has distance ``inf``: the shapelet does not occur in it.
* A split point ``tau`` divides the series into ``D_near = {dist <= tau}`` and
  ``D_far = {dist > tau}``. The candidates are the midpoints between consecutive distinct
  finite sorted distances (gaps of at most ``SPLIT_TOL`` count as ties). The split that
  only separates the series with a finite distance from those without one is not a
  candidate, because it does not depend on the shapelet.
* Information gain, in bits:
  ``IG = H(D) - |D_near|/|D| * H(D_near) - |D_far|/|D| * H(D_far)``, with ``H`` the binary
  entropy (log2).
* ``margin`` is the separation gap of Ye & Keogh: the mean finite distance in ``D_far``
  minus the mean distance in ``D_near``. IG ties (within 1e-12) are broken by the larger
  margin, then by the smaller threshold.
* ``direction`` is ``"le"`` if the positive rate is higher in ``D_near``, i.e.
  ``distance <= threshold`` indicates the positive class, and ``"gt"`` otherwise. A
  ``"gt"`` program also fires on an episode where the feature has no valid window, which is
  exactly where that episode falls in the training split.
* Candidates are ranked by IG (desc), margin (desc), threshold, length, series, start.
  Candidates with no information gain are dropped. Similarity pruning: a candidate is
  skipped if its template is within ``max(threshold_a, threshold_b)`` (or ``prune_radius``)
  of an accepted template of the same length. Identical templates (equal after rounding
  to 1e-10) are evaluated once, at their first occurrence.

The stored v1 arrays are robust-scaled per episode, so ``normalize=False`` compares values
on per-episode scales, not clinical units. The z-normalized shape is the meaningful
default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import entr

from src.icu.v2.baselines.motif import (
    DEFAULT_EPS,
    DEFAULT_LENGTHS,
    MATCH_ATOL,
    EpisodeArrays,
    WindowBank,
    _as_series_list,
    _binary_label,
    _check_float,
    _check_int,
    _check_lengths,
    _check_policy,
    _feature_order,
    block_starts,
    build_window_bank,
    chunk_rows,
    feature_series,
    sq_distances,
    template_bank,
)
from src.icu.v2.types import ProgramRecord, coarse_group

SPLIT_TOL = 1e-8  # adjacent sorted distances closer than this are ties (no split between them)
GAIN_TOL = 1e-12
MARGIN_TOL = 1e-12
RANK_DECIMALS = 12
DIRECTIONS: tuple[str, ...] = ("le", "gt")
_DEDUP_DECIMALS = 10
_LN2 = math.log(2.0)


def subsequence_distance(
    series: np.ndarray,
    shapelet: np.ndarray,
    normalize: bool = True,
    *,
    eps: float = DEFAULT_EPS,
) -> float:
    """Minimum length-normalized distance from ``shapelet`` to the valid windows of ``series``.

    Returns ``inf`` if ``series`` has no valid window (too short, unobserved, or only
    constant windows when ``normalize``). Raises ValueError for a constant shapelet under
    ``normalize``, since it has no shape.
    """
    query = template_bank(np.asarray(shapelet, dtype=float).reshape(-1), normalize=normalize, eps=eps)
    windows = build_window_bank([series], query.m, normalize=normalize, eps=eps)
    if len(windows) == 0:
        return math.inf
    return float(math.sqrt(float(sq_distances(query, windows).min())))


# ------------------------------------------------------------------------ split search


@dataclass(frozen=True, eq=False)
class _Splits:
    threshold: np.ndarray
    gain: np.ndarray
    margin: np.ndarray
    positive_near: np.ndarray
    n_near: np.ndarray
    n_near_positive: np.ndarray


def _binary_labels(labels: Sequence[int], n: int) -> np.ndarray:
    y = np.asarray(labels).reshape(-1)
    if y.shape[0] != n:
        raise ValueError(f"{y.shape[0]} labels for {n} series")
    if y.size and not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return y.astype(np.int64)


def _entropy_bits(p: np.ndarray) -> np.ndarray:
    return (entr(p) + entr(1.0 - p)) / _LN2


def _best_splits(distances: np.ndarray, y: np.ndarray) -> _Splits:
    """Exact best split of every row of ``distances`` (rows = shapelets, columns = series)."""
    rows, n = distances.shape
    out = _Splits(
        threshold=np.full(rows, np.nan),
        gain=np.zeros(rows),
        margin=np.zeros(rows),
        positive_near=np.ones(rows, dtype=bool),
        n_near=np.zeros(rows, dtype=np.int64),
        n_near_positive=np.zeros(rows, dtype=np.int64),
    )
    if rows == 0 or n < 2:
        return out
    order = np.argsort(distances, axis=1, kind="stable")
    ordered = np.take_along_axis(distances, order, axis=1)
    finite = np.isfinite(ordered)
    n_finite = finite.sum(axis=1)
    n_pos = int(y.sum())
    near_size = np.arange(1, n)  # split k puts the k smallest distances in D_near
    with np.errstate(invalid="ignore", divide="ignore"):
        gaps = ordered[:, 1:] - ordered[:, :-1]
        valid = (near_size[None, :] < n_finite[:, None]) & (gaps > SPLIT_TOL)

        near_pos = np.cumsum(y[order], axis=1)[:, :-1]
        near_n = near_size[None, :].astype(float)
        far_n = n - near_n
        gain = (
            _entropy_bits(np.asarray(n_pos / n))
            - (near_n / n) * _entropy_bits(near_pos / near_n)
            - (far_n / n) * _entropy_bits((n_pos - near_pos) / far_n)
        )
        gain = np.where(valid, np.maximum(gain, 0.0), -np.inf)

        sums = np.cumsum(np.where(finite, ordered, 0.0), axis=1)
        near_sum = sums[:, :-1]
        far_sum = sums[:, -1:] - near_sum
        far_finite = n_finite[:, None] - near_size[None, :]
        margin = far_sum / far_finite - near_sum / near_n
        threshold = (ordered[:, :-1] + ordered[:, 1:]) / 2.0

    best_gain = gain.max(axis=1)
    tied = valid & (gain >= best_gain[:, None] - GAIN_TOL)
    tied_margin = np.where(tied, margin, -np.inf)
    tied &= tied_margin >= tied_margin.max(axis=1)[:, None] - MARGIN_TOL
    pick = np.argmax(tied, axis=1)  # first tied split = smallest threshold
    has = np.isfinite(best_gain)
    index = np.arange(rows)[has]
    k = pick[has]
    n_near = k + 1
    n_near_positive = near_pos[index, k]
    out.threshold[has] = threshold[index, k]
    out.gain[has] = gain[index, k]
    out.margin[has] = margin[index, k]
    out.positive_near[has] = n_near_positive / n_near >= (n_pos - n_near_positive) / (n - n_near)
    out.n_near[has] = n_near
    out.n_near_positive[has] = n_near_positive
    return out


def best_split(distances: Sequence[float], labels: Sequence[int]) -> tuple[float, float, float]:
    """Exact optimal split point of one shapelet (Ye & Keogh 2009).

    ``distances`` holds one value per series (``inf`` = no valid window) and ``labels`` holds
    0/1. Returns ``(threshold, information_gain, margin)``, with the information gain in
    bits and the margin equal to the separation gap. Returns ``(nan, 0.0, 0.0)`` if there is
    no split point, i.e. fewer than two distinct finite distances.
    """
    d = np.asarray(distances, dtype=float).reshape(1, -1)
    if np.isnan(d).any():
        raise ValueError("distances must not be NaN (use inf for series without a valid window)")
    split = _best_splits(d, _binary_labels(labels, d.shape[1]))
    return float(split.threshold[0]), float(split.gain[0]), float(split.margin[0])


# -------------------------------------------------------------------------- discovery


@dataclass(frozen=True)
class ShapeletCandidate:
    """One shapelet. ``series_index`` indexes the input series list.

    ``template`` is the z-normalized source window when ``normalize``, else the raw window.
    ``raw_template`` holds the stored values. ``n_near`` / ``n_near_positive`` describe
    ``D_near`` of the training split.
    """

    length: int
    template: tuple[float, ...]
    raw_template: tuple[float, ...]
    series_index: int
    start: int
    threshold: float
    information_gain: float
    margin: float
    direction: str
    n_near: int
    n_near_positive: int
    n_positive: int
    n_total: int


@dataclass(frozen=True, eq=False)
class _ShapeletPool:
    bank: WindowBank
    splits: _Splits
    n_positive: int
    n_total: int

    def candidate(self, i: int) -> ShapeletCandidate:
        if self.bank.normalize:
            template = self.bank.shape[i] * math.sqrt(self.bank.m)
        else:
            template = self.bank.raw[i]
        return ShapeletCandidate(
            length=self.bank.m,
            template=tuple(float(v) for v in template),
            raw_template=tuple(float(v) for v in self.bank.raw[i]),
            series_index=int(self.bank.owner[i]),
            start=int(self.bank.start[i]),
            threshold=float(self.splits.threshold[i]),
            information_gain=float(self.splits.gain[i]),
            margin=float(self.splits.margin[i]),
            direction="le" if self.splits.positive_near[i] else "gt",
            n_near=int(self.splits.n_near[i]),
            n_near_positive=int(self.splits.n_near_positive[i]),
            n_positive=self.n_positive,
            n_total=self.n_total,
        )


def _first_unique_rows(bank: WindowBank, rows: np.ndarray) -> np.ndarray:
    """``rows`` without duplicate templates (rounded to 1e-10), keeping first occurrences."""
    if rows.size < 2:
        return rows
    keys = np.column_stack([bank.shape[rows], bank.level[rows]])
    keys = np.round(keys, _DEDUP_DECIMALS) + 0.0  # + 0.0 turns -0.0 into 0.0
    _, first = np.unique(keys, axis=0, return_index=True)
    return rows[np.sort(first)]


def _concat_splits(parts: list[_Splits]) -> _Splits:
    fields = ("threshold", "gain", "margin", "positive_near", "n_near", "n_near_positive")
    return _Splits(**{name: np.concatenate([getattr(part, name) for part in parts]) for name in fields})


def discover_shapelets(
    series_list: Sequence[np.ndarray],
    labels: Sequence[int],
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    top_k: int = 3,
    *,
    normalize: bool = True,
    positive_sources_only: bool = False,
    prune_radius: float | None = None,
    eps: float = DEFAULT_EPS,
) -> list[ShapeletCandidate]:
    """Top-``k`` information-gain shapelets of one feature (exact brute force).

    Every valid window of every series (only of positive series if
    ``positive_sources_only``) is a candidate and is scored against all series with
    :func:`best_split`. See the module docstring for ranking and pruning. Returns ``[]``
    when both classes are not present.
    """
    top_k = _check_int("top_k", top_k, 0)
    if prune_radius is not None:
        prune_radius = _check_float("prune_radius", prune_radius)
    checked_lengths = _check_lengths(lengths, 2 if normalize else 1)
    series_list = _as_series_list(series_list)
    y = _binary_labels(labels, len(series_list))
    n = len(series_list)
    if top_k == 0 or n < 2 or y.min() == y.max():
        return []

    pools: list[_ShapeletPool] = []
    for m in checked_lengths:
        bank = build_window_bank(series_list, m, normalize=normalize, eps=eps)
        if len(bank) == 0:
            continue
        rows = np.arange(len(bank))
        if positive_sources_only:
            rows = rows[y[bank.owner] == 1]
        rows = _first_unique_rows(bank, rows)
        if rows.size == 0:
            continue
        candidates = bank.take(rows)
        starts = block_starts(bank.owner)
        series_with_windows = bank.owner[starts]
        step = chunk_rows(len(bank))
        parts: list[_Splits] = []
        for r0 in range(0, len(candidates), step):
            r1 = min(len(candidates), r0 + step)
            d2 = sq_distances(candidates.take(slice(r0, r1)), bank)
            per_series = np.full((r1 - r0, n), np.inf)
            per_series[:, series_with_windows] = np.sqrt(np.minimum.reduceat(d2, starts, axis=1))
            parts.append(_best_splits(per_series, y))
        splits = _concat_splits(parts)
        keep = np.flatnonzero(np.isfinite(splits.threshold) & (splits.gain > GAIN_TOL))
        if keep.size == 0:
            continue
        kept_splits = _Splits(**{name: getattr(splits, name)[keep] for name in _Splits.__dataclass_fields__})
        pools.append(_ShapeletPool(candidates.take(keep), kept_splits, int(y.sum()), n))
    if not pools:
        return []

    pool_id = np.concatenate([np.full(len(pool.bank), p) for p, pool in enumerate(pools)])
    local = np.concatenate([np.arange(len(pool.bank)) for pool in pools])
    order = np.lexsort(
        (
            np.concatenate([pool.bank.start for pool in pools]),
            np.concatenate([pool.bank.owner for pool in pools]),
            np.concatenate([np.full(len(pool.bank), pool.bank.m) for pool in pools]),
            np.concatenate([pool.splits.threshold for pool in pools]),
            -np.round(np.concatenate([pool.splits.margin for pool in pools]), RANK_DECIMALS),
            -np.round(np.concatenate([pool.splits.gain for pool in pools]), RANK_DECIMALS),
        )
    )
    alive = [np.ones(len(pool.bank), dtype=bool) for pool in pools]
    chosen: list[tuple[int, int]] = []
    for position in order:
        p, i = int(pool_id[position]), int(local[position])
        if not alive[p][i]:
            continue
        chosen.append((p, i))
        if len(chosen) == top_k:
            break
        pool = pools[p]
        distance = np.sqrt(sq_distances(pool.bank.take(slice(i, i + 1)), pool.bank)[0])
        if prune_radius is None:
            radius = np.maximum(pool.splits.threshold[i], pool.splits.threshold)
        else:
            radius = np.full(len(pool.bank), prune_radius)
        alive[p] &= distance > radius + MATCH_ATOL
    return [pools[p].candidate(i) for p, i in chosen]


def mine_shapelet_programs(
    episodes: Mapping[str, EpisodeArrays],
    labels: Mapping[str, int],
    *,
    feature_names: Sequence[str] | None = None,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    top_k_per_feature: int = 3,
    normalize: bool = True,
    positive_sources_only: bool = False,
    prune_radius: float | None = None,
    missing_policy: str = "strict",
    eps: float = DEFAULT_EPS,
    method: str = "shapelet",
) -> list[ProgramRecord]:
    """Shapelet programs from TRAIN episodes, one :func:`discover_shapelets` run per feature.

    ``episodes`` maps episode id -> ``(values, mask, feature_names)`` (mask True = missing)
    and ``labels`` maps episode id -> 0/1. Each feature uses the episodes whose schema has
    it. The result is sorted by score (= information gain in bits), margin, feature
    order, and per-feature rank.
    """
    _check_policy(missing_policy)
    episode_ids = sorted(episodes)
    label_of = {eid: _binary_label(labels, eid) for eid in episode_ids}
    features = list(dict.fromkeys(feature_names)) if feature_names is not None else _feature_order(episodes)

    ranked: list[tuple[tuple, ProgramRecord]] = []
    for feature_index, feature in enumerate(features):
        kept: list[str] = []
        series_list: list[np.ndarray] = []
        for eid in episode_ids:
            values, mask, names = episodes[eid]
            series = feature_series(values, mask, names, feature, missing_policy)
            if series is not None:
                kept.append(eid)
                series_list.append(series)
        candidates = discover_shapelets(
            series_list,
            [label_of[eid] for eid in kept],
            lengths,
            top_k_per_feature,
            normalize=normalize,
            positive_sources_only=positive_sources_only,
            prune_radius=prune_radius,
            eps=eps,
        )
        for rank, shapelet in enumerate(candidates):
            program = ProgramRecord(
                program_id=f"{method}:{feature}:L{shapelet.length}:{rank}",
                method=method,
                kind="shapelet",
                features=frozenset({feature}),
                anchor_signature=frozenset({coarse_group(feature)}),
                score=shapelet.information_gain,
                params={
                    "feature": feature,
                    "template": shapelet.template,
                    "threshold": shapelet.threshold,
                    "length": shapelet.length,
                    "normalize": bool(normalize),
                    "information_gain": shapelet.information_gain,
                    "direction": shapelet.direction,
                    "margin": shapelet.margin,
                    "n_near": shapelet.n_near,
                    "n_near_positive": shapelet.n_near_positive,
                    "n_positive": shapelet.n_positive,
                    "n_total": shapelet.n_total,
                    "raw_template": shapelet.raw_template,
                    "source_episode": kept[shapelet.series_index],
                    "source_start": shapelet.start,
                    "positive_sources_only": bool(positive_sources_only),
                    "missing_policy": missing_policy,
                    "eps": eps,
                    "rank": rank,
                },
            )
            sort_key = (-round(shapelet.information_gain, RANK_DECIMALS), -round(shapelet.margin, RANK_DECIMALS))
            ranked.append((sort_key + (feature_index, rank), program))
    ranked.sort(key=lambda item: item[0])
    return [program for _, program in ranked]
