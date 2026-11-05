"""Exact cross-episode matrix-profile motifs (v2 numeric baseline).

The v1 comparator ``src/icu/comparators/run_matrix_profile.py`` only approximated motif
discovery. It used the first 20 episodes, the first 10 features, and 2 windows per episode
as templates, and it never computed a matrix profile. Its ``z_normalize`` mapped flat
windows to all-zero templates. The resulting top-10 was a single flat Lactate template
``[0, 0, 0]`` with mean distance ~1e-18, ranked by a 1e-6 score floor. ICU windows are short
(about 12 hourly rows), so the exact brute-force definitions are cheap. This module
implements them.

Conventions shared by all v2 numeric programs
---------------------------------------------
``shapelet.py`` and ``numeric_firing.py`` import these primitives from this module.

* Episode arrays: ``values`` (rows x features, NaN allowed), a ``mask`` of the same shape
  with True = missing (the polarity of ``shock_episode_*_missing_mask.npy``; ``None`` means
  missing = non-finite), and ``feature_names``. A feature is looked up by name.
* Observed points are finite and not masked. At missing positions the stored v1 arrays
  hold imputed values (interpolation in both time directions, then ffill/bfill). Those
  values are never used.
* ``missing_policy="strict"`` (default): a window is valid only if every point is observed.
  ``"ffill"`` (flagged option): a missing point takes the last observed value of the same
  episode (LOCF, never backward in time), and points before the first observation stay
  missing.
* Constant rule: a window is constant if ``std <= eps * max(1, max|x|)`` (``eps = 1e-8``,
  i.e. constant up to rounding). A constant window has no z-normalized shape, so it is
  never a motif candidate and never matches a z-normalized program. All-zero channels
  and carried-forward values produce such windows, and this rule removes the v1
  flat-template artifact.
* Distance: length-normalized Euclidean distance ``||a - b|| / sqrt(m)``. On z-normalized
  windows it equals ``sqrt(2 * (1 - rho))`` in ``[0, 2]``, where ``rho`` is the Pearson
  correlation. It is the usual matrix-profile (stumpy) distance divided by ``sqrt(m)``, so
  motifs of different lengths can be ranked together (the MOEN/VALMOD convention).
  Squared distances at most ``1e-12`` times the summed squared norms are rounding noise
  and are set to 0, as in stumpy. Exactly repeated shapes therefore have distance exactly
  0 in mining and in firing. "Within radius r" means ``d <= r + 1e-9`` everywhere.
* Scale: the stored arrays are robust-scaled per episode, ``(x - median) / IQR``
  (``src/icu/mimic_iv_features.py::_robust_scale_columns``). Absolute levels are therefore
  not comparable across episodes, and raw-unit thresholds cannot be recovered from the
  stored arrays. Z-normalized windows are invariant to that per-episode affine map, so
  shape matching is the meaningful default.

Algorithm
---------
:func:`cross_episode_matrix_profile` is an AB-join. The nearest neighbour of each valid
subsequence is searched only in OTHER series, and with ``groups`` only in series of
another group (e.g. another patient). A pattern repeated inside one episode is therefore
not a motif. It is exact brute force, ``O(W^2 m)`` for ``W`` valid windows, computed in
chunks of at most ``CHUNK_ELEMENTS`` distances to bound memory.

:func:`discover_motifs` ranks every valid subsequence of every length by its profile value
(nearest-neighbour distance). Ties go to more supporting series, then the shorter length,
then the earlier series and start. Each candidate gets the range-motif radius
``r = max(radius_factor * nn_distance, min_radius)``. The default factor 2.0 is the
Mueen-style motif-set radius of twice the pair distance. Candidates are accepted greedily.
A candidate is rejected if its template is within ``max(r_a, r_b)`` of an accepted template
of the same length, so two returned motifs are never within each other's radius (this
also drops the accepted motif's own pair partner). ``support`` counts the other-group
series that contain a window within ``r``.

:func:`mine_motif_programs` runs the discovery per feature on the positive training
episodes (``positive_only=True``, as for the symbolic baselines). ``positive_only=False``
mines over all training episodes, the label-agnostic setting of the v1 comparator. It
returns ``ProgramRecord`` objects. Note that the nn distance shrinks as the number of
episodes grows (and is often exactly 0 for discrete channels). The default radius then
gives very specific programs, and ``min_radius`` sets a fixed matching range instead.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.icu.v2.types import ProgramRecord, coarse_group

DEFAULT_EPS = 1e-8
DEFAULT_LENGTHS: tuple[int, ...] = (3, 4, 5)
MISSING_POLICIES: tuple[str, ...] = ("strict", "ffill")
SNAP_REL = 1e-12  # squared distances <= SNAP_REL * (|a|^2 + |b|^2) are rounding noise -> 0
MATCH_ATOL = 1e-9  # "within radius r" means d <= r + MATCH_ATOL
CHUNK_ELEMENTS = 2_000_000  # max entries of one distance block (16 MB in float64)

EpisodeArrays = tuple[np.ndarray, np.ndarray | None, Sequence[str]]


# ------------------------------------------------------------------------ validation


def _check_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _check_lengths(lengths: Sequence[int], minimum: int) -> tuple[int, ...]:
    checked = tuple(sorted({_check_int("length", m, minimum) for m in lengths}))
    if not checked:
        raise ValueError("lengths must not be empty")
    return checked


def _check_float(name: str, value: object, *, positive: bool = False) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number < 0.0 or (positive and number == 0.0):
        bound = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be a finite number {bound}, got {value!r}")
    return number


def _check_policy(missing_policy: str) -> str:
    if missing_policy not in MISSING_POLICIES:
        raise ValueError(f"missing_policy must be one of {MISSING_POLICIES}, got {missing_policy!r}")
    return missing_policy


def _binary_label(labels: Mapping[str, int], episode_id: str) -> int:
    if episode_id not in labels:
        raise ValueError(f"no label for episode {episode_id!r}")
    value = labels[episode_id]
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if value not in (0, 1):
        raise ValueError(f"label of episode {episode_id!r} must be 0 or 1, got {value!r}")
    return int(value)


def _feature_order(episodes: Mapping[str, EpisodeArrays]) -> list[str]:
    """Feature names in order of first appearance over the episodes sorted by id."""
    seen: dict[str, None] = {}
    for episode_id in sorted(episodes):
        for name in episodes[episode_id][2]:
            seen.setdefault(str(name), None)
    return list(seen)


# --------------------------------------------------------------------- series access


def observed_series(
    column: np.ndarray,
    missing: np.ndarray | None = None,
    missing_policy: str = "strict",
) -> np.ndarray:
    """One feature column as a float series with NaN at unobserved points.

    ``missing`` uses True = missing; observed points must also be finite. ``"strict"`` keeps
    the observed points only. ``"ffill"`` carries the last observed value forward (never
    backward), so points before the first observation stay NaN.
    """
    _check_policy(missing_policy)
    series = np.array(column, dtype=float).reshape(-1)
    observed = np.isfinite(series)
    if missing is not None:
        missing_flags = np.asarray(missing, dtype=bool).reshape(-1)
        if missing_flags.shape != series.shape:
            raise ValueError(f"missing shape {missing_flags.shape} != column shape {series.shape}")
        observed &= ~missing_flags
    series[~observed] = np.nan
    if missing_policy == "ffill" and series.size:
        last = np.where(observed, np.arange(series.size), -1)
        np.maximum.accumulate(last, out=last)
        filled = np.full(series.size, np.nan)
        has_value = last >= 0
        filled[has_value] = series[last[has_value]]
        series = filled
    return series


def feature_series(
    values: np.ndarray,
    mask: np.ndarray | None,
    feature_names: Sequence[str],
    feature: str,
    missing_policy: str = "strict",
) -> np.ndarray | None:
    """Series of ``feature`` in one episode (see :func:`observed_series`).

    Returns None when the episode's schema has no column named ``feature``.
    """
    names = [str(name) for name in feature_names]
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(f"values shape {matrix.shape} does not match {len(names)} feature names")
    missing = None
    if mask is not None:
        missing = np.asarray(mask, dtype=bool)
        if missing.shape != matrix.shape:
            raise ValueError(f"mask shape {missing.shape} != values shape {matrix.shape}")
    _check_policy(missing_policy)
    if feature not in names:
        return None
    column = names.index(feature)
    return observed_series(matrix[:, column], None if missing is None else missing[:, column], missing_policy)


def sliding_windows(series: np.ndarray, m: int) -> np.ndarray:
    """All length-``m`` windows of a 1-D series as rows, shape ``(max(n - m + 1, 0), m)``."""
    m = _check_int("m", m, 1)
    values = np.asarray(series, dtype=float).reshape(-1)
    if values.size < m:
        return np.empty((0, m))
    return sliding_window_view(values, m)


# --------------------------------------------------------------- z-normalized windows


def znorm_rows(windows: np.ndarray, eps: float = DEFAULT_EPS) -> tuple[np.ndarray, np.ndarray]:
    """Z-normalize each row (population std). Returns ``(z, valid)``.

    A row is invalid (all-NaN in ``z``) if it has a non-finite value or is constant, i.e.
    ``std <= eps * max(1, max|x|)``.
    """
    rows = np.asarray(windows, dtype=float)
    if rows.ndim != 2:
        raise ValueError(f"windows must be 2-D, got shape {rows.shape}")
    eps = _check_float("eps", eps)
    n, m = rows.shape
    z = np.full((n, m), np.nan)
    valid = np.zeros(n, dtype=bool)
    if n == 0 or m == 0:
        return z, valid
    finite = np.flatnonzero(np.isfinite(rows).all(axis=1))
    if finite.size == 0:
        return z, valid
    block = rows[finite]
    centered = block - block.mean(axis=1, keepdims=True)
    std = np.sqrt(np.mean(centered * centered, axis=1))
    scale = np.maximum(1.0, np.max(np.abs(block), axis=1))
    shaped = std > eps * scale
    kept = finite[shaped]
    valid[kept] = True
    z[kept] = centered[shaped] / std[shaped, None]
    return z, valid


def znorm(x: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray | None:
    """Z-normalize one subsequence; None if it is not finite or is constant (excluded)."""
    z, valid = znorm_rows(np.asarray(x, dtype=float).reshape(1, -1), eps)
    return z[0] if valid[0] else None


@dataclass(frozen=True, eq=False)
class WindowBank:
    """Valid length-``m`` windows prepared for :func:`sq_distances`.

    ``shape`` rows are z-normalized windows divided by ``sqrt(m)`` (unit norm, ``level`` 0)
    when ``normalize``. Otherwise they are mean-centered raw windows divided by ``sqrt(m)``,
    with the window mean in ``level``. ``owner`` (the series index, non-decreasing for banks
    built from series) and ``start`` locate each window, and ``raw`` holds its values.
    """

    m: int
    normalize: bool
    shape: np.ndarray
    level: np.ndarray
    sq_norm: np.ndarray
    owner: np.ndarray
    start: np.ndarray
    raw: np.ndarray

    def __len__(self) -> int:
        return int(self.owner.shape[0])

    def take(self, index: slice | np.ndarray) -> WindowBank:
        """Subset of the rows (a slice or an integer index array)."""
        return WindowBank(
            self.m,
            self.normalize,
            self.shape[index],
            self.level[index],
            self.sq_norm[index],
            self.owner[index],
            self.start[index],
            self.raw[index],
        )


def _prepare(raw: np.ndarray, normalize: bool, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, m = raw.shape
    root = math.sqrt(m)
    if normalize:
        z, valid = znorm_rows(raw, eps)
        shape = z / root
        level = np.zeros(n)
    else:
        valid = np.isfinite(raw).all(axis=1)
        shape = np.full((n, m), np.nan)
        level = np.full(n, np.nan)
        if valid.any():
            means = raw[valid].mean(axis=1)
            level[valid] = means
            shape[valid] = (raw[valid] - means[:, None]) / root
    sq_norm = np.einsum("ij,ij->i", shape, shape)
    return shape, level, sq_norm, valid


def build_window_bank(
    series_list: Sequence[np.ndarray],
    m: int,
    *,
    normalize: bool = True,
    eps: float = DEFAULT_EPS,
) -> WindowBank:
    """All valid windows of all series: fully observed and, when ``normalize``, non-constant."""
    m = _check_int("m", m, 2 if normalize else 1)
    raws: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    for index, series in enumerate(series_list):
        windows = sliding_windows(series, m)
        if windows.shape[0] == 0:
            continue
        raws.append(windows)
        owners.append(np.full(windows.shape[0], index, dtype=np.int64))
        starts.append(np.arange(windows.shape[0], dtype=np.int64))
    if not raws:
        empty = np.empty((0, m))
        none = np.empty(0, dtype=np.int64)
        return WindowBank(m, normalize, empty, np.empty(0), np.empty(0), none, none.copy(), empty.copy())
    raw = np.concatenate(raws, axis=0)
    shape, level, sq_norm, valid = _prepare(raw, normalize, eps)
    return WindowBank(
        m,
        normalize,
        shape[valid],
        level[valid],
        sq_norm[valid],
        np.concatenate(owners)[valid],
        np.concatenate(starts)[valid],
        raw[valid],
    )


def template_bank(templates: np.ndarray, *, normalize: bool = True, eps: float = DEFAULT_EPS) -> WindowBank:
    """Bank of explicit templates (one per row) for matching against windows.

    Raises ValueError if a template is not finite or, when ``normalize``, is constant.
    """
    raw = np.asarray(templates, dtype=float)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2:
        raise ValueError(f"templates must be 1-D or 2-D, got shape {raw.shape}")
    _check_int("template length", raw.shape[1], 2 if normalize else 1)
    shape, level, sq_norm, valid = _prepare(raw, normalize, eps)
    if not valid.all():
        reason = "not finite or constant (no z-normalized shape)" if normalize else "not finite"
        raise ValueError(f"template is {reason}")
    count = raw.shape[0]
    return WindowBank(
        raw.shape[1],
        normalize,
        shape,
        level,
        sq_norm,
        np.arange(count, dtype=np.int64),
        np.zeros(count, dtype=np.int64),
        raw.copy(),
    )


def sq_distances(a: WindowBank, b: WindowBank) -> np.ndarray:
    """Squared length-normalized distances between the rows of ``a`` and of ``b`` (``|a| x |b|``)."""
    if a.m != b.m or a.normalize != b.normalize:
        raise ValueError("banks must share the window length and the normalization")
    d2 = a.shape @ b.shape.T
    d2 *= -2.0
    norms = a.sq_norm[:, None] + b.sq_norm[None, :]
    d2 += norms
    norms *= SNAP_REL
    d2[d2 <= norms] = 0.0  # rounding noise of the shape term; also clips negatives
    if not a.normalize:
        d2 += np.square(a.level[:, None] - b.level[None, :])
    return d2


def chunk_rows(n_columns: int) -> int:
    """Rows per distance block so that one block has at most ``CHUNK_ELEMENTS`` entries."""
    return max(1, CHUNK_ELEMENTS // max(1, int(n_columns)))


def block_starts(owner: np.ndarray) -> np.ndarray:
    """Offsets where a run of equal ``owner`` values begins (``owner`` non-decreasing)."""
    if owner.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])


# ------------------------------------------------------------------ matrix profile


def _group_codes(groups: Sequence[Hashable] | None, n_series: int) -> np.ndarray:
    if groups is None:
        return np.arange(n_series, dtype=np.int64)
    labels = list(groups)
    if len(labels) != n_series:
        raise ValueError(f"groups has {len(labels)} entries for {n_series} series")
    codes: dict[Hashable, int] = {}
    return np.array([codes.setdefault(label, len(codes)) for label in labels], dtype=np.int64)


def _cross_join(
    bank: WindowBank,
    series_group: np.ndarray,
    radius_factor: float | None,
    min_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest other-group window of every window (exact, chunked).

    Returns ``(nn_sq, nn_row, support)``: squared distance (inf if none), bank row of the
    neighbour (-1 if none), and, if ``radius_factor`` is given, the number of other-group
    series with a window within ``max(radius_factor * nn, min_radius)``.
    """
    n = len(bank)
    nn_sq = np.full(n, np.inf)
    nn_row = np.full(n, -1, dtype=np.int64)
    support = np.zeros(n, dtype=np.int64)
    if n == 0:
        return nn_sq, nn_row, support
    window_group = series_group[bank.owner]
    starts = block_starts(bank.owner)
    step = chunk_rows(n)
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        d2 = sq_distances(bank.take(slice(r0, r1)), bank)
        d2[window_group[r0:r1, None] == window_group[None, :]] = np.inf
        best = np.argmin(d2, axis=1)
        best_sq = d2[np.arange(r1 - r0), best]
        found = np.isfinite(best_sq)
        nn_sq[r0:r1] = best_sq
        nn_row[r0:r1] = np.where(found, best, -1)
        if radius_factor is not None:
            radius = np.maximum(radius_factor * np.sqrt(best_sq), min_radius) + MATCH_ATOL
            per_series = np.minimum.reduceat(d2, starts, axis=1)
            within = (per_series <= np.square(radius)[:, None]).sum(axis=1)
            support[r0:r1] = np.where(found, within, 0)
    return nn_sq, nn_row, support


def _as_series_list(series_list: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(series, dtype=float).reshape(-1) for series in series_list]


def cross_episode_matrix_profile(
    series_list: Sequence[np.ndarray],
    m: int,
    groups: Sequence[Hashable] | None = None,
    *,
    eps: float = DEFAULT_EPS,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Exact cross-episode (AB-join) matrix profile of a collection of 1-D series.

    ``profile[i][s]`` is the length-normalized z-normalized distance from window ``s`` of
    series ``i`` to its nearest valid window in another series (in another group when
    ``groups`` is given). ``index[i][s]`` is that neighbour's ``(series, start)``. Invalid
    windows (not fully observed, or constant) and windows without an other-group
    neighbour get ``inf`` and ``(-1, -1)``. Ties go to the earliest series, then start.
    """
    m = _check_int("m", m, 2)
    series_list = _as_series_list(series_list)
    bank = build_window_bank(series_list, m, normalize=True, eps=eps)
    nn_sq, nn_row, _ = _cross_join(bank, _group_codes(groups, len(series_list)), None, 0.0)
    profile = [np.full(max(series.size - m + 1, 0), np.inf) for series in series_list]
    index = [np.full((max(series.size - m + 1, 0), 2), -1, dtype=np.int64) for series in series_list]
    for row in np.flatnonzero(nn_row >= 0):
        owner, start, neighbour = int(bank.owner[row]), int(bank.start[row]), int(nn_row[row])
        profile[owner][start] = math.sqrt(nn_sq[row])
        index[owner][start] = (bank.owner[neighbour], bank.start[neighbour])
    return profile, index


# ------------------------------------------------------------------------ discovery


@dataclass(frozen=True)
class MotifCandidate:
    """One motif; ``series_index`` and ``neighbor_series`` index the input series list.

    ``template`` is the z-normalized source window and ``raw_template`` its stored values.
    ``threshold`` is the radius ``r``. ``support`` counts the other-group series with a
    window within ``r``, out of ``n_other_series`` other-group series that have a valid
    window.
    """

    length: int
    template: tuple[float, ...]
    raw_template: tuple[float, ...]
    series_index: int
    start: int
    nn_distance: float
    neighbor_series: int
    neighbor_start: int
    threshold: float
    support: int
    n_other_series: int


@dataclass(frozen=True, eq=False)
class _MotifPool:
    bank: WindowBank
    nn: np.ndarray
    radius: np.ndarray
    support: np.ndarray
    n_other: np.ndarray
    neighbor_series: np.ndarray
    neighbor_start: np.ndarray

    def candidate(self, i: int) -> MotifCandidate:
        template = self.bank.shape[i] * math.sqrt(self.bank.m)
        return MotifCandidate(
            length=self.bank.m,
            template=tuple(float(v) for v in template),
            raw_template=tuple(float(v) for v in self.bank.raw[i]),
            series_index=int(self.bank.owner[i]),
            start=int(self.bank.start[i]),
            nn_distance=float(self.nn[i]),
            neighbor_series=int(self.neighbor_series[i]),
            neighbor_start=int(self.neighbor_start[i]),
            threshold=float(self.radius[i]),
            support=int(self.support[i]),
            n_other_series=int(self.n_other[i]),
        )


def discover_motifs(
    series_list: Sequence[np.ndarray],
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    top_k: int = 3,
    *,
    groups: Sequence[Hashable] | None = None,
    radius_factor: float = 2.0,
    min_radius: float = 0.0,
    eps: float = DEFAULT_EPS,
) -> list[MotifCandidate]:
    """Top-``k`` cross-episode motifs of one feature over all ``lengths`` (module docstring).

    Ranking: nn distance ascending, then support descending, length, series, start. A
    candidate is rejected if its template is within ``max(r_a, r_b)`` of an accepted
    template of the same length.
    """
    top_k = _check_int("top_k", top_k, 0)
    radius_factor = _check_float("radius_factor", radius_factor, positive=True)
    min_radius = _check_float("min_radius", min_radius)
    checked_lengths = _check_lengths(lengths, 2)
    series_list = _as_series_list(series_list)
    codes = _group_codes(groups, len(series_list))
    if top_k == 0 or len(series_list) < 2:
        return []

    pools: list[_MotifPool] = []
    for m in checked_lengths:
        bank = build_window_bank(series_list, m, normalize=True, eps=eps)
        nn_sq, nn_row, support = _cross_join(bank, codes, radius_factor, min_radius)
        rows = np.flatnonzero(nn_row >= 0)
        if rows.size == 0:
            continue
        nn = np.sqrt(nn_sq[rows])
        owners = np.unique(bank.owner)
        per_group = np.bincount(codes[owners], minlength=int(codes.max()) + 1)
        pools.append(
            _MotifPool(
                bank=bank.take(rows),
                nn=nn,
                radius=np.maximum(radius_factor * nn, min_radius),
                support=support[rows],
                n_other=owners.size - per_group[codes[bank.owner[rows]]],
                neighbor_series=bank.owner[nn_row[rows]],
                neighbor_start=bank.start[nn_row[rows]],
            )
        )
    if not pools:
        return []

    pool_id = np.concatenate([np.full(len(pool.nn), p) for p, pool in enumerate(pools)])
    local = np.concatenate([np.arange(len(pool.nn)) for pool in pools])
    order = np.lexsort(
        (
            np.concatenate([pool.bank.start for pool in pools]),
            np.concatenate([pool.bank.owner for pool in pools]),
            np.concatenate([np.full(len(pool.nn), pool.bank.m) for pool in pools]),
            -np.concatenate([pool.support for pool in pools]),
            np.concatenate([pool.nn for pool in pools]),
        )
    )
    alive = [np.ones(len(pool.nn), dtype=bool) for pool in pools]
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
        alive[p] &= distance > np.maximum(pool.radius[i], pool.radius) + MATCH_ATOL
    return [pools[p].candidate(i) for p, i in chosen]


def mine_motif_programs(
    episodes: Mapping[str, EpisodeArrays],
    labels: Mapping[str, int] | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    groups: Mapping[str, Hashable] | None = None,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    top_k_per_feature: int = 3,
    positive_only: bool = True,
    radius_factor: float = 2.0,
    min_radius: float = 0.0,
    missing_policy: str = "strict",
    eps: float = DEFAULT_EPS,
    method: str = "matrix_profile",
) -> list[ProgramRecord]:
    """Motif programs from TRAIN episodes, one :func:`discover_motifs` run per feature.

    ``episodes`` maps episode id -> ``(values, mask, feature_names)`` (mask True = missing).
    ``groups`` maps episode id -> patient id so that a motif's neighbour comes from another
    patient (default: another episode). ``feature_names`` selects the features to mine
    (default: all, in order of first appearance). Episodes whose schema lacks a feature are
    skipped for that feature. The result is sorted by score ``1 / (1 + nn_distance)``,
    then support, feature order, and per-feature rank.
    """
    _check_policy(missing_policy)
    episode_ids = sorted(episodes)
    if positive_only:
        if labels is None:
            raise ValueError("positive_only=True needs labels")
        episode_ids = [eid for eid in episode_ids if _binary_label(labels, eid) == 1]
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
        if len(kept) < 2:
            continue
        group_list = None
        if groups is not None:
            missing_groups = [eid for eid in kept if eid not in groups]
            if missing_groups:
                raise ValueError(f"no group for episodes {missing_groups[:5]}")
            group_list = [groups[eid] for eid in kept]
        candidates = discover_motifs(
            series_list,
            lengths,
            top_k_per_feature,
            groups=group_list,
            radius_factor=radius_factor,
            min_radius=min_radius,
            eps=eps,
        )
        for rank, motif in enumerate(candidates):
            score = 1.0 / (1.0 + motif.nn_distance)
            program = ProgramRecord(
                program_id=f"{method}:{feature}:L{motif.length}:{rank}",
                method=method,
                kind="motif",
                features=frozenset({feature}),
                anchor_signature=frozenset({coarse_group(feature)}),
                score=score,
                params={
                    "feature": feature,
                    "template": motif.template,
                    "threshold": motif.threshold,
                    "length": motif.length,
                    "normalize": True,
                    "nn_distance": motif.nn_distance,
                    "radius_factor": radius_factor,
                    "min_radius": min_radius,
                    "support": motif.support,
                    "support_fraction": motif.support / motif.n_other_series if motif.n_other_series else 0.0,
                    "n_discovery_episodes": len(kept),
                    "raw_template": motif.raw_template,
                    "source_episode": kept[motif.series_index],
                    "source_start": motif.start,
                    "neighbor_episode": kept[motif.neighbor_series],
                    "neighbor_start": motif.neighbor_start,
                    "discovery": "positive_train" if positive_only else "all_train",
                    "missing_policy": missing_policy,
                    "eps": eps,
                    "rank": rank,
                },
            )
            ranked.append(((-score, -motif.support, feature_index, rank), program))
    ranked.sort(key=lambda item: item[0])
    return [program for _, program in ranked]
