"""Established pattern-set selection methods, applied to any candidate pool (baselines for
the v2 selector).

Every function takes the same TRAIN-only inputs as :func:`src.icu.v2.selector.select` -- the
candidate programs (in generator rank order) and their TRAIN firing sets, plus TRAIN labels
where the method is supervised -- and returns at most ``k`` programs in selection order.
None of them sees validation, test or external data; their single hyper-parameter (if any)
is chosen on VALIDATION by the caller (:func:`bridge_validation_score`).

* :func:`mmr` -- Maximal Marginal Relevance (Carbonell & Goldstein, SIGIR 1998):
  ``argmax lam * rel(p) - (1 - lam) * max_{q in S} sim(p, q)`` with rel = the pool-normalized
  generator score (the selector's utility) and sim = paper Definition 3
  (``program_metrics.pairwise_redundancy``), i.e. the redundancy the v2 selector penalizes.
* :func:`mms_greedy` -- redundancy-aware top-k patterns, Maximal Marginal Significance
  (Xin, Cheng, Yan & Han, KDD 2006, Algorithm 1): significance S = pool-normalized generator
  score, redundancy ``R(p, q) = (1 - D(p, q)) * min(S(p), S(q))`` with D the Jaccard distance
  of the TRAIN firing sets (their Eq. 2); first the most significant pattern, then
  ``argmax S(p) - max_{q in P} R(p, q)``.
* :func:`mmrfs` -- MMRFS (Cheng, Yan, Han & Hsu, ICDE 2007, Algorithm 1): relevance S =
  information gain of the firing indicator on TRAIN labels, redundancy as above with that S,
  greedy gain ``S(p) - max R(p, q)``; a pattern is kept only if it correctly covers at least
  one TRAIN episode that is not yet covered ``delta`` times (an episode is correctly covered
  when the pattern fires on it and its label is the pattern's majority TRAIN class), and the
  loop stops when every episode is covered ``delta`` times, no pattern is left, or k are kept.
* :func:`mrmr` -- minimum Redundancy Maximum Relevance, MID scheme (Peng, Long & Ding, TPAMI
  2005): ``argmax I(x; y) - mean_{s in S} I(x; s)`` on the binary TRAIN firing indicators.
* :func:`dpp_greedy` -- greedy MAP inference of a determinantal point process (Chen, Zhang &
  Zhou, NeurIPS 2018): ``L = diag(q) K diag(q)`` with K the Jaccard (Tanimoto) kernel of the
  TRAIN firing sets (positive semi-definite on binary vectors) and ``q = exp(alpha * rel)``;
  each step adds the program that maximizes ``log det L_S``; it stops early when no program
  increases the determinant.
* :func:`l1_logistic` -- embedded feature selection: an L1-penalized logistic regression on
  the TRAIN firing indicators of all candidates; the smallest C on a log grid with at least
  k non-zero coefficients, then the k largest ``|coef|``.

Added for the PVLDB revision (``EXTRA_BASELINES`` and :func:`l1_schema_dropout`; the E1-E3
drivers iterate ``BASELINES`` only):

* :func:`jmi` -- Joint Mutual Information (Yang & Moody, NIPS 1999): first ``argmax I(x; y)``,
  then ``argmax sum_{s in S} I(x, s; y)``.
* :func:`cmim` -- Conditional Mutual Information Maximization (Fleuret, JMLR 2004): first
  ``argmax I(x; y)``, then ``argmax min_{s in S} I(x; y | s)``.
  JMI and CMIM are the best-performing information-theoretic filters in the comparison of
  Brown, Pocock, Zhao & Lujan (JMLR 2012), which derives them as approximations of
  conditional-likelihood maximization. Both skip exact firing duplicates of a kept program
  (JMI would otherwise re-select them: ``I(x, s; y) = I(s; y)`` for ``x = s``).
* :func:`l1_schema_dropout` -- a schema-robust embedded selector: :func:`l1_logistic` trained on
  the TRAIN rows plus ``n_copies`` copies in which every row loses each coarse feature group
  with probability q (the columns of programs that cannot execute are zeroed), i.e. dropout
  training at the schema level (Wager, Wang & Liang, NIPS 2013).

Ties are broken by generator rank (the order of ``candidates``), so every method is
deterministic and independent of dict order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.icu.v2.program_metrics import jaccard, pairwise_redundancy
from src.icu.v2.types import ProgramRecord

FiringSets = Mapping[str, frozenset]


@dataclass(frozen=True)
class SelectionInput:
    """TRAIN-only view of one candidate pool. ``matrix`` (optional) is the binary TRAIN firing
    matrix, rows = sorted ``train_ids`` and columns = ``candidates``; the matrix-based selectors
    (SRS, JMI, CMIM, L1, vectorized mRMR) use it instead of the firing sets, which large or
    synthetic pools may leave empty."""

    candidates: tuple[ProgramRecord, ...]
    train_firings: Mapping[str, frozenset[str]]
    train_ids: frozenset[str]
    train_positives: frozenset[str]
    matrix: Any = field(default=None, compare=False, repr=False)

    def firing(self, program: ProgramRecord) -> frozenset[str]:
        return frozenset(self.train_firings.get(program.program_id, frozenset())) & self.train_ids


def normalized_scores(candidates: Sequence[ProgramRecord]) -> list[float]:
    """Min-max normalized generator scores in [0, 1] (all 1.0 when the scores are equal),
    the same utility the v2 selector uses."""
    scores = [float(p.score) for p in candidates]
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low <= 0:
        return [1.0] * len(scores)
    return [(s - low) / (high - low) for s in scores]


def _entropy(p: float) -> float:
    return 0.0 if p <= 0.0 or p >= 1.0 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def information_gain(fired: frozenset[str], ids: frozenset[str], positives: frozenset[str]) -> float:
    """IG (bits) of the binary indicator "program fires" about the label, on ``ids``."""
    n = len(ids)
    if n == 0:
        return 0.0
    pos = len(positives & ids)
    fired = fired & ids
    n1 = len(fired)
    n0 = n - n1
    pos1 = len(fired & positives)
    pos0 = pos - pos1
    conditional = 0.0
    if n1:
        conditional += n1 / n * _entropy(pos1 / n1)
    if n0:
        conditional += n0 / n * _entropy(pos0 / n0)
    return max(0.0, _entropy(pos / n) - conditional)


def mutual_information(a: frozenset[str], b: frozenset[str], ids: frozenset[str]) -> float:
    """MI (bits) between two binary indicators (membership in ``a`` and in ``b``) on ``ids``."""
    n = len(ids)
    if n == 0:
        return 0.0
    a, b = a & ids, b & ids
    n11 = len(a & b)
    n10 = len(a) - n11
    n01 = len(b) - n11
    n00 = n - n11 - n10 - n01
    pa, pb = len(a) / n, len(b) / n
    total = 0.0
    for count, px, py in ((n11, pa, pb), (n10, pa, 1 - pb), (n01, 1 - pa, pb), (n00, 1 - pa, 1 - pb)):
        if count > 0 and px > 0 and py > 0:
            pxy = count / n
            total += pxy * math.log2(pxy / (px * py))
    return max(0.0, total)


def _argmax(values: Sequence[float], allowed: Sequence[int]) -> int | None:
    """Index in ``allowed`` with the largest value; ties -> earliest (generator rank)."""
    best = None
    for i in allowed:
        if best is None or values[i] > values[best] + 1e-12:
            best = i
    return best


# ------------------------------------------------------------------------------ selectors


def mmr(inp: SelectionInput, k: int, lam: float = 0.7) -> list[ProgramRecord]:
    cands = list(inp.candidates)
    rel = normalized_scores(cands)
    firings = {p.program_id: inp.firing(p) for p in cands}
    chosen: list[int] = []
    max_sim = [0.0] * len(cands)
    remaining = list(range(len(cands)))
    while remaining and len(chosen) < k:
        if not chosen:
            pick = _argmax(rel, remaining)
        else:
            gains = [lam * rel[i] - (1.0 - lam) * max_sim[i] for i in range(len(cands))]
            pick = _argmax(gains, remaining)
        chosen.append(pick)
        remaining.remove(pick)
        for i in remaining:
            sim = pairwise_redundancy(cands[i], cands[pick], firings)["value"]
            max_sim[i] = max(max_sim[i], sim)
    return [cands[i] for i in chosen]


def _significance_redundancy_greedy(
    cands: Sequence[ProgramRecord],
    significance: Sequence[float],
    firing: Sequence[frozenset[str]],
    k: int,
    eligible: Callable[[int], bool] | None = None,
    on_select: Callable[[int], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> list[int]:
    """Greedy of Xin et al. (KDD'06, Alg. 1) with R(p, q) = J(p, q) * min(S(p), S(q));
    the hooks add the MMRFS coverage rule."""
    chosen: list[int] = []
    max_red = [0.0] * len(cands)
    remaining = list(range(len(cands)))
    while remaining and len(chosen) < k:
        if stop is not None and stop():
            break
        gains = [significance[i] - max_red[i] for i in range(len(cands))]
        order = sorted(remaining, key=lambda i: (-round(gains[i], 12), i))
        pick = None
        for i in order:
            if eligible is None or eligible(i):
                pick = i
                break
            remaining.remove(i)  # MMRFS: a pattern that covers nothing new is discarded
        if pick is None:
            break
        chosen.append(pick)
        remaining.remove(pick)
        if on_select is not None:
            on_select(pick)
        for i in remaining:
            red = jaccard(firing[i], firing[pick]) * min(significance[i], significance[pick])
            max_red[i] = max(max_red[i], red)
    return chosen


def mms_greedy(inp: SelectionInput, k: int) -> list[ProgramRecord]:
    cands = list(inp.candidates)
    firing = [inp.firing(p) for p in cands]
    return [cands[i] for i in _significance_redundancy_greedy(cands, normalized_scores(cands), firing, k)]


def mmrfs(inp: SelectionInput, k: int, delta: float = 1) -> list[ProgramRecord]:
    cands = list(inp.candidates)
    firing = [inp.firing(p) for p in cands]
    significance = [information_gain(f, inp.train_ids, inp.train_positives) for f in firing]
    prevalence = len(inp.train_positives & inp.train_ids) / max(len(inp.train_ids), 1)
    # majority TRAIN class of each pattern, relative to the base rate (precision >= prevalence -> positive)
    pattern_class = [1 if (len(f & inp.train_positives) / len(f) if f else 0.0) >= prevalence else 0 for f in firing]
    covered: dict[str, int] = {e: 0 for e in inp.train_ids}

    def correctly_covered(i: int) -> frozenset[str]:
        if pattern_class[i] == 1:
            return firing[i] & inp.train_positives
        return firing[i] - inp.train_positives

    def eligible(i: int) -> bool:
        return any(covered[e] < delta for e in correctly_covered(i))

    def on_select(i: int) -> None:
        for e in correctly_covered(i):
            covered[e] += 1

    def stop() -> bool:
        return all(c >= delta for c in covered.values())

    chosen = _significance_redundancy_greedy(cands, significance, firing, k, eligible, on_select, stop)
    return [cands[i] for i in chosen]


def mrmr(inp: SelectionInput, k: int) -> list[ProgramRecord]:
    cands = list(inp.candidates)
    firing = [inp.firing(p) for p in cands]
    relevance = [mutual_information(f, inp.train_positives, inp.train_ids) for f in firing]
    chosen: list[int] = []
    red_sum = [0.0] * len(cands)
    remaining = list(range(len(cands)))
    while remaining and len(chosen) < k:
        if not chosen:
            pick = _argmax(relevance, remaining)
        else:
            scores = [relevance[i] - red_sum[i] / len(chosen) for i in range(len(cands))]
            pick = _argmax(scores, remaining)
        chosen.append(pick)
        remaining.remove(pick)
        for i in remaining:
            red_sum[i] += mutual_information(firing[i], firing[pick], inp.train_ids)
    return [cands[i] for i in chosen]


def dpp_greedy(inp: SelectionInput, k: int, alpha: float = 1.0, eps: float = 1e-10) -> list[ProgramRecord]:
    """Fast greedy MAP (Chen et al. 2018, Alg. 1) with incremental Cholesky updates."""
    cands = list(inp.candidates)
    n = len(cands)
    if n == 0:
        return []
    rel = np.array(normalized_scores(cands))
    quality = np.exp(alpha * rel)
    firing = [inp.firing(p) for p in cands]
    kernel = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            kernel[i, j] = kernel[j, i] = jaccard(firing[i], firing[j])
    L = quality[:, None] * kernel * quality[None, :]
    cis = np.zeros((k, n))
    di2 = np.diag(L).copy()
    chosen: list[int] = []
    remaining = np.ones(n, dtype=bool)
    while len(chosen) < k:
        candidates_left = np.flatnonzero(remaining)
        if candidates_left.size == 0:
            break
        # largest marginal gain d_i^2; ties -> earliest rank
        best = candidates_left[np.argmax(di2[candidates_left])]
        if di2[best] < eps:
            break
        j = len(chosen)
        chosen.append(int(best))
        remaining[best] = False
        if len(chosen) == k:
            break
        ej = (L[best, :] - cis[:j, best] @ cis[:j, :]) / math.sqrt(di2[best])
        cis[j, :] = ej
        di2 = di2 - ej ** 2
        di2[~remaining] = -np.inf
    return [cands[i] for i in chosen]


def l1_logistic(inp: SelectionInput, k: int, c_grid: Sequence[float] | None = None) -> list[ProgramRecord]:
    from sklearn.linear_model import LogisticRegression

    if not inp.candidates:
        return []
    cands, X, y = train_matrix(inp)
    y = y.astype(int)
    grid = list(c_grid) if c_grid is not None else list(np.logspace(-4, 2, 31))
    coef = None
    for c in grid:
        # pure L1 (scikit-learn >= 1.8 spells it l1_ratio=1; ``penalty`` is deprecated there)
        model = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=float(c), max_iter=2000, random_state=0)
        model.fit(X, y)
        coef = model.coef_.ravel()
        if np.count_nonzero(coef) >= k:
            break
    order = sorted((j for j in range(len(cands)) if coef[j] != 0.0), key=lambda j: (-abs(coef[j]), j))
    return [cands[j] for j in order[:k]]


def train_matrix(inp: SelectionInput) -> tuple[list[ProgramRecord], np.ndarray, np.ndarray]:
    """Candidates, the binary TRAIN firing matrix (episodes in sorted id order x candidates)
    and the TRAIN labels."""
    cands = list(inp.candidates)
    ids = sorted(inp.train_ids)
    y = np.array([1.0 if e in inp.train_positives else 0.0 for e in ids])
    if inp.matrix is not None:
        return cands, np.asarray(inp.matrix, dtype=float), y
    row = {e: i for i, e in enumerate(ids)}
    X = np.zeros((len(ids), len(cands)))
    for j, p in enumerate(cands):
        for e in inp.firing(p):
            X[row[e], j] = 1.0
    return cands, X, y


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return np.nan_to_num(h, nan=0.0)


def _conditional_entropy(cells: Sequence[tuple[np.ndarray, np.ndarray]], n: int) -> np.ndarray:
    """H(y | cell) in bits for a partition given as (cell size, positives in cell) pairs."""
    total = 0.0
    for size, positives in cells:
        size = np.asarray(size, dtype=float)
        rate = np.divide(positives, size, out=np.zeros_like(size), where=size > 0)
        total = total + size / n * _binary_entropy(rate)
    return np.asarray(total)


def _information_greedy(inp: SelectionInput, k: int, criterion: str) -> list[ProgramRecord]:
    cands, X, y = train_matrix(inp)
    n, m = X.shape
    if m == 0 or k < 1 or n == 0:
        return []
    positives = float(y.sum())
    fires = X.sum(axis=0)
    fired_pos = X.T @ y
    h_y = float(_binary_entropy(np.array(positives / n)))
    relevance = h_y - _conditional_entropy([(fires, fired_pos), (n - fires, positives - fired_pos)], n)
    score = np.zeros(m) if criterion == "jmi" else np.full(m, np.inf)
    packed = np.packbits(X.astype(bool), axis=0)
    signature = [packed[:, j].tobytes() for j in range(m)]
    available = np.ones(m, dtype=bool)
    chosen: list[int] = []
    while len(chosen) < k:
        left = np.flatnonzero(available)
        if left.size == 0:
            break
        values = (relevance if not chosen else score)[left]
        best = int(left[np.flatnonzero(values >= values.max() - 1e-12)[0]])  # ties -> generator rank
        chosen.append(best)
        available[[j for j in left if signature[j] == signature[best]]] = False  # best and its exact duplicates
        xs = X[:, best]
        both = X.T @ xs
        both_pos = X.T @ (xs * y)
        s_fires, s_pos = fires[best], fired_pos[best]
        joint = _conditional_entropy([
            (both, both_pos),
            (fires - both, fired_pos - both_pos),
            (s_fires - both, s_pos - both_pos),
            (n - fires - s_fires + both, positives - fired_pos - s_pos + both_pos),
        ], n)
        if criterion == "jmi":
            score += h_y - joint  # I(x, s; y)
        else:
            h_given_s = float(_conditional_entropy([(np.array(s_fires), np.array(s_pos)), (np.array(n - s_fires), np.array(positives - s_pos))], n))
            score = np.minimum(score, h_given_s - joint)  # I(x; y | s)
    return [cands[j] for j in chosen]


def _argmax_scan(values: np.ndarray, allowed: np.ndarray) -> int:
    """:func:`_argmax` on arrays: scanning ``allowed`` in rank order, the best index changes only
    when a later value exceeds the current best by more than 1e-12."""
    position, best = 0, int(allowed[0])
    while True:
        bigger = np.flatnonzero(values[allowed[position + 1:]] > values[best] + 1e-12)
        if bigger.size == 0:
            return best
        position += 1 + int(bigger[0])
        best = int(allowed[position])


def _mutual_information_counts(a_count: np.ndarray, b_count: float, both: np.ndarray, n: int) -> np.ndarray:
    """:func:`mutual_information` from counts, vectorized over the first indicator."""
    pa, pb = a_count / n, b_count / n
    total = np.zeros_like(pa, dtype=float)
    for count, px, py in ((both, pa, pb), (a_count - both, pa, 1 - pb), (b_count - both, 1 - pa, pb), (n - a_count - b_count + both, 1 - pa, 1 - pb)):
        ok = (count > 0) & (px > 0) & (py > 0)
        pxy = np.where(ok, count / n, 1.0)
        total = total + np.where(ok, pxy * np.log2(pxy / np.where(ok, px * py, 1.0)), 0.0)
    return np.maximum(total, 0.0)


def mrmr_vectorized(inp: SelectionInput, k: int) -> list[ProgramRecord]:
    """:func:`mrmr` on the firing matrix (same criterion and tie rule; used where the firing sets
    are too large, e.g. the runtime benchmark)."""
    cands, X, y = train_matrix(inp)
    n, m = X.shape
    if m == 0 or k < 1 or n == 0:
        return []
    fires = X.sum(axis=0)
    relevance = _mutual_information_counts(fires, float(y.sum()), X.T @ y, n)
    red_sum = np.zeros(m)
    available = np.ones(m, dtype=bool)
    chosen: list[int] = []
    while available.any() and len(chosen) < k:
        values = relevance if not chosen else relevance - red_sum / len(chosen)
        pick = _argmax_scan(values, np.flatnonzero(available))
        chosen.append(pick)
        available[pick] = False
        red_sum += _mutual_information_counts(fires, float(fires[pick]), X.T @ X[:, pick], n)
    return [cands[j] for j in chosen]


def jmi(inp: SelectionInput, k: int) -> list[ProgramRecord]:
    return _information_greedy(inp, k, "jmi")


def nmifs(inp: SelectionInput, k: int) -> list[ProgramRecord]:
    """NMIFS (Estevez et al., IEEE TNN 2009): mRMR with the redundancy normalized by entropy,
    I(x; y) - mean_s I(x; s) / min(H(x), H(s)), so that programs with very different firing rates
    are compared on one scale. Same tie rule as :func:`mrmr_vectorized`."""
    cands, X, y = train_matrix(inp)
    n, m = X.shape
    if m == 0 or k < 1 or n == 0:
        return []
    fires = X.sum(axis=0)
    relevance = _mutual_information_counts(fires, float(y.sum()), X.T @ y, n)
    entropy = _binary_entropy(fires / n)
    red_sum = np.zeros(m)
    available = np.ones(m, dtype=bool)
    chosen: list[int] = []
    while available.any() and len(chosen) < k:
        values = relevance if not chosen else relevance - red_sum / len(chosen)
        pick = _argmax_scan(values, np.flatnonzero(available))
        chosen.append(pick)
        available[pick] = False
        mi = _mutual_information_counts(fires, float(fires[pick]), X.T @ X[:, pick], n)
        scale = np.minimum(entropy, entropy[pick])
        red_sum += np.divide(mi, scale, out=np.zeros_like(mi), where=scale > 0)
    return [cands[j] for j in chosen]


def mrmr_stratified(inp: SelectionInput, k: int) -> list[ProgramRecord]:
    """Generator-stratified mRMR: mRMR within each generator's candidates, then round-robin over
    generators (ordered by the relevance of their first pick), so that every generator's language
    is represented and redundancy is only judged within one language."""
    by_method: dict[str, list[ProgramRecord]] = {}
    for p in inp.candidates:
        by_method.setdefault(p.method, []).append(p)
    orders = []
    for method, cands in by_method.items():
        sub = SelectionInput(tuple(cands), inp.train_firings, inp.train_ids, inp.train_positives)
        order = mrmr_vectorized(sub, k)
        if order:
            first = mutual_information(inp.firing(order[0]), inp.train_positives, inp.train_ids)
            orders.append((-first, method, order))
    orders.sort()
    chosen: list[ProgramRecord] = []
    depth = 0
    while len(chosen) < k and any(depth < len(o) for _, _, o in orders):
        for _, _, order in orders:
            if depth < len(order) and len(chosen) < k:
                chosen.append(order[depth])
        depth += 1
    return chosen


def with_relevance(inp: SelectionInput, relevance: str) -> SelectionInput:
    """``inp`` with every candidate's generator score replaced, for the score-based selectors (MMR,
    MMS, DPP), whose generator scores are not on one scale across generators: ``rank`` = percentile
    rank of the score within the candidate's generator; ``ig`` = TRAIN information gain."""
    from dataclasses import replace

    cands = list(inp.candidates)
    if relevance == "ig":
        values = [information_gain(inp.firing(p), inp.train_ids, inp.train_positives) for p in cands]
    elif relevance == "rank":
        values = [0.0] * len(cands)
        groups: dict[str, list[int]] = {}
        for i, p in enumerate(cands):
            groups.setdefault(p.method, []).append(i)
        for idx in groups.values():
            scores = np.array([float(cands[i].score) for i in idx])
            order = scores.argsort(kind="stable")
            ranks = np.empty(len(idx))
            ranks[order] = np.arange(len(idx))
            for value in np.unique(scores):  # ties share their mean rank
                same = scores == value
                ranks[same] = ranks[same].mean()
            for i, r in zip(idx, ranks):
                values[i] = float(r / max(len(idx) - 1, 1))
    else:
        raise ValueError(f"unknown relevance {relevance!r}")
    rescored = tuple(replace(p, score=v) for p, v in zip(cands, values))
    firings = {r.program_id: inp.train_firings.get(p.program_id, frozenset()) for p, r in zip(cands, rescored)}
    return SelectionInput(rescored, firings, inp.train_ids, inp.train_positives, inp.matrix)


def cmim(inp: SelectionInput, k: int) -> list[ProgramRecord]:
    return _information_greedy(inp, k, "cmim")


def l1_schema_dropout(inp: SelectionInput, k: int, q: float = 0.25, source_features: frozenset[str] = frozenset(),
                      n_copies: int = 4, seed: int = 0, c_grid: Sequence[float] | None = None) -> list[ProgramRecord]:
    from sklearn.linear_model import LogisticRegression

    from src.icu.v2.types import coarse_group

    cands, X, y = train_matrix(inp)
    if not cands:
        return []
    groups = sorted({coarse_group(f) for f in source_features} | {coarse_group(f) for p in cands for f in p.features})
    gid = {g: i for i, g in enumerate(groups)}
    need = np.zeros((len(cands), len(groups)), dtype=np.float32)
    for j, p in enumerate(cands):
        for f in p.features:
            need[j, gid[coarse_group(f)]] = 1.0
    rng = np.random.default_rng(seed)
    blocks = [X]
    for _ in range(n_copies):
        keep = rng.random((X.shape[0], len(groups))) >= q
        empty = np.flatnonzero(~keep.any(axis=1))
        keep[empty, rng.integers(len(groups), size=empty.size)] = True  # never drop every group
        executable = ((~keep).astype(np.float32) @ need.T) == 0
        blocks.append(X * executable)
    Xa, ya = np.vstack(blocks), np.tile(y, n_copies + 1)
    grid = list(c_grid) if c_grid is not None else list(np.logspace(-4, 2, 31))
    coef = None
    for c in grid:
        model = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=float(c), max_iter=2000, random_state=0)
        model.fit(Xa, ya)
        coef = model.coef_.ravel()
        if np.count_nonzero(coef) >= k:
            break
    order = sorted((j for j in range(len(cands)) if coef[j] != 0.0), key=lambda j: (-abs(coef[j]), j))
    return [cands[j] for j in order[:k]]


#: name -> (function, hyper-parameter name or None, validation grid)
BASELINES: dict[str, tuple[Callable[..., list[ProgramRecord]], str | None, tuple[Any, ...]]] = {
    "mmr": (mmr, "lam", (0.3, 0.5, 0.7, 0.9)),
    "mms_xin": (mms_greedy, None, (None,)),
    "mmrfs": (mmrfs, "delta", (1, 3, 10, math.inf)),
    "mrmr": (mrmr, None, (None,)),
    "dpp": (dpp_greedy, "alpha", (0.5, 1.0, 2.0, 4.0)),
    "l1_logistic": (l1_logistic, None, (None,)),
}

#: the information-theoretic filters added for the revision (same interface as ``BASELINES``)
EXTRA_BASELINES: dict[str, tuple[Callable[..., list[ProgramRecord]], str | None, tuple[Any, ...]]] = {
    "jmi": (jmi, None, (None,)),
    "cmim": (cmim, None, (None,)),
    "mrmr_vectorized": (mrmr_vectorized, None, (None,)),
    "nmifs": (nmifs, None, (None,)),
}


def run_baseline(name: str, inp: SelectionInput, k: int, value: Any = None) -> list[ProgramRecord]:
    function, param, _ = BASELINES[name] if name in BASELINES else EXTRA_BASELINES[name]
    return function(inp, k) if param is None else function(inp, k, **{param: value})
