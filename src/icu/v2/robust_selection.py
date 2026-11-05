"""Schema-robust selection (SRS): choose K programs for deployment to sources with other schemas.

Deployment model (the one the v2 evaluation uses for every method): a registry R of K programs
is fixed on the source; one probe h (L2 logistic regression on the K binary firing
indicators) is fit on source TRAIN; at a target with available features F, every program
whose required features sigma_f are not all in F cannot execute and its coordinate is 0; h is
not refit. SRS chooses R to maximize the expected value of that deployment,

    J(R) = sum_s pi_s * U_s(R),

over schema scenarios s with available features F_s and weights pi_s (the source schema
itself, the known target schemas, and optionally schemas drawn by dropping whole coarse
mechanism groups at random). U_s(R) is estimated on TRAIN only, by patient-grouped
cross-validation: fit h on the training folds with all K coordinates, score the held-out fold
with the coordinates that cannot execute under F_s set to 0, and take the mean log-likelihood
gain over the fold's base rate (bits per episode). Relevance, redundancy and executability
therefore enter through one quantity, the expected predictive value of the deployed probe,
instead of through separately weighted terms.

Greedy forward selection: start empty; at each step add the screened candidate with the
largest J(R + p); stop at K programs or when the best candidate raises J by less than
``min_gain`` bits per episode (a parsimony threshold: under L2 a near-copy of a selected
program can raise the cross-validated likelihood by a vanishing amount; exact copies, same
TRAIN firing set, are skipped outright). By default every candidate is evaluated at every
step (``screen=None``); a screen of ``m`` keeps the union of the top ``m // 2`` candidates by
TRAIN information gain and the top ``m // 2`` by information gain times expected
executability (sum of pi_s over the scenarios where the program can execute). On the
MIMIC-IV pools a screen of 60 lost up to 0.06 internal AUPRC on the 200-candidate SPMF pools,
because low-ranked but complementary programs were cut; the full search takes 10-20 s per
200-candidate pool. Ties are broken by generator rank. Validation, test and target data are
never used.

Fast SRS (``shortlist=m``): at every step all candidates are scored at once by one Newton step
from the current per-fold fits (the second-order, score-test approximation of adding a
coordinate, computed for every candidate with a Schur complement), and only the ``m`` best by
that approximation are re-fit exactly (warm-started) and compared on the exact objective. The
result equals exact greedy whenever the exact best candidate is in the shortlist; the cost per
step drops from |C| full fits per fold to one O(n |C| |R|) matrix pass plus m warm-started fits.

SRS-C (``surrogates=True``): every candidate is a composite program, i.e. the candidate plus
its CART surrogates chosen on TRAIN firings (``src/icu/v2/surrogates.py``). Under a scenario
the composite's coordinate takes the firing of its first executable branch instead of 0, so
the objective values a primary by how well it (or its surrogates) serves every schema. The
probe is still fit on the primaries' source firings. Only ``mode="frozen"`` is supported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from src.icu.v2.selection_baselines import SelectionInput, information_gain
from src.icu.v2.selection_baselines import _binary_entropy, _mutual_information_counts
from src.icu.v2.surrogates import MAX_SURROGATES, branch_index, surrogate_chains
from src.icu.v2.types import ProgramRecord, coarse_group

_EPS = 1e-12


@dataclass(frozen=True)
class SchemaScenario:
    name: str
    features: frozenset[str]
    weight: float


@dataclass
class SRSResult:
    registry: list[ProgramRecord]
    objective: list[float] = field(default_factory=list)  # J after each added program
    screened: int = 0
    stop_reason: str = ""
    #: SRS-C only: primary program id -> its surrogate programs (deployment order)
    surrogates: dict[str, tuple[ProgramRecord, ...]] = field(default_factory=dict)


def scenario_set(
    source_features: frozenset[str],
    known_targets: Mapping[str, frozenset[str]],
    dropout_q: float = 0.0,
    n_dropout: int = 20,
    seed: int = 0,
) -> list[SchemaScenario]:
    """Source schema, known target schemas, and (dropout_q > 0) random group-dropout schemas.

    Weights: the source, the known targets and the random schemas each get an equal share of
    the total (the known targets and the random schemas split their share evenly).
    """
    blocks: list[list[SchemaScenario]] = [[SchemaScenario("source", frozenset(source_features), 1.0)]]
    if known_targets:
        blocks.append([SchemaScenario(name, frozenset(features), 1.0) for name, features in sorted(known_targets.items())])
    if dropout_q > 0:
        rng = np.random.default_rng(seed)
        groups = sorted({coarse_group(f) for f in source_features})
        sampled = []
        for i in range(n_dropout):
            keep = [g for g in groups if rng.random() >= dropout_q]
            if not keep:  # never drop everything
                keep = [groups[int(rng.integers(len(groups)))]]
            features = frozenset(f for f in source_features if coarse_group(f) in keep)
            sampled.append(SchemaScenario(f"dropout_{i}", features, 1.0))
        blocks.append(sampled)
    scenarios = []
    for block in blocks:
        for s in block:
            scenarios.append(SchemaScenario(s.name, s.features, 1.0 / (len(blocks) * len(block))))
    return scenarios


def grouped_folds(ids: Sequence[str], subject_of: Mapping[str, object], n_folds: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Patient-grouped K folds over the rows of ``ids`` (deterministic in ``seed``)."""
    subjects = sorted({str(subject_of[e]) for e in ids})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subjects))
    fold_of = {subjects[j]: i % n_folds for i, j in enumerate(order)}
    assignment = np.array([fold_of[str(subject_of[e])] for e in ids])
    return [(np.flatnonzero(assignment != f), np.flatnonzero(assignment == f)) for f in range(n_folds)]


def _fit_logistic(X: np.ndarray, y: np.ndarray, C: float, iterations: int = 25, init: np.ndarray | None = None) -> np.ndarray:
    """L2 logistic regression by Newton's method (intercept unpenalized); returns [b0, w...].
    ``init`` warm-starts the iterations (the optimum is unique, so only the speed changes)."""
    n, d = X.shape
    Z = np.hstack([np.ones((n, 1)), X])
    if init is not None and len(init) == d + 1:
        beta = np.array(init, dtype=float)
    else:
        beta = np.zeros(d + 1)
        prior = min(max(y.mean(), 1e-6), 1 - 1e-6)
        beta[0] = math.log(prior / (1 - prior))
    penalty = np.full(d + 1, 1.0 / C)
    penalty[0] = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(Z @ beta)))
        gradient = Z.T @ (y - p) - penalty * beta
        hessian = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(penalty) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


class _Objective:
    """Cross-validated expected log-likelihood gain of the frozen, schema-masked probe.

    ``mode="frozen"`` (deployment-faithful): one probe per fold on all selected programs, scored
    with the non-executable coordinates zeroed. ``mode="projected"``: per scenario, the probe is
    refit on the executable programs only (a source-side, per-schema model); with
    ``cv=False`` (fit and score on all TRAIN rows) this is the penalized in-sample log-likelihood
    for which greedy forward selection is (1 - e^-gamma)-optimal under restricted strong
    convexity (Elenberg et al., Ann. Statist. 2018), the L2 penalty giving the convexity.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, folds, executable: np.ndarray, weights: np.ndarray, C: float, mode: str = "frozen",
                 branch: np.ndarray | None = None):
        if mode not in ("frozen", "projected"):
            raise ValueError(f"unknown mode {mode!r}")
        if branch is not None and mode != "frozen":
            raise ValueError("surrogate branches are only defined for the frozen objective")
        self.X, self.y, self.folds, self.executable, self.weights, self.C, self.mode = X, y, folds, executable, weights, C, mode
        # SRS-C: (scenarios x candidates) branch column, -1 -> index X.shape[1], read as all zeros
        self.branch = None
        if branch is not None:
            self.branch = np.where(branch < 0, X.shape[1], branch)
        self.base = []
        for train, test in folds:
            prior = min(max(y[train].mean(), 1e-6), 1 - 1e-6)
            yt = y[test]
            self.base.append(float(np.mean(yt * math.log2(prior) + (1 - yt) * math.log2(1 - prior))))

    def columns(self, rows: np.ndarray, index) -> np.ndarray:
        """``X[rows, index]`` where the index ``X.shape[1]`` is an all-zero column."""
        index = np.asarray(index, dtype=np.int64)
        valid = index < self.X.shape[1]
        out = self.X[np.ix_(rows, np.where(valid, index, 0))]
        if not valid.all():
            out[:, ~valid] = 0.0
        return out

    @staticmethod
    def _loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        logits = beta[0] + X @ beta[1:]
        p = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-9, 1 - 1e-9)
        return float(np.mean(y * np.log2(p) + (1 - y) * np.log2(1 - p)))

    def __call__(self, cols: Sequence[int]) -> float:
        if not cols:
            return 0.0
        cols = list(cols)
        total = 0.0
        for (train, test), base in zip(self.folds, self.base):
            yt = self.y[test]
            if self.mode == "frozen":
                beta = _fit_logistic(self.X[np.ix_(train, cols)], self.y[train], self.C)
                if self.branch is not None:
                    for branch, weight in zip(self.branch[:, cols], self.weights):
                        total += weight * (self._loglik(beta, self.columns(test, branch), yt) - base)
                    continue
                Xt = self.X[np.ix_(test, cols)]
                for mask, weight in zip(self.executable[:, cols], self.weights):
                    total += weight * (self._loglik(beta, Xt * mask, yt) - base)
            else:
                cache: dict[tuple[int, ...], float] = {}
                for mask, weight in zip(self.executable[:, cols], self.weights):
                    keep = tuple(c for c, m in zip(cols, mask) if m > 0)
                    if keep not in cache:
                        if keep:
                            beta = _fit_logistic(self.X[np.ix_(train, list(keep))], self.y[train], self.C)
                            cache[keep] = self._loglik(beta, self.X[np.ix_(test, list(keep))], yt) - base
                        else:
                            cache[keep] = 0.0
                    total += weight * cache[keep]
        return total / len(self.folds)


def _log2_sigmoid_ll(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean log2-likelihood per column of ``logits`` (rows = episodes)."""
    p = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-9, 1 - 1e-9)
    return (y[:, None] * np.log2(p) + (1 - y[:, None]) * np.log2(1 - p)).mean(axis=0)


class _FastFrozen:
    """Frozen-mode objective with per-fold warm state: one-step Newton screening of every
    candidate, exact warm-started evaluation of a shortlist."""

    def __init__(self, objective: _Objective):
        self.o = objective
        self.betas = []
        for train, _ in objective.folds:  # intercept-only start
            self.betas.append(_fit_logistic(np.zeros((len(train), 0)), objective.y[train], objective.C))

    def _scenario_columns(self, rows: np.ndarray, cols) -> list[np.ndarray]:
        """Per scenario, the (rows x len(cols)) coordinates seen under that scenario."""
        o = self.o
        cols = np.asarray(cols, dtype=np.int64)
        if o.branch is not None:
            return [o.columns(rows, o.branch[s, cols]) for s in range(len(o.weights))]
        base = o.X[np.ix_(rows, cols)]
        return [base * o.executable[s, cols] for s in range(len(o.weights))]

    def screen(self, selected: list[int], candidates: list[int], chunk: int = 8192) -> np.ndarray:
        """One-step Newton value of every candidate. Candidates are processed in blocks of
        ``chunk`` and the scenario columns one scenario at a time, so memory is O(n x chunk)
        whatever the pool size; every candidate's value is computed as with a single block."""
        o = self.o
        candidates = np.asarray(candidates, dtype=np.int64)
        total = np.zeros(len(candidates))
        penalty = 1.0 / o.C
        for f, ((train, test), base) in enumerate(zip(o.folds, o.base)):
            beta = self.betas[f]
            Zs = np.hstack([np.ones((len(train), 1)), o.X[np.ix_(train, selected)]])
            p = 1.0 / (1.0 + np.exp(-(Zs @ beta)))
            w = p * (1 - p)
            r = o.y[train] - p
            pen = np.full(Zs.shape[1], penalty)
            pen[0] = 0.0
            g_s = Zs.T @ r - pen * beta
            H_s = Zs.T @ (Zs * w[:, None]) + np.diag(pen) + 1e-9 * np.eye(Zs.shape[1])
            Hinv_gs = np.linalg.solve(H_s, g_s)
            yt = o.y[test]
            Zts = [np.hstack([np.ones((len(test), 1)), cols]) for cols in self._scenario_columns(test, selected)]
            for first in range(0, len(candidates), chunk):
                block = candidates[first:first + chunk]
                Xc = o.X[np.ix_(train, block)]
                g_c = Xc.T @ r
                H_sc = Zs.T @ (Xc * w[:, None])
                h_cc = w @ Xc + penalty  # binary coordinates: x^2 = x
                Hinv_sc = np.linalg.solve(H_s, H_sc)
                schur = np.maximum(h_cc - np.einsum("ij,ij->j", H_sc, Hinv_sc), 1e-12)
                d_c = (g_c - H_sc.T @ Hinv_gs) / schur
                d_s = Hinv_gs[:, None] - Hinv_sc * d_c[None, :]
                plain = o.X[np.ix_(test, block)] if o.branch is None else None
                for s, weight in enumerate(o.weights):
                    Zt = Zts[s]
                    cand = o.columns(test, o.branch[s, block]) if plain is None else plain * o.executable[s, block]
                    logits = (Zt @ beta)[:, None] + Zt @ d_s + cand * d_c[None, :]
                    total[first:first + len(block)] += weight * (_log2_sigmoid_ll(logits, yt) - base)
        return total / len(o.folds)

    def exact(self, selected: list[int], j: int) -> tuple[float, list[np.ndarray]]:
        o = self.o
        cols = selected + [j]
        total, betas = 0.0, []
        for f, ((train, test), base) in enumerate(zip(o.folds, o.base)):
            beta = _fit_logistic(o.X[np.ix_(train, cols)], o.y[train], o.C, init=np.append(self.betas[f], 0.0))
            betas.append(beta)
            yt = o.y[test]
            for Xs, weight in zip(self._scenario_columns(test, cols), o.weights):
                total += weight * (o._loglik(beta, Xs, yt) - base)
        return total / len(o.folds), betas

    def heldout_auprc(self, cols: list[int], betas: list[np.ndarray]) -> float:
        """Scenario-weighted held-out-fold AUPRC of the fold probes ``betas`` on ``cols`` (SRS-AP).

        The score is computed once per distinct coordinate row, so episodes with the same coordinates
        tie exactly. Scoring every row separately lets floating-point noise in the matrix product
        (which depends, e.g., on the BLAS thread count) break those ties and move the AUPRC by up to
        about 1e-4, more than the gap between near-identical candidates."""
        from sklearn.metrics import average_precision_score

        o = self.o
        total = 0.0
        for f, (_, test) in enumerate(o.folds):
            yt = o.y[test]
            for Xs, weight in zip(self._scenario_columns(test, cols), o.weights):
                rows, inverse = np.unique(Xs, axis=0, return_inverse=True)
                score = (betas[f][0] + rows @ betas[f][1:])[inverse.reshape(-1)]
                total += weight * float(average_precision_score(yt, score))
        return total / len(o.folds)


def schema_robust_select(
    inp: SelectionInput,
    k: int,
    scenarios: Sequence[SchemaScenario],
    subject_of: Mapping[str, object],
    n_folds: int = 3,
    screen: int | None = None,
    C: float = 1.0,
    seed: int = 0,
    min_gain: float = 1e-4,
    mode: str = "frozen",
    cv: bool = True,
    surrogates: bool = False,
    max_surrogates: int = MAX_SURROGATES,
    shortlist: int | None = None,
    n_repeats: int = 1,
    chains: Sequence[Sequence[int]] | None = None,
    decision: str = "loglik",
    screener: str = "newton",
    trace: list | None = None,
) -> SRSResult:
    """Greedy SRS (see the module docstring). ``cv=False`` fits and scores every probe on all
    TRAIN rows (in-sample objective, used for the analyzable ``mode="projected"`` variant).
    ``surrogates=True`` is SRS-C: composite candidates, surrogates returned in the result.
    ``shortlist=m`` is fast SRS (frozen mode only): one-step Newton screening, exact re-check of m.
    ``n_repeats=r`` averages the objective over r patient-grouped fold splits (seeds seed..seed+r-1),
    which lowers the variance of each step's gain estimate on large candidate pools.
    ``chains`` reuses surrogate chains computed by :func:`surrogate_chains` for the same candidates
    and TRAIN rows (SRS-C-V runs SRS-C twice on one pool). With ``inp.matrix`` the TRAIN firing
    matrix is taken as given and exact duplicates are found from its columns.
    ``decision="auprc"`` is SRS-AP (fast SRS only): screening, the shortlist and the stop rule are
    unchanged, but of the shortlisted candidates the one with the best scenario-weighted
    held-out-fold AUPRC is added; selection stops when no shortlisted candidate raises the
    cross-validated log-likelihood by ``min_gain`` (the stop rule of SRS).
    ``screener="nmifs"`` (fast SRS only) replaces the one-step Newton screening by NMIFS's conditional
    score (relevance minus mean entropy-normalized redundancy to the selected set): the hybrid
    baseline NMIFS+AP, which isolates the screening from the decision.
    ``trace`` (fast SRS only, diagnostics): a list that receives one dict per step with the shortlist's
    exact likelihood values and decision keys; it does not change the selection."""
    if decision not in ("loglik", "auprc"):
        raise ValueError(f"unknown decision {decision!r}")
    if decision == "auprc" and shortlist is None:
        raise ValueError("decision='auprc' is defined for fast SRS (shortlist) only")
    if screener not in ("newton", "nmifs") or (screener == "nmifs" and shortlist is None):
        raise ValueError(f"screener {screener!r} needs fast SRS (shortlist)")
    cands = list(inp.candidates)
    if not cands or k < 1:
        return SRSResult([], stop_reason="no candidates")
    ids = sorted(inp.train_ids)
    if inp.matrix is not None:
        X = np.asarray(inp.matrix, dtype=float)
        packed = np.packbits(X.astype(bool), axis=0)
        signature = [packed[:, j].tobytes() for j in range(len(cands))]
    else:
        row = {e: i for i, e in enumerate(ids)}
        X = np.zeros((len(ids), len(cands)))
        for j, p in enumerate(cands):
            for e in inp.firing(p):
                X[row[e], j] = 1.0
        signature = [inp.firing(p) for p in cands]
    y = np.array([1.0 if e in inp.train_positives else 0.0 for e in ids])
    weights = np.array([s.weight for s in scenarios], dtype=float)
    weights = weights / weights.sum()
    feature_sets = [frozenset(p.features) for p in cands]
    executable = np.array([[1.0 if f <= s.features else 0.0 for f in feature_sets] for s in scenarios])

    expected_exec = weights @ executable
    if screen is None or screen >= len(cands):
        screened = list(range(len(cands)))
    else:
        if inp.matrix is not None:
            ig = np.array([information_gain(frozenset(ids[i] for i in np.flatnonzero(X[:, j])), inp.train_ids, inp.train_positives)
                           for j in range(len(cands))])
        else:
            ig = np.array([information_gain(inp.firing(p), inp.train_ids, inp.train_positives) for p in cands])
        half = max(1, screen // 2)
        by_ig = sorted(range(len(cands)), key=lambda j: (-ig[j], j))[:half]
        by_exec = sorted(range(len(cands)), key=lambda j: (-ig[j] * expected_exec[j], j))[:half]
        screened = sorted(set(by_ig) | set(by_exec))

    if cv:
        folds = [fold for r in range(max(1, int(n_repeats))) for fold in grouped_folds(ids, subject_of, n_folds, seed + r)]
    else:
        folds = [(np.arange(len(ids)), np.arange(len(ids)))]
    if surrogates:
        chains = surrogate_chains(cands, X, max_surrogates) if chains is None else chains
    else:
        chains = None
    branch = branch_index(chains, cands, [s.features for s in scenarios]) if surrogates else None
    objective = _Objective(X, y, folds, executable, weights, C, mode, branch)
    selected: list[int] = []
    values: list[float] = []
    current = 0.0
    stop = "budget reached"
    if shortlist is not None:
        if mode != "frozen":
            raise ValueError("fast SRS (shortlist) supports the frozen objective only")
        fast = _FastFrozen(objective)
        if screener == "nmifs":
            n_rows = X.shape[0]
            fires = X.sum(axis=0)
            relevance = _mutual_information_counts(fires, float(y.sum()), X.T @ y, n_rows)
            entropy = _binary_entropy(fires / n_rows)
            red_sum = np.zeros(X.shape[1])
        while len(selected) < k:
            chosen_signatures = {signature[j] for j in selected}
            pending = [j for j in screened if j not in selected and signature[j] not in chosen_signatures]
            if not pending:
                stop = "screened candidates exhausted"
                break
            if screener == "nmifs":
                approx = (relevance if not selected else relevance - red_sum / len(selected))[pending]
            else:
                approx = fast.screen(selected, pending)
            order = [pending[i] for i in sorted(range(len(pending)), key=lambda i: (-approx[i], pending[i]))[:shortlist]]
            best, best_value = None, -math.inf
            seen = {}
            for j in sorted(order):  # generator rank breaks ties, as in exact greedy
                value, betas = fast.exact(selected, j)
                best_value = max(best_value, value)
                key = value if decision == "loglik" else fast.heldout_auprc(selected + [j], betas)
                seen[j] = (value, key)
                if best is None or key > best[0] + _EPS:
                    best = (key, j, betas, value)
            if trace is not None:
                by_value = max(sorted(seen), key=lambda j: (seen[j][0], -j))
                trace.append({"step": len(selected), "n_pending": len(pending), "current": current, "chosen": cands[best[1]].program_id,
                              "chosen_value": best[3], "chosen_key": best[0], "loglik_best": cands[by_value].program_id,
                              "best_value": best_value, "loglik_best_key": seen[by_value][1],
                              "chosen_screen_rank": order.index(best[1]) + 1, "stopped": bool(best_value < current + min_gain)})
            if best_value < current + min_gain:
                stop = f"no candidate increases the objective by min_gain={min_gain:g} bits/episode"
                break
            selected.append(best[1])
            values.append(best[3])
            current = best[3]
            fast.betas = best[2]
            if screener == "nmifs":
                pick = best[1]
                mi = _mutual_information_counts(fires, float(fires[pick]), X.T @ X[:, pick], n_rows)
                scale = np.minimum(entropy, entropy[pick])
                red_sum += np.divide(mi, scale, out=np.zeros_like(mi), where=scale > 0)
        chosen_surrogates = {cands[j].program_id: tuple(cands[q] for q in chains[j][1:]) for j in selected} if surrogates else {}
        return SRSResult([cands[j] for j in selected], values, len(screened), stop, chosen_surrogates)
    while len(selected) < k:
        best = None
        chosen_signatures = {signature[j] for j in selected}
        for j in screened:
            # a program that fires on exactly the same TRAIN episodes as a selected one is the same
            # coordinate for the probe; under L2 it would only dilute the penalty, so it is skipped
            if j in selected or signature[j] in chosen_signatures:
                continue
            value = objective(selected + [j])
            if best is None or value > best[0] + _EPS:
                best = (value, j)
        if best is None:
            stop = "screened candidates exhausted"
            break
        if best[0] < current + min_gain:
            stop = f"no candidate increases the objective by min_gain={min_gain:g} bits/episode"
            break
        selected.append(best[1])
        values.append(best[0])
        current = best[0]
    chosen_surrogates = {cands[j].program_id: tuple(cands[q] for q in chains[j][1:]) for j in selected} if surrogates else {}
    return SRSResult([cands[j] for j in selected], values, len(screened), stop, chosen_surrogates)
