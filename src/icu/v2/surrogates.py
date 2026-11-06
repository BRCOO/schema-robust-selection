"""Surrogate programs: schema-portable registries without refitting the frozen probe.

Problem. A registry R of K programs and a frozen probe h (fit on the source with one binary
coordinate per program) are shipped to a target whose schema F lacks some features. A
program p whose required features sigma_f(p) are not all in F cannot execute; the plain
deployment zeroes its coordinate.

Composite programs. Each program p ships with a short, ordered list of surrogate programs
q_1, q_2, ... (CART's surrogate splits, Breiman et al. 1984, sec. 5.3, applied to whole programs
instead of split variables). On a schema F the coordinate of p takes the firing of the first
branch of (p, q_1, q_2, ...) that can execute under F, and 0 when none can. On the source every
primary executes, so the probe is fit exactly as before; nothing is refit or repaired at the
target, which only supplies its schema.

Choosing surrogates (source TRAIN firings only, no labels). For a primary p with firing rate
pi_p, the blind rule (never fires) disagrees with p on min(pi_p, 1 - pi_p) of the episodes; a
candidate q disagrees on d(p, q) = P(x_p != x_q). CART's predictive measure of association

    lambda(p, q) = (min(pi_p, 1 - pi_p) - d(p, q)) / min(pi_p, 1 - pi_p)

is positive iff q predicts p better than the blind rule. Surrogates are the candidates with
lambda > 0 in decreasing lambda, skipping a candidate whose feature set contains the feature
set of an earlier branch (it could never be the first executable branch), up to
``max_surrogates`` (rpart's default of 5).

Why this criterion. For the frozen probe with weights beta, replacing coordinate x_p by x_q
changes the logit of an episode by at most |beta_p| * 1[x_p != x_q], and the logistic loss is
1-Lipschitz in the logit, so on source-distributed episodes the expected loss increase under F
is at most sum_p |beta_p| * d(p, b_p(F)), where b_p(F) is the branch used (d = pi_p for the
zeroed coordinate). Every surrogate with lambda > 0 has d(p, q) < pi_p (for pi_p <= 1/2) and so
tightens this bound relative to zeroing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import AbstractSet, Mapping, Sequence

import numpy as np

from src.icu.v2.bridge import refire_external
from src.icu.v2.types import ProgramRecord

MAX_SURROGATES = 5  # rpart's default maxsurrogate


def firing_matrix(programs: Sequence[ProgramRecord], firings: Mapping[str, AbstractSet[str]], episode_ids: Sequence[str]) -> np.ndarray:
    """Binary (episodes x programs) matrix of firings on ``episode_ids``."""
    row = {e: i for i, e in enumerate(episode_ids)}
    X = np.zeros((len(episode_ids), len(programs)), dtype=np.float64)
    for j, p in enumerate(programs):
        for e in firings.get(p.program_id, ()):
            i = row.get(e)
            if i is not None:
                X[i, j] = 1.0
    return X


def association_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CART association lambda[p, q] of candidate q as a surrogate of primary p, and the
    disagreement rates d[p, q], from a binary (episodes x candidates) matrix."""
    n = max(X.shape[0], 1)
    fires = X.sum(axis=0)
    both = X.T @ X
    d = (fires[:, None] + fires[None, :] - 2.0 * both) / n
    rate = fires / n
    blind = np.minimum(rate, 1.0 - rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(blind[:, None] > 0, (blind[:, None] - d) / blind[:, None], -np.inf)
    np.fill_diagonal(lam, -np.inf)
    return lam, d


def surrogate_chains(
    candidates: Sequence[ProgramRecord],
    X_train: np.ndarray,
    max_surrogates: int = MAX_SURROGATES,
    block_cells: int = 50_000_000,
) -> list[list[int]]:
    """For every candidate (row/column order of ``X_train``), the ordered candidate indices of
    its branches: the primary itself, then up to ``max_surrogates`` surrogates (see the module
    docstring). Ties in lambda keep the generator's rank order.

    The association rows are computed in blocks of primaries (about ``block_cells`` entries
    each), so memory is O(block x |C|) instead of O(|C|^2); co-firing counts use float32, which
    is exact for counts below 2^24, so lambda is the same as :func:`association_matrix`'s. Only the
    largest lambdas of a row are sorted (with every tie at the cut), and the full row only when
    their surrogates do not fill the chain."""
    m = len(candidates)
    n = max(X_train.shape[0], 1)
    if m == 0:
        return []
    if X_train.shape[0] >= 2 ** 24:
        raise ValueError("float32 co-firing counts are exact only below 2^24 episodes")
    feats = [frozenset(p.features) for p in candidates]
    fires = X_train.sum(axis=0)
    rate = fires / n
    blind = np.minimum(rate, 1.0 - rate)
    X32 = np.asarray(X_train, dtype=np.float32)
    block = max(1, min(m, block_cells // m))
    top = max(4 * max_surrogates, 32)

    def chain_from(i: int, order) -> list[int]:
        kept = [i]
        for j in order:
            if len(kept) - 1 >= max_surrogates:
                break
            if any(feats[r] <= feats[j] for r in kept):
                continue  # an earlier branch executes whenever this one does
            kept.append(j)
        return kept

    chains: list[list[int]] = []
    for start in range(0, m, block):
        rows = np.arange(start, min(start + block, m))
        both = (X32[:, rows].T @ X32).astype(np.float64)
        d = (fires[rows, None] + fires[None, :] - 2.0 * both) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(blind[rows, None] > 0, (blind[rows, None] - d) / blind[rows, None], -np.inf)
        lam[np.arange(len(rows)), rows] = -np.inf
        for local, i in enumerate(rows.tolist()):
            row = lam[local]
            positive = np.flatnonzero(row > 0)
            key = lambda j: (-row[j], j)
            if positive.size > top:
                cut = np.partition(row[positive], positive.size - top)[positive.size - top]
                head = positive[row[positive] >= cut]  # the largest lambdas, every tie at the cut kept
                kept = chain_from(i, sorted(head.tolist(), key=key))
                if len(kept) - 1 < max_surrogates and head.size < positive.size:
                    kept = chain_from(i, sorted(positive.tolist(), key=key))
            else:
                kept = chain_from(i, sorted(positive.tolist(), key=key))
            chains.append(kept)
    return chains


def branch_index(chains: Sequence[Sequence[int]], candidates: Sequence[ProgramRecord], schemas: Sequence[AbstractSet[str]]) -> np.ndarray:
    """(schemas x candidates) index of the branch each composite uses under each schema, -1 if
    no branch can execute."""
    feats = [frozenset(p.features) for p in candidates]
    out = np.full((len(schemas), len(candidates)), -1, dtype=np.int64)
    for s, schema in enumerate(schemas):
        available = frozenset(schema)
        executable: dict[frozenset[str], bool] = {}  # feature sets repeat, so test each once
        for i, chain in enumerate(chains):
            for j in chain:
                ok = executable.get(feats[j])
                if ok is None:
                    ok = executable[feats[j]] = feats[j] <= available
                if ok:
                    out[s, i] = j
                    break
    return out


@dataclass(frozen=True)
class CompositeRegistry:
    """A registry of primaries plus, per primary id, its ordered surrogate programs."""

    primaries: tuple[ProgramRecord, ...]
    surrogates: Mapping[str, tuple[ProgramRecord, ...]]

    def branches(self, program: ProgramRecord) -> tuple[ProgramRecord, ...]:
        return (program, *self.surrogates.get(program.program_id, ()))

    def resolve(self, schema: AbstractSet[str]) -> list[ProgramRecord | None]:
        """The branch each primary uses under ``schema`` (None when none can execute)."""
        available = frozenset(schema)
        resolved = []
        for p in self.primaries:
            resolved.append(next((b for b in self.branches(p) if frozenset(b.features) <= available), None))
        return resolved


def composite_registry(registry: Sequence[ProgramRecord], candidates: Sequence[ProgramRecord], train_firings: Mapping[str, AbstractSet[str]],
                       train_ids: Sequence[str], max_surrogates: int = MAX_SURROGATES) -> CompositeRegistry:
    """Attach surrogates (chosen among ``candidates`` on TRAIN firings) to every program of ``registry``."""
    pool = list(candidates)
    index = {p.program_id: i for i, p in enumerate(pool)}
    for p in registry:  # a registry program missing from the pool still gets surrogates from the pool
        if p.program_id not in index:
            index[p.program_id] = len(pool)
            pool.append(p)
    X = firing_matrix(pool, train_firings, list(train_ids))
    chains = surrogate_chains(pool, X, max_surrogates)
    surrogates = {p.program_id: tuple(pool[j] for j in chains[index[p.program_id]][1:]) for p in registry}
    return CompositeRegistry(tuple(registry), surrogates)


def refire_composite(
    model_bundle: Mapping[str, object],
    composite: CompositeRegistry,
    target_features: AbstractSet[str],
    external_firings: Mapping[str, AbstractSet[str]],
    external_episode_ids: Sequence[str],
) -> dict:
    """``bridge.refire_external`` for a composite registry: each coordinate keeps its primary's id
    and position but takes the firing (and executability) of the branch it resolves to under
    ``target_features``. The frozen probe is not refit."""
    resolved = composite.resolve(target_features)
    programs, firings = [], {}
    n_surrogate = 0
    for primary, branch in zip(composite.primaries, resolved):
        if branch is None:
            programs.append(primary)  # not executable: refire_external zeroes it
            firings[primary.program_id] = external_firings.get(primary.program_id, frozenset())
            continue
        if branch.program_id != primary.program_id:
            n_surrogate += 1
        programs.append(replace(primary, features=frozenset(branch.features)))
        firings[primary.program_id] = external_firings.get(branch.program_id, frozenset())
    result = refire_external(model_bundle, programs, target_features, firings, external_episode_ids)
    result["surrogate_fraction"] = n_surrogate / max(len(programs), 1)
    result["primary_survival"] = float(np.mean([frozenset(p.features) <= frozenset(target_features) for p in composite.primaries])) if programs else 0.0
    return result
