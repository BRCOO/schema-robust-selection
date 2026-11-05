"""Firing semantics of numeric programs (``motif``, ``shapelet``) on episode arrays (v2).

A numeric program runs on one episode's ``(values, mask, feature_names)``. Series access,
window validity, and the distance are the ones the miners use (the conventions in
``baselines/motif.py``; mask True = missing), so a program fires on the training
episodes exactly as its miner assumed.

* The program's feature (``params["feature"]``) is looked up by name. If the episode
  schema has no such column, the program cannot run and does not fire. This matches the
  sigma_f survival rule in ``bridge.py``.
* ``d`` is the minimum, over the valid windows of length ``params["length"]``, of the
  length-normalized distance to ``params["template"]``. The distance is z-normalized
  when ``params["normalize"]`` is true (the default). Validity uses
  ``params["missing_policy"]`` (default ``"strict"``) and ``params["eps"]``. ``d`` is
  ``inf`` if no window is valid, e.g. the feature is not observed for ``length``
  consecutive rows, or only constant windows exist under z-normalization.
* ``motif`` fires iff ``d <= threshold``. ``shapelet`` with ``params["direction"] == "le"``
  (the default) fires iff ``d <= threshold``. With ``"gt"`` it fires iff ``d > threshold``,
  including ``d = inf``, which reproduces the training split of ``shapelet.best_split``.

The comparison uses ``threshold + MATCH_ATOL`` (1e-9). Shapelet split points lie more
than 5e-9 from every training distance (``shapelet.SPLIT_TOL``), and exactly repeated
shapes have distance 0, so the tolerance only absorbs recomputation noise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.icu.v2.baselines.motif import (
    DEFAULT_EPS,
    MATCH_ATOL,
    EpisodeArrays,
    WindowBank,
    _check_policy,
    build_window_bank,
    feature_series,
    sq_distances,
    template_bank,
)
from src.icu.v2.types import ProgramRecord

NUMERIC_KINDS: tuple[str, ...] = ("motif", "shapelet")
DIRECTIONS: tuple[str, ...] = ("le", "gt")


@dataclass(frozen=True, eq=False)
class _NumericSpec:
    program_id: str
    feature: str
    template: WindowBank
    threshold: float
    missing_policy: str
    eps: float
    fires_when_close: bool

    @property
    def key(self) -> tuple:
        """Programs with equal keys share the episode's windows."""
        return (self.feature, self.template.m, self.template.normalize, self.missing_policy, self.eps)


def _spec(program: ProgramRecord) -> _NumericSpec:
    if program.kind not in NUMERIC_KINDS:
        raise ValueError(f"program {program.program_id!r} has kind {program.kind!r}, not one of {NUMERIC_KINDS}")
    params = program.params
    feature = params.get("feature")
    if not isinstance(feature, str) or not feature:
        raise ValueError(f"program {program.program_id!r} needs params['feature']")
    if program.features and frozenset(program.features) != frozenset({feature}):
        raise ValueError(f"program {program.program_id!r}: features {set(program.features)} != {{{feature!r}}}")
    template = np.asarray(params.get("template", ()), dtype=float).reshape(-1)
    length = params.get("length", template.size)
    if template.size == 0 or isinstance(length, bool) or int(length) != template.size:
        raise ValueError(f"program {program.program_id!r}: template length must equal params['length']")
    threshold = float(params.get("threshold", math.nan))
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"program {program.program_id!r}: threshold must be finite and >= 0")
    normalize = params.get("normalize", True)
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError(f"program {program.program_id!r}: params['normalize'] must be a bool")
    direction = params.get("direction", "le")
    if direction not in DIRECTIONS or (program.kind == "motif" and direction != "le"):
        raise ValueError(f"program {program.program_id!r}: invalid direction {direction!r}")
    eps = float(params.get("eps", DEFAULT_EPS))
    return _NumericSpec(
        program_id=program.program_id,
        feature=feature,
        template=template_bank(template, normalize=bool(normalize), eps=eps),
        threshold=threshold,
        missing_policy=_check_policy(str(params.get("missing_policy", "strict"))),
        eps=eps,
        fires_when_close=direction == "le",
    )


def _min_distances(
    templates: WindowBank,
    key: tuple,
    values: np.ndarray,
    mask: np.ndarray | None,
    feature_names: Sequence[str],
) -> np.ndarray | None:
    """Per-template minimum distance over the episode's valid windows; None if not executable."""
    feature, m, normalize, missing_policy, eps = key
    series = feature_series(values, mask, feature_names, feature, missing_policy)
    if series is None:
        return None
    windows = build_window_bank([series], m, normalize=normalize, eps=eps)
    if len(windows) == 0:
        return np.full(len(templates), np.inf)
    return np.sqrt(sq_distances(templates, windows).min(axis=1))


def _decide(spec: _NumericSpec, distance: float | None) -> bool:
    if distance is None:
        return False
    close = distance <= spec.threshold + MATCH_ATOL
    return close if spec.fires_when_close else not close


def _stack(banks: list[WindowBank]) -> WindowBank:
    count = len(banks)
    return WindowBank(
        banks[0].m,
        banks[0].normalize,
        np.vstack([bank.shape for bank in banks]),
        np.concatenate([bank.level for bank in banks]),
        np.concatenate([bank.sq_norm for bank in banks]),
        np.arange(count, dtype=np.int64),
        np.zeros(count, dtype=np.int64),
        np.vstack([bank.raw for bank in banks]),
    )


def program_distance(
    program: ProgramRecord,
    values: np.ndarray,
    mask: np.ndarray | None,
    feature_names: Sequence[str],
) -> float | None:
    """Minimum distance from the program's template to the episode's valid windows.

    Returns ``inf`` if no window is valid, and None if the episode schema lacks the feature.
    """
    spec = _spec(program)
    distances = _min_distances(spec.template, spec.key, values, mask, feature_names)
    return None if distances is None else float(distances[0])


def fires_numeric(
    program: ProgramRecord,
    values: np.ndarray,
    mask: np.ndarray | None,
    feature_names: Sequence[str],
) -> bool:
    """Whether a motif or shapelet program fires on one episode (module docstring)."""
    spec = _spec(program)
    distances = _min_distances(spec.template, spec.key, values, mask, feature_names)
    return _decide(spec, None if distances is None else float(distances[0]))


def numeric_firing_sets(
    programs: Sequence[ProgramRecord],
    episodes: Mapping[str, EpisodeArrays],
) -> dict[str, frozenset[str]]:
    """``program_id -> frozenset(episode ids where it fires)`` for every program.

    ``episodes`` maps episode id -> ``(values, mask, feature_names)``. Programs on the same
    feature, length, normalization, and missing policy are evaluated together.
    """
    specs = [_spec(program) for program in programs]
    ids = [spec.program_id for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("program ids must be unique")
    members: dict[tuple, list[int]] = {}
    for index, spec in enumerate(specs):
        members.setdefault(spec.key, []).append(index)
    banks = {key: _stack([specs[i].template for i in indexes]) for key, indexes in members.items()}
    fired: dict[str, list[str]] = {pid: [] for pid in ids}
    for episode_id, (values, mask, names) in episodes.items():
        for key, indexes in members.items():
            distances = _min_distances(banks[key], key, values, mask, names)
            for row, index in enumerate(indexes):
                spec = specs[index]
                if _decide(spec, None if distances is None else float(distances[row])):
                    fired[spec.program_id].append(episode_id)
    return {pid: frozenset(episode_ids) for pid, episode_ids in fired.items()}
