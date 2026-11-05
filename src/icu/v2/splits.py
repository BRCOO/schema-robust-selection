"""Patient-level train/val/test split for the v2 protocol.

v1 grouped episodes by ``stay_id`` on the first N rows of a stay-sorted index: 31
patients, and a validation split made of one patient with 144 episodes. v2 groups by
``subject_id`` and assigns whole patients with a seeded procedure that does not depend on
the input order:

1. Optional cap: keep at most ``max_episodes_per_patient`` episodes per patient, chosen
   by a seeded hash of ``episode_id``; a patient with both classes keeps at least one
   positive and one control.
2. Stratify patients by whether they have any positive episode.
3. Within each stratum, visit patients in seeded hash order and put each one in the split
   whose relative deviation ``|filled - target| / target`` grows least (target = fraction
   x stratum episodes): small splits take patients that fit, and a patient too large to
   fit anywhere goes where its overshoot is relatively smallest, so one large patient
   cannot swamp val.
4. Repair: while a split lacks a class, move the patient that restores it with the
   smallest total relative deviation from the episode targets, never removing a class
   from the donor split.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from src.icu.v2.types import EpisodeRecord

SPLIT_NAMES = ("train", "val", "test")
UNASSIGNED = "unassigned"
_CLASS_FIELDS = ("positives", "controls")


@dataclass
class _Tally:
    episodes: int = 0
    positives: int = 0
    controls: int = 0


def _stable_key(seed: int, salt: str, value: object) -> str:
    """Seeded sort key that is identical across runs and platforms (``hash()`` is salted)."""
    return hashlib.sha256(f"{seed}|{salt}|{value}".encode("utf-8")).hexdigest()


def _validate_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in fractions)
    if len(values) != len(SPLIT_NAMES):
        raise ValueError(f"fractions must give (train, val, test), got {len(values)} values")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(f"every split fraction must be positive, got {values}")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {sum(values):.6f}")
    return values


def _validate_cap(cap: int | None) -> int | None:
    if cap is None:
        return None
    if isinstance(cap, bool) or int(cap) != cap or cap < 2:
        raise ValueError(
            "max_episodes_per_patient must be None or an integer >= 2 "
            f"(a patient with both classes keeps one of each), got {cap!r}"
        )
    return int(cap)


def _validate_episodes(episodes: Sequence[EpisodeRecord]) -> None:
    if not episodes:
        raise ValueError("no episodes to split")
    seen: set[str] = set()
    subject_of_stay: dict[int, int] = {}
    for episode in episodes:
        if episode.episode_id in seen:
            raise ValueError(f"duplicate episode_id {episode.episode_id!r}")
        seen.add(episode.episode_id)
        if episode.label not in (0, 1):
            raise ValueError(f"episode {episode.episode_id!r} has label {episode.label!r}; expected 0 or 1")
        owner = subject_of_stay.setdefault(episode.stay_id, episode.subject_id)
        if owner != episode.subject_id:
            raise ValueError(f"stay_id {episode.stay_id} is linked to subject_id {owner} and {episode.subject_id}")


def _cap_patient(episodes: list[EpisodeRecord], cap: int | None, seed: int) -> list[EpisodeRecord]:
    """Keep at most ``cap`` episodes by seeded hash order, one of each class if present."""
    if cap is None or len(episodes) <= cap:
        return episodes
    ranked = sorted(episodes, key=lambda episode: _stable_key(seed, "cap", episode.episode_id))
    first_positive = next((episode for episode in ranked if episode.label == 1), None)
    first_control = next((episode for episode in ranked if episode.label == 0), None)
    kept: list[EpisodeRecord] = []
    if first_positive is not None and first_control is not None:
        kept = [first_positive, first_control]
    kept_ids = {episode.episode_id for episode in kept}
    for episode in ranked:
        if len(kept) >= cap:
            break
        if episode.episode_id not in kept_ids:
            kept.append(episode)
            kept_ids.add(episode.episode_id)
    return kept


def _check_feasible(tallies: dict[int, _Tally]) -> None:
    with_positive = sum(1 for tally in tallies.values() if tally.positives > 0)
    with_control = sum(1 for tally in tallies.values() if tally.controls > 0)
    if with_positive < len(SPLIT_NAMES) or with_control < len(SPLIT_NAMES):
        raise ValueError(
            "cannot give train, val, and test both classes with whole patients: need at least 3 "
            f"patients with a positive episode and 3 with a control episode, found {with_positive} "
            f"and {with_control}"
        )


def _placement_cost(filled: int, size: int, target: float) -> float:
    """Change of ``|filled - target| / target`` when a patient of ``size`` episodes joins."""
    return (abs(filled + size - target) - abs(filled - target)) / target


def _assign_patients(tallies: dict[int, _Tally], fractions: tuple[float, ...], seed: int) -> dict[int, str]:
    assignment: dict[int, str] = {}
    for has_positive in (True, False):
        members = sorted(
            (pid for pid, tally in tallies.items() if (tally.positives > 0) == has_positive),
            key=lambda pid: _stable_key(seed, "order", pid),
        )
        total = sum(tallies[pid].episodes for pid in members)
        targets = [fraction * total for fraction in fractions]
        filled = [0] * len(SPLIT_NAMES)
        for pid in members:
            size = tallies[pid].episodes
            # Smallest growth of relative deviation; ties go to the larger deficit, then split order.
            best = min(
                range(len(SPLIT_NAMES)),
                key=lambda j: (_placement_cost(filled[j], size, targets[j]), filled[j] - targets[j], j),
            )
            assignment[pid] = SPLIT_NAMES[best]
            filled[best] += size
    return assignment


def _split_rows(assignment: dict[int, str], tallies: dict[int, _Tally]) -> dict[str, dict[str, int]]:
    rows = {name: {"episodes": 0, "positives": 0, "controls": 0} for name in SPLIT_NAMES}
    for pid, name in assignment.items():
        rows[name]["episodes"] += tallies[pid].episodes
        rows[name]["positives"] += tallies[pid].positives
        rows[name]["controls"] += tallies[pid].controls
    return rows


def _repair_class_coverage(
    assignment: dict[int, str],
    tallies: dict[int, _Tally],
    fractions: tuple[float, ...],
    seed: int,
) -> int:
    """Move patients until every split has both classes; returns the number of moves.

    With at least 3 carriers of each class (checked beforehand) one move per missing
    (split, class) always exists: some other split holds two carriers and can give one
    away without losing a class, so the loop ends after at most six moves.
    """
    total = sum(tally.episodes for tally in tallies.values())
    targets = {name: fraction * total for name, fraction in zip(SPLIT_NAMES, fractions)}
    moves = 0
    while True:
        rows = _split_rows(assignment, tallies)
        gap = next(
            ((name, field) for name in SPLIT_NAMES for field in _CLASS_FIELDS if rows[name][field] == 0),
            None,
        )
        if gap is None:
            return moves
        receiver, field = gap
        best: tuple[tuple[float, str], int] | None = None
        for pid, tally in tallies.items():
            donor = assignment[pid]
            if donor == receiver or getattr(tally, field) == 0:
                continue
            # The donor must keep every class it has now (which also keeps it non-empty).
            if any(rows[donor][f] > 0 and rows[donor][f] == getattr(tally, f) for f in _CLASS_FIELDS):
                continue
            sizes = {name: rows[name]["episodes"] for name in SPLIT_NAMES}
            sizes[donor] -= tally.episodes
            sizes[receiver] += tally.episodes
            deviation = sum(abs(sizes[name] - targets[name]) / targets[name] for name in SPLIT_NAMES)
            candidate = (deviation, _stable_key(seed, "repair", pid))
            if best is None or candidate < best[0]:
                best = (candidate, pid)
        if best is None:
            raise ValueError(f"cannot give split {receiver!r} any {field} without taking a class from another split")
        assignment[best[1]] = receiver
        moves += 1


def patient_level_split(
    episodes: Sequence[EpisodeRecord],
    fractions: Sequence[float] = (0.7, 0.1, 0.2),
    seed: int = 0,
    max_episodes_per_patient: int | None = None,
) -> tuple[list[EpisodeRecord], dict]:
    """Assign whole patients (``subject_id``, not ``stay_id``) to train/val/test.

    Patients are stratified by whether they have any positive episode, and each stratum
    is split toward the requested episode fractions. ``max_episodes_per_patient`` (>= 2)
    caps the episodes kept per patient deterministically, keeping one positive and one
    control when the patient has both; capped-out episodes are left out of the result.
    The assignment depends only on the episode set and ``seed``, not on the input order.

    Raises ValueError when the splits cannot all hold both classes, which needs at least
    3 patients with a positive and 3 with a control episode.

    Returns the kept episodes (input order, ``split`` set with ``dataclasses.replace``)
    and a manifest with per-split episode/positive/control/patient counts, requested and
    achieved fractions, seed, cap, and sorted patient id lists.
    """
    episodes = list(episodes)
    fractions = _validate_fractions(fractions)
    cap = _validate_cap(max_episodes_per_patient)
    seed = int(seed)
    _validate_episodes(episodes)

    by_patient: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for episode in episodes:
        by_patient[episode.subject_id].append(episode)
    kept_ids: set[str] = set()
    tallies: dict[int, _Tally] = {}
    n_capped = 0
    for pid, patient_episodes in by_patient.items():
        kept = _cap_patient(patient_episodes, cap, seed)
        if len(kept) < len(patient_episodes):
            n_capped += 1
        kept_ids.update(episode.episode_id for episode in kept)
        tallies[pid] = _Tally(
            episodes=len(kept),
            positives=sum(1 for episode in kept if episode.label == 1),
            controls=sum(1 for episode in kept if episode.label == 0),
        )

    _check_feasible(tallies)
    assignment = _assign_patients(tallies, fractions, seed)
    n_moves = _repair_class_coverage(assignment, tallies, fractions, seed)

    split_episodes = [
        replace(episode, split=assignment[episode.subject_id]) for episode in episodes if episode.episode_id in kept_ids
    ]
    assert_patient_disjoint(split_episodes)
    summary = split_summary(split_episodes)
    manifest = {
        "group_key": "subject_id",
        "stratify_by": "patient_has_positive_episode",
        "seed": seed,
        "max_episodes_per_patient": cap,
        "fractions_requested": dict(zip(SPLIT_NAMES, fractions)),
        "fractions_achieved": {name: summary["splits"][name]["episode_fraction"] for name in SPLIT_NAMES},
        "n_episodes_input": len(episodes),
        "n_episodes": len(split_episodes),
        "n_episodes_dropped_by_cap": len(episodes) - len(split_episodes),
        "n_patients": len(tallies),
        "n_patients_capped": n_capped,
        "n_class_repair_moves": n_moves,
        "splits": {
            name: dict(
                summary["splits"][name],
                patient_ids=sorted(pid for pid, split in assignment.items() if split == name),
            )
            for name in SPLIT_NAMES
        },
    }
    return split_episodes, manifest


def assert_patient_disjoint(episodes: Iterable[EpisodeRecord]) -> None:
    """Raise ValueError if an episode has no split or a patient or stay spans several splits."""
    episodes = list(episodes)
    unassigned = [episode.episode_id for episode in episodes if not episode.split]
    if unassigned:
        raise ValueError(f"{len(unassigned)} episodes have no split, e.g. {unassigned[:3]}")
    for key in ("subject_id", "stay_id"):
        splits_of: dict[int, set[str]] = defaultdict(set)
        for episode in episodes:
            splits_of[getattr(episode, key)].add(episode.split)
        overlap = sorted(value for value, names in splits_of.items() if len(names) > 1)
        if overlap:
            example = overlap[0]
            raise ValueError(
                f"{len(overlap)} {key} values appear in more than one split, "
                f"e.g. {key}={example} in {sorted(splits_of[example])}"
            )


def split_summary(episodes: Iterable[EpisodeRecord]) -> dict:
    """Per-split episode, class, patient, and stay counts with episode fractions.

    train/val/test are always present (zeros when empty); any other split label is added,
    and an empty label is reported as ``"unassigned"``.
    """
    episodes = list(episodes)
    names = list(SPLIT_NAMES) + sorted({episode.split or UNASSIGNED for episode in episodes} - set(SPLIT_NAMES))
    counts = {name: {"episodes": 0, "positives": 0, "controls": 0} for name in names}
    patients: dict[str, set[int]] = {name: set() for name in names}
    positive_patients: dict[str, set[int]] = {name: set() for name in names}
    stays: dict[str, set[int]] = {name: set() for name in names}
    for episode in episodes:
        name = episode.split or UNASSIGNED
        counts[name]["episodes"] += 1
        counts[name]["positives" if episode.label == 1 else "controls"] += 1
        patients[name].add(episode.subject_id)
        stays[name].add(episode.stay_id)
        if episode.label == 1:
            positive_patients[name].add(episode.subject_id)
    total = len(episodes)
    return {
        "n_episodes": total,
        "n_patients": len({episode.subject_id for episode in episodes}),
        "splits": {
            name: {
                **counts[name],
                "patients": len(patients[name]),
                "patients_with_positive": len(positive_patients[name]),
                "stays": len(stays[name]),
                "episode_fraction": counts[name]["episodes"] / total if total else 0.0,
            }
            for name in names
        },
    }
