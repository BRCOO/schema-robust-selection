"""Build the v2 episode cohort: task eligibility, patient-level split, pre-registration.

This is the step between the v1 episode index and every v2 experiment:

1. stream ``shock_episode_index.json`` (560 MB for MIMIC-IV) without loading it whole;
2. apply a :class:`~src.icu.v2.task.TaskSpec` (prediction gap, excluded features,
   full-window rule, control eligibility) using an anchor index built from the WHOLE
   index, so control rules see every anchor of a stay;
3. split whole patients (``subject_id``) into train/val/test with
   :func:`~src.icu.v2.splits.patient_level_split`;
4. write a pre-registration record (spec, split parameters, source hashes) that is meant
   to be committed before any model or miner is run on the cohort.

Unlike v1, records are never taken as "the first N rows": the index is sorted by
``stay_id``, so first-N sampling gave 31 patients for 500 episodes. ``limit`` exists only
for smoke tests and is reported in the outputs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.icu.v2.splits import patient_level_split, split_summary
from src.icu.v2.task import TaskSpec, apply_task, build_stay_anchor_index, episode_label, summarize_exclusions
from src.icu.v2.types import EpisodeRecord

KEEP_FIELDS: tuple[str, ...] = (
    "episode_id",
    "episode_type",
    "stay_id",
    "subject_id",
    "hadm_id",
    "anchor_kind",
    "anchor_time",
    "window_start",
    "window_end",
    "start_time",
    "end_time",
    "shape",
    "resample_minutes",
)

V2_SOURCE_FILES: tuple[str, ...] = (
    "src/icu/v2/types.py",
    "src/icu/v2/task.py",
    "src/icu/v2/splits.py",
    "src/icu/v2/cohort.py",
)


_SEPARATORS = " \t\r\n,"


def _skip(text: str, pos: int, chars: str) -> int:
    end = len(text)
    while pos < end and text[pos] in chars:
        pos += 1
    return pos


def iter_index_records(path: str | Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Yield the objects of a top-level JSON array one at a time.

    Decodes in place with ``json.JSONDecoder.raw_decode(buffer, pos)`` and only compacts
    the buffer when more text must be read, so memory stays proportional to one record
    plus one chunk and no text is copied per record.
    """
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as handle:
        buffer = handle.read(chunk_size)
        eof = not buffer
        pos = _skip(buffer, 0, " \t\r\n")
        if pos >= len(buffer) or buffer[pos] != "[":
            raise ValueError(f"{path}: expected a top-level JSON array")
        pos += 1
        while True:
            pos = _skip(buffer, pos, _SEPARATORS)
            if pos < len(buffer):
                if buffer[pos] == "]":
                    return
                try:
                    record, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    pass  # record continues in the next chunk (or the file is malformed)
                else:
                    if not isinstance(record, dict):
                        raise ValueError(f"{path}: array items must be JSON objects")
                    yield record
                    pos = end
                    continue
            if eof:
                raise ValueError(f"{path}: truncated or malformed JSON array")
            more = handle.read(chunk_size)
            eof = not more
            buffer = buffer[pos:] + more
            pos = 0


def trim_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the metadata fields the task and split steps need."""
    return {key: record[key] for key in KEEP_FIELDS if key in record}


def to_episode_record(meta: Mapping[str, Any]) -> EpisodeRecord:
    """Convert kept metadata to an ``EpisodeRecord``; a missing ``subject_id`` is an error.

    v1 silently fell back to ``stay_id`` when ``subject_id`` was missing
    (``src/icu/comparators/data_contract.py``), which turns patient grouping into stay
    grouping. v2 refuses instead.
    """
    subject = meta.get("subject_id")
    if subject is None or (isinstance(subject, float) and subject != subject):
        raise ValueError(f"episode {meta.get('episode_id')!r} has no subject_id")
    return EpisodeRecord(
        episode_id=str(meta["episode_id"]),
        subject_id=int(subject),
        stay_id=int(meta["stay_id"]),
        label=episode_label(meta),
        anchor_kind=str(meta.get("anchor_kind") or ""),
        anchor_time=str(meta.get("anchor_time") or ""),
        window_start=str(meta.get("window_start") or ""),
        window_end=str(meta.get("window_end") or ""),
    )


def task_spec_from_dict(payload: Mapping[str, Any]) -> TaskSpec:
    """Inverse of ``TaskSpec.to_dict()``; JSON lists become tuples, unknown keys raise."""
    known = {field.name for field in fields(TaskSpec)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown TaskSpec fields {unknown}")
    return TaskSpec(**{key: tuple(value) if isinstance(value, list) else value for key, value in payload.items()})


def build_cohort(
    records: Sequence[Mapping[str, Any]],
    spec: TaskSpec,
    *,
    fractions: Sequence[float] = (0.7, 0.1, 0.2),
    seed: int = 0,
    max_episodes_per_patient: int | None = None,
    sample_patients: int | None = None,
    sample_seed: int = 0,
) -> dict[str, Any]:
    """Apply the task to trimmed index records and split the kept episodes by patient.

    The stay anchor index includes the paired anchors of controls, which are real anchors
    of the stay; an index built from a subsample would under-count exclusions. It is
    therefore always built from ALL records, and the optional patient sample
    (:func:`sample_patient_ids`) is drawn only after eligibility.
    """
    anchors = build_stay_anchor_index(records, include_control_anchors=True)
    kept, excluded = apply_task(records, spec, stay_anchors=anchors)
    episodes = [to_episode_record(meta) for meta in kept]
    n_patients_eligible = len({episode.subject_id for episode in episodes})
    if sample_patients is not None:
        chosen = sample_patient_ids({episode.subject_id for episode in episodes}, sample_patients, sample_seed)
        episodes = [episode for episode in episodes if episode.subject_id in chosen]
    split_episodes, manifest = patient_level_split(
        episodes,
        fractions=fractions,
        seed=seed,
        max_episodes_per_patient=max_episodes_per_patient,
    )
    return {
        "episodes": split_episodes,
        "split_manifest": manifest,
        "split_summary": split_summary(split_episodes),
        "exclusions": excluded,
        "exclusion_summary": summarize_exclusions(excluded),
        "n_input_records": len(records),
        "n_eligible": len(kept),
        "n_patients_eligible": n_patients_eligible,
        "n_patients_sampled": len({episode.subject_id for episode in episodes}),
        "n_episodes_sampled": len(episodes),
    }


def sample_patient_ids(subject_ids: Iterable[int], n: int, seed: int) -> frozenset[int]:
    """A uniform random sample of ``n`` patients that does not depend on input order.

    Patients are ranked by ``sha256(f"{seed}:sample:{subject_id}")`` and the first ``n``
    are kept, so the sample is reproducible and independent of how the index is sorted
    (the v1 index is sorted by ``stay_id``, which made first-N prefixes tiny cohorts).
    """
    if int(n) < 1:
        raise ValueError(f"sample size must be >= 1, got {n!r}")
    ranked = sorted(set(int(s) for s in subject_ids), key=lambda s: hashlib.sha256(f"{seed}:sample:{s}".encode()).hexdigest())
    return frozenset(ranked[: int(n)])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def preregistration_record(
    spec: TaskSpec,
    split_params: Mapping[str, Any],
    *,
    root: str | Path,
    index_path: str | Path | None = None,
    hash_index: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the analysis settings before any result is seen.

    ``content_sha256`` covers the spec, split parameters and ``extra`` (e.g. metric
    definitions, selector weights, which prediction gap is primary), so a later change is
    detectable. Hashing the 560 MB index is optional because it takes a while.
    """
    root = Path(root)
    content = {
        "task_spec": spec.to_dict(),
        "split_params": dict(split_params),
        "extra": dict(extra or {}),
    }
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return {
        **content,
        "content_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(root),
        "v2_source_sha256": {
            rel: sha256_file(root / rel) for rel in V2_SOURCE_FILES if (root / rel).exists()
        },
        "index_path": str(index_path) if index_path is not None else None,
        "index_sha256": sha256_file(index_path) if (index_path is not None and hash_index) else None,
    }


def write_cohort_outputs(result: Mapping[str, Any], spec: TaskSpec, prereg: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    """Write the cohort files the v2 runners consume; returns name -> path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payloads = {
        "episodes_split.json": [asdict(episode) for episode in result["episodes"]],
        "task_spec.json": spec.to_dict(),
        "split_manifest.json": result["split_manifest"],
        "cohort_summary.json": {
            "n_input_records": result["n_input_records"],
            "n_eligible": result["n_eligible"],
            "n_patients_eligible": result["n_patients_eligible"],
            "n_patients_sampled": result["n_patients_sampled"],
            "n_episodes_sampled": result["n_episodes_sampled"],
            "split_summary": result["split_summary"],
            "exclusion_summary": result["exclusion_summary"],
        },
        "exclusions.json": result["exclusions"],
        "preregistration.json": prereg,
    }
    written: dict[str, str] = {}
    for name, payload in payloads.items():
        path = out / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        written[name] = str(path)
    return written
