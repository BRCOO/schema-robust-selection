#!/usr/bin/env python3
"""Matched runtime and memory scalability benchmark of the v2 discovery stage.

Answers the reviewer request for "matched end-to-end runtime and memory against
PrefixSpan, RuleGrowth and the other mining baselines under identical hardware and
candidate budgets" and for runtime versus cohort size.

Protocol
--------
* Cohort: the TRAIN split of a v2 cohort (default ``artifacts/v2/task/x_mimic_iv_gap1h``).
  Fraction ``f`` keeps the first ``round(f * N)`` of the N TRAIN patients in the seeded
  hash order of :func:`src.icu.v2.cohort.sample_patient_ids`, with all their TRAIN
  episodes. The subsamples are patient-level, nested, and independent of input order.
* Every (method, fraction) runs in a fresh job process, one job at a time: method by
  method with fractions ascending, FMG-Bridge last, each job started only once enough RAM
  and disk are free (other experiments share the machine). The driver samples the
  resident set size (the working set on Windows) of the job process plus all of its
  descendants (the SPMF JVM, the v1 FMG mining processes) every ``--interval`` seconds
  from a psutil polling thread. The sampler lives in the driver, so it never competes
  with the measured job for its GIL. BLAS/OpenMP pools of every job are limited to
  ``--blas-threads`` (default 1). Every job and its descendants are pinned to the
  performance cores of a hybrid CPU (``--cpu-affinity``) and run at above-normal priority
  (``--priority``), so the core type does not change with the background load.
* Only the discovery stage is timed, from the episode files on disk to the ranked
  candidate pool, on TRAIN episodes only. Interpreter start-up and imports of the job,
  firing on other episodes, the v2 selector, the bridge, and evaluation are excluded.

  =================  ==================================================================
  prefixspan,        load arrays -> global TRAIN bins + eventization -> SPMF 2.64b in one
  erminer            JVM on the positive TRAIN sequences -> contrast ranking on TRAIN ->
                     dedup -> pool cap (``pools.build_symbolic_pool`` without firing)
  matrix_profile     load arrays -> ``mine_motif_programs`` (positive TRAIN episodes,
                     neighbours from other patients: groups = subject ids)
  shapelet           load arrays -> ``mine_shapelet_programs`` (all TRAIN episodes)
  fmg_bridge         per_episode_mining: unmodified v1 ``run_mimic_experiment.py`` per
                     shard of ``--fmg-shard-size`` episodes of the FMG view, shards run one
                     after another (``run_fmg_mining_sharded`` with one worker), and each
                     shard cache is merged into a compact part and deleted at once;
                     support_prefilter: ``fmg_prefilter`` support counts over the parts,
                     then the filtered, compacted episodes are loaded into memory (the
                     production scripts write them to disk and stream them back);
                     contrastive_stage: ``fmg_adapter.run_fmg_contrastive_stage`` twice
                     (without, then with the transfer target), as in
                     ``run_fmg_train_only.py``, and conversion to ``ProgramRecord``.
  =================  ==================================================================

Outputs in ``--out-dir``: ``scalability.csv`` (one row per method x fraction, plus one row
per FMG stage), ``scalability.json`` (metadata, subsamples, and every job with per-phase
timings and memory), ``memory_traces.json`` (1-s maxima of the sampled RSS),
``README.md``, and one log per job in ``logs/``. Temporary files go to ``<out-dir>/tmp``,
are deleted after each job, and the directory is removed at the end.

Only when the machine is free enough (the full run takes about 1-2 hours):

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python \\
      scripts/v2/run_scalability_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence

import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Project modules are imported lazily, so a job process loads only what its method needs
# before the measured stage starts.
SCRIPT = Path(__file__).resolve()
MIB = float(1 << 20)
GIB = float(1 << 30)

DEFAULT_COHORT = Path("artifacts/v2/task/x_mimic_iv_gap1h/episodes_split.json")
DEFAULT_EPISODE_DIR = Path("data/processed/v2/extract/mimic_iv_gap1h_r2")
DEFAULT_FMG_DIR = Path("data/processed/v2/extract/mimic_iv_gap1h_r2_fmg")
DEFAULT_OUT_DIR = Path("artifacts/v2/scalability")
DEFAULT_FRACTIONS = (0.25, 0.5, 1.0)

FMG_METHOD = "fmg_bridge"
SPMF_METHODS = ("prefixspan", "erminer")
NUMERIC_METHODS = ("matrix_profile", "shapelet")
# Default job order: method by method, FMG-Bridge (the largest memory footprint) last.
METHODS = (*SPMF_METHODS, *NUMERIC_METHODS, FMG_METHOD)
FMG_STAGES = ("per_episode_mining", "support_prefilter", "contrastive_stage")
TOTAL_STAGE = "discovery_total"
REPRESENTATION = {"n_bins": 4, "observation_tokens": True, "trend_tokens": False}

# Candidate budgets and miner parameters (same as artifacts/v2/comparison/configs/x_mimic_iv_gap1h.json).
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "prefixspan": {"minsup": 0.2, "max_pattern_length": 3, "pool_cap": 200, "timeout_s": 7200},
    "erminer": {
        "minsup": 0.2,
        "minconf": 0.6,
        "max_antecedent": 2,
        "max_consequent": 1,
        "pool_cap": 200,
        "timeout_s": 7200,
    },
    "matrix_profile": {"lengths": [3, 4, 5], "top_k_per_feature": 3},
    "shapelet": {"lengths": [3, 4, 5], "top_k_per_feature": 3},
    FMG_METHOD: {
        "shard_size": 220,
        "transfer_target": "two-pass",
        "compact_memory": True,
        "miner_config": "v1_locked_reference_profile",
        "domain": "mimic_iv",
    },
}

def method_note(method: str, params: Mapping[str, Any]) -> str:
    """One-line description of a method with the parameters actually used."""
    p = dict(params)
    if method == "prefixspan":
        return (
            f"SPMF 2.64b PrefixSpan in one JVM (-Xmx2g) on positive TRAIN sequences (minsup {p.get('minsup')}, "
            f"max length {p.get('max_pattern_length')}); contrast ranking on TRAIN, dedup, pool cap {p.get('pool_cap')}"
        )
    if method == "erminer":
        return (
            f"SPMF 2.64b ERMiner (complete RuleGrowth rule set) in one JVM (-Xmx2g) on positive TRAIN sequences "
            f"(minsup {p.get('minsup')}, minconf {p.get('minconf')}, antecedent <= {p.get('max_antecedent')}, "
            f"consequent <= {p.get('max_consequent')}); contrast ranking on TRAIN, dedup, pool cap {p.get('pool_cap')}"
        )
    lengths = "/".join(str(m) for m in p.get("lengths", ()))
    if method == "matrix_profile":
        return (
            f"exact cross-episode matrix-profile motifs on positive TRAIN episodes, neighbours from other patients "
            f"(groups = subject ids), lengths {lengths}, top-{p.get('top_k_per_feature')} per feature"
        )
    if method == "shapelet":
        return f"exact information-gain shapelets on all TRAIN episodes, lengths {lengths}, top-{p.get('top_k_per_feature')} per feature"
    if method == FMG_METHOD:
        return (
            f"v1 per-episode mining on the FMG view (shards of {p.get('shard_size')} run one after another) + exact "
            f"support pre-filter + v1 contrastive stage ({p.get('transfer_target')}, locked profile); v1 keeps at most 20 programs"
        )
    raise ValueError(f"unknown method {method!r}")


FMG_STAGE_NOTES = {
    "per_episode_mining": (
        "v1 run_mimic_experiment.py --stop-after-contrastive-cache, one subprocess per shard (includes its "
        "interpreter start-up), compact merge of each shard cache"
    ),
    "support_prefilter": "fmg_prefilter support counts on the mined TRAIN episodes, filtered episodes loaded into memory",
    "contrastive_stage": "fmg_adapter.run_fmg_contrastive_stage x2 (without / with transfer target) + ProgramRecord conversion",
}

CSV_COLUMNS = (
    "method",
    "stage",
    "fraction",
    "time_s",
    "peak_rss_mb",
    "child_peak_rss_mb",
    "n_train_episodes",
    "n_candidates",
    "notes",
    "status",
    "n_train_patients",
    "n_train_positives",
    "n_train_controls",
    "n_mined_raw",
    "self_peak_rss_mb",
    "rss_at_start_mb",
    "peak_private_mb",
    "cpu_s",
    "load_s",
    "represent_s",
    "mine_s",
    "rank_s",
)
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
CODE_FILES = (
    "scripts/v2/run_scalability_benchmark.py",
    "scripts/v2/run_fmg_mining_sharded.py",
    "scripts/run_mimic_experiment.py",
    "Non-stationary time series.py",
    "src/icu/icu_eventizer.py",
    "src/icu/shock_contrastive_miner.py",
    "src/icu/v2/fmg_adapter.py",
    "src/icu/v2/fmg_prefilter.py",
    "src/icu/v2/pools.py",
    "src/icu/v2/events.py",
    "src/icu/v2/patterns.py",
    "src/icu/v2/baselines/spmf.py",
    "src/icu/v2/baselines/motif.py",
    "src/icu/v2/baselines/shapelet.py",
)


# ------------------------------------------------------------------------------ small helpers


def _rel(path: str | Path) -> str:
    """Path relative to the repo root when possible (for readable outputs)."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(lines: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def json_ready(value: Any) -> Any:
    """Strict-JSON view: non-finite floats -> None, sets -> sorted lists, numpy scalars and Paths -> plain."""
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(json_ready(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, payload: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(json_ready(payload), indent=indent, allow_nan=False, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, path)


def safe_rmtree(path: Path, inside: Path) -> None:
    """``shutil.rmtree(path)`` only if ``path`` lies inside ``inside`` (guards against typos)."""
    path, inside = Path(path).resolve(), Path(inside).resolve()
    if path == inside or inside in path.parents:
        shutil.rmtree(path, ignore_errors=True)
    else:
        raise ValueError(f"refusing to delete {path}: not inside {inside}")


def _tail(path: Path, n_lines: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])


def load_script_module(name: str) -> ModuleType:
    """Import ``scripts/v2/<name>.py`` by path (``scripts`` is not a regular package)."""
    module_name = f"_scalability_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = SCRIPT.parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------- memory sampling


@dataclass(frozen=True)
class MemorySample:
    """One poll: wall-clock time and bytes of the root process and of all its descendants."""

    t: float
    self_rss: int
    child_rss: int
    self_private: int
    child_private: int
    n_children: int


def _private_bytes(info: Any) -> int:
    """Private committed bytes (Windows ``PrivateUsage``); falls back to the RSS elsewhere."""
    value = getattr(info, "private", None)
    return int(value if value is not None else info.rss)


def _os_peak_bytes(info: Any) -> int | None:
    """OS-tracked lifetime peak working set (Windows only)."""
    value = getattr(info, "peak_wset", None)
    return None if value is None else int(value)


class ProcessTreeSampler:
    """Poll the RSS of one process and of all its descendants from a background thread.

    ``rss`` is the resident set size (the working set on Windows). ``private`` is the
    private committed memory on Windows (``PrivateUsage``) and the RSS elsewhere.
    Descendants are re-enumerated on every poll, so a child that lives for at least one
    interval (the SPMF JVM, a v1 mining shard) is always seen. For every child the last
    seen CPU time and the OS-tracked peak working set are kept as well, and
    ``on_new_child`` (if given) is called once per new descendant, e.g. to apply the CPU
    placement of the job (Windows children inherit the affinity but not the priority).
    """

    def __init__(
        self,
        pid: int | None = None,
        interval: float = 0.1,
        on_new_child: Callable[[psutil.Process], str | None] | None = None,
    ) -> None:
        interval = float(interval)
        if not 0.0 < interval <= 5.0:
            raise ValueError(f"interval must be in (0, 5] seconds, got {interval!r}")
        self.pid = os.getpid() if pid is None else int(pid)
        self.interval = interval
        self.on_new_child = on_new_child
        self.samples: list[MemorySample] = []
        self.children: dict[tuple[int, float], dict[str, Any]] = {}
        self._root = psutil.Process(self.pid)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> MemorySample | None:
        now = time.time()
        try:
            info = self._root.memory_info()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
        try:
            children = self._root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []
        child_rss = child_private = n_children = 0
        for child in children:
            try:
                with child.oneshot():
                    child_info = child.memory_info()
                    cpu = child.cpu_times()
                    key = (child.pid, child.create_time())
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
                continue
            n_children += 1
            child_rss += int(child_info.rss)
            child_private += _private_bytes(child_info)
            record = self.children.get(key)
            if record is None:
                try:
                    name = child.name()
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    name = "?"
                record = {"pid": child.pid, "name": name, "first_seen": now, "peak_rss_mb": 0.0, "os_peak_wset_mb": None}
                if self.on_new_child is not None:
                    record["placement_error"] = self.on_new_child(child)
                self.children[key] = record
            record["last_seen"] = now
            record["cpu_s"] = float(cpu.user + cpu.system)
            record["peak_rss_mb"] = max(record["peak_rss_mb"], child_info.rss / MIB)
            peak = _os_peak_bytes(child_info)
            if peak is not None:
                record["os_peak_wset_mb"] = max(record["os_peak_wset_mb"] or 0.0, peak / MIB)
        sample = MemorySample(now, int(info.rss), child_rss, _private_bytes(info), child_private, n_children)
        self.samples.append(sample)
        return sample

    def _run(self) -> None:
        deadline = time.perf_counter()
        while not self._stop.is_set():
            self.sample_once()
            deadline += self.interval
            delay = deadline - time.perf_counter()
            if delay < 0.0:  # polling slower than the interval: restart the schedule
                deadline, delay = time.perf_counter(), 0.0
            self._stop.wait(delay)

    def start(self) -> ProcessTreeSampler:
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> ProcessTreeSampler:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample_once()
        return self

    def __enter__(self) -> ProcessTreeSampler:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def summarize_samples(
    samples: Sequence[MemorySample],
    t0: float | None = None,
    t1: float | None = None,
) -> dict[str, Any]:
    """Peaks over the samples taken in ``[t0, t1]`` (MiB; ``None`` bounds are open).

    A window shorter than one interval may hold no sample; the two samples around it are
    then used and ``n_samples`` is 0. ``rss_at_start_mb`` is the first sample of the window.
    """
    chosen = [s for s in samples if (t0 is None or s.t >= t0) and (t1 is None or s.t <= t1)]
    n_samples = len(chosen)
    if not chosen:
        before = [s for s in samples if t0 is not None and s.t < t0]
        after = [s for s in samples if t1 is not None and s.t > t1]
        chosen = before[-1:] + after[:1]
    if not chosen:
        return {
            "n_samples": 0,
            "peak_rss_mb": None,
            "child_peak_rss_mb": None,
            "self_peak_rss_mb": None,
            "peak_private_mb": None,
            "rss_at_start_mb": None,
            "max_children": None,
        }
    return {
        "n_samples": n_samples,
        "peak_rss_mb": max(s.self_rss + s.child_rss for s in chosen) / MIB,
        "child_peak_rss_mb": max(s.child_rss for s in chosen) / MIB,
        "self_peak_rss_mb": max(s.self_rss for s in chosen) / MIB,
        "peak_private_mb": max(s.self_private + s.child_private for s in chosen) / MIB,
        "rss_at_start_mb": (chosen[0].self_rss + chosen[0].child_rss) / MIB,
        "max_children": max(s.n_children for s in chosen),
    }


def children_in_window(children: Mapping[Any, Mapping[str, Any]], t0: float, t1: float) -> list[dict[str, Any]]:
    """Children whose observed lifetime overlaps ``[t0, t1]``."""
    return [dict(c) for c in children.values() if c["first_seen"] <= t1 and c.get("last_seen", c["first_seen"]) >= t0]


def downsample_trace(samples: Sequence[MemorySample], origin: float, bucket_s: float = 1.0) -> list[list[float]]:
    """Per ``bucket_s`` the sample with the largest total RSS: ``[t - origin, total_mb, child_mb]``."""
    buckets: dict[int, MemorySample] = {}
    for sample in samples:
        key = int((sample.t - origin) // bucket_s)
        best = buckets.get(key)
        if best is None or sample.self_rss + sample.child_rss > best.self_rss + best.child_rss:
            buckets[key] = sample
    return [
        [round(s.t - origin, 2), round((s.self_rss + s.child_rss) / MIB, 1), round(s.child_rss / MIB, 1)]
        for _, s in sorted(buckets.items())
    ]


# ------------------------------------------------------------------------------- subsampling


def load_cohort(path: str | Path) -> list[dict[str, Any]]:
    """Episodes of an ``episodes_split.json`` as plain dicts (id, subject, label, split)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    episodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in seen:
            raise ValueError(f"duplicate episode_id {episode_id!r} in {path}")
        seen.add(episode_id)
        label = row.get("label")
        if label is None:
            label = 1 if row.get("episode_type") == "positive" else 0
        if int(label) not in (0, 1):
            raise ValueError(f"episode {episode_id!r} has label {label!r}")
        episodes.append(
            {"episode_id": episode_id, "subject_id": int(row["subject_id"]), "label": int(label), "split": str(row.get("split", ""))}
        )
    return episodes


def check_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    values = sorted({float(f) for f in fractions})
    if not values or any(not math.isfinite(f) or not 0.0 < f <= 1.0 for f in values):
        raise ValueError(f"fractions must be in (0, 1], got {list(fractions)}")
    return tuple(values)


def nested_train_subsamples(
    episodes: Sequence[Mapping[str, Any]],
    fractions: Sequence[float],
    seed: int = 0,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Patient-level, nested, seeded subsamples of the ``split`` patients.

    Patients are ranked once by :func:`sample_patient_ids` (sha256 of seed and subject
    id), and fraction ``f`` keeps the first ``round(f * N)`` (at least 1) with every one of
    their ``split`` episodes, so smaller fractions are prefixes of larger ones and the
    result does not depend on the order of ``episodes``.
    """
    from src.icu.v2.cohort import sample_patient_ids

    fractions = check_fractions(fractions)
    train = [dict(e) for e in episodes if e["split"] == split]
    if not train:
        raise ValueError(f"no {split!r} episodes in the cohort")
    patients = sorted({int(e["subject_id"]) for e in train})
    n_total = len(patients)
    out: list[dict[str, Any]] = []
    previous: frozenset[int] = frozenset()
    for fraction in fractions:
        n = n_total if fraction >= 1.0 else max(1, int(math.floor(fraction * n_total + 0.5)))
        chosen = sample_patient_ids(patients, n, seed)
        if not previous <= chosen:  # cannot happen with one ranking; kept as a guard
            raise AssertionError("patient subsamples are not nested")
        previous = chosen
        kept = sorted((e for e in train if int(e["subject_id"]) in chosen), key=lambda e: e["episode_id"])
        out.append(
            {
                "fraction": fraction,
                "seed": int(seed),
                "n_patients_total": n_total,
                "n_patients": len(chosen),
                "n_episodes": len(kept),
                "n_positives": sum(1 for e in kept if e["label"] == 1),
                "n_controls": sum(1 for e in kept if e["label"] == 0),
                "patients_sha256": _sha256_text([str(p) for p in sorted(chosen)]),
                "episodes_sha256": _sha256_text([e["episode_id"] for e in kept]),
                "patient_ids": sorted(chosen),
                "episodes": kept,
            }
        )
    return out


# ------------------------------------------------------------------------------- job process


class PhaseClock:
    """Timed phases of a job: wall-clock bounds (``time.time``, shared with the driver's
    sampler), ``perf_counter`` durations, and this process's CPU time."""

    def __init__(self) -> None:
        self.phases: list[dict[str, Any]] = []
        self._process = psutil.Process()

    def _cpu(self) -> float:
        times = self._process.cpu_times()
        return float(times.user + times.system)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        cpu0, t0, p0 = self._cpu(), time.time(), time.perf_counter()
        try:
            yield
        finally:
            self.phases.append(
                {
                    "name": name,
                    "t0": t0,
                    "t1": time.time(),
                    "duration_s": time.perf_counter() - p0,
                    "self_cpu_s": self._cpu() - cpu0,
                }
            )


def _train_view(spec: Mapping[str, Any]) -> tuple[list[str], dict[str, int], dict[str, int]]:
    ids = [str(e["episode_id"]) for e in spec["train"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episode ids in the job spec")
    labels = {str(e["episode_id"]): int(e["label"]) for e in spec["train"]}
    groups = {str(e["episode_id"]): int(e["subject_id"]) for e in spec["train"]}
    return ids, labels, groups


def run_spmf_job(spec: Mapping[str, Any], clock: PhaseClock) -> dict[str, Any]:
    """``pools.build_symbolic_pool`` on TRAIN, phase by phase, without firing sets."""
    from src.icu.v2.baselines.spmf import SPMF_VERSION
    from src.icu.v2.patterns import dedup_programs, rank_by_contrast
    from src.icu.v2.pools import SPMF_MINERS, build_sequences, load_arrays

    method = str(spec["method"])
    params = dict(spec["params"])
    pool_cap = int(params.pop("pool_cap"))
    timeout_s = float(params.pop("timeout_s", 7200))
    miner = SPMF_MINERS[method]
    ids, labels, _ = _train_view(spec)
    spmf_tmp = Path(spec["tmp_dir"]) / "spmf"
    with clock.phase("load"):
        arrays = load_arrays(spec["episode_dir"], ids)
    with clock.phase("represent"):
        sequences, _, _ = build_sequences(arrays, ids, **spec["representation"])
    positives = [sequences[i] for i in ids if labels[i] == 1]
    with clock.phase("mine"):
        mined = miner(positives, **params, timeout_s=timeout_s, tmp_root=spmf_tmp)
    with clock.phase("rank"):
        ranked = rank_by_contrast(mined, sequences, labels)
        candidates = dedup_programs(ranked)[:pool_cap]
    return {
        "n_train_episodes": len(ids),
        "n_positive_sequences": len(positives),
        "n_mined_raw": len(mined),
        "n_candidates": len(candidates),
        "n_tokens": len({token for seq in sequences.values() for itemset in seq for token in itemset}),
        "top_candidate_ids": [program.program_id for program in candidates[:5]],
        "spmf_version": SPMF_VERSION,
    }


def run_numeric_job(spec: Mapping[str, Any], clock: PhaseClock) -> dict[str, Any]:
    """``pools.build_numeric_pool`` on TRAIN without firing sets."""
    from src.icu.v2.baselines.motif import mine_motif_programs
    from src.icu.v2.baselines.shapelet import mine_shapelet_programs
    from src.icu.v2.pools import load_arrays

    method = str(spec["method"])
    params = dict(spec["params"])
    lengths = tuple(int(m) for m in params.pop("lengths"))
    top_k = int(params.pop("top_k_per_feature"))
    ids, labels, groups = _train_view(spec)
    with clock.phase("load"):
        arrays = load_arrays(spec["episode_dir"], ids)
    with clock.phase("mine"):
        if method == "matrix_profile":
            programs = mine_motif_programs(arrays, labels, groups=groups, lengths=lengths, top_k_per_feature=top_k, **params)
        else:
            programs = mine_shapelet_programs(arrays, labels, lengths=lengths, top_k_per_feature=top_k, **params)
    return {
        "n_train_episodes": len(ids),
        "n_discovery_episodes": sum(labels.values()) if method == "matrix_profile" else len(ids),
        "n_candidates": len(programs),
        "n_features_with_programs": len({next(iter(p.features)) for p in programs}),
        "top_candidate_ids": [program.program_id for program in programs[:5]],
    }


def run_fmg_job(spec: Mapping[str, Any], clock: PhaseClock) -> dict[str, Any]:
    """v1 per-episode mining (sharded, one worker) + support pre-filter + v1 contrastive stage."""
    from src.icu.v2 import fmg_adapter as fa
    from src.icu.v2.fmg_prefilter import (
        compact_episode,
        count_rule_support,
        iter_filtered_inputs,
        kept_rule_ids,
        prefilter_thresholds,
    )

    sharded = load_script_module("run_fmg_mining_sharded")
    params = dict(spec["params"])
    shard_size = int(params["shard_size"])
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    two_pass = str(params.get("transfer_target", "two-pass")) == "two-pass"
    compact = bool(params.get("compact_memory", True))
    domain = str(params.get("domain", "mimic_iv"))
    miner_config = params.get("miner_config")
    if miner_config in (None, "v1_locked_reference_profile"):
        requested = fa.v1_locked_reference_profile()
    else:
        requested = dict(miner_config)
    # Untimed set-up: resolving the profile also imports the v1 modules once.
    min_positive, min_control = prefilter_thresholds(fa.resolve_v1_profile(requested))
    ids, _, _ = _train_view(spec)
    fmg_dir = Path(spec["fmg_dir"])
    tmp = Path(spec["tmp_dir"])
    parts_dir, shards_root = tmp / "parts", tmp / "shards"
    parts_dir.mkdir(parents=True, exist_ok=True)
    python = str(spec.get("python") or sys.executable)

    shard_reports: list[dict[str, Any]] = []
    parts: list[Path] = []
    n_mined_episodes = 0
    with clock.phase("per_episode_mining"):
        index = json.loads((fmg_dir / "shock_episode_index.json").read_text(encoding="utf-8"))
        by_id = {str(meta["episode_id"]): meta for meta in index}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(f"{len(missing)} TRAIN episodes are not in the FMG view index, e.g. {missing[:3]}")
        metas = [by_id[i] for i in ids]
        del index, by_id
        shards = sharded.shard_metas(metas, max(1, math.ceil(len(metas) / shard_size)))
        for k, shard in enumerate(shards):
            shard_root = shards_root / f"shard_{k:03d}"
            started = time.perf_counter()
            shard_input = sharded.prepare_shard_input(fmg_dir, shard_root / "input", shard)
            prepare_s = time.perf_counter() - started
            report = sharded.run_shard(python, shard_input, shard_root / "out")
            cache = shard_root / "out" / "contrastive_inputs.json"
            if report["returncode"] != 0 or not report["manifest"] or not cache.is_file():
                raise RuntimeError(
                    f"v1 mining failed on shard {k} (exit code {report['returncode']}):\n{_tail(shard_root / 'out' / 'shard.log')}"
                )
            cache_bytes = cache.stat().st_size
            part = parts_dir / f"part_{k:03d}.json"
            started = time.perf_counter()
            merged = sharded.merge_shard_caches([cache], part)
            merge_s = time.perf_counter() - started
            started = time.perf_counter()
            safe_rmtree(shard_root, tmp)  # links, indented cache, per-episode result files
            delete_s = time.perf_counter() - started
            parts.append(part)
            n_mined_episodes += int(merged["n_episodes"])
            shard_reports.append(
                {
                    "shard": k,
                    "n_episodes": len(shard),
                    "n_mined": int(merged["n_episodes"]),
                    "prepare_s": prepare_s,
                    "v1_process_s": float(report["elapsed_seconds"]),
                    "merge_s": merge_s,
                    "delete_s": delete_s,
                    "indented_cache_mb": cache_bytes / MIB,
                    "compact_part_mb": part.stat().st_size / MIB,
                }
            )

    with clock.phase("support_prefilter"):
        started = time.perf_counter()
        positives: Counter = Counter()
        controls: Counter = Counter()
        counted: Counter = Counter()
        for part in parts:
            part_positives, part_controls, part_episodes = count_rule_support(part)
            positives.update(part_positives)
            controls.update(part_controls)
            counted.update(part_episodes)
            del part_positives, part_controls
        keep = kept_rule_ids(positives, controls, min_positive, min_control)
        n_distinct_rules = len(positives.keys() | controls.keys())
        del positives, controls
        count_s = time.perf_counter() - started
        started = time.perf_counter()
        inputs: list[dict] = []
        feature_cache: dict = {}
        for part in parts:
            for item in iter_filtered_inputs(part, keep):
                inputs.append(compact_episode(item, feature_cache) if compact else item)
        load_filtered_s = time.perf_counter() - started
        n_rule_records_kept = sum(len(item["rules"]) for item in inputs)

    with clock.phase("contrastive_stage"):
        mining_inputs = fa.filter_contrastive_inputs(inputs, set(ids))
        started = time.perf_counter()
        if two_pass:
            first = fa.run_fmg_contrastive_stage(mining_inputs, fa.without_transfer_target(requested), domain=domain)
            final_config = fa.transfer_target_config(requested, first)
        else:
            final_config = dict(requested)
        first_pass_s = time.perf_counter() - started
        started = time.perf_counter()
        result = fa.run_fmg_contrastive_stage(mining_inputs, final_config, domain=domain)
        records = fa.fmg_programs_to_records(result.programs)
        final_pass_s = time.perf_counter() - started

    return {
        "n_train_episodes": len(ids),
        "n_mined_episodes": n_mined_episodes,
        "n_shards": len(shard_reports),
        "shard_size": shard_size,
        "n_mined_raw": n_distinct_rules,
        "n_distinct_rules_kept": len(keep),
        "n_rule_records_kept": n_rule_records_kept,
        "prefilter_thresholds": [min_positive, min_control],
        "prefilter_episode_counts": dict(counted),
        "n_scored_rules": int(result.rule_count),
        "n_candidates": len(records),
        "top_candidate_ids": [record.program_id for record in records[:10]],
        "gates_overall_pass": bool(result.gates.get("overall_pass", False)),
        "compact_parts_mb": sum(r["compact_part_mb"] for r in shard_reports),
        "indented_caches_mb": sum(r["indented_cache_mb"] for r in shard_reports),
        "substeps_s": {
            "shard_prepare": sum(r["prepare_s"] for r in shard_reports),
            "v1_processes": sum(r["v1_process_s"] for r in shard_reports),
            "merge": sum(r["merge_s"] for r in shard_reports),
            "delete_shard_files": sum(r["delete_s"] for r in shard_reports),
            "prefilter_count": count_s,
            "prefilter_load_filtered": load_filtered_s,
            "contrastive_first_pass": first_pass_s,
            "contrastive_final_pass": final_pass_s,
        },
        "transfer_target": "two-pass" if two_pass else "off",
        "shards": shard_reports,
    }


JOB_RUNNERS: dict[str, Callable[[Mapping[str, Any], PhaseClock], dict[str, Any]]] = {
    "prefixspan": run_spmf_job,
    "erminer": run_spmf_job,
    "matrix_profile": run_numeric_job,
    "shapelet": run_numeric_job,
    FMG_METHOD: run_fmg_job,
}


def _threadpools() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    return [
        {"api": info.get("internal_api"), "num_threads": info.get("num_threads"), "version": info.get("version")}
        for info in threadpool_info()
    ]


def job_main(spec_path: Path, result_path: Path) -> int:
    """Run one measured job; the driver samples this process tree from outside."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    method = str(spec["method"])
    result: dict[str, Any] = {"method": method, "fraction": spec["fraction"], "status": "error", "pid": os.getpid()}
    clock = PhaseClock()
    try:
        runner = JOB_RUNNERS[method]
        import numpy  # noqa: F401  (loads the BLAS so threadpools can be reported)

        result["threadpools"] = _threadpools()
        result.update(runner(spec, clock))
        result["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001 - reported to the driver, which records the failure
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    result["phases"] = clock.phases
    me = psutil.Process()
    info = me.memory_info()
    peak = _os_peak_bytes(info)
    result["self_os_peak_wset_mb"] = None if peak is None else peak / MIB
    try:
        result["cpu_affinity"] = me.cpu_affinity()
        result["priority"] = me.nice()
    except (AttributeError, psutil.Error, OSError):
        pass
    write_json_atomic(Path(result_path), result)
    return 0 if result["status"] == "ok" else 1


# --------------------------------------------------------------------------------- the driver


def _kill_tree(pid: int) -> None:
    try:
        root = psutil.Process(pid)
        processes = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for process in processes:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(processes, timeout=30)


def job_env(blas_threads: int) -> dict[str, str]:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="")
    if blas_threads > 0:
        env.update({name: str(int(blas_threads)) for name in THREAD_ENV_VARS})
    return env


# ------------------------------------------------------------------------------ CPU placement

PRIORITY_CLASS_NAMES = {
    "normal": "NORMAL_PRIORITY_CLASS",
    "above_normal": "ABOVE_NORMAL_PRIORITY_CLASS",
    "high": "HIGH_PRIORITY_CLASS",
}


def windows_cpu_sets() -> list[dict[str, int]] | None:
    """Logical processors with core index and efficiency class (Windows CPU sets API).

    On hybrid Intel CPUs the performance cores have the higher ``efficiency_class``.
    Returns None elsewhere or when the API is unavailable.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        import struct
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        length = wintypes.ULONG(0)
        kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(length), None, 0)
        if not length.value:
            return None
        buffer = ctypes.create_string_buffer(length.value)
        if not kernel32.GetSystemCpuSetInformation(buffer, length, ctypes.byref(length), None, 0):
            return None
        rows, offset = [], 0
        while offset < length.value:
            size, kind = struct.unpack_from("<II", buffer.raw, offset)
            if size == 0:
                break
            if kind == 0:  # CpuSetInformation
                _, group, logical, core, _, _, efficiency, _ = struct.unpack_from("<IHBBBBBB", buffer.raw, offset + 8)
                rows.append({"group": group, "logical": logical, "core": core, "efficiency_class": efficiency})
            offset += size
        return rows or None
    except (OSError, AttributeError, ValueError):
        return None


def parse_cpu_list(text: str) -> list[int]:
    """``"0-3,6"`` -> ``[0, 1, 2, 3, 6]``."""
    cpus: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        low, _, high = part.partition("-")
        start, stop = int(low), int(high) if high else int(low)
        if start < 0 or stop < start:
            raise ValueError(f"bad CPU range {part!r}")
        cpus.update(range(start, stop + 1))
    if not cpus:
        raise ValueError(f"empty CPU list {text!r}")
    return sorted(cpus)


def _ranges(cpus: Sequence[int]) -> str:
    """``[0, 1, 2, 5]`` -> ``"0-2,5"`` (inverse of :func:`parse_cpu_list`)."""
    runs: list[list[int]] = []
    for cpu in sorted(set(int(c) for c in cpus)):
        if runs and cpu == runs[-1][1] + 1:
            runs[-1][1] = cpu
        else:
            runs.append([cpu, cpu])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def resolve_affinity(spec: str) -> tuple[list[int] | None, str]:
    """CPU list for ``--cpu-affinity`` and a description; None means "not pinned".

    ``performance`` pins to the logical CPUs of the highest efficiency class on hybrid
    CPUs (P-cores of Intel 12th-14th gen) and does nothing on homogeneous CPUs.
    """
    spec = str(spec).strip().lower()
    if spec in ("", "all", "none"):
        return None, "not pinned"
    if spec == "performance":
        sets = windows_cpu_sets()
        if not sets or any(row["group"] != 0 for row in sets):
            return None, "not pinned (CPU topology unavailable)"
        classes = {row["efficiency_class"] for row in sets}
        if len(classes) < 2:
            return None, "not pinned (all cores have the same efficiency class)"
        top = max(classes)
        cpus = sorted(row["logical"] for row in sets if row["efficiency_class"] == top)
        return cpus, f"performance cores only (efficiency class {top}): logical CPUs {_ranges(cpus)}"
    cpus = parse_cpu_list(spec)
    return cpus, f"logical CPUs {_ranges(cpus)}"


def priority_value(name: str) -> int | None:
    """psutil priority class for ``--priority`` on Windows; None = leave unchanged."""
    if name not in PRIORITY_CLASS_NAMES:
        raise ValueError(f"unknown priority {name!r}; choose from {sorted(PRIORITY_CLASS_NAMES)}")
    if sys.platform != "win32" or name == "normal":
        return None
    return int(getattr(psutil, PRIORITY_CLASS_NAMES[name]))


def apply_placement(process: psutil.Process, cpus: Sequence[int] | None, priority: int | None) -> str | None:
    """Set the affinity and priority class of one process; returns an error text or None."""
    errors = []
    if cpus is not None:
        try:
            process.cpu_affinity(list(cpus))
        except (psutil.Error, OSError, ValueError) as exc:
            errors.append(f"affinity: {exc}")
    if priority is not None:
        try:
            process.nice(priority)
        except (psutil.Error, OSError, ValueError) as exc:
            errors.append(f"priority: {exc}")
    return "; ".join(errors) or None


def estimated_disk_need_gb(method: str, n_episodes: int, params: Mapping[str, Any]) -> float:
    """Rough temporary disk need: FMG compact parts (~1.6 MB/episode) plus one indented shard cache."""
    if method != FMG_METHOD:
        return 0.2
    shard = min(n_episodes, int(params.get("shard_size", 220)))
    return (1.6 * n_episodes + 2.8 * shard) / 1024.0 + 0.2


def estimated_ram_need_gb(method: str, n_episodes: int) -> float:
    """Rough peak RSS of one job in GB, only used to wait for enough free RAM before starting it.

    FMG-Bridge holds the pre-filtered rule records of every TRAIN episode in its contrastive
    stage (2.4 MB per episode at the peak in a 928-episode pilot, growing slowly with the
    cohort; 2.6 MB assumed). SPMF jobs run a JVM of at most 2 GB next to the ranking
    structures.
    """
    n = max(0, int(n_episodes))
    if method == FMG_METHOD:
        return 0.5 + 2.6 * n / 1024.0
    if method in SPMF_METHODS:
        return 0.5 + 2.0 * min(1.0, n / 1000.0) + 0.5 * n / 1024.0
    return 0.5 + 0.3 * n / 1024.0


def wait_for_resources(
    need_ram_gb: float,
    need_disk_gb: float,
    *,
    min_free_ram_gb: float,
    min_free_disk_gb: float,
    disk_path: Path,
    max_wait_s: float,
    poll_s: float,
    label: str,
) -> tuple[bool, float, str]:
    """Wait until RAM and disk allow a job (other experiments share the machine).

    Returns ``(admitted, waited_seconds, what_was_missing)``. The job needs
    ``need_ram_gb + min_free_ram_gb`` of available RAM and must leave ``min_free_disk_gb``
    of disk after ``need_disk_gb`` of temporary files.
    """
    started = time.time()
    last_report = -math.inf
    while True:
        available = psutil.virtual_memory().available / GIB
        free = shutil.disk_usage(disk_path).free / GIB
        missing = []
        if available < need_ram_gb + min_free_ram_gb:
            missing.append(f"RAM available {available:.1f} GB < {need_ram_gb + min_free_ram_gb:.1f} GB")
        if free - need_disk_gb < min_free_disk_gb:
            missing.append(f"disk free {free:.1f} GB minus ~{need_disk_gb:.1f} GB < {min_free_disk_gb} GB")
        waited = time.time() - started
        if not missing:
            return True, waited, ""
        if waited >= max_wait_s:
            return False, waited, "; ".join(missing)
        if time.time() - last_report >= 300.0:
            print(f"[wait] {label}: {'; '.join(missing)} (waited {waited:.0f} s)", flush=True)
            last_report = time.time()
        time.sleep(max(0.1, min(poll_s, max_wait_s - waited)))


def run_job(
    spec: Mapping[str, Any],
    *,
    job_dir: Path,
    log_path: Path,
    python: str,
    env: Mapping[str, str],
    interval: float,
    timeout_s: float,
    min_free_ram_gb: float,
    min_free_disk_gb: float,
    disk_path: Path,
    cpus: Sequence[int] | None = None,
    priority: int | None = None,
) -> dict[str, Any]:
    """Run one job process under the memory sampler and the watchdog; return its record.

    ``cpus`` / ``priority`` are applied to the job process right after it starts and to
    every descendant as soon as the sampler sees it.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path, result_path = job_dir / "job_spec.json", job_dir / "job_result.json"
    write_json_atomic(spec_path, spec, indent=None)
    command = [python, str(SCRIPT), "--job-spec", str(spec_path), "--job-result", str(result_path)]
    status, reason = "ok", ""
    cpu_percent: list[float] = []
    min_available = math.inf
    min_disk_free = math.inf
    started = time.time()
    psutil.cpu_percent(None)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=dict(env), stdout=log, stderr=subprocess.STDOUT)
        placement_error = apply_placement(psutil.Process(process.pid), cpus, priority)
        sampler = ProcessTreeSampler(
            process.pid, interval, on_new_child=lambda child: apply_placement(child, cpus, priority)
        ).start()
        try:
            while True:
                try:
                    returncode = process.wait(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    pass
                available = psutil.virtual_memory().available
                disk_free = shutil.disk_usage(disk_path).free
                min_available = min(min_available, available)
                min_disk_free = min(min_disk_free, disk_free)
                cpu_percent.append(psutil.cpu_percent(None))
                if time.time() - started > timeout_s:
                    reason = f"timeout after {timeout_s:.0f} s"
                elif available < min_free_ram_gb * GIB:
                    reason = f"system RAM available {available / GIB:.2f} GB < {min_free_ram_gb} GB"
                elif disk_free < min_free_disk_gb * GIB:
                    reason = f"disk free {disk_free / GIB:.1f} GB < {min_free_disk_gb} GB"
                if reason:
                    _kill_tree(process.pid)
                    returncode = process.wait()
                    status = "aborted"
                    break
        except BaseException:
            _kill_tree(process.pid)
            raise
        finally:
            sampler.stop()
    ended = time.time()
    job: dict[str, Any] = {}
    if result_path.is_file():
        job = json.loads(result_path.read_text(encoding="utf-8"))
    if status == "ok" and (returncode != 0 or job.get("status") != "ok"):
        status = "failed"
        reason = str(job.get("error") or f"job exited with code {returncode}; see {_rel(log_path)}")

    phases = job.get("phases") or []
    if phases:
        t0, t1 = min(p["t0"] for p in phases), max(p["t1"] for p in phases)
    else:
        t0, t1 = started, ended
    memory = {
        "discovery": summarize_samples(sampler.samples, t0, t1),
        "job_lifetime": summarize_samples(sampler.samples),
        "phases": {p["name"]: summarize_samples(sampler.samples, p["t0"], p["t1"]) for p in phases},
        "children": children_in_window(sampler.children, started, ended),
        "sampler": {"interval_s": interval, "n_samples": len(sampler.samples)},
    }
    return {
        "status": status,
        "reason": reason,
        "returncode": returncode,
        "job_wall_s": ended - started,
        "job": job,
        "memory": memory,
        "system": {
            "cpu_percent_mean": sum(cpu_percent) / len(cpu_percent) if cpu_percent else None,
            "cpu_percent_max": max(cpu_percent) if cpu_percent else None,
            "available_ram_min_gb": None if min_available == math.inf else min_available / GIB,
            "disk_free_min_gb": None if min_disk_free == math.inf else min_disk_free / GIB,
        },
        "trace": downsample_trace(sampler.samples, t0),
        "discovery_window": [t0, t1],
        "placement": {
            "cpus": None if cpus is None else list(cpus),
            "priority_class": priority,
            "job_error": placement_error,
            "child_errors": sorted({c["placement_error"] for c in memory["children"] if c.get("placement_error")}),
        },
        "log": _rel(log_path),
    }


def _round(value: Any, digits: int) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    return value


def _children_cpu(record: Mapping[str, Any], t0: float, t1: float) -> float:
    return sum(float(c.get("cpu_s", 0.0)) for c in children_in_window({i: c for i, c in enumerate(record["memory"]["children"])}, t0, t1))


def rows_for_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """CSV rows of one job: the discovery total, plus one row per FMG stage."""
    job = record.get("job") or {}
    phases = job.get("phases") or []
    ok = record["status"] == "ok"
    subsample = record["subsample"]
    method = record["method"]
    base = {
        "method": method,
        "fraction": record["fraction"],
        "n_train_episodes": subsample["n_episodes"],
        "n_train_patients": subsample["n_patients"],
        "n_train_positives": subsample["n_positives"],
        "n_train_controls": subsample["n_controls"],
        "status": record["status"],
    }
    failure = "" if ok else f" [{record['status']}: {record.get('reason', '')}]"
    discovery = record["memory"]["discovery"]
    t0, t1 = record["discovery_window"]
    total = {
        **base,
        "stage": TOTAL_STAGE,
        "time_s": _round(sum(p["duration_s"] for p in phases), 3) if ok else None,
        "peak_rss_mb": _round(discovery["peak_rss_mb"], 1),
        "child_peak_rss_mb": _round(discovery["child_peak_rss_mb"], 1),
        "self_peak_rss_mb": _round(discovery["self_peak_rss_mb"], 1),
        "rss_at_start_mb": _round(discovery["rss_at_start_mb"], 1),
        "peak_private_mb": _round(discovery["peak_private_mb"], 1),
        "cpu_s": _round(sum(p["self_cpu_s"] for p in phases) + _children_cpu(record, t0, t1), 3) if ok else None,
        "n_candidates": job.get("n_candidates"),
        "n_mined_raw": job.get("n_mined_raw"),
        "notes": method_note(method, record.get("params") or DEFAULT_PARAMS[method]) + failure,
    }
    for phase in phases:
        total[f"{phase['name']}_s"] = _round(phase["duration_s"], 3)
    rows = [total]
    if method == FMG_METHOD:
        by_name = {p["name"]: p for p in phases}
        for stage in FMG_STAGES:
            phase = by_name.get(stage)
            memory = record["memory"]["phases"].get(stage) or {}
            child_cpu = _children_cpu(record, phase["t0"], phase["t1"]) if phase else 0.0
            rows.append(
                {
                    **base,
                    "stage": stage,
                    "time_s": _round(phase["duration_s"], 3) if phase and ok else None,
                    "peak_rss_mb": _round(memory.get("peak_rss_mb"), 1),
                    "child_peak_rss_mb": _round(memory.get("child_peak_rss_mb"), 1),
                    "self_peak_rss_mb": _round(memory.get("self_peak_rss_mb"), 1),
                    "rss_at_start_mb": _round(memory.get("rss_at_start_mb"), 1),
                    "peak_private_mb": _round(memory.get("peak_private_mb"), 1),
                    "cpu_s": _round(phase["self_cpu_s"] + child_cpu, 3) if phase and ok else None,
                    "n_candidates": job.get("n_candidates") if stage == "contrastive_stage" else None,
                    "n_mined_raw": job.get("n_mined_raw") if stage == "per_episode_mining" else None,
                    "notes": FMG_STAGE_NOTES[stage] + failure,
                }
            )
    return rows


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> list[str]:
    """Rows to CSV: the fixed ``CSV_COLUMNS`` first, then any other keys (sorted)."""
    extras = sorted({key for row in rows for key in row} - set(CSV_COLUMNS))
    columns = list(CSV_COLUMNS) + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})
    os.replace(partial, path)
    return columns


def write_outputs(out_dir: Path, meta: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``scalability.csv``, ``scalability.json`` and ``memory_traces.json``; returns the rows."""
    rows = [row for record in records for row in rows_for_record(record)]
    write_csv(rows, out_dir / "scalability.csv")
    jobs = [{key: value for key, value in record.items() if key != "trace"} for record in records]
    write_json_atomic(out_dir / "scalability.json", {"meta": meta, "rows": rows, "jobs": jobs})
    traces = {
        record["key"]: {"columns": ["t_s", "total_rss_mb", "child_rss_mb"], "phases": _phase_marks(record), "samples": record.get("trace", [])}
        for record in records
    }
    write_json_atomic(out_dir / "memory_traces.json", traces, indent=None)
    return rows


def _phase_marks(record: Mapping[str, Any]) -> list[list[Any]]:
    t0 = record["discovery_window"][0]
    return [[p["name"], round(p["t0"] - t0, 2), round(p["t1"] - t0, 2)] for p in (record.get("job") or {}).get("phases", [])]


# ------------------------------------------------------------------------------- metadata


def cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _java_version() -> str | None:
    java = shutil.which("java")
    if java is None:
        return None
    try:
        completed = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (completed.stderr or completed.stdout).strip().splitlines()
    return " | ".join(line.strip() for line in lines[:2]) or None


def _blas_name() -> str | None:
    try:
        import numpy

        blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f"{blas.get('name')} {blas.get('version')}"
    except Exception:  # noqa: BLE001 - informational only
        return None


def hardware_info(disk_path: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    frequency = psutil.cpu_freq()
    try:
        import numpy

        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    sets = windows_cpu_sets()
    cpu_sets = None
    if sets:
        cpu_sets = {
            f"efficiency_class_{c}": _ranges([row["logical"] for row in sets if row["efficiency_class"] == c])
            for c in sorted({row["efficiency_class"] for row in sets})
        }
    return {
        "cpu_model": cpu_model(),
        "cpu_identifier": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "cpu_sets_logical_cpus": cpu_sets,
        "cpu_max_mhz": None if frequency is None else frequency.max,
        "ram_total_gb": memory.total / GIB,
        "ram_available_gb_at_start": memory.available / GIB,
        "disk_free_gb_at_start": shutil.disk_usage(disk_path).free / GIB,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": numpy_version,
        "blas": _blas_name(),
        "psutil": psutil.__version__,
        "java": _java_version(),
    }


def _git_head() -> str | None:
    try:
        completed = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _subsample_summary(subsample: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in subsample.items() if key not in ("patient_ids", "episodes")}


def _record_key(method: str, fraction: float) -> str:
    return f"{method}@{fraction:g}"


def _previous_records(out_dir: Path) -> dict[str, dict[str, Any]]:
    path = out_dir / "scalability.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    traces_path = out_dir / "memory_traces.json"
    traces = json.loads(traces_path.read_text(encoding="utf-8")) if traces_path.is_file() else {}
    previous = {}
    for job in payload.get("jobs", []):
        record = dict(job)
        record["trace"] = (traces.get(record["key"]) or {}).get("samples", [])
        previous[record["key"]] = record
    return previous


# ---------------------------------------------------------------------------------- README


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def results_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| method | stage | fraction | train episodes | time (s) | peak RSS (MB) | child peak RSS (MB) | "
        "candidates | raw patterns / rules | status |"
    )
    lines = [header, "|" + "---|" * 10]
    for row in rows:
        lines.append(
            "| {method} | {stage} | {fraction:g} | {episodes} | {time} | {peak} | {child} | {cands} | {raw} | {status} |".format(
                method=row["method"],
                stage=row["stage"],
                fraction=row["fraction"],
                episodes=_fmt(row.get("n_train_episodes")),
                time=_fmt(row.get("time_s")),
                peak=_fmt(row.get("peak_rss_mb"), 0),
                child=_fmt(row.get("child_peak_rss_mb"), 0),
                cands=_fmt(row.get("n_candidates")),
                raw=_fmt(row.get("n_mined_raw")),
                status=row.get("status", ""),
            )
        )
    return "\n".join(lines)


def write_readme(out_dir: Path, meta: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    hw = meta["hardware"]
    cohort = meta["cohort"]
    subsamples = meta["subsamples"]
    fmg = meta["params"].get(FMG_METHOD, DEFAULT_PARAMS[FMG_METHOD])
    notes = {method: method_note(method, meta["params"].get(method, DEFAULT_PARAMS[method])) for method in METHODS}
    sub_lines = "\n".join(
        f"| {s['fraction']:g} | {s['n_patients']:,} | {s['n_episodes']:,} | {s['n_positives']:,} | {s['n_controls']:,} |"
        for s in subsamples
    )
    overrides = meta.get("params_overrides") or {}
    overrides_note = (
        f" This run overrode the defaults below with `{json.dumps(overrides, sort_keys=True)}`." if overrides else ""
    )
    superseded_note = (
        " (kept in `superseded_unpinned_run/` for reference only)" if (out_dir / "superseded_unpinned_run").is_dir() else ""
    )
    text = f"""# Discovery-stage scalability: matched runtime and memory

Generated by `scripts/v2/run_scalability_benchmark.py` ({meta['created_utc']}, git `{meta.get('git_head') or 'unknown'}`).
Raw numbers: `scalability.csv` (one row per method x fraction, plus one row per FMG-Bridge
stage), `scalability.json` (all metadata and per-phase details), `memory_traces.json`
(1-s maxima of the sampled RSS), job logs in `logs/`.

## Protocol

- **Cohort**: TRAIN split of `{cohort['path']}` ({cohort['n_train_patients']:,} patients,
  {cohort['n_train_episodes']:,} episodes). Baselines read the raw episodes in
  `{cohort['episode_dir']}`; FMG-Bridge reads the FMG view `{cohort['fmg_dir']}` (same
  episodes, v1 representation), as in the v2 pipeline.
- **Cohort-size subsamples**: TRAIN patients ranked once by
  `sha256("{meta['seed']}:sample:<subject_id>")` (`src.icu.v2.cohort.sample_patient_ids`);
  fraction f keeps the first round(f x N) patients and all their TRAIN episodes. The
  subsamples are patient-level, nested (0.25 within 0.5 within 1.0), deterministic, and
  independent of input order; hashes of the patient and episode lists are in
  `scalability.json`.

| fraction | patients | episodes | positives | controls |
|---|---|---|---|---|
{sub_lines}

- **Methods and candidate budgets** (same parameters as the v2 comparison
  `artifacts/v2/comparison/configs/x_mimic_iv_gap1h.json`; all mining and ranking on the
  TRAIN subsample only; the parameters used are in `scalability.json`, `meta.params`).{overrides_note}
  - `prefixspan`: {notes['prefixspan']}.
  - `erminer`: {notes['erminer']}.
  - `matrix_profile`: {notes['matrix_profile']}.
  - `shapelet`: {notes['shapelet']}.
  - `fmg_bridge`: v1 per-episode mining on the FMG view in shards of {fmg.get('shard_size')}
    episodes run one after another (unmodified `scripts/run_mimic_experiment.py
    --stop-after-contrastive-cache`, one subprocess per shard, as
    `run_fmg_mining_sharded.py --workers 1`); each shard cache is merged into a compact
    part and deleted at once; exact support pre-filter (`fmg_prefilter`, thresholds of
    the locked profile); v1 contrastive stage via `fmg_adapter.run_fmg_contrastive_stage`
    twice (without, then with the transfer target, as `run_fmg_train_only.py`) on the
    locked reference profile. v1 consolidates at most 20 programs.
- **What is timed** (`time_s`): the discovery stage only, from the episode files on disk
  to the ranked candidate pool. Included: loading the TRAIN episode files, building the
  shared event representation (SPMF methods: global TRAIN bins + eventization), mining
  (SPMF JVM start-up, input/output files and parsing; v1 per-shard interpreter start-up),
  ranking/selection, and for FMG-Bridge the shard file preparation, cache merge/deletion
  and the support pre-filter. Excluded: interpreter start-up and imports of the job
  process, firing on validation/test/external episodes, the shared v2 selector, the
  bridge, and evaluation. Sub-phase times are in the CSV (`load_s`, `represent_s`,
  `mine_s`, `rank_s`; FMG stage rows).
- **Memory**: each (method, fraction) runs in a fresh job process, one job at a time. A
  psutil polling thread in the driver samples, every {meta['interval_s']} s, the resident
  set size (Windows working set) of the job process plus all its descendants (SPMF JVM,
  v1 mining subprocesses). `peak_rss_mb` is the maximum of that sum during the discovery
  stage, `child_peak_rss_mb` the maximum of the descendants' part, `rss_at_start_mb` the
  sum at the start of the stage (interpreter, imports, job spec). MB = 2^20 bytes.
  `peak_private_mb` (Windows private commit) and the OS peak working sets of every process
  are recorded as cross-checks in `scalability.json`.
- **Single process, single thread, fixed core type**: every job and its children run one
  at a time; BLAS/OpenMP pools are limited to {meta['blas_threads']} thread(s)
  (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, ...); SPMF runs in one JVM (`-Xmx2g`, JVM
  default GC threads). CPU placement: {meta['cpu_placement']['description']}, priority
  class {meta['cpu_placement']['priority']}, applied to the job process and to every
  descendant (SPMF JVM, v1 mining processes). On this hybrid CPU Windows otherwise places
  a single-threaded job on a performance or an efficiency core depending on the
  background load, which changed the CPU time per unit of work by up to about 1.7x in a
  first, unpinned run{superseded_note}.
  `cpu_s` is the CPU time of the job process plus its children (children: last poll); it
  is less sensitive to waiting for a core than the wall-clock `time_s`.
- **Scheduling**: jobs run method by method with fractions ascending, FMG-Bridge last.
  Before each job the driver waits until the system has enough available RAM and disk
  for it (other experiments share the machine); a running job is killed if available RAM
  drops below {meta['args'].get('min_free_ram_gb')} GB or free disk below
  {meta['args'].get('min_free_disk_gb')} GB.

## Hardware

- CPU: {hw['cpu_model']} ({hw['physical_cores']} physical cores, {hw['logical_cpus']} logical CPUs, max {_fmt(hw.get('cpu_max_mhz'), 0)} MHz; logical CPUs by efficiency class: {hw.get('cpu_sets_logical_cpus')})
- RAM: {hw['ram_total_gb']:.1f} GB total, {hw['ram_available_gb_at_start']:.1f} GB available at the start
- OS: {hw['os']}; Python {hw['python']} (`{hw['python_executable']}`), numpy {hw.get('numpy')} ({hw.get('blas')}), psutil {hw['psutil']}
- Java: {hw.get('java')}; SPMF {meta.get('spmf_version')}
- Background load: other experiments were running on this machine; the system CPU
  utilisation and the minimum available RAM during every job are in `scalability.json`
  (`jobs[].system`).

## Results

{results_table(rows)}

## Caveats

- One repetition per (method, fraction); wall-clock times include background load from
  concurrent experiments (see `jobs[].system`).
- FMG-Bridge mines every TRAIN episode (positives and controls) per episode and then
  contrasts them; PrefixSpan, ERMiner and the matrix-profile motifs mine positive TRAIN
  episodes only and shapelets use all TRAIN episodes, following each method's v2 protocol.
  Candidate budgets differ by design: 200 for the SPMF pools, 3 per feature for the
  numeric miners (at most 51), at most 20 programs for FMG-Bridge (v1 consolidation).
- FMG-Bridge memory is bounded by the shard size in per-episode mining; the pre-filtered
  rule records of all TRAIN episodes are held in memory in the contrastive stage. The
  production scripts additionally write the pre-filtered cache to disk and read it back,
  which this benchmark skips (identical content).
- Peaks are sampled every {meta['interval_s']} s; a spike shorter than that can be missed
  (the OS peak working sets in `scalability.json` are exact per process).
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", default=str(DEFAULT_COHORT), help="episodes_split.json of the v2 cohort")
    parser.add_argument("--episode-dir", default=str(DEFAULT_EPISODE_DIR), help="raw v2 episode files (baselines)")
    parser.add_argument("--fmg-dir", default=str(DEFAULT_FMG_DIR), help="FMG view with shock_episode_index.json")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--seed", type=int, default=0, help="patient subsampling seed")
    parser.add_argument("--interval", type=float, default=0.1, help="RSS polling interval in seconds")
    parser.add_argument("--fmg-shard-size", type=int, default=int(DEFAULT_PARAMS[FMG_METHOD]["shard_size"]))
    parser.add_argument(
        "--params-json",
        default=None,
        help="JSON object or file {method: {param: value}} merged into the default parameters (recorded in the outputs)",
    )
    parser.add_argument("--blas-threads", type=int, default=1, help="BLAS/OpenMP threads per job (0 = leave unset)")
    parser.add_argument(
        "--cpu-affinity",
        default="performance",
        help="'performance' (P-cores of a hybrid CPU; default), 'all', or a list such as '0-11'; inherited by children",
    )
    parser.add_argument(
        "--priority",
        default="above_normal",
        choices=sorted(PRIORITY_CLASS_NAMES),
        help="Windows priority class of every job process and descendant",
    )
    parser.add_argument("--job-timeout-s", type=float, default=4 * 3600.0)
    parser.add_argument("--min-free-ram-gb", type=float, default=1.0, help="kill a job when system RAM available drops below")
    parser.add_argument("--min-free-disk-gb", type=float, default=15.0, help="kill a job when free disk drops below")
    parser.add_argument(
        "--max-wait-s",
        type=float,
        default=3 * 3600.0,
        help="wait at most this long for enough free RAM/disk before a job (then it is skipped)",
    )
    parser.add_argument("--poll-s", type=float, default=30.0, help="polling period while waiting for resources")
    parser.add_argument("--python", default=sys.executable, help="interpreter for the job processes")
    parser.add_argument("--resume", action="store_true", help="keep finished jobs of an earlier run in --out-dir")
    parser.add_argument("--keep-tmp", action="store_true", help="debugging: keep <out-dir>/tmp")
    parser.add_argument("--job-spec", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--job-result", default=None, help=argparse.SUPPRESS)
    return parser


def method_params(methods: Sequence[str], fmg_shard_size: int, overrides: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Default parameters of ``methods`` with the FMG shard size and ``overrides`` applied."""
    params = {method: dict(DEFAULT_PARAMS[method]) for method in methods}
    if FMG_METHOD in params:
        params[FMG_METHOD]["shard_size"] = int(fmg_shard_size)
    for method, values in (overrides or {}).items():
        if method not in DEFAULT_PARAMS:
            raise ValueError(f"--params-json names unknown method {method!r}")
        if not isinstance(values, Mapping):
            raise ValueError(f"--params-json[{method!r}] must be an object")
        if method in params:
            params[method].update(values)
    return params


def _load_overrides(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    text = value if value.strip().startswith("{") else _abs(value).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise SystemExit("--params-json must be a JSON object {method: {param: value}}")
    return payload


def run_benchmark(args: argparse.Namespace) -> int:
    out_dir = _abs(args.out_dir)
    tmp_root, logs_dir = out_dir / "tmp", out_dir / "logs"
    cohort_path, episode_dir, fmg_dir = _abs(args.cohort), _abs(args.episode_dir), _abs(args.fmg_dir)
    methods = list(dict.fromkeys(args.methods))
    episodes = load_cohort(cohort_path)
    subsamples = nested_train_subsamples(episodes, args.fractions, args.seed)
    overrides = _load_overrides(args.params_json)
    params = method_params(methods, args.fmg_shard_size, overrides)
    cpus, cpu_description = resolve_affinity(args.cpu_affinity)
    priority = priority_value(args.priority)
    out_dir.mkdir(parents=True, exist_ok=True)
    disk_path = out_dir
    spmf_version = None
    if any(m in SPMF_METHODS for m in methods):
        from src.icu.v2.baselines.spmf import SPMF_VERSION

        spmf_version = SPMF_VERSION
    full = subsamples[-1]
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "args": {key: value for key, value in vars(args).items() if key not in ("job_spec", "job_result")},
        "seed": int(args.seed),
        "interval_s": float(args.interval),
        "blas_threads": int(args.blas_threads),
        "cpu_placement": {
            "affinity": cpus,
            "description": cpu_description,
            "priority": args.priority if priority is not None or args.priority == "normal" else f"{args.priority} (not applied on this OS)",
        },
        "methods": methods,
        "params": params,
        "params_overrides": overrides,
        "representation": REPRESENTATION,
        "spmf_version": spmf_version,
        "cohort": {
            "path": _rel(cohort_path),
            "sha256": _sha256_file(cohort_path),
            "episode_dir": _rel(episode_dir),
            "fmg_dir": _rel(fmg_dir),
            "n_episodes": len(episodes),
            "n_train_episodes": sum(1 for e in episodes if e["split"] == "train"),
            "n_train_patients": full["n_patients_total"],
        },
        "subsamples": [_subsample_summary(s) for s in subsamples],
        "hardware": hardware_info(disk_path),
        "code_sha256": {name: _sha256_file(ROOT / name) for name in CODE_FILES if (ROOT / name).is_file()},
    }
    previous = _previous_records(out_dir) if args.resume else {}
    env = job_env(int(args.blas_threads))
    records: list[dict[str, Any]] = []
    try:
        for method in methods:  # method by method, so each scaling curve is measured in one stretch
            for subsample in subsamples:
                key = _record_key(method, subsample["fraction"])
                summary = _subsample_summary(subsample)
                old = previous.get(key)
                if (
                    old is not None
                    and old.get("status") == "ok"
                    and old.get("params") == json_ready(params[method])
                    and (old.get("subsample") or {}).get("episodes_sha256") == summary["episodes_sha256"]
                ):
                    records.append(old)
                    print(f"[resume] {key}: kept the earlier measurement", flush=True)
                    continue
                base_record = {"key": key, "method": method, "fraction": subsample["fraction"], "params": params[method], "subsample": summary}
                need_disk = estimated_disk_need_gb(method, subsample["n_episodes"], params[method])
                need_ram = estimated_ram_need_gb(method, subsample["n_episodes"])
                admitted, waited, why = wait_for_resources(
                    need_ram,
                    need_disk,
                    min_free_ram_gb=args.min_free_ram_gb,
                    min_free_disk_gb=args.min_free_disk_gb,
                    disk_path=disk_path,
                    max_wait_s=args.max_wait_s,
                    poll_s=args.poll_s,
                    label=key,
                )
                admission = {"waited_s": waited, "need_ram_gb": need_ram, "need_disk_gb": need_disk}
                if not admitted:
                    record = {
                        **base_record,
                        "status": "skipped",
                        "reason": f"not started after waiting {waited:.0f} s: {why}",
                        "admission": admission,
                        "job": {},
                        "memory": {"discovery": summarize_samples([]), "phases": {}, "children": []},
                        "discovery_window": [0.0, 0.0],
                        "trace": [],
                    }
                else:
                    job_tmp = tmp_root / key.replace("@", "_f")
                    spec = {
                        "method": method,
                        "fraction": subsample["fraction"],
                        "params": params[method],
                        "representation": REPRESENTATION,
                        "episode_dir": str(episode_dir),
                        "fmg_dir": str(fmg_dir),
                        "tmp_dir": str(job_tmp),
                        "python": args.python,
                        "train": [
                            {"episode_id": e["episode_id"], "subject_id": e["subject_id"], "label": e["label"]}
                            for e in subsample["episodes"]
                        ],
                    }
                    print(
                        f"[run] {key}: {subsample['n_episodes']} TRAIN episodes of {subsample['n_patients']} patients "
                        f"({datetime.now().strftime('%H:%M:%S')}, waited {waited:.0f} s for resources)",
                        flush=True,
                    )
                    try:
                        outcome = run_job(
                            spec,
                            job_dir=job_tmp,
                            log_path=logs_dir / f"{key.replace('@', '_f')}.log",
                            python=args.python,
                            env=env,
                            interval=args.interval,
                            timeout_s=args.job_timeout_s,
                            min_free_ram_gb=args.min_free_ram_gb,
                            min_free_disk_gb=args.min_free_disk_gb,
                            disk_path=disk_path,
                            cpus=cpus,
                            priority=priority,
                        )
                    finally:
                        if not args.keep_tmp:
                            safe_rmtree(job_tmp, tmp_root)
                    record = {**base_record, **outcome, "admission": admission}
                records.append(record)
                rows = write_outputs(out_dir, meta, records)
                total = next(row for row in rows_for_record(record) if row["stage"] == TOTAL_STAGE)
                print(
                    f"[done] {key}: status={record['status']} time_s={total.get('time_s')} "
                    f"peak_rss_mb={total.get('peak_rss_mb')} child_peak_rss_mb={total.get('child_peak_rss_mb')} "
                    f"n_candidates={total.get('n_candidates')} {record.get('reason', '')}",
                    flush=True,
                )
    finally:
        if not args.keep_tmp and tmp_root.exists():
            safe_rmtree(tmp_root, out_dir)
    rows = write_outputs(out_dir, meta, records)
    write_readme(out_dir, meta, rows)
    return 0 if all(record["status"] == "ok" for record in records) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.job_spec:
        if not args.job_result:
            raise SystemExit("--job-spec needs --job-result")
        return job_main(Path(args.job_spec), Path(args.job_result))
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
