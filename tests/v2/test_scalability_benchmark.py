"""Tests for scripts/v2/run_scalability_benchmark.py (memory sampler, subsampling, writers, jobs)."""

from __future__ import annotations

import csv
import functools
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "v2" / "run_scalability_benchmark.py"
MIB = 1 << 20

HAVE_SPMF = shutil.which("java") is not None and (ROOT / "tools" / "spmf" / "spmf.jar").is_file()
requires_spmf = pytest.mark.skipif(not HAVE_SPMF, reason="java or tools/spmf/spmf.jar unavailable")
HAVE_V1 = (ROOT / "scripts" / "run_mimic_experiment.py").is_file() and (ROOT / "Non-stationary time series.py").is_file()
requires_v1 = pytest.mark.skipif(not HAVE_V1, reason="v1 FMG mining code unavailable")

REQUIRED_COLUMNS = [
    "method",
    "stage",
    "fraction",
    "time_s",
    "peak_rss_mb",
    "child_peak_rss_mb",
    "n_train_episodes",
    "n_candidates",
    "notes",
]
FEATURES = ("HR", "RR", "SpO2", "MAP", "Lactate", "Temp")


@functools.lru_cache(maxsize=1)
def _module():
    name = "run_scalability_benchmark"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _strict_json(path: Path):
    def _reject(constant: str):
        raise AssertionError(f"non-strict JSON constant {constant} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


# ------------------------------------------------------------------------------ memory sampler


def test_sampler_sees_own_memory_and_a_child_process():
    m = _module()
    sampler = m.ProcessTreeSampler(interval=0.05).start()
    time.sleep(0.3)
    t_alloc = time.time()
    block = np.ones(160 * MIB // 8)  # written, hence resident
    time.sleep(0.4)
    t_freed = time.time()
    del block
    time.sleep(0.2)
    t_child = time.time()
    child = subprocess.Popen([sys.executable, "-c", "import time; b = b'x' * (120 * 2**20); time.sleep(1.0)"])
    child.wait(timeout=120)
    t_end = time.time()
    sampler.stop()

    before = m.summarize_samples(sampler.samples, None, t_alloc)
    during = m.summarize_samples(sampler.samples, t_alloc, t_freed)
    with_child = m.summarize_samples(sampler.samples, t_child, t_end)
    assert before["n_samples"] >= 3 and during["n_samples"] >= 3
    assert during["self_peak_rss_mb"] - before["self_peak_rss_mb"] > 120
    assert before["child_peak_rss_mb"] == 0 and during["child_peak_rss_mb"] == 0
    assert with_child["child_peak_rss_mb"] > 100
    assert with_child["max_children"] >= 1
    assert with_child["peak_rss_mb"] >= with_child["child_peak_rss_mb"]
    seen = [c for c in sampler.children.values() if c["pid"] == child.pid]
    assert len(seen) == 1 and seen[0]["peak_rss_mb"] > 100 and seen[0]["last_seen"] >= seen[0]["first_seen"]
    with pytest.raises(ValueError):
        m.ProcessTreeSampler(interval=0.0)


def test_sampler_follows_grandchildren_of_an_external_process():
    m = _module()
    code = (
        "import subprocess, sys; "
        "subprocess.run([sys.executable, '-c', \"import time; b = b'y' * (100 * 2**20); time.sleep(1.0)\"], check=True)"
    )
    root = subprocess.Popen([sys.executable, "-c", code])
    sampler = m.ProcessTreeSampler(root.pid, interval=0.05).start()
    root.wait(timeout=120)
    sampler.stop()
    whole = m.summarize_samples(sampler.samples)
    assert whole["n_samples"] >= 5
    assert whole["child_peak_rss_mb"] > 80  # the grandchild, seen through the child
    assert whole["peak_rss_mb"] > whole["self_peak_rss_mb"]


def test_summarize_samples_windows_bracketing_and_empty():
    m = _module()
    sample = m.MemorySample
    samples = [
        sample(0.0, 100 * MIB, 0, 110 * MIB, 0, 0),
        sample(1.0, 150 * MIB, 50 * MIB, 160 * MIB, 60 * MIB, 1),
        sample(2.0, 120 * MIB, 0, 125 * MIB, 0, 0),
    ]
    whole = m.summarize_samples(samples)
    assert whole["n_samples"] == 3
    assert (whole["peak_rss_mb"], whole["child_peak_rss_mb"], whole["self_peak_rss_mb"]) == (200, 50, 150)
    assert whole["peak_private_mb"] == 220 and whole["rss_at_start_mb"] == 100 and whole["max_children"] == 1
    window = m.summarize_samples(samples, 0.5, 1.5)
    assert window["n_samples"] == 1 and window["rss_at_start_mb"] == 200
    late = m.summarize_samples(samples, 1.5, 2.5)
    assert late["peak_rss_mb"] == 120 and late["child_peak_rss_mb"] == 0
    gap = m.summarize_samples(samples, 1.2, 1.4)  # no sample inside: the two neighbours are used
    assert gap["n_samples"] == 0 and gap["peak_rss_mb"] == 200 and gap["self_peak_rss_mb"] == 150
    empty = m.summarize_samples([])
    assert empty["n_samples"] == 0 and empty["peak_rss_mb"] is None
    trace = m.downsample_trace(samples + [sample(1.5, 300 * MIB, 0, 0, 0, 0)], origin=0.0)
    assert [row[0] for row in trace] == [0.0, 1.5, 2.0] and trace[1][1] == 300.0


# ---------------------------------------------------------------------------------- subsampling


def _cohort_rows(n_patients: int = 40) -> list[dict]:
    rows = []
    for p in range(n_patients):
        split = ("train", "train", "train", "val", "test")[p % 5]
        for j in range(2 if p % 3 == 0 else 1):
            rows.append({"episode_id": f"stay_{p}_{j}", "subject_id": 1000 + p, "label": (p + j) % 2, "split": split})
    return rows


def test_subsamples_are_patient_level_nested_and_deterministic():
    m = _module()
    rows = _cohort_rows()
    subsamples = m.nested_train_subsamples(rows, (1.0, 0.25, 0.5), seed=3)
    assert [s["fraction"] for s in subsamples] == [0.25, 0.5, 1.0]
    train = [r for r in rows if r["split"] == "train"]
    train_patients = {r["subject_id"] for r in train}
    n = len(train_patients)
    assert [s["n_patients"] for s in subsamples] == [round(0.25 * n), round(0.5 * n), n]
    for s in subsamples:
        patients = set(s["patient_ids"])
        assert patients <= train_patients and len(patients) == s["n_patients"]
        # patient-level: every TRAIN episode of a sampled patient, and nothing else
        expected = {r["episode_id"] for r in train if r["subject_id"] in patients}
        assert {e["episode_id"] for e in s["episodes"]} == expected
        assert s["n_episodes"] == len(expected) == s["n_positives"] + s["n_controls"]
        assert all(e["split"] == "train" for e in s["episodes"])
    assert set(subsamples[0]["patient_ids"]) < set(subsamples[1]["patient_ids"]) < set(subsamples[2]["patient_ids"])
    assert set(subsamples[2]["patient_ids"]) == train_patients

    shuffled = m.nested_train_subsamples(list(reversed(rows)), (0.25, 0.5, 1.0), seed=3)
    for key in ("patients_sha256", "episodes_sha256", "patient_ids"):
        assert [s[key] for s in shuffled] == [s[key] for s in subsamples]
    other_seed = m.nested_train_subsamples(rows, (0.25,), seed=4)
    assert other_seed[0]["patient_ids"] != subsamples[0]["patient_ids"]


def test_subsampling_rejects_bad_fractions_and_missing_train():
    m = _module()
    rows = _cohort_rows()
    for bad in ([0.0], [1.5], [float("nan")], []):
        with pytest.raises(ValueError):
            m.nested_train_subsamples(rows, bad)
    with pytest.raises(ValueError, match="no 'train' episodes"):
        m.nested_train_subsamples([dict(r, split="test") for r in rows], [0.5])
    tiny = m.nested_train_subsamples(rows, [0.01])
    assert tiny[0]["n_patients"] == 1


def test_load_cohort_derives_labels_and_rejects_duplicates(tmp_path):
    m = _module()
    path = tmp_path / "cohort.json"
    path.write_text(
        json.dumps(
            [
                {"episode_id": "a", "subject_id": 1, "episode_type": "positive", "split": "train"},
                {"episode_id": "b", "subject_id": 2, "label": 0, "split": "test"},
            ]
        ),
        encoding="utf-8",
    )
    assert [(e["episode_id"], e["label"]) for e in m.load_cohort(path)] == [("a", 1), ("b", 0)]
    path.write_text(json.dumps([{"episode_id": "a", "subject_id": 1, "label": 1}] * 2), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        m.load_cohort(path)


# ---------------------------------------------------------------------------------- the writers


def _record(m, method: str, fraction: float, phases: list[tuple[str, float]], *, status: str = "ok", child_mb: int = 0):
    """A driver record built from synthetic samples, as run_job would return it."""
    sample = m.MemorySample
    samples, timed, t = [], [], 100.0
    for name, duration in phases:
        timed.append({"name": name, "t0": t, "t1": t + duration, "duration_s": duration, "self_cpu_s": duration * 0.9})
        samples.append(sample(t + duration / 2, int((200 + 10 * len(timed)) * MIB), child_mb * MIB, 0, 0, int(child_mb > 0)))
        t += duration
    window = (timed[0]["t0"], timed[-1]["t1"])
    children = [{"pid": 7, "name": "java.exe", "first_seen": window[0] + 0.1, "last_seen": window[0] + 0.2, "cpu_s": 2.0}] if child_mb else []
    return {
        "key": f"{method}@{fraction:g}",
        "method": method,
        "fraction": fraction,
        "status": status,
        "reason": "" if status == "ok" else "job exited with code 1",
        "params": {"x": 1},
        "subsample": {"n_episodes": 10, "n_patients": 9, "n_positives": 4, "n_controls": 6, "episodes_sha256": "e"},
        "job": {"phases": timed, "n_candidates": 7, "n_mined_raw": 99, "odd": float("nan")} if status == "ok" else {},
        "memory": {
            "discovery": m.summarize_samples(samples, *window),
            "phases": {p["name"]: m.summarize_samples(samples, p["t0"], p["t1"]) for p in timed},
            "children": children,
        },
        "discovery_window": list(window),
        "trace": m.downsample_trace(samples, window[0]),
    }


def test_rows_csv_and_json_writers(tmp_path):
    m = _module()
    records = [
        _record(m, "prefixspan", 0.25, [("load", 1.0), ("represent", 0.5), ("mine", 2.0), ("rank", 3.0)], child_mb=50),
        _record(m, "fmg_bridge", 0.25, [("per_episode_mining", 10.0), ("support_prefilter", 2.0), ("contrastive_stage", 4.0)], child_mb=300),
        _record(m, "shapelet", 0.25, [("load", 1.0)], status="failed"),
    ]
    rows = m.write_outputs(tmp_path, {"created_utc": "now"}, records)
    written = _read_csv(tmp_path / "scalability.csv")
    assert list(written[0].keys())[: len(REQUIRED_COLUMNS)] == REQUIRED_COLUMNS
    assert [(r["method"], r["stage"]) for r in written] == [
        ("prefixspan", "discovery_total"),
        ("fmg_bridge", "discovery_total"),
        ("fmg_bridge", "per_episode_mining"),
        ("fmg_bridge", "support_prefilter"),
        ("fmg_bridge", "contrastive_stage"),
        ("shapelet", "discovery_total"),
    ]
    prefixspan = written[0]
    assert float(prefixspan["time_s"]) == 6.5 and float(prefixspan["mine_s"]) == 2.0
    assert float(prefixspan["child_peak_rss_mb"]) == 50.0 and float(prefixspan["peak_rss_mb"]) == 290.0
    assert prefixspan["n_candidates"] == "7" and prefixspan["n_mined_raw"] == "99" and prefixspan["n_train_episodes"] == "10"
    assert math.isclose(float(prefixspan["cpu_s"]), 6.5 * 0.9 + 2.0)  # own CPU plus the child's
    fmg = {r["stage"]: r for r in written if r["method"] == "fmg_bridge"}
    assert float(fmg["discovery_total"]["time_s"]) == sum(float(fmg[s]["time_s"]) for s in m.FMG_STAGES) == 16.0
    assert float(fmg["per_episode_mining"]["child_peak_rss_mb"]) == 300.0
    assert fmg["contrastive_stage"]["n_candidates"] == "7" and fmg["support_prefilter"]["n_candidates"] == ""
    failed = written[-1]
    assert failed["status"] == "failed" and failed["time_s"] == "" and "failed" in failed["notes"]

    payload = _strict_json(tmp_path / "scalability.json")
    assert payload["meta"] == {"created_utc": "now"}
    assert payload["rows"] == json.loads(json.dumps(m.json_ready(rows)))
    assert payload["jobs"][0]["job"]["odd"] is None and "trace" not in payload["jobs"][0]
    traces = _strict_json(tmp_path / "memory_traces.json")
    assert set(traces) == {"prefixspan@0.25", "fmg_bridge@0.25", "shapelet@0.25"}
    assert [mark[0] for mark in traces["fmg_bridge@0.25"]["phases"]] == list(m.FMG_STAGES)
    assert not list(tmp_path.glob("*.partial"))


def test_method_params_merge_overrides():
    m = _module()
    params = m.method_params(["prefixspan", "fmg_bridge"], 50, {"prefixspan": {"minsup": 0.5}})
    assert params["prefixspan"]["minsup"] == 0.5 and params["prefixspan"]["pool_cap"] == 200
    assert params["fmg_bridge"]["shard_size"] == 50 and "erminer" not in params
    assert m.DEFAULT_PARAMS["prefixspan"]["minsup"] == 0.2
    assert "minsup 0.5" in m.method_note("prefixspan", params["prefixspan"])
    assert "shards of 50" in m.method_note("fmg_bridge", params["fmg_bridge"])
    with pytest.raises(ValueError):
        m.method_params(["prefixspan"], 50, {"nope": {}})


def test_cpu_placement_helpers():
    m = _module()
    assert m.parse_cpu_list("0-3, 6,5") == [0, 1, 2, 3, 5, 6]
    assert m._ranges([6, 0, 1, 2, 3, 5]) == "0-3,5-6" and m._ranges([4]) == "4"
    for bad in ("", "3-1", "-1"):
        with pytest.raises(ValueError):
            m.parse_cpu_list(bad)
    assert m.resolve_affinity("all") == (None, "not pinned")
    assert m.resolve_affinity("0,1")[0] == [0, 1]
    cpus, description = m.resolve_affinity("performance")
    assert cpus is None or (cpus and all(0 <= c < (os.cpu_count() or 1) for c in cpus)), description
    assert m.priority_value("normal") is None
    with pytest.raises(ValueError):
        m.priority_value("realtime")


def test_resource_gate_admits_or_reports_what_is_missing(tmp_path):
    m = _module()
    admitted, waited, why = m.wait_for_resources(
        0.0, 0.0, min_free_ram_gb=0.0, min_free_disk_gb=0.0, disk_path=tmp_path, max_wait_s=0.0, poll_s=0.1, label="x"
    )
    assert admitted and why == ""
    admitted, waited, why = m.wait_for_resources(
        1e6, 1e6, min_free_ram_gb=0.0, min_free_disk_gb=0.0, disk_path=tmp_path, max_wait_s=0.2, poll_s=0.1, label="x"
    )
    assert not admitted and waited >= 0.2 and "RAM available" in why and "disk free" in why
    assert m.estimated_ram_need_gb("fmg_bridge", 3697) > m.estimated_ram_need_gb("fmg_bridge", 924)
    assert m.estimated_disk_need_gb("prefixspan", 3697, {}) < m.estimated_disk_need_gb("fmg_bridge", 3697, {"shard_size": 220})


# ------------------------------------------------------------------------ end to end (synthetic)


def _write_synthetic_cohort(root: Path, n_patients: int = 16, seed: int = 0) -> tuple[Path, Path, Path]:
    """Raw and FMG-view episodes in the v1 file layout plus an episodes_split.json.

    Positives carry a falling MAP and a rising Lactate so that every miner finds something.
    """
    rng = np.random.default_rng(seed)
    raw_dir, fmg_dir = root / "raw", root / "fmg"
    raw_dir.mkdir(parents=True)
    fmg_dir.mkdir(parents=True)
    rows, metas = [], []
    hours = np.arange(12)
    for p in range(n_patients):
        split = "train" if p < n_patients - 4 else ("val" if p % 2 else "test")
        labels = [p % 2] + ([1 - p % 2] if p % 4 == 0 else [])
        for j, label in enumerate(labels):
            kind = "positive" if label else "control"
            episode_id = f"stay_{3000 + p}_{j}_{kind}"
            values = rng.normal(0.0, 1.0, size=(12, len(FEATURES)))
            if label:
                values[:, FEATURES.index("MAP")] -= 0.5 * hours
                values[:, FEATURES.index("Lactate")] += 0.5 * hours
            mask = rng.random(values.shape) < 0.2
            mask[:, :2] = False
            raw = values.copy()
            raw[mask] = np.nan
            stem = f"shock_episode_{episode_id}"
            meta = {
                "episode_id": episode_id,
                "episode_type": kind,
                "stay_id": 30000 + 10 * p + j,
                "subject_id": 700 + p,
                "resample_minutes": 60,
                "shape": [12, len(FEATURES)],
                "feature_names": list(FEATURES),
                "missing_mask_file": f"{stem}_missing_mask.npy",
                "intervention_features": [],
                "missingness_features": list(FEATURES),
                "observation_density_features": list(FEATURES),
            }
            for directory, array in ((raw_dir, raw), (fmg_dir, values)):
                np.save(directory / f"{stem}.npy", array)
                np.save(directory / f"{stem}_missing_mask.npy", mask)
                (directory / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
            metas.append(meta)
            rows.append({"episode_id": episode_id, "subject_id": 700 + p, "stay_id": meta["stay_id"], "label": label, "split": split})
    (fmg_dir / "shock_episode_index.json").write_text(json.dumps(metas), encoding="utf-8")
    cohort = root / "episodes_split.json"
    cohort.write_text(json.dumps(rows), encoding="utf-8")
    return cohort, raw_dir, fmg_dir


def _run_driver(m, tmp_path: Path, methods: list[str], fractions: list[str], *extra: str) -> tuple[int, Path]:
    cohort, raw_dir, fmg_dir = _write_synthetic_cohort(tmp_path / "data")
    out = tmp_path / "out"
    argv = [
        "--cohort", str(cohort),
        "--episode-dir", str(raw_dir),
        "--fmg-dir", str(fmg_dir),
        "--out-dir", str(out),
        "--methods", *methods,
        "--fractions", *fractions,
        "--interval", "0.05",
        "--min-free-disk-gb", "0",
        "--min-free-ram-gb", "0",
        "--max-wait-s", "0",
        *extra,
    ]  # fmt: skip
    return m.main(argv), out


def _logs(out: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace")[-3000:] for path in sorted((out / "logs").glob("*.log")))


def _check_common_outputs(out: Path) -> dict:
    assert (out / "README.md").is_file() and "## Hardware" in (out / "README.md").read_text(encoding="utf-8")
    assert not (out / "tmp").exists()
    assert (out / "logs").is_dir()
    payload = _strict_json(out / "scalability.json")
    for job in payload["jobs"]:
        pools = [p for p in job["job"].get("threadpools", []) if p["api"] in ("openblas", "mkl", "openmp")]
        assert all(p["num_threads"] == 1 for p in pools), pools
        placement = job["placement"]
        assert placement["job_error"] is None and placement["child_errors"] == []
        if placement["cpus"] is not None:
            assert job["job"]["cpu_affinity"] == placement["cpus"]
        if sys.platform == "win32":
            import psutil

            assert job["job"]["priority"] == psutil.ABOVE_NORMAL_PRIORITY_CLASS
    return payload


def test_driver_numeric_methods_end_to_end(tmp_path):
    m = _module()
    code, out = _run_driver(m, tmp_path, ["matrix_profile", "shapelet"], ["0.5", "1.0"], "--cpu-affinity", "0")
    assert code == 0, _logs(out)
    rows = _read_csv(out / "scalability.csv")
    assert {(r["method"], float(r["fraction"])) for r in rows} == {
        (method, fraction) for method in ("matrix_profile", "shapelet") for fraction in (0.5, 1.0)
    }
    for row in rows:
        assert row["status"] == "ok" and row["stage"] == "discovery_total"
        assert float(row["time_s"]) > 0 and float(row["peak_rss_mb"]) > 0 and float(row["child_peak_rss_mb"]) == 0
        assert int(row["n_candidates"]) >= 1 and float(row["load_s"]) >= 0 and float(row["mine_s"]) > 0
    sizes = {float(r["fraction"]): int(r["n_train_episodes"]) for r in rows if r["method"] == "shapelet"}
    assert sizes[0.5] < sizes[1.0]
    payload = _check_common_outputs(out)
    assert [s["fraction"] for s in payload["meta"]["subsamples"]] == [0.5, 1.0]
    assert payload["meta"]["hardware"]["logical_cpus"] >= 1
    assert payload["meta"]["cpu_placement"]["affinity"] == [0]
    assert all(job["job"]["cpu_affinity"] == [0] for job in payload["jobs"])


@requires_spmf
def test_driver_spmf_methods_see_the_jvm(tmp_path):
    m = _module()
    small = json.dumps({"prefixspan": {"minsup": 0.6, "max_pattern_length": 2}, "erminer": {"minsup": 0.6, "max_antecedent": 1}})
    code, out = _run_driver(m, tmp_path, ["prefixspan", "erminer"], ["1.0"], "--params-json", small)
    assert code == 0, _logs(out)
    rows = _read_csv(out / "scalability.csv")
    assert [r["method"] for r in rows] == ["prefixspan", "erminer"]
    for row in rows:
        assert row["status"] == "ok"
        assert float(row["child_peak_rss_mb"]) > 10  # the SPMF JVM
        assert float(row["peak_rss_mb"]) > float(row["child_peak_rss_mb"])
        assert 1 <= int(row["n_candidates"]) <= min(200, int(row["n_mined_raw"]))
        assert all(float(row[f"{phase}_s"]) >= 0 for phase in ("load", "represent", "mine", "rank"))
    payload = _check_common_outputs(out)
    assert payload["meta"]["params"]["prefixspan"]["minsup"] == 0.6
    assert any("java" in child["name"].lower() for child in payload["jobs"][0]["memory"]["children"])
    assert "overrode the defaults" in (out / "README.md").read_text(encoding="utf-8")


@requires_v1
def test_driver_fmg_bridge_runs_v1_mining_shards_and_contrastive_stage(tmp_path):
    m = _module()
    code, out = _run_driver(m, tmp_path, ["fmg_bridge"], ["1.0"], "--fmg-shard-size", "5")
    assert code == 0, _logs(out)
    rows = {r["stage"]: r for r in _read_csv(out / "scalability.csv")}
    assert set(rows) == {"discovery_total", *m.FMG_STAGES}
    assert all(r["status"] == "ok" for r in rows.values())
    total = rows["discovery_total"]
    assert math.isclose(float(total["time_s"]), sum(float(rows[s]["time_s"]) for s in m.FMG_STAGES), abs_tol=0.01)
    assert float(rows["per_episode_mining"]["child_peak_rss_mb"]) > 10  # the v1 mining subprocess
    assert float(total["child_peak_rss_mb"]) == float(rows["per_episode_mining"]["child_peak_rss_mb"])
    assert float(rows["contrastive_stage"]["child_peak_rss_mb"]) == 0
    assert int(total["n_mined_raw"]) > 0 and total["n_candidates"] == rows["contrastive_stage"]["n_candidates"]
    job = _check_common_outputs(out)["jobs"][0]["job"]
    n_train = int(total["n_train_episodes"])
    assert job["n_shards"] == math.ceil(n_train / 5) and job["n_mined_episodes"] == n_train
    assert job["n_distinct_rules_kept"] <= job["n_mined_raw"]
    assert sum(shard["n_episodes"] for shard in job["shards"]) == n_train
