"""Tests for the v2 supervised reference models (tiny synthetic data, CPU only)."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch a GPU, even with a CUDA build of torch

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.comparators.registry import get_default_registry  # noqa: E402
from src.icu.v2.supervised import (  # noqa: E402
    DEFAULT_EXTERNAL_N_BOOT,
    MISSINGNESS_BY_MODEL,
    SEQUENCE_MODELS,
    SPLITS,
    SUPERVISED_MODELS,
    EarlyStopping,
    ExternalSource,
    FileArrayLoader,
    SupervisedConfig,
    TaskWindowLoader,
    assign_splits,
    build_external_cohort,
    build_supervised_inputs,
    fit_model,
    fit_supervised_models,
    json_safe,
    load_episode_list,
    load_split_assignments,
    read_feature_names,
    run_supervised_v2,
    validate_episode_splits,
)
from src.icu.v2.task import TaskSpec  # noqa: E402
from src.icu.v2.types import EpisodeRecord  # noqa: E402

FEATURES = ("HR", "MAP", "Lactate", "vasopressor_on")
LEAK = FEATURES.index("vasopressor_on")
LATE = FEATURES.index("MAP")
N_ROWS = 6
# Window [00:00, 06:00) with the anchor at 05:30: with a 5 h window and a 1 h gap the
# task cutoff is 04:30, so rows 0-3 are kept and the two anchor-side rows are cut.
TEST_SPEC = TaskSpec(window_hours=5.0, prediction_gap_hours=1.0)
CLI_PATH = ROOT / "scripts" / "v2" / "run_supervised_v2.py"


def _write_episode(
    data_dir: Path,
    episode_id: str,
    subject: int,
    label: int,
    values: np.ndarray,
    missing: np.ndarray,
    feature_names: tuple[str, ...] = FEATURES,
) -> None:
    stem = f"shock_episode_{episode_id}"
    np.save(data_dir / f"{stem}.npy", values)
    np.save(data_dir / f"{stem}_missing_mask.npy", missing)
    meta = {
        "episode_id": episode_id,
        "episode_type": "positive" if label else "control",
        "stay_id": 500 + subject,
        "subject_id": 100 + subject,
        "anchor_kind": "initiation",
        "feature_names": list(feature_names),
        "shape": [int(values.shape[0]), len(feature_names)],
        "resample_minutes": 60,
        "start_time": "2150-01-01T00:00:00",
        "end_time": f"2150-01-01T{values.shape[0] - 1:02d}:00:00",
        "window_start": "2150-01-01T00:00:00",
        "window_end": "2150-01-01T05:30:00",
        "anchor_time": "2150-01-01T05:30:00" if label else "2150-01-01T17:30:00",
        "missing_mask_file": f"{stem}_missing_mask.npy",
    }
    (data_dir / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")


def make_dataset(root: Path, late_scale: float = 0.0, ragged: bool = False) -> tuple[Path, list[EpisodeRecord]]:
    """40 episodes x 6 rows x 4 features in the v1 file layout, 2 episodes per patient.

    Patients 0-11 are train, 12-15 val, 16-19 test; each has one positive and one control.
    HR carries a weak learnable shift, ``vasopressor_on`` equals the label in every row
    (the v1 leak), and MAP in the two anchor-side rows is shifted by ``late_scale * label``.
    ``ragged`` shortens the control of every third patient to 5 rows (a length leak).
    """
    data_dir = root / "episodes"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    episodes: list[EpisodeRecord] = []
    for subject in range(20):
        split = "train" if subject < 12 else ("val" if subject < 16 else "test")
        for label in (1, 0):
            episode_id = f"p{subject:02d}_{'pos' if label else 'ctl'}"
            n_rows = N_ROWS - 1 if (ragged and label == 0 and subject % 3 == 0) else N_ROWS
            values = rng.normal(size=(n_rows, len(FEATURES)))
            missing = rng.random((n_rows, len(FEATURES))) < 0.2
            values[:, 0] += 1.5 * label
            values[-2:, LATE] += late_scale * label
            values[:, LEAK] = float(label)
            missing[:, LEAK] = False
            _write_episode(data_dir, episode_id, subject, label, values, missing)
            episodes.append(EpisodeRecord(episode_id, 100 + subject, 500 + subject, label, split=split))
    return data_dir, episodes


def _config(**overrides: object) -> SupervisedConfig:
    settings: dict[str, object] = {
        "seeds": (0,),
        "max_epochs": 3,
        "early_stopping_patience": 2,
        "batch_size": 8,
        "eval_batch_size": 64,
        "hidden_size": 8,
        "lr_c_grid": (0.1, 1.0),
        "gbdt_max_depth_grid": (2, 3),
        "gbdt_max_iter": 5,
        "num_threads": 1,
    }
    settings.update(overrides)
    return SupervisedConfig(**settings)


class PerturbColumn:
    """Loader wrapper that replaces one column (values and mask) with fresh noise."""

    def __init__(self, base: FileArrayLoader, column: int, seed: int) -> None:
        self.base, self.column, self.rng = base, column, np.random.default_rng(seed)

    def __call__(self, episode: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
        values, missing = self.base(episode)
        values, missing = values.copy(), missing.copy()
        values[:, self.column] = self.rng.normal(scale=100.0, size=values.shape[0])
        missing[:, self.column] = self.rng.random(values.shape[0]) < 0.5
        return values, missing


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_supervised_v2_cli", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------ config


def test_default_config_trains_the_seven_v1_models_without_vasopressor():
    config = SupervisedConfig()
    assert config.models == tuple(get_default_registry()) == SUPERVISED_MODELS
    assert len(config.models) == 7
    assert config.excluded_features == ("vasopressor_on",)
    assert config.device == "cpu"
    assert config.max_epochs > 1 and config.early_stopping_patience >= 1
    assert SupervisedConfig(models=("grud_lite",), hidden_size=6).hidden_size == 6  # no attention heads


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"models": ("lr_stats", "transformer_seq")}, "unknown supervised models"),
        ({"models": ()}, "models must not be empty"),
        ({"seeds": (1, 1)}, "seeds must be distinct"),
        ({"max_epochs": 0}, "max_epochs"),
        ({"early_stopping_patience": 0}, "early_stopping_patience"),
        ({"models": ("mtan_seq",), "hidden_size": 6}, "divisible by 4"),
        ({"learning_rate": -1e-3}, "learning_rate"),
        ({"lr_c_grid": (0.0,)}, "lr_c_grid"),
    ],
)
def test_config_rejects_invalid_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        SupervisedConfig(**overrides)


def test_early_stopping_counts_epochs_without_improvement():
    stopper = EarlyStopping(patience=2)
    assert stopper.update(1, 0.50) and stopper.update(2, 0.75)
    assert not stopper.update(3, 0.75)  # a tie is not an improvement
    assert not stopper.should_stop
    assert not stopper.update(4, 0.5)
    assert stopper.should_stop and stopper.best_epoch == 2 and stopper.best_score == 0.75


def test_early_stopping_min_delta_nan_and_patience_bounds():
    stopper = EarlyStopping(patience=1, min_delta=0.25)
    assert stopper.update(1, 0.5)
    assert not stopper.update(2, 0.75)  # exactly min_delta better is not enough
    assert stopper.should_stop
    stopper = EarlyStopping(patience=3)
    assert not stopper.update(1, math.nan)
    assert stopper.best_epoch == 0 and stopper.bad_epochs == 1
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=0)


# ------------------------------------------------------------------------------ splits


def test_validation_split_must_exist_and_hold_both_classes(tmp_path):
    data_dir, episodes = make_dataset(tmp_path)
    loader = FileArrayLoader(data_dir)
    config = _config(models=("lr_stats",))
    no_val = [replace(e, split="train") if e.split == "val" else e for e in episodes]
    with pytest.raises(ValueError, match="validation split is empty"):
        run_supervised_v2(no_val, FEATURES, loader, config)
    one_class = [e for e in episodes if not (e.split == "val" and e.label == 0)]
    with pytest.raises(ValueError, match="validation split has a single class"):
        run_supervised_v2(one_class, FEATURES, loader, config)


def test_split_contract_rejects_patient_overlap_and_malformed_rows(tmp_path):
    _, episodes = make_dataset(tmp_path)
    groups = validate_episode_splits(episodes)
    assert {split: len(rows) for split, rows in groups.items()} == {"train": 24, "val": 8, "test": 8}
    moved = next(e for e in episodes if e.split == "val")
    overlap = [replace(e, subject_id=100) if e.episode_id == moved.episode_id else e for e in episodes]
    with pytest.raises(ValueError, match="patient-disjoint"):
        validate_episode_splits(overlap)
    with pytest.raises(ValueError, match="duplicate episode_id"):
        validate_episode_splits(episodes + [episodes[0]])
    with pytest.raises(ValueError, match="expected one of"):
        validate_episode_splits([replace(episodes[0], split="validation")] + episodes[1:])
    with pytest.raises(ValueError, match="expected 0 or 1"):
        validate_episode_splits([replace(episodes[0], label=2)] + episodes[1:])
    with pytest.raises(ValueError, match="empty"):
        validate_episode_splits([])


# ------------------------------------------------------------------------------ inputs


def test_excluded_feature_is_absent_from_every_model_input(tmp_path):
    data_dir, episodes = make_dataset(tmp_path)
    base = FileArrayLoader(data_dir)
    clean = build_supervised_inputs(episodes, FEATURES, base)
    assert clean.input_features == ("HR", "MAP", "Lactate")
    assert clean.excluded_features == ("vasopressor_on",)
    assert not any(name.startswith("vasopressor_on") for name in clean.tabular_feature_names)
    assert len(clean.tabular_feature_names) == 3 * 7  # v1 summary stats of the kept channels
    perturbed = build_supervised_inputs(episodes, FEATURES, PerturbColumn(base, LEAK, seed=1))
    for split in SPLITS:
        assert clean.splits[split].values.shape[-1] == 3
        for field in ("tabular", "values", "observed", "deltas", "lengths"):
            np.testing.assert_array_equal(getattr(clean.splits[split], field), getattr(perturbed.splits[split], field))
    config = _config(models=("lr_stats", "grud_lite"), max_epochs=2)
    for name in config.models:
        original = fit_model(name, clean, config, seed=0)
        assert original.test_probabilities == fit_model(name, perturbed, config, seed=0).test_probabilities
        assert original.n_model_inputs == (21 if name == "lr_stats" else 3)
    with pytest.raises(ValueError, match="unknown supervised model"):
        fit_model("transformer_seq", clean, config, seed=0)


def test_without_exclusion_the_leak_reaches_the_inputs(tmp_path):
    """Control for the test above: the same perturbation is visible when nothing is excluded."""
    data_dir, episodes = make_dataset(tmp_path)
    base = FileArrayLoader(data_dir)
    open_inputs = build_supervised_inputs(episodes, FEATURES, base, excluded_features=())
    perturbed = build_supervised_inputs(episodes, FEATURES, PerturbColumn(base, LEAK, seed=1), excluded_features=())
    assert "vasopressor_on__last" in open_inputs.tabular_feature_names
    assert not np.array_equal(open_inputs.splits["train"].tabular, perturbed.splits["train"].tabular)
    with pytest.raises(ValueError, match="not among the episode features"):
        build_supervised_inputs(episodes, FEATURES, base, excluded_features=("vasopresor_on",))
    with pytest.raises(ValueError, match="rebuild them"):
        fit_model("lr_stats", open_inputs, _config(models=("lr_stats",)), seed=0)
    with pytest.raises(ValueError, match=r"expected \(rows, 3\)"):
        build_supervised_inputs(episodes, FEATURES[:3], base, excluded_features=())


def test_task_window_loader_cuts_the_anchor_bins_before_features_are_built(tmp_path):
    data_a, episodes = make_dataset(tmp_path / "a", late_scale=0.0)
    data_b, _ = make_dataset(tmp_path / "b", late_scale=50.0)
    inputs_a = build_supervised_inputs(episodes, FEATURES, TaskWindowLoader(data_a, TEST_SPEC, FEATURES))
    loader_b = TaskWindowLoader(data_b, TEST_SPEC, FEATURES)
    inputs_b = build_supervised_inputs(episodes, FEATURES, loader_b)
    assert inputs_a.max_len == 4 and loader_b.describe()["rows_after_cutoff"] == {"4": 40}
    for split in SPLITS:
        for field in ("tabular", "values", "observed", "deltas", "lengths"):
            a, b = getattr(inputs_a.splits[split], field), getattr(inputs_b.splits[split], field)
            np.testing.assert_array_equal(a, b)
    stored_a = build_supervised_inputs(episodes, FEATURES, FileArrayLoader(data_a))
    stored_b = build_supervised_inputs(episodes, FEATURES, FileArrayLoader(data_b))
    assert stored_a.max_len == N_ROWS
    assert not np.array_equal(stored_a.splits["train"].tabular, stored_b.splits["train"].tabular)
    with pytest.raises(ValueError, match="task spec excludes"):
        build_supervised_inputs(episodes, FEATURES, loader_b, excluded_features=())
    with pytest.raises(ValueError, match="disagrees with metadata"):
        loader_b(replace(episodes[0], label=1 - episodes[0].label))
    with pytest.raises(ValueError, match="not a full 12 h window"):
        TaskWindowLoader(data_a, TaskSpec(window_hours=12.0), FEATURES)(episodes[0])


# ---------------------------------------------------------------------------- training


def test_lr_and_gbdt_select_on_validation_and_key_test_probabilities(tmp_path):
    data_dir, episodes = make_dataset(tmp_path)
    config = _config(models=("lr_stats", "gbdt_stats"), seeds=(0, 1))
    result = run_supervised_v2(episodes, FEATURES, FileArrayLoader(data_dir), config)
    test_ids = {e.episode_id for e in episodes if e.split == "test"}
    assert [(r.model, r.seed) for r in result.runs] == [
        ("lr_stats", 0),
        ("lr_stats", 1),
        ("gbdt_stats", 0),
        ("gbdt_stats", 1),
    ]
    for run in result.runs:
        assert set(run.test_probabilities) == test_ids
        assert all(0.0 <= p <= 1.0 for p in run.test_probabilities.values())
        assert run.test_metrics["count"] == len(test_ids)
        best = max(step["val_auprc"] for step in run.selection_trace)
        assert run.val_metrics["auprc"] == pytest.approx(best)  # the chosen candidate is the validation best
    lr_run, gbdt_run = result.runs[0], result.runs[2]
    assert lr_run.selected["C"] in config.lr_c_grid and len(lr_run.selection_trace) == len(config.lr_c_grid)
    assert gbdt_run.selected["max_depth"] in config.gbdt_max_depth_grid
    assert 1 <= gbdt_run.selected["max_iter"] <= config.gbdt_max_iter
    rows = result.prediction_rows()
    assert len(rows) == 4 * len(test_ids)
    assert {row["subject_id"] for row in rows} == {e.subject_id for e in episodes if e.split == "test"}
    summary = result.summary()
    json.dumps(summary, allow_nan=False)
    assert summary["models"]["lr_stats"]["n_seeds"] == 2


@pytest.mark.parametrize("name", SEQUENCE_MODELS)
def test_sequence_models_train_on_minibatches_and_restore_the_best_epoch(tmp_path, name):
    data_dir, episodes = make_dataset(tmp_path, ragged=True)
    with pytest.warns(UserWarning, match="window length differs by label"):
        inputs = build_supervised_inputs(episodes, FEATURES, FileArrayLoader(data_dir))
    run = fit_model(name, inputs, _config(models=(name,), max_epochs=2, early_stopping_patience=5), seed=0)
    assert run.epochs_trained == 2 and run.stopped_early is False
    assert [step["epoch"] for step in run.selection_trace] == [1, 2]
    assert run.best_epoch in (1, 2) and run.selected == {"epochs": run.best_epoch}
    best = run.selection_trace[run.best_epoch - 1]["val_auprc"]
    assert best == max(step["val_auprc"] for step in run.selection_trace)
    assert run.val_metrics["auprc"] == pytest.approx(best)  # the best epoch's weights were restored
    assert run.n_model_inputs == 3
    assert set(run.test_probabilities) == set(inputs.splits["test"].episode_ids)


def test_early_stopping_triggers_on_a_flat_validation_curve(tmp_path):
    data_dir, episodes = make_dataset(tmp_path)
    inputs = build_supervised_inputs(episodes, FEATURES, FileArrayLoader(data_dir))
    # learning_rate=0 freezes the network, so validation AUPRC is exactly flat.
    config = _config(models=("retain_seq",), learning_rate=0.0, max_epochs=20, early_stopping_patience=3)
    run = fit_model("retain_seq", inputs, config, seed=0)
    assert run.stopped_early is True
    assert run.epochs_trained == 4 and run.best_epoch == 1
    assert [step["improved"] for step in run.selection_trace] == [True, False, False, False]
    assert len({step["val_auprc"] for step in run.selection_trace}) == 1


def test_test_labels_never_influence_selection_or_predictions(tmp_path):
    data_dir, episodes = make_dataset(tmp_path)
    flipped = [replace(e, label=1 - e.label) if e.split == "test" else e for e in episodes]
    config = _config(models=("lr_stats", "grud_lite"), max_epochs=3, early_stopping_patience=1)
    loader = FileArrayLoader(data_dir)
    original = run_supervised_v2(episodes, FEATURES, loader, config)
    relabeled = run_supervised_v2(flipped, FEATURES, loader, config)
    for a, b in zip(original.runs, relabeled.runs):
        assert a.selected == b.selected and a.best_epoch == b.best_epoch
        assert a.selection_trace == b.selection_trace
        assert a.test_probabilities == b.test_probabilities  # also: same seed, same model
        assert a.test_metrics["auroc"] == pytest.approx(1.0 - b.test_metrics["auroc"])


# -------------------------------------------------------------------------- file input


def test_episode_lists_load_from_json_jsonl_and_csv(tmp_path):
    rows = [
        {"episode_id": "e1", "subject_id": 1, "stay_id": 10, "label": 1, "split": "train"},
        {"episode_id": "e2", "subject_id": "2", "stay_id": "", "episode_type": "control", "split": "val"},
    ]
    json_path = tmp_path / "episodes.json"
    json_path.write_text(json.dumps({"episodes": rows}), encoding="utf-8")
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    csv_path = tmp_path / "episodes.csv"
    columns = ["episode_id", "subject_id", "stay_id", "label", "episode_type", "split"]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    expected = [EpisodeRecord("e1", 1, 10, 1, split="train"), EpisodeRecord("e2", 2, -1, 0, split="val")]
    for path in (json_path, jsonl_path, csv_path):
        assert load_episode_list(path) == expected
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"episode_id": "e3", "label": 1}]), encoding="utf-8")
    with pytest.raises(ValueError, match="subject_id"):
        load_episode_list(bad)


def test_split_manifest_assigns_episode_level_splits(tmp_path):
    episodes = [EpisodeRecord("e1", 1, 10, 1), EpisodeRecord("e2", 2, 20, 0), EpisodeRecord("e3", 3, 30, 0)]
    manifest = tmp_path / "split.json"
    lists = {"seed": 3, "train": ["e1"], "val": ["e2"], "test": []}
    manifest.write_text(json.dumps({"split": lists}), encoding="utf-8")
    assignment = load_split_assignments(manifest)
    assert assignment == {"e1": "train", "e2": "val"}
    assigned = assign_splits(episodes, assignment)
    assert [(e.episode_id, e.split) for e in assigned] == [("e1", "train"), ("e2", "val")]  # e3 unlisted
    with pytest.raises(ValueError, match="missing from the episode list"):
        assign_splits(episodes[:1], assignment)
    with pytest.raises(ValueError, match="manifest says"):
        assign_splits([replace(episodes[0], split="test")], {"e1": "train"})
    twice = tmp_path / "twice.json"
    twice.write_text(json.dumps({"train": ["e1"], "val": ["e1"], "test": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="listed in"):
        load_split_assignments(twice)
    patient_level = tmp_path / "patients.json"
    patient_level.write_text(json.dumps({"splits": {"train": {"patient_ids": [1]}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="patient-level"):
        load_split_assignments(patient_level)
    rows = tmp_path / "rows.json"  # dumped EpisodeRecord rows, as fmg_adapter.load_split_manifest reads them
    rows.write_text(json.dumps([{"episode_id": "e1", "split": "train"}, {"episode_id": "e3", "split": ""}]), "utf-8")
    assert load_split_assignments(rows) == {"e1": "train"}
    rows.write_text(json.dumps([{"episode_id": "e1", "split": "unassigned"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected one of"):
        load_split_assignments(rows)


def test_file_loader_and_feature_names(tmp_path):
    data_dir, episodes = make_dataset(tmp_path)
    assert read_feature_names(data_dir, episodes[0].episode_id) == FEATURES
    values, missing = FileArrayLoader(data_dir)(episodes[0])
    assert values.shape == (N_ROWS, len(FEATURES)) and missing.dtype == bool
    np.save(data_dir / "shock_episode_broken.npy", np.zeros((3, 4)))
    np.save(data_dir / "shock_episode_broken_missing_mask.npy", np.zeros((2, 4), dtype=bool))
    with pytest.raises(ValueError, match="equal 2-D shapes"):
        FileArrayLoader(data_dir)(replace(episodes[0], episode_id="broken"))
    (data_dir / "shock_episode_nometa.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="feature_names"):
        read_feature_names(data_dir, "nometa")


def test_json_safe_turns_nan_into_null():
    payload = json_safe({"a": float("nan"), "b": np.int64(3), "c": (np.float32(0.5), np.bool_(True)), 4: Path("x")})
    assert payload == {"a": None, "b": 3, "c": [0.5, True], "4": "x"}
    json.dumps(payload, allow_nan=False)


# --------------------------------------------------------------------------------- CLI


def test_cli_help_lists_inputs_and_protocol(capsys):
    cli = _load_cli()
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--episodes", "--data-dir", "--split-manifest", "--task-spec", "--patience", "--exclude-features"):
        assert flag in out


def test_cli_runs_end_to_end_on_tiny_synthetic_data(tmp_path):
    cli = _load_cli()
    data_dir, episodes = make_dataset(tmp_path, late_scale=50.0)
    episodes_path = tmp_path / "episodes.json"
    episodes_path.write_text(json.dumps([dict(asdict(e), split="") for e in episodes]), encoding="utf-8")
    manifest = tmp_path / "split.json"
    split_lists = {split: [e.episode_id for e in episodes if e.split == split] for split in SPLITS}
    manifest.write_text(json.dumps({"split": split_lists}), encoding="utf-8")
    spec_path = tmp_path / "task_spec.json"
    spec_path.write_text(json.dumps({"task_spec": TEST_SPEC.to_dict()}), encoding="utf-8")
    out_dir = tmp_path / "out"
    argv = [
        "--episodes", str(episodes_path),
        "--split-manifest", str(manifest),
        "--data-dir", str(data_dir),
        "--task-spec", str(spec_path),
        "--out-dir", str(out_dir),
        "--models", "lr_stats,grud_lite",
        "--seeds", "0,1",
        "--max-epochs", "2",
        "--patience", "1",
        "--batch-size", "8",
        "--hidden-size", "8",
        "--lr-c-grid", "0.1,1",
        "--num-threads", "1",
        "--quiet",
    ]  # fmt: skip
    assert cli.main(argv) == 0
    summary = json.loads((out_dir / "supervised_v2_summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["excluded_features"] == ["vasopressor_on"]
    assert summary["data"]["input_features"] == ["HR", "MAP", "Lactate"]
    assert summary["data"]["max_len"] == 4  # anchor-side bins cut by the task spec
    assert summary["provenance"]["loader"]["window"] == "task"
    assert set(summary["models"]) == {"lr_stats", "grud_lite"}
    with open(out_dir / "supervised_v2_test_predictions.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 2 * len(split_lists["test"])
    assert {row["episode_id"] for row in rows} == set(split_lists["test"])
    assert sorted(path.name for path in (out_dir / "runs").iterdir()) == [
        "grud_lite__seed0.json",
        "grud_lite__seed1.json",
        "lr_stats__seed0.json",
        "lr_stats__seed1.json",
    ]
    with pytest.raises(SystemExit):  # an existing summary is not overwritten silently
        cli.main(argv)


# -------------------------------------------------------------------- external cohorts

HR = FEATURES.index("HR")
LACTATE = FEATURES.index("Lactate")
ARRAY_FIELDS = ("tabular", "values", "observed", "deltas", "lengths", "y")
INTERNAL_RESULT_FIELDS = (
    "selected",
    "selection_trace",
    "best_epoch",
    "epochs_trained",
    "stopped_early",
    "val_metrics",
    "test_metrics",
    "test_probabilities",
)


def make_external(
    root: Path,
    tag: str,
    n_patients: int = 6,
    hr_shift: float = 0.0,
    unavailable: str | None = None,
    drop_column: bool = False,
    fill: float = math.nan,
    seed: int = 11,
) -> tuple[Path, list[EpisodeRecord], tuple[str, ...]]:
    """An external cohort in the v1 layout: ``n_patients`` patients with one positive and one control.

    ``unavailable`` is a feature the source does not provide: missing in every bin with
    stored value ``fill`` (NaN, as the v2 extraction writes it) or, with ``drop_column``,
    absent from the arrays and the metadata. For a given ``seed`` the other columns do not
    depend on ``unavailable``, ``drop_column`` or ``fill``.
    """
    names = tuple(name for name in FEATURES if not (drop_column and name == unavailable))
    keep = [FEATURES.index(name) for name in names]
    data_dir = root / f"external_{tag}"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    episodes: list[EpisodeRecord] = []
    for patient in range(n_patients):
        subject = 1000 + patient  # subject_id 1100+, disjoint from the internal 100-119
        for label in (1, 0):
            episode_id = f"{tag}{patient:02d}_{'pos' if label else 'ctl'}"
            values = rng.normal(size=(N_ROWS, len(FEATURES)))
            missing = rng.random((N_ROWS, len(FEATURES))) < 0.2
            values[:, HR] += 1.5 * label + hr_shift
            values[:, LEAK] = float(label)
            missing[:, LEAK] = False
            if unavailable is not None:
                values[:, FEATURES.index(unavailable)] = fill
                missing[:, FEATURES.index(unavailable)] = True
            _write_episode(data_dir, episode_id, subject, label, values[:, keep], missing[:, keep], names)
            episodes.append(EpisodeRecord(episode_id, 100 + subject, 500 + subject, label, split="external"))
    return data_dir, episodes, names


def copy_as_external(data_dir: Path, episodes: list[EpisodeRecord], root: Path) -> tuple[Path, list[EpisodeRecord]]:
    """The internal test episodes, byte for byte, under new episode and subject ids."""
    target = root / "external_copy"
    target.mkdir(parents=True, exist_ok=True)
    copies: list[EpisodeRecord] = []
    for episode in episodes:
        if episode.split != "test":
            continue
        new_id = f"copy_{episode.episode_id}"
        for suffix in (".npy", "_missing_mask.npy"):
            shutil.copyfile(
                data_dir / f"shock_episode_{episode.episode_id}{suffix}", target / f"shock_episode_{new_id}{suffix}"
            )
        meta = json.loads((data_dir / f"shock_episode_{episode.episode_id}.json").read_text(encoding="utf-8"))
        meta["episode_id"] = new_id
        (target / f"shock_episode_{new_id}.json").write_text(json.dumps(meta), encoding="utf-8")
        copies.append(replace(episode, episode_id=new_id, subject_id=episode.subject_id + 10_000, split="external"))
    return target, copies


class MaskColumn:
    """Loader wrapper: one column becomes missing in every row (value NaN) for the given episodes (default all)."""

    def __init__(self, base: FileArrayLoader, column: int, episode_ids: set[str] | None = None) -> None:
        self.base, self.column, self.episode_ids = base, column, episode_ids

    def __call__(self, episode: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
        values, missing = self.base(episode)
        if self.episode_ids is None or episode.episode_id in self.episode_ids:
            values, missing = values.copy(), missing.copy()
            values[:, self.column] = np.nan
            missing[:, self.column] = True
        return values, missing


def test_external_cohort_is_prepared_with_the_internal_training_statistics(tmp_path):
    data_dir, episodes = make_dataset(tmp_path)
    inputs = build_supervised_inputs(episodes, FEATURES, FileArrayLoader(data_dir))
    copy_dir, copies = copy_as_external(data_dir, episodes, tmp_path)
    cohort = build_external_cohort("copy", copies, FileArrayLoader(copy_dir), inputs)
    assert cohort.preprocessing is inputs.preprocessing
    test = inputs.splits["test"]
    for name in ARRAY_FIELDS:  # identical data -> identical model inputs: no statistic is refit on the cohort
        np.testing.assert_array_equal(getattr(cohort.arrays, name), getattr(test, name))
    assert cohort.arrays.episode_ids == tuple(f"copy_{episode_id}" for episode_id in test.episode_ids)
    assert cohort.info["all_missing_features"] == [] and cohort.info["unused_columns"] == ["vasopressor_on"]
    # Frozen scoring of the copies reproduces the internal test predictions (selected C / depth /
    # iterations, best-epoch weights): the external pass uses the model the test pass used.
    models = ("lr_stats", "gbdt_stats", "grud_lite", "retain_seq")
    config = _config(models=models, max_epochs=6, early_stopping_patience=6)
    result = fit_supervised_models(inputs, config, externals=[cohort])
    assert any(run.best_epoch is not None and run.best_epoch < run.epochs_trained for run in result.runs)
    for run in result.runs:
        copied = run.external_probabilities["copy"]
        for episode_id, prob in run.test_probabilities.items():
            assert copied[f"copy_{episode_id}"] == pytest.approx(prob, rel=1e-6, abs=1e-7), run.model
        assert run.external_metrics["copy"]["auprc"] == pytest.approx(run.test_metrics["auprc"])

    # A cohort shifted by +100 in HR keeps the shift; re-centring on the cohort would remove it.
    base_dir, base_episodes, _ = make_external(tmp_path, "a")
    shift_dir, shift_episodes, _ = make_external(tmp_path, "b", hr_shift=100.0)
    base = build_external_cohort("a", base_episodes, FileArrayLoader(base_dir), inputs).arrays
    shifted = build_external_cohort("b", shift_episodes, FileArrayLoader(shift_dir), inputs).arrays
    observed = base.observed[..., HR] > 0
    np.testing.assert_array_equal(observed, shifted.observed[..., HR] > 0)
    gap = (shifted.values - base.values)[..., HR][observed]
    np.testing.assert_allclose(gap, 100.0 / inputs.preprocessing.sequence_stds[HR], rtol=1e-4)
    names = list(inputs.tabular_feature_names)
    shift_by_stat = {"min": 100, "max": 100, "mean": 100, "last": 100, "std": 0, "slope": 0, "missing_rate": 0}
    for stat, delta in shift_by_stat.items():
        column = names.index(f"HR__{stat}")
        np.testing.assert_allclose(shifted.tabular[:, column] - base.tabular[:, column], delta, atol=1e-6)
    for array in (inputs.preprocessing.sequence_means, inputs.preprocessing.tabular_medians):
        assert not array.flags.writeable


def test_external_data_never_change_internal_selection_or_test_scores(tmp_path):
    data_dir, episodes = make_dataset(tmp_path)
    inputs = build_supervised_inputs(episodes, FEATURES, FileArrayLoader(data_dir))
    a_dir, a_episodes, _ = make_external(tmp_path, "a")
    b_dir, b_episodes, _ = make_external(tmp_path, "b", n_patients=9, hr_shift=-50.0, unavailable="MAP", seed=5)
    cohort_a = build_external_cohort("site_a", a_episodes, FileArrayLoader(a_dir), inputs)
    cohort_b = build_external_cohort("site_b", b_episodes, FileArrayLoader(b_dir), inputs)
    config = _config(models=SUPERVISED_MODELS, max_epochs=2)
    plain = fit_supervised_models(inputs, config)
    with_a = fit_supervised_models(inputs, config, externals=[cohort_a])
    with_both = fit_supervised_models(inputs, config, externals=[cohort_b, cohort_a])
    for runs in (with_a.runs, with_both.runs):
        for reference, run in zip(plain.runs, runs, strict=True):
            for name in INTERNAL_RESULT_FIELDS:
                assert getattr(run, name) == getattr(reference, name), (run.model, name)
    for run_a, run_both in zip(with_a.runs, with_both.runs, strict=True):
        assert set(run_both.external_probabilities) == {"site_a", "site_b"}
        assert run_a.external_probabilities["site_a"] == run_both.external_probabilities["site_a"]
        assert run_a.external_metrics["site_a"] == run_both.external_metrics["site_a"]
        assert run_both.external_metrics["site_b"]["count"] == len(b_episodes)
    assert all(not run.external_probabilities and not run.external_metrics for run in plain.runs)


def test_feature_unavailable_in_an_external_cohort_enters_as_missing(tmp_path):
    data_dir, episodes = make_dataset(tmp_path)
    inputs = build_supervised_inputs(episodes, FEATURES, FileArrayLoader(data_dir))
    stats = inputs.preprocessing
    channel = inputs.input_features.index("Lactate")
    nan_dir, nan_episodes, _ = make_external(tmp_path, "n", unavailable="Lactate")
    junk_dir, junk_episodes, _ = make_external(tmp_path, "j", unavailable="Lactate", fill=1e6)
    drop_dir, drop_episodes, drop_names = make_external(tmp_path, "d", unavailable="Lactate", drop_column=True)
    assert "Lactate" not in drop_names
    stored = build_external_cohort("stored_nan", nan_episodes, FileArrayLoader(nan_dir), inputs, FEATURES)
    junk = build_external_cohort("stored_junk", junk_episodes, FileArrayLoader(junk_dir), inputs, FEATURES)
    no_column = build_external_cohort("no_column", drop_episodes, FileArrayLoader(drop_dir), inputs, drop_names)
    for cohort in (junk, no_column):  # stored values under the mask and a missing column change nothing
        for name in ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(cohort.arrays, name), getattr(stored.arrays, name))
    assert stored.info["all_missing_features"] == ["Lactate"] and stored.info["absent_features"] == []
    assert no_column.info["all_missing_features"] == ["Lactate"] and no_column.info["absent_features"] == ["Lactate"]
    assert stored.info["observed_fraction"]["Lactate"] == 0.0 and stored.info["observed_fraction"]["HR"] > 0.5

    part, names = stored.arrays, list(stats.tabular_feature_names)
    for stat in ("min", "max", "mean", "std", "last"):  # undefined -> the INTERNAL training median
        column = names.index(f"Lactate__{stat}")
        assert (part.tabular[:, column] == stats.tabular_medians[column]).all()
    assert (part.tabular[:, names.index("Lactate__slope")] == 0.0).all()
    assert (part.tabular[:, names.index("Lactate__missing_rate")] == 1.0).all()
    assert not part.observed[..., channel].any() and not part.values[..., channel].any()
    expected_deltas = np.broadcast_to(np.arange(N_ROWS, dtype=np.float32), part.deltas.shape[:2])
    np.testing.assert_array_equal(part.deltas[..., channel], expected_deltas)

    # The encoding is exactly what an internal test episode gets when that channel is never observed.
    test_ids = {episode.episode_id for episode in episodes if episode.split == "test"}
    masked = build_supervised_inputs(episodes, FEATURES, MaskColumn(FileArrayLoader(data_dir), LACTATE, test_ids))
    np.testing.assert_array_equal(masked.preprocessing.tabular_medians, stats.tabular_medians)
    np.testing.assert_array_equal(masked.preprocessing.sequence_means, stats.sequence_means)
    copy_dir, copies = copy_as_external(data_dir, episodes, tmp_path)
    copied = build_external_cohort("copy", copies, MaskColumn(FileArrayLoader(copy_dir), LACTATE), inputs)
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(copied.arrays, name), getattr(masked.splits["test"], name))

    config = _config(models=SUPERVISED_MODELS, max_epochs=2)
    result = fit_supervised_models(inputs, config, externals=[stored, no_column])
    for run in result.runs:
        probabilities = list(run.external_probabilities["stored_nan"].values())
        assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probabilities)
        assert probabilities == list(run.external_probabilities["no_column"].values())


def test_unavailable_intervention_channel_gets_the_v1_unobserved_encoding(tmp_path):
    """Like eICU ventilation_on: an intervention channel that is never missing internally."""
    data_dir, episodes = make_dataset(tmp_path)
    inputs = build_supervised_inputs(episodes, FEATURES, FileArrayLoader(data_dir), excluded_features=())
    ext_dir, ext_episodes, _ = make_external(tmp_path, "v", unavailable="vasopressor_on")
    tabular = build_external_cohort("v", ext_episodes, FileArrayLoader(ext_dir), inputs).arrays.tabular
    names = list(inputs.tabular_feature_names)
    assert (inputs.splits["train"].tabular[:, names.index("vasopressor_on__missing_rate")] == 0.0).all()
    assert (tabular[:, names.index("vasopressor_on__missing_rate")] == 1.0).all()
    for stat in ("slope", "transition_count", "on_ratio"):
        assert (tabular[:, names.index(f"vasopressor_on__{stat}")] == 0.0).all()
    for stat in ("min", "max", "mean", "std", "last"):
        column = names.index(f"vasopressor_on__{stat}")
        assert (tabular[:, column] == inputs.preprocessing.tabular_medians[column]).all()


def test_external_cohort_contract(tmp_path):
    data_dir, episodes = make_dataset(tmp_path)
    inputs = build_supervised_inputs(episodes, FEATURES, FileArrayLoader(data_dir))
    ext_dir, ext_episodes, _ = make_external(tmp_path, "x")
    loader = FileArrayLoader(ext_dir)
    with pytest.raises(ValueError, match="single class"):
        build_external_cohort("x", [e for e in ext_episodes if e.label == 1], loader, inputs)
    with pytest.raises(ValueError, match="duplicate episode_id"):
        build_external_cohort("x", ext_episodes + ext_episodes[:1], loader, inputs)
    with pytest.raises(ValueError, match="shares 1 episode ids with the internal episodes"):
        build_external_cohort("x", ext_episodes + [episodes[0]], loader, inputs)
    with pytest.raises(ValueError, match="no episodes"):
        build_external_cohort("x", [], loader, inputs)
    with pytest.raises(ValueError, match="must match"):
        build_external_cohort("e icu", ext_episodes, loader, inputs)
    with pytest.raises(ValueError, match="feature names must be unique"):
        build_external_cohort("x", ext_episodes, loader, inputs, ("HR", "HR", "MAP", "Lactate"))
    with pytest.raises(ValueError, match=r"expected \(rows, 3\)"):
        build_external_cohort("x", ext_episodes, loader, inputs, ("HR", "MAP", "Lactate"))
    with pytest.raises(ValueError, match="no training statistics"):
        build_external_cohort("x", ext_episodes, loader, replace(inputs, preprocessing=None))
    task_inputs = build_supervised_inputs(episodes, FEATURES, TaskWindowLoader(data_dir, TEST_SPEC, FEATURES))
    with pytest.raises(ValueError, match="at most 4 rows"):  # stored external windows have 6 rows
        build_external_cohort("x", ext_episodes, loader, task_inputs)
    open_inputs = build_supervised_inputs(episodes, FEATURES, FileArrayLoader(data_dir), excluded_features=())
    with pytest.raises(ValueError, match="task spec excludes"):
        build_external_cohort("x", ext_episodes, TaskWindowLoader(ext_dir, TEST_SPEC, FEATURES), open_inputs)
    with pytest.warns(UserWarning, match="also occur in the internal episodes"):
        build_external_cohort("x", [replace(e, subject_id=100) for e in ext_episodes], loader, inputs)
    config = _config(models=("lr_stats",))
    other_inputs = build_supervised_inputs(episodes, FEATURES, FileArrayLoader(data_dir))
    foreign = build_external_cohort("x", ext_episodes, loader, other_inputs)
    with pytest.raises(ValueError, match="not prepared with these inputs"):
        fit_model("lr_stats", inputs, config, seed=0, externals=[foreign])
    cohort = build_external_cohort("x", ext_episodes, loader, inputs)
    with pytest.raises(ValueError, match="duplicate external cohort names"):
        fit_supervised_models(inputs, config, externals=[cohort, cohort])


def test_external_summary_reports_seed_spread_patient_bootstrap_and_counts(tmp_path):
    data_dir, episodes = make_dataset(tmp_path)
    a_dir, a_episodes, _ = make_external(tmp_path, "a")
    b_dir, b_episodes, b_names = make_external(
        tmp_path, "b", n_patients=8, unavailable="Lactate", drop_column=True, seed=3
    )
    sources = [
        ExternalSource("site_a", a_episodes, FileArrayLoader(a_dir)),
        ExternalSource("site_b", b_episodes, FileArrayLoader(b_dir), b_names),
    ]
    config = _config(models=("lr_stats", "grud_lite"), seeds=(1, 0), max_epochs=2)
    result = run_supervised_v2(episodes, FEATURES, FileArrayLoader(data_dir), config, externals=sources)

    rows = result.external_prediction_rows()
    assert len(rows) == 2 * 2 * (len(a_episodes) + len(b_episodes))
    assert set(rows[0]) == {"model", "seed", "cohort", "episode_id", "subject_id", "label", "prob"}
    expected = {("site_a", e.episode_id, e.subject_id, e.label) for e in a_episodes}
    expected |= {("site_b", e.episode_id, e.subject_id, e.label) for e in b_episodes}
    for model in config.models:
        for seed in config.seeds:
            chosen = [r for r in rows if r["model"] == model and r["seed"] == seed]
            assert {(r["cohort"], r["episode_id"], r["subject_id"], r["label"]) for r in chosen} == expected

    summary = result.summary(external_n_boot=40, external_bootstrap_seed=7)
    json.dumps(summary, allow_nan=False)
    external = summary["external"]
    assert external["bootstrap"] == {
        "function": "src.icu.v2.stats.patient_cluster_bootstrap",
        "resample_unit": "subject_id",
        "model_seed": 0,
        "n_boot": 40,
        "ci": 0.95,
        "seed": 7,
    }
    assert set(external["models"]) == {"lr_stats", "grud_lite"}
    assert external["missingness_by_model"] == {name: MISSINGNESS_BY_MODEL[name] for name in config.models}
    assert external["cohorts"]["site_b"]["absent_features"] == ["Lactate"]
    assert external["frozen_preprocessing"]["input_features"] == ["HR", "MAP", "Lactate"]
    assert len(external["frozen_preprocessing"]["tabular_medians"]) == 21
    assert set(external["frozen_preprocessing"]["internal_train_observed_fraction"]) == {"HR", "MAP", "Lactate"}
    for per_cohort in external["models"].values():
        assert set(per_cohort) == {"site_a", "site_b"}
        for cohort, block in per_cohort.items():
            cohort_episodes = a_episodes if cohort == "site_a" else b_episodes
            positives = sum(e.label for e in cohort_episodes)
            assert block["episodes"] == len(cohort_episodes) and block["positives"] == positives
            assert block["controls"] == len(cohort_episodes) - positives
            assert block["patients"] == len({e.subject_id for e in cohort_episodes})
            assert block["prevalence"] == pytest.approx(positives / len(cohort_episodes))
            assert block["n_seeds"] == 2 and [step["seed"] for step in block["per_seed"]] == [1, 0]
            for metric in ("auprc", "auroc"):
                per_seed = [step[metric] for step in block["per_seed"]]
                assert block[f"{metric}_mean"] == pytest.approx(np.mean(per_seed))
                assert block[f"{metric}_std"] == pytest.approx(np.std(per_seed, ddof=1))
                interval = block["bootstrap"][metric]
                assert interval["point"] == pytest.approx(per_seed[1])  # model seed 0
                assert interval["n_boot"] == 40 and interval["seed"] == 7 and interval["ci"] == 0.95
                assert interval["n_groups"] == block["patients"] and interval["resample_unit"] == "cluster"
                assert 0.0 <= interval["low"] <= interval["high"] <= 1.0
            assert block["bootstrap"]["model_seed"] == 0
    seed_block = summary["models"]["lr_stats"]["seeds"][0]
    assert set(seed_block["external_metrics"]) == {"site_a", "site_b"}
    assert "external_probabilities" not in seed_block and "external_probabilities" in result.runs[0].to_dict()

    plain = run_supervised_v2(episodes, FEATURES, FileArrayLoader(data_dir), config)
    assert "external" not in plain.summary() and plain.external_prediction_rows() == []
    assert set(plain.runs[0].to_dict()) == {
        "model", "seed", "family", "selected", "best_epoch", "epochs_trained", "stopped_early",
        "n_model_inputs", "val_metrics", "test_metrics", "fit_seconds", "selection_trace", "test_probabilities",
    }  # fmt: skip
    with pytest.raises(ValueError, match="no external cohort"):
        plain.external_summary()
    assert DEFAULT_EXTERNAL_N_BOOT == 1000


def test_cli_parses_repeatable_external_cohorts(capsys):
    cli = _load_cli()
    spec = cli.parse_external_spec(
        "eicu=artifacts/v2/task/x_eicu_gap1h/episodes_all.json::data/processed/v2/extract/eicu_gap1h_r2"
    )
    assert tuple(spec) == (
        "eicu",
        "artifacts/v2/task/x_eicu_gap1h/episodes_all.json",
        "data/processed/v2/extract/eicu_gap1h_r2",
    )
    assert spec.name == "eicu" and spec.data_dir.endswith("eicu_gap1h_r2")
    windows = cli.parse_external_spec(r"hirid=D:\task\episodes_all.json::D:\extract\hirid")
    assert windows == ("hirid", r"D:\task\episodes_all.json", r"D:\extract\hirid")
    for bad in ("eicu", "eicu=a.json", "=a.json::d", "e icu=a.json::d", "eicu=a.json::", "eicu=::d", "eicu=a::b::c"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_external_spec(bad)
    base = ["--episodes", "e.json", "--data-dir", "d", "--out-dir", "o"]
    args = cli.build_parser().parse_args(base)
    assert args.external is None and args.external_n_boot == 1000 and args.external_bootstrap_seed == 0
    args = cli.build_parser().parse_args(
        base + ["--external", "eicu=a.json::x", "--external", "hirid=b.json::y", "--external-n-boot", "50"]
    )
    assert [tuple(spec) for spec in args.external] == [("eicu", "a.json", "x"), ("hirid", "b.json", "y")]
    assert args.external_n_boot == 50
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(base + ["--external", "eicu=a.json"])
    assert "NAME=EPISODES_JSON::DATA_DIR" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "--external NAME=EPISODES_JSON::DATA_DIR" in out and "--external-n-boot" in out


def test_cli_scores_external_cohorts_with_frozen_models(tmp_path, capsys):
    cli = _load_cli()
    data_dir, episodes = make_dataset(tmp_path, late_scale=50.0)
    episodes_path = tmp_path / "episodes.json"
    episodes_path.write_text(json.dumps([asdict(e) for e in episodes]), encoding="utf-8")
    spec_path = tmp_path / "task_spec.json"
    spec_path.write_text(json.dumps(TEST_SPEC.to_dict()), encoding="utf-8")
    a_dir, a_episodes, _ = make_external(tmp_path, "a")
    b_dir, b_episodes, _ = make_external(tmp_path, "b", n_patients=5, unavailable="MAP", seed=4)
    lists = {}
    for tag, cohort_episodes in (("a", a_episodes), ("b", b_episodes)):
        lists[tag] = tmp_path / f"external_{tag}.json"
        lists[tag].write_text(json.dumps([asdict(e) for e in cohort_episodes]), encoding="utf-8")
    common = [
        "--episodes", str(episodes_path),
        "--data-dir", str(data_dir),
        "--task-spec", str(spec_path),
        "--models", "lr_stats,gbdt_stats,grud_lite",
        "--seeds", "0,1",
        "--max-epochs", "2",
        "--patience", "1",
        "--batch-size", "8",
        "--hidden-size", "8",
        "--lr-c-grid", "0.1,1",
        "--gbdt-max-depths", "2",
        "--gbdt-max-iter", "5",
        "--num-threads", "1",
        "--quiet",
    ]  # fmt: skip
    externals = ["--external", f"site_a={lists['a']}::{a_dir}", "--external", f"site_b={lists['b']}::{b_dir}"]
    out_dir = tmp_path / "out"
    assert cli.main(common + externals + ["--out-dir", str(out_dir), "--external-n-boot", "30"]) == 0

    summary = json.loads((out_dir / "supervised_v2_summary.json").read_text(encoding="utf-8"))
    assert summary["data"]["max_len"] == 4  # the external windows were cut by the same task spec
    external = summary["external"]
    assert external["bootstrap"]["n_boot"] == 30 and external["bootstrap"]["model_seed"] == 0
    assert external["cohorts"]["site_a"]["loader"]["rows_after_cutoff"] == {"4": len(a_episodes)}
    assert external["cohorts"]["site_b"]["all_missing_features"] == ["MAP"]
    for model in ("lr_stats", "gbdt_stats", "grud_lite"):
        for cohort in ("site_a", "site_b"):
            block = external["models"][model][cohort]
            for key in ("auprc_mean", "auprc_std", "auroc_mean", "auroc_std", "prevalence", "episodes", "patients"):
                assert block[key] is not None, (model, cohort, key)
            assert {"low", "high", "point"} <= set(block["bootstrap"]["auprc"])
            assert {"low", "high", "point"} <= set(block["bootstrap"]["auroc"])
    assert [entry["name"] for entry in summary["provenance"]["externals"]] == ["site_a", "site_b"]
    assert summary["provenance"]["externals"][1]["episodes_listed"] == len(b_episodes)

    with open(out_dir / "supervised_v2_external_predictions.csv", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        external_rows = list(reader)
    assert reader.fieldnames == ["model", "seed", "cohort", "episode_id", "subject_id", "label", "prob"]
    assert len(external_rows) == 3 * 2 * (len(a_episodes) + len(b_episodes))
    assert {row["cohort"] for row in external_rows} == {"site_a", "site_b"}
    with open(out_dir / "supervised_v2_test_predictions.csv", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        test_rows = list(reader)
    assert reader.fieldnames == ["model", "seed", "episode_id", "subject_id", "label", "prob"]
    assert len(test_rows) == 3 * 2 * sum(e.split == "test" for e in episodes)
    run_file = json.loads((out_dir / "runs" / "gbdt_stats__seed1.json").read_text(encoding="utf-8"))
    assert set(run_file["external_metrics"]) == set(run_file["external_probabilities"]) == {"site_a", "site_b"}

    # Bad --external arguments fail before anything is loaded or written.
    capsys.readouterr()
    bad_runs = {
        "distinct": common + externals[:2] + ["--external", f"site_a={lists['b']}::{b_dir}"],
        "shares": common + ["--external", f"site_x={episodes_path}::{data_dir}"],
        "does not exist": common + ["--external", f"site_x={lists['a']}::{tmp_path / 'nowhere'}"],
        "external-n-boot": common + externals + ["--external-n-boot", "0"],
    }
    for message, argv in bad_runs.items():
        with pytest.raises(SystemExit):
            cli.main(argv + ["--out-dir", str(tmp_path / "rejected")])
        assert message in capsys.readouterr().err
    assert not (tmp_path / "rejected").exists()
