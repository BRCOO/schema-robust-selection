#!/usr/bin/env python3
"""Train the v2 supervised reference models: leakage-safe inputs, validation early stopping.

Fixes audit P0-5: v1 trained these seven models for one full-batch step (epochs=1),
without validation data, on inputs that contained vasopressor_on. The protocol and the
exact differences from v1 are documented in src/icu/v2/supervised.py.

Inputs
  --episodes        v2 episode list: JSON list (or {"episodes": [...]}), JSONL or CSV of
                    EpisodeRecord rows with episode_id, subject_id, label (or episode_type)
                    and split (train/val/test) unless --split-manifest assigns it. Save the
                    episodes returned by splits.patient_level_split here.
  --split-manifest  optional episode-level manifest: {"split": {"train": [...], ...}} (v1
                    layout) or a list of {episode_id, split} rows, as for run_fmg_train_only.py;
                    episodes it does not list are dropped.
  --data-dir        per-episode files shock_episode_<id>.npy, *_missing_mask.npy and
                    shock_episode_<id>.json (v1 layout). Episodes are opened by constructed
                    path only; the directory is never listed.
  --window task     (default) cut each v1 window at the v2 prediction cutoff with
                    task.transform_episode, using --task-spec or the TaskSpec defaults;
                    the spec's excluded_features are always excluded too.
  --window stored   use the arrays as stored (only for arrays that already hold v2 windows).
  --external        repeatable NAME=EPISODES_JSON::DATA_DIR. An external cohort (e.g. eICU,
                    HiRID) scored with every frozen (model, seed). EPISODES_JSON lists
                    EpisodeRecord rows (episode_id, subject_id, label; the split field is
                    ignored); DATA_DIR holds the cohort's per-episode files, read with the
                    same --window, --task-spec and file patterns as the internal ones. Column
                    names come from the cohort's first episode metadata and are matched to the
                    internal input features by name.

External cohorts are scored frozen: once a (model, seed) has been selected on internal
validation data and has scored the internal test split, each external cohort is scored once
with that model and the INTERNAL preprocessing (training medians, z-scoring statistics,
input features, padded length). Nothing is refit or recalibrated, and external data never
reach selection. A feature a source does not provide (NaN with mask True in every bin, or
no column) enters as missing in every bin, encoded like an unobserved internal channel; how
each model family consumes missingness is documented in src/icu/v2/supervised.py.

Outputs (in --out-dir)
  supervised_v2_summary.json          config, inputs, per-seed selections and metrics; with
                                      --external also an "external" block: per model and
                                      cohort the seed mean/std of AUPRC and AUROC, a patient-
                                      clustered bootstrap CI for model seed 0, prevalence and
                                      counts, plus the frozen preprocessing statistics
  supervised_v2_test_predictions.csv  model, seed, episode_id, subject_id, label, prob
  supervised_v2_external_predictions.csv
                                      with --external: model, seed, cohort, episode_id,
                                      subject_id, label, prob
  runs/<model>__seed<seed>.json       one file per finished fit, so partial runs survive
                                      (with --external also external metrics/probabilities)

CPU only, and only when the machine is free:
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python \\
      scripts/v2/run_supervised_v2.py --episodes <episodes.json> \\
      --data-dir data/processed/mimic-iv-full-cw3-gap0-real/mimic-iv \\
      --task-spec <task_spec.json> --out-dir artifacts/v2/supervised/<run>

Frozen transfer to external cohorts (stored v2 extraction windows):
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python \\
      scripts/v2/run_supervised_v2.py \\
      --episodes artifacts/v2/task/x_mimic_iv_gap1h/episodes_split.json \\
      --data-dir data/processed/v2/extract/mimic_iv_gap1h_r2 --window stored --exclude-features '' \\
      --external eicu=artifacts/v2/task/x_eicu_gap1h/episodes_all.json::data/processed/v2/extract/eicu_gap1h_r2 \\
      --external hirid=artifacts/v2/task/x_hirid_gap1h/episodes_all.json::data/processed/v2/extract/hirid_gap1h_r2 \\
      --out-dir artifacts/v2/supervised/<run> --num-threads 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, NamedTuple, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.supervised import (  # noqa: E402
    DATA_PATTERN,
    DEFAULT_EXTERNAL_N_BOOT,
    EXTERNAL_NAME_PATTERN,
    MASK_PATTERN,
    ExternalCohort,
    FileArrayLoader,
    SeedResult,
    SupervisedConfig,
    TaskWindowLoader,
    assign_splits,
    build_external_cohort,
    build_supervised_inputs,
    fit_supervised_models,
    json_safe,
    load_episode_list,
    load_split_assignments,
    read_feature_names,
    validate_external_episodes,
)

SUMMARY_NAME = "supervised_v2_summary.json"
PREDICTIONS_NAME = "supervised_v2_test_predictions.csv"
PREDICTION_FIELDS = ("model", "seed", "episode_id", "subject_id", "label", "prob")
EXTERNAL_PREDICTIONS_NAME = "supervised_v2_external_predictions.csv"
EXTERNAL_PREDICTION_FIELDS = ("model", "seed", "cohort", "episode_id", "subject_id", "label", "prob")
EXTERNAL_SEPARATOR = "::"
RUNS_DIR = "runs"
_DEFAULTS = SupervisedConfig()


class ExternalSpec(NamedTuple):
    """One ``--external NAME=EPISODES_JSON::DATA_DIR`` argument."""

    name: str
    episodes: str
    data_dir: str


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keep the module docstring layout and show argument defaults (except None/False)."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)


def _csv_list(text: str | None) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _log(message: str) -> None:
    print(f"[supervised-v2 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _fmt(value: float | None) -> str:
    return "nan" if value is None else f"{value:.4f}"


def parse_external_spec(text: str) -> ExternalSpec:
    """Parse ``NAME=EPISODES_JSON::DATA_DIR``; only ``::`` separates, so ``D:\\...`` paths work."""
    name, equals, rest = str(text).partition("=")
    name = name.strip()
    if not equals or EXTERNAL_NAME_PATTERN.fullmatch(name) is None:
        raise argparse.ArgumentTypeError(
            f"expected NAME=EPISODES_JSON::DATA_DIR with NAME matching {EXTERNAL_NAME_PATTERN.pattern}, got {text!r}"
        )
    if rest.count(EXTERNAL_SEPARATOR) != 1:
        raise argparse.ArgumentTypeError(
            f"expected exactly one {EXTERNAL_SEPARATOR!r} between EPISODES_JSON and DATA_DIR, got {text!r}"
        )
    episodes, _, data_dir = (part.strip() for part in rest.partition(EXTERNAL_SEPARATOR))
    if not episodes or not data_dir:
        raise argparse.ArgumentTypeError(f"EPISODES_JSON and DATA_DIR must not be empty, got {text!r}")
    return ExternalSpec(name, episodes, data_dir)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface; training defaults come from ``SupervisedConfig``."""
    parser = argparse.ArgumentParser(prog="run_supervised_v2.py", description=__doc__, formatter_class=_HelpFormatter)
    io = parser.add_argument_group("inputs and outputs")
    io.add_argument("--episodes", required=True, help="v2 episode list (.json, .jsonl or .csv)")
    io.add_argument("--data-dir", required=True, help="directory with the per-episode arrays and metadata")
    io.add_argument("--out-dir", required=True, help="output directory, e.g. artifacts/v2/supervised/<run>")
    io.add_argument("--split-manifest", default=None, help="episode-level split manifest JSON")
    io.add_argument("--window", choices=("task", "stored"), default="task", help="how episode windows are cut")
    io.add_argument("--task-spec", default=None, help="TaskSpec JSON (bare, or under 'task_spec'); default TaskSpec()")
    io.add_argument("--feature-names", default=None, help="comma list; default: first episode's metadata")
    io.add_argument("--data-pattern", default=DATA_PATTERN, help="array file name (--window stored)")
    io.add_argument("--mask-pattern", default=MASK_PATTERN, help="mask file name (--window stored)")
    io.add_argument("--overwrite", action="store_true", help="replace an existing summary in --out-dir")
    io.add_argument("--quiet", action="store_true", help="no per-epoch log lines")

    train = parser.add_argument_group("training protocol")
    train.add_argument("--models", default=",".join(_DEFAULTS.models), help="comma list of v1 registry names")
    train.add_argument("--seeds", default=",".join(str(seed) for seed in _DEFAULTS.seeds), help="comma list")
    train.add_argument("--max-epochs", type=int, default=_DEFAULTS.max_epochs, help="neural epoch cap")
    train.add_argument(
        "--patience", type=int, default=_DEFAULTS.early_stopping_patience, help="epochs without val gain"
    )
    train.add_argument("--min-delta", type=float, default=_DEFAULTS.early_stopping_min_delta, help="val AUPRC gain")
    train.add_argument("--batch-size", type=int, default=_DEFAULTS.batch_size, help="training mini-batch")
    train.add_argument("--eval-batch-size", type=int, default=_DEFAULTS.eval_batch_size, help="scoring batch")
    train.add_argument("--learning-rate", type=float, default=_DEFAULTS.learning_rate, help="AdamW learning rate")
    train.add_argument("--weight-decay", type=float, default=_DEFAULTS.weight_decay, help="AdamW weight decay")
    train.add_argument("--hidden-size", type=int, default=_DEFAULTS.hidden_size, help="width of every neural model")
    train.add_argument(
        "--lr-c-grid", default=",".join(f"{c:g}" for c in _DEFAULTS.lr_c_grid), help="LR C candidates"
    )
    train.add_argument(
        "--gbdt-max-depths",
        default=",".join(str(depth) for depth in _DEFAULTS.gbdt_max_depth_grid),
        help="GBDT max_depth candidates",
    )
    train.add_argument("--gbdt-max-iter", type=int, default=_DEFAULTS.gbdt_max_iter, help="GBDT iteration cap")
    train.add_argument(
        "--exclude-features",
        default=",".join(_DEFAULTS.excluded_features),
        help="comma list removed before feature building ('' for none)",
    )
    train.add_argument("--device", default=_DEFAULTS.device, help="torch device; keep cpu")
    train.add_argument("--num-threads", type=int, default=None, help="cap torch/BLAS/OpenMP threads")

    external = parser.add_argument_group("external cohorts (frozen transfer)")
    external.add_argument(
        "--external",
        action="append",
        type=parse_external_spec,
        default=None,
        metavar="NAME=EPISODES_JSON::DATA_DIR",
        help="external cohort scored with every frozen (model, seed); repeatable",
    )
    external.add_argument(
        "--external-n-boot",
        type=int,
        default=DEFAULT_EXTERNAL_N_BOOT,
        help="patient-clustered bootstrap resamples per model and cohort (model seed 0)",
    )
    external.add_argument("--external-bootstrap-seed", type=int, default=0, help="random seed of that bootstrap")
    return parser


def _load_task_spec(path: str | None) -> Any:
    """TaskSpec from JSON (a bare object or one with a 'task_spec' key); defaults if no path."""
    from src.icu.v2.task import TaskSpec

    if not path:
        return TaskSpec()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("task_spec"), dict):
        payload = payload["task_spec"]
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: a TaskSpec must be a JSON object")
    unknown = sorted(set(payload) - {field.name for field in fields(TaskSpec)})
    if unknown:
        raise ValueError(f"{path}: unknown TaskSpec fields {unknown}")
    return TaskSpec(**payload)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _versions() -> dict[str, str]:
    import numpy
    import sklearn

    versions = {"python": platform.python_version(), "numpy": numpy.__version__, "sklearn": sklearn.__version__}
    if "torch" in sys.modules:
        versions["torch"] = str(sys.modules["torch"].__version__)
    return versions


def _write_predictions(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] = PREDICTION_FIELDS) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _read_external_lists(
    parser: argparse.ArgumentParser,
    specs: Sequence[ExternalSpec],
    internal_ids: set[str],
) -> list[tuple[ExternalSpec, list[Any], tuple[str, ...]]]:
    """Load and check every external episode list before the internal data are read."""
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        parser.error(f"--external names must be distinct, got {names}")
    lists = []
    for spec in specs:
        if not Path(spec.episodes).is_file():
            parser.error(f"--external {spec.name}: episode list {spec.episodes} does not exist")
        if not Path(spec.data_dir).is_dir():
            parser.error(f"--external {spec.name}: data directory {spec.data_dir} does not exist")
        try:
            episodes = load_episode_list(spec.episodes)
            validate_external_episodes(spec.name, episodes, internal_ids)
            feature_names = read_feature_names(spec.data_dir, episodes[0].episode_id)
        except (OSError, ValueError) as exc:
            parser.error(f"--external {spec.name}: {exc}")
        lists.append((spec, episodes, feature_names))
    return lists


def _describe_cohort(cohort: ExternalCohort) -> str:
    info = cohort.info
    return (
        f"external {cohort.name}: {info['episodes']} episodes, {info['positives']} positives "
        f"(prevalence {info['prevalence']:.3f}), {info['patients']} patients; "
        f"all-missing input features {info['all_missing_features']}; no column for {info['absent_features']}; "
        f"unused columns {info['unused_columns']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # keep torch off the GPU even with a CUDA build
    out_dir = Path(args.out_dir)
    summary_path = out_dir / SUMMARY_NAME
    if summary_path.exists() and not args.overwrite:
        parser.error(f"{summary_path} exists; pass --overwrite to replace it")
    if args.external_n_boot < 1:
        parser.error("--external-n-boot must be >= 1")
    started = time.perf_counter()
    log = None if args.quiet else _log

    episodes = load_episode_list(args.episodes)
    n_listed = len(episodes)
    if args.split_manifest:
        episodes = assign_splits(episodes, load_split_assignments(args.split_manifest))
    if not episodes:
        parser.error("no episodes to train on")
    data_dir = Path(args.data_dir)
    if args.feature_names:
        feature_names = tuple(_csv_list(args.feature_names))
    else:
        feature_names = read_feature_names(data_dir, episodes[0].episode_id)
    excluded = _csv_list(args.exclude_features)
    task_spec: Any = None
    if args.window == "task":
        task_spec = _load_task_spec(args.task_spec)
        loader: Any = TaskWindowLoader(data_dir, task_spec, feature_names)
        excluded = list(dict.fromkeys(excluded + list(task_spec.excluded_features)))
    else:
        loader = FileArrayLoader(data_dir, args.data_pattern, args.mask_pattern)
    try:
        config = SupervisedConfig(
            models=_csv_list(args.models),
            seeds=[int(seed) for seed in _csv_list(args.seeds)],
            max_epochs=args.max_epochs,
            early_stopping_patience=args.patience,
            early_stopping_min_delta=args.min_delta,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_size=args.hidden_size,
            lr_c_grid=[float(c) for c in _csv_list(args.lr_c_grid)],
            gbdt_max_depth_grid=[int(depth) for depth in _csv_list(args.gbdt_max_depths)],
            gbdt_max_iter=args.gbdt_max_iter,
            excluded_features=excluded,
            device=args.device,
            num_threads=args.num_threads,
        )
    except ValueError as exc:
        parser.error(str(exc))
    external_lists = _read_external_lists(
        parser, list(args.external or ()), {episode.episode_id for episode in episodes}
    )
    if not config.excluded_features:
        _log("WARNING: no feature is excluded; under the v1 task vasopressor_on reveals the label")
    _log(
        f"{len(episodes)} of {n_listed} listed episodes; window={args.window}; "
        f"excluded={list(config.excluded_features)}; models={list(config.models)}; seeds={list(config.seeds)}"
    )

    inputs = build_supervised_inputs(episodes, feature_names, loader, config.excluded_features, log=log)
    _log(f"inputs: {json.dumps(inputs.summary()['splits'])}; max_len={inputs.max_len}")
    cohorts: list[ExternalCohort] = []
    external_provenance: list[dict[str, Any]] = []
    for spec, external_episodes, external_names in external_lists:
        external_dir = Path(spec.data_dir)
        if args.window == "task":
            external_loader: Any = TaskWindowLoader(external_dir, task_spec, external_names)
        else:
            external_loader = FileArrayLoader(external_dir, args.data_pattern, args.mask_pattern)
        cohort = build_external_cohort(spec.name, external_episodes, external_loader, inputs, external_names, log=log)
        cohorts.append(cohort)
        _log(_describe_cohort(cohort))
        external_provenance.append(
            {
                "name": spec.name,
                "episodes": spec.episodes,
                "episodes_sha256": _sha256(spec.episodes),
                "data_dir": spec.data_dir,
                "loader": cohort.info.get("loader"),
                "feature_names": list(external_names),
                "episodes_listed": len(external_episodes),
            }
        )
    runs_dir = out_dir / RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)

    def on_result(result: SeedResult) -> None:
        path = runs_dir / f"{result.model}__seed{result.seed}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
        external = "".join(
            f" {name}_auprc={_fmt(metrics.get('auprc'))}" for name, metrics in result.external_metrics.items()
        )
        _log(
            f"{result.model} seed={result.seed} selected={dict(result.selected)} "
            f"val_auprc={_fmt(result.val_metrics.get('auprc'))} "
            f"test_auprc={_fmt(result.test_metrics.get('auprc'))}{external} ({result.fit_seconds:.1f}s)"
        )

    result = fit_supervised_models(inputs, config, on_result=on_result, log=log, externals=cohorts)
    if cohorts:
        _log(f"patient-clustered bootstrap of the external predictions (n_boot={args.external_n_boot})")
    summary = result.summary(
        external_n_boot=args.external_n_boot, external_bootstrap_seed=args.external_bootstrap_seed
    )
    provenance: dict[str, Any] = {
        "script": "scripts/v2/run_supervised_v2.py",
        "argv": list(sys.argv[1:] if argv is None else argv),
        "episodes": str(args.episodes),
        "episodes_sha256": _sha256(args.episodes),
        "split_manifest": args.split_manifest,
        "split_manifest_sha256": _sha256(args.split_manifest) if args.split_manifest else None,
        "data_dir": str(data_dir),
        "loader": loader.describe(),
        "feature_names": list(feature_names),
        "episodes_listed": n_listed,
        "episodes_used": len(episodes),
        "wall_seconds": time.perf_counter() - started,
        "versions": _versions(),
    }
    if external_provenance:
        provenance["externals"] = external_provenance
    summary["provenance"] = json_safe(provenance)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    _write_predictions(out_dir / PREDICTIONS_NAME, result.prediction_rows())
    if cohorts:
        _write_predictions(
            out_dir / EXTERNAL_PREDICTIONS_NAME, result.external_prediction_rows(), EXTERNAL_PREDICTION_FIELDS
        )
    for name, block in summary["models"].items():
        over = block["over_seeds"]
        _log(
            f"{name}: test AUPRC {_fmt(over['test_auprc_mean'])} +/- {_fmt(over['test_auprc_std'])} "
            f"over {block['n_seeds']} seeds"
        )
    for name, per_cohort in summary.get("external", {}).get("models", {}).items():
        for cohort_name, block in per_cohort.items():
            line = (
                f"{name} -> {cohort_name}: AUPRC {_fmt(block['auprc_mean'])} +/- {_fmt(block['auprc_std'])}, "
                f"AUROC {_fmt(block['auroc_mean'])} +/- {_fmt(block['auroc_std'])} over {block['n_seeds']} seeds"
            )
            boot = block.get("bootstrap")
            if boot:
                auprc, auroc = boot["auprc"], boot["auroc"]
                line += (
                    f"; seed {boot['model_seed']} AUPRC {_fmt(auprc['point'])} "
                    f"[{_fmt(auprc['low'])}, {_fmt(auprc['high'])}], AUROC {_fmt(auroc['point'])} "
                    f"[{_fmt(auroc['low'])}, {_fmt(auroc['high'])}]"
                )
            _log(f"{line}; prevalence {_fmt(block['prevalence'])}")
    if cohorts:
        _log(f"wrote {summary_path}, {out_dir / PREDICTIONS_NAME} and {out_dir / EXTERNAL_PREDICTIONS_NAME}")
    else:
        _log(f"wrote {summary_path} and {out_dir / PREDICTIONS_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
