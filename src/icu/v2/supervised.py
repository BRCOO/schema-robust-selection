"""Supervised reference models for the v2 protocol.

Audit finding P0-5 (``docs/reviews/ICDE2027_558_audit_and_revision_plan.md``): the v1
internal utility table trained its supervised reference models with ``epochs=1``
(``scripts/run_comparator_supervised_consistent.py`` -> ``comparators/fit_all.py``), which
is a single full-batch gradient step for the neural models; the evaluated slice had no
validation episodes; and ``comparators/feature_builder.py`` gave every model
``vasopressor_on`` (for example ``vasopressor_on__last``), the feature that defines the
anchor and therefore the label under the v1 task.

v2 trains the same seven models under these rules:

* Leakage-safe inputs. ``SupervisedConfig.excluded_features`` (default
  ``("vasopressor_on",)``) is removed from the raw arrays before any feature is built.
  :class:`TaskWindowLoader` also cuts every v1 window at the v2 prediction cutoff
  (``task.transform_episode``) before that.
* Selection on validation only. LR ``C``, GBDT ``max_depth`` and boosting iterations,
  and the neural stopping epoch are chosen by validation AUPRC. An empty or
  single-class validation split raises; there is no fallback to training data.
* One test pass per (model, seed), after selection. Test probabilities are returned
  keyed by episode id, with subject ids, for the patient-clustered bootstrap
  (``stats.patient_cluster_bootstrap``).

Reused from v1 (imported only; nothing is modified or patched):

* the model classes ``GRUDLite``, ``RETAINClassifier``, ``MTANClassifier``,
  ``AdaCareClassifier`` and ``STraTSClassifier`` with the ``fit_all.py`` architecture
  settings (hidden size 32, dropout 0.1, one STraTS layer);
* model names and families from ``comparators.registry``;
* the per-episode handcrafted feature kernel of ``feature_builder.py`` (min, max, mean,
  std, last, slope and missing rate per channel; transition count and on-ratio for
  intervention channels);
* ``sequence_bundle.py`` preprocessing: right padding, observed mask, steps since the
  last observation (``_compute_delta``), z-scoring with training statistics and zeros
  at missing positions (``_fit_normalization``, ``_apply_normalization``);
* the ``fit_all.py`` optimisation settings: AdamW (weight decay 1e-3), gradient-norm
  clipping at 2.0, ``BCEWithLogitsLoss`` with ``pos_weight = n_controls / n_positives``;
* ``metrics.compute_binary_metrics`` for every reported metric.

Re-implemented here, and why:

* Training loops. ``fit_all._fit_torch_model``, ``train_evaluate_grud`` and
  ``sequence_bundle.fit_binary_sequence_model`` take ``epochs`` full-batch gradient
  steps, never look at validation data, return no model, and score the test split in
  the same call. v2 trains on shuffled mini-batches, scores validation AUPRC after every
  epoch, stops after ``early_stopping_patience`` epochs without improvement, restores
  the best epoch's weights, and only then scores the test split, once.
* ``train_evaluate_lr`` and ``train_evaluate_gbdt`` take no hyperparameters. v2 keeps
  their estimators (scaler + class-balanced ``LogisticRegression``;
  ``HistGradientBoostingClassifier`` with learning rate 0.05) and adds the validation
  choice of ``C`` and of ``max_depth`` plus the number of boosting iterations. sklearn's
  internal early stopping, which holds out a random 10% of the *training* split when it
  has more than 10,000 rows, is disabled.
* Inputs. ``build_handcrafted_feature_matrix`` and ``build_sequence_bundle`` read every
  column of the stored arrays (no way to drop one), impute missing summary statistics
  with medians over all records including test, silently skip unreadable episodes, and
  do not return episode ids. v2 loads each episode once, drops the excluded columns,
  runs the v1 kernels above, imputes with training medians only, and raises on any
  malformed episode.

External cohorts (frozen transfer, e.g. eICU and HiRID under schema shift):

* :func:`build_external_cohort` prepares an external cohort with the INTERNAL preprocessing
  (:class:`FrozenPreprocessing`): the internal input features and their order, the
  training-split z-scoring means/stds of the sequence inputs, the training medians of the
  handcrafted features and the internal padded length. Nothing is fitted, refit or
  recalibrated on external data, and every external episode runs through the same
  per-episode code path as an internal one.
* Each (model, seed) is trained and selected on internal train/val as above and scores
  the internal test split once; its result is then final, and only afterwards is every
  external cohort scored once with that frozen model (the selected LR ``C``, GBDT depth
  and iteration count, the neural best-epoch weights). External data cannot reach any
  selection.
* Target columns are matched to the internal input features by name. A feature the target
  source does not provide -- stored by the v2 extraction as NaN with mask True in every
  bin (eICU has no ``ventilation_on``), or absent from the target's columns -- is missing
  in every bin, which is exactly how an internal episode encodes a channel it never
  observed. No target-specific value is fitted or imputed for it.

How each model family consumes missingness (identical for internal and external episodes;
also written to the summary JSON, see ``MISSINGNESS_BY_MODEL``):

* ``lr_stats``, ``gbdt_stats``: the v1 kernel summarises each channel over its observed
  bins only (min, max, mean, std, last, slope) and adds its missing rate; intervention
  channels also get a transition count and an on-ratio. For a channel with no observed bin
  min/max/mean/std/last are undefined and take the internal training medians of those
  columns, slope is 0, missing rate is 1, and transition count and on-ratio are 0 (the v1
  kernel's values for an unobserved channel). LR applies its internally fitted scaler and
  coefficients; GBDT gets the same median-imputed matrix (sklearn's native NaN routing is
  not used, internally or externally).
* ``retain_seq``, ``grud_lite``, ``mtan_seq``, ``adacare_seq``, ``strats_seq``: each bin
  carries, per channel, the z-scored value, the observed mask and the number of bins since
  the channel was last observed (v1 ``_compute_delta``). A missing bin has value 0 (the
  internal training mean), mask 0 and a growing delta; right padding is missing too. Every
  model passes this triple through its input layer, so missingness reaches it only through
  the mask and delta channels.

If a channel was never missing in the internal training data (MIMIC-IV records
``ventilation_on`` in every bin), the frozen models have no internal example of that
encoding: its missing-rate column was constant during training, so LR gives it zero weight
and GBDT never splits on it (both then see training-median statistics and on-ratio 0),
and the sequence models extrapolate from a mask they only saw at 1. This is reported in
the per-cohort ``all_missing_features`` and ``observed_fraction``, not repaired.
"""

from __future__ import annotations

import contextlib
import csv
import itertools
import json
import math
import re
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

from src.icu.comparators.feature_builder import DEFAULT_INTERVENTION_FEATURES, DEFAULT_SUMMARY_STATS
from src.icu.comparators.feature_builder import _compute_episode_feature_vector as _v1_episode_features
from src.icu.comparators.metrics import compute_binary_metrics
from src.icu.comparators.registry import get_default_registry
from src.icu.comparators.sequence_bundle import _apply_normalization as _v1_apply_normalization
from src.icu.comparators.sequence_bundle import _compute_delta as _v1_compute_delta
from src.icu.comparators.sequence_bundle import _fit_normalization as _v1_fit_normalization
from src.icu.v2.stats import patient_cluster_bootstrap
from src.icu.v2.types import EpisodeRecord

if TYPE_CHECKING:
    from src.icu.v2.task import TaskSpec

SPLITS: tuple[str, ...] = ("train", "val", "test")
TABULAR_MODELS: tuple[str, ...] = ("lr_stats", "gbdt_stats")
SEQUENCE_MODELS: tuple[str, ...] = ("retain_seq", "grud_lite", "mtan_seq", "adacare_seq", "strats_seq")
SUPERVISED_MODELS: tuple[str, ...] = TABULAR_MODELS + SEQUENCE_MODELS  # v1 registry order
DEFAULT_EXCLUDED_FEATURES: tuple[str, ...] = ("vasopressor_on",)
DATA_PATTERN = "shock_episode_{episode_id}.npy"
MASK_PATTERN = "shock_episode_{episode_id}_missing_mask.npy"
META_PATTERN = "shock_episode_{episode_id}.json"

_SPLIT_NAMES = {"train": "training", "val": "validation", "test": "test"}
_ATTENTION_MODELS = ("mtan_seq", "strats_seq")  # the v1 classes hard-code 4 attention heads
_LR_MAX_ITER = 1000  # as in train_lr.py and fit_all.py

EXTERNAL_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")  # cohort names become CSV values and JSON keys
DEFAULT_EXTERNAL_N_BOOT = 1000
EXTERNAL_METRICS: tuple[str, ...] = ("auprc", "auroc", "brier", "ece")
_SEQUENCE_MISSINGNESS = (
    "per bin and channel the inputs are the value z-scored with the internal training mean/std, the observed "
    "mask and the bins since the channel was last observed (v1 _compute_delta); a missing bin (or right "
    "padding) has value 0, mask 0 and a growing delta; {model}"
)
MISSINGNESS_BY_MODEL: Mapping[str, str] = {
    "lr_stats": (
        "v1 handcrafted features: per channel min/max/mean/std/last/slope over observed bins plus missing_rate "
        "(intervention channels also transition_count and on_ratio). For a channel with no observed bin "
        "min/max/mean/std/last take the internal training medians, slope=0, missing_rate=1, transition_count=0, "
        "on_ratio=0. Scaled with the StandardScaler fitted on the internal training split."
    ),
    "gbdt_stats": (
        "the same median-imputed handcrafted matrix as lr_stats (internal training medians; sklearn's native NaN "
        "routing is not used), routed through the frozen trees with the selected number of iterations."
    ),
    "retain_seq": _SEQUENCE_MISSINGNESS.format(model="the triple is projected per bin; lengths mask the attention."),
    "grud_lite": _SEQUENCE_MISSINGNESS.format(
        model="the triple is encoded per bin; the GRU reads the last (possibly padded) bin, as in v1."
    ),
    "mtan_seq": _SEQUENCE_MISSINGNESS.format(
        model="the triple is projected per bin; lengths mask the attention and the pooling."
    ),
    "adacare_seq": _SEQUENCE_MISSINGNESS.format(
        model="the triple drives the channel gates; gated values are packed by length into the GRU."
    ),
    "strats_seq": _SEQUENCE_MISSINGNESS.format(
        model="the triple is projected per bin; lengths mask the encoder and the pooling."
    ),
}

ArrayLoader = Callable[[EpisodeRecord], tuple[np.ndarray, np.ndarray]]
Logger = Callable[[str], None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# ------------------------------------------------------------------------------ config


@dataclass(frozen=True)
class SupervisedConfig:
    """Training protocol for the v2 supervised reference models.

    Neural models train on shuffled mini-batches for at most ``max_epochs`` epochs, are
    scored on validation AUPRC after every epoch, stop once ``early_stopping_patience``
    epochs pass without an improvement larger than ``early_stopping_min_delta``, and are
    restored to their best epoch. LR picks ``C`` from ``lr_c_grid``; GBDT picks
    ``max_depth`` from ``gbdt_max_depth_grid`` and its number of boosting iterations
    (at most ``gbdt_max_iter``). Every choice uses validation AUPRC only; ties keep the
    earlier candidate (smaller ``C``, shallower tree, fewer iterations, earlier epoch).
    ``learning_rate=0`` is accepted and freezes the networks (a flat validation curve).
    ``num_threads`` caps torch, BLAS and OpenMP threads when set.
    """

    models: tuple[str, ...] = SUPERVISED_MODELS
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    batch_size: int = 256
    eval_batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    grad_clip_norm: float = 2.0
    hidden_size: int = 32
    dropout: float = 0.1
    strats_num_layers: int = 1
    lr_c_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    gbdt_learning_rate: float = 0.05
    gbdt_max_depth_grid: tuple[int, ...] = (3, 4, 6)
    gbdt_max_iter: int = 500
    excluded_features: tuple[str, ...] = DEFAULT_EXCLUDED_FEATURES
    device: str = "cpu"
    num_threads: int | None = None

    def __post_init__(self) -> None:
        for name, cast in (
            ("models", str),
            ("seeds", int),
            ("lr_c_grid", float),
            ("gbdt_max_depth_grid", int),
            ("excluded_features", str),
        ):
            value = getattr(self, name)
            value = (value,) if isinstance(value, (str, int, float)) else value
            object.__setattr__(self, name, tuple(cast(item) for item in value))
        registry = get_default_registry()
        unknown = [m for m in self.models if m not in SUPERVISED_MODELS or m not in registry]
        _require(bool(self.models), "models must not be empty")
        _require(not unknown, f"unknown supervised models {unknown}; expected a subset of {list(SUPERVISED_MODELS)}")
        _require(len(set(self.models)) == len(self.models), f"duplicate model names in {list(self.models)}")
        _require(bool(self.seeds) and len(set(self.seeds)) == len(self.seeds), "seeds must be distinct integers")
        _require(int(self.max_epochs) >= 1, "max_epochs must be >= 1")
        _require(int(self.early_stopping_patience) >= 1, "early_stopping_patience must be >= 1")
        _require(float(self.early_stopping_min_delta) >= 0.0, "early_stopping_min_delta must be >= 0")
        _require(int(self.batch_size) >= 1 and int(self.eval_batch_size) >= 1, "batch sizes must be >= 1")
        _require(float(self.learning_rate) >= 0.0, "learning_rate must be >= 0")
        _require(float(self.weight_decay) >= 0.0, "weight_decay must be >= 0")
        _require(float(self.grad_clip_norm) > 0.0, "grad_clip_norm must be > 0")
        _require(int(self.hidden_size) >= 1, "hidden_size must be >= 1")
        if any(m in _ATTENTION_MODELS for m in self.models):
            _require(int(self.hidden_size) % 4 == 0, "hidden_size must be divisible by 4 (v1 mTAN/STraTS use 4 heads)")
        _require(0.0 <= float(self.dropout) < 1.0, "dropout must be in [0, 1)")
        _require(int(self.strats_num_layers) >= 1, "strats_num_layers must be >= 1")
        _require(bool(self.lr_c_grid) and all(c > 0 for c in self.lr_c_grid), "lr_c_grid needs positive values")
        depths = self.gbdt_max_depth_grid
        _require(bool(depths) and all(d >= 1 for d in depths), "gbdt_max_depth_grid needs depths >= 1")
        _require(float(self.gbdt_learning_rate) > 0.0, "gbdt_learning_rate must be > 0")
        _require(int(self.gbdt_max_iter) >= 1, "gbdt_max_iter must be >= 1")
        _require(isinstance(self.device, str) and bool(self.device), "device must be a torch device string")
        _require(self.num_threads is None or int(self.num_threads) >= 1, "num_threads must be None or >= 1")

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON reports."""
        return asdict(self)


@dataclass
class EarlyStopping:
    """Patience-based early stopping on a validation score where higher is better.

    A score counts as an improvement only if it is finite and exceeds the best score by
    more than ``min_delta``; anything else (including NaN) is a bad epoch.
    """

    patience: int
    min_delta: float = 0.0
    best_score: float = -math.inf
    best_epoch: int = 0
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        _require(int(self.patience) >= 1, "patience must be >= 1")
        _require(float(self.min_delta) >= 0.0, "min_delta must be >= 0")

    def update(self, epoch: int, score: float) -> bool:
        """Record the validation score of ``epoch``; return True if it is a new best."""
        if math.isfinite(score) and score > self.best_score + self.min_delta:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


# ----------------------------------------------------------------------- episode lists


def _as_int(value: Any, what: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = "" if value is None else str(value).strip()
    _require(bool(text), f"{what} is missing")
    try:
        return int(text)
    except ValueError:
        number = float(text)
        _require(number.is_integer(), f"{what} must be an integer, got {value!r}")
        return int(number)


def _episode_from_row(row: Mapping[str, Any]) -> EpisodeRecord:
    episode_id = str(row.get("episode_id") or "").strip()
    _require(bool(episode_id), f"episode row without episode_id: {dict(row)!r}")
    raw_label = row.get("label")
    if raw_label is not None and str(raw_label).strip() != "":
        label = _as_int(raw_label, f"label of {episode_id!r}")
    else:
        kind = str(row.get("episode_type") or "").strip()
        _require(kind in ("positive", "control"), f"episode {episode_id!r} has neither label nor episode_type")
        label = 1 if kind == "positive" else 0
    _require(label in (0, 1), f"episode {episode_id!r} has label {label!r}; expected 0 or 1")
    stay = row.get("stay_id")
    return EpisodeRecord(
        episode_id=episode_id,
        subject_id=_as_int(row.get("subject_id"), f"subject_id of {episode_id!r}"),
        stay_id=-1 if stay is None or str(stay).strip() == "" else _as_int(stay, f"stay_id of {episode_id!r}"),
        label=label,
        anchor_kind=str(row.get("anchor_kind") or ""),
        anchor_time=str(row.get("anchor_time") or ""),
        window_start=str(row.get("window_start") or ""),
        window_end=str(row.get("window_end") or ""),
        split=str(row.get("split") or "").strip(),
    )


def load_episode_list(path: Path | str) -> list[EpisodeRecord]:
    """Read a v2 episode list: JSON (a list, or ``{"episodes": [...]}``), JSONL, or CSV.

    Rows follow :class:`~src.icu.v2.types.EpisodeRecord` (for example the episodes returned
    by ``splits.patient_level_split``). ``episode_id`` and ``subject_id`` are required; the
    label comes from ``label`` (0/1) or else ``episode_type`` (positive/control).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as handle:  # tolerate an Excel BOM
            rows: Any = list(csv.DictReader(handle))
    else:
        text = path.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            rows = payload.get("episodes") if isinstance(payload, dict) else payload
    _require(isinstance(rows, list), f"{path}: expected a list of episode rows (or an 'episodes' list)")
    return [_episode_from_row(row) for row in rows]


def load_split_assignments(path: Path | str) -> dict[str, str]:
    """Read an episode-level split manifest into ``{episode_id: split}``.

    Accepts the v1 layout ``{"split": {"train": [...], "val": [...], "test": [...]}}``, the
    bare ``{"train": [...], ...}`` lists, or a JSON list of ``{"episode_id", "split"}`` rows
    such as dumped ``EpisodeRecord`` objects (rows with an empty split are skipped); the v1
    layout and the row list are also what ``fmg_adapter.load_split_manifest`` reads. The
    patient-level manifest written by ``splits.patient_level_split`` is refused: it cannot
    tell which episodes a per-patient cap dropped, so pass the split episode list instead.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    assignment: dict[str, str] = {}

    def assign(episode_id: Any, split: str) -> None:
        key = str(episode_id).strip()
        _require(bool(key), f"{path}: empty episode id")
        _require(split in SPLITS, f"{path}: episode {key!r} has split {split!r}; expected one of {list(SPLITS)}")
        previous = assignment.setdefault(key, split)
        _require(previous == split, f"{path}: episode {key!r} is listed in {previous!r} and {split!r}")

    if isinstance(payload, list):
        for row in payload:
            _require(isinstance(row, Mapping) and "episode_id" in row, f"{path}: split rows need an 'episode_id'")
            split = str(row.get("split") or "").strip()
            if split:
                assign(row["episode_id"], split)
        return assignment
    _require(isinstance(payload, dict), f"{path}: split manifest must be a JSON object or a list of rows")
    if isinstance(payload.get("splits"), dict) and "split" not in payload:
        raise ValueError(
            f"{path} looks like a patient-level manifest from splits.patient_level_split; "
            "pass the returned episode list (rows with a 'split' field) as --episodes instead"
        )
    lists = payload["split"] if isinstance(payload.get("split"), dict) else payload
    for split in SPLITS:
        ids = lists.get(split)
        _require(isinstance(ids, list), f"{path}: split manifest needs an episode-id list under {split!r}")
        for episode_id in ids:
            assign(episode_id, split)
    return assignment


def assign_splits(episodes: Sequence[EpisodeRecord], assignment: Mapping[str, str]) -> list[EpisodeRecord]:
    """Set ``split`` from an episode-level manifest; episodes it does not list are dropped.

    Raises if the manifest names an episode that is not in the list, or contradicts a
    split already present on an episode.
    """
    known = {episode.episode_id for episode in episodes}
    unknown = [episode_id for episode_id in assignment if episode_id not in known]
    _require(
        not unknown,
        f"split manifest lists {len(unknown)} episodes missing from the episode list, e.g. {unknown[:3]}",
    )
    assigned: list[EpisodeRecord] = []
    for episode in episodes:
        split = assignment.get(episode.episode_id)
        if split is None:
            continue
        _require(
            not episode.split or episode.split == split,
            f"episode {episode.episode_id!r} has split {episode.split!r} but the manifest says {split!r}",
        )
        assigned.append(replace(episode, split=split))
    return assigned


def validate_episode_splits(episodes: Sequence[EpisodeRecord]) -> dict[str, list[EpisodeRecord]]:
    """Group episodes by split and enforce the v2 split contract.

    Every episode needs a unique id, a 0/1 label and a split in train/val/test; each
    split must be non-empty and hold both classes (no fallback to training data for
    selection); and no patient (``subject_id``) may appear in two splits.
    """
    _require(bool(episodes), "episode list is empty")
    by_split: dict[str, list[EpisodeRecord]] = {split: [] for split in SPLITS}
    seen: set[str] = set()
    for episode in episodes:
        _require(episode.episode_id not in seen, f"duplicate episode_id {episode.episode_id!r}")
        seen.add(episode.episode_id)
        _require(
            episode.label in (0, 1),
            f"episode {episode.episode_id!r} has label {episode.label!r}; expected 0 or 1",
        )
        _require(
            episode.split in by_split,
            f"episode {episode.episode_id!r} has split {episode.split!r}; expected one of {list(SPLITS)}",
        )
        by_split[episode.split].append(episode)
    for split in SPLITS:
        name = _SPLIT_NAMES[split]
        rows = by_split[split]
        _require(
            bool(rows),
            f"{name} split is empty; v2 selects hyperparameters and stopping epochs on validation "
            "data only and never falls back to the training split",
        )
        classes = sorted({int(episode.label) for episode in rows})
        _require(
            len(classes) == 2,
            f"{name} split has a single class (label {classes[0]}); AUPRC selection and evaluation "
            "need positives and controls",
        )
    owner: dict[int, str] = {}
    for split in SPLITS:
        for episode in by_split[split]:
            first = owner.setdefault(int(episode.subject_id), split)
            _require(
                first == split,
                f"patient subject_id={episode.subject_id} appears in the {first} and {split} splits; "
                "splits must be patient-disjoint",
            )
    return by_split


# ----------------------------------------------------------------------------- loaders


def read_feature_names(data_dir: Path | str, episode_id: str, meta_pattern: str = META_PATTERN) -> tuple[str, ...]:
    """Feature names from one episode's metadata JSON (v1 layout)."""
    path = Path(data_dir) / meta_pattern.format(episode_id=episode_id)
    names = json.loads(path.read_text(encoding="utf-8")).get("feature_names")
    _require(isinstance(names, list) and bool(names), f"{path} has no 'feature_names' list")
    return tuple(str(name) for name in names)


@dataclass(frozen=True)
class FileArrayLoader:
    """Return ``(values, missing_mask)`` exactly as stored in per-episode ``.npy`` files.

    Use it only for arrays that already hold v2 windows. For v1 arrays, whose last bin
    contains the anchor, use :class:`TaskWindowLoader`.
    """

    data_dir: Path
    data_pattern: str = DATA_PATTERN
    mask_pattern: str = MASK_PATTERN

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))

    def __call__(self, episode: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
        episode_id = episode.episode_id
        values = np.load(self.data_dir / self.data_pattern.format(episode_id=episode_id), allow_pickle=False)
        missing = np.load(self.data_dir / self.mask_pattern.format(episode_id=episode_id), allow_pickle=False)
        _require(
            values.ndim == 2 and missing.shape == values.shape,
            f"episode {episode_id!r}: values {values.shape} and mask {missing.shape} must be equal 2-D shapes",
        )
        return values, missing.astype(bool)

    def describe(self) -> dict[str, Any]:
        return {"window": "stored", "data_pattern": self.data_pattern, "mask_pattern": self.mask_pattern}


class TaskWindowLoader:
    """Cut v1-layout episodes with the v2 task window before any feature is built.

    Each call opens ``shock_episode_<id>.json/.npy/_missing_mask.npy`` by constructed path
    (``leakage_audit.load_episode``) and applies ``task.transform_episode``: bins that end
    after ``reference_time - prediction_gap`` are dropped and v1's per-episode imputation
    is refit on the kept rows. The transform runs with ``excluded_features=()`` so that
    :func:`build_supervised_inputs` drops excluded features for every loader the same way;
    ``required_exclusions`` makes that function refuse an exclusion list that misses the
    spec's features. The episode-list label must match the metadata class, and a window
    that is not full under a ``require_full_window`` spec raises (length leak).
    """

    def __init__(
        self,
        data_dir: Path | str,
        spec: TaskSpec | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        from src.icu.v2.leakage_audit import load_episode
        from src.icu.v2.task import TaskSpec as _TaskSpec
        from src.icu.v2.task import episode_label, transform_episode

        self.data_dir = Path(data_dir)
        self.spec = spec if spec is not None else _TaskSpec()
        self.required_exclusions: tuple[str, ...] = tuple(self.spec.excluded_features)
        self.feature_names = None if feature_names is None else tuple(str(name) for name in feature_names)
        self.rows_after: Counter[int] = Counter()
        self._transform_spec = replace(self.spec, excluded_features=())
        self._load_episode = load_episode
        self._episode_label = episode_label
        self._transform_episode = transform_episode

    def __call__(self, episode: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
        arrays = self._load_episode(self.data_dir, episode.episode_id)
        meta = arrays.meta
        _require(
            self._episode_label(meta) == int(episode.label),
            f"episode {episode.episode_id!r}: list label {episode.label} disagrees with metadata "
            f"episode_type {meta.get('episode_type')!r}",
        )
        names = tuple(str(name) for name in meta.get("feature_names", ()))
        _require(
            self.feature_names is None or names == self.feature_names,
            f"episode {episode.episode_id!r} has features {list(names)}, expected {list(self.feature_names or ())}",
        )
        values, missing, _, info = self._transform_episode(arrays.values, arrays.mask, meta, self._transform_spec)
        _require(
            not (self.spec.require_full_window and not info["full_window"]),
            f"episode {episode.episode_id!r} is not a full {self.spec.window_hours:g} h window under the task "
            "spec; build the episode list with the same TaskSpec (task.apply_task)",
        )
        _require(values.shape[0] >= 1, f"episode {episode.episode_id!r} keeps no rows before the prediction cutoff")
        self.rows_after[int(values.shape[0])] += 1
        return values, missing

    def describe(self) -> dict[str, Any]:
        return {
            "window": "task",
            "task_spec": self.spec.to_dict(),
            "rows_after_cutoff": {str(rows): count for rows, count in sorted(self.rows_after.items())},
        }


# ------------------------------------------------------------------------------ inputs


@dataclass(frozen=True, eq=False)
class SplitArrays:
    """Model inputs of one split, row-aligned with ``episode_ids``.

    ``tabular`` holds the v1 handcrafted features (training-median imputed). ``values``,
    ``observed`` and ``deltas`` are ``(episodes, max_len, channels)`` float32 arrays,
    right-padded; ``values`` is z-scored with training statistics and 0 where missing.
    """

    episode_ids: tuple[str, ...]
    subject_ids: np.ndarray
    y: np.ndarray
    tabular: np.ndarray
    values: np.ndarray
    observed: np.ndarray
    deltas: np.ndarray
    lengths: np.ndarray


@dataclass(frozen=True, eq=False)
class FrozenPreprocessing:
    """Statistics fitted on the internal training split; external cohorts reuse them unchanged.

    ``feature_names`` are the columns the internal loader returns and ``input_features``
    the model channels left after exclusion, in order. ``sequence_means`` and
    ``sequence_stds`` z-score the observed sequence values (v1 ``_fit_normalization``),
    ``tabular_medians`` replace undefined handcrafted features, and ``max_len`` is the
    padded length the neural models were trained on. The arrays are read-only copies.
    """

    feature_names: tuple[str, ...]
    input_features: tuple[str, ...]
    excluded_features: tuple[str, ...]
    intervention_features: tuple[str, ...]
    tabular_feature_names: tuple[str, ...]
    max_len: int
    sequence_means: np.ndarray
    sequence_stds: np.ndarray
    tabular_medians: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready statistics that every external cohort was scored with."""
        return json_safe(
            {
                "fitted_on": "internal training split",
                "feature_names": list(self.feature_names),
                "input_features": list(self.input_features),
                "excluded_features": list(self.excluded_features),
                "max_len": int(self.max_len),
                "sequence_means": dict(zip(self.input_features, self.sequence_means.tolist())),
                "sequence_stds": dict(zip(self.input_features, self.sequence_stds.tolist())),
                "tabular_medians": dict(zip(self.tabular_feature_names, self.tabular_medians.tolist())),
            }
        )


@dataclass(frozen=True, eq=False)
class SupervisedInputs:
    """Inputs for train/val/test built after feature exclusion (see :func:`build_supervised_inputs`).

    ``preprocessing`` holds the training statistics used to build them; external cohorts
    are prepared with it (:func:`build_external_cohort`).
    """

    input_features: tuple[str, ...]
    excluded_features: tuple[str, ...]
    tabular_feature_names: tuple[str, ...]
    max_len: int
    splits: Mapping[str, SplitArrays]
    nonfinite_observed_values: int = 0
    preprocessing: FrozenPreprocessing | None = None

    def length_by_label(self) -> dict[str, dict[str, float]]:
        """Window length (rows) by label over all splits; unequal rows mean a length leak."""
        lengths = np.concatenate([self.splits[split].lengths for split in SPLITS])
        labels = np.concatenate([self.splits[split].y for split in SPLITS])
        summary: dict[str, dict[str, float]] = {}
        for label in (0, 1):
            chosen = lengths[labels == label]
            summary[str(label)] = {"min": int(chosen.min()), "max": int(chosen.max()), "mean": float(chosen.mean())}
        return summary

    def summary(self) -> dict[str, Any]:
        """JSON-ready description of the inputs (features, split sizes, length check)."""
        by_label = self.length_by_label()
        return {
            "input_features": list(self.input_features),
            "excluded_features": list(self.excluded_features),
            "n_tabular_features": len(self.tabular_feature_names),
            "tabular_feature_names": list(self.tabular_feature_names),
            "max_len": int(self.max_len),
            "splits": {split: _split_counts(self.splits[split]) for split in SPLITS},
            "length_by_label": by_label,
            "length_differs_by_label": by_label["0"] != by_label["1"],
            "nonfinite_observed_values": int(self.nonfinite_observed_values),
        }


def _split_counts(part: SplitArrays) -> dict[str, int]:
    positives = int(part.y.sum())
    return {
        "episodes": int(part.y.size),
        "positives": positives,
        "controls": int(part.y.size) - positives,
        "patients": int(np.unique(part.subject_ids).size),
    }


def _kept_columns(
    feature_names: Sequence[str],
    excluded_features: Iterable[str],
) -> tuple[list[int], tuple[str, ...], tuple[str, ...]]:
    names = tuple(str(name) for name in feature_names)
    _require(bool(names) and len(set(names)) == len(names), f"feature names must be unique and non-empty: {names}")
    excluded = {str(name) for name in excluded_features}
    unknown = sorted(excluded - set(names))
    _require(not unknown, f"excluded features {unknown} are not among the episode features {list(names)}")
    keep = [index for index, name in enumerate(names) if name not in excluded]
    _require(bool(keep), "every feature is excluded; nothing is left to train on")
    return keep, tuple(names[index] for index in keep), tuple(name for name in names if name in excluded)


def _pad_sequences(
    rows: Sequence[tuple[np.ndarray, np.ndarray]],
    max_len: int,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Right-pad like ``sequence_bundle._load_split_arrays`` (padding counts as missing)."""
    count = len(rows)
    values = np.zeros((count, max_len, dim), dtype=np.float32)
    observed = np.zeros((count, max_len, dim), dtype=np.float32)
    deltas = np.zeros((count, max_len, dim), dtype=np.float32)
    lengths = np.zeros((count,), dtype=np.int64)
    padded_missing = np.ones((max_len, dim), dtype=bool)
    for index, (episode_values, episode_missing) in enumerate(rows):
        steps = episode_values.shape[0]
        padded_missing[:] = True
        padded_missing[:steps] = episode_missing
        values[index, :steps] = episode_values
        observed[index] = ~padded_missing
        deltas[index] = _v1_compute_delta(padded_missing)
        lengths[index] = steps
    return values, observed, deltas, lengths


def _training_medians(train: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(train), train, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(finite, axis=0)
    return np.where(np.isfinite(medians), medians, 0.0)


def _frozen(array: np.ndarray) -> np.ndarray:
    copy = np.array(array, dtype=float, copy=True)
    copy.setflags(write=False)
    return copy


def _episode_model_inputs(
    episode: EpisodeRecord,
    raw_values: Any,
    raw_missing: Any,
    n_columns: int,
    columns: np.ndarray,
    input_features: Sequence[str],
    intervention: Sequence[str],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, int]:
    """One episode -> (handcrafted vector, its names, float32 values, missing mask, non-finite count).

    The loader returned ``n_columns`` columns; ``columns[k]`` is the column holding
    ``input_features[k]``, or -1 if the source has none, in which case that channel is
    missing in every row, like a channel the source provides but never observed.
    Non-finite observed values become missing; values at missing positions are set to 0
    (the v1 kernels never read them). Internal and external episodes both run through here.
    """
    raw_values = np.asarray(raw_values)
    raw_missing = np.asarray(raw_missing, dtype=bool)
    expected_shape = raw_values.ndim == 2 and raw_values.shape[1] == n_columns
    _require(
        expected_shape and raw_missing.shape == raw_values.shape,
        f"episode {episode.episode_id!r}: expected (rows, {n_columns}) values and mask, "
        f"got {raw_values.shape} and {raw_missing.shape}",
    )
    _require(raw_values.shape[0] >= 1, f"episode {episode.episode_id!r} has no rows")
    present = columns >= 0
    if present.all():
        values = raw_values[:, columns].astype(float)
        missing = raw_missing[:, columns]
    else:
        values = np.full((raw_values.shape[0], columns.size), np.nan)
        missing = np.ones((raw_values.shape[0], columns.size), dtype=bool)
        values[:, present] = raw_values[:, columns[present]]
        missing[:, present] = raw_missing[:, columns[present]]
    bad = ~missing & ~np.isfinite(values)
    n_bad = int(bad.sum())
    if n_bad:
        missing = missing | bad
    values[missing] = 0.0  # the v1 kernels never read missing positions
    vector, names = _v1_episode_features(
        values, missing, list(input_features), tuple(intervention), DEFAULT_SUMMARY_STATS
    )
    return np.asarray(vector, dtype=float), list(names), values.astype(np.float32), missing, n_bad


def _split_arrays(
    episodes: Sequence[EpisodeRecord],
    tabular: np.ndarray,
    padded: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    preprocessing: FrozenPreprocessing,
) -> SplitArrays:
    """Apply the training statistics: median imputation (tabular) and z-scoring (sequences)."""
    values, observed, deltas, lengths = padded
    return SplitArrays(
        episode_ids=tuple(episode.episode_id for episode in episodes),
        subject_ids=np.asarray([int(episode.subject_id) for episode in episodes], dtype=np.int64),
        y=np.asarray([int(episode.label) for episode in episodes], dtype=np.int64),
        tabular=np.where(np.isfinite(tabular), tabular, preprocessing.tabular_medians),
        values=_v1_apply_normalization(
            values, observed, preprocessing.sequence_means, preprocessing.sequence_stds
        ).astype(np.float32),
        observed=observed,
        deltas=deltas,
        lengths=lengths,
    )


def build_supervised_inputs(
    episodes: Sequence[EpisodeRecord],
    feature_names: Sequence[str],
    loader: ArrayLoader,
    excluded_features: Iterable[str] = DEFAULT_EXCLUDED_FEATURES,
    log: Logger | None = None,
) -> SupervisedInputs:
    """Load every episode once, drop excluded features, and build tabular and sequence inputs.

    ``loader`` returns ``(values, missing_mask)`` with one column per ``feature_names``
    entry (mask True = missing). Excluded columns are removed before the v1 handcrafted
    kernel and the v1 sequence preprocessing run. Observed values that are not finite are
    treated as missing and counted. Normalisation statistics and imputation medians come
    from the training split only and are kept in ``preprocessing`` for external cohorts.
    Raises on split-contract violations (:func:`validate_episode_splits`), unknown excluded
    features, an exclusion list that misses ``loader.required_exclusions``, and malformed
    arrays.
    """
    by_split = validate_episode_splits(episodes)
    all_names = tuple(str(name) for name in feature_names)
    excluded_features = tuple(str(name) for name in excluded_features)
    required = [name for name in getattr(loader, "required_exclusions", ()) if name not in excluded_features]
    _require(not required, f"the loader's task spec excludes {required}; add them to excluded_features")
    keep, input_features, excluded = _kept_columns(all_names, excluded_features)
    intervention = tuple(name for name in DEFAULT_INTERVENTION_FEATURES if name in input_features)
    columns = np.asarray(keep, dtype=np.int64)

    tabular_names: tuple[str, ...] | None = None
    tabular_rows: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    sequence_rows: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {split: [] for split in SPLITS}
    nonfinite = 0
    total = sum(len(rows) for rows in by_split.values())
    done = 0
    for split in SPLITS:
        for episode in by_split[split]:
            raw_values, raw_missing = loader(episode)
            vector, names, values, missing, n_bad = _episode_model_inputs(
                episode, raw_values, raw_missing, len(all_names), columns, input_features, intervention
            )
            nonfinite += n_bad
            if tabular_names is None:
                tabular_names = tuple(names)
            tabular_rows[split].append(vector)
            sequence_rows[split].append((values, missing))
            done += 1
            if log is not None and done % 10000 == 0:
                log(f"loaded {done}/{total} episodes")
    if nonfinite:
        warnings.warn(f"{nonfinite} observed values were not finite and were treated as missing", stacklevel=2)

    max_len = max(values.shape[0] for split in SPLITS for values, _ in sequence_rows[split])
    padded = {split: _pad_sequences(sequence_rows[split], max_len, len(input_features)) for split in SPLITS}
    sequence_rows.clear()
    means, stds = _v1_fit_normalization(padded["train"][0], padded["train"][1])
    tabular = {split: np.vstack(tabular_rows[split]) for split in SPLITS}
    preprocessing = FrozenPreprocessing(
        feature_names=all_names,
        input_features=input_features,
        excluded_features=excluded,
        intervention_features=intervention,
        tabular_feature_names=tabular_names or (),
        max_len=int(max_len),
        sequence_means=_frozen(means),
        sequence_stds=_frozen(stds),
        tabular_medians=_frozen(_training_medians(tabular["train"])),
    )
    parts = {split: _split_arrays(by_split[split], tabular[split], padded[split], preprocessing) for split in SPLITS}
    inputs = SupervisedInputs(
        input_features=input_features,
        excluded_features=excluded,
        tabular_feature_names=tabular_names or (),
        max_len=int(max_len),
        splits=parts,
        nonfinite_observed_values=nonfinite,
        preprocessing=preprocessing,
    )
    by_label = inputs.length_by_label()
    if by_label["0"] != by_label["1"]:
        warnings.warn(
            f"window length differs by label {by_label}; the sequence models see lengths, so this "
            "can leak the label (use a task spec with require_full_window)",
            stacklevel=2,
        )
    return inputs


# -------------------------------------------------------------------- external cohorts


@dataclass(frozen=True, eq=False)
class ExternalCohort:
    """An external cohort prepared for frozen scoring (see :func:`build_external_cohort`).

    ``arrays`` were built with ``preprocessing``, the internal training statistics; ``info``
    describes the cohort (counts, prevalence, column alignment, all-missing features).
    """

    name: str
    arrays: SplitArrays
    preprocessing: FrozenPreprocessing
    info: Mapping[str, Any]


@dataclass(frozen=True)
class ExternalSource:
    """Where :func:`run_supervised_v2` reads an external cohort from.

    ``feature_names`` name the columns ``loader`` returns (default: the internal ones).
    """

    name: str
    episodes: Sequence[EpisodeRecord]
    loader: ArrayLoader
    feature_names: Sequence[str] | None = None


def validate_external_episodes(
    name: str,
    episodes: Sequence[EpisodeRecord],
    internal_episode_ids: Iterable[str] = (),
) -> None:
    """Check an external episode list before anything is loaded.

    The name must match ``EXTERNAL_NAME_PATTERN``; episodes need unique ids and 0/1 labels,
    both classes must be present (AUPRC and AUROC), and no episode may be an internal
    train/val/test episode. The ``split`` field is ignored: every listed episode is scored.
    """
    _require(
        isinstance(name, str) and EXTERNAL_NAME_PATTERN.fullmatch(name) is not None,
        f"external cohort name {name!r} must match {EXTERNAL_NAME_PATTERN.pattern}",
    )
    _require(bool(episodes), f"external cohort {name!r} has no episodes")
    seen: set[str] = set()
    for episode in episodes:
        _require(
            episode.episode_id not in seen,
            f"external cohort {name!r}: duplicate episode_id {episode.episode_id!r}",
        )
        seen.add(episode.episode_id)
        _require(
            episode.label in (0, 1),
            f"external cohort {name!r}: episode {episode.episode_id!r} has label {episode.label!r}; expected 0 or 1",
        )
    classes = sorted({int(episode.label) for episode in episodes})
    _require(
        len(classes) == 2,
        f"external cohort {name!r} has a single class (label {classes[0]}); AUPRC and AUROC need "
        "positives and controls",
    )
    shared = sorted(seen.intersection(internal_episode_ids))
    _require(
        not shared,
        f"external cohort {name!r} shares {len(shared)} episode ids with the internal episodes, e.g. "
        f"{shared[:3]}; an external cohort must be disjoint from the training, validation and test data",
    )


def _observed_fraction(part: SplitArrays, features: Sequence[str]) -> dict[str, float]:
    """Share of the real (unpadded) bins in which each channel was observed."""
    observed = part.observed.sum(axis=(0, 1), dtype=np.float64)  # padding is never observed
    bins = float(part.lengths.sum())
    return {name: float(count) / bins for name, count in zip(features, observed)}


def build_external_cohort(
    name: str,
    episodes: Sequence[EpisodeRecord],
    loader: ArrayLoader,
    inputs: SupervisedInputs,
    feature_names: Sequence[str] | None = None,
    log: Logger | None = None,
) -> ExternalCohort:
    """Prepare an external cohort for frozen scoring with the internal training statistics.

    ``loader`` returns ``(values, missing_mask)`` with one column per ``feature_names``
    entry (default: the internal loader's columns). Columns are matched to the internal
    input features by name: target columns the models do not use are ignored, and an input
    feature without a target column is missing in every bin. Each episode then runs through
    the internal per-episode code path, is right-padded to the internal ``max_len``,
    z-scored with the internal training means/stds and median-imputed with the internal
    training medians (``inputs.preprocessing``). Nothing is fitted on the cohort. Raises on
    contract violations (:func:`validate_external_episodes`), malformed arrays and
    episodes longer than the internal ``max_len``.
    """
    preprocessing = inputs.preprocessing
    _require(preprocessing is not None, "inputs carry no training statistics; build them with build_supervised_inputs")
    assert preprocessing is not None
    internal_ids = {episode_id for split in SPLITS for episode_id in inputs.splits[split].episode_ids}
    validate_external_episodes(name, episodes, internal_ids)
    source_names = tuple(str(f) for f in (preprocessing.feature_names if feature_names is None else feature_names))
    _require(
        bool(source_names) and len(set(source_names)) == len(source_names),
        f"external cohort {name!r}: feature names must be unique and non-empty: {list(source_names)}",
    )
    required = [f for f in getattr(loader, "required_exclusions", ()) if f not in preprocessing.excluded_features]
    _require(
        not required,
        f"external cohort {name!r}: the loader's task spec excludes {required}, but the internal models were "
        "trained with them; use the internal task spec",
    )
    position = {feature: index for index, feature in enumerate(source_names)}
    columns = np.asarray([position.get(feature, -1) for feature in preprocessing.input_features], dtype=np.int64)
    absent = [feature for feature, column in zip(preprocessing.input_features, columns) if column < 0]
    unused = [feature for feature in source_names if feature not in preprocessing.input_features]

    tabular_rows: list[np.ndarray] = []
    sequence_rows: list[tuple[np.ndarray, np.ndarray]] = []
    nonfinite = 0
    for done, episode in enumerate(episodes, start=1):
        raw_values, raw_missing = loader(episode)
        vector, names, values, missing, n_bad = _episode_model_inputs(
            episode,
            raw_values,
            raw_missing,
            len(source_names),
            columns,
            preprocessing.input_features,
            preprocessing.intervention_features,
        )
        _require(
            done > 1 or tuple(names) == preprocessing.tabular_feature_names,
            f"external cohort {name!r}: handcrafted features differ from the internal ones",
        )
        _require(
            values.shape[0] <= preprocessing.max_len,
            f"external episode {episode.episode_id!r} has {values.shape[0]} rows, but the frozen models were "
            f"trained on windows of at most {preprocessing.max_len} rows; load it with the internal window",
        )
        nonfinite += n_bad
        tabular_rows.append(vector)
        sequence_rows.append((values, missing))
        if log is not None and done % 10000 == 0:
            log(f"{name}: loaded {done}/{len(episodes)} external episodes")
    if nonfinite:
        warnings.warn(
            f"external cohort {name!r}: {nonfinite} observed values were not finite and were treated as missing",
            stacklevel=2,
        )
    padded = _pad_sequences(sequence_rows, preprocessing.max_len, len(preprocessing.input_features))
    sequence_rows.clear()
    arrays = _split_arrays(episodes, np.vstack(tabular_rows), padded, preprocessing)

    by_label: dict[str, dict[str, float]] = {}
    for label in (0, 1):
        chosen = arrays.lengths[arrays.y == label]
        by_label[str(label)] = {"min": int(chosen.min()), "max": int(chosen.max()), "mean": float(chosen.mean())}
    if by_label["0"] != by_label["1"]:
        warnings.warn(f"external cohort {name!r}: window length differs by label {by_label}", stacklevel=2)
    internal_subjects = {int(s) for split in SPLITS for s in inputs.splits[split].subject_ids.tolist()}
    shared_subjects = len(set(arrays.subject_ids.tolist()) & internal_subjects)
    if shared_subjects:
        warnings.warn(
            f"external cohort {name!r}: {shared_subjects} subject_id values also occur in the internal episodes; "
            "if both cohorts come from the same database, these patients overlap",
            stacklevel=2,
        )
    counts = _split_counts(arrays)
    observed_fraction = _observed_fraction(arrays, preprocessing.input_features)
    describe = getattr(loader, "describe", None)
    info = {
        "name": name,
        **counts,
        "prevalence": counts["positives"] / counts["episodes"],
        "split_values": dict(sorted(Counter(episode.split for episode in episodes).items())),
        "feature_names": list(source_names),
        "absent_features": absent,
        "unused_columns": unused,
        "all_missing_features": [feature for feature, share in observed_fraction.items() if share == 0.0],
        "observed_fraction": observed_fraction,
        "nonfinite_observed_values": nonfinite,
        "length_by_label": by_label,
        "subjects_shared_with_internal": shared_subjects,
        "loader": describe() if callable(describe) else None,
    }
    return ExternalCohort(name=name, arrays=arrays, preprocessing=preprocessing, info=json_safe(info))


# ----------------------------------------------------------------------------- results


@dataclass(frozen=True)
class SeedResult:
    """One (model, seed) fit. Selection used validation only; test was scored once.

    ``selection_trace`` lists per-epoch validation AUPRC (neural models) or per-candidate
    validation AUPRC (LR, GBDT). ``n_model_inputs`` is the number of input channels
    (sequence models) or handcrafted columns (LR, GBDT) the fitted model consumed.
    ``external_metrics`` and ``external_probabilities`` (keyed by cohort name) come from
    scoring each external cohort once with the frozen selected model, after everything
    else here was final; they are empty when no external cohort was given.
    """

    model: str
    family: str
    seed: int
    selected: Mapping[str, Any]
    selection_trace: tuple[Mapping[str, Any], ...]
    best_epoch: int | None
    epochs_trained: int | None
    stopped_early: bool | None
    n_model_inputs: int
    val_metrics: Mapping[str, Any]
    test_metrics: Mapping[str, Any]
    test_probabilities: Mapping[str, float]
    fit_seconds: float
    external_metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    external_probabilities: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def to_dict(self, include_probabilities: bool = True) -> dict[str, Any]:
        """JSON-ready dict; NaN becomes None. External keys appear only for external cohorts."""
        row = {
            "model": self.model,
            "family": self.family,
            "seed": self.seed,
            "selected": dict(self.selected),
            "best_epoch": self.best_epoch,
            "epochs_trained": self.epochs_trained,
            "stopped_early": self.stopped_early,
            "n_model_inputs": self.n_model_inputs,
            "val_metrics": dict(self.val_metrics),
            "test_metrics": dict(self.test_metrics),
            "fit_seconds": self.fit_seconds,
            "selection_trace": [dict(step) for step in self.selection_trace],
        }
        if self.external_metrics:
            row["external_metrics"] = {cohort: dict(metrics) for cohort, metrics in self.external_metrics.items()}
        if include_probabilities:
            row["test_probabilities"] = dict(self.test_probabilities)
            if self.external_probabilities:
                row["external_probabilities"] = {
                    cohort: dict(probabilities) for cohort, probabilities in self.external_probabilities.items()
                }
        return json_safe(row)


@dataclass(frozen=True)
class SupervisedRunResult:
    """All (model, seed) fits of one run plus what is needed to resample them by patient.

    The ``external_*`` fields are filled only when external cohorts were scored:
    labels and subject ids per cohort and episode, the cohort descriptions, and the frozen
    internal preprocessing every cohort was scored with.
    """

    config: SupervisedConfig
    data_summary: Mapping[str, Any]
    test_labels: Mapping[str, int]
    test_subjects: Mapping[str, int]
    runs: tuple[SeedResult, ...]
    external_labels: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    external_subjects: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    external_info: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    frozen_preprocessing: Mapping[str, Any] = field(default_factory=dict)

    def summary(
        self,
        external_n_boot: int = DEFAULT_EXTERNAL_N_BOOT,
        external_ci: float = 0.95,
        external_bootstrap_seed: int = 0,
    ) -> dict[str, Any]:
        """JSON-ready report: config, inputs, per-seed results and their mean/std per model.

        With external cohorts it also has an ``external`` block (:meth:`external_summary`,
        which the ``external_*`` arguments configure); without them it is unchanged.
        """
        models: dict[str, Any] = {}
        for name in self.config.models:
            runs = [run for run in self.runs if run.model == name]
            if not runs:
                continue
            aggregate: dict[str, Any] = {}
            for split, attribute in (("val", "val_metrics"), ("test", "test_metrics")):
                for metric in ("auprc", "auroc", "brier", "ece"):
                    mean, std = _mean_std([getattr(run, attribute).get(metric) for run in runs])
                    aggregate[f"{split}_{metric}_mean"] = mean
                    aggregate[f"{split}_{metric}_std"] = std
            models[name] = {
                "family": runs[0].family,
                "n_seeds": len(runs),
                "over_seeds": aggregate,
                "seeds": [run.to_dict(include_probabilities=False) for run in runs],
            }
        payload = {
            "protocol": "v2_supervised_reference",
            "config": self.config.to_dict(),
            "data": dict(self.data_summary),
            "models": models,
            "notes": [
                "LR C, GBDT depth and iterations, and the neural stopping epoch were chosen on "
                "validation AUPRC only.",
                "The test split was scored once per (model, seed), after selection.",
                "Seed mean/std is not a confidence interval; resample the per-episode test predictions by patient.",
            ],
        }
        if self.external_labels:
            payload["external"] = self.external_summary(external_n_boot, external_ci, external_bootstrap_seed)
        return json_safe(payload)

    def prediction_rows(self) -> list[dict[str, Any]]:
        """Long-format test predictions: model, seed, episode_id, subject_id, label, prob."""
        rows: list[dict[str, Any]] = []
        for run in self.runs:
            for episode_id, prob in run.test_probabilities.items():
                rows.append(
                    {
                        "model": run.model,
                        "seed": run.seed,
                        "episode_id": episode_id,
                        "subject_id": int(self.test_subjects[episode_id]),
                        "label": int(self.test_labels[episode_id]),
                        "prob": float(prob),
                    }
                )
        return rows

    def external_prediction_rows(self) -> list[dict[str, Any]]:
        """Long-format external predictions: model, seed, cohort, episode_id, subject_id, label, prob."""
        rows: list[dict[str, Any]] = []
        for run in self.runs:
            for cohort, labels in self.external_labels.items():
                subjects = self.external_subjects[cohort]
                for episode_id, prob in run.external_probabilities.get(cohort, {}).items():
                    rows.append(
                        {
                            "model": run.model,
                            "seed": run.seed,
                            "cohort": cohort,
                            "episode_id": episode_id,
                            "subject_id": int(subjects[episode_id]),
                            "label": int(labels[episode_id]),
                            "prob": float(prob),
                        }
                    )
        return rows

    def external_summary(
        self,
        n_boot: int = DEFAULT_EXTERNAL_N_BOOT,
        ci: float = 0.95,
        bootstrap_seed: int = 0,
    ) -> dict[str, Any]:
        """Frozen-transfer report per model x external cohort.

        For each pair: mean/std over seeds of AUPRC and AUROC (also Brier and ECE), the
        per-seed values, a patient-clustered bootstrap CI (``stats.patient_cluster_bootstrap``,
        resampling ``subject_id``) of AUPRC and AUROC for model seed 0 (the first configured
        seed if 0 was not run), prevalence and counts. Also the cohort descriptions, the
        frozen internal preprocessing and how each model family consumes missingness.
        """
        _require(bool(self.external_labels), "this run scored no external cohort")
        model_seed = 0 if 0 in self.config.seeds else int(self.config.seeds[0])
        models: dict[str, Any] = {}
        for name in self.config.models:
            runs = [run for run in self.runs if run.model == name]
            if runs:
                models[name] = {
                    cohort: self._external_block(runs, cohort, model_seed, n_boot, ci, bootstrap_seed)
                    for cohort in self.external_labels
                }
        return json_safe(
            {
                "protocol": "v2_supervised_external_frozen",
                "bootstrap": {
                    "function": "src.icu.v2.stats.patient_cluster_bootstrap",
                    "resample_unit": "subject_id",
                    "model_seed": model_seed,
                    "n_boot": int(n_boot),
                    "ci": float(ci),
                    "seed": int(bootstrap_seed),
                },
                "models": models,
                "cohorts": {cohort: dict(info) for cohort, info in self.external_info.items()},
                "frozen_preprocessing": dict(self.frozen_preprocessing),
                "missingness_by_model": {name: MISSINGNESS_BY_MODEL[name] for name in self.config.models},
                "notes": [
                    "Each (model, seed) was trained and selected on the internal train/val splits only, scored the "
                    "internal test split once, and then scored every external cohort once with the frozen model.",
                    "External episodes used the internal preprocessing (training medians, z-scoring statistics, "
                    "input features, padded length); nothing was refit or recalibrated on external data.",
                    "A feature a source does not provide is missing in every bin, encoded like an unobserved "
                    "channel of an internal episode (see missingness_by_model and cohorts.*.all_missing_features).",
                    "Seed mean/std is not a confidence interval; the bootstrap block resamples patients for one "
                    "model seed.",
                ],
            }
        )

    def _external_block(
        self,
        runs: Sequence[SeedResult],
        cohort: str,
        model_seed: int,
        n_boot: int,
        ci: float,
        bootstrap_seed: int,
    ) -> dict[str, Any]:
        info = self.external_info.get(cohort, {})
        block: dict[str, Any] = {"family": runs[0].family, "n_seeds": len(runs)}
        for key in ("episodes", "positives", "controls", "patients", "prevalence"):
            block[key] = info.get(key)
        for metric in EXTERNAL_METRICS:
            mean, std = _mean_std([run.external_metrics.get(cohort, {}).get(metric) for run in runs])
            block[f"{metric}_mean"] = mean
            block[f"{metric}_std"] = std
        block["per_seed"] = [
            {"seed": run.seed, **{m: run.external_metrics.get(cohort, {}).get(m) for m in EXTERNAL_METRICS}}
            for run in runs
        ]
        chosen = next((run for run in runs if run.seed == model_seed), None)
        if chosen is not None and cohort in chosen.external_probabilities:
            labels, subjects = self.external_labels[cohort], self.external_subjects[cohort]
            probabilities = chosen.external_probabilities[cohort]
            episode_ids = list(labels)
            y = np.asarray([labels[episode_id] for episode_id in episode_ids], dtype=np.int64)
            scores = np.asarray([probabilities[episode_id] for episode_id in episode_ids], dtype=float)
            groups = np.asarray([subjects[episode_id] for episode_id in episode_ids], dtype=np.int64)
            block["bootstrap"] = {"model_seed": int(chosen.seed)}
            for metric in ("auprc", "auroc"):
                block["bootstrap"][metric] = patient_cluster_bootstrap(
                    y, scores, groups, metric=metric, n_boot=n_boot, seed=bootstrap_seed, ci=ci
                )
        return block


def json_safe(value: Any) -> Any:
    """Convert numpy scalars/arrays, tuples and paths to JSON types; non-finite floats to None."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _mean_std(values: Iterable[Any]) -> tuple[float | None, float | None]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None, None
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return float(np.mean(finite)), std


# ---------------------------------------------------------------------------- training


def _auprc(y: np.ndarray, prob: np.ndarray) -> float:
    return float(average_precision_score(y, prob))


def _metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    metrics = compute_binary_metrics(np.asarray(y), np.asarray(prob, dtype=float))
    metrics.pop("lead_time_precision_at_recall_050", None)  # needs lead times; always NaN here
    return metrics


def _thread_limit(num_threads: int | None) -> contextlib.AbstractContextManager:
    if num_threads is None:
        return contextlib.nullcontext()
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=int(num_threads))


def _result(
    name: str,
    seed: int,
    selected: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    n_inputs: int,
    val: SplitArrays,
    val_prob: np.ndarray,
    test: SplitArrays,
    test_prob: np.ndarray,
    started: float,
    best_epoch: int | None = None,
    epochs_trained: int | None = None,
    stopped_early: bool | None = None,
) -> SeedResult:
    test_prob = np.asarray(test_prob, dtype=float)
    return SeedResult(
        model=name,
        family=get_default_registry()[name].family,
        seed=int(seed),
        selected=dict(selected),
        selection_trace=tuple(dict(step) for step in trace),
        best_epoch=best_epoch,
        epochs_trained=epochs_trained,
        stopped_early=stopped_early,
        n_model_inputs=int(n_inputs),
        val_metrics=_metrics(val.y, val_prob),
        test_metrics=_metrics(test.y, test_prob),
        test_probabilities={episode_id: float(p) for episode_id, p in zip(test.episode_ids, test_prob)},
        fit_seconds=float(time.perf_counter() - started),
    )


def _with_external(
    result: SeedResult,
    externals: Sequence[ExternalCohort],
    predict: Callable[[SplitArrays], np.ndarray],
) -> SeedResult:
    """Score every external cohort once with the frozen model behind ``result``.

    ``result`` is already final (selection, validation and the single test pass), so
    external data cannot change it; ``predict`` maps prepared arrays to probabilities with
    the selected model and nothing else.
    """
    if not externals:
        return result
    metrics: dict[str, dict[str, Any]] = {}
    probabilities: dict[str, dict[str, float]] = {}
    for cohort in externals:
        part = cohort.arrays
        prob = np.asarray(predict(part), dtype=float).reshape(-1)
        if prob.shape != part.y.shape or not np.isfinite(prob).all():
            raise FloatingPointError(
                f"{result.model} (seed {result.seed}): external cohort {cohort.name!r} needs {part.y.size} "
                f"finite probabilities, got shape {prob.shape}"
            )
        metrics[cohort.name] = _metrics(part.y, prob)
        probabilities[cohort.name] = {episode_id: float(p) for episode_id, p in zip(part.episode_ids, prob)}
    return replace(result, external_metrics=metrics, external_probabilities=probabilities)


def _fit_lr(
    inputs: SupervisedInputs,
    config: SupervisedConfig,
    seed: int,
    externals: Sequence[ExternalCohort] = (),
) -> SeedResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    started = time.perf_counter()
    train, val, test = (inputs.splits[split] for split in SPLITS)
    trace: list[dict[str, Any]] = []
    best: tuple[float, float, Any, np.ndarray] | None = None
    with _thread_limit(config.num_threads):
        for c in config.lr_c_grid:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=c, max_iter=_LR_MAX_ITER, random_state=seed, class_weight="balanced")),
                ]
            )
            model.fit(train.tabular, train.y)
            val_prob = model.predict_proba(val.tabular)[:, 1]
            score = _auprc(val.y, val_prob)
            trace.append({"C": c, "val_auprc": score})
            if best is None or score > best[0]:
                best = (score, c, model, val_prob)
        assert best is not None
        _, c_best, model, val_prob = best
        test_prob = model.predict_proba(test.tabular)[:, 1]  # the single test pass
    n_inputs = train.tabular.shape[1]
    result = _result("lr_stats", seed, {"C": c_best}, trace, n_inputs, val, val_prob, test, test_prob, started)
    with _thread_limit(config.num_threads):
        return _with_external(result, externals, lambda part: model.predict_proba(part.tabular)[:, 1])


def _staged_proba(model: Any, features: np.ndarray, n_iter: int) -> np.ndarray:
    """Probabilities of the first ``n_iter`` boosting iterations (equals refitting with max_iter=n_iter)."""
    return next(itertools.islice(model.staged_predict_proba(features), n_iter - 1, None))[:, 1]


def _fit_gbdt(
    inputs: SupervisedInputs,
    config: SupervisedConfig,
    seed: int,
    externals: Sequence[ExternalCohort] = (),
) -> SeedResult:
    from sklearn.ensemble import HistGradientBoostingClassifier

    started = time.perf_counter()
    train, val, test = (inputs.splits[split] for split in SPLITS)
    trace: list[dict[str, Any]] = []
    best: tuple[float, int, int, Any] | None = None
    with _thread_limit(config.num_threads):
        for depth in config.gbdt_max_depth_grid:
            model = HistGradientBoostingClassifier(
                learning_rate=config.gbdt_learning_rate,
                max_depth=depth,
                max_iter=config.gbdt_max_iter,
                early_stopping=False,
                random_state=seed,
            )
            model.fit(train.tabular, train.y)
            curve = np.asarray([_auprc(val.y, stage[:, 1]) for stage in model.staged_predict_proba(val.tabular)])
            n_iter = int(np.argmax(np.where(np.isfinite(curve), curve, -np.inf))) + 1
            trace.append({"max_depth": depth, "max_iter": n_iter, "val_auprc": float(curve[n_iter - 1])})
            if best is None or curve[n_iter - 1] > best[0]:
                best = (float(curve[n_iter - 1]), depth, n_iter, model)
        assert best is not None
        _, depth, n_iter, model = best
        val_prob = _staged_proba(model, val.tabular, n_iter)
        test_prob = _staged_proba(model, test.tabular, n_iter)  # the single test pass
    selected = {"max_depth": depth, "max_iter": n_iter}
    result = _result(
        "gbdt_stats", seed, selected, trace, train.tabular.shape[1], val, val_prob, test, test_prob, started
    )
    with _thread_limit(config.num_threads):
        return _with_external(result, externals, lambda part: _staged_proba(model, part.tabular, n_iter))


def _build_sequence_model(name: str, input_dim: int, max_len: int, config: SupervisedConfig) -> Any:
    """Instantiate the v1 model class with the ``fit_all.py`` architecture settings."""
    hidden, dropout = int(config.hidden_size), float(config.dropout)
    if name == "grud_lite":
        from src.icu.comparators.models.grud import GRUDLite

        return GRUDLite(input_dim=input_dim, hidden_size=hidden, dropout=dropout)
    if name == "retain_seq":
        from src.icu.comparators.train_retain import RETAINClassifier

        return RETAINClassifier(input_dim=input_dim, hidden_size=hidden, dropout=dropout, max_len=max_len)
    if name == "mtan_seq":
        from src.icu.comparators.train_mtan import MTANClassifier

        return MTANClassifier(input_dim=input_dim, latent_dim=hidden, dropout=dropout, max_len=max_len)
    if name == "adacare_seq":
        from src.icu.comparators.train_adacare import AdaCareClassifier

        return AdaCareClassifier(input_dim=input_dim, hidden_size=hidden, dropout=dropout, max_len=max_len)
    if name == "strats_seq":
        from src.icu.comparators.train_strats import STraTSClassifier

        return STraTSClassifier(
            input_dim=input_dim,
            d_model=hidden,
            num_layers=int(config.strats_num_layers),
            dropout=dropout,
            max_len=max_len,
        )
    raise ValueError(f"{name!r} is not a sequence model")


def _forward(model: Any, name: str, x: Any, m: Any, d: Any, lengths: Any) -> Any:
    # GRUDLite takes no lengths and reads the last (possibly padded) step, as in v1.
    return model(x, m, d) if name == "grud_lite" else model(x, m, d, lengths)


def _predict_sequence(model: Any, name: str, tensors: tuple[Any, ...], batch_size: int) -> np.ndarray:
    import torch

    x, m, d, lengths = tensors
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), batch_size):
            stop = start + batch_size
            logits = _forward(model, name, x[start:stop], m[start:stop], d[start:stop], lengths[start:stop])
            chunks.append(torch.sigmoid(logits.reshape(-1)).cpu().numpy().astype(float))
    prob = np.concatenate(chunks)
    if not np.isfinite(prob).all():
        raise FloatingPointError(f"{name}: non-finite predicted probabilities")
    return prob


def _fit_sequence_model(
    name: str,
    inputs: SupervisedInputs,
    config: SupervisedConfig,
    seed: int,
    log: Logger | None,
    externals: Sequence[ExternalCohort] = (),
) -> SeedResult:
    import torch

    started = time.perf_counter()
    if config.num_threads is not None:
        torch.set_num_threads(int(config.num_threads))
    device = torch.device(config.device)
    train, val, test = (inputs.splits[split] for split in SPLITS)

    def as_tensors(part: SplitArrays) -> tuple[Any, ...]:
        arrays = (part.values, part.observed, part.deltas, part.lengths)
        return tuple(torch.from_numpy(array).to(device) for array in arrays)

    tensors = {split: as_tensors(part) for split, part in inputs.splits.items()}
    y_train = torch.as_tensor(train.y, dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    model = _build_sequence_model(name, len(inputs.input_features), inputs.max_len, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    n_positive = float(train.y.sum())
    pos_weight = torch.tensor([(train.y.size - n_positive) / n_positive], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator().manual_seed(seed)
    stopper = EarlyStopping(patience=config.early_stopping_patience, min_delta=config.early_stopping_min_delta)

    x, m, d, lengths = tensors["train"]
    n_train = int(train.y.size)
    best_state: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = []
    stopped_early = False
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        loss_sum = 0.0
        for start in range(0, n_train, config.batch_size):
            batch = order[start : start + config.batch_size]
            logits = _forward(model, name, x[batch], m[batch], d[batch], lengths[batch]).reshape(-1)
            loss = criterion(logits, y_train[batch])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{name} (seed {seed}): non-finite training loss in epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            optimizer.step()
            loss_sum += float(loss.detach()) * int(batch.numel())
        val_auprc = _auprc(val.y, _predict_sequence(model, name, tensors["val"], config.eval_batch_size))
        improved = stopper.update(epoch, val_auprc)
        if improved:
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
        trace.append({"epoch": epoch, "train_loss": loss_sum / n_train, "val_auprc": val_auprc, "improved": improved})
        if log is not None:
            mark = " *" if improved else ""
            log(f"{name} seed={seed} epoch={epoch} train_loss={loss_sum / n_train:.4f} val_auprc={val_auprc:.4f}{mark}")
        if stopper.should_stop:
            stopped_early = True
            break
    if best_state is None:
        raise RuntimeError(f"{name} (seed {seed}): validation AUPRC was never finite")
    model.load_state_dict(best_state)
    val_prob = _predict_sequence(model, name, tensors["val"], config.eval_batch_size)
    test_prob = _predict_sequence(model, name, tensors["test"], config.eval_batch_size)  # the single test pass
    result = _result(
        name,
        seed,
        {"epochs": stopper.best_epoch},
        trace,
        len(inputs.input_features),
        val,
        val_prob,
        test,
        test_prob,
        started,
        best_epoch=stopper.best_epoch,
        epochs_trained=len(trace),
        stopped_early=stopped_early,
    )
    return _with_external(
        result,
        externals,
        lambda part: _predict_sequence(model, name, as_tensors(part), config.eval_batch_size),
    )


def _check_inputs_match_config(inputs: SupervisedInputs, config: SupervisedConfig) -> None:
    _require(
        set(inputs.excluded_features) == set(config.excluded_features),
        f"inputs were built without {sorted(set(config.excluded_features) - set(inputs.excluded_features))} "
        f"or with extra exclusions {sorted(set(inputs.excluded_features) - set(config.excluded_features))}; "
        "rebuild them with config.excluded_features",
    )


def _check_externals(
    inputs: SupervisedInputs,
    externals: Iterable[ExternalCohort] | None,
) -> tuple[ExternalCohort, ...]:
    """External cohorts must have distinct names and be prepared from these very inputs."""
    cohorts = tuple(externals or ())
    names = [cohort.name for cohort in cohorts]
    _require(len(set(names)) == len(names), f"duplicate external cohort names {names}")
    for cohort in cohorts:
        _require(
            inputs.preprocessing is not None and cohort.preprocessing is inputs.preprocessing,
            f"external cohort {cohort.name!r} was not prepared with these inputs' training statistics; "
            "build it with build_external_cohort(..., inputs)",
        )
    return cohorts


def fit_model(
    name: str,
    inputs: SupervisedInputs,
    config: SupervisedConfig,
    seed: int,
    log: Logger | None = None,
    externals: Sequence[ExternalCohort] = (),
) -> SeedResult:
    """Fit one reference model for one seed: select on validation, then score test once.

    Each cohort in ``externals`` (:func:`build_external_cohort` on the same ``inputs``) is
    scored once afterwards with the frozen selected model.
    """
    _require(name in SUPERVISED_MODELS, f"unknown supervised model {name!r}")
    _check_inputs_match_config(inputs, config)
    cohorts = _check_externals(inputs, externals)
    if name == "lr_stats":
        return _fit_lr(inputs, config, int(seed), cohorts)
    if name == "gbdt_stats":
        return _fit_gbdt(inputs, config, int(seed), cohorts)
    return _fit_sequence_model(name, inputs, config, int(seed), log, cohorts)


def fit_supervised_models(
    inputs: SupervisedInputs,
    config: SupervisedConfig,
    on_result: Callable[[SeedResult], None] | None = None,
    log: Logger | None = None,
    externals: Sequence[ExternalCohort] = (),
) -> SupervisedRunResult:
    """Fit every configured model for every seed; ``on_result`` sees each fit as it finishes.

    ``externals`` (prepared with :func:`build_external_cohort` from ``inputs``) are scored
    with each frozen (model, seed) after its internal result is final.
    """
    _check_inputs_match_config(inputs, config)
    cohorts = _check_externals(inputs, externals)
    runs: list[SeedResult] = []
    for name in config.models:
        for seed in config.seeds:
            result = fit_model(name, inputs, config, seed, log=log, externals=cohorts)
            runs.append(result)
            if on_result is not None:
                on_result(result)
    test = inputs.splits["test"]
    external: dict[str, Any] = {}
    if cohorts:
        assert inputs.preprocessing is not None
        external = {
            "external_labels": {
                cohort.name: dict(zip(cohort.arrays.episode_ids, cohort.arrays.y.tolist())) for cohort in cohorts
            },
            "external_subjects": {
                cohort.name: dict(zip(cohort.arrays.episode_ids, cohort.arrays.subject_ids.tolist()))
                for cohort in cohorts
            },
            "external_info": {cohort.name: dict(cohort.info) for cohort in cohorts},
            "frozen_preprocessing": {
                **inputs.preprocessing.to_dict(),
                "internal_train_observed_fraction": _observed_fraction(inputs.splits["train"], inputs.input_features),
            },
        }
    return SupervisedRunResult(
        config=config,
        data_summary=inputs.summary(),
        test_labels={episode_id: int(label) for episode_id, label in zip(test.episode_ids, test.y)},
        test_subjects={episode_id: int(subject) for episode_id, subject in zip(test.episode_ids, test.subject_ids)},
        runs=tuple(runs),
        **external,
    )


def run_supervised_v2(
    episodes: Sequence[EpisodeRecord],
    feature_names: Sequence[str],
    loader: ArrayLoader,
    config: SupervisedConfig | None = None,
    on_result: Callable[[SeedResult], None] | None = None,
    log: Logger | None = None,
    externals: Sequence[ExternalSource] = (),
) -> SupervisedRunResult:
    """Build leakage-safe inputs (``config.excluded_features``) and fit all models and seeds.

    Each :class:`ExternalSource` is prepared with the internal training statistics and
    scored with every frozen (model, seed).
    """
    config = config if config is not None else SupervisedConfig()
    inputs = build_supervised_inputs(episodes, feature_names, loader, config.excluded_features, log=log)
    cohorts = [
        build_external_cohort(source.name, source.episodes, source.loader, inputs, source.feature_names, log=log)
        for source in externals
    ]
    return fit_supervised_models(inputs, config, on_result=on_result, log=log, externals=cohorts)
