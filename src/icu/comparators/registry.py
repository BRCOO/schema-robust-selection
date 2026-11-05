"""Model registry for mandatory supervised and mining comparator baselines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComparatorSpec:
    """Specification for a single comparator baseline.

    Attributes:
        name: Unique key used to reference this model in the registry.
        family: High-level family grouping (statistical, classic_clinical, etc.).
        kind: 'supervised' for end-to-end predictors, 'mining' for discovery-only.
        trainer: Fully-qualified import path or function name for the trainer.
    """

    name: str
    family: str
    kind: str
    trainer: str


def get_default_registry() -> dict[str, ComparatorSpec]:
    """Return the default registry of all mandatory comparator baselines."""
    return {
        "lr_stats": ComparatorSpec("lr_stats", "statistical", "supervised", "train_evaluate_lr"),
        "gbdt_stats": ComparatorSpec("gbdt_stats", "statistical", "supervised", "train_evaluate_gbdt"),
        "retain_seq": ComparatorSpec("retain_seq", "classic_clinical", "supervised", "train_evaluate_retain"),
        "grud_lite": ComparatorSpec("grud_lite", "missingness_aware", "supervised", "train_evaluate_grud"),
        "mtan_seq": ComparatorSpec("mtan_seq", "modern_irregular", "supervised", "train_evaluate_mtan"),
        "adacare_seq": ComparatorSpec("adacare_seq", "modern_clinical", "supervised", "train_evaluate_adacare"),
        "strats_seq": ComparatorSpec("strats_seq", "modern_dependency", "supervised", "train_evaluate_strats"),
    }


def resolve_trainer(trainer_name: str):
    """Dynamically resolve a trainer function by name.

    This function is a placeholder for the registry-based resolution in
    pack_runner. Each trainer module is expected to expose the named function.
    """
    import importlib

    trainer_map = {
        "train_evaluate_lr": "src.icu.comparators.train_lr",
        "train_evaluate_gbdt": "src.icu.comparators.train_gbdt",
        "train_evaluate_grud": "src.icu.comparators.train_grud",
        "train_evaluate_retain": "src.icu.comparators.train_retain",
        "train_evaluate_mtan": "src.icu.comparators.train_mtan",
        "train_evaluate_adacare": "src.icu.comparators.train_adacare",
        "train_evaluate_strats": "src.icu.comparators.train_strats",
    }

    if trainer_name not in trainer_map:
        raise ValueError(f"Unknown trainer: {trainer_name}")

    module_path = trainer_map[trainer_name]
    module = importlib.import_module(module_path)
    return getattr(module, trainer_name)
