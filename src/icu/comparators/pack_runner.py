from __future__ import annotations

from pathlib import Path
import json
import inspect

from .data_contract import build_episode_records, build_split_manifest, load_or_build_split, split_episode_records, write_split_manifest
from .feature_builder import build_handcrafted_feature_matrix
from .metrics import compute_binary_metrics
from .registry import get_default_registry, resolve_trainer
from .train_gbdt import train_evaluate_gbdt
from .train_grud import train_evaluate_grud
from .train_lr import train_evaluate_lr
from .train_transformer import train_evaluate_transformer


def run_selected_models(
    registry: dict,
    enabled_names: list[str],
    records: list,
    split: dict,
    config: dict | None = None,
) -> dict:
    """Run a subset of models from the registry with optional per-model config."""
    config = config or {}
    results = {}
    handcrafted_bundle = None
    for name in enabled_names:
        if name not in registry:
            raise KeyError(f"Model '{name}' not found in registry")
        spec = registry[name]
        trainer = resolve_trainer(spec.trainer)
        model_config = config.get(name, {})
        trainer_params = inspect.signature(trainer).parameters
        if "bundle" in trainer_params:
            if handcrafted_bundle is None:
                handcrafted_bundle = build_handcrafted_feature_matrix(records)
            results[name] = trainer(bundle=handcrafted_bundle, split=split, **model_config)
        elif "records" in trainer_params:
            results[name] = trainer(records=records, split=split, **model_config)
        else:
            raise TypeError(
                f"Trainer '{spec.trainer}' for model '{name}' must accept either 'bundle' or 'records'"
            )
    return results


def build_artifact_manifest(
    summary: dict,
    split_manifest_path: str | None,
    feature_schema_path: str | None,
) -> dict:
    """Build a frozen artifact manifest consumed by downstream transfer/stress runners."""
    return {
        "split_manifest_path": str(split_manifest_path or ""),
        "feature_schema_path": str(feature_schema_path or ""),
        "model_artifacts": {
            model_name: {
                "model_family": summary.get("_registry", {}).get(model_name, {}).get("family", ""),
            }
            for model_name in summary.get("models", {})
        },
    }


def run_comparator_pack(
    input_dir: Path | str,
    out_dir: Path | str,
    limit: int = 120,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    balance_episode_types: bool = True,
    run_lr: bool = True,
    run_gbdt: bool = True,
    run_grud: bool = True,
    grud_epochs: int = 12,
    grud_hidden_size: int = 64,
    run_transformer: bool = True,
    transformer_epochs: int = 10,
    transformer_d_model: int = 64,
    transformer_nhead: int = 4,
    transformer_num_layers: int = 2,
    transformer_learning_rate: float = 1e-3,
    split_manifest_path: Path | str | None = None,
    execution_manifest: dict | None = None,
    enabled_models: list[str] | None = None,
) -> dict:
    input_path = Path(input_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = build_episode_records(input_path, limit=0, episode_types=("positive", "control"))

    if int(limit) > 0:
        if balance_episode_types:
            positives = [record for record in records if record.episode_type == "positive"]
            controls = [record for record in records if record.episode_type == "control"]
            target_each = int(limit) // 2
            selected_ids = set()
            selected: list = []

            def _append_candidates(candidates, quota):
                added = 0
                for candidate in candidates:
                    if added >= quota:
                        break
                    if candidate.episode_id in selected_ids:
                        continue
                    selected.append(candidate)
                    selected_ids.add(candidate.episode_id)
                    added += 1

            _append_candidates(positives, target_each)
            _append_candidates(controls, target_each)

            remaining = int(limit) - len(selected)
            if remaining > 0:
                leftovers = [record for record in records if record.episode_id not in selected_ids]
                _append_candidates(leftovers, remaining)
            records = selected
        else:
            records = records[: int(limit)]

    split = load_or_build_split(
        records,
        split_manifest_path=split_manifest_path,
        seed=int(seed),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
    )

    split_manifest = build_split_manifest(
        dataset_name=input_path.name,
        dataset_dir=input_path,
        split={"train": split.get("train", []), "val": split.get("val", []), "test": split.get("test", [])},
        seed=int(seed),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
    )
    write_split_manifest(split_manifest, output_path / "comparator_split_manifest.json")

    if enabled_models:
        registry = get_default_registry()
        config = {}
        if execution_manifest:
            pass  # future: load per-model config from execution_manifest
        models = run_selected_models(registry, enabled_models, records, split, config=config)
    else:
        handcrafted_bundle = build_handcrafted_feature_matrix(records)
        models = {}
        if run_lr:
            models["lr_stats"] = train_evaluate_lr(handcrafted_bundle, split, random_state=int(seed))
        if run_gbdt:
            models["gbdt_stats"] = train_evaluate_gbdt(handcrafted_bundle, split, random_state=int(seed))

        if run_grud:
            models["grud_lite"] = train_evaluate_grud(
                records,
                split,
                random_state=int(seed),
                epochs=int(grud_epochs),
                hidden_size=int(grud_hidden_size),
            )

        if run_transformer:
            models["transformer_seq"] = train_evaluate_transformer(
                records,
                split,
                random_state=int(seed),
                epochs=int(transformer_epochs),
                d_model=int(transformer_d_model),
                nhead=int(transformer_nhead),
                num_layers=int(transformer_num_layers),
                learning_rate=float(transformer_learning_rate),
            )

    if not models:
        raise ValueError("At least one comparator model must be enabled")

    summary = {
        "dataset": {
            "input_dir": str(input_path),
            "episode_count": int(len(records)),
            "feature_dim": int(len(records[0].feature_names)) if records else 0,
            "balanced_episode_types": bool(balance_episode_types),
        },
        "split": split_manifest,
        "models": models,
    }

    predictions = {
        model_name: model_result.get("predictions", {})
        for model_name, model_result in models.items()
    }

    (output_path / "comparator_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_path / "comparator_metrics.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
    (output_path / "comparator_predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    artifact_manifest = build_artifact_manifest(
        summary,
        split_manifest_path=str(split_manifest_path) if split_manifest_path else None,
        feature_schema_path=str(output_path / "feature_schema.json") if (output_path / "feature_schema.json").exists() else None,
    )
    (output_path / "comparator_artifact_manifest.json").write_text(json.dumps(artifact_manifest, indent=2), encoding="utf-8")
    return summary
