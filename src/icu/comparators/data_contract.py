from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    stay_id: int
    subject_id: int
    episode_type: str
    data_path: Path
    meta_path: Path
    mask_path: Path
    feature_names: tuple[str, ...]
    shape: tuple[int, int]


def load_episode_index(dataset_dir: Path | str) -> list[dict]:
    root = Path(dataset_dir)
    index_path = root / "shock_episode_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"shock episode index not found: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("shock_episode_index.json must be a list")
    return payload


def build_episode_records(
    dataset_dir: Path | str,
    limit: int = 0,
    episode_types: tuple[str, ...] | None = None,
) -> list[EpisodeRecord]:
    root = Path(dataset_dir)
    rows = load_episode_index(root)
    allowed_types = set(episode_types or ())

    records: list[EpisodeRecord] = []
    for row in rows:
        episode_id = str(row.get("episode_id", "")).strip()
        if not episode_id:
            continue
        episode_type = str(row.get("episode_type", "")).strip() or "unknown"
        if allowed_types and episode_type not in allowed_types:
            continue

        stay_raw = row.get("stay_id")
        if stay_raw is None:
            continue
        try:
            stay_id = int(stay_raw)
        except (TypeError, ValueError):
            continue

        subject_raw = row.get("subject_id", stay_raw)
        try:
            subject_id = int(subject_raw)
        except (TypeError, ValueError):
            subject_id = stay_id

        data_path = root / f"shock_episode_{episode_id}.npy"
        meta_path = root / f"shock_episode_{episode_id}.json"
        mask_name = str(row.get("missing_mask_file") or f"shock_episode_{episode_id}_missing_mask.npy")
        mask_path = root / mask_name

        if not data_path.exists() or not meta_path.exists() or not mask_path.exists():
            continue

        feature_names = tuple(str(name) for name in row.get("feature_names", []))
        shape = row.get("shape", [0, 0])
        if not isinstance(shape, list) or len(shape) != 2:
            shape = [0, 0]
        record = EpisodeRecord(
            episode_id=episode_id,
            stay_id=stay_id,
            subject_id=subject_id,
            episode_type=episode_type,
            data_path=data_path,
            meta_path=meta_path,
            mask_path=mask_path,
            feature_names=feature_names,
            shape=(int(shape[0]), int(shape[1])),
        )
        records.append(record)
        if int(limit) > 0 and len(records) >= int(limit):
            break
    return records


def split_episode_records(
    records: list[EpisodeRecord],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> dict[str, list[str]]:
    if not records:
        return {"train": [], "val": [], "test": []}

    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("split ratios must be non-negative and train must be positive")
    total_ratio = float(train_ratio + val_ratio + test_ratio)
    if total_ratio <= 0:
        raise ValueError("sum of ratios must be positive")

    split_names = ("train", "val", "test")
    ratio_map = {
        "train": float(train_ratio) / total_ratio,
        "val": float(val_ratio) / total_ratio,
        "test": float(test_ratio) / total_ratio,
    }

    total_episodes = len(records)
    targets = {
        name: int(round(total_episodes * ratio_map[name]))
        for name in split_names
    }
    targets["test"] = total_episodes - targets["train"] - targets["val"]
    for name in split_names:
        if ratio_map[name] > 0.0 and targets[name] <= 0 and total_episodes >= len([n for n in split_names if ratio_map[n] > 0.0]):
            targets[name] = 1
    while sum(targets.values()) > total_episodes:
        donor = max(split_names, key=lambda n: targets[n])
        targets[donor] -= 1
    while sum(targets.values()) < total_episodes:
        receiver = max(split_names, key=lambda n: ratio_map[n])
        targets[receiver] += 1

    stay_index: dict[int, dict] = {}
    for record in records:
        info = stay_index.setdefault(
            int(record.stay_id),
            {
                "stay_id": int(record.stay_id),
                "episode_ids": [],
                "positive": 0,
                "control": 0,
                "total": 0,
            },
        )
        info["episode_ids"].append(record.episode_id)
        info["total"] += 1
        if record.episode_type == "positive":
            info["positive"] += 1
        elif record.episode_type == "control":
            info["control"] += 1

    stay_infos = [stay_index[sid] for sid in sorted(stay_index)]
    rng = random.Random(int(seed))
    rng.shuffle(stay_infos)
    for order_idx, info in enumerate(stay_infos):
        info["_order_idx"] = order_idx
        info["_seed_noise"] = float(rng.random())

    split_stays: dict[str, list[int]] = {name: [] for name in split_names}
    split_episode_count: dict[str, int] = {name: 0 for name in split_names}
    split_positive_count: dict[str, int] = {name: 0 for name in split_names}
    split_control_count: dict[str, int] = {name: 0 for name in split_names}

    def _stay_has_positive(info: dict) -> bool:
        return int(info["positive"]) > 0

    def _stay_has_control(info: dict) -> bool:
        return int(info["control"]) > 0

    def _apply_assign(split_name: str, info: dict) -> None:
        split_stays[split_name].append(int(info["stay_id"]))
        split_episode_count[split_name] += int(info["total"])
        split_positive_count[split_name] += int(info["positive"])
        split_control_count[split_name] += int(info["control"])

    # initialize with all stays in train, then move to val/test by target episodes
    for info in stay_infos:
        _apply_assign("train", info)

    global_has_positive = any(record.episode_type == "positive" for record in records)
    global_has_control = any(record.episode_type == "control" for record in records)

    def _move_stay(donor: str, receiver: str, stay_id: int) -> None:
        info = stay_index[int(stay_id)]
        split_stays[donor].remove(int(stay_id))
        split_episode_count[donor] -= int(info["total"])
        split_positive_count[donor] -= int(info["positive"])
        split_control_count[donor] -= int(info["control"])
        _apply_assign(receiver, info)

    def _pick_best_train_move(receiver: str, require_positive: bool | None = None, require_control: bool | None = None) -> int | None:
        best: tuple[float, int] | None = None
        for stay_id in list(split_stays["train"]):
            info = stay_index[int(stay_id)]
            if require_positive is True and int(info["positive"]) <= 0:
                continue
            if require_control is True and int(info["control"]) <= 0:
                continue
            train_after = split_episode_count["train"] - int(info["total"])
            if train_after <= 0:
                continue
            train_pos_after = split_positive_count["train"] - int(info["positive"])
            train_ctrl_after = split_control_count["train"] - int(info["control"])
            if global_has_positive and train_pos_after <= 0:
                continue
            if global_has_control and train_ctrl_after <= 0:
                continue

            recv_after = split_episode_count[receiver] + int(info["total"])
            delta = abs(recv_after - targets[receiver]) + abs(train_after - targets["train"])
            if require_positive is None and split_positive_count[receiver] == 0 and int(info["positive"]) > 0:
                delta -= 5.0
            if require_control is None and split_control_count[receiver] == 0 and int(info["control"]) > 0:
                delta -= 5.0
            jitter = 0.499 * float(info.get("_seed_noise", 0.0))
            candidate = (float(delta) + jitter, int(info.get("_order_idx", 0)))
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return None
        selected_rank = int(best[1])
        for stay_id in list(split_stays["train"]):
            info = stay_index[int(stay_id)]
            if int(info.get("_order_idx", -1)) == selected_rank:
                return int(stay_id)
        return None

    for receiver in ("val", "test"):
        while targets[receiver] > 0 and split_episode_count[receiver] < targets[receiver]:
            stay_id = _pick_best_train_move(receiver)
            if stay_id is None:
                break
            _move_stay("train", receiver, stay_id)

    def _ensure_receiver_has_class(receiver: str, need_positive: bool) -> None:
        if split_episode_count[receiver] <= 0:
            return
        if need_positive and split_positive_count[receiver] > 0:
            return
        if (not need_positive) and split_control_count[receiver] > 0:
            return
        stay_id = _pick_best_train_move(
            receiver,
            require_positive=True if need_positive else None,
            require_control=True if not need_positive else None,
        )
        if stay_id is not None:
            _move_stay("train", receiver, stay_id)

    if global_has_positive and global_has_control:
        for receiver in ("val", "test"):
            _ensure_receiver_has_class(receiver, need_positive=True)
            _ensure_receiver_has_class(receiver, need_positive=False)

    stay_to_split: dict[int, str] = {}
    for split_name in split_names:
        for stay_id in split_stays[split_name]:
            stay_to_split[int(stay_id)] = split_name

    split = {"train": [], "val": [], "test": []}
    for record in records:
        split_name = stay_to_split.get(int(record.stay_id))
        if split_name is None:
            raise RuntimeError("record stay not assigned to any split")
        split[split_name].append(record.episode_id)

    split["train"].sort()
    split["val"].sort()
    split["test"].sort()
    return split


def build_split_manifest(
    dataset_name: str,
    dataset_dir: Path | str,
    split: dict[str, list[str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict:
    train = list(split.get("train", []))
    val = list(split.get("val", []))
    test = list(split.get("test", []))
    return {
        "dataset_name": str(dataset_name),
        "dataset_dir": str(Path(dataset_dir)),
        "split": {
            "seed": int(seed),
            "ratios": {
                "train": float(train_ratio),
                "val": float(val_ratio),
                "test": float(test_ratio),
            },
            "train": train,
            "val": val,
            "test": test,
        },
        "counts": {
            "episodes_train": len(train),
            "episodes_val": len(val),
            "episodes_test": len(test),
            "episodes_total": len(train) + len(val) + len(test),
        },
    }


def write_split_manifest(manifest: dict, out_path: Path | str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")


def load_split_manifest(path: Path | str) -> dict:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("split manifest must be a JSON object")

    split = payload.get("split")
    if not isinstance(split, dict):
        raise ValueError("split manifest must include a 'split' object")

    for key in ("train", "val", "test"):
        value = split.get(key)
        if not isinstance(value, list):
            raise ValueError(f"split manifest 'split.{key}' must be a list")
    return payload


def load_or_build_split(
    records: list[EpisodeRecord],
    split_manifest_path: Path | str | None = None,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> dict[str, list[str]]:
    """Load a frozen split manifest, or build and return a new split.

    If split_manifest_path is provided and the file exists, the frozen split
    is loaded. Otherwise a new patient-grouped split is generated.
    """
    if split_manifest_path:
        manifest_path = Path(split_manifest_path)
        if manifest_path.exists():
            manifest = load_split_manifest(manifest_path)
            return manifest["split"]

    return split_episode_records(
        records,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def episode_record_to_dict(record: EpisodeRecord) -> dict:
    row = asdict(record)
    row["data_path"] = str(record.data_path)
    row["meta_path"] = str(record.meta_path)
    row["mask_path"] = str(record.mask_path)
    row["feature_names"] = list(record.feature_names)
    row["shape"] = list(record.shape)
    return row
