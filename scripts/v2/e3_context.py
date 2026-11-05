"""Shared loader for the add-on analyses of finished E3 runs (reviewer-driven experiments, 2026-11).

Not a pipeline step. ``load_run(e3_dir, extra_dir)`` reads one E3 run (``run_union_selection.py``)
and, when given, its add-on run (``run_union_extra.py``): the pools state, the splits, the union
pool with TRAIN firings, and every saved registry as a ``CompositeRegistry``. Registries are read,
never re-selected, so every analysis sees exactly the registries reported in the paper.

Import from another script with ``e3_context()`` below (the module is registered in
``sys.modules`` first, which dataclasses need).
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.bridge import fit_frozen_bridge_v2  # noqa: E402
from src.icu.v2.pipeline import episodes_by_split, restrict_firings  # noqa: E402
from src.icu.v2.selection_baselines import SelectionInput  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, composite_registry  # noqa: E402
from src.icu.v2.union import union_pool  # noqa: E402

STANDARD = ("f_mimic_iv_gap1h", "f_mimic_iv_gap2h", "f_mimic_iv_gap4h", "f_mimic_iv_aki_gap6h")
LARGE = ("f_mimic_iv_gap1h", "f_mimic_iv_aki_gap6h")
#: the 16 ICU runs of the paper: (E3 root, label); the add-on run is union_extra/<root>/<label>
ICU_RUNS = tuple((root, label) for root in ("union_k5", "union", "union_k20") for label in STANDARD) + \
    tuple((root, label) for root in ("union_large_k10", "union_large_k20") for label in LARGE)
K_OF_ROOT = {"union_k5": 5, "union": 10, "union_k20": 20, "union_large_k10": 10, "union_large_k20": 20}


def run_dirs(root: str, label: str) -> tuple[Path, Path]:
    if root == "union_resplit":  # robustness re-splits keep their add-on runs in a sibling folder
        return ROOT / "artifacts/v2/union_resplit" / label, ROOT / "artifacts/v2/union_resplit_extra" / label
    return ROOT / "artifacts/v2" / root / label, ROOT / "artifacts/v2/union_extra" / root / label


def drive_dirs(k: int, label: str = "f_backblaze_seagate_gap2d") -> tuple[Path, Path]:
    return ROOT / f"artifacts/v2/drive/union_k{k}" / label, ROOT / f"artifacts/v2/drive/union_extra_k{k}" / label


def selection_baselines_module():
    name = "run_selection_baselines"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "v2" / "run_selection_baselines.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _composite(saved: dict[str, Any], by_id: dict[str, Any]) -> CompositeRegistry:
    return CompositeRegistry(tuple(by_id[i] for i in saved["primaries"]),
                             {pid: tuple(by_id[q] for q in qs) for pid, qs in saved["surrogates"].items()})


@dataclass
class Run:
    e3_dir: Path
    extra_dir: Path | None
    pools_path: Path
    state: dict
    source: frozenset
    grouped: dict
    ids: dict
    labels: dict
    train_ids: frozenset
    train_pos: frozenset
    subject_of: dict
    targets: list
    candidates: list
    firings: dict
    external: dict
    by_id: dict
    train_firings: dict
    k: int
    max_surrogates: int
    shortlist: int | None
    registries: dict
    meta: dict

    @property
    def inp(self) -> SelectionInput:
        return SelectionInput(tuple(self.candidates), self.train_firings, self.train_ids, self.train_pos)

    def fit(self, registry) -> dict:
        """The frozen probe of the paper: TRAIN fit, C on VAL (``bridge.fit_frozen_bridge_v2``)."""
        f = {p.program_id: self.firings.get(p.program_id, frozenset()) for p in registry}
        return fit_frozen_bridge_v2(list(registry), {s: restrict_firings(f, frozenset(self.ids[s])) for s in self.ids}, self.ids, self.labels)

    def composite(self, registry) -> CompositeRegistry:
        return composite_registry(registry, self.candidates, self.train_firings, sorted(self.train_ids), self.max_surrogates)

    def matrix(self, programs, episode_ids, firings=None) -> np.ndarray:
        """0/1 matrix, one column per program (None = all-zero column); ``firings`` defaults to
        the internal firings (all source splits)."""
        firings = self.firings if firings is None else firings
        row = {e: i for i, e in enumerate(episode_ids)}
        X = np.zeros((len(episode_ids), len(programs)))
        for j, p in enumerate(programs):
            if p is None:
                continue
            for e in firings.get(p.program_id, ()):
                i = row.get(e)
                if i is not None:
                    X[i, j] = 1.0
        return X


def load_run(e3_dir: Path, extra_dir: Path | None = None) -> Run:
    sb = selection_baselines_module()
    e3_dir = Path(e3_dir)
    manifest = json.loads((e3_dir / "manifest.json").read_text(encoding="utf-8"))
    args = manifest["args"]
    pools_path = ROOT / args["pools"]
    with open(pools_path, "rb") as handle:
        state = pickle.load(handle)
    source = sb.source_schema(state, pools_path.parent / "config.json")
    grouped = episodes_by_split(state["episodes"])
    ids = {name: [e.episode_id for e in rows] for name, rows in grouped.items()}
    labels = {name: [e.label for e in rows] for name, rows in grouped.items()}
    train_ids = frozenset(ids["train"])
    candidates, firings, external = union_pool(state)
    by_id = {c.program_id: c for pool in state["pools"] for c in pool.candidates}
    registries = {name: _composite(reg, by_id) for name, reg in json.loads((e3_dir / "registries.json").read_text(encoding="utf-8")).items()}
    meta = dict(manifest.get("meta", {}))
    if extra_dir is not None and (Path(extra_dir) / "registries.json").exists():
        extra_dir = Path(extra_dir)
        for name, reg in json.loads((extra_dir / "registries.json").read_text(encoding="utf-8")).items():
            registries[name] = _composite(reg, by_id)
        meta.update(json.loads((extra_dir / "manifest.json").read_text(encoding="utf-8")).get("meta", {}))
    else:
        extra_dir = None
    # registries of the later add-on runs, when they exist for this run (same pool, same splits)
    relative = e3_dir.relative_to(ROOT / "artifacts/v2") if (ROOT / "artifacts/v2") in e3_dir.parents else None
    if relative is not None:
        label = relative.as_posix()
        if label.startswith("drive/"):
            k = label.split("/")[1].replace("union_k", "")
            label = f"drive_k{k}/{label.split('/')[-1]}"
        # SRS_AP_ROOT points the SRS-AP registries at another output root (e.g. a re-selection under artifacts/v2)
        for root in ("union_fairness", os.environ.get("SRS_AP_ROOT", "union_ap")):
            path = ROOT / "artifacts/v2" / root / label / "registries.json"
            if path.exists():
                for name, reg in json.loads(path.read_text(encoding="utf-8")).items():
                    registries.setdefault(name, _composite(reg, by_id))
    return Run(
        e3_dir=e3_dir, extra_dir=extra_dir, pools_path=pools_path, state=state, source=source, grouped=grouped, ids=ids, labels=labels,
        train_ids=train_ids, train_pos=frozenset(e.episode_id for e in grouped["train"] if e.label == 1),
        subject_of={e.episode_id: e.subject_id for e in grouped["train"]}, targets=list(state["targets"]),
        candidates=candidates, firings=firings, external=external, by_id=by_id,
        train_firings={c.program_id: firings[c.program_id] & train_ids for c in candidates},
        k=int(args["k"]), max_surrogates=int(args["max_surrogates"]), shortlist=int(args.get("shortlist") or 0) or None,
        registries=registries, meta=meta)


def write_csv(path: Path, records: list[dict[str, Any]], first: tuple[str, ...] = ("run", "registry", "mode", "target", "a", "b")) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({k for r in records for k in r}, key=lambda c: (c not in first, first.index(c) if c in first else 0, c))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def e3_context():
    """This module, registered in ``sys.modules`` (for scripts that load it by path)."""
    return sys.modules[__name__]
