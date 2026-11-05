"""Mining contract: shared threshold and budget configuration for mining baselines.

Reads only from icde_mining_contract.json to ensure all discovery methods
use identical time handling, search budget, and threshold grids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MiningContract:
    """Frozen mining contract derived from icde_mining_contract.json."""
    time_bin: str
    top_k: int
    candidate_budget: int
    support_grid: list[float]
    confidence_grid: list[float]
    motif_distance_quantile: list[float]
    shapelet_length_grid: list[int]


def load_mining_contract(path: str | Path) -> MiningContract:
    """Load and validate the mining contract from a JSON manifest."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    return MiningContract(
        time_bin=str(payload.get("time_discretization", "relative_1h_bin")),
        top_k=int(payload.get("top_k", 10)),
        candidate_budget=int(payload.get("candidate_budget_cap", 2000)),
        support_grid=list(payload.get("support_grid", [0.02, 0.05, 0.08])),
        confidence_grid=list(payload.get("confidence_grid", [0.5, 0.6, 0.7])),
        motif_distance_quantile=list(payload.get("motif_distance_quantile", [0.1, 0.2])),
        shapelet_length_grid=[int(v) for v in payload.get("shapelet_length_grid", [3, 4, 5])],
    )
