#!/usr/bin/env python3
"""Add the window-statistic predicate generator (src/icu/v2/baselines/predicates.py) to finished candidate pools.

Reads a comparison folder's pools.pkl and config.json, mines the predicates on the source TRAIN split (the episodes,
splits and targets of the other generators), fires them on every internal episode and every target (controlled
targets fire on internal TEST, as in run_discovery_comparison.add_schema_shift_targets), and writes pools_stat.pkl next
to the original with the new pool appended. The other generators are not re-mined. The pool cap is the SPMF
generators' (``methods.prefixspan.pool_cap``: 200 for the standard pools, 1,000 for the large ones).

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/add_statistic_predicates.py --comparison artifacts/v2/comparison/f_mimic_iv_gap1h
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.baselines.predicates import METHOD, mine_statistic_predicates, predicate_firing_sets, statistic_table  # noqa: E402
from src.icu.v2.pipeline import MethodPool  # noqa: E402
from src.icu.v2.pools import load_arrays  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", required=True, help="comparison folder with pools.pkl and config.json")
    parser.add_argument("--out-name", default="pools_stat.pkl")
    parser.add_argument("--pool-cap", type=int, default=None)
    args = parser.parse_args(argv)
    folder = ROOT / args.comparison if not Path(args.comparison).is_absolute() else Path(args.comparison)
    out = folder / args.out_name
    if out.exists():
        print(f"[stat] {out}: exists, skipped", flush=True)
        return 0
    t0 = time.perf_counter()
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    with open(folder / "pools.pkl", "rb") as handle:
        state = pickle.load(handle)
    if any(p.method == METHOD for p in state["pools"]):
        raise SystemExit(f"{folder}: already has a {METHOD} pool")
    episodes = state["episodes"]
    features = list(cfg["internal_features"])
    arrays = load_arrays(cfg["episode_dir"], [e.episode_id for e in episodes])
    ids, columns, table = statistic_table(arrays, features)
    train_ids = [e.episode_id for e in episodes if e.split == "train"]
    labels = {e.episode_id: int(e.label) for e in episodes}
    cap = int(args.pool_cap or cfg["methods"]["prefixspan"]["pool_cap"])
    candidates = mine_statistic_predicates(train_ids, ids, columns, table, labels, cap)
    firings = predicate_firing_sets(candidates, ids, columns, table)
    print(f"[stat] {folder.name}: {len(candidates)} predicates from {len(train_ids)} TRAIN episodes ({time.perf_counter() - t0:.0f}s)", flush=True)
    controlled = set((cfg.get("schema_shift_targets") or {}).keys())
    test_ids = frozenset(e.episode_id for e in episodes if e.split == "test")
    external = {}
    for target in state["targets"]:
        if target.name in controlled:
            external[target.name] = {pid: fs & test_ids for pid, fs in firings.items()}
            continue
        spec = cfg["targets"][target.name]
        t_arrays = load_arrays(spec["episode_dir"], [e.episode_id for e in target.episodes])
        t_ids, t_columns, t_table = statistic_table(t_arrays, features)
        external[target.name] = predicate_firing_sets(candidates, t_ids, t_columns, t_table)
        print(f"[stat] {folder.name}: fired at {target.name} ({len(t_ids)} episodes, {time.perf_counter() - t0:.0f}s)", flush=True)
    state["pools"].append(MethodPool(method=METHOD, candidates=candidates, firings=firings, external_firings=external))
    if isinstance(state.get("n_candidates"), dict):
        state["n_candidates"][METHOD] = len(candidates)
    with open(out, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {"comparison": str(folder), "out": str(out), "pool_cap": cap, "n_predicates": len(candidates),
               "by_stat": dict(Counter(p.params["stat"] for p in candidates)), "by_feature": dict(Counter(p.params["feature"] for p in candidates)),
               "seconds": time.perf_counter() - t0}
    (folder / "pools_stat_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[stat] wrote {out} ({summary['seconds']:.0f}s): {summary['by_stat']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
