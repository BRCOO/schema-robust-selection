#!/usr/bin/env python3
"""Build the drive-failure cohorts of the second domain (Backblaze drive stats) in the v2 episode format.

Task rules: ``src/icu/v2/extract/backblaze.py``. Writes, per manufacturer, an episode directory
``<out-root>/backblaze_<manufacturer>_gap<g>d_full/`` (``shock_episode_<id>.npy``, ``_missing_mask.npy``,
``.json`` and ``extraction_manifest.json``, the same layout as the ICU extractions), a drive-level
TRAIN/VAL/TEST split of the source manufacturer (``<task-root>/f_backblaze_<source>_gap<g>d/episodes_split.json``)
and the external episode lists of the other manufacturers (``.../episodes_all.json``). Pass 1 (the
drive scan) is cached in ``--cache``.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/extract_backblaze_cohort.py \\
      --zips "data/raw/backblaze/data_Q*_2024.zip"
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.extract import backblaze as bb  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zips", required=True, help="glob of the Backblaze quarterly archives")
    parser.add_argument("--gap-days", type=int, default=2)
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--source", default="seagate")
    parser.add_argument("--seed", type=int, default=20261102, help="control sampling and the split")
    parser.add_argument("--fractions", default="0.7,0.1,0.2")
    parser.add_argument("--out-root", default="data/processed/v2/extract")
    parser.add_argument("--task-root", default="artifacts/v2/task")
    parser.add_argument("--cache", default="data/processed/v2/cache/backblaze/scan.pkl")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    zips = [Path(p) for p in sorted(glob.glob(str(ROOT / args.zips) if not Path(args.zips).is_absolute() else args.zips))]
    if not zips:
        raise SystemExit(f"no archives match {args.zips}")
    cache = ROOT / args.cache
    key = [p.name for p in zips]
    if cache.exists() and pickle.loads(cache.read_bytes())["zips"] == key:
        days, drives = (lambda c: (c["days"], c["drives"]))(pickle.loads(cache.read_bytes()))
    else:
        days, drives = bb.scan(zips)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps({"zips": key, "days": days, "drives": drives}))
    print(f"[bb] scan: {len(days)} days, {len(drives)} drives ({time.perf_counter() - started:.0f}s)", flush=True)
    rows, counts = bb.select_episodes(days, drives, gap_days=args.gap_days, window_days=args.window_days, horizon_days=args.horizon_days, seed=args.seed)
    print(f"[bb] episodes: {len(rows)}; {counts} ({time.perf_counter() - started:.0f}s)", flush=True)
    values, mask = bb.extract_windows(zips, days, rows)
    schemas = bb.manufacturer_schemas(rows, mask)
    print(f"[bb] windows extracted ({time.perf_counter() - started:.0f}s); schemas: { {k: len(v) for k, v in schemas.items()} }", flush=True)

    names = [n for n, _ in bb.FEATURES]
    drive_id = {serial: i + 1 for i, serial in enumerate(sorted({r["serial"] for r in rows}))}
    tag = f"gap{args.gap_days}d"
    spec = {"task": "drive_failure", "gap_days": args.gap_days, "window_days": args.window_days, "horizon_days": args.horizon_days,
            "seed": args.seed, "features": {n: c for n, c in bb.FEATURES}, "feature_meaning": bb.FEATURE_MEANING,
            "schema_threshold": bb.SCHEMA_THRESHOLD, "archives": key}
    episodes: dict[str, list[dict]] = {}
    for e, row in enumerate(rows):
        vendor = row["manufacturer"]
        if vendor in ("other", "ssd"):
            continue
        out = ROOT / args.out_root / f"backblaze_{vendor}_{tag}_full"
        out.mkdir(parents=True, exist_ok=True)
        anchor = days[row["anchor"]]
        did = drive_id[row["serial"]]
        episode_id = f"bb_{vendor}_{did}_{anchor:%Y%m%d}_{row['role']}"
        stem = f"shock_episode_{episode_id}"
        record = {"episode_id": episode_id, "subject_id": did, "stay_id": did, "label": int(row["role"] == "positive"),
                  "anchor_kind": "failure" if row["role"] == "positive" else "landmark_control", "anchor_time": anchor.date().isoformat(),
                  "window_start": days[row["start"]].date().isoformat(), "window_end": days[row["end"]].date().isoformat()}
        meta = {"dataset": "backblaze-2024", **{k: v for k, v in record.items() if k != "label"}, "episode_type": row["role"],
                "start_time": record["window_start"], "end_time": record["window_end"], "resample_minutes": 1440,
                "shape": [values.shape[1], values.shape[2]], "feature_names": names, "missing_mask_file": f"{stem}_missing_mask.npy",
                "intervention_features": [], "missingness_features": [], "observation_density_features": [],
                "source_available_features": schemas[vendor], "model": row["model"], "manufacturer": vendor,
                "matched_drive_id": drive_id[row["matched_serial"]],
                "values_units": "SMART raw or normalized value per extraction_manifest spec.features; NaN = not reported; not imputed, not scaled"}
        np.save(out / f"{stem}.npy", values[e])
        np.save(out / f"{stem}_missing_mask.npy", mask[e])
        (out / f"{stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        episodes.setdefault(vendor, []).append(record)
    for vendor, records in episodes.items():
        out = ROOT / args.out_root / f"backblaze_{vendor}_{tag}_full"
        n_pos = sum(r["label"] for r in records)
        (out / "extraction_manifest.json").write_text(json.dumps({
            "spec": {**spec, "manufacturer": vendor}, "available_features": schemas[vendor], "feature_names": names,
            "counts": {"episodes": len(records), "positives": n_pos, "controls": len(records) - n_pos, **({"selection": counts} if vendor == args.source else {})},
            "adapter": {"source": "backblaze", "archives": key}}, indent=2), encoding="utf-8")
    rng = np.random.default_rng(args.seed)
    fractions = [float(x) for x in args.fractions.split(",")]
    source = sorted(episodes[args.source], key=lambda r: r["episode_id"])
    order = rng.permutation(len(source))
    cut1, cut2 = int(round(fractions[0] * len(source))), int(round((fractions[0] + fractions[1]) * len(source)))
    for rank, i in enumerate(order):
        source[i]["split"] = "train" if rank < cut1 else ("val" if rank < cut2 else "test")
    task = ROOT / args.task_root / f"f_backblaze_{args.source}_{tag}"
    task.mkdir(parents=True, exist_ok=True)
    (task / "episodes_split.json").write_text(json.dumps(source, indent=1), encoding="utf-8")
    for vendor, records in episodes.items():
        if vendor == args.source:
            continue
        target = ROOT / args.task_root / f"f_backblaze_{vendor}_{tag}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "episodes_all.json").write_text(json.dumps([{**r, "split": "external"} for r in sorted(records, key=lambda r: r["episode_id"])], indent=1), encoding="utf-8")
    summary = {v: {"episodes": len(r), "positives": sum(x["label"] for x in r), "schema": schemas[v]} for v, r in episodes.items()}
    summary[args.source]["splits"] = {s: sum(1 for r in source if r["split"] == s) for s in ("train", "val", "test")}
    print(json.dumps(summary, indent=1), flush=True)
    print(f"[bb] done ({time.perf_counter() - started:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
