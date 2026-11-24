#!/usr/bin/env python3
"""Reviewer item G: registries compiled to SQL and executed inside a database (DuckDB).

For a finished E3 run, every compilable candidate program (all but FMG's legacy rules) is compiled
to SQL (``src/icu/v2/sql_registry.py``) and executed on the internal TEST episodes and on every
real target database, each loaded as the long table ``obs`` with its own schema. The firing sets
are compared, program by program, with the Python engine's (``pools.pkl``); they must be equal.
Timings: deriving the event tokens in SQL, all programs, and the shipped SRS+T registry (each
coordinate resolved against the target schema, i.e. the queries a site would run). A scale-up
replicates the eICU episodes ``--scales`` times and times tokens plus the registry.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_sql_execution.py [--runs union:f_mimic_iv_gap1h]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _context():
    name = "e3_context"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "v2" / "e3_context.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


ctx = _context()

from src.icu.v2.events import fit_feature_bins  # noqa: E402
from src.icu.v2.pools import load_arrays  # noqa: E402
from src.icu.v2.sql_registry import compilable, edges_rows, executable, fire, obs_rows, token_sql  # noqa: E402


def connect(threads: int):
    con = duckdb.connect()
    con.execute(f"SET threads TO {threads}")
    return con


def load_obs(con, arrays, schema, bins, replicate: int = 1) -> dict:
    t0 = time.perf_counter()
    ep, tt, ff, vv = obs_rows(arrays, schema)
    frame = pd.DataFrame({"episode_id": ep, "t": np.asarray(tt, dtype=np.int32), "feature": ff, "value": np.asarray(vv, dtype=float)})
    con.register("obs_df", frame)
    con.execute("CREATE OR REPLACE TABLE obs0 AS SELECT episode_id, t, feature, CASE WHEN isnan(value) THEN NULL ELSE value END AS value FROM obs_df")
    con.unregister("obs_df")
    if replicate > 1:
        con.execute(f"CREATE OR REPLACE TABLE obs AS SELECT o.episode_id || '#' || CAST(r.i AS VARCHAR) AS episode_id, o.t, o.feature, o.value "
                    f"FROM obs0 o CROSS JOIN range({replicate}) r(i)")
    else:
        con.execute("CREATE OR REPLACE TABLE obs AS SELECT * FROM obs0")
    con.execute("DROP TABLE obs0")
    con.execute("CREATE OR REPLACE TABLE edges (feature VARCHAR, c1 DOUBLE, c2 DOUBLE, c3 DOUBLE)")
    rows = edges_rows(bins)
    if rows:
        con.executemany("INSERT INTO edges VALUES (?, ?, ?, ?)", rows)
    load = time.perf_counter() - t0
    t1 = time.perf_counter()
    con.execute(token_sql())
    tokens = time.perf_counter() - t1
    n_obs = con.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    n_tok = con.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]
    return {"seconds_load": load, "seconds_tokens": tokens, "n_obs_rows": int(n_obs), "n_tokens": int(n_tok)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="union:f_mimic_iv_gap1h,union:f_mimic_iv_aki_gap6h")
    parser.add_argument("--registry", default="srs_c_val")
    parser.add_argument("--scales", default="1,10,30")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--out-root", default=str(ROOT / "artifacts/v2/sql_execution"))
    args = parser.parse_args(argv)
    for spec in [x for x in args.runs.split(",") if x]:
        root, label = spec.split(":")
        out = Path(args.out_root) / root / label
        run = ctx.load_run(*ctx.run_dirs(root, label))
        config = json.loads((run.pools_path.parent / "config.json").read_text(encoding="utf-8"))
        internal_dir = config["episode_dir"]
        t0 = time.perf_counter()
        train = load_arrays(internal_dir, run.ids["train"])
        names = next(iter(train.values()))[2]
        bins = fit_feature_bins([a[0] for a in train.values()], names, n_bins=int(config["representation"]["n_bins"]),
                                missing_masks=[a[1] for a in train.values()])
        print(f"[sql] {label}: bins refit on {len(train)} TRAIN episodes ({time.perf_counter() - t0:.0f}s)", flush=True)
        del train
        programs = [p for p in run.candidates if compilable(p)]
        comp = run.registries[args.registry]
        sites = [("internal_test", internal_dir, run.ids["test"], run.source, {p: run.firings[p] & frozenset(run.ids["test"]) for p in run.firings})]
        for target in run.targets:
            if target.name in config["targets"]:
                eids = [e.episode_id for e in target.episodes]
                sites.append((target.name, config["targets"][target.name]["episode_dir"], eids, frozenset(target.features),
                              {p: f & frozenset(eids) for p, f in run.external[target.name].items()}))
        records, mismatches, scale_rows = [], [], []
        for name, episode_dir, eids, schema, reference in sites:
            arrays = load_arrays(episode_dir, eids)
            con = connect(args.threads)
            info = load_obs(con, arrays, schema, bins)
            runnable = [p for p in programs if executable(p, schema)]
            t1 = time.perf_counter()
            fired = fire(con, runnable, schema)
            all_seconds = time.perf_counter() - t1
            bad = [p for p in runnable if fired[p.program_id] != reference.get(p.program_id, frozenset())]
            for p in bad:
                got, want = fired[p.program_id], reference.get(p.program_id, frozenset())
                mismatches.append({"site": name, "program_id": p.program_id, "kind": p.kind, "sql_only": len(got - want), "python_only": len(want - got)})
            branches = [b for b in comp.resolve(schema) if b is not None]
            resolved = [b for b in branches if compilable(b)]
            t2 = time.perf_counter()
            fire(con, resolved, schema)
            registry_seconds = time.perf_counter() - t2
            records.append({"run": f"{root}/{label}", "site": name, "n_episodes": len(eids), "n_programs": len(runnable),
                            "n_mismatched_programs": len(bad), "seconds_all_programs": all_seconds,
                            "registry_programs": len(resolved), "registry_not_compiled": len(branches) - len(resolved),
                            "seconds_registry": registry_seconds, **info})
            print(f"[sql] {label} {name}: {len(runnable)} programs on {len(eids)} episodes, mismatched {len(bad)}; "
                  f"tokens {info['seconds_tokens']:.2f}s, all programs {all_seconds:.1f}s, registry {registry_seconds:.2f}s", flush=True)
            con.close()
            if name == "eicu":
                for r in [int(x) for x in args.scales.split(",") if x]:
                    con = connect(args.threads)
                    info_r = load_obs(con, arrays, schema, bins, replicate=r)
                    t3 = time.perf_counter()
                    fire(con, resolved, schema)
                    scale_rows.append({"run": f"{root}/{label}", "site": name, "replicate": r, "n_episodes": len(eids) * r,
                                       "registry_programs": len(resolved), "seconds_registry": time.perf_counter() - t3, **info_r})
                    print(f"[sql] {label} eicu x{r}: {len(eids) * r} episodes, tokens {info_r['seconds_tokens']:.1f}s, "
                          f"registry {scale_rows[-1]['seconds_registry']:.1f}s", flush=True)
                    con.close()
        ctx.write_csv(out / "exactness.csv", records, first=("run", "site"))
        ctx.write_csv(out / "mismatches.csv", mismatches or [{"site": "-", "program_id": "-"}], first=("site", "program_id"))
        ctx.write_csv(out / "scale.csv", scale_rows, first=("run", "site", "replicate"))
        (out / "manifest.json").write_text(json.dumps({"args": vars(args), "duckdb": duckdb.__version__}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
