#!/usr/bin/env python3
"""Registries on three SQL engines (review of 2026-11-09, item 1): exactness of the portable dialect in DuckDB, PostgreSQL
and SQLite against the Python reference engine, program by program, as run_sql_execution.py does for the DuckDB dialect.

For each run, every compilable candidate program is compiled in the portable dialect (``sql_registry``) and executed on
the internal TEST episodes and on every real target database, each loaded with its own schema (``sql_engines``); firing
sets must equal the reference engine's (``pools.pkl``). Also timed: loading, indexing, deriving the tokens, all programs.
PostgreSQL is the pip-packaged server of ``pgserver`` (in ``--pg-site``), listening on 127.0.0.1 only, with its data
directory under artifacts/ (deleted at exit).

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_sql_engines.py \\
      [--runs union:f_mimic_iv_gap1h] [--engines duckdb_portable,postgres,sqlite]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIX_POOLS = ("union:f_mimic_iv_gap1h,union:f_mimic_iv_gap2h,union:f_mimic_iv_gap4h,union:f_mimic_iv_aki_gap6h,"
             "union_large_k10:f_mimic_iv_gap1h,union_large_k10:f_mimic_iv_aki_gap6h")


def _context():
    name = "e3_context"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "v2" / "e3_context.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


ctx = _context()

from src.icu.v2 import sql_engines as eng  # noqa: E402
from src.icu.v2.events import fit_feature_bins  # noqa: E402
from src.icu.v2.pools import load_arrays  # noqa: E402
from src.icu.v2.sql_registry import compilable, executable, fire  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=SIX_POOLS)
    parser.add_argument("--engines", default="duckdb_portable,postgres,sqlite")
    parser.add_argument("--sites", default="internal_test,eicu,hirid")
    parser.add_argument("--max-programs", type=int, default=0, help="first N programs only (smoke test)")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--pg-site", default=str(ROOT / "tmp/pg_site"))
    parser.add_argument("--out-root", default=str(ROOT / "artifacts/v2/sql_engines"))
    args = parser.parse_args(argv)
    engines = [e for e in args.engines.split(",") if e]
    pg = None
    pgdata = Path(args.out_root) / "pgdata"
    if "postgres" in engines:
        sys.path.insert(0, args.pg_site)
        import pgserver

        shutil.rmtree(pgdata, ignore_errors=True)
        pgdata.parent.mkdir(parents=True, exist_ok=True)
        pg = pgserver.get_server(str(pgdata), cleanup_mode="delete")
    try:
        for spec in [x for x in args.runs.split(",") if x]:
            root, label = spec.split(":")
            out = Path(args.out_root) / root / label
            run = ctx.load_run(*ctx.run_dirs(root, label))
            config = json.loads((run.pools_path.parent / "config.json").read_text(encoding="utf-8"))
            internal_dir = config["episode_dir"]
            train = load_arrays(internal_dir, run.ids["train"])
            names = next(iter(train.values()))[2]
            bins = fit_feature_bins([a[0] for a in train.values()], names, n_bins=int(config["representation"]["n_bins"]),
                                    missing_masks=[a[1] for a in train.values()])
            del train
            programs = [p for p in run.candidates if compilable(p)]
            if args.max_programs:
                programs = programs[: args.max_programs]
            sites = [("internal_test", internal_dir, run.ids["test"], run.source, {p: run.firings[p] & frozenset(run.ids["test"]) for p in run.firings})]
            for target in run.targets:
                if target.name in config["targets"]:
                    eids = [e.episode_id for e in target.episodes]
                    sites.append((target.name, config["targets"][target.name]["episode_dir"], eids, frozenset(target.features),
                                  {p: f & frozenset(eids) for p, f in run.external[target.name].items()}))
            records = {e: [] for e in engines}
            mismatches = {e: [] for e in engines}
            versions = {}
            for name, episode_dir, eids, schema, reference in sites:
                if name not in args.sites.split(","):
                    continue
                arrays = load_arrays(episode_dir, eids)
                runnable = [p for p in programs if executable(p, schema)]
                for engine in engines:
                    con = eng.connect(engine, threads=args.threads, pg_uri=pg.get_uri() if pg else None)
                    versions[engine] = eng.versions(con, engine)
                    info = eng.load(con, engine, arrays, schema, bins)
                    info.update(eng.tokens(con, engine))
                    t1 = time.perf_counter()
                    fired = fire(con, runnable, schema, dialect=eng.dialect(engine))
                    seconds = time.perf_counter() - t1
                    bad = [p for p in runnable if fired[p.program_id] != reference.get(p.program_id, frozenset())]
                    for p in bad:
                        got, want = fired[p.program_id], reference.get(p.program_id, frozenset())
                        mismatches[engine].append({"site": name, "program_id": p.program_id, "kind": p.kind,
                                                   "sql_only": len(got - want), "python_only": len(want - got)})
                    records[engine].append({"run": f"{root}/{label}", "engine": engine, "site": name, "n_episodes": len(eids),
                                            "n_programs": len(runnable), "n_mismatched_programs": len(bad),
                                            "n_decisions": len(runnable) * len(eids), "seconds_all_programs": seconds, **info})
                    print(f"[engines] {label} {name} {engine}: {len(runnable)} programs on {len(eids)} episodes, mismatched {len(bad)}; "
                          f"load {info['seconds_load']:.1f}s, index {info['seconds_index']:.1f}s, tokens {info['seconds_tokens']:.1f}s, "
                          f"programs {seconds:.1f}s", flush=True)
                    for table in ("tokens", "edges", "obs"):
                        con.execute(f"DROP TABLE IF EXISTS {table}")
                    con.close()
            for engine in engines:
                ctx.write_csv(out / f"exactness_{engine}.csv", records[engine], first=("run", "engine", "site"))
                ctx.write_csv(out / f"mismatches_{engine}.csv", mismatches[engine] or [{"site": "-", "program_id": "-"}], first=("site", "program_id"))
            (out / "manifest.json").write_text(json.dumps({"args": vars(args), "versions": versions}, indent=2), encoding="utf-8")
    finally:
        if pg is not None:
            pg.cleanup()
            shutil.rmtree(pgdata, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
