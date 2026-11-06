#!/usr/bin/env python3
"""Case study of one registry of an E3 run, program by program (a paper table).

For ``--registry`` (default ``srs_c_val``) of the E3 run in ``--e3-dir``: generator, kind and a
readable form of every program, its required features sigma_f, TRAIN firing rate, TRAIN
precision when it fires (and lift over the base rate), the frozen probe's coefficient, the
surrogate chain with the CART association lambda of every surrogate (TRAIN firings), and for
every target the branch each coordinate executes (primary, surrogate i, or zeroed).

Token glossary (``events.py``): ``f:q<l>`` observed value in global TRAIN quantile bin l
(q0 lowest); ``f:on`` intervention active in the bin; ``f:missing`` not measured;
``f:sparse_measurement`` trailing window only sparsely measured.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/export_case_study.py \\
      --e3-dir artifacts/v2/union/f_mimic_iv_gap1h --out-dir artifacts/v2/case_study/gap1h_k10
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.bridge import fit_frozen_bridge_v2  # noqa: E402
from src.icu.v2.pipeline import episodes_by_split, restrict_firings  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry  # noqa: E402
from src.icu.v2.union import union_pool  # noqa: E402


def readable(p) -> str:
    items = lambda s: "{" + ", ".join(sorted(s)) + "}"
    if p.kind == "sequence":
        return " -> ".join(items(s) for s in p.elements)
    if p.kind in ("unordered_rule", "ordered_rule"):
        return " -> ".join(items(s) for s in p.elements) + " => " + " -> ".join(items(s) for s in p.consequent)
    raw = ", ".join(f"{v:g}" for v in p.params.get("raw_template", ()))
    text = f"{p.kind}({p.params.get('feature')}, L={p.params.get('length')}, raw template [{raw}]"
    if p.kind == "shapelet":
        text += f", distance {'<=' if p.params.get('direction') == 'le' else '>'} {float(p.params.get('threshold', 0.0)):.2f}"
    return text + ")"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e3-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--registry", default="srs_c_val")
    parser.add_argument("--registries-file", default="", help="registries.json holding --registry (default: the E3 run's)")
    args = parser.parse_args(argv)
    e3 = Path(args.e3_dir)
    manifest = json.loads((e3 / "manifest.json").read_text(encoding="utf-8"))
    with open(ROOT / manifest["args"]["pools"], "rb") as handle:
        state = pickle.load(handle)
    grouped = episodes_by_split(state["episodes"])
    ids = {name: [e.episode_id for e in rows] for name, rows in grouped.items()}
    labels = {name: [e.label for e in rows] for name, rows in grouped.items()}
    train = frozenset(ids["train"])
    positives = frozenset(e.episode_id for e in grouped["train"] if e.label == 1)
    _, firings, _ = union_pool(state)
    by_id = {c.program_id: c for pool in state["pools"] for c in pool.candidates}
    saved = json.loads(Path(args.registries_file or e3 / "registries.json").read_text(encoding="utf-8"))[args.registry]
    comp = CompositeRegistry(tuple(by_id[i] for i in saved["primaries"]),
                             {pid: tuple(by_id[q] for q in qs) for pid, qs in saved["surrogates"].items()})
    f = {p.program_id: firings[p.program_id] for p in comp.primaries}
    bridge = fit_frozen_bridge_v2(list(comp.primaries), {s: restrict_firings(f, frozenset(ids[s])) for s in ids}, ids, labels)
    coef = dict(zip(bridge["model_bundle"]["coordinate_names"], bridge["model_bundle"]["coef"]))
    n, base_rate = len(train), len(positives) / len(train)
    resolved = {t.name: comp.resolve(t.features) for t in state["targets"]}

    def association(p, q) -> float:
        fp, fq = firings[p.program_id] & train, firings[q.program_id] & train
        rate = len(fp) / n
        blind = min(rate, 1 - rate)
        return (blind - len(fp ^ fq) / n) / blind if blind > 0 else float("nan")

    rows = []
    for position, p in enumerate(comp.primaries, start=1):
        fired = firings[p.program_id] & train
        precision = len(fired & positives) / len(fired) if fired else float("nan")
        chain = comp.branches(p)
        row = {"position": position, "program_id": p.program_id, "generator": p.method, "kind": p.kind, "program": readable(p),
               "features": ", ".join(sorted(p.features)), "train_firing_rate": len(fired) / n, "train_precision": precision,
               "lift": precision / base_rate, "probe_coef": coef[p.program_id],
               "surrogates": " | ".join(f"{q.program_id} [{', '.join(sorted(q.features))}] lambda={association(p, q):.2f}" for q in chain[1:])}
        for target, branches in resolved.items():
            b = branches[position - 1]
            row[f"branch_{target}"] = "zeroed" if b is None else ("primary" if b.program_id == p.program_id else f"surrogate {chain.index(b)}")
        rows.append(row)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "case_study.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    targets = list(resolved)
    lines = [f"# Case study: `{args.registry}` of `{e3.as_posix()}` (K = {len(rows)}; TRAIN base rate {base_rate:.3f})", "",
             "| # | generator | program | features | fires | precision (lift) | beta | " + " | ".join(targets) + " |",
             "|---|---|---|---|---|---|---|" + "---|" * len(targets)]
    for r in rows:
        lines.append(f"| {r['position']} | {r['generator']} | `{r['program']}` | {r['features']} | {r['train_firing_rate']:.3f} | "
                     f"{r['train_precision']:.2f} ({r['lift']:.2f}) | {r['probe_coef']:+.2f} | " + " | ".join(r[f"branch_{t}"] for t in targets) + " |")
    lines += ["", "Surrogate chains:", ""]
    for r in rows:
        lines.append(f"- {r['position']}. `{r['program_id']}`: {r['surrogates'] or '(none)'}")
    (out / "case_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
