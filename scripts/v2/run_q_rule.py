#!/usr/bin/env python3
"""Reviewer item C: choosing the dropout rate q by a masked validation objective.

SRS picks q in {0, 0.25} by the frozen probe's AUPRC on the clean VAL split (pre-specified). The
reviewer's concern: that criterion cannot see schema loss. Alternatives, both computed on VAL with
the frozen probe (TRAIN fit, C on VAL) and every coordinate resolved through its surrogate chain
under each scenario of the q = 0.25 scenario distribution (source, known targets for SRS+T, and the
20 random group-dropout schemas; equal block weights as in selection):

* ``masked_auprc``: the scenario-weighted VAL AUPRC;
* ``masked_ll``: the scenario-weighted VAL log-likelihood.

The candidate registries are the saved ones (q = 0: ``fs_cv`` / ``srs_c``; q = 0.25:
``srs_c_unknown25`` / ``srs_c_robust25``), so nothing is re-selected; the outcome of each rule is
the chosen registry's internal and mean target AUPRC as reported (composite deployment).

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_q_rule.py [--drive-k 5,10,20]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

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

from src.icu.v2.robust_selection import scenario_set  # noqa: E402
from src.icu.v2.stats import metric_value  # noqa: E402

#: method -> (q = 0 registry, q = 0.25 registry, registry reported as the method, uses known targets)
METHODS = {"SRS": ("fs_cv", "srs_c_unknown25", "srs_c_unknown_val", False), "SRS+T": ("srs_c", "srs_c_robust25", "srs_c_val", True)}
EPS = 1e-6


def outcomes(e3_dir: Path, extra_dir: Path) -> dict[str, dict[str, float]]:
    out = {}
    for d in (e3_dir, extra_dir):
        for r in csv.DictReader(open(d / "rows.csv", encoding="utf-8")):
            if r["variant"] == "composite":
                out[r["registry"]] = {"internal": float(r["internal_auprc"]), "target": float(r["mean_target_auprc"])}
    return out


def masked_scores(run, comp, scenarios) -> tuple[float, float]:
    prim = list(comp.primaries)
    model = run.fit(prim)["model_bundle"]["model"]
    val = run.ids["val"]
    y = np.asarray(run.labels["val"])
    auprc = ll = 0.0
    for s in scenarios:
        p = model.predict_proba(run.matrix(comp.resolve(s.features), val))[:, 1]
        pc = np.clip(p, EPS, 1 - EPS)
        auprc += s.weight * metric_value(y, p, "auprc")
        ll += s.weight * float(np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    return auprc, ll


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="")
    parser.add_argument("--drive-k", default="5,10,20")
    parser.add_argument("--out", default=str(ROOT / "artifacts/v2/q_rule/q_rule.csv"))
    args = parser.parse_args(argv)
    runs = [tuple(x.split(":")) for x in args.runs.split(",") if x] or list(ctx.ICU_RUNS)
    jobs = [(f"{root}/{label}", *ctx.run_dirs(root, label)) for root, label in runs]
    jobs += [(f"drive_k{k}", *ctx.drive_dirs(int(k))) for k in args.drive_k.split(",") if k]
    records = []
    for label, e3_dir, extra_dir in jobs:
        run = ctx.load_run(e3_dir, extra_dir)
        res = outcomes(e3_dir, extra_dir)
        known = {t.name: t.features for t in run.targets if t.name in json.loads((e3_dir / "manifest.json").read_text(encoding="utf-8"))["args"]["known_targets"].split(",")}
        for method, (r0, r25, reported, uses_known) in METHODS.items():
            scen = scenario_set(run.source, known if uses_known else {}, dropout_q=0.25)
            scores = {}
            for q, name in ((0.0, r0), (0.25, r25)):
                comp = run.registries[name]
                val_auprc = run.fit(list(comp.primaries))["validation_score"]
                m_auprc, m_ll = masked_scores(run, comp, scen)
                scores[q] = {"val_auprc": val_auprc, "masked_auprc": m_auprc, "masked_ll": m_ll, "registry": name}
            for rule in ("val_auprc", "masked_auprc", "masked_ll"):
                q = 0.25 if scores[0.25][rule] > scores[0.0][rule] + 1e-12 else 0.0
                chosen = scores[q]["registry"]
                records.append({"run": label, "method": method, "rule": rule, "chosen_q": q, "registry": chosen,
                                "internal_auprc": res[chosen]["internal"], "mean_target_auprc": res[chosen]["target"],
                                "score_q0": scores[0.0][rule], "score_q25": scores[0.25][rule],
                                "reported_matches": (res[chosen] == res[reported]) if rule == "val_auprc" else ""})
            line = " ".join(f"{r['rule']}:q={r['chosen_q']:g}/{r['mean_target_auprc']:.3f}" for r in records[-3:])
            print(f"[q-rule] {label} {method}: {line}", flush=True)
    ctx.write_csv(Path(args.out), records, first=("run", "method", "rule"))
    print(f"[q-rule] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
