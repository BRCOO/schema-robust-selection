#!/usr/bin/env python3
"""Can a target see, without labels, that a surrogate will not stand in for its primary? (review of 2026-11-09, item 2;
post hoc and exploratory)

For every ICU run, the six registries of the deployment-mode analysis (Figure 8c) and every target with lost features,
each coordinate that resolves to a surrogate b (a branch other than its primary p) is scored twice with the frozen probe
of run_deployment_modes.py: as resolved, and with that one coordinate zeroed (all others as resolved). Its effect is the
AUPRC difference (x100; label-based, the outcome to predict). Label-free diagnostics, from the source TRAIN firings that
ship with the registry and the target's own unlabeled firings:

* ``rate_shift``: |logit r_T(b) - logit r_S(b)|, the shift of the substitute's firing rate;
* ``cofire_shift``: the mean, over the other coordinates that run at the target, of |phi_T(b, c) - phi_S(b, c)| (phi: the
  correlation of two firing vectors), i.e. how much the substitute's relation to the rest of the registry moved;
* ``chain_shift``: the same over the other executable members of p's chain, the substitutes that should agree with b;
* ``base_cofire_shift``: cofire_shift of the coordinates that run as selected, the target's general shift.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_chain_diagnostic.py

Output: artifacts/v2/chain_diagnostic/coordinates.csv (one row per substituted coordinate) and registries.csv.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


ctx = _module("e3_context", ROOT / "scripts/v2/e3_context.py")
dm = _module("run_deployment_modes", ROOT / "scripts/v2/run_deployment_modes.py")

from src.icu.v2.stats import metric_value  # noqa: E402

REGISTRIES = ("srs_ap_t", "srs_ap", "nmifs", "mrmr", "l1_dropout", "fs_cv")
OUT = ROOT / "artifacts/v2/chain_diagnostic"


def logit(r: float, eps: float = 1e-3) -> float:
    r = min(max(r, eps), 1 - eps)
    return float(np.log(r / (1 - r)))


def phi(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def shift(xs: np.ndarray, xt: np.ndarray, Ys: np.ndarray, Yt: np.ndarray) -> float | None:
    """Mean |phi_T - phi_S| between one column and the columns of Y (skipping undefined correlations)."""
    d = []
    for j in range(Ys.shape[1]):
        ps, pt = phi(xs, Ys[:, j]), phi(xt, Yt[:, j])
        if ps is not None and pt is not None:
            d.append(abs(pt - ps))
    return float(np.mean(d)) if d else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="")
    args = parser.parse_args(argv)
    runs = [tuple(x.split(":")) for x in args.runs.split(",") if x] or list(ctx.ICU_RUNS)
    coords, regs = [], []
    for root, label in runs:
        run = ctx.load_run(*ctx.run_dirs(root, label))
        task = "AKI" if "aki" in label else "vaso"
        train = run.ids["train"]
        base_rate = float(np.mean(run.labels["train"]))
        for name in REGISTRIES:
            if name not in run.registries:
                continue
            comp = run.registries[name]
            prim = list(comp.primaries)
            frozen = dm.probe(run, {s: run.matrix(prim, run.ids[s]) for s in ("train", "val")})
            beta = frozen.coef_[0] if frozen is not None else np.zeros(len(prim))
            Xp_tr = run.matrix(prim, train)
            for target in run.targets:
                schema = frozenset(target.features)
                resolved = comp.resolve(schema)
                subs = [k for k, b in enumerate(resolved) if b is not None and b.program_id != prim[k].program_id]
                if not subs:
                    continue
                eids = [e.episode_id for e in target.episodes]
                y = np.array([e.label for e in target.episodes])
                ext = run.external[target.name]
                X_t = run.matrix(resolved, eids, ext)
                X_s = run.matrix(resolved, train)
                ap_full = metric_value(y, dm.predict(frozen, X_t, base_rate), "auprc")
                running = [k for k, b in enumerate(resolved) if b is not None]
                selected = [k for k in running if k not in subs]
                base = [shift(X_s[:, k], X_t[:, k], X_s[:, [j for j in running if j != k]], X_t[:, [j for j in running if j != k]])
                        for k in selected]
                base = [v for v in base if v is not None]
                X_zero = X_t.copy()
                X_zero[:, subs] = 0.0
                regs.append({"run": f"{root}/{label}", "task": task, "registry": name, "target": target.name, "n_substituted": len(subs),
                             "auprc_resolved": ap_full, "auprc_zeroed": metric_value(y, dm.predict(frozen, X_zero, base_rate), "auprc"),
                             "base_cofire_shift": float(np.mean(base)) if base else None})
                for k in subs:
                    b = resolved[k]
                    X_k0 = X_t.copy()
                    X_k0[:, k] = 0.0
                    effect = 100 * (ap_full - metric_value(y, dm.predict(frozen, X_k0, base_rate), "auprc"))
                    others = [j for j in running if j != k]
                    members = [q for q in comp.branches(prim[k]) if q.program_id != b.program_id and frozenset(q.features) <= schema]
                    chain = None
                    if members:
                        chain = shift(X_s[:, k], X_t[:, k], run.matrix(members, train), run.matrix(members, eids, ext))
                    pi = float(Xp_tr[:, k].mean())
                    minor = min(pi, 1 - pi)
                    lam = (minor - float(np.mean(Xp_tr[:, k] != X_s[:, k]))) / minor if minor > 0 else None
                    coords.append({"run": f"{root}/{label}", "task": task, "registry": name, "target": target.name, "k": k,
                                   "primary": prim[k].program_id, "branch": b.program_id, "branch_kind": b.kind,
                                   "lambda_source": lam, "beta": float(beta[k]), "rate_primary_source": pi,
                                   "rate_branch_source": float(X_s[:, k].mean()), "rate_branch_target": float(X_t[:, k].mean()),
                                   "rate_shift": abs(logit(float(X_t[:, k].mean())) - logit(float(X_s[:, k].mean()))),
                                   "cofire_shift": shift(X_s[:, k], X_t[:, k], X_s[:, others], X_t[:, others]) if others else None,
                                   "chain_shift": chain, "base_cofire_shift": regs[-1]["base_cofire_shift"], "effect_auprc_x100": effect})
        print(f"[chain] {root}/{label}: {sum(1 for c in coords if c['run'] == f'{root}/{label}')} substituted coordinates", flush=True)
        ctx.write_csv(OUT / "coordinates.csv", coords, first=("run", "task", "registry", "target", "k"))
        ctx.write_csv(OUT / "registries.csv", regs, first=("run", "task", "registry", "target"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
