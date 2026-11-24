"""Executability of the drive registries at each manufacturer (Figure X7): how many of the K programs run as selected
(primary), how many run through a surrogate, how many are zeroed. Registries are read, never re-selected; NMIFS is
recomputed with the deterministic selector of the paper (selection_baselines.nmifs) because the drive runs did not save it.

  PYTHONDONTWRITEBYTECODE=1 python scripts/v2/paper_v3/drive_executability.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("e3_context", ROOT / "scripts/v2/e3_context.py")
ctx = importlib.util.module_from_spec(spec)
sys.modules["e3_context"] = ctx
spec.loader.exec_module(ctx)

from src.icu.v2.selection_baselines import nmifs  # noqa: E402

LABEL = "f_backblaze_seagate_gap2d"
SHOW = ("srs_ap_t", "srs_ap", "nmifs_ap_t", "nmifs", "mrmr", "l1_dropout", "fs_cv", "l1_logistic", "jmi", "cmim", "mmrfs")
rows = []
for k in (5, 10, 20):
    run = ctx.load_run(*ctx.drive_dirs(k, LABEL))
    regs = dict(run.registries)
    hyb = json.loads((ROOT / f"artifacts/v2/nmifs_ap/drive_k{k}/{LABEL}/registries.json").read_text(encoding="utf-8"))
    regs["nmifs_ap_t"] = ctx._composite(hyb["nmifs_ap_t"], run.by_id)
    regs["nmifs"] = run.composite(nmifs(run.inp, run.k))
    for name in SHOW:
        if name not in regs:
            print("missing", k, name)
            continue
        comp = regs[name]
        for t in run.targets:
            schema = frozenset(t.features)
            primary = sum(frozenset(p.features) <= schema for p in comp.primaries)
            resolved = sum(b is not None for b in comp.resolve(schema))
            rows.append({"k": k, "registry": name, "target": t.name, "n_programs": len(comp.primaries), "primary": primary,
                         "surrogate": resolved - primary, "zeroed": len(comp.primaries) - resolved})
    print("done", k, flush=True)
out = ROOT / "artifacts/v2/paper_v3_cache/drive_executability.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(pd.DataFrame(rows)[lambda d: d.k == 10].pivot_table(index="registry", columns="target", values=["primary", "surrogate"]))
print("wrote", out)
