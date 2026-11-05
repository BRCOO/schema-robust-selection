"""Cache the TRAIN firing rate and generator of every candidate in the two standard union pools (vasopressor 1 h, AKI),
for the mechanism figure. Output: paper/v3_assets/source_data/firing_rates.csv."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("e3_context", ROOT / "scripts/v2/e3_context.py")
ctx = importlib.util.module_from_spec(spec)
sys.modules["e3_context"] = ctx
spec.loader.exec_module(ctx)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import generator_of  # noqa: E402

rows = []
for label, task in (("f_mimic_iv_gap1h", "Vaso 1h"), ("f_mimic_iv_aki_gap6h", "AKI")):
    run = ctx.load_run(*ctx.run_dirs("union", label))
    n = len(run.train_ids)
    for c in run.candidates:
        rows.append({"task": task, "program_id": c.program_id, "generator": generator_of(c.program_id), "kind": c.kind,
                     "train_rate": len(run.train_firings[c.program_id]) / n})
    print(task, len(run.candidates), "candidates", flush=True)
out = ROOT / "paper/v3_assets/source_data/firing_rates.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
