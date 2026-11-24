"""Supervised references at the source TEST split and at the real targets (paper v3, RQ1 reference block): AUPRC of the
supervised models of artifacts/v2/supervised (trained on full windows of MIMIC-IV TRAIN), mean over their seeds.

  PYTHONDONTWRITEBYTECODE=1 python scripts/v2/paper_v3/supervised_refs.py
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
rows = []
for label in ("f_mimic_iv_gap1h", "f_mimic_iv_gap2h", "f_mimic_iv_gap4h", "f_mimic_iv_aki_gap6h"):
    base = ROOT / "artifacts/v2/supervised" / label
    for name, cohort_col in (("supervised_v2_test_predictions.csv", None), ("supervised_v2_external_predictions.csv", "cohort")):
        path = base / name
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if cohort_col is None or cohort_col not in d.columns:
            d["cohort"] = "internal_test"
        for (model, seed, cohort), g in d.groupby(["model", "seed", "cohort"]):
            rows.append({"label": label, "model": model, "seed": seed, "cohort": cohort, "n": len(g),
                         "auprc": average_precision_score(g.label, g.prob)})
out = pd.DataFrame(rows)
summary = out.groupby(["label", "model", "cohort"]).auprc.mean().unstack("cohort").round(3)
print(summary.to_string())
dest = ROOT / "artifacts/v2/paper_v3_cache/supervised_refs.csv"
out.to_csv(dest, index=False)
print("wrote", dest)
