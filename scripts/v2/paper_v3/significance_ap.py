"""Multiple-comparison control for the SRS-AP comparisons (paper v3, RQ1), the same rule as
scripts/v2/summarize_significance.py: two-sided percentile p-values from each paired bootstrap, Holm within one run's tests of
one reference (all opponents, both metrics), Benjamini-Hochberg over all tests of one reference; alpha = 0.05.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/significance_ap.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("sig", ROOT / "scripts/v2/summarize_significance.py")
sig = importlib.util.module_from_spec(spec)
sys.modules["sig"] = sig
spec.loader.exec_module(sig)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

OPPONENTS = ("nmifs", "mrmr", "mrmr_stratified", "jmi", "cmim", "mmrfs", "mmr_ig", "mms_xin_ig", "dpp_ig", "l1_logistic", "l1_dropout",
             "mrmr_filter_known")
N_BOOT = 1000
rows = []
for ref in ("srs_ap_t", "srs_ap"):
    p = data.paired_all()
    p = p[(p.a == ref) & (p.b.isin(OPPONENTS)) & (p.metric.isin(["internal", "mean_target"]))].copy()
    if p.empty:
        continue
    p["p"] = [sig.p_values({"frac_le_zero": r.frac_le_zero, "high": r.high, "low": r.low, "point_diff": r.point_diff}, N_BOOT)[0]
              for r in p.itertuples()]
    p["holm"] = np.nan
    for _, g in p.groupby("name"):
        p.loc[g.index, "holm"] = sig.holm(list(g.p))
    p["bh"] = sig.bh(list(p.p))
    for (b, metric), g in p.groupby(["b", "metric"]):
        def tally(signif):
            better = int(((g.point_diff > 0) & signif).sum())
            worse = int(((g.point_diff < 0) & signif).sum())
            return f"{better}/{len(g) - better - worse}/{worse}"
        rows.append({"reference": ref, "opponent": b, "metric": metric, "n_runs": len(g),
                     "ci": tally((g.low > 0) | (g.high < 0)), "holm": tally(g.holm < 0.05), "bh": tally(g.bh < 0.05)})
out = pd.DataFrame(rows)
path = ROOT / "artifacts/v2/paper_v3_cache/significance_ap.csv"
out.to_csv(path, index=False)
print(out.to_string())
print("wrote", path)
