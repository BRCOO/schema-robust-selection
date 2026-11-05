#!/bin/bash
# Review-driven add-on runs for the v3 experiments (2026-11-05). Sequential; each step is resumable.
cd "/d/ALL PROJECT/Non-stationary times series"
export CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PY=/d/BRCoo/miniconda/python
$PY scripts/v2/run_reference_registries.py --phase refs > artifacts/v2/logs/v3_refs.log 2>&1
$PY scripts/v2/run_reference_registries.py --phase seeds > artifacts/v2/logs/v3_seeds.log 2>&1
for l in f_mimic_iv_gap1h f_mimic_iv_gap2h f_mimic_iv_gap4h f_mimic_iv_aki_gap6h; do
  if [ ! -f artifacts/v2/proxy_correlation/heldout11_$l.csv ]; then
    $PY scripts/v2/analyze_selection_proxies_heldout.py --pools artifacts/v2/comparison/$l/pools.pkl --out artifacts/v2/proxy_correlation/heldout11_$l.csv --steps 11 > artifacts/v2/logs/v3_proxy11_$l.log 2>&1
  fi
done
echo done > artifacts/v2/logs/v3_review_runs.done
