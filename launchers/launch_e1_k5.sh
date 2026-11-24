#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
PY="${PYTHON:-python}"
for g in 1 2 4; do
  $PY scripts/v2/run_selection_baselines.py --pools artifacts/v2/comparison/f_mimic_iv_gap${g}h/pools.pkl \
    --selector-analysis-dir artifacts/v2/selector_analysis/f_mimic_iv_gap${g}h --out-dir artifacts/v2/selection_baselines_k5/f_mimic_iv_gap${g}h \
    --selector-targets eicu,hirid --k 5 > artifacts/v2/logs/selection_baselines_k5_gap${g}h.log 2>&1 || exit 1
done
echo "[e1-k5] done"
