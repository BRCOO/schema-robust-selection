#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
PY="${PYTHON:-python}"
for g in 1 2 4; do
  $PY scripts/v2/run_schema_aware_selection.py --pools artifacts/v2/comparison/f_mimic_iv_gap${g}h/pools.pkl \
    --selector-analysis-dir artifacts/v2/selector_analysis/f_mimic_iv_gap${g}h --out-dir artifacts/v2/schema_aware/f_mimic_iv_gap${g}h \
    --agnostic-targets eicu,hirid --unaware-registries artifacts/v2/selection_baselines/f_mimic_iv_gap${g}h/registries.json || exit 1
done
echo "[e2-unaware] done"
