#!/usr/bin/env bash
# E2 rerun for gap 4 (filter + unaware baselines), then E1 at K=5 for gaps 1, 2, 4.
cd "$(dirname "$0")/.." || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
PY="${PYTHON:-python}"
$PY scripts/v2/run_schema_aware_selection.py --pools artifacts/v2/comparison/f_mimic_iv_gap4h/pools.pkl \
  --selector-analysis-dir artifacts/v2/selector_analysis/f_mimic_iv_gap4h --out-dir artifacts/v2/schema_aware/f_mimic_iv_gap4h \
  --agnostic-targets eicu,hirid --unaware-registries artifacts/v2/selection_baselines/f_mimic_iv_gap4h/registries.json \
  > artifacts/v2/logs/e2_filter_baselines_gap4h.log 2>&1 || exit 1
bash tmp/v2_helpers/launch_e1_k5.sh > artifacts/v2/logs/e1_k5_driver.log 2>&1 || exit 1
echo "[e2g4+k5] done"
