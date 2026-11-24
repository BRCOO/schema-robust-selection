#!/usr/bin/env bash
# launch E1 (selection baselines) and E2 (schema-aware) for one gap once its selector analysis is done
g=$1
cd "$(dirname "$0")/.."
while ! grep -q "registry:" artifacts/v2/logs/selector_analysis_gap${g}h_sv.log 2>/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2
"${PYTHON:-python}" scripts/v2/run_selection_baselines.py --pools artifacts/v2/comparison/f_mimic_iv_gap${g}h/pools.pkl --selector-analysis-dir artifacts/v2/selector_analysis/f_mimic_iv_gap${g}h --out-dir artifacts/v2/selection_baselines/f_mimic_iv_gap${g}h --selector-targets eicu,hirid > artifacts/v2/logs/selection_baselines_gap${g}h.log 2>&1 &
"${PYTHON:-python}" scripts/v2/run_schema_aware_selection.py --pools artifacts/v2/comparison/f_mimic_iv_gap${g}h/pools.pkl --selector-analysis-dir artifacts/v2/selector_analysis/f_mimic_iv_gap${g}h --out-dir artifacts/v2/schema_aware/f_mimic_iv_gap${g}h --agnostic-targets eicu,hirid > artifacts/v2/logs/schema_aware_gap${g}h.log 2>&1 &
wait
echo "gap$g E1/E2 done"
