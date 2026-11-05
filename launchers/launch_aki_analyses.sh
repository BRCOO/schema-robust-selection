#!/usr/bin/env bash
# AKI task (tag aki_gap6h): selector analysis -> E1 (K=10) -> E2 (+ unaware baselines) -> SRS ablation -> registry pairs.
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
PY=/d/BRCoo/miniconda/python
T=aki_gap6h
C=artifacts/v2/comparison/f_mimic_iv_$T
while [ ! -f $C/comparison_rows.csv ]; do sleep 60; done
L=artifacts/v2/logs
set -e
[ -f artifacts/v2/selector_analysis/f_mimic_iv_$T/tuned.csv ] || \
  $PY scripts/v2/run_selector_analysis.py --pools $C/pools.pkl --out-dir artifacts/v2/selector_analysis/f_mimic_iv_$T --selector-targets eicu,hirid > $L/selector_analysis_$T.log 2>&1
[ -f artifacts/v2/selection_baselines/f_mimic_iv_$T/rows.csv ] || \
  $PY scripts/v2/run_selection_baselines.py --pools $C/pools.pkl --selector-analysis-dir artifacts/v2/selector_analysis/f_mimic_iv_$T \
    --out-dir artifacts/v2/selection_baselines/f_mimic_iv_$T --selector-targets eicu,hirid > $L/selection_baselines_$T.log 2>&1
[ -f artifacts/v2/schema_aware/f_mimic_iv_$T/single_target.csv ] || \
  $PY scripts/v2/run_schema_aware_selection.py --pools $C/pools.pkl --selector-analysis-dir artifacts/v2/selector_analysis/f_mimic_iv_$T \
    --out-dir artifacts/v2/schema_aware/f_mimic_iv_$T --agnostic-targets eicu,hirid \
    --unaware-registries artifacts/v2/selection_baselines/f_mimic_iv_$T/registries.json > $L/schema_aware_$T.log 2>&1
[ -f artifacts/v2/srs_ablation/f_mimic_iv_$T/rows.csv ] || \
  $PY scripts/v2/run_srs_ablation.py --pools $C/pools.pkl --out-dir artifacts/v2/srs_ablation/f_mimic_iv_$T > $L/srs_ablation_$T.log 2>&1
E1=artifacts/v2/selection_baselines/f_mimic_iv_$T/registries.json
[ -f artifacts/v2/registry_pairs/f_mimic_iv_${T}_fs_vs_baselines.csv ] || \
  $PY scripts/v2/run_registry_pairs.py --pools $C/pools.pkl --a artifacts/v2/srs_ablation/f_mimic_iv_$T/registries.json:srs_source_only \
    --b $E1:mrmr --b $E1:mmrfs --b $E1:l1_logistic --out artifacts/v2/registry_pairs/f_mimic_iv_${T}_fs_vs_baselines.csv > $L/registry_pairs_$T.log 2>&1
echo "[aki-analyses] done"
