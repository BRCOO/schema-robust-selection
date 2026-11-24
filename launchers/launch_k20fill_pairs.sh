#!/usr/bin/env bash
# K=20: budget-filling SRS against the K=20 established selectors (paired, internal + targets).
cd "$(dirname "$0")/.." || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
PY="${PYTHON:-python}"
for g in 1 2 4; do
  E1=artifacts/v2/selection_baselines_k20/f_mimic_iv_gap${g}h/registries.json
  $PY scripts/v2/run_registry_pairs.py --pools artifacts/v2/comparison/f_mimic_iv_gap${g}h/pools.pkl \
    --a artifacts/v2/srs_ablation_k20fill/f_mimic_iv_gap${g}h/registries.json:srs_known_k20_fill \
    --b $E1:mmrfs --b $E1:mrmr --b $E1:l1_logistic --b $E1:srs_known \
    --out artifacts/v2/registry_pairs/f_mimic_iv_gap${g}h_k20fill_vs_baselines.csv || exit 1
done
echo "[k20fill-pairs] done"
