#!/usr/bin/env bash
# Phase 1 of the statistic-predicate extension (exploratory, K = 10, the four standard settings): E3 union selection,
# SRS with the AUPRC decision, NMIFS / stratified mRMR / IG selectors, add-on comparators. Two workers in parallel.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
PY="${PYTHON:-python}"
run_one() {
  label=$1
  log=artifacts/v2/logs/union_stat_${label}.log
  e3=artifacts/v2/union_stat/$label
  extra=artifacts/v2/union_extra/union_stat/$label
  [ -f $e3/manifest.json ] || $PY scripts/v2/run_union_selection.py --pools artifacts/v2/comparison/$label/pools_stat.pkl --e1-dir "" --k 10 --out-dir $e3 >> $log 2>&1
  [ -f artifacts/v2/union_ap/union_stat/$label/select.json ] || $PY scripts/v2/run_union_ap.py --phase select --runs union_stat:$label --drive-k "" >> $log 2>&1
  [ -f artifacts/v2/union_fairness/union_stat/$label/registries.json ] || $PY scripts/v2/run_union_fairness.py --runs union_stat:$label >> $log 2>&1
  [ -f $extra/manifest.json ] || $PY scripts/v2/run_union_extra.py --e3-dir $e3 --out-dir $extra --stress-levels 0.5 --stress-draws 1 >> $log 2>&1
  echo "[phase1] $label done" >> $log
}
( run_one f_mimic_iv_gap1h; run_one f_mimic_iv_gap4h ) &
( run_one f_mimic_iv_aki_gap6h; run_one f_mimic_iv_gap2h ) &
wait
echo "[phase1] all done"
