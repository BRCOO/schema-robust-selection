#!/bin/bash
# Selection timing repeats on an idle workstation (review of the v3 experiments, round 2, item M7): two more repeats of the
# size axis for SRS+T (AUPRC decision), the likelihood variant, exact greedy and the filters, then one AUPRC-decision run at
# 10^5 candidates. Each repeat writes its own folder; the first repeat is artifacts/v2/selection_scalability.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python}"
L=artifacts/v2/logs
# Hold until the remaining heavy analyses are done, so that the timings run on an idle machine (touch $L/timing.go).
while [ ! -f $L/timing.go ]; do sleep 60; done
for r in 2 3; do
  $PY scripts/v2/run_selection_scalability.py --methods srs_ap_v,srs_c_v,srs_c_exact,nmifs,mrmr --sizes 500,1000,2274,5000,10000,20000 \
    --fractions 1.0 --budgets 10 --exact-limit 20000 --out-dir artifacts/v2/selection_scalability_repeats/r$r > $L/v3r2_scal_r$r.log 2>&1
done
# exact greedy at 10^4 and 2*10^4 for the first repeat too (it stopped at 5,000), so every size has three measurements
$PY scripts/v2/run_selection_scalability.py --methods srs_c_exact --sizes 10000,20000 --fractions 1.0 --budgets 10 --exact-limit 20000 --out-dir artifacts/v2/selection_scalability_repeats/r1_exact > $L/v3r2_scal_r1_exact.log 2>&1
$PY scripts/v2/run_selection_scalability.py --methods srs_ap_v --sizes 50000,100000 --fractions 1.0 --budgets 10 --timeout 14400 \
  --out-dir artifacts/v2/selection_scalability_repeats/ap_large > $L/v3r2_scal_ap_large.log 2>&1
echo done > $L/v3r2_scal.done
