#!/usr/bin/env bash
# Idle-machine timings for the paper (2026-11-04): SRS-AP selection time against the pool size (srs_ap_v, sizes
# 500-20,000, K = 10, P-cores, 6 BLAS threads), then the SQL scale-up (eICU x1/x10/x30, shipped SRS+T-AP registry).
# A load logger records system CPU use once a minute (artifacts/v2/logs/timing_load_2026-11-04.csv).
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
PY=/d/BRCoo/miniconda/python
L=artifacts/v2/logs
STOP=$L/timing_load.stop
rm -f $STOP
$PY tmp/v2_helpers/log_load.py $L/timing_load_2026-11-04.csv $STOP &
$PY scripts/v2/run_selection_scalability.py --methods srs_ap_v --sizes 500,1000,2274,5000,10000,20000 --fractions 1.0 --budgets 10 \
  > artifacts/v2/selection_scalability/pass_ap_sizes.log 2>&1 || echo "failed scalability"
$PY scripts/v2/run_sql_execution.py --runs union:f_mimic_iv_gap1h,union:f_mimic_iv_aki_gap6h --registry srs_ap_t --scales 1,10,30 \
  --out-root artifacts/v2/sql_execution > $L/sql_execution_scale.log 2>&1 || echo "failed sql"
touch $STOP
echo "ALL DONE"
