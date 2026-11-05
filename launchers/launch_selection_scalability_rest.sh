#!/usr/bin/env bash
# After pass 1 (pool sizes <= 20k), the remaining selection-scalability passes, one after another.
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
PY=/d/BRCoo/miniconda/python
S=scripts/v2/run_selection_scalability.py
L=artifacts/v2/selection_scalability
while [ ! -f $L/README.md ]; do sleep 60; done  # pass 1 writes README.md when it ends
$PY $S --sizes 500,1000 --fractions 1.0 --budgets 10 --methods mmrfs > $L/pass1b_mmrfs.log 2>&1
$PY $S --sizes "" --fractions 0.25,0.5 --budgets 10 > $L/pass1b_episodes.log 2>&1
$PY $S --sizes "" --fractions 1.0 --budgets 5,20,40 --methods srs_c_v,fs_cv,mrmr,jmi,cmim,l1_logistic,mmrfs,dpp > $L/pass1b_budgets.log 2>&1
$PY $S --sizes "" --fractions 1.0 --budgets 5,20 --methods srs_c_exact > $L/pass1b_budgets_exact.log 2>&1
$PY $S --sizes 50000,100000 --fractions 1.0 --budgets 10 --methods srs_c_v,fs_cv,mrmr,jmi,cmim,l1_logistic --timeout 14400 > $L/pass2_large.log 2>&1
echo "ALL DONE"
