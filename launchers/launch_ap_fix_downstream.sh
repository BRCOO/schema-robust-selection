#!/usr/bin/env bash
# Downstream re-runs after the exact-tie re-selection of SRS-AP (2026-11-04), against the registries in
# artifacts/v2/union_ap_fix (SRS_AP_ROOT). Waits for the main selections; copies the outputs of unchanged runs
# (tmp/v2_helpers/ap_fix_changed.py) and re-runs the changed ones; stress is re-run for all 16 runs (13 s each).
# Then waits for the re-split selections and evaluates the changed re-splits; finally re-exports the case studies.
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 SRS_AP_ROOT=union_ap_fix OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/d/BRCoo/miniconda/python
FIX=artifacts/v2/union_ap_fix
L=artifacts/v2/logs
while [ ! -f $FIX/drive_k20/f_backblaze_seagate_gap2d/select.json ]; do sleep 60; done
$PY tmp/v2_helpers/ap_fix_changed.py main > $L/ap_fix_changed_main.log 2>&1 || { echo "changed main failed"; exit 1; }
ICU=$(cat tmp/v2_helpers/ap_fix_changed_icu.txt); DRV=$(cat tmp/v2_helpers/ap_fix_changed_drive.txt)
$PY scripts/v2/run_union_ap.py --phase decide --out-root $FIX > $L/union_ap_fix_decide.log 2>&1 || echo "failed decide"
$PY scripts/v2/run_union_ap.py --phase evaluate --out-root $FIX --runs "$ICU" --drive-k "$DRV" > $L/union_ap_fix_evaluate.log 2>&1 || echo "failed evaluate"
$PY scripts/v2/run_stress.py --out-root artifacts/v2/stress_ap_fix \
  --registries srs_ap_t,srs_ap,srs_c_val,srs_c_unknown_val,nmifs,mrmr,mrmr_filter_known,l1_dropout,fs_cv,cmim > $L/stress_ap_fix.log 2>&1 || echo "failed stress"
$PY scripts/v2/run_deployment_modes.py --drive-k 5,10,20 --registries srs_ap_t,srs_ap,nmifs,mrmr,l1_dropout,fs_cv --dropout-grid 0.25,0.5 \
  --copies 4 --n-boot 1000 --seed 0 --out-root artifacts/v2/deployment_modes_ap_fix --ref-t srs_ap_t --ref srs_ap > $L/deployment_modes_ap_fix.log 2>&1 || echo "failed deploy"
$PY scripts/v2/run_hospital_schemas.py --registries srs_ap_t,srs_ap,nmifs,mrmr,mrmr_filter_known,mmrfs,cmim,l1_dropout,fs_cv --coverage 0.05 \
  --min-episodes 1 --n-boot 1000 --seed 0 --out-root artifacts/v2/hospital_schemas_ap_fix --ref-t srs_ap_t --ref srs_ap > $L/hospital_schemas_ap_fix.log 2>&1 || echo "failed hospitals"
for lab in gap1h gap2h gap4h aki_gap6h; do
  [ -f artifacts/v2/union_robustness_ap_fix/f_mimic_iv_$lab/manifest.json ] && continue
  $PY scripts/v2/run_union_robustness.py --e3-dir artifacts/v2/union/f_mimic_iv_$lab \
    --extra-dir artifacts/v2/union_extra/union/f_mimic_iv_$lab --out-dir artifacts/v2/union_robustness_ap_fix/f_mimic_iv_$lab \
    --reference srs_ap_t --registries srs_ap_t,srs_ap,nmifs,fs_cv,mrmr,mrmr_filter_known,mmrfs,l1_logistic,l1_dropout,jmi,cmim \
    --more-registries $FIX/union/f_mimic_iv_$lab/registries.json,artifacts/v2/union_fairness/union/f_mimic_iv_$lab/registries.json \
    > $L/union_robustness_ap_fix_$lab.log 2>&1 || echo "failed robustness $lab"
done
echo "MAIN DOWNSTREAM DONE"
while [ ! -f $FIX/union_resplit/r5_mimic_iv_aki_gap6h/select.json ]; do sleep 60; done
$PY tmp/v2_helpers/ap_fix_changed.py resplit > $L/ap_fix_changed_resplit.log 2>&1 || { echo "changed resplit failed"; exit 1; }
RS=$(cat tmp/v2_helpers/ap_fix_changed_resplit.txt)
if [ -n "$RS" ]; then
  $PY scripts/v2/run_union_ap.py --phase evaluate --out-root $FIX --runs "$RS" --drive-k "" > $L/union_ap_fix_evaluate_resplit.log 2>&1 || echo "failed resplit evaluate"
fi
for lab in gap1h aki_gap6h; do
  $PY scripts/v2/export_case_study.py --e3-dir artifacts/v2/union/f_mimic_iv_$lab --out-dir artifacts/v2/case_study_ap_fix/${lab}_k10 \
    --registry srs_ap_t --registries-file $FIX/union/f_mimic_iv_$lab/registries.json > $L/case_study_ap_fix_$lab.log 2>&1 || echo "failed case study $lab"
done
echo "ALL DONE"
