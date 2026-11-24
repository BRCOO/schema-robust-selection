#!/usr/bin/env bash
# Re-select every SRS-AP registry with the exact-tie AUPRC decision (fix of 2026-11-04) into artifacts/v2/union_ap_fix:
# the 16 ICU runs and the three drive budgets (select, then ablate), and the ten re-splits (select only, as before).
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python}"
OUTR=artifacts/v2/union_ap_fix
RESPLITS=union_resplit:r1_mimic_iv_gap1h,union_resplit:r2_mimic_iv_gap1h,union_resplit:r3_mimic_iv_gap1h,union_resplit:r4_mimic_iv_gap1h,union_resplit:r5_mimic_iv_gap1h,union_resplit:r1_mimic_iv_aki_gap6h,union_resplit:r2_mimic_iv_aki_gap6h,union_resplit:r3_mimic_iv_aki_gap6h,union_resplit:r4_mimic_iv_aki_gap6h,union_resplit:r5_mimic_iv_aki_gap6h
$PY scripts/v2/run_union_ap.py --phase select --out-root $OUTR > artifacts/v2/logs/union_ap_fix_select.log 2>&1 || echo "failed select"
$PY scripts/v2/run_union_ap.py --phase select --out-root $OUTR --runs $RESPLITS --drive-k "" > artifacts/v2/logs/union_ap_fix_select_resplit.log 2>&1 || echo "failed resplit select"
$PY scripts/v2/run_union_ap.py --phase ablate --out-root $OUTR > artifacts/v2/logs/union_ap_fix_ablate.log 2>&1 || echo "failed ablate"
echo "ALL DONE"
