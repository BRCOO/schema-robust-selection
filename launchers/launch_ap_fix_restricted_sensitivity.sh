#!/usr/bin/env bash
# Re-run the analyses that select SRS-AP themselves, with the exact-tie AUPRC decision (fix of 2026-11-04):
# restricted pools (H: care, care_obs; I: single generators on the large pools) into artifacts/v2/restricted_pools_fix,
# then the hyper-parameter / seed sensitivity into artifacts/v2/srs_sensitivity_ap_fix (its default must reproduce the
# re-selected srs_ap_t in artifacts/v2/union_ap_fix).
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/d/BRCoo/miniconda/python
R=artifacts/v2/restricted_pools_fix
LARGE=union_large_k10:f_mimic_iv_gap1h,union_large_k10:f_mimic_iv_aki_gap6h
for spec in care care_obs; do
  $PY scripts/v2/run_restricted_pools.py --restrict $spec --out-root $R > artifacts/v2/logs/restricted_fix_$spec.log 2>&1 || echo "failed $spec"
done
for g in erminer prefixspan; do
  $PY scripts/v2/run_restricted_pools.py --restrict generator:$g --runs $LARGE --out-root $R > artifacts/v2/logs/restricted_fix_$g.log 2>&1 || echo "failed $g"
done
echo "RESTRICTED DONE"
for lab in gap1h gap2h gap4h aki_gap6h; do
  ref=artifacts/v2/union_ap_fix/union/f_mimic_iv_$lab/registries.json
  while [ ! -f "$ref" ]; do sleep 60; done
  [ -f artifacts/v2/srs_sensitivity_ap_fix/f_mimic_iv_$lab/manifest.json ] && continue
  $PY scripts/v2/run_srs_sensitivity.py --e3-dir artifacts/v2/union/f_mimic_iv_$lab \
    --out-dir artifacts/v2/srs_sensitivity_ap_fix/f_mimic_iv_$lab --decision auprc --reference $ref:srs_ap_t \
    > artifacts/v2/logs/srs_sensitivity_ap_fix_$lab.log 2>&1 || echo "failed sensitivity $lab"
done
echo "ALL DONE"
