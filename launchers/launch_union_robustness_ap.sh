#!/usr/bin/env bash
# Decoder + subgroup robustness of SRS-AP (K = 10, four settings), reference srs_ap_t.
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for lab in gap1h gap2h gap4h aki_gap6h; do
  [ -f artifacts/v2/union_robustness_ap/f_mimic_iv_$lab/manifest.json ] && continue
  /d/BRCoo/miniconda/python scripts/v2/run_union_robustness.py --e3-dir artifacts/v2/union/f_mimic_iv_$lab \
    --extra-dir artifacts/v2/union_extra/union/f_mimic_iv_$lab --out-dir artifacts/v2/union_robustness_ap/f_mimic_iv_$lab \
    --reference srs_ap_t --registries srs_ap_t,srs_ap,nmifs,fs_cv,mrmr,mrmr_filter_known,mmrfs,l1_logistic,l1_dropout,jmi,cmim \
    --more-registries artifacts/v2/union_ap/union/f_mimic_iv_$lab/registries.json,artifacts/v2/union_fairness/union/f_mimic_iv_$lab/registries.json \
    > artifacts/v2/logs/union_robustness_ap_$lab.log 2>&1 || echo "failed $lab"
done
echo "ALL DONE"
