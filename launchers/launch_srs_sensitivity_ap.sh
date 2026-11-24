#!/usr/bin/env bash
# Hyper-parameter and CV-seed sensitivity of SRS+T-AP (K = 10, four settings).
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for lab in gap1h gap2h gap4h aki_gap6h; do
  [ -f artifacts/v2/srs_sensitivity_ap/f_mimic_iv_$lab/manifest.json ] && continue
  "${PYTHON:-python}" scripts/v2/run_srs_sensitivity.py --e3-dir artifacts/v2/union/f_mimic_iv_$lab \
    --out-dir artifacts/v2/srs_sensitivity_ap/f_mimic_iv_$lab --decision auprc \
    --reference artifacts/v2/union_ap/union/f_mimic_iv_$lab/registries.json:srs_ap_t \
    > artifacts/v2/logs/srs_sensitivity_ap_$lab.log 2>&1 || echo "failed $lab"
done
echo "ALL DONE"
