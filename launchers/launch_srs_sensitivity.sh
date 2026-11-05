#!/usr/bin/env bash
# Hyper-parameter sensitivity of SRS-C-V (K = 10) on the four settings, one after another.
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for lab in gap1h gap2h gap4h aki_gap6h; do
  [ -f artifacts/v2/srs_sensitivity/f_mimic_iv_$lab/manifest.json ] && continue
  /d/BRCoo/miniconda/python scripts/v2/run_srs_sensitivity.py --e3-dir artifacts/v2/union/f_mimic_iv_$lab \
    --out-dir artifacts/v2/srs_sensitivity/f_mimic_iv_$lab > artifacts/v2/logs/srs_sensitivity_$lab.log 2>&1 || echo "failed $lab"
done
echo "ALL DONE"
