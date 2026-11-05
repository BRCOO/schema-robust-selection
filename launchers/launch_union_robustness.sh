#!/usr/bin/env bash
# Decoder + subgroup robustness of the E3 registries (K = 10) for the remaining settings, one after another.
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for lab in gap2h gap4h aki_gap6h; do
  [ -f artifacts/v2/union_robustness/f_mimic_iv_$lab/manifest.json ] && continue
  /d/BRCoo/miniconda/python scripts/v2/run_union_robustness.py --e3-dir artifacts/v2/union/f_mimic_iv_$lab \
    --extra-dir artifacts/v2/union_extra/union/f_mimic_iv_$lab --out-dir artifacts/v2/union_robustness/f_mimic_iv_$lab \
    > artifacts/v2/logs/union_robustness_$lab.log 2>&1 || echo "failed $lab"
done
echo "ALL DONE"
