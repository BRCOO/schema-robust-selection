#!/bin/bash
cd "/d/ALL PROJECT/Non-stationary times series"
export CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for l in f_mimic_iv_gap1h f_mimic_iv_gap2h f_mimic_iv_gap4h f_mimic_iv_aki_gap6h; do
  if [ ! -f artifacts/v2/proxy_correlation/ext_corr_$l.csv ]; then
    /d/BRCoo/miniconda/python scripts/v2/paper_v3/proxy_fidelity_ext.py --pools artifacts/v2/comparison/$l/pools.pkl --tag $l --steps 20 > artifacts/v2/logs/v3r2_fidelity_$l.log 2>&1
  fi
done
echo done > artifacts/v2/logs/v3r2_fidelity.done
