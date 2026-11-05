#!/usr/bin/env bash
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
PY=/d/BRCoo/miniconda/python
for g in 1 2 4; do
  $PY scripts/v2/run_srs_ablation.py --pools artifacts/v2/comparison/f_mimic_iv_gap${g}h/pools.pkl \
    --out-dir artifacts/v2/srs_ablation_k20fill/f_mimic_iv_gap${g}h --variants srs_known,srs_known_k20,srs_known_k20_fill || exit 1
done
echo "[k20-fill] done"
