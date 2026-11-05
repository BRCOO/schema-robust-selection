#!/usr/bin/env bash
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
PY=/d/BRCoo/miniconda/python
for lab in "$@"; do
  $PY scripts/v2/run_union_schema_aware.py --pools artifacts/v2/comparison/f_mimic_iv_${lab}/pools.pkl \
    --e3-dir artifacts/v2/union/f_mimic_iv_${lab} --out-dir artifacts/v2/union_schema_aware/f_mimic_iv_${lab} \
    > artifacts/v2/logs/union_e2_${lab}.log 2>&1 || exit 1
done
echo "[union-e2] $* done"
