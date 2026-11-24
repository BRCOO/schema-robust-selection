#!/usr/bin/env bash
# E3 (union of generator pools): two labels per queue; usage: launch_union.sh <label> [<label> ...]
cd "$(dirname "$0")/.." || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
PY="${PYTHON:-python}"
for lab in "$@"; do
  $PY scripts/v2/run_union_selection.py --pools artifacts/v2/comparison/f_mimic_iv_${lab}/pools.pkl \
    --e1-dir artifacts/v2/selection_baselines/f_mimic_iv_${lab} --out-dir artifacts/v2/union/f_mimic_iv_${lab} \
    > artifacts/v2/logs/union_${lab}.log 2>&1 || exit 1
done
echo "[union] $* done"
