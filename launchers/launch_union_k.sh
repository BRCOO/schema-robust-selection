#!/usr/bin/env bash
# E3 (union of generator pools) at budget K; usage: launch_union_k.sh <k> <label> [<label> ...]
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
PY=/d/BRCoo/miniconda/python
k=$1; shift
if [ "$k" = "10" ]; then out=artifacts/v2/union; e1=artifacts/v2/selection_baselines; else out=artifacts/v2/union_k$k; e1=artifacts/v2/selection_baselines_k$k; fi
for lab in "$@"; do
  e1dir=""
  [ -f $e1/f_mimic_iv_${lab}/rows.csv ] && e1dir=$e1/f_mimic_iv_${lab}
  $PY scripts/v2/run_union_selection.py --pools artifacts/v2/comparison/f_mimic_iv_${lab}/pools.pkl --k $k \
    --e1-dir "$e1dir" --out-dir $out/f_mimic_iv_${lab} > artifacts/v2/logs/union_k${k}_${lab}.log 2>&1 || exit 1
done
echo "[union K=$k] $* done"
