#!/usr/bin/env bash
# Run scripts/v2/run_union_extra.py on every finished E3 run, 3 at a time (skips finished ones).
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
jobs=""
for root in union_large_k20 union_large_k10 union_k20 union union_k5; do
  for lab in aki_gap6h gap1h gap2h gap4h; do
    [ -f "artifacts/v2/$root/f_mimic_iv_$lab/manifest.json" ] || continue
    [ -f "artifacts/v2/union_extra/$root/f_mimic_iv_$lab/manifest.json" ] && continue
    jobs="$jobs $root:$lab"
  done
done
echo $jobs | tr ' ' '\n' | grep -v '^$' | xargs -P 3 -I{} bash -c '
  root=${1%%:*}; lab=${1##*:}
  "${PYTHON:-python}" scripts/v2/run_union_extra.py --e3-dir artifacts/v2/$root/f_mimic_iv_$lab \
      --out-dir artifacts/v2/union_extra/$root/f_mimic_iv_$lab > artifacts/v2/union_extra/logs/${root}_$lab.log 2>&1
  echo "done $1 (exit $?)"' _ {}
echo ALL DONE
