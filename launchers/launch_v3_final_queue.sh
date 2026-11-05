#!/bin/bash
# Final compute queue (2026-11-05): compilable-only SRS+T on the remaining roots, then the selection timing repeats.
cd "/d/ALL PROJECT/Non-stationary times series"
export CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PY=/d/BRCoo/miniconda/python
L=artifacts/v2/logs
for root in union_k5 union union_large_k10 union_large_k20; do
  $PY scripts/v2/run_reference_registries.py --phase compilable --root $root > $L/v3r2_compilable_$root.log 2>&1
done
echo done > $L/v3r2_compilable_all.done
bash tmp/v2_helpers/launch_v3_timing_repeats.sh
