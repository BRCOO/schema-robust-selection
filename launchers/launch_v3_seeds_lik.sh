#!/bin/bash
# Review round 3 (2026-11-05), items H2/M3: CV seeds 1-4 for the likelihood versions (SRS+T, lik. = srs_c_val; Wrapper, lik. = fs_cv),
# so that every CV-based selector in Table 2 averages five seeds. Two lanes; the AKI large K = 20 run is the longest.
cd "/d/ALL PROJECT/Non-stationary times series"
export CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PY=/d/BRCoo/miniconda/python
L=artifacts/v2/logs
( for root in union_k5 union union_k20 union_large_k10; do
    $PY scripts/v2/run_reference_registries.py --phase seeds --root $root --configs srs_c_val,fs_cv --tag lik > $L/v3r3_seeds_lik_$root.log 2>&1
  done; echo done > $L/v3r3_seeds_lik_a.done ) &
( $PY scripts/v2/run_reference_registries.py --phase seeds --root union_large_k20 --configs srs_c_val,fs_cv --tag lik > $L/v3r3_seeds_lik_union_large_k20.log 2>&1
  echo done > $L/v3r3_seeds_lik_b.done ) &
wait
