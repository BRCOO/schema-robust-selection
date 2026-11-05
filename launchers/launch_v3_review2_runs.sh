#!/bin/bash
# Second review round (2026-11-05): compute queue. Sequential; every step is resumable.
cd "/d/ALL PROJECT/Non-stationary times series"
export CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PY=/d/BRCoo/miniconda/python
L=artifacts/v2/logs
$PY scripts/v2/run_reference_registries.py --phase compilable --root union_k20 > $L/v3r2_compilable.log 2>&1
RS=""
for t in gap1h aki_gap6h; do for i in 1 2 3 4 5; do RS="$RS,union_resplit:r${i}_mimic_iv_${t}"; done; done
RS=${RS#,}
$PY scripts/v2/run_nmifs_ap.py --runs "$RS" --drive-k "" > $L/v3r2_resplit_nmifs_ap.log 2>&1
$PY scripts/v2/run_union_ap.py --phase ablate --runs "$RS" --drive-k "" > $L/v3r2_resplit_ablate.log 2>&1
for c in 0.01 0.1; do
  $PY scripts/v2/run_hospital_schemas.py --coverage $c --registries srs_ap_t,srs_ap,nmifs,mrmr,mrmr_filter_known,mmrfs,cmim,l1_dropout,fs_cv \
    --ref-t srs_ap_t --ref srs_ap --out-root artifacts/v2/hospital_schemas_ap > $L/v3r2_hospitals_$c.log 2>&1
done
$PY scripts/v2/run_reference_registries.py --phase seeds --root union_k5 --configs srs_ap_t,srs_ap,nmifs_ap_t,srs_ap_unaware --tag all4 > $L/v3r2_seeds_k5.log 2>&1
$PY scripts/v2/run_reference_registries.py --phase seeds --root union --configs srs_ap --tag srs > $L/v3r2_seeds_k10_srs.log 2>&1
$PY scripts/v2/run_reference_registries.py --phase seeds --root union_k20 --configs srs_ap --tag srs > $L/v3r2_seeds_k20_srs.log 2>&1
$PY scripts/v2/run_reference_registries.py --phase seeds --root union_large_k10 --configs srs_ap_t,srs_ap,nmifs_ap_t,srs_ap_unaware --tag all4 > $L/v3r2_seeds_l10.log 2>&1
$PY scripts/v2/run_reference_registries.py --phase seeds --root union_large_k20 --configs srs_ap_t,srs_ap,nmifs_ap_t,srs_ap_unaware --tag all4 > $L/v3r2_seeds_l20.log 2>&1
echo done > $L/v3r2_queue.done
