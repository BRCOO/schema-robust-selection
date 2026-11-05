#!/usr/bin/env bash
# AKI-onset cohorts (gap 6 h) for the three sources, sequentially.
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
PY=/d/BRCoo/miniconda/python
X=data/processed/v2/extract
set -e
$PY scripts/v2/extract_v2_cohort.py --task aki_onset --gap-hours 6 --fmg-view --source mimic_iv --raw-dir data/raw/mimic-iv/mimic-iv-3.1 --cache-dir data/processed/v2/cache/mimic_iv --out-dir $X/mimic_iv_aki_gap6h_full
$PY scripts/v2/extract_v2_cohort.py --task aki_onset --gap-hours 6 --fmg-view --source eicu --raw-dir "data/raw/EICU 2.0数据" --cache-dir data/processed/v2/cache/eicu --out-dir $X/eicu_aki_gap6h_full --match-columns hospitalid
$PY scripts/v2/extract_v2_cohort.py --task aki_onset --gap-hours 6 --fmg-view --source hirid --raw-dir data/raw/hirid/hirid-a-high-time-resolution-icu-dataset-1.1.1 --cache-dir data/processed/v2/cache/hirid --out-dir $X/hirid_aki_gap6h_full
echo "[aki-extract] all done"
