#!/bin/bash
cd "/d/ALL PROJECT/Non-stationary times series"
export CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
/d/BRCoo/miniconda/python scripts/v2/paper_v3/sql_timing_repeats.py --repeats 5 --scales 1,10,30 --python-scales 1,10 --memory-scales 1,10 > artifacts/v2/logs/v3_sql_repeats.log 2>&1
echo done > artifacts/v2/logs/v3_sql_repeats.done
