#!/bin/bash
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
"${PYTHON:-python}" scripts/v2/run_reference_registries.py --phase seeds --root union_k20 --with-srs > artifacts/v2/logs/v3_seeds20.log 2>&1
echo done > artifacts/v2/logs/v3_seeds20.done
