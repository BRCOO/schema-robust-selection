#!/usr/bin/env bash
cd "/d/ALL PROJECT/Non-stationary times series" || exit 1
bash tmp/v2_helpers/launch_e2_unaware.sh > artifacts/v2/logs/e2_filter_baselines.log 2>&1 || exit 1
bash tmp/v2_helpers/launch_e1_k5.sh > artifacts/v2/logs/e1_k5_driver.log 2>&1 || exit 1
echo "[e2+k5] done"
