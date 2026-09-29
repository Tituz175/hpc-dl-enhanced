#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS="$SCRIPT_DIR"
cd "$PROJ/notebooks"

echo "=== [$(date)] STAGE 0: setup (load + feature engineering) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/fdata_power_00_setup.py"

echo "=== [$(date)] STAGE 1: RandomForest (fresh process) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/fdata_power_01_rf.py"

echo "=== [$(date)] STAGE 2: XGBoost (fresh process) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/fdata_power_02_xgb.py"

echo "=== [$(date)] STAGE 3: LightGBM (fresh process) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/fdata_power_03_lgbm.py"

echo "=== [$(date)] ALL STAGES COMPLETE ==="
