#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS="$SCRIPT_DIR"
cd "$PROJ/notebooks"

echo "=== [$(date)] STAGE 0: setup (load PM100 + feature engineering) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/pm100_duration_00_setup.py"

echo "=== [$(date)] STAGE 1: RandomForest (fresh process) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/pm100_duration_01_rf.py"

echo "=== [$(date)] STAGE 2: XGBoost (fresh process) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/pm100_duration_02_xgb.py"

echo "=== [$(date)] STAGE 3: LightGBM (fresh process) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/pm100_duration_03_lgbm.py"

echo "=== [$(date)] ALL STAGES COMPLETE ==="
