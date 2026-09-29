#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS="$SCRIPT_DIR"
cd "$PROJ/notebooks"

echo "=== [$(date)] STAGE 0: setup (F-DATA duration Result 2 feature matrix) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/fdata_duration_r2_00_setup.py"

echo "=== [$(date)] STAGE 1: RandomForest (3 seeds) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/fdata_duration_r2_seed_rf.py"

echo "=== [$(date)] STAGE 2: XGBoost (5 seeds) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/fdata_duration_r2_seed_xgb.py"

echo "=== [$(date)] STAGE 3: LightGBM (5 seeds) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/fdata_duration_r2_seed_lgbm.py"

echo "=== [$(date)] ALL STAGES COMPLETE ==="
