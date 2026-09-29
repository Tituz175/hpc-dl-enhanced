#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS="$SCRIPT_DIR"
cd "$PROJ/notebooks"

echo "=== [$(date)] STAGE 1: RandomForest (5 seeds) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/pm100_duration_seed_rf.py"

echo "=== [$(date)] STAGE 2: XGBoost (hist + exact, both full 5-seed sweeps) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/pm100_duration_seed_xgb.py"

echo "=== [$(date)] STAGE 3: LightGBM (5 seeds) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/pm100_duration_seed_lgbm.py"

echo "=== [$(date)] ALL STAGES COMPLETE ==="
