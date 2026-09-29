#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS="$SCRIPT_DIR"
cd "$PROJ/notebooks"

echo "=== [$(date)] EXT200 STAGE 1: RandomForest ==="
uv run --project "$PROJ" python -u "$SCRIPTS/pm100_duration_ext200_rf.py"

echo "=== [$(date)] EXT200 STAGE 2: XGBoost ==="
uv run --project "$PROJ" python -u "$SCRIPTS/pm100_duration_ext200_xgb.py"

echo "=== [$(date)] EXT200 STAGE 3: LightGBM ==="
uv run --project "$PROJ" python -u "$SCRIPTS/pm100_duration_ext200_lgbm.py"

echo "=== [$(date)] EXT200 ALL STAGES COMPLETE ==="
