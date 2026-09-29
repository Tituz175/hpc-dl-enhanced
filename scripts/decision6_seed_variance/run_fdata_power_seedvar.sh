#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS="$SCRIPT_DIR"
cd "$PROJ/notebooks"

echo "=== [$(date)] SEEDVAR STAGE 1: RandomForest (3 seeds) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/fdata_power_seed_rf.py"

echo "=== [$(date)] SEEDVAR STAGE 2: XGBoost (5 seeds, 1M-row sample, exact) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/fdata_power_seed_xgb.py"

echo "=== [$(date)] SEEDVAR STAGE 3: LightGBM (5 seeds) ==="
uv run --project "$PROJ" python -u "$SCRIPTS/fdata_power_seed_lgbm.py"

echo "=== [$(date)] SEEDVAR ALL STAGES COMPLETE ==="
