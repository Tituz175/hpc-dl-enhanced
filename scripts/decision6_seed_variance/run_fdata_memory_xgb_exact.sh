#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJ/notebooks"
uv run --project "$PROJ" python -u "$SCRIPT_DIR/fdata_memory_seed_xgb_exact.py"
