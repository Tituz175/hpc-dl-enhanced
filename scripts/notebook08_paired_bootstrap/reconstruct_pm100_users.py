"""
Reconstructs test-set user_id for all three PM100 combinations (power, duration, memory
clean-4). None of the three target-specific pipelines drop any rows after
filter_completed_jobs (no dropna/resolution step, unlike F-DATA power), so all three share
the identical chronological_split on the same filtered dataframe -- confirmed by asserting
the reconstructed test_df index matches each combination's own saved X_test index
independently below, rather than assumed.

Saves data/interim/notebook08_pm100_user_ids.joblib: {combo: user_ids array, aligned to that
combo's saved X_test row order}. Also prints each combination's top-5-user row-share and
sum-of-squares-share up front, per spec point 6/the follow-up request to see this before
running any pairwise comparisons.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

from src import features, splits
from bootstrap_lib import compute_per_user_stats, top5_user_share

INTERIM_DIR = ROOT / "data" / "interim"
PM100_PATH = ROOT / "data" / "raw" / "pm100" / "pm100_job_table.parquet"

print("loading PM100 raw data and applying the shared (dropna-free) pipeline prefix")
pm100 = pd.read_parquet(PM100_PATH)
pm100 = features.filter_completed_jobs(pm100, "pm100")
_, test_df = splits.chronological_split(pm100, "pm100")
print(f"shared test_df: {len(test_df):,} rows")

combos = {
    "pm100_power": "pm100_power_prepared.joblib",
    "pm100_duration": "pm100_duration_prepared.joblib",
    "pm100_memory_c4": "pm100_memory_clean4_prepared.joblib",
}

results = {}
for combo, fname in combos.items():
    prep = joblib.load(INTERIM_DIR / fname)
    X_test = prep["X_test"]
    assert list(test_df.index) == list(X_test.index), \
        f"{combo}: shared test_df index does not match its saved X_test index"
    user_ids = test_df.loc[X_test.index, "user_id"].to_numpy()
    y_raw = prep["y_test_raw"]
    stats = compute_per_user_stats(user_ids, y_raw, y_raw, y_raw)
    row_share, ss_share = top5_user_share(stats)
    results[combo] = user_ids
    print(f"  {combo}: {stats['n_users']} unique users, {stats['n_rows']:,} rows, "
          f"top-5-user row-share={row_share:.2%}, top-5-user sum-of-squares-share={ss_share:.2%}")

joblib.dump(results, INTERIM_DIR / "notebook08_pm100_user_ids.joblib")
print(f"saved {INTERIM_DIR / 'notebook08_pm100_user_ids.joblib'}")
