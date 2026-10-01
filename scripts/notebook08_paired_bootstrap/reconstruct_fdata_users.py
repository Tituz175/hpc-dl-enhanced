"""
Reconstructs test-set usr (username) for all three F-DATA combinations (power, duration
Result 2, memory). Loads the full 38-month F-DATA (no embedding, ~27GB) exactly ONCE and
shares the filter_completed_jobs + handle_mszl_sentinel prefix across all three targets,
since those steps don't depend on the target column.

Row filtering DOES differ by target after that shared prefix: power additionally drops rows
via sanitize_fdata_power + dropna(avgpcon) BEFORE its own chronological_split, which shifts
its split boundary relative to duration/memory (which have no target-specific row-dropping
step and so share an identical split with each other, confirmed by the two independent
index-match assertions below rather than assumed).

Saves data/interim/notebook08_fdata_user_ids.joblib: {combo: usr array, aligned to that
combo's saved X_test row order}. Prints each combination's top-5-user row-share and
sum-of-squares-share up front.
"""
import glob
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import joblib

from src import features, splits
from bootstrap_lib import compute_per_user_stats, top5_user_share

INTERIM_DIR = ROOT / "data" / "interim"
FDATA_DIR = ROOT / "data" / "raw" / "fdata"
N_CONSECUTIVE_MONTHS = 38

t0 = time.perf_counter()
fdata_files = sorted(glob.glob(str(FDATA_DIR / "*.parquet")))[:N_CONSECUTIVE_MONTHS]
print(f"[{time.strftime('%H:%M:%S')}] loading {len(fdata_files)} F-DATA months (no embedding)")
fdata_base = features.load_fdata_no_embedding(fdata_files)
fdata_base = features.filter_completed_jobs(fdata_base, "fdata")
fdata_base = features.handle_mszl_sentinel(fdata_base)
print(f"[{time.strftime('%H:%M:%S')}] shared prefix done: {len(fdata_base):,} rows, "
      f"elapsed={time.perf_counter() - t0:.0f}s")

results = {}

# ---- power: extra sanitize + dropna before its own split ----
fdata_power = features.sanitize_fdata_power(fdata_base.copy())
fdata_power = fdata_power.dropna(subset=["avgpcon"])
_, test_df_power = splits.chronological_split(fdata_power, "fdata")
prep = joblib.load(INTERIM_DIR / "fdata_power_prepared.joblib")
assert list(test_df_power.index) == list(prep["X_test"].index), \
    "fdata_power: reconstructed test_df index does not match saved X_test index"
results["fdata_power"] = test_df_power.loc[prep["X_test"].index, "usr"].to_numpy()
print(f"[{time.strftime('%H:%M:%S')}] fdata_power: index match confirmed, "
      f"{len(test_df_power):,} test rows")

# ---- duration (Result 2): no extra row-dropping step ----
_, test_df_duration = splits.chronological_split(fdata_base, "fdata")
prep = joblib.load(INTERIM_DIR / "fdata_duration_r2_prepared.joblib")
assert list(test_df_duration.index) == list(prep["X_test"].index), \
    "fdata_duration: reconstructed test_df index does not match saved X_test index"
results["fdata_duration"] = test_df_duration.loc[prep["X_test"].index, "usr"].to_numpy()
print(f"[{time.strftime('%H:%M:%S')}] fdata_duration: index match confirmed, "
      f"{len(test_df_duration):,} test rows")

# ---- memory: resolve_fdata_memory_target doesn't drop rows, but compute independently ----
fdata_memory = features.resolve_fdata_memory_target(fdata_base.copy())
_, test_df_memory = splits.chronological_split(fdata_memory, "fdata")
prep = joblib.load(INTERIM_DIR / "fdata_memory_prepared.joblib")
assert list(test_df_memory.index) == list(prep["X_test"].index), \
    "fdata_memory: reconstructed test_df index does not match saved X_test index"
results["fdata_memory"] = test_df_memory.loc[prep["X_test"].index, "usr"].to_numpy()
print(f"[{time.strftime('%H:%M:%S')}] fdata_memory: index match confirmed, "
      f"{len(test_df_memory):,} test rows")

print(f"\nsplit sharing check: duration and memory test sets "
      f"{'ARE identical' if list(test_df_duration.index) == list(test_df_memory.index) else 'DIFFER'} "
      f"(both have no target-specific row-dropping step)")

print("\ntop-5-user shares:")
for combo in ["fdata_power", "fdata_duration", "fdata_memory"]:
    fname = {"fdata_power": "fdata_power_prepared.joblib",
             "fdata_duration": "fdata_duration_r2_prepared.joblib",
             "fdata_memory": "fdata_memory_prepared.joblib"}[combo]
    prep = joblib.load(INTERIM_DIR / fname)
    y_raw = prep["y_test_raw"]
    user_ids = results[combo]
    stats = compute_per_user_stats(user_ids, y_raw, y_raw, y_raw)
    row_share, ss_share = top5_user_share(stats)
    print(f"  {combo}: {stats['n_users']} unique users, {stats['n_rows']:,} rows, "
          f"top-5-user row-share={row_share:.2%}, top-5-user sum-of-squares-share={ss_share:.2%}")

joblib.dump(results, INTERIM_DIR / "notebook08_fdata_user_ids.joblib")
print(f"\n[{time.strftime('%H:%M:%S')}] saved {INTERIM_DIR / 'notebook08_fdata_user_ids.joblib'}, "
      f"total elapsed={time.perf_counter() - t0:.0f}s")
