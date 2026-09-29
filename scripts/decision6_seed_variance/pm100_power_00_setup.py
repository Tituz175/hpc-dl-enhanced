import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from src import baselines, config, features, metrics, models, plotting, roofline, splits

N_TRIALS = models.TuningBudget().n_trials
SEED = 0

# ===== next cell =====

PM100_PATH = "../data/raw/pm100/pm100_job_table.parquet"
PM100_TARGET_COL = "power_mean"   # derived scalar -- see Step 14, not a raw column
PM100_TUNE_TRAIN_FRAC = 0.8       # same chronological tune/val carve-out pattern as Part 1

print(f"PM100_TARGET_COL={PM100_TARGET_COL}  N_TRIALS={N_TRIALS}")

# ===== next cell =====

pm100 = pd.read_parquet(PM100_PATH)
print(f"PM100 raw rows: {len(pm100):,}")

pm100 = features.filter_completed_jobs(pm100, "pm100")
pm100["power_mean"] = pm100["node_power_consumption"].apply(np.mean)
print(f"rows after filtering: {len(pm100):,}")

# ===== next cell =====

pm100 = features.add_pm100_derived_indicators(pm100)
features.assert_reserved_columns_preserved(pm100)
print("reserved columns preserved:", [c for c in features.PM100_RESERVED_COLUMNS if c in pm100.columns])
print("num_tasks_missing present:", "num_tasks_missing" in pm100.columns)

# ===== next cell =====

pm100 = features.add_user_rolling_stat(pm100, "pm100", PM100_TARGET_COL, window=5)
pm100_rolling_col = f"{PM100_TARGET_COL}_user_rolling_mean"
print(f"rolling stat available for {pm100[pm100_rolling_col].notna().sum():,} / {len(pm100):,} jobs")

# ===== next cell =====

pm100_train_df, pm100_test_df = splits.chronological_split(pm100, "pm100")
print(f"train={len(pm100_train_df):,} test={len(pm100_test_df):,} "
      f"(test/train ratio={len(pm100_test_df) / len(pm100_train_df):.4f})")

# ===== next cell =====

pm100_train_tier_a = features.build_tier_a_features(pm100_train_df, "pm100")
pm100_test_tier_a = features.build_tier_a_features(pm100_test_df, "pm100")

features.assert_no_tier_leakage(list(pm100_train_tier_a.columns), "A", "pm100")
features.assert_no_tier_leakage(list(pm100_test_tier_a.columns), "A", "pm100")
print(f"PM100 Tier A: {len(pm100_train_tier_a.columns)} columns, OK")

# ===== next cell =====

pm100_X_train = features.build_pm100_numeric_matrix(
    pm100_train_tier_a, extra_columns=pm100_train_df[[pm100_rolling_col]]
).fillna(0.0)
pm100_X_test = features.build_pm100_numeric_matrix(
    pm100_test_tier_a, extra_columns=pm100_test_df[[pm100_rolling_col]]
).fillna(0.0)

pm100_y_train_raw = pm100_train_df[PM100_TARGET_COL].to_numpy(dtype=float)
pm100_y_test_raw = pm100_test_df[PM100_TARGET_COL].to_numpy(dtype=float)
pm100_y_train = features.transform_target(pm100_y_train_raw)
pm100_y_test = features.transform_target(pm100_y_test_raw)

metrics.expm1_round_trip_check(pm100_y_train_raw)
metrics.expm1_round_trip_check(pm100_y_test_raw)

print(f"X_train shape: {pm100_X_train.shape}  X_test shape: {pm100_X_test.shape}")
print(f"feature columns: {list(pm100_X_train.columns)}")

# ===== next cell =====

pm100_user_medians, pm100_global_median = baselines.fit_naive_baseline(pm100_train_df, "pm100", PM100_TARGET_COL)
pm100_naive_pred = baselines.predict_naive_baseline(pm100_test_df, "pm100", pm100_user_medians, pm100_global_median)
pm100_naive_metrics = metrics.regression_metrics(pm100_y_test_raw, pm100_naive_pred)

print("Naive per-user-median baseline (PM100 power_mean, test split):")
for k, v in pm100_naive_metrics.items():
    print(f"  {k}: {v:,.4f}")

# ===== next cell =====


import joblib
import pathlib
from src import models

N_TRIALS = models.TuningBudget().n_trials
SEED = 0

INTERIM_DIR = pathlib.Path("../data/interim")
INTERIM_DIR.mkdir(exist_ok=True)

best_params = {
    "RandomForest": {"n_estimators": 229, "max_depth": 9, "min_samples_leaf": 8, "max_features": 0.48374603344981765},
    "XGBoost": {"n_estimators": 128, "max_depth": 4, "learning_rate": 0.05882341368895798, "subsample": 0.9337098444135781, "colsample_bytree": 0.9294864662794635},
    "LightGBM": {"n_estimators": 89, "num_leaves": 19, "learning_rate": 0.16977763163336626, "subsample": 0.8890783754749252, "colsample_bytree": 0.9350060741234096},
}

prepared = dict(
    X_train=pm100_X_train, X_test=pm100_X_test,
    y_train_raw=pm100_y_train_raw, y_test_raw=pm100_y_test_raw,
    y_train=pm100_y_train, y_test=pm100_y_test,
    naive_pred=pm100_naive_pred, naive_metrics=pm100_naive_metrics,
    best_params=best_params,
    SEED=SEED, N_TRIALS=N_TRIALS,
)
joblib.dump(prepared, INTERIM_DIR / "pm100_power_prepared.joblib")
print(f"saved prepared data: X_train {pm100_X_train.shape}, X_test {pm100_X_test.shape}")
