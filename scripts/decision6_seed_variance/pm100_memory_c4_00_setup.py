import sys
sys.path.append("..")

import glob
import time

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src import baselines, features, metrics, models, plotting, splits

optuna.logging.set_verbosity(optuna.logging.WARNING)

PM100_PATH = "../data/raw/pm100/pm100_job_table.parquet"
PM100_TARGET_COL = "mem_alloc"    # raw scalar column -- no reduction needed (unlike power)
PM100_TUNE_TRAIN_FRAC = 0.8
SHAP_SAMPLE = 200_000
N_TRIALS = models.TuningBudget().n_trials
SEED = 0

print(f"PM100_TARGET_COL={PM100_TARGET_COL}  N_TRIALS={N_TRIALS}")

# ===== next cell =====

pm100 = pd.read_parquet(PM100_PATH)
print(f"PM100 raw rows: {len(pm100):,}")

pm100 = features.filter_completed_jobs(pm100, "pm100")
print(f"rows after filtering: {len(pm100):,}")
print(f"mem_alloc null: {pm100['mem_alloc'].isna().sum():,}  dtype: {pm100['mem_alloc'].dtype}")

# ===== next cell =====

pm100 = features.add_pm100_derived_indicators(pm100)
features.assert_reserved_columns_preserved(pm100)
print("reserved columns preserved:", [c for c in features.PM100_RESERVED_COLUMNS if c in pm100.columns])

# ===== next cell =====

_window_candidates = [5, 20, 50]
_window_results = {}
for _w in _window_candidates:
    _pr = features.add_user_rolling_stat(pm100, "pm100", PM100_TARGET_COL, window=_w)
    _col = f"{PM100_TARGET_COL}_user_rolling_mean"
    _cov = _pr[_col].notna().sum()
    _samp = _pr.dropna(subset=[_col, PM100_TARGET_COL])
    _corr = float(np.corrcoef(_samp[_col], _samp[PM100_TARGET_COL])[0, 1])
    _window_results[_w] = _corr
    print(f"  window={_w:3d}: coverage={_cov:,}/{len(_pr):,} ({_cov/len(_pr):.4%})  corr(rolling_mean, {PM100_TARGET_COL})={_corr:.4f}")
    del _pr, _samp

PM100_ROLLING_WINDOW = max(_window_results, key=_window_results.get)
print(f"\nchosen window: {PM100_ROLLING_WINDOW} (corr={_window_results[PM100_ROLLING_WINDOW]:.4f})")
print("Same direction as Part 1: correlation strictly decreases as the window widens. Also, per the "
      "05b planning report, this rolling feature (corr ~0.77) is weaker than num_nodes_req alone "
      "(corr ~0.89) -- unlike Part 1, the resource-request block is expected to matter more here "
      "than the historical feature; the stripped-feature ablation below quantifies this directly.")

# ===== next cell =====

pm100 = features.add_user_rolling_stat(pm100, "pm100", PM100_TARGET_COL, window=PM100_ROLLING_WINDOW)
pm100_rolling_col = f"{PM100_TARGET_COL}_user_rolling_mean"
print(f"rolling stat (window={PM100_ROLLING_WINDOW}) available for {pm100[pm100_rolling_col].notna().sum():,} / {len(pm100):,} jobs")

# ===== next cell =====

pm100_train_df, pm100_test_df = splits.chronological_split(pm100, "pm100")
print(f"train={len(pm100_train_df):,} test={len(pm100_test_df):,} "
      f"(test/train ratio={len(pm100_test_df) / len(pm100_train_df):.4f})")

pm100_y_train_raw = pm100_train_df[PM100_TARGET_COL].to_numpy(dtype=float)
pm100_y_test_raw = pm100_test_df[PM100_TARGET_COL].to_numpy(dtype=float)

# ===== next cell =====

pm100_user_medians, pm100_global_median = baselines.fit_naive_baseline(pm100_train_df, "pm100", PM100_TARGET_COL)
pm100_naive_pred = baselines.predict_naive_baseline(pm100_test_df, "pm100", pm100_user_medians, pm100_global_median)
pm100_naive_metrics = metrics.regression_metrics(pm100_y_test_raw, pm100_naive_pred)

print("Naive per-user-median baseline (PM100 mem_alloc, test split):")
for k, v in pm100_naive_metrics.items():
    print(f"  {k}: {v:,.4f}")

# ===== next cell =====


import joblib
import pathlib

INTERIM_DIR = pathlib.Path("../data/interim")
INTERIM_DIR.mkdir(exist_ok=True)

CLEAN4_COLS = ["num_nodes_req", "num_gpus_req", "num_cores_req", pm100_rolling_col]
pm100_X_train_c4 = pm100_train_df[CLEAN4_COLS].astype(float).fillna(0.0)
pm100_X_test_c4 = pm100_test_df[CLEAN4_COLS].astype(float).fillna(0.0)
print(f"clean-4 matrix: X_train {pm100_X_train_c4.shape}  X_test {pm100_X_test_c4.shape}")
print(f"columns: {list(pm100_X_train_c4.columns)}")

pm100_y_train_raw = pm100_train_df[PM100_TARGET_COL].to_numpy(dtype=float)
pm100_y_test_raw = pm100_test_df[PM100_TARGET_COL].to_numpy(dtype=float)
pm100_y_train = features.transform_target(pm100_y_train_raw)
pm100_y_test = features.transform_target(pm100_y_test_raw)
metrics.expm1_round_trip_check(pm100_y_train_raw)
metrics.expm1_round_trip_check(pm100_y_test_raw)

best_params = {
    "RandomForest": {"n_estimators": 149, "max_depth": 10, "min_samples_leaf": 5, "max_features": 0.9286795150728064},
    "XGBoost": {"n_estimators": 188, "max_depth": 11, "learning_rate": 0.030862925609952782, "subsample": 0.8768040541890576, "colsample_bytree": 0.8921051173321578},
    "LightGBM": {"n_estimators": 188, "num_leaves": 101, "learning_rate": 0.11965759404600158, "subsample": 0.8612368964975208, "colsample_bytree": 0.8312766647047717},
}

prepared = dict(
    X_train=pm100_X_train_c4, X_test=pm100_X_test_c4,
    y_train_raw=pm100_y_train_raw, y_test_raw=pm100_y_test_raw,
    y_train=pm100_y_train, y_test=pm100_y_test,
    naive_pred=pm100_naive_pred, naive_metrics=pm100_naive_metrics,
    best_params=best_params,
    SEED=SEED, N_TRIALS=N_TRIALS,
)
joblib.dump(prepared, INTERIM_DIR / "pm100_memory_clean4_prepared.joblib")
print(f"saved prepared data: X_train {pm100_X_train_c4.shape}, X_test {pm100_X_test_c4.shape}")
