import sys, time, pathlib
sys.path.append("..")
import glob
import joblib
import numpy as np
import pandas as pd
from src import baselines, features, metrics, models, splits

PM100_PATH = "../data/raw/pm100/pm100_job_table.parquet"
PM100_TARGET_COL = "run_time"
PM100_TUNE_TRAIN_FRAC = 0.8
N_TRIALS = models.TuningBudget().n_trials
SEED = 0
print(f"PM100_TARGET_COL={PM100_TARGET_COL}  N_TRIALS={N_TRIALS}")

pm100 = pd.read_parquet(PM100_PATH)
print(f"PM100 raw rows: {len(pm100):,}")
pm100 = features.filter_completed_jobs(pm100, "pm100")
print(f"rows after filtering: {len(pm100):,}")
print(f"run_time null: {pm100['run_time'].isna().sum():,}  dtype: {pm100['run_time'].dtype}")

pm100 = features.add_pm100_derived_indicators(pm100)
features.assert_reserved_columns_preserved(pm100)
print("reserved columns preserved:", [c for c in features.PM100_RESERVED_COLUMNS if c in pm100.columns])

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
print(f"chosen window: {PM100_ROLLING_WINDOW} (corr={_window_results[PM100_ROLLING_WINDOW]:.4f})")

pm100 = features.add_user_rolling_stat(pm100, "pm100", PM100_TARGET_COL, window=PM100_ROLLING_WINDOW)
pm100_rolling_col = f"{PM100_TARGET_COL}_user_rolling_mean"
print(f"rolling stat (window={PM100_ROLLING_WINDOW}) available for {pm100[pm100_rolling_col].notna().sum():,} / {len(pm100):,} jobs")

pm100_train_df, pm100_test_df = splits.chronological_split(pm100, "pm100")
print(f"train={len(pm100_train_df):,} test={len(pm100_test_df):,}")

pm100_train_tier_a = features.build_tier_a_features(pm100_train_df, "pm100")
pm100_test_tier_a = features.build_tier_a_features(pm100_test_df, "pm100")
features.assert_no_tier_leakage(list(pm100_train_tier_a.columns), "A", "pm100")
features.assert_no_tier_leakage(list(pm100_test_tier_a.columns), "A", "pm100")
print(f"PM100 Tier A: {len(pm100_train_tier_a.columns)} columns, OK")

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

pm100_user_medians, pm100_global_median = baselines.fit_naive_baseline(pm100_train_df, "pm100", PM100_TARGET_COL)
pm100_naive_pred = baselines.predict_naive_baseline(pm100_test_df, "pm100", pm100_user_medians, pm100_global_median)
pm100_naive_metrics = metrics.regression_metrics(pm100_y_test_raw, pm100_naive_pred)
print("Naive per-user-median baseline (PM100 run_time, test split):")
for k, v in pm100_naive_metrics.items():
    print(f"  {k}: {v:,.4f}")

pm100_tune_train_raw, pm100_tune_val_raw = splits.chronological_split(
    pm100_train_df, "pm100", train_frac=PM100_TUNE_TRAIN_FRAC
)
print(f"tune_train={len(pm100_tune_train_raw):,} tune_val={len(pm100_tune_val_raw):,}")

pm100_tune_train_tier_a = features.build_tier_a_features(pm100_tune_train_raw, "pm100")
pm100_tune_val_tier_a = features.build_tier_a_features(pm100_tune_val_raw, "pm100")
pm100_X_tune_train = features.build_pm100_numeric_matrix(
    pm100_tune_train_tier_a, extra_columns=pm100_tune_train_raw[[pm100_rolling_col]]
).fillna(0.0)
pm100_X_tune_val = features.build_pm100_numeric_matrix(
    pm100_tune_val_tier_a, extra_columns=pm100_tune_val_raw[[pm100_rolling_col]]
).fillna(0.0)
pm100_y_tune_train = features.transform_target(pm100_tune_train_raw[PM100_TARGET_COL].to_numpy(dtype=float))
pm100_y_tune_val = features.transform_target(pm100_tune_val_raw[PM100_TARGET_COL].to_numpy(dtype=float))
print(f"X_tune_train shape: {pm100_X_tune_train.shape}  X_tune_val shape: {pm100_X_tune_val.shape}")

INTERIM_DIR = pathlib.Path("../data/interim")
INTERIM_DIR.mkdir(exist_ok=True)
joblib.dump(dict(
    X_train=pm100_X_train, X_test=pm100_X_test,
    y_train_raw=pm100_y_train_raw, y_test_raw=pm100_y_test_raw,
    y_train=pm100_y_train, y_test=pm100_y_test,
    X_tune_train=pm100_X_tune_train, X_tune_val=pm100_X_tune_val,
    y_tune_train=pm100_y_tune_train, y_tune_val=pm100_y_tune_val,
    naive_pred=pm100_naive_pred, naive_metrics=pm100_naive_metrics,
    rolling_col=pm100_rolling_col, rolling_window=PM100_ROLLING_WINDOW,
    N_TRIALS=N_TRIALS, SEED=SEED, TARGET_COL=PM100_TARGET_COL,
    num_cores_req_test=pm100_test_df["num_cores_req"].to_numpy(),
), INTERIM_DIR / "pm100_duration_prepared.joblib")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] saved prepared data")
