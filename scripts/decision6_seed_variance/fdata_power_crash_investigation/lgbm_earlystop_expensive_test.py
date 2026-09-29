import sys
sys.path.append("..")

from src import baselines, config, features, metrics, models, plotting, roofline, splits

# ===== next cell =====

import glob
import time

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

FDATA_DIR = "../data/raw/fdata"
N_CONSECUTIVE_MONTHS = 38     # full F-DATA -- every month
SAMPLE_SIZE = 1_000_000       # rows for the Optuna TUNING sample only (stratified within the
                             # train split); the final fit uses the full train split below
RF_FINAL_FIT_CAP = 5_000_000  # same disclosed deviation as notebook 05 -- RF final fit only
SHAP_SAMPLE = 200_000
PLOT_SAMPLE = 200_000
TARGET_COL = "avgpcon"   # F-DATA power (job-total draw, corruption-sanitized -- see Step 3)
N_TRIALS = models.TuningBudget().n_trials  # fixed tuning budget, same as every other target
SEED = 0
N_BUCKETS = 5
EMBEDDING_VARIANCE_THRESHOLD = 0.90
TUNE_TRAIN_FRAC = 0.8

print(f"N_CONSECUTIVE_MONTHS={N_CONSECUTIVE_MONTHS}  tuning SAMPLE_SIZE={SAMPLE_SIZE:,}  "
      f"RF_FINAL_FIT_CAP={RF_FINAL_FIT_CAP:,}  PLOT_SAMPLE={PLOT_SAMPLE:,}  TARGET_COL={TARGET_COL}")

# ===== next cell =====

fdata_files = sorted(glob.glob(f"{FDATA_DIR}/*.parquet"))[:N_CONSECUTIVE_MONTHS]
print(f"Loading {len(fdata_files)} months: {[f.split('/')[-1] for f in fdata_files]}")

fdata = features.load_fdata_no_embedding(fdata_files)
fdata = features.filter_completed_jobs(fdata, "fdata")
print(f"rows after filtering: {len(fdata):,}")

# ===== next cell =====

fdata = features.handle_mszl_sentinel(fdata)
features.assert_mszl_sanitized(fdata)
print(f"mszl_unlimited True: {fdata['mszl_unlimited'].sum():,} ({fdata['mszl_unlimited'].mean():.1%})")

# ===== next cell =====

n_before = len(fdata)
fdata = features.sanitize_fdata_power(fdata)
features.assert_fdata_power_sanitized(fdata)
fdata = fdata.dropna(subset=["avgpcon"])
print(f"rows before power-sanitize dropna: {n_before:,}")
print(f"rows after power-sanitize dropna: {len(fdata):,} ({n_before - len(fdata)} corrupted/invalid avgpcon rows removed)")

# ===== next cell =====

_window_candidates = [5, 20, 50]
_window_results = {}
for _w in _window_candidates:
    _fr = features.add_user_rolling_stat(fdata, "fdata", TARGET_COL, window=_w)
    _col = f"{TARGET_COL}_user_rolling_mean"
    _cov = _fr[_col].notna().sum()
    _samp = _fr.dropna(subset=[_col, TARGET_COL]).sample(n=min(1_000_000, len(_fr)), random_state=SEED)
    _corr = float(np.corrcoef(_samp[_col], _samp[TARGET_COL])[0, 1])
    _window_results[_w] = _corr
    print(f"  window={_w:3d}: coverage={_cov:,}/{len(_fr):,} ({_cov/len(_fr):.4%})  corr(rolling_mean, {TARGET_COL})={_corr:.4f}")
    del _fr, _samp

ROLLING_WINDOW = max(_window_results, key=_window_results.get)
print(f"\nchosen window: {ROLLING_WINDOW} (corr={_window_results[ROLLING_WINDOW]:.4f})")
print("Correlation strictly decreases as the window widens -- a user's most recent jobs are the "
      "most representative of their current power footprint; averaging over more history dilutes "
      "the signal rather than stabilizing it, the opposite of what duration's Result 2 found. "
      "window=5 wins outright, not a coin-flip pick."
      if ROLLING_WINDOW == min(_window_candidates) else
      "Correlation increases with window width here -- kept for the record, chosen on the evidence.")

# ===== next cell =====

fdata = features.add_user_rolling_stat(fdata, "fdata", TARGET_COL, window=ROLLING_WINDOW)
rolling_col = f"{TARGET_COL}_user_rolling_mean"
print(f"rolling stat (window={ROLLING_WINDOW}) available for {fdata[rolling_col].notna().sum():,} / {len(fdata):,} jobs")

# ===== next cell =====

train_df, test_df = splits.chronological_split(fdata, "fdata")
split_ratio = len(test_df) / len(train_df)
print(f"train={len(train_df):,} test={len(test_df):,} (test/train ratio={split_ratio:.4f})")

# ===== next cell =====

test_target_n = round(SAMPLE_SIZE * split_ratio)

train_sample = features.stratified_sample_by_job_size(train_df, "fdata", SAMPLE_SIZE, n_buckets=N_BUCKETS, seed=SEED)
test_sample = features.stratified_sample_by_job_size(test_df, "fdata", test_target_n, n_buckets=N_BUCKETS, seed=SEED)

print(f"train_sample={len(train_sample):,} (target {SAMPLE_SIZE:,}), "
      f"test_sample={len(test_sample):,} (target {test_target_n:,})")

# ===== next cell =====

train_tier_a = features.build_tier_a_features(train_df, "fdata", include_embedding=False)
test_tier_a = features.build_tier_a_features(test_df, "fdata", include_embedding=False)

features.assert_no_tier_leakage(list(train_tier_a.columns), "A", "fdata")
features.assert_no_tier_leakage(list(test_tier_a.columns), "A", "fdata")
print(f"F-DATA Tier A: {len(train_tier_a.columns)} columns | "
      f"train {len(train_tier_a):,} rows, test {len(test_tier_a):,} rows")

# ===== next cell =====

embed_full = pd.concat(
    [pd.read_parquet(f, columns=["embedding"]) for f in fdata_files], ignore_index=True
)
embed_train = embed_full.loc[train_sample.index]
embed_test = embed_full.loc[test_sample.index]

pca_check = features.compute_embedding_explained_variance(embed_train)
n_components = features.n_components_for_variance(pca_check, threshold=EMBEDDING_VARIANCE_THRESHOLD)
print(f"embedding PCA: {n_components} components reach {EMBEDDING_VARIANCE_THRESHOLD:.0%} "
      f"explained variance (fit on train_sample only, {len(embed_train):,} rows)")

embedding_pca = features.fit_fdata_embedding_pca(embed_train, n_components=n_components)
embed_train_pca = features.transform_fdata_embedding(embed_train, embedding_pca)
embed_test_pca = features.transform_fdata_embedding(embed_test, embedding_pca)

embed_train_full = embed_full.loc[train_df.index]
embed_test_full = embed_full.loc[test_df.index]
del embed_full
embed_train_full_pca = features.transform_fdata_embedding(embed_train_full, embedding_pca)
embed_test_full_pca = features.transform_fdata_embedding(embed_test_full, embedding_pca)
print(f"full-split embedding PCA: train {embed_train_full_pca.shape}, test {embed_test_full_pca.shape}")

# ===== next cell =====

X_train = features.build_fdata_numeric_matrix(
    train_tier_a, embed_train_full_pca, extra_columns=train_df[[rolling_col]]
).fillna(0.0)
X_test = features.build_fdata_numeric_matrix(
    test_tier_a, embed_test_full_pca, extra_columns=test_df[[rolling_col]]
).fillna(0.0)

y_train_raw = train_df[TARGET_COL].to_numpy(dtype=float)
y_test_raw = test_df[TARGET_COL].to_numpy(dtype=float)
y_train = features.transform_target(y_train_raw)
y_test = features.transform_target(y_test_raw)

metrics.expm1_round_trip_check(y_train_raw)
metrics.expm1_round_trip_check(y_test_raw)

print(f"X_train shape: {X_train.shape}  X_test shape: {X_test.shape}")
print(f"feature columns: {list(X_train.columns)}")

# ===== next cell =====

user_medians, global_median = baselines.fit_naive_baseline(train_df, "fdata", TARGET_COL)
naive_pred = baselines.predict_naive_baseline(test_df, "fdata", user_medians, global_median)
naive_metrics = metrics.regression_metrics(y_test_raw, naive_pred)

print(f"Naive per-user-median baseline (full test split, n={len(test_df):,}):")
for k, v in naive_metrics.items():
    print(f"  {k}: {v:,.4f}")

# ===== next cell =====

tune_train_raw, tune_val_raw = splits.chronological_split(train_sample, "fdata", train_frac=TUNE_TRAIN_FRAC)
print(f"tune_train={len(tune_train_raw):,} tune_val={len(tune_val_raw):,}")

tune_train_tier_a = features.build_tier_a_features(tune_train_raw, "fdata", include_embedding=False)
tune_val_tier_a = features.build_tier_a_features(tune_val_raw, "fdata", include_embedding=False)

embed_tune_train = embed_train.loc[tune_train_raw.index]
embed_tune_val = embed_train.loc[tune_val_raw.index]
embed_tune_train_pca = features.transform_fdata_embedding(embed_tune_train, embedding_pca)
embed_tune_val_pca = features.transform_fdata_embedding(embed_tune_val, embedding_pca)

X_tune_train = features.build_fdata_numeric_matrix(
    tune_train_tier_a, embed_tune_train_pca, extra_columns=tune_train_raw[[rolling_col]]
).fillna(0.0)
X_tune_val = features.build_fdata_numeric_matrix(
    tune_val_tier_a, embed_tune_val_pca, extra_columns=tune_val_raw[[rolling_col]]
).fillna(0.0)

y_tune_train = features.transform_target(tune_train_raw[TARGET_COL].to_numpy(dtype=float))
y_tune_val = features.transform_target(tune_val_raw[TARGET_COL].to_numpy(dtype=float))

print(f"X_tune_train shape: {X_tune_train.shape}  X_tune_val shape: {X_tune_val.shape}")

# ===== next cell =====


import lightgbm as lgb
import numpy as np

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] setup complete, starting early-stopping test on the EXPENSIVE point (n_estimators=500 ceiling, num_leaves=255 -- the true top of the search space, the point that was never safely measured before)")

params = dict(n_estimators=500, num_leaves=255, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)

t0 = time.perf_counter()
model = lgb.LGBMRegressor(**params, bagging_freq=1, n_jobs=-1, random_state=SEED, verbose=-1)
model.fit(
    X_tune_train, y_tune_train,
    eval_set=[(X_tune_val, y_tune_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=True), lgb.log_evaluation(period=10)],
)
elapsed = time.perf_counter() - t0
pred = model.predict(X_tune_val)
rmse = float(np.sqrt(np.mean((y_tune_val - pred) ** 2)))

print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] EXPENSIVE-POINT EARLY-STOPPING TEST RESULT")
print(f"  n_estimators ceiling: 500, num_leaves: 255")
print(f"  best_iteration_: {model.best_iteration_}  (rounds actually used, out of 500 allowed)")
print(f"  elapsed: {elapsed:.1f}s")
print(f"  rmse: {rmse:.4f}")
