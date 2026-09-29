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
TARGET_COL = "mmszu_resolved"   # F-DATA used memory, msza-resolved -- see Step 3
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

n_mmszu_null_before = int(fdata["mmszu"].isna().sum())
fdata = features.resolve_fdata_memory_target(fdata)
n_fallback_used = int((fdata["mmszu"].isna() & fdata["msza"].notna()).sum())
print(f"mmszu null before resolution: {n_mmszu_null_before:,} / {len(fdata):,}")
print(f"rows that actually needed the msza fallback: {n_fallback_used:,}")
print(f"mmszu_resolved null after resolution: {fdata['mmszu_resolved'].isna().sum():,} (must be 0)")

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
      "most representative of their current memory footprint; averaging over more history dilutes "
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


import joblib
import pathlib

INTERIM_DIR = pathlib.Path("../data/interim")
INTERIM_DIR.mkdir(exist_ok=True)

best_params = {
    "RandomForest": {"n_estimators": 135, "max_depth": 13, "min_samples_leaf": 8, "max_features": 0.5135016306843436},
    "XGBoost": {"n_estimators": 79, "max_depth": 5, "learning_rate": 0.1922013171258541, "subsample": 0.890961327267642, "colsample_bytree": 0.8938787758491833},
    "LightGBM": {"n_estimators": 262, "num_leaves": 15, "learning_rate": 0.12640286235344467, "subsample": 0.939378128127307, "colsample_bytree": 0.9675308653219115},
}

prepared = dict(
    X_train=X_train, X_test=X_test,
    y_train_raw=y_train_raw, y_test_raw=y_test_raw,
    y_train=y_train, y_test=y_test,
    naive_pred=naive_pred, naive_metrics=naive_metrics,
    rf_cap=min(RF_FINAL_FIT_CAP, len(X_train)),
    best_params=best_params,
    SEED=SEED,
)
joblib.dump(prepared, INTERIM_DIR / "fdata_memory_prepared.joblib")
print(f"saved prepared data: X_train {X_train.shape}, X_test {X_test.shape}")
