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
RF_FINAL_FIT_CAP = 5_000_000  # RandomForest final fit is capped here (sklearn RF has no GPU
                             # path and gains little past a few million rows); XGBoost and
                             # LightGBM fit on the entire train split
SHAP_SAMPLE = 200_000        # SHAP on a sample of the full test set (exact SHAP over ~7M rows
                             # is too slow)
PLOT_SAMPLE = 200_000        # rows sampled from the full test split for the scatter / residual /
                             # error-by-size / boxplot figures (Steps 13b, 13b-ii/iii, 13f)
TARGET_COL = "duration"   # F-DATA execution time
N_TRIALS = models.TuningBudget().n_trials  # fixed tuning budget, same for every model family
SEED = 0
N_BUCKETS = 5              # job-size stratification buckets (cnumr-based)
EMBEDDING_VARIANCE_THRESHOLD = 0.90
TUNE_TRAIN_FRAC = 0.8      # chronological carve-out WITHIN train_sample: first 80% tunes, last 20% validates

print(f"N_CONSECUTIVE_MONTHS={N_CONSECUTIVE_MONTHS}  tuning SAMPLE_SIZE={SAMPLE_SIZE:,}  "
      f"RF_FINAL_FIT_CAP={RF_FINAL_FIT_CAP:,}  PLOT_SAMPLE={PLOT_SAMPLE:,}  TARGET_COL={TARGET_COL}")

# ===== next cell =====

fdata_files = sorted(glob.glob(f"{FDATA_DIR}/*.parquet"))[:N_CONSECUTIVE_MONTHS]
print(f"Loading {len(fdata_files)} months: {[f.split('/')[-1] for f in fdata_files]}")

fdata = features.load_fdata_no_embedding(fdata_files)
fdata = features.filter_completed_jobs(fdata, "fdata")
print(f"rows after filtering: {len(fdata):,}")

# ===== next cell =====

n_sentinel_before = int((fdata["mszl"] >= 1e15).sum())
print(f"mszl sentinel rows before fix: {n_sentinel_before:,} / {len(fdata):,} "
      f"({n_sentinel_before / len(fdata):.1%})")

fdata = features.handle_mszl_sentinel(fdata)
features.assert_mszl_sanitized(fdata)

print(f"mszl range after fix: [{fdata['mszl'].min():.3g}, {fdata['mszl'].max():.3g}]")
print(f"mszl_unlimited True: {fdata['mszl_unlimited'].sum():,} ({fdata['mszl_unlimited'].mean():.1%})")

# ===== next cell =====

fdata = features.add_user_rolling_stat(fdata, "fdata", TARGET_COL, window=5)
rolling_col = f"{TARGET_COL}_user_rolling_mean"
print(f"rolling stat available for {fdata[rolling_col].notna().sum():,} / {len(fdata):,} jobs")

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
embed_train = embed_full.loc[train_sample.index]   # tuning sample (PCA fit + Step 10 tuning)
embed_test = embed_full.loc[test_sample.index]

pca_check = features.compute_embedding_explained_variance(embed_train)
n_components = features.n_components_for_variance(pca_check, threshold=EMBEDDING_VARIANCE_THRESHOLD)
print(f"embedding PCA: {n_components} components reach {EMBEDDING_VARIANCE_THRESHOLD:.0%} "
      f"explained variance (fit on train_sample only, {len(embed_train):,} rows)")

embedding_pca = features.fit_fdata_embedding_pca(embed_train, n_components=n_components)
embed_train_pca = features.transform_fdata_embedding(embed_train, embedding_pca)
embed_test_pca = features.transform_fdata_embedding(embed_test, embedding_pca)

# Full train/test splits, for the final fit (Step 12).
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

# add_user_rolling_stat writes a fixed column name, so each extra call is renamed
# before merging. `rolling_col` (window=5) is already on `fdata` from Step 3.
ROLLING_COL_W20 = f"{TARGET_COL}_user_rolling_mean_w20"
ROLLING_COL_W50 = f"{TARGET_COL}_user_rolling_mean_w50"
_roll20 = features.add_user_rolling_stat(fdata, "fdata", TARGET_COL, window=20)[rolling_col].rename(ROLLING_COL_W20)
_roll50 = features.add_user_rolling_stat(fdata, "fdata", TARGET_COL, window=50)[rolling_col].rename(ROLLING_COL_W50)
fdata_r2 = fdata.join(_roll20).join(_roll50)
rolling_cols_r2 = [rolling_col, ROLLING_COL_W20, ROLLING_COL_W50]
for c in rolling_cols_r2:
    print(f"  {c}: {fdata_r2[c].notna().sum():,} / {len(fdata_r2):,} non-null")

train_df_r2 = train_df.join(fdata_r2[[ROLLING_COL_W20, ROLLING_COL_W50]])
test_df_r2 = test_df.join(fdata_r2[[ROLLING_COL_W20, ROLLING_COL_W50]])
assert len(train_df_r2) == len(train_df) and len(test_df_r2) == len(test_df)

# ===== next cell =====

X_train_r2 = features.build_fdata_numeric_matrix(
    train_tier_a, embed_train_full_pca, extra_columns=train_df_r2[rolling_cols_r2]
).fillna(0.0)
X_test_r2 = features.build_fdata_numeric_matrix(
    test_tier_a, embed_test_full_pca, extra_columns=test_df_r2[rolling_cols_r2]
).fillna(0.0)
print(f"Result 2: X_train_r2 {X_train_r2.shape}  X_test_r2 {X_test_r2.shape}  "
      f"(+{X_train_r2.shape[1] - X_train.shape[1]} cols vs Result 1)")
assert X_train_r2.shape[0] == len(y_train) and X_test_r2.shape[0] == len(y_test_raw)
assert X_train_r2.shape[1] == X_train.shape[1] + 2

# ===== next cell =====


import joblib
import pathlib

INTERIM_DIR = pathlib.Path("../data/interim")
INTERIM_DIR.mkdir(exist_ok=True)

best_params = {
    "RandomForest": {"n_estimators": 105, "max_depth": 18, "min_samples_leaf": 19, "max_features": 0.5495121930980689},
    "XGBoost": {"n_estimators": 411, "max_depth": 7, "learning_rate": 0.01696003021391471, "subsample": 0.7137365087386797, "colsample_bytree": 0.8996675930517088},
    "LightGBM": {"n_estimators": 493, "num_leaves": 207, "learning_rate": 0.014920507084345887, "subsample": 0.685307832922302, "colsample_bytree": 0.9415149135271289},
}

prepared = dict(
    X_train=X_train_r2, X_test=X_test_r2,
    y_train_raw=y_train_raw, y_test_raw=y_test_raw,
    y_train=y_train, y_test=y_test,
    naive_pred=naive_pred, naive_metrics=naive_metrics,
    rf_cap=min(RF_FINAL_FIT_CAP, len(X_train_r2)),
    best_params=best_params,
    SEED=SEED,
)
joblib.dump(prepared, INTERIM_DIR / "fdata_duration_r2_prepared.joblib")
print(f"saved prepared data: X_train {X_train_r2.shape}, X_test {X_test_r2.shape}")
