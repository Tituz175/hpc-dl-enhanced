"""
Shared data loading for the deep learning sweep (3 architectures x 6 combinations). One
function, load_combo(combo), handles all six combos' quirks (F-DATA's shared base load +
per-target row-dropping differences already established in earlier notebook 06/08 work;
PM100's three targets sharing one split) and returns everything a sweep pair needs: train/test
feature frames (straight from each combo's already-saved *_prepared.joblib -- no feature
reconstruction from raw data needed), user identity + time for the FULL (train+test)
population so sequences can be built once with test rows correctly reaching back into train
history, and the dataset name needed by splits.chronological_split.

Sequences are built ONCE over the concatenated (train+test) population per combo -- train
rows occupy positions [0, n_train), test rows [n_train, n_train+n_test) -- so a test row's
sequence can legitimately reference train-split rows (same user, earlier time), the same
cross-split-history design already used for notebook 08's bootstrap and confirmed correct by
notebook 06's own leakage check.

Shared between scripts/notebook06_deep_learning/run_sweep_pair.py (the detached CLI sweep
that originally produced every pair's Optuna study + checkpoint) and notebook 06's own results
cells (which reuse those same studies/checkpoints live) -- one implementation, not two copies
that could drift apart.
"""
import glob
import pathlib

import joblib
import numpy as np
import pandas as pd

from src import features, sequences, splits

ROOT = pathlib.Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT / "data" / "interim"
FDATA_DIR = ROOT / "data" / "raw" / "fdata"
PM100_PATH = ROOT / "data" / "raw" / "pm100" / "pm100_job_table.parquet"

_fdata_base_cache = None  # shared across fdata_power/duration/memory within ONE process
_pm100_base_cache = None  # shared across pm100_power/duration/memory within ONE process


def _load_fdata_base():
    global _fdata_base_cache
    if _fdata_base_cache is None:
        fdata_files = sorted(glob.glob(str(FDATA_DIR / "*.parquet")))[:38]
        df = features.load_fdata_no_embedding(fdata_files)
        df = features.filter_completed_jobs(df, "fdata")
        df = features.handle_mszl_sentinel(df)
        _fdata_base_cache = df
    return _fdata_base_cache


def _load_pm100_base():
    global _pm100_base_cache
    if _pm100_base_cache is None:
        df = pd.read_parquet(PM100_PATH)
        df = features.filter_completed_jobs(df, "pm100")
        _pm100_base_cache = df
    return _pm100_base_cache


_PREPARED_FILE = {
    "fdata_power": "fdata_power_prepared.joblib",
    "fdata_duration": "fdata_duration_r2_prepared.joblib",
    "fdata_memory": "fdata_memory_prepared.joblib",
    "pm100_power": "pm100_power_prepared.joblib",
    "pm100_duration": "pm100_duration_prepared.joblib",
    "pm100_memory_c4": "pm100_memory_clean4_prepared.joblib",
}


def load_combo(combo: str) -> dict:
    prep = joblib.load(INTERIM_DIR / _PREPARED_FILE[combo])
    X_train_df, X_test_df = prep["X_train"], prep["X_test"]
    y_train = prep["y_train"].astype(np.float32)
    y_test_log = prep["y_test"].astype(np.float32)
    y_test_raw = prep["y_test_raw"].astype(np.float64)

    # Drop absolute epoch-timestamp columns for the DL models specifically (classical ML
    # keeps them -- tree splits tolerate a monotonically-drifting feature fine). Confirmed via
    # a real failing smoke test (PM100 power/FNN): after standardizing on tune_train,
    # submit_time_epoch/eligible_time_epoch land ~2.9 std devs out of range on the test split,
    # since test is chronologically later and epoch seconds only increase -- the model has to
    # extrapolate into territory it never saw scaled values for, and test R2 came back
    # catastrophic (-3.48 log-space) despite excellent validation loss during training. This is
    # not a leakage issue (a job's own submission time is legitimately Tier A) -- it is a
    # standardized-neural-net-specific modeling issue with feeding an ever-increasing absolute
    # value through a scaler fit on an earlier time window. `time_limit` is a duration, not an
    # absolute point in time, and is kept.
    epoch_cols = [c for c in X_train_df.columns if c.endswith("_epoch")]
    if epoch_cols:
        X_train_df = X_train_df.drop(columns=epoch_cols)
        X_test_df = X_test_df.drop(columns=epoch_cols)

    # Drop the factorize()-coded categorical columns here (replaced below, once train_df/
    # test_df are reconstructed from raw data, with a one-hot encoding fit consistently on the
    # combined train+test population instead -- see the one-hot block further down).
    # build_pm100_numeric_matrix/build_fdata_numeric_matrix (src/features.py) call
    # pd.factorize() SEPARATELY on the train and test Tier A frames -- the integer code
    # assigned to a given category is just its order of first appearance WITHIN that split, so
    # the same real-world category can get a different code in train vs test (confirmed
    # directly: group_id_code's test-period values landed 22 std devs outside the
    # tune_train-fit scaler's range). Classical ML tolerates this because tree splits don't
    # weight every feature equally; a dense/sequence NN does, and is far more exposed.
    code_cols = [c for c in X_train_df.columns if c.endswith("_code")]
    if code_cols:
        X_train_df = X_train_df.drop(columns=code_cols)
        X_test_df = X_test_df.drop(columns=code_cols)

    # Drop per-split user-frequency columns for the DL models specifically -- the SAME bug
    # class as the categorical codes above, found by tracing the worst individual test-row
    # errors after the first two fixes still failed: build_fdata_numeric_matrix/
    # build_pm100_numeric_matrix compute usr_freq/user_id_freq as tier_a_df["usr"].value_counts()
    # SEPARATELY per split (src/features.py lines ~664-665, ~720-721) -- a user's test-period
    # job count has nothing to do with their train-period count, so this feature is just as
    # train/test-incompatible as the factorize codes, not a smaller version of the same
    # concern. Confirmed directly: 9 of the 10 worst-error test rows for PM100 power/FNN all
    # shared the identical scaled user_id_freq value (4.12, the single largest outlier found in
    # the earlier column-by-column scan) despite having otherwise-unremarkable other features.
    freq_cols = [c for c in X_train_df.columns if c.endswith("_freq")]
    if freq_cols:
        X_train_df = X_train_df.drop(columns=freq_cols)
        X_test_df = X_test_df.drop(columns=freq_cols)

    if combo.startswith("fdata"):
        dataset_name = "fdata"
        base = _load_fdata_base()
        if combo == "fdata_power":
            df = features.sanitize_fdata_power(base.copy())
            df = df.dropna(subset=["avgpcon"])
        else:
            df = base  # duration and memory share the shared-base split exactly
        train_df, test_df = splits.chronological_split(df, "fdata")
        assert list(train_df.index) == list(X_train_df.index), f"{combo}: train index mismatch"
        assert list(test_df.index) == list(X_test_df.index), f"{combo}: test index mismatch"
        user_col, time_col = "usr", "adt"
    else:
        dataset_name = "pm100"
        base = _load_pm100_base()
        train_df, test_df = splits.chronological_split(base, "pm100")
        assert list(train_df.index) == list(X_train_df.index), f"{combo}: train index mismatch"
        assert list(test_df.index) == list(X_test_df.index), f"{combo}: test index mismatch"
        user_col, time_col = "user_id", "submit_time"

    user_ids_full = np.concatenate([train_df[user_col].to_numpy(), test_df[user_col].to_numpy()])
    times_full = pd.concat([pd.to_datetime(train_df[time_col]),
                             pd.to_datetime(test_df[time_col])]).to_numpy()
    n_train = len(train_df)

    # One-hot encode the real (pre-factorize) categorical columns, fit on train+test COMBINED
    # so both splits get the exact same dummy columns in the exact same meaning -- replaces
    # the train/test-incompatible factorize codes dropped above. These are the raw string/
    # categorical columns straight from the already-loaded base dataframe (jobenv_req for
    # F-DATA, PM100_TIER_A_CATEGORICAL_COLUMNS for PM100) -- no extra reconstruction needed,
    # they were never factorized in `base` itself, only in the *_prepared.joblib matrices.
    cat_source_cols = ["jobenv_req"] if dataset_name == "fdata" else list(
        features.PM100_TIER_A_CATEGORICAL_COLUMNS)
    cat_source_cols = [c for c in cat_source_cols if c in train_df.columns]
    if cat_source_cols:
        combined_cat = pd.concat([train_df[cat_source_cols], test_df[cat_source_cols]],
                                  ignore_index=True)
        dummies = pd.get_dummies(combined_cat.astype(str), columns=cat_source_cols, dtype=float)
        dummies_train = dummies.iloc[:n_train].reset_index(drop=True)
        dummies_test = dummies.iloc[n_train:].reset_index(drop=True)
        X_train_df = pd.concat([X_train_df.reset_index(drop=True), dummies_train], axis=1)
        X_test_df = pd.concat([X_test_df.reset_index(drop=True), dummies_test], axis=1)

    return dict(
        dataset_name=dataset_name,
        X_train_df=X_train_df, X_test_df=X_test_df,
        y_train=y_train, y_test_log=y_test_log, y_test_raw=y_test_raw,
        user_ids_full=user_ids_full, times_full=times_full, n_train=n_train,
    )
