"""
Feature standardization for notebook 06's deep learning models (FNN/LSTM/TCN). Not used by
the classical ML models (05/05b/05c) -- tree ensembles are scale-invariant, so this was never
needed there. It IS needed here: confirmed directly (not assumed) that no standardization was
ever applied anywhere in the DL pipeline up to this point -- FlatDataset/SequenceDataset cast
the raw Tier A feature matrix straight to float32 with no transform, meaning columns ranging
from 0/1 flags to multi-billion-scale epoch timestamps and byte counts were fed to the network
side by side, unscaled. This is a strong, plausible root cause for both the TCN divergence and
LSTM's weak plateau seen in earlier diagnostics: a single huge-magnitude input directly
inflates the first layer's pre-activation regardless of gradient clipping or learning rate.

fit_scaler is fit on a combination's TRAINING split only, applied identically (same fitted
parameters) to validation and test -- the same train/val/test parameter-leakage discipline as
every other transform in this project (chronological_split, target transforms, etc.).

Per-column treatment, decided per column from the TRAINING data itself (not fixed by name) --
a column gets a signed-log1p pass before standardization only if it is both heavy-tailed
(|skew| above SKEW_THRESHOLD) and not a near-binary/categorical-code column (more than
MIN_UNIQUE_FOR_LOG distinct training values) -- this reliably catches genuine continuous
heavy-tailed quantities (byte counts, memory sizes, user-frequency counts, rolling-mean
durations) without distorting 0/1 flags or small integer category codes, and does not assume a
column is heavy-tailed just because of its name (e.g. F-DATA's epoch-timestamp columns are
large in magnitude but not actually skewed within a dataset's time span, so they are left to
plain standardization, not log-transformed, if that is what their real distribution shows).
signed log1p (sign(x) * log1p(|x|)), not plain log1p, since some feature columns can be
negative (e.g. a difference-of-counts feature), unlike the non-negative targets log1p alone is
used for elsewhere in this project.
"""
import numpy as np
import pandas as pd

SKEW_THRESHOLD = 1.0
MIN_UNIQUE_FOR_LOG = 10


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def fit_scaler(X_train: pd.DataFrame) -> dict:
    """Returns {column_name: {use_log, mean, std, skew, is_binary}}, in X_train's own column
    order. Binary (0/1) columns -- one-hot/dummy features -- are left UNSCALED (mean=0,
    std=1, use_log=False, an identity transform): found directly, not assumed, after one-hot
    encoding the categorical columns elsewhere in this pipeline and standardizing them the
    same way as every other column -- a category that's rare in tune_train gets a tiny fitted
    std, so a single 1 anywhere else (a more common occurrence of that category in a different
    split) produces an enormous z-score (confirmed: one real PM100 one-hot column reached
    129 std devs on the test split). Standardizing a binary indicator is not standard practice
    for exactly this reason; left as a plain 0/1 instead."""
    params = {}
    for col in X_train.columns:
        vals = X_train[col].to_numpy(dtype=float)
        unique_vals = np.unique(vals)
        is_binary = set(unique_vals.tolist()) <= {0.0, 1.0}
        if is_binary:
            params[col] = dict(use_log=False, mean=0.0, std=1.0, skew=0.0,
                                n_unique=len(unique_vals), is_binary=True)
            continue
        n_unique = len(unique_vals)
        skew = float(pd.Series(vals).skew())
        use_log = n_unique > MIN_UNIQUE_FOR_LOG and abs(skew) > SKEW_THRESHOLD
        transformed = _signed_log1p(vals) if use_log else vals
        mean = float(transformed.mean())
        std = float(transformed.std())
        if std < 1e-8:
            std = 1.0  # constant (or near-constant) column guard -- avoid divide-by-zero;
            # mean-centering alone is harmless for a column with ~zero variance
        params[col] = dict(use_log=use_log, mean=mean, std=std, skew=skew, n_unique=n_unique,
                            is_binary=False)
    return params


def apply_scaler(X: pd.DataFrame, params: dict) -> np.ndarray:
    """Applies a fitted scaler to X (any split). Returns a float32 (n_rows, n_cols) array in
    params' column order -- asserts X's columns match exactly, so a silent column-order
    mismatch can't silently mis-scale the wrong column."""
    assert list(X.columns) == list(params.keys()), \
        "X's columns don't match the fitted scaler's columns/order"
    out = np.empty((len(X), len(params)), dtype=np.float32)
    for i, col in enumerate(params):
        vals = X[col].to_numpy(dtype=float)
        p = params[col]
        if p["use_log"]:
            vals = _signed_log1p(vals)
        out[:, i] = (vals - p["mean"]) / p["std"]
    return out


def summarize_scaler(params: dict) -> pd.DataFrame:
    """For reporting/disclosure: which columns got log-transformed and why."""
    return pd.DataFrame(params).T.sort_values("skew", key=abs, ascending=False)
