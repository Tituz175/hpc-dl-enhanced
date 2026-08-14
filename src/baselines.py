"""Naive/trivial baseline predictor.

Predicts each test-period job's target as that job's user's historical
median, computed from the training period only — a floor showing how
much of any model's accuracy (Roofline, RF/XGBoost/LightGBM, FNN/LSTM/TCN,
Hybrid) is genuine learning versus just reflecting a user's recent typical
behavior. Falls back to the training period's global median for any user
with no training-period history at all (a user appearing only in the test
period, or for the first time near the split boundary).
"""
import pandas as pd

_USER_COL: dict[str, str] = {"fdata": "usr", "pm100": "user_id"}


def fit_naive_baseline(
    train_df: pd.DataFrame, dataset: str, target_col: str
) -> tuple[pd.Series, float]:
    """Returns (per-user training-period median, training-period global
    median) — both computed from train_df only, never test data."""
    user_col = _USER_COL[dataset]
    user_medians = train_df.groupby(user_col)[target_col].median()
    global_median = float(train_df[target_col].median())
    return user_medians, global_median


def predict_naive_baseline(
    test_df: pd.DataFrame, dataset: str, user_medians: pd.Series, global_median: float
):
    """Per-row prediction: that row's user's training-period median target,
    or the training-period global median if the user has no training history."""
    user_col = _USER_COL[dataset]
    return test_df[user_col].map(user_medians).fillna(global_median).to_numpy()
