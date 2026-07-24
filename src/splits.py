"""Chronological train/test split (Shared Context: train on earlier time,
test on later time, for both datasets, to avoid temporal leakage).

Single source of truth for the split boundary so every notebook from 03
onward (calibration, baselines, classical ML, DL, evaluation) uses the
same boundary rather than each notebook picking its own — a different
boundary per notebook would make cross-notebook comparisons invalid
without anyone noticing.
"""
import pandas as pd

_TIME_COL: dict[str, str] = {"fdata": "adt", "pm100": "submit_time"}


def split_boundary(df: pd.DataFrame, dataset: str, train_frac: float = 0.7) -> pd.Timestamp:
    """Timestamp such that `train_frac` of rows (by row count) fall at or
    before it. Uses submission/arrival time, since that's what's known at
    decision time for both datasets (not start/end time, which are Tier B)."""
    times = pd.to_datetime(df[_TIME_COL[dataset]])
    return times.quantile(train_frac)


def chronological_split(
    df: pd.DataFrame, dataset: str, train_frac: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (train_df, test_df) split at the train_frac boundary."""
    times = pd.to_datetime(df[_TIME_COL[dataset]])
    boundary = times.quantile(train_frac)
    train_mask = times <= boundary
    return df[train_mask].copy(), df[~train_mask].copy()
