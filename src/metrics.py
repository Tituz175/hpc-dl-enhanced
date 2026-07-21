"""Evaluation metrics (Decision #8) and statistical rigor helpers (Decision #6).

Headline metrics: MAE, RMSE, R², MAPE — computed in both log-space and
back-transformed real units (Decision #3), plus stratified breakdowns by
job-size bucket and (for PM100) CPU-only vs. GPU jobs.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE/RMSE/R²/MAPE for one model on one target."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def expm1_round_trip_check(x: np.ndarray, atol: float = 1e-8) -> None:
    """Sanity-check assertion (Decision #19): log1p/expm1 must round-trip."""
    recovered = np.expm1(np.log1p(x))
    assert np.allclose(x, recovered, atol=atol), "log1p/expm1 round-trip failed"


def paired_significance_test(errors_a: np.ndarray, errors_b: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank test between two models' per-job/per-fold absolute
    errors (Decision #6) — use before claiming one model 'beats' another."""
    statistic, p_value = stats.wilcoxon(errors_a, errors_b)
    return {"statistic": float(statistic), "p_value": float(p_value)}
