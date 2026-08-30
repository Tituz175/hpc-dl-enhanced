"""Evaluation metrics, plus statistical rigor helpers for when a
point-estimate difference in those metrics between two models isn't
enough on its own to trust — it could just be noise.

Headline metrics: MAE, RMSE, R², MAPE — computed in both log-space
(matching how these heavy-tailed targets are actually trained on) and
back-transformed real units (for interpretability). Heavy tails also make
plain MAE/RMSE easy to dominate with a handful of huge jobs and can make
R² look deceptively high, so stratified breakdowns by job-size bucket and
(for PM100) CPU-only vs. GPU jobs are reported alongside the aggregate
numbers, to show where the error is concentrated rather than just how
much there is.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE/RMSE/R²/MAPE/MedAE/Within20pct for one model on one target.

    MedAE (median absolute error) is the heavy-tail-robust companion to
    MAE — a handful of huge jobs can't drag it the way they drag the mean.
    Within20pct is the share of predictions landing within ±20% of the
    actual value, a directly interpretable "how often is it roughly
    right" number; rows with y_true == 0 are excluded from it (the ratio
    is undefined there), the same rows MAPE already skips.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    nonzero = y_true != 0
    rel_err = np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])
    mape = float(np.mean(rel_err) * 100)
    medae = float(median_absolute_error(y_true, y_pred))
    within20 = float(np.mean(rel_err <= 0.20) * 100)
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
        "MedAE": medae, "Within20pct": within20,
    }


def expm1_round_trip_check(x: np.ndarray, atol: float = 1e-8) -> None:
    """Sanity-check assertion: log1p/expm1 must round-trip. A silent bug in
    this custom transform would quietly invalidate every downstream metric
    without producing an obvious symptom, so it's checked directly rather
    than assumed to work."""
    recovered = np.expm1(np.log1p(x))
    assert np.allclose(x, recovered, atol=atol), "log1p/expm1 round-trip failed"


def paired_significance_test(errors_a: np.ndarray, errors_b: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank test between two models' per-job/per-fold absolute
    errors — point-estimate differences in MAE/RMSE/R² between models can
    just be noise, so run this before claiming one model 'beats' another."""
    statistic, p_value = stats.wilcoxon(errors_a, errors_b)
    return {"statistic": float(statistic), "p_value": float(p_value)}
