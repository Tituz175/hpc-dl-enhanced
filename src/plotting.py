"""Shared plotting functions (Track B5 Visualization Plan).

Kept here rather than duplicated per-notebook so figure style stays
consistent across F-DATA and PM100 and across the ~9 notebooks that
generate the thesis's figures.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_target_distribution(raw: np.ndarray, log_transformed: np.ndarray, target_name: str):
    """Raw vs. log1p-transformed target distribution (motivates Decision #3)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(raw, bins=50)
    axes[0].set_title(f"{target_name} (raw)")
    axes[1].hist(log_transformed, bins=50)
    axes[1].set_title(f"{target_name} (log1p)")
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    """Regression analog of a confusion matrix (Track B5)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    """Residual-vs-predicted + residual histogram (Track B5)."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=8)
    axes[0].axhline(0, color="r", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[1].hist(residuals, bins=50)
    axes[1].set_xlabel("Residual")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
