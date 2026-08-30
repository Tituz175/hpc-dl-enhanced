"""Shared plotting functions (Track B5 Visualization Plan).

Kept here rather than duplicated per-notebook so figure style stays
consistent across F-DATA and PM100 and across the ~9 notebooks that
generate the thesis's figures.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_target_distribution(raw: np.ndarray, log_transformed: np.ndarray, target_name: str):
    """Raw vs. log1p-transformed target distribution, side by side — meant to
    make the heavy right tail (many short/small jobs, a handful of huge
    ones) visible directly, which is why these targets get trained on in
    log-space rather than raw units."""
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


def plot_roofline(
    opint: np.ndarray,
    achieved_flops_per_node: np.ndarray,
    peak_flops_per_node: float,
    peak_bw_per_node: float,
    pclass=None,
    title: str = "F-DATA Roofline (A64FX, per-node)",
):
    """Arithmetic-intensity-vs-performance scatter with the roofline
    ceiling line and shaded compute/memory-bound regions (Track B5).
    Both axes are per-node quantities — opint is node-count invariant by
    construction, and achieved performance should be pre-divided by nnuma
    (see roofline.hierarchical_roofline_fdata) so every job plots against
    the same single-node ceiling regardless of how many nodes it used."""
    ridge_point = peak_flops_per_node / peak_bw_per_node
    fig, ax = plt.subplots(figsize=(7, 6))
    valid = (opint > 0) & (achieved_flops_per_node > 0)
    if pclass is not None:
        for label, color in [("compute-bound", "tab:orange"), ("memory-bound", "tab:blue")]:
            mask = valid & (pclass == label)
            ax.scatter(opint[mask], achieved_flops_per_node[mask], alpha=0.15, s=5,
                       color=color, label=label)
    else:
        ax.scatter(opint[valid], achieved_flops_per_node[valid], alpha=0.15, s=5)

    x = np.logspace(np.log10(opint[valid].min()), np.log10(opint[valid].max()), 200)
    ceiling = np.minimum(peak_flops_per_node, x * peak_bw_per_node)
    ax.plot(x, ceiling, "r-", linewidth=2, label="Roofline ceiling")
    ax.axvline(ridge_point, color="gray", linestyle="--", linewidth=1, label="Ridge point")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational intensity (FLOP/byte)")
    ax.set_ylabel("Performance (FLOP/s, per node)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_model_comparison(summary_df: pd.DataFrame, title: str = "Model comparison"):
    """R²/MAPE side by side across models (notebook 05) — two panels
    rather than one chart, since R² (unitless, roughly [-1, 1]) and MAPE
    (a percentage, often well over 100% for heavy-tailed targets) sit on
    incompatible scales. Same model order in both panels so a model that
    wins on one metric and loses on the other — the recurring R²-vs-MAPE
    disagreement seen on both F-DATA and PM100 — is visible at a glance
    rather than hidden by picking a single metric to plot. `summary_df`
    is indexed by model name with "R2"/"MAPE" columns, the same frame
    already built for the printed results table."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(summary_df))
    for ax, col, label in [(axes[0], "R2", "R²"), (axes[1], "MAPE", "MAPE (%)")]:
        ax.bar(x, summary_df[col])
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df.index, rotation=20, ha="right")
        ax.set_title(label)
        ax.axhline(0, color="gray", linewidth=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_feature_importance_comparison(
    gain_importance: pd.Series, shap_importance: pd.Series,
    title: str = "Feature importance", top_n: int = 10,
):
    """Gain-based feature_importances_ vs. SHAP mean |value|, side by
    side, ranked by SHAP — the more trustworthy metric when one feature
    dominates early splits (gain systematically starves other features'
    credit in that case, notebook 05's PM100 finding). Same feature
    order in both panels so a feature the two methods disagree about is
    visible directly, not hidden by sorting each panel independently.
    `gain_importance`/`shap_importance` are raw (not pre-normalized)
    per-feature values indexed by feature name — normalized to percent
    of total here."""
    top_features = shap_importance.sort_values(ascending=False).head(top_n).index
    gain_pct = gain_importance / gain_importance.sum() * 100
    shap_pct = shap_importance / shap_importance.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    y = np.arange(len(top_features))
    axes[0].barh(y, shap_pct.loc[top_features])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(top_features)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("% of total |SHAP|")
    axes[0].set_title("SHAP (mean |value|)")

    axes[1].barh(y, gain_pct.loc[top_features])
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(top_features)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("% of total gain")
    axes[1].set_title("Gain-based importance")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_metrics_heatmap(summary_df, title):
    """Heatmap of MAE/RMSE/R2/MAPE across models, each column normalized
    independently -- these metrics live on completely different scales
    (raw seconds/Watts, unitless R2, percent), so a shared color scale
    would be meaningless. Green = best-in-column, not best overall."""
    display_df = summary_df.copy()
    normed = display_df.copy()
    for col in display_df.columns:
        col_min, col_max = display_df[col].min(), display_df[col].max()
        span = col_max - col_min + 1e-12
        if col == "R2":  # higher is better
            normed[col] = (display_df[col] - col_min) / span
        else:  # MAE/RMSE/MAPE -- lower is better, so invert
            normed[col] = 1 - (display_df[col] - col_min) / span

    fig, ax = plt.subplots(figsize=(6, 0.6 * len(display_df) + 1.5))
    sns.heatmap(
        normed, annot=display_df.round(3), fmt="", cmap="RdYlGn",
        cbar_kws={"label": "Relative performance (green=better, per column)"},
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig
