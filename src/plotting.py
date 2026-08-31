"""Shared plotting functions (Track B5 Visualization Plan).

Kept here rather than duplicated per-notebook so figure style stays
consistent across F-DATA and PM100 and across the ~9 notebooks that
generate the thesis's figures.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# One fixed colour per model, reused by every seaborn call in this module so
# a given model is the same colour in every figure. "Naive" also matches the
# longer "Naive (per-user median)" label used as a results-table index.
MODEL_COLORS: dict[str, str] = {
    "Naive": "#8C8C8C",         # grey — the trivial floor, visually recessive
    "RandomForest": "#4C72B0",  # blue
    "XGBoost": "#DD8452",       # orange
    "LightGBM": "#55A868",      # green
}
_DEFAULT_COLOR = "#BBBBBB"

# Fixed panel colours for plot_feature_importance_comparison — one per
# method, the same in every call regardless of which model is shown.
SHAP_PANEL_COLOR = "#8172B3"   # muted purple
GAIN_PANEL_COLOR = "#CCB974"   # muted gold


def _palette_for(labels) -> dict:
    """Map each label to its fixed model colour; anything starting with
    'Naive' collapses to the Naive colour, unknown labels get a neutral grey."""
    out = {}
    for lab in labels:
        key = "Naive" if str(lab).startswith("Naive") else str(lab)
        out[lab] = MODEL_COLORS.get(key, _DEFAULT_COLOR)
    return out


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
    """Residual (actual − predicted) against actual, actual on a log x-axis
    since these targets span several orders of magnitude. A horizontal
    band centred on the red zero line is the ideal; systematic curvature
    or a fan opening toward large jobs shows where the model breaks down.
    alpha is low (0.1) so density is still readable at millions of points."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    pos = y_true > 0  # log x-axis needs strictly positive actuals
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_true[pos], residuals[pos], alpha=0.1, s=6, edgecolors="none")
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title(title)
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
    incompatible scales. Same model order and the module's fixed
    per-model colour in both panels so a model that wins on one metric
    and loses on the other — the recurring R²-vs-MAPE disagreement seen
    on both F-DATA and PM100 — is visible at a glance. `summary_df` is
    indexed by model name with "R2"/"MAPE" columns, the same frame
    already built for the printed results table."""
    df = summary_df.reset_index()
    model_col = df.columns[0]
    palette = _palette_for(df[model_col])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, label in [(axes[0], "R2", "R²"), (axes[1], "MAPE", "MAPE (%)")]:
        sns.barplot(data=df, x=model_col, y=col, hue=model_col, palette=palette,
                    legend=False, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.axhline(0, color="gray", linewidth=0.8)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_feature_importance_comparison(
    gain_importance: pd.Series, shap_importance: pd.Series,
    title: str = "Feature importance", top_n: int = 10,
):
    """Gain-based importance vs. SHAP mean |value|, side by side, ranked
    by SHAP — the more trustworthy metric when one feature dominates
    early splits (gain systematically starves other features' credit in
    that case, notebook 05's PM100 finding). Same feature order in both
    panels so a feature the two methods disagree about is visible
    directly. `gain_importance`/`shap_importance` are raw per-feature
    values indexed by feature name, normalised to percent of total here.
    The two panels get fixed colours (SHAP purple, gain gold) the same in
    every call; the model name, if any, lives in `title` as text only."""
    top_features = list(shap_importance.sort_values(ascending=False).head(top_n).index)
    gain_pct = gain_importance / gain_importance.sum() * 100
    shap_pct = shap_importance / shap_importance.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, series, xlabel, panel, color in [
        (axes[0], shap_pct, "% of total |SHAP|", "SHAP (mean |value|)", SHAP_PANEL_COLOR),
        (axes[1], gain_pct, "% of total gain", "Gain-based importance", GAIN_PANEL_COLOR),
    ]:
        sns.barplot(x=series.loc[top_features].to_numpy(), y=top_features,
                    color=color, ax=ax, orient="h")
        ax.set_xlabel(xlabel)
        ax.set_title(panel)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


_HIGHER_IS_BETTER = {"R2", "Within20pct"}

_HEATMAP_DEV_CAP = 15.0  # % deviation from best at which a cell is fully red


def plot_metrics_heatmap(summary_df, title, exclude_from_scale=None):
    """Heatmap of MAE/RMSE/R2/MAPE/MedAE/Within20pct across models.

    Colour encodes how far each cell is from the best value in its
    column, as a percentage: 0% (the best) is full green, ≥ 15% worse is
    full red, linear between. "Best" and the 15% span are computed only
    over the *scale-setting* rows — every row named in
    `exclude_from_scale` is still shown and still coloured (by its own %
    deviation from the scale-setting group's best), it just doesn't get a
    vote in where the scale's endpoints land. This keeps a trivial floor
    like the naive baseline, or notebook 03's Roofline, from flattening
    the colours of the models actually being compared.

    Annotation text is the real, unchanged raw value, always white. No
    colourbar — a caption states what the colour means."""
    import matplotlib as mpl

    exclude_from_scale = list(exclude_from_scale or [])
    display_df = summary_df.copy()
    scale_rows = [r for r in display_df.index if r not in exclude_from_scale]
    scale_df = display_df.loc[scale_rows] if scale_rows else display_df

    frac = pd.DataFrame(index=display_df.index, columns=display_df.columns, dtype=float)
    for col in display_df.columns:
        best = scale_df[col].max() if col in _HIGHER_IS_BETTER else scale_df[col].min()
        denom = abs(best) if abs(best) > 1e-12 else 1.0
        # Signed % deviation from the scale-setting best: positive = worse,
        # negative = better than the whole scale-setting group (only an
        # excluded row can reach this). Negative clamps to 0 -> full green;
        # beating the group is never rendered as worse than the group.
        direction = -1.0 if col in _HIGHER_IS_BETTER else 1.0
        signed_dev = direction * (display_df[col] - best) / denom * 100.0
        frac[col] = signed_dev.clip(lower=0.0, upper=_HEATMAP_DEV_CAP) / _HEATMAP_DEV_CAP
    frac = frac.clip(0.0, 1.0)

    cmap = mpl.colormaps["RdYlGn_r"]  # 0.0 -> green (best or better), 1.0 -> red (>= cap% worse)
    annot_df = display_df.round(3)
    fig, ax = plt.subplots(figsize=(1.3 * len(display_df.columns) + 1, 0.6 * len(display_df) + 1.5))
    sns.heatmap(frac, cmap=cmap, vmin=0.0, vmax=1.0, cbar=False, linewidths=0.5, ax=ax)
    for i, row in enumerate(display_df.index):
        for j, col in enumerate(display_df.columns):
            ax.text(j + 0.5, i + 0.5, f"{annot_df.loc[row, col]}",
                    ha="center", va="center", fontsize=9, color="white")
    ax.set_title(title)
    _excl = ", ".join(exclude_from_scale) if exclude_from_scale else "none"
    ax.text(0.5, -0.18,
            f"Color: % worse than the best scale-setting model in each column "
            f"(green = best, red ≥ {int(_HEATMAP_DEV_CAP)}%). Excluded from the scale: {_excl}.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, style="italic")
    fig.tight_layout()
    return fig


def _sig_range(left: float, right: float, sig: int = 2) -> str:
    """Bucket-edge label: each edge rounded to `sig` significant figures,
    formatted as a plain integer with thousands separators when it lands
    on a whole number, else compact general format."""
    def fmt(x: float) -> str:
        if not np.isfinite(x):
            return str(x)
        if x == 0:
            return "0"
        r = float(f"{x:.{sig}g}")
        return f"{int(round(r)):,}" if r == int(r) else f"{r:g}"
    return f"{fmt(left)}–{fmt(right)}"


def plot_error_boxplot(y_true: np.ndarray, models_preds: dict, title: str):
    """Absolute-error distribution per model, on a log y-axis (errors span
    orders of magnitude). Outliers are not drawn — the handful of
    huge-job errors would compress every box to a line, and
    plot_residuals already shows that tail. `models_preds` maps model
    name -> that model's raw-space prediction array, aligned with
    `y_true`. Drawn with seaborn using the module's fixed per-model
    palette."""
    y_true = np.asarray(y_true, dtype=float)
    frames = []
    for name, pred in models_preds.items():
        e = np.abs(y_true - np.asarray(pred, dtype=float))
        e = e[e > 0]  # log axis can't place exact-zero errors
        frames.append(pd.DataFrame({"model": name, "abs_error": e}))
    long = pd.concat(frames, ignore_index=True)
    order = list(models_preds)

    fig, ax = plt.subplots(figsize=(1.5 * len(order) + 2, 5))
    sns.boxplot(data=long, x="model", y="abs_error", hue="model", order=order,
                palette=_palette_for(order), legend=False, fliersize=0, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Absolute error")
    ax.set_title(title)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    fig.tight_layout()
    return fig


def plot_error_by_size_bucket(
    job_size_series, y_true: np.ndarray, y_pred: np.ndarray,
    title: str, n_buckets: int = 5, model: str | None = None,
):
    """One model's absolute error per job-size bucket, on a log y-axis.
    `job_size_series` (e.g. cores requested) is split into `n_buckets`
    equal-count bins by *rank*, not by raw value — `cnumr` / `num_cores_req`
    are so concentrated that value-based `pd.qcut` collapses to one or two
    bins, whereas ranking first always yields exactly `n_buckets` bins of
    equal size. Each bin is labelled with its real size range (edges
    rounded to 2 significant figures). Aligned by position with
    `y_true`/`y_pred`. `model` (optional) picks the box colour from the
    module's fixed palette."""
    size = np.asarray(job_size_series, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)

    ranks = pd.Series(size).rank(method="first")
    bin_idx = pd.qcut(ranks, n_buckets, labels=False).to_numpy()

    rows, order, seen = [], [], {}
    for k in range(int(bin_idx.max()) + 1):
        m = bin_idx == k
        lab = _sig_range(size[m].min(), size[m].max())
        if lab in seen:  # adjacent rank-bins covering an identical size range
            seen[lab] += 1
            lab = f"{lab} [{seen[lab]}]"
        else:
            seen[lab] = 1
        e = abs_err[m]
        rows.append(pd.DataFrame({"bucket": lab, "abs_error": e[e > 0]}))
        order.append(lab)
    long = pd.concat(rows, ignore_index=True)
    color = _palette_for([model])[model] if model is not None else MODEL_COLORS["LightGBM"]

    fig, ax = plt.subplots(figsize=(1.6 * len(order) + 2, 5))
    sns.boxplot(data=long, x="bucket", y="abs_error", order=order,
                color=color, fliersize=0, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Job size bucket (quantile bins)")
    ax.set_ylabel("Absolute error")
    ax.set_title(title)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    fig.tight_layout()
    return fig
