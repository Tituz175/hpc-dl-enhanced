"""Analytical baselines (Decision #15).

F-DATA: Hierarchical Roofline (Yang et al. 2019), computed from A64FX's
published peak specs and F-DATA's own measured flops/mbwidth/opint fields.

PM100: NOT Roofline — PM100's schema has zero FLOP/instruction/
performance-counter fields (verified directly against its
documentation/job_features.md), so no Roofline-family metric is
computable there. Instead, a calibrated resource-utilization power model
(see the module-level note below `PM100PowerModelCoefficients` for why
the form scales P_idle by num_nodes_alloc rather than applying it once
per job).

These two analytical baselines are fundamentally different kinds of
models and must never be presented as directly comparable, even
informally — footnote both wherever they appear together.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- F-DATA: Hierarchical Roofline ------------------------------------------
# A64FX peak specs (Fujitsu A64FX / Fugaku published specifications,
# boost clock 2.2GHz): double-precision peak 3.3792 TFLOP/s and HBM2 peak
# bandwidth 1024 GB/s, both PER NODE.
#
# IMPORTANT — F-DATA's flops/mbwidth fields are JOB-WIDE TOTALS, not
# per-node (verified directly, same check as the avgpcon investigation in
# features.py): only 387/713,617 jobs in a 1-file sample exceed the
# single-node peak FLOP/s when nnuma is ignored, and every one of those is
# a large multi-node job (median nnuma ~39,744); dividing flops/duration
# by nnuma, zero jobs exceed the single-node peak. So the ceiling used
# below scales both peak FLOPs and peak bandwidth by nnuma — the ridge
# point (their ratio) is unaffected by this scaling since it cancels out,
# which is also why F-DATA's own precomputed `pclass` threshold lines up
# with a single-node ridge point of ~3.3 FLOP/byte regardless of job size.

A64FX_PEAK_DP_FLOPS_PER_NODE: float = 3.3792e12  # FLOP/s
A64FX_PEAK_HBM_BW_PER_NODE: float = 1024e9  # bytes/s


def hierarchical_roofline_fdata(df: pd.DataFrame) -> pd.DataFrame:
    """Per-job Roofline diagnostics from F-DATA's measured flops/mbwidth/
    opint fields (opint == flops/mbwidth exactly, verified against the
    real data — F-DATA's authors precompute it, we don't recompute it).

    Adds: achieved_flops_per_sec, roofline_ceiling_flops_per_sec (the
    performance a job at its measured operational intensity could reach
    under a memory-bound or compute-bound ceiling, whichever binds),
    roofline_pclass (recomputed compute/memory-bound label, for
    cross-checking against F-DATA's own precomputed `pclass`), and
    roofline_predicted_duration (the analytical baseline's execution-time
    prediction: how long the job would take if it achieved the ceiling
    performance for its actual FLOP count).
    """
    out = df.copy()
    peak_flops = out["nnuma"] * A64FX_PEAK_DP_FLOPS_PER_NODE
    peak_bw = out["nnuma"] * A64FX_PEAK_HBM_BW_PER_NODE
    ridge_point = A64FX_PEAK_DP_FLOPS_PER_NODE / A64FX_PEAK_HBM_BW_PER_NODE

    out["achieved_flops_per_sec"] = out["flops"] / out["duration"].replace(0, np.nan)
    # Per-node-normalized view for plotting against a single fixed ceiling
    # curve — valid because opint (flops/mbwidth) is already node-count
    # invariant (both numerator and denominator scale by nnuma equally).
    out["achieved_flops_per_sec_per_node"] = out["achieved_flops_per_sec"] / out["nnuma"]
    memory_bound_ceiling = out["opint"] * peak_bw
    out["roofline_ceiling_flops_per_sec"] = np.minimum(peak_flops, memory_bound_ceiling)
    out["roofline_pclass"] = np.where(
        out["opint"] >= ridge_point, "compute-bound", "memory-bound"
    )
    out["roofline_predicted_duration"] = (
        out["flops"] / out["roofline_ceiling_flops_per_sec"].replace(0, np.nan)
    )
    return out


# --- PM100: calibrated resource-utilization power model ---------------------
# CORRECTED FORM (2026-07-23): node_power_consumption is a job-wide total
# (verified in notebook 03 — see features.py's correction note), not a
# per-node reading. cores_allocated/num_gpus_alloc/mem_alloc are likewise
# job-wide totals (they scale near-linearly with num_nodes_alloc). Only the
# idle baseline is a genuinely per-node physical quantity, so it must be
# scaled by num_nodes_alloc rather than applied once per job:
#
#     P_total = num_nodes_alloc * P_idle + alpha * num_cores_alloc
#                                         + beta  * num_gpus_alloc
#                                         + gamma * mem_alloc
#
# The form is fixed by hardware/physics reasoning (idle draw scales with
# node count; resource contributions scale with total resources granted);
# only P_idle/alpha/beta/gamma are calibrated, via ordinary least squares
# on a training-period-only calibration subset (never validation/test).
# Plausibility check, not a hard constraint: published TDPs put a
# Marconi100 node's two POWER9 CPUs at ~190W each and its four V100 GPUs
# at ~300W each — P_idle should land noticeably below the ~680-733W
# per-node average measured in notebook 03 (that average already includes
# whatever cores/GPUs happen to be active), and alpha/beta should be
# positive and physically small per unit (a node isn't idle-to-full-TDP
# swing from one core or one GPU).

@dataclass
class PM100PowerModelCoefficients:
    """Calibrated coefficients — fit once on a training-period-only
    calibration subset, then frozen before validation/test (Decision #15)."""
    p_idle: float
    alpha: float  # per allocated core (job-total)
    beta: float  # per allocated GPU (job-total)
    gamma: float  # per unit of allocated memory (job-total)


def calibrate_pm100_power_model(
    calibration_df: pd.DataFrame,
) -> PM100PowerModelCoefficients:
    """Calibrate P_idle/alpha/beta/gamma against measured job-total
    node_power_consumption (reduced to its per-job mean, since it's a
    20s-interval time series) on a training-period-only calibration
    subset via OLS. Must never touch validation/test data."""
    X = calibration_df[["num_nodes_alloc", "num_cores_alloc", "num_gpus_alloc", "mem_alloc"]].to_numpy(
        dtype=float
    )
    y = np.stack(calibration_df["node_power_consumption"].apply(np.mean).to_numpy())
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    p_idle, alpha, beta, gamma = reg.coef_
    return PM100PowerModelCoefficients(
        p_idle=float(p_idle), alpha=float(alpha), beta=float(beta), gamma=float(gamma)
    )


def predict_pm100_power(
    df: pd.DataFrame, coeffs: PM100PowerModelCoefficients
) -> np.ndarray:
    """P_total = num_nodes_alloc*P_idle + alpha*num_cores_alloc + beta*num_gpus_alloc + gamma*mem_alloc"""
    return (
        coeffs.p_idle * df["num_nodes_alloc"].to_numpy()
        + coeffs.alpha * df["num_cores_alloc"].to_numpy()
        + coeffs.beta * df["num_gpus_alloc"].to_numpy()
        + coeffs.gamma * df["mem_alloc"].to_numpy()
    )
