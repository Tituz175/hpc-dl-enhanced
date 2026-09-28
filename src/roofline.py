"""Analytical baselines.

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
    calibration subset, then frozen before validation/test, so this stays
    a genuine analytical baseline with only its coefficients calibrated to
    data, not a disguised end-to-end regression."""
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


# --- F-DATA: calibrated resource-utilization power model --------------------
# A direct correlation check (EXPERIMENT_TRACKER.md, 2026-09-21) found avgpcon correlates
# 0.9990 with nnuma and 0.9993 with cnumat on a real check, an even
# stronger starting relationship than PM100's power model had — but nnuma
# and cnumat are themselves correlated 0.9996 with each other, tighter
# than PM100's four predictors ever were (0.85-0.97), because cnumat is
# essentially nnuma times A64FX's fixed 48-cores-per-node ratio: the ratio
# cnumat/nnuma sits at exactly 48.0 for 98.97% of jobs in a real check.
# cnumat is therefore almost entirely redundant with nnuma at this
# dataset's dominant configuration, not an independent data-quality
# problem the way msza's sentinel is, but functionally the same
# disqualifying property for a joint OLS fit: cnumat is dropped, nnuma
# kept, because nnuma is also the clearer, more direct hardware quantity
# (it is the field already cross-validated by the Roofline ceiling check
# above, and matches nnumr, the requested node count, 98.97% of the time,
# always at or above it — well-behaved allocation semantics; cnumat's own
# match rate against cnumr, 99.999%, does not distinguish the two on data
# quality, only on redundancy). No GPU term, since F-DATA is CPU-only.
#
#     P_total = nnuma * P_idle + gamma * msza_sanitized_GB
#
# msza (memory allocated, Tier B) needs features.py's handle_msza_sentinel
# applied first — it carries the same 2**64-1 "no allocation recorded"
# sentinel mszl does. Verified directly against all 38 raw F-DATA files:
# mszl and msza are identical on every row of every file (both sentinel
# and non-sentinel values alike, checked separately). Full-scale sentinel
# rate (raw rows, all 38 months): 88.55%, ranging 59.3%-99.8% by file.
# Dropping cnumat here is an OLS-identifiability fix only; it does not
# disqualify cnumat as a classical-ML Tier A/B feature candidate (see
# the Tier A/B feature-vetting pass documented in EXPERIMENT_TRACKER.md).
#
# UNIT-SCALE FINDING (2026-09-21, found by fitting on the real
# full-scale data, not assumed): msza in raw bytes (up to ~3.0e10) next to
# nnuma (up to 1.6e5) makes the design matrix so ill-conditioned that
# sklearn's no-intercept OLS solver returns a near-zero garbage solution
# for BOTH coefficients (p_idle ~1.8e-16) rather than erroring — a silent
# numerical failure, not a modeling one. Rescaling msza to gigabytes
# before fitting fixes it: nnuma's coefficient recovers to ~108.0 (matching
# a single-predictor nnuma-only fit almost exactly, train R^2 0.993 either
# way), confirming the two-column version was a units bug, not evidence
# msza has no room to matter. Once fixed, msza's OWN full-population
# correlation with avgpcon is only -0.007 (essentially zero) — unlike
# PM100, where all four predictors correlated with power, F-DATA's power
# draw is overwhelmingly a function of node count alone, and memory
# allocation isn't expected to add real explanatory power on top of it.
FDATA_MSZA_BYTES_PER_GB: float = 1e9


@dataclass
class FDataPowerModelCoefficients:
    """Calibrated coefficients for F-DATA's power model — same discipline
    as PM100PowerModelCoefficients: fit once on a training-period-only
    calibration subset, then frozen before validation/test."""
    p_idle: float
    gamma: float  # per GB of allocated memory (job-total, msza_sanitized)


def calibrate_fdata_power_model(
    calibration_df: pd.DataFrame,
) -> FDataPowerModelCoefficients:
    """Calibrate P_idle/gamma against avgpcon (already run through
    sanitize_fdata_power) on a training-period-only calibration subset via
    OLS. calibration_df must already have handle_msza_sentinel applied.
    msza is rescaled from bytes to gigabytes internally — see the
    UNIT-SCALE FINDING note above; fitting on raw bytes silently produces
    a garbage near-zero solution. Must never touch validation/test data."""
    msza_gb = calibration_df["msza"].to_numpy(dtype=float) / FDATA_MSZA_BYTES_PER_GB
    X = np.column_stack([calibration_df["nnuma"].to_numpy(dtype=float), msza_gb])
    y = calibration_df["avgpcon"].to_numpy(dtype=float)
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    p_idle, gamma = reg.coef_
    return FDataPowerModelCoefficients(p_idle=float(p_idle), gamma=float(gamma))


def predict_fdata_power(
    df: pd.DataFrame, coeffs: FDataPowerModelCoefficients
) -> np.ndarray:
    """P_total = nnuma*P_idle + gamma*msza_gb (msza already sentinel-sanitized;
    rescaled bytes->GB here to match calibrate_fdata_power_model)."""
    msza_gb = df["msza"].to_numpy(dtype=float) / FDATA_MSZA_BYTES_PER_GB
    return (
        coeffs.p_idle * df["nnuma"].to_numpy(dtype=float)
        + coeffs.gamma * msza_gb
    )
