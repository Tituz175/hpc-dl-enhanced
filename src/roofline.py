"""Analytical baselines (Decision #15).

F-DATA: Hierarchical Roofline (Yang et al. 2019), computed from A64FX's
own memory hierarchy and the real FLOP/performance-counter fields F-DATA
provides.

PM100: NOT Roofline — PM100's schema has zero FLOP/instruction/
performance-counter fields (verified directly against its
documentation/job_features.md), so no Roofline-family metric is
computable there. Instead, a calibrated resource-utilization power model:

    P_estimated = P_idle + alpha * cores_allocated
                          + beta  * gpus_allocated
                          + gamma * mem_alloc

The form is fixed by hardware/physics reasoning; only the coefficients
are calibrated (from published POWER9/V100 TDP specs and/or a
training-period-only calibration subset — never validation/test).

These two analytical baselines are fundamentally different kinds of
models and must never be presented as directly comparable, even
informally — footnote both wherever they appear together.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


def hierarchical_roofline_fdata(df: pd.DataFrame) -> pd.DataFrame:
    """Arithmetic intensity + performance ceiling per job, from F-DATA's
    real FLOP/memory-bandwidth fields. Implement once F-DATA schema is
    loaded (notebook 03)."""
    raise NotImplementedError("Implement once F-DATA schema is loaded (notebook 03)")


@dataclass
class PM100PowerModelCoefficients:
    """Calibrated coefficients — fit once on a training-period-only
    calibration subset, then frozen before validation/test (Decision #15)."""
    p_idle: float
    alpha: float  # per allocated core
    beta: float  # per allocated GPU
    gamma: float  # per unit of allocated memory


def calibrate_pm100_power_model(
    calibration_df: pd.DataFrame,
) -> PM100PowerModelCoefficients:
    """Calibrate P_idle/alpha/beta/gamma against measured
    node_power_consumption on a training-period-only calibration subset.
    Must never touch validation/test data. Implement once PM100 is
    loaded (notebook 03)."""
    raise NotImplementedError("Implement once PM100 is loaded (notebook 03)")


def predict_pm100_power(
    df: pd.DataFrame, coeffs: PM100PowerModelCoefficients
) -> np.ndarray:
    """P_estimated = P_idle + alpha*cores_allocated + beta*gpus_allocated + gamma*mem_alloc"""
    return (
        coeffs.p_idle
        + coeffs.alpha * df["cores_allocated"].to_numpy()
        + coeffs.beta * df["num_gpus_alloc"].to_numpy()
        + coeffs.gamma * df["mem_alloc"].to_numpy()
    )
