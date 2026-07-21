"""Tier A/B feature-set builders (Decision #1).

Tier A — submission-time features (known before the job runs): the only
tier valid for a genuine pre-execution prediction claim, and what RQ1/H1
are evaluated on as the headline result.

Tier B — execution-time/post-hoc features (measured FLOPs, actual power,
performance counters): valid for characterization (Roofline analysis)
and as input to the Hybrid residual model, never as a scheduling predictor.

Keeping these as two separate builder functions (rather than one function
with a flag) is the point — it makes it structurally hard for an
experiment to accidentally mix tiers.
"""
import pandas as pd

TIER_A_COLUMNS: list[str] = [
    # requested cores/nodes/memory, requested walltime, queue,
    # node frequency requested, job-name embedding, user/app historical
    # rolling stats — finalize exact column names against the real
    # F-DATA/PM100 schemas during notebook 02.
]

TIER_B_COLUMNS: list[str] = [
    # measured FLOPs, actual power, performance counters —
    # finalize against the real schemas during notebook 02.
]


def build_tier_a_features(df: pd.DataFrame) -> pd.DataFrame:
    """Submission-time-only feature matrix. Valid for RQ1/H1 headline results."""
    raise NotImplementedError("Implement once F-DATA/PM100 schemas are loaded (notebook 02)")


def build_tier_b_features(df: pd.DataFrame) -> pd.DataFrame:
    """Execution-time/post-hoc feature matrix. For characterization and Hybrid only."""
    raise NotImplementedError("Implement once F-DATA/PM100 schemas are loaded (notebook 02)")


def assert_no_tier_leakage(feature_columns: list[str], tier: str) -> None:
    """Sanity-check assertion (Decision #19): fail loudly if a Tier B column
    ends up in a Tier A feature matrix, or vice versa."""
    other = TIER_B_COLUMNS if tier == "A" else TIER_A_COLUMNS
    leaked = set(feature_columns) & set(other)
    assert not leaked, f"Tier leakage detected in Tier {tier} features: {leaked}"
