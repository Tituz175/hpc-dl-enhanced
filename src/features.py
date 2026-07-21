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

Column classifications below are grounded in the official documentation:
- F-DATA: https://github.com/francescoantici/F-DATA/blob/main/docs/feature_list.md
- PM100:  https://github.com/francescoantici/PM100-data (documentation/job_features.md)
not guessed from abbreviated names.
"""
import pandas as pd

# --- F-DATA -------------------------------------------------------------
# Targets (not features): duration (execution time), mmszu (memory used,
# per Decision #2's primary choice), avgpcon (power — average node power
# consumption). minpcon/maxpcon are auxiliary, not the primary power target.

FDATA_TIER_A_COLUMNS: list[str] = [
    "usr",          # username — for historical rolling stats
    "jnam",         # job name
    "cnumr",        # cores requested
    "nnumr",        # nodes requested
    "adt",          # arrival/submission datetime
    "qdt",          # time of insertion in job queue (pre-execution)
    "schedsdt",     # time of completed scheduling choice (still pre-execution)
    "elpl",         # elapsed time limit requested
    "mszl",         # memory size limit requested
    "pri",          # priority
    "jobenv_req",   # job environment requested
    "freq_req",     # node frequency requested
    "embedding",    # SBert encoding of job name/sensitive data (Decision #7)
]

FDATA_TIER_B_COLUMNS: list[str] = [
    "cnumat", "cnumut",             # cores allocated / used
    "deldt",                        # job deletion time
    "ec",                           # exit code
    "sdt", "edt",                   # start / end datetime
    "nnuma", "nnumu",               # nodes allocated / used
    "idle_time_ave",                # average idle time
    "perf1", "perf2", "perf3", "perf4", "perf5", "perf6",  # HW performance counters
    "econ",                         # energy consumption
    "avgpcon", "minpcon", "maxpcon",  # power consumption (avgpcon is the power target)
    "msza", "mmszu",                # memory allocated / used (mmszu is the memory target)
    "uctmut", "sctmut", "usctmut",  # CPU time used
    "freq_alloc",                   # node frequency actually allocated
    "flops", "mbwidth", "opint", "pclass",  # Roofline inputs — measured, post-hoc only
    "exit state",                   # completed / failed
]

FDATA_TARGETS: dict[str, str] = {
    "execution_time": "duration",
    "memory": "mmszu",   # used memory (Decision #2); fall back to msza (allocated) if null
    "power": "avgpcon",
}

# --- PM100 ----------------------------------------------------------------
# No "used" memory field exists at all (only requested/allocated) — so
# PM100's memory target is necessarily mem_alloc, the Decision #2 fallback
# case, not a choice. No FLOP/performance-counter fields exist (verified
# directly against both the docs and the actual parquet schema), so PM100
# has no Tier B Roofline-relevant columns — its Tier B is just the
# runtime/actuals needed for post-hoc characterization and Hybrid
# residual learning on the Power target (Decision #15).

PM100_TIER_A_COLUMNS: list[str] = [
    "user_id", "group_id", "partition", "qos", "priority",
    "submit_time", "eligible_time", "time_limit",
    "num_cores_req", "num_nodes_req", "num_gpus_req", "mem_req",
    "cores_per_task", "num_tasks", "threads_per_core", "shared",
    "req_nodes", "req_switch",
]

PM100_TIER_B_COLUMNS: list[str] = [
    "start_time", "end_time",       # run_time (target) is derived from these
    "job_state", "state_reason", "derived_ec",
    "cores_allocated", "cores_alloc_layout",
    "num_cores_alloc", "num_nodes_alloc", "num_gpus_alloc",
    "mem_alloc", "nodes",
    "node_power_consumption",       # power target (time series, Decision #13's LSTM/TCN sequence)
    "mem_power_consumption", "cpu_power_consumption",
]

PM100_TARGETS: dict[str, str] = {
    "execution_time": "run_time",
    "memory": "mem_alloc",   # no "used" field exists for PM100 — always the fallback case
    "power": "node_power_consumption",
}

# num_gpus_alloc is Tier B (not requested), but also used as the CPU-only
# vs. GPU stratification variable for evaluation slicing (Decision #8) —
# that's a reporting/grouping use, not a predictive-feature use, so it
# doesn't need Tier A/B treatment for that purpose.


def _assert_no_overlap(tier_a: list[str], tier_b: list[str]) -> None:
    overlap = set(tier_a) & set(tier_b)
    assert not overlap, f"Column(s) listed in both tiers: {overlap}"


_assert_no_overlap(FDATA_TIER_A_COLUMNS, FDATA_TIER_B_COLUMNS)
_assert_no_overlap(PM100_TIER_A_COLUMNS, PM100_TIER_B_COLUMNS)


def build_tier_a_features(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Submission-time-only feature matrix. Valid for RQ1/H1 headline results."""
    columns = FDATA_TIER_A_COLUMNS if dataset == "fdata" else PM100_TIER_A_COLUMNS
    return df[columns]


def build_tier_b_features(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Execution-time/post-hoc feature matrix. For characterization and Hybrid only."""
    columns = FDATA_TIER_B_COLUMNS if dataset == "fdata" else PM100_TIER_B_COLUMNS
    return df[columns]


def assert_no_tier_leakage(feature_columns: list[str], tier: str, dataset: str) -> None:
    """Sanity-check assertion (Decision #19): fail loudly if a Tier B column
    ends up in a Tier A feature matrix, or vice versa."""
    if dataset == "fdata":
        tier_a, tier_b = FDATA_TIER_A_COLUMNS, FDATA_TIER_B_COLUMNS
    else:
        tier_a, tier_b = PM100_TIER_A_COLUMNS, PM100_TIER_B_COLUMNS
    other = tier_b if tier == "A" else tier_a
    leaked = set(feature_columns) & set(other)
    assert not leaked, f"Tier leakage detected in Tier {tier} features ({dataset}): {leaked}"
