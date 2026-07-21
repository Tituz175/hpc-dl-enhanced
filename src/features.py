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
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# --- F-DATA -------------------------------------------------------------
# Targets (not features): duration (execution time), mmszu (memory used,
# per Decision #2's primary choice), avgpcon (power). minpcon/maxpcon are
# auxiliary, not the primary power target.
#
# IMPORTANT — avgpcon/minpcon/maxpcon semantics (verified against real data,
# not just the column docs): despite being named "average/min/max NODE
# power consumption", these are actually the JOB's TOTAL power (summed
# across all allocated nodes at each time-sample), then min/avg/max'd over
# the job's duration — NOT a single node's power. Verified two ways: (1)
# avgpcon correlates 0.98 with nnuma (nodes allocated); single-node jobs
# (nnuma==1) show avgpcon in [39.6, 149.9] W, matching known A64FX
# per-node power figures, while multi-node jobs scale up proportionally.
# (2) Unit cross-check: econ / (avgpcon * duration) ≈ 1/3600 consistently
# across the sample, confirming avgpcon is genuine Watts, duration is
# seconds, and econ is Watt-hours (W·s / 3600 = Wh) — the units are
# self-consistent, nothing is broken.
#
# This makes F-DATA's power target semantically DIFFERENT from PM100's
# node_power_consumption (which is genuinely per-node, per Decision #15) —
# not just a different unit, a different physical quantity (whole-job
# total vs. single-node). Footnote this alongside the existing
# Roofline-vs-power-model comparability caveat wherever F-DATA and PM100
# power numbers appear in the same table/figure.

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


# --- Failed/cancelled job exclusion (Decision #4) --------------------------
# F-DATA's "exit state" is a clean binary (completed/failed). PM100's
# job_state has 6 values (COMPLETED, FAILED, CANCELLED, TIMEOUT,
# OUT_OF_MEMORY, NODE_FAIL) — only COMPLETED reflects real workload
# behavior; the rest are artifacts of the job not finishing normally.

_COMPLETED_FILTER: dict[str, tuple[str, str]] = {
    "fdata": ("exit state", "completed"),
    "pm100": ("job_state", "COMPLETED"),
}


def filter_completed_jobs(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Exclude non-completed jobs (Decision #4). Returns a copy; also
    logs the exclusion rate since the thesis needs to report it, not just
    silently drop rows."""
    column, value = _COMPLETED_FILTER[dataset]
    kept = df[df[column] == value].copy()
    excluded_frac = 1 - len(kept) / len(df) if len(df) else 0.0
    print(f"[{dataset}] excluded {excluded_frac:.1%} of jobs as non-completed "
          f"({len(df) - len(kept)} / {len(df)})")
    return kept


# --- Embedding dimensionality reduction (Decision #7) -----------------------
# F-DATA only — PM100 has no job-name embedding field. Tree-based models
# get a handful of PCA components; FNN/LSTM/TCN can still use the full
# 384-dim embedding directly (this function doesn't replace that option,
# it's specifically for the tree-based-model feature matrix).

def reduce_fdata_embedding(df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """PCA-reduce F-DATA's 384-dim `embedding` column to `n_components`
    columns named emb_pc_0..N-1. Returns a new DataFrame with those columns
    only (index-aligned with `df`), to be concatenated onto a Tier A/B
    feature matrix in place of the raw embedding column."""
    stacked = np.stack(df["embedding"].to_numpy())
    n_components = min(n_components, stacked.shape[0], stacked.shape[1])
    components = PCA(n_components=n_components, random_state=0).fit_transform(stacked)
    columns = [f"emb_pc_{i}" for i in range(n_components)]
    return pd.DataFrame(components, columns=columns, index=df.index)


# --- Historical rolling-stat features (Tier A) ------------------------------
# The plan's example: "that user's mean duration over their last N jobs."
# Computed using only STRICTLY PAST jobs (shift(1) before rolling) so it
# can never leak the current job's own outcome — this is itself part of
# what keeps Tier A genuinely submission-time-only.
#
# With only 1 of F-DATA's 38 months present (and a small dev sample), most
# users won't have many prior jobs to compute this from yet — expect sparse/
# noisy values until more months are loaded (see EXPERIMENT_TRACKER.md).

_ROLLING_CONFIG: dict[str, dict[str, str]] = {
    "fdata": {"user_col": "usr", "time_col": "adt"},
    "pm100": {"user_col": "user_id", "time_col": "submit_time"},
}


def add_user_rolling_stat(
    df: pd.DataFrame, dataset: str, target_col: str, window: int = 5
) -> pd.DataFrame:
    """Add a `{target_col}_user_rolling_mean` column: each user's mean
    `target_col` over their previous `window` jobs (strictly before the
    current one). Returns a copy of `df` with the new column added."""
    cfg = _ROLLING_CONFIG[dataset]
    out = df.sort_values([cfg["user_col"], cfg["time_col"]]).copy()
    grouped = out.groupby(cfg["user_col"])[target_col]
    out[f"{target_col}_user_rolling_mean"] = (
        grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    return out.sort_index()


# --- Target transforms (Decision #3) ----------------------------------------
# Heavy-tailed targets (execution time, memory, power) are trained on in
# log-space; metrics get reported in both log-space and back-transformed
# real units. Kept as named functions (not inline np.log1p/np.expm1) so the
# round-trip sanity check (Decision #19, src/metrics.py) checks the exact
# functions actually used in the pipeline.

def transform_target(values: np.ndarray) -> np.ndarray:
    """log1p — requires non-negative input; targets here (time/memory/power) always are."""
    return np.log1p(values)


def inverse_transform_target(values: np.ndarray) -> np.ndarray:
    return np.expm1(values)
