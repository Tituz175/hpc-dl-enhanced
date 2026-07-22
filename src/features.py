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
    "cores_per_task", "num_tasks", "shared", "req_switch",
    "num_tasks_missing",  # promoted after feature vetting — see below
]

# --- PM100 missing-by-design fields (feature vetting) -----------------------
# Discovered during EDA: num_tasks (4.1% missing), req_nodes (88.6%
# missing), threads_per_core (99.6% missing) have very different
# missingness mechanisms — verified against the FULL PM100 dataset, not a
# sample (see notebook 02's "Feature Vetting" section for the complete
# analysis with visualizations: effect sizes, redundancy cross-tabs, and
# user-concentration checks).
#
# - req_nodes: Slurm's explicit node-pinning flag (--nodelist) — null by
#   design for the ~88.6% of jobs that don't request specific named
#   hardware. Its derived flag (node_pinned) shows a real effect on
#   targets (power ~35% lower when pinned) BUT is concentrated in just 19
#   users, with the top 5 accounting for 99.3% of all pinned jobs (73.5%
#   from a single user) — too concentrated to trust as a generalizable
#   Tier A feature. RESERVED, not active.
# - threads_per_core: Slurm's explicit SMT/hyperthreading override — null
#   by design for ~99.6% of jobs. Its derived flag (threads_per_core_set)
#   shows an even larger effect (run_time ~4x higher, memory ~4x lower)
#   but is concentrated in only 6 users (85% from a single user), and
#   isn't explained by partition/qos/GPU-allocation (checked and ruled
#   out as redundant with those). Same verdict despite the strong effect
#   size: RESERVED, not active.
# - num_tasks: ordinary incomplete logging (~4.1% missing), NOT an
#   optional Slurm flag — no single dominant user. Its derived
#   missingness flag (num_tasks_missing) shows a real effect (run_time
#   ~2x higher when missing) AND is reasonably distributed across 101
#   users (top-5 share = 47.1%, not dominated by one) — PROMOTED to
#   active Tier A.
#
# Both raw columns (req_nodes, threads_per_core) and all three derived
# flags are preserved in the processed dataset regardless of this split —
# only num_tasks_missing is added to the active PM100_TIER_A_COLUMNS list.
# This concentration check generalizes: both datasets are dominated by a
# handful of power users (PM100's top-5 users = 45.4% of all jobs, top-10
# = 59.9%; F-DATA's top-5 = 41.4% in a 5-file sample) — worth the same
# scrutiny for any future derived feature, not just these three.

PM100_RESERVED_COLUMNS: list[str] = ["req_nodes", "threads_per_core", "node_pinned", "threads_per_core_set"]


def add_pm100_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Binary indicators for PM100's missing-by-design/incomplete fields.
    Returns a copy of df with node_pinned, threads_per_core_set, and
    num_tasks_missing added. Only num_tasks_missing belongs in the active
    Tier A feature set — node_pinned and threads_per_core_set are
    computed and preserved but reserved (see module comment above)."""
    out = df.copy()
    out["node_pinned"] = out["req_nodes"].notna()
    out["threads_per_core_set"] = out["threads_per_core"].notna()
    out["num_tasks_missing"] = out["num_tasks"].isna()
    return out


def assert_reserved_columns_preserved(df: pd.DataFrame) -> None:
    """Sanity-check assertion (Decision #19 discipline): reserved columns
    must exist in the processed dataframe (not silently dropped by a
    future refactor) and must NOT appear in the active Tier A feature
    list (not silently promoted back in without re-running the vetting)."""
    for col in PM100_RESERVED_COLUMNS:
        assert col in df.columns, (
            f"Reserved column '{col}' missing from dataframe — it must be "
            "preserved (not dropped), even though it's excluded from active Tier A."
        )
        assert col not in PM100_TIER_A_COLUMNS, (
            f"Reserved column '{col}' leaked into active PM100_TIER_A_COLUMNS "
            "— this was deliberately excluded after feature vetting found it "
            "too concentrated in a handful of users to generalize."
        )

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
# F-DATA only — PM100 has no job-name embedding field.
#
# Option C (confirmed): the PCA-reduced embedding is used EVERYWHERE — tree
# models and FNN/LSTM/TCN alike. The plan's original Decision #7 language
# ("FNN/LSTM/TCN can take the full embedding or a smaller learned
# projection") is intentionally narrowed here: the full 384-dim embedding
# is never loaded for more than one file/sample at a time, for any model
# family. This keeps the memory profile safe unconditionally, at the cost
# of not giving the DL models the option of the raw embedding — judged an
# acceptable trade given the uncertain payoff (SHAP/ablation, Decisions
# #9/#20, will show whether the retained components matter at all).
#
# IMPORTANT — memory: loading F-DATA's raw `embedding` column for all ~26M
# rows at once measured at ~100GB+ RSS (each row holds an individually
# allocated 384-dim array object — very memory-inefficient at this scale),
# versus ~27GB for all other 44 columns combined. Never do
# `PCA(...).fit_transform(everything)` across the full dataset. Fit once
# on a sample (fit_fdata_embedding_pca), then apply the already-fitted
# model file-by-file (transform_fdata_embedding) so the raw embedding for
# more than one month is never held in memory simultaneously.

def compute_embedding_explained_variance(sample_df: pd.DataFrame, max_components: int = 100) -> PCA:
    """Fit PCA with many components on a sample, purely to inspect
    cumulative explained variance — used to pick n_components empirically
    (via n_components_for_variance) rather than guessing a "handful"."""
    stacked = np.stack(sample_df["embedding"].to_numpy())
    max_components = min(max_components, stacked.shape[0], stacked.shape[1])
    return PCA(n_components=max_components, random_state=0).fit(stacked)


def n_components_for_variance(pca: PCA, threshold: float = 0.90) -> int:
    """Smallest number of components whose cumulative explained variance
    ratio reaches `threshold`, given a PCA already fit with many
    components (from compute_embedding_explained_variance)."""
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, threshold) + 1)


def fit_fdata_embedding_pca(sample_df: pd.DataFrame, n_components: int = 10) -> PCA:
    """Fit PCA on a representative sample (e.g. the SAMPLE_SIZE dev sample,
    or a dedicated larger fitting sample) — this is the only place the raw
    embedding should be stacked across more than one file's worth of rows.
    `n_components` should normally come from n_components_for_variance,
    not the arbitrary default below (kept only as a fallback)."""
    stacked = np.stack(sample_df["embedding"].to_numpy())
    n_components = min(n_components, stacked.shape[0], stacked.shape[1])
    return PCA(n_components=n_components, random_state=0).fit(stacked)


def transform_fdata_embedding(df: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    """Apply an already-fitted PCA (from fit_fdata_embedding_pca) to `df`.
    Safe to call per-file in a loop over all 38 months — never re-fits,
    so it never needs the full-dataset embedding in memory at once."""
    stacked = np.stack(df["embedding"].to_numpy())
    components = pca.transform(stacked)
    columns = [f"emb_pc_{i}" for i in range(pca.n_components_)]
    return pd.DataFrame(components, columns=columns, index=df.index)


def reduce_fdata_embedding(df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """Convenience wrapper for small/dev-sample use (fits and transforms in
    one call, as notebook 02 does on the 5000-row sample). For full-scale
    work across many months, use fit_fdata_embedding_pca once +
    transform_fdata_embedding per file instead — see the memory note above."""
    return transform_fdata_embedding(df, fit_fdata_embedding_pca(df, n_components))


def load_fdata_no_embedding(paths: list[str]) -> pd.DataFrame:
    """Load and concatenate multiple F-DATA monthly parquet files WITHOUT
    the embedding column — this is the memory-safe way to load many/all
    months at once (~27GB for all 38 months vs ~100GB+ with embedding
    included). Use fit_fdata_embedding_pca/transform_fdata_embedding
    separately (per-file) if embedding-derived features are also needed."""
    cols = [c for c in dict.fromkeys(FDATA_TIER_A_COLUMNS + FDATA_TIER_B_COLUMNS
                                     + list(FDATA_TARGETS.values())) if c != "embedding"] + ["jid"]
    return pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)


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
