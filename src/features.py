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
# CORRECTION (2026-07-23, notebook 03): PM100's node_power_consumption
# turns out to be the SAME kind of quantity as avgpcon above, not the
# per-node reading its name implies — verified directly (not just from
# the docs) in notebook 03: dividing each job's mean node_power_consumption
# by num_nodes_alloc gives a stable ~680-733W across every node count from
# 1 to 32 nodes (single-node jobs average 733W; 16-node jobs average
# 10,876W, i.e. 680W/node), which is only possible if the recorded value
# is already summed across all allocated nodes at each 20s sample. An
# earlier pass over this file asserted node_power_consumption was
# "genuinely per-node" — that was wrong, based on the field name and docs
# rather than a direct check against the data, the same mistake the
# avgpcon investigation above was originally trying to avoid. Both
# datasets' power fields are job-wide totals; the calibrated power model
# in src/roofline.py accounts for this (P_idle scales by num_nodes_alloc,
# not applied once per job).

FDATA_TIER_A_COLUMNS: list[str] = [
    "usr",          # username — for historical rolling stats
    "jnam",         # job name
    "cnumr",        # cores requested
    "nnumr",        # nodes requested
    "adt",          # arrival/submission datetime
    "qdt",          # time of insertion in job queue (pre-execution)
    "schedsdt",     # time of completed scheduling choice (still pre-execution)
    "elpl",         # elapsed time limit requested
    "mszl",         # memory size limit requested (sentinel-sanitized — see handle_mszl_sentinel)
    "mszl_unlimited",  # True where mszl was the "no limit requested" sentinel
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

# --- mszl sentinel handling (Decision #19) ---------------------------------
# `mszl` (memory size limit requested, Tier A) turns out to use an
# unsigned-int sentinel for "no limit requested" rather than a null: found
# 2026-07-31 while scratch-timing notebook 05's SAMPLE_SIZE decision, when
# it broke XGBoost/LightGBM's histogram binning (see EXPERIMENT_TRACKER.md
# Data Gotchas). Verified against the 6-month dev slice: 99.5% of rows
# (2,882,158 / 2,897,734) sit exactly at 2**64 - 1; the rest have real
# requested limits spanning ~1e9-3e10 (bytes) — a 10-order-of-magnitude
# range if the sentinel is left mixed in raw.
MSZL_SENTINEL: float = float(2**64 - 1)


def handle_mszl_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """Split mszl into a clean boolean flag (mszl_unlimited) plus a
    sanitized numeric mszl column. Sentinel rows get mszl set to 0.0 (not
    NaN) so the numeric column stays finite and directly usable by RF
    without a separate imputation step — mszl_unlimited alone carries the
    "no limit requested" signal for every model family (RF/XGBoost/
    LightGBM all handle a binary flag natively). Returns a copy of df with
    both columns present; only `mszl_unlimited` needs adding to
    FDATA_TIER_A_COLUMNS (mszl itself already lists there), and this must
    run before build_tier_a_features (same pattern as
    add_pm100_derived_indicators for PM100)."""
    out = df.copy()
    is_sentinel = out["mszl"] >= 1e15  # sentinel is ~1.8e19; real requests are <=~3e10
    out["mszl_unlimited"] = is_sentinel
    out.loc[is_sentinel, "mszl"] = 0.0
    return out


def assert_mszl_sanitized(df: pd.DataFrame) -> None:
    """Sanity-check assertion (Decision #19): fail loudly if mszl still
    contains uint64-sentinel-scale values after handle_mszl_sentinel —
    guards against a future refactor silently reintroducing the bug."""
    assert (df["mszl"] < 1e15).all(), "mszl still contains sentinel-scale (>=1e15) values"


def assert_avgpcon_is_job_total(df: pd.DataFrame, corr_threshold: float = 0.9) -> None:
    """Sanity-check assertion: fail loudly if avgpcon's correlation with
    nnuma drops below threshold -- guards against a future data refresh
    silently changing this field's semantics back to genuinely per-node."""
    corr = df["avgpcon"].corr(df["nnuma"])
    assert corr >= corr_threshold, (
        f"avgpcon-nnuma correlation ({corr:.4f}) fell below {corr_threshold} -- "
        "check whether avgpcon is still a job-total quantity."
    )

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
    "cores_per_task", "num_tasks", "shared",
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
# RETROACTIVE AUDIT (2026-07-22): the same concentration check was run
# against columns already sitting in PM100_TIER_A_COLUMNS before this
# vetting process existed, not just the three new candidates above.
# req_switch failed it just as badly as the excluded candidates: only 13
# non-zero occurrences across 180,310 completed jobs, and ALL 13 from a
# single user (100% concentration) — moved to reserved below. shared and
# the minority categories of qos/partition were also checked: shared has
# no comparable rare-value problem (its two populated categories have
# 149,226 and 31,084 jobs — nowhere near single-user territory); qos/
# partition's smallest categories are either negligible in size (n=2) or
# only moderately concentrated (partition's smallest category: n=111,
# top-1 user share 55%) — both are legitimate multi-valued scheduling
# categoricals rather than purpose-built rare binary flags, so ordinary
# small-category noise is expected and not treated the same way.
# F-DATA's analogous check (jobenv_req, its only near-constant Tier A
# column) found real but much milder concentration: the minority category
# (1,088 of ~3.66M jobs in a 5-file sample) spans 21 users, top-1 share
# 39.2% — above F-DATA's own baseline (16.0%) but far short of PM100's
# excluded flags. Left active for now; worth re-checking once more months
# are available.
#
# Both raw columns (req_nodes, threads_per_core, req_switch) and all four
# derived flags are preserved in the processed dataset regardless of this
# split — only num_tasks_missing is added to the active
# PM100_TIER_A_COLUMNS list. This concentration check generalizes: both
# datasets are dominated by a handful of power users (PM100's top-5 users
# = 45.4% of all jobs, top-10 = 59.9%; F-DATA's top-5 = 41.4% in a 5-file
# sample) — worth the same scrutiny for any future derived feature, not
# just these three.
#
# Split-strategy note: PM100's top-5 heaviest users are active across
# 97-100% of the full 159-day dataset span (one exception at 76%,
# starting a month in but continuing to the end) — a chronological
# train/test split would not strand any of them entirely in one side of
# the split. This has only been checked for PM100; F-DATA spans 38
# months rather than 6, and the same check has not yet been run there.

PM100_RESERVED_COLUMNS: list[str] = [
    "req_nodes", "threads_per_core", "req_switch",
    "node_pinned", "threads_per_core_set", "req_switch_set",
]


def add_pm100_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Binary indicators for PM100's missing-by-design/incomplete/
    near-constant fields. Returns a copy of df with node_pinned,
    threads_per_core_set, req_switch_set, and num_tasks_missing added.
    Only num_tasks_missing belongs in the active Tier A feature set — the
    other three are computed and preserved but reserved (see module
    comment above)."""
    out = df.copy()
    out["node_pinned"] = out["req_nodes"].notna()
    out["threads_per_core_set"] = out["threads_per_core"].notna()
    out["req_switch_set"] = out["req_switch"] != 0
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


def build_tier_a_features(df: pd.DataFrame, dataset: str, include_embedding: bool = True) -> pd.DataFrame:
    """Submission-time-only feature matrix. Valid for RQ1/H1 headline results.
    Set include_embedding=False when the raw embedding column was never
    loaded (the memory-safe load_fdata_no_embedding pattern) and is
    instead PCA-reduced separately, positionally aligned, per
    fit_fdata_embedding_pca/transform_fdata_embedding (notebook 05
    onward)."""
    columns = FDATA_TIER_A_COLUMNS if dataset == "fdata" else PM100_TIER_A_COLUMNS
    if not include_embedding:
        columns = [c for c in columns if c != "embedding"]
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
    _derived_only = {"embedding", "mszl_unlimited"}  # not real parquet columns — added by
    # handle_mszl_sentinel()/etc. after loading, never present in the raw files
    cols = [c for c in dict.fromkeys(FDATA_TIER_A_COLUMNS + FDATA_TIER_B_COLUMNS
                                     + list(FDATA_TARGETS.values())) if c not in _derived_only] + ["jid"]
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


# --- Stratified sampling by job-size bucket (Decision #10) ------------------
# Draws a fixed-size sample from an ALREADY-SPLIT frame (train or test),
# never from the full pre-split data — stratifying before splitting could
# let the draw disturb which rows fall before/after the chronological
# boundary the whole evaluation depends on. Bucketed on cnumr (cores
# requested, Tier A) since it's a genuine submission-time job-size proxy
# available for both datasets' analogous column.

_JOB_SIZE_COL: dict[str, str] = {"fdata": "cnumr", "pm100": "num_cores_req"}


def stratified_sample_by_job_size(
    df: pd.DataFrame, dataset: str, n: int, n_buckets: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Stratified sample of `n` rows from `df` (call separately on an
    already-split train_df/test_df — see module docstring above), grouped
    into `n_buckets` quantile buckets of the job-size column."""
    n = min(n, len(df))
    col = _JOB_SIZE_COL[dataset]
    buckets = pd.qcut(df[col], q=n_buckets, duplicates="drop")
    frac = n / len(df)
    sampled = df.groupby(buckets, group_keys=False, observed=True).apply(
        lambda g: g.sample(frac=frac, random_state=seed)
    )
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        remainder = df.drop(sampled.index).sample(n=n - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, remainder])
    return sampled


# --- Datetime -> epoch-seconds encoding -------------------------------------
# NOT a blind `.astype("int64") // 10**9`: pandas datetime64's internal
# storage unit varies (ns/us/etc.) and a fixed divisor silently gives the
# wrong answer depending on that unit — found the hard way while
# scratch-timing notebook 05's SAMPLE_SIZE decision. F-DATA's datetime
# columns are datetime64[us, UTC+09:00]; `.astype("int64")` on them gives
# MICROSECONDS since epoch, not nanoseconds, so a 1e9 divisor is 1000x too
# small. Subtracting a matching tz-aware reference timestamp and taking
# total_seconds() sidesteps the storage-unit question entirely.

def datetime_to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Epoch-seconds encoding of a (possibly tz-aware) datetime column,
    robust to pandas' internal storage-unit dtype (ns/us/etc.)."""
    ts = pd.to_datetime(series)
    tz = ts.dt.tz
    epoch = pd.Timestamp("1970-01-01", tz=tz) if tz is not None else pd.Timestamp("1970-01-01")
    return (ts - epoch).dt.total_seconds()


# --- Model-ready numeric matrix for F-DATA Tier A (notebook 05 onward) ------
# build_tier_a_features only selects/validates tier membership — it
# returns raw, mixed-dtype columns (strings, datetimes, the embedding
# array), not something a model can fit on directly. This is the encoding
# step that makes it model-ready: numeric passthrough, mszl_unlimited as
# int, datetimes to epoch seconds (via datetime_to_epoch_seconds above,
# not a blind divisor), jobenv_req factorized (near-constant categorical,
# see the Feature Vetting notes above), usr frequency-encoded (raw
# high-cardinality username; the per-user rolling-stat feature already
# captures generalizable per-user history — this adds a coarse
# concentration signal, not identity), and jnam dropped entirely (its
# semantic content is already captured by the PCA-reduced embedding
# passed in separately; a raw near-unique job-name string adds noise, not
# signal, if frequency-encoded).

FDATA_TIER_A_NUMERIC_PASSTHROUGH: list[str] = ["cnumr", "nnumr", "elpl", "mszl", "pri", "freq_req"]
FDATA_TIER_A_DATETIME_COLUMNS: list[str] = ["adt", "qdt", "schedsdt"]


def build_fdata_numeric_matrix(
    tier_a_df: pd.DataFrame, embedding_pca_df: pd.DataFrame, extra_columns: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Fully numeric Tier A feature matrix for F-DATA, ready for RF/
    XGBoost/LightGBM (or FNN/LSTM/TCN later). `tier_a_df` must already
    have gone through handle_mszl_sentinel and be the output of
    build_tier_a_features. `embedding_pca_df` is the already-fitted PCA
    transform of this same row set (fit on TRAIN only — see
    fit_fdata_embedding_pca/transform_fdata_embedding — never re-fit per
    split, to avoid leaking test rows into the PCA fit). `extra_columns`
    is for target-specific additions such as the rolling-stat feature
    (`{target}_user_rolling_mean`), which isn't in the fixed Tier A column
    list because its name depends on which target is being modeled."""
    out = pd.DataFrame(index=tier_a_df.index)
    for col in FDATA_TIER_A_NUMERIC_PASSTHROUGH:
        out[col] = tier_a_df[col].to_numpy(dtype=float)
    out["mszl_unlimited"] = tier_a_df["mszl_unlimited"].astype(int)
    for col in FDATA_TIER_A_DATETIME_COLUMNS:
        out[f"{col}_epoch"] = datetime_to_epoch_seconds(tier_a_df[col])
    out["jobenv_req_code"] = pd.factorize(tier_a_df["jobenv_req"])[0]
    user_counts = tier_a_df["usr"].value_counts()
    out["usr_freq"] = tier_a_df["usr"].map(user_counts).to_numpy(dtype=float)
    out = pd.concat([out, embedding_pca_df.reindex(out.index)], axis=1)
    if extra_columns is not None:
        out = pd.concat([out, extra_columns.reindex(out.index)], axis=1)
    return out


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
