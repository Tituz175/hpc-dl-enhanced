# Experiment Tracker

Thesis: *Deep Learning Enhanced Prediction of Execution Time, Memory Usage, and Power Consumption in Heterogeneous HPC Systems*

Tracks execution of **Track B** from the conceptualization plan (`~/.claude/plans/i-have-a-master-s-encapsulated-cat.md`). Update this as work happens — it's a living document, not a one-time writeup. See `WRITING_TRACKER.md` for the thesis-writing side.

Status values used throughout: `Not Started` / `In Progress` / `Blocked` / `Done`.

Last updated: 2026-07-21

---

## ⚠️ Sample-size reminder

Notebooks 01 and 02 currently run on `SAMPLE_SIZE = 5000` (a plain random subsample) — this is a **dev-only speed setting**, not the experiment. Per Decision #10:

- **Notebooks 05-07 (classical ML, DL, Hybrid model training)**: step `SAMPLE_SIZE` up to **1–5 million rows** (stratified by job-size/chronological, not plain random) before any results from these are meaningful for hyperparameter search.
- **Final reported thesis numbers**: `SAMPLE_SIZE = None` — the full ~24M F-DATA rows (compute permitting) and the full ~231K PM100 rows (already always used in full).
- PM100 never needs subsampling — it's small enough to use in full throughout.

I'll flag this explicitly again when we reach notebook 05, but it's recorded here too so it isn't lost either way.

---

## Workstation Capacity Check (2026-07-21)

Real specs, measured directly, in response to a concern about whether this workstation can actually handle full-scale F-DATA:

| Resource | Spec |
|---|---|
| RAM | 125GB total, 115GB available |
| CPU | Intel i9-10900X, 10 cores / 20 threads |
| GPU | RTX 3090, 24GB VRAM |
| Disk | 161GB free (after all 38 F-DATA files + PM100 = ~26GB) |

**The real constraint found**: F-DATA's `embedding` column (384-dim SBert vector per job) is memory-inefficient at full scale — measured ~5.54GB/month with it included vs. ~1.39GB/month without, on the ~1.3M-row Jan 2022 file. Extrapolated across all ~25.87M rows (38 files, confirmed via parquet metadata): **~100GB+ with the raw embedding included** (would leave almost no headroom) vs. **~27GB without it** (very comfortable).

**Fix applied** (`src/features.py`, commit `ca908ed`): embedding PCA reduction is now split into `fit_fdata_embedding_pca` (fit once, on a sample) and `transform_fdata_embedding` (apply the fitted model per-file, in a loop over months) — verified working across different files. `load_fdata_no_embedding` loads all/many months at once safely when embedding-derived features aren't needed for that step.

**Bottom line**: full-scale F-DATA (all 38 months, all non-embedding columns) is very feasible on this machine (~27GB of 115GB available). The RTX 3090's 24GB VRAM isn't a separate concern for this — DL training only needs mini-batches on the GPU at a time, not the whole dataset resident in VRAM. The 10-core CPU is comfortable for RF/XGBoost/LightGBM's parallel training. The only thing that actually needed a code change was the embedding-loading pattern, which is now fixed.

---

## Milestone Status

| Months | Milestone | Status | Target Date | Actual Date | Notes |
|---|---|---|---|---|---|
| 1–2 | Git repo set up (local) | Done | 2026-07-21 | 2026-07-21 | Root commit `d709812`. Remote not yet created — `gh` CLI unavailable on this machine; create manually on GitHub/GitLab (private) and add as `origin` |
| 1–2 | Repo scaffolding (src/ package, 9 notebooks, .gitignore) | Done | 2026-07-21 | 2026-07-21 | src/ modules are structural stubs (raise NotImplementedError) pending real data |
| 1–2 | uv-managed environment (Python 3.12, pyproject.toml + uv.lock) | Done | 2026-07-21 | 2026-07-21 | torch 2.13.0+cu130 confirmed detecting the RTX 3090. Jupyter kernel `hpc-dl-enhanced` registered. Note: pinned Python to 3.12 (not 3.13) since numba/llvmlite lack working 3.13 support; also needed an explicit `numba>=0.60` constraint to stop uv's resolver from picking a broken ancient numba — see commit `bcc6149` |
| 1–2 | F-DATA + PM100 acquired, EDA done | Done | 2026-07-21 | 2026-07-21 | **All 38 F-DATA monthly files now present** (`21_03.parquet` through `24_04.parquet`, ~25.87M rows total confirmed via parquet metadata) + full PM100. Notebooks 01-02 execute clean end-to-end on a 5000-row sample; full-scale run pending (see Workstation Capacity Check above — feasible at ~27GB RAM if embedding column handled via the new fit/transform split, commit `ca908ed`). Real schemas confirmed against official docs: F-DATA `docs/feature_list.md`, PM100 `documentation/job_features.md` |
| 1–2 | Tier A/B feature split + sanity-check assertions | Done | 2026-07-21 | 2026-07-21 | `src/features.py` grounded in official column docs, validated against real data (every listed column exists, every real column accounted for). Decisions #1, #19 |
| 1–2 | Chronological split implemented, target transforms decided | In Progress | | | log1p transform implemented + round-trip-verified on all 6 targets (Decision #3, notebook 02, commit `6087afc`). Chronological split itself still not implemented — needs the remaining 37 F-DATA months first |
| 1–2 | NPB/Rodinia microbenchmark validation (RTX 3090) | Not Started | | | Decision #16, runs independently of trace pipeline |
| 2–3 | Naive baseline predictor | Not Started | | | Decision #17 |
| 2–3 | Analytical baselines: F-DATA Roofline + PM100 power model | Not Started | | | Decision #15 |
| 2–3 | RF/XGBoost/LightGBM baselines (both datasets) | Not Started | | | Decisions #5, #14 |
| 3–5 | FNN + LSTM + TCN (multi-task and single-target variants) | Not Started | | | Decisions #12, #13 |
| 3–5 | Hybrid Analytical+DL model | Not Started | | | Power-only on PM100 (Decision #15) |
| 5–6 | Full evaluation sweep + statistical testing | Not Started | | | Decision #6, rolling-window drift check (#18) |
| 5–6 | SHAP + feature-ablation interpretability | Not Started | | | Decisions #9, #20 |
| 5–6 | Cross-dataset discussion write-up (feeds Track A Ch.5) | Not Started | | | |

---

## Notebook Status

All 9 notebooks are scaffolded (title cell + `src` imports) as of 2026-07-21 — "Not Started" below refers to actual analysis content, not file existence.

| # | Notebook | Status | Decisions Covered | Notes |
|---|---|---|---|---|
| 01 | `data_acquisition_eda.ipynb` | In Progress | EDA groundwork | Runs clean end-to-end on a 5000-row sample (commit `d2518d4`). Pending: full-scale run once remaining 37 F-DATA months arrive |
| 02 | `feature_engineering.ipynb` | Done (dev-sample) | #1, #2, #3, #4, #7 | Runs clean end-to-end on the 5000-row sample (commit `6087afc`): failed-job exclusion, Tier A/B re-validation, embedding PCA, rolling-stat features, log1p round-trip — all pass. Will need re-running once `SAMPLE_SIZE` steps up (see sample-size reminder below) |
| 03 | `analytical_baselines.ipynb` | Not Started | #15 | Scaffolded. F-DATA Roofline + PM100 calibrated power model |
| 04 | `microbenchmark_validation.ipynb` | Not Started | #16 | Scaffolded. Independent of main data pipeline |
| 05 | `classical_ml_baselines.ipynb` | Not Started | #5, #14 | Scaffolded. RF, XGBoost, LightGBM |
| 06 | `deep_learning_models.ipynb` | Not Started | #12, #13 | Scaffolded. FNN, LSTM, TCN |
| 07 | `hybrid_model.ipynb` | Not Started | #15 (Hybrid) | Scaffolded. Power-only for PM100 |
| 08 | `evaluation_and_statistics.ipynb` | Not Started | #6, #8, #18 | Scaffolded |
| 09 | `interpretability.ipynb` | Not Started | #9, #20 | Scaffolded |

---

## Decision Implementation Checklist

| # | Decision | Status | Notebook | Notes |
|---|---|---|---|---|
| 1 | Tier A/B feature-leakage split | Done (dev-sample) | 01, 02 | Column classification + historical rolling-stat feature (`add_user_rolling_stat`, strictly-past-only via shift(1)) both implemented and validated on real data. Re-validated after failed-job exclusion too (Decision #19 discipline). Still needs advisor sign-off — see WRITING_TRACKER |
| 2 | Target definition (used memory, fallback allocated) | Done | 01 | Confirmed: PM100 has no "used" field at all (`mem_alloc` always the fallback case, not a choice) |
| 3 | Target skew (log1p transform) | Done (dev-sample) | 02 | `transform_target`/`inverse_transform_target` implemented, round-trip verified on all 6 targets (both datasets) |
| 4 | Failed/cancelled job exclusion | Done (dev-sample) | 02 | `filter_completed_jobs` implemented. Real exclusion rates: F-DATA 5.1%, PM100 21.0% (PM100 notably higher — worth a mention in the thesis) |
| 5 | Fixed tuning budget across model families | Not Started | 05, 06, 07 | |
| 6 | Multi-seed + paired significance testing | Not Started | 08 | |
| 7 | SBert embedding dimensionality reduction | Done (dev-sample) | 02 | `reduce_fdata_embedding` (PCA, 10 components) implemented and validated. F-DATA only — PM100 has no embedding field |
| 8 | MAPE + stratified error breakdowns | Not Started | 08 | |
| 9 | SHAP + permutation/integrated-gradients interpretability | Not Started | 09 | |
| 10 | F-DATA stratified subsample → full-scale strategy | In Progress | 01, all training notebooks | `SAMPLE_SIZE` config added to notebook 01 (currently random 5000-row sample, not yet the stratified-by-job-size/chronological version described in the decision). Full-scale feasibility confirmed: ~27GB RAM for all 38 months minus embedding (see Workstation Capacity Check) — no cluster escalation needed for RAM; embedding handled via fit-once/transform-per-file (`load_fdata_no_embedding`, `fit_fdata_embedding_pca`, `transform_fdata_embedding`) |
| 11 | Git repo/remote + backup discipline | Not Started | — | Repo-level, not notebook-specific |
| 12 | Plain FNN (no entity-embedding upgrade) | Not Started | 06 | |
| 13 | LSTM + TCN on PM100 intra-job trace | Not Started | 06 | |
| 14 | XGBoost + LightGBM ablation | Not Started | 05 | |
| 15 | F-DATA Roofline / PM100 calibrated power model | Not Started | 03, 07 | Hybrid scoped to Power-only on PM100 |
| 16 | Microbenchmark validation side-study | Not Started | 04 | |
| 17 | Naive/trivial baseline predictor | Not Started | 05 (or its own step) | |
| 18 | Rolling-window temporal-drift check (F-DATA) | Not Started | 08 | |
| 19 | Sanity-check assertions on pipeline code | Done (dev-sample) | 01, 02, 03 | `assert_no_tier_leakage` and `expm1_round_trip_check` both implemented and passing on real data for both datasets, re-checked after every transformation step |
| 20 | Feature ablation study alongside SHAP | Not Started | 09 | |

---

## Data Gotchas Discovered So Far

Worth keeping visible since they'll recur in every notebook that touches these columns, not just notebook 01:

- **F-DATA's `embedding` column** is a 384-dim `numpy.ndarray` per row (object dtype). Never call generic `describe()`/uniqueness-style operations on it directly — pandas tries to hash/compare unhashable arrays and grinds for a very long time instead of erroring cleanly. Exclude it explicitly first.
- **PM100's `node_power_consumption`** (and likely `mem_power_consumption`, `cpu_power_consumption`) is a per-job **time series** (20-second-interval samples, variable length per job), not a scalar — this is the genuine intra-job sequence data Decision #13 relies on for LSTM/TCN. Any scalar treatment of it (plots, describe, simple stats) needs an explicit per-job reduction (e.g. mean) first; the raw array is what feeds the sequence models later.
- **PM100 has no "used" memory field at all** — only `mem_req`/`mem_alloc` — so its memory target is always the Decision #2 fallback (`mem_alloc`), never a choice between used/allocated the way F-DATA's `mmszu`/`msza` are.
- **PM100's failure rate (21.0%) is notably higher than F-DATA's (5.1%)** in the dev-sample — worth stating explicitly in the thesis's data-characterization section rather than treating both datasets as symmetric on this axis.
- **RESOLVED — F-DATA's `avgpcon`/`minpcon`/`maxpcon` are the job's TOTAL power (summed across all allocated nodes), not a single node's power**, despite the "node power consumption" naming. Verified two ways: (1) `avgpcon` correlates 0.98 with `nnuma` (nodes allocated); single-node jobs (`nnuma==1`) show `avgpcon` in [39.6, 149.9] W, matching known A64FX per-node figures, confirming genuine Watts, not a units bug. (2) `econ / (avgpcon * duration) ≈ 1/3600` consistently — confirms `econ` is in Watt-hours, units are self-consistent. **Consequence**: F-DATA's power target (whole-job total) is semantically different from PM100's `node_power_consumption` (genuinely per-node) — a different physical quantity, not just a different unit. Documented in `src/features.py`; footnote this alongside the existing Roofline-vs-power-model comparability caveat (Decision #15) wherever both appear together.

---

## Risk Watch

Mirrors the Risk Register (Track B3) — track whether each risk actually materialized and what was done about it.

| Risk | Materialized? | Action Taken | Notes |
|---|---|---|---|
| Feature leakage undermines RQ1 claims | | | |
| 24M-row F-DATA too slow to iterate on with 1 GPU | | | |
| LSTM has nothing genuinely sequential on F-DATA | | | Expected — PM100 is primary sequence testbed |
| Unfair baseline comparison weakens H1/H2 | | | |
| Heavy-tailed targets make R²/MAE misleading | | | |
| Single-vendor/single-system findings don't generalize | | | Expected limitation, not a flaw |
| RTX 3090 microbenchmark doesn't match production hardware | | | Expected — scoped as methodology validation only |
| F-DATA workload mix drifts over 37 months | | | |
| PM100 has zero FLOP/performance-counter fields | Yes (confirmed during planning) | Replaced Roofline with calibrated power model | Resolved before implementation began |
| F-DATA Roofline and PM100 power model not comparable | Yes (confirmed during planning) | Footnoting discipline adopted | Resolved before implementation began |
| Silent bug in Roofline/tiering/transform code | | | |
| Experiment code/results lost to disk failure | | | |
| Timeline slips past 6-9 months | | | See Descope Log below |

---

## Descope Log

Empty unless/until something from the Descope Priority Order (Track B7) actually gets cut. Record what, when, and why — Track A's Discussion/Threats-to-Validity chapter should cite this log accurately rather than from memory.

| Date | Item Cut | Reason | Impact |
|---|---|---|---|
| | | | |
