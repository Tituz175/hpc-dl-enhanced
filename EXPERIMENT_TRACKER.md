# Experiment Tracker

Thesis: *Deep Learning Enhanced Prediction of Execution Time, Memory Usage, and Power Consumption in Heterogeneous HPC Systems*

Tracks execution of **Track B** from the conceptualization plan (`~/.claude/plans/i-have-a-master-s-encapsulated-cat.md`). Update this as work happens — it's a living document, not a one-time writeup. See `WRITING_TRACKER.md` for the thesis-writing side.

Status values used throughout: `Not Started` / `In Progress` / `Blocked` / `Done`.

Last updated: 2026-07-21

---

## Milestone Status

| Months | Milestone | Status | Target Date | Actual Date | Notes |
|---|---|---|---|---|---|
| 1–2 | Git repo + remote set up | Not Started | | | First implementation action |
| 1–2 | F-DATA + PM100 acquired, EDA done | Not Started | | | Includes temporal-drift trend plot (Decision #18) |
| 1–2 | Tier A/B feature split + sanity-check assertions | Not Started | | | Decisions #1, #19 |
| 1–2 | Chronological split implemented, target transforms decided | Not Started | | | Decisions #2, #3 |
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

| # | Notebook | Status | Decisions Covered | Notes |
|---|---|---|---|---|
| 01 | `data_acquisition_eda.ipynb` | Not Started | EDA groundwork | |
| 02 | `feature_engineering.ipynb` | Not Started | #1, #2, #3, #4, #7 | |
| 03 | `analytical_baselines.ipynb` | Not Started | #15 | F-DATA Roofline + PM100 calibrated power model |
| 04 | `microbenchmark_validation.ipynb` | Not Started | #16 | Independent of main data pipeline |
| 05 | `classical_ml_baselines.ipynb` | Not Started | #5, #14 | RF, XGBoost, LightGBM |
| 06 | `deep_learning_models.ipynb` | Not Started | #12, #13 | FNN, LSTM, TCN |
| 07 | `hybrid_model.ipynb` | Not Started | #15 (Hybrid) | Power-only for PM100 |
| 08 | `evaluation_and_statistics.ipynb` | Not Started | #6, #8, #18 | |
| 09 | `interpretability.ipynb` | Not Started | #9, #20 | |

---

## Decision Implementation Checklist

| # | Decision | Status | Notebook | Notes |
|---|---|---|---|---|
| 1 | Tier A/B feature-leakage split | Not Started | 02 | Still needs advisor sign-off — see WRITING_TRACKER |
| 2 | Target definition (used memory, fallback allocated) | Not Started | 02 | |
| 3 | Target skew (log1p transform) | Not Started | 02 | |
| 4 | Failed/cancelled job exclusion | Not Started | 02 | |
| 5 | Fixed tuning budget across model families | Not Started | 05, 06, 07 | |
| 6 | Multi-seed + paired significance testing | Not Started | 08 | |
| 7 | SBert embedding dimensionality reduction | Not Started | 02 | |
| 8 | MAPE + stratified error breakdowns | Not Started | 08 | |
| 9 | SHAP + permutation/integrated-gradients interpretability | Not Started | 09 | |
| 10 | F-DATA stratified subsample → full-scale strategy | Not Started | 01, all training notebooks | |
| 11 | Git repo/remote + backup discipline | Not Started | — | Repo-level, not notebook-specific |
| 12 | Plain FNN (no entity-embedding upgrade) | Not Started | 06 | |
| 13 | LSTM + TCN on PM100 intra-job trace | Not Started | 06 | |
| 14 | XGBoost + LightGBM ablation | Not Started | 05 | |
| 15 | F-DATA Roofline / PM100 calibrated power model | Not Started | 03, 07 | Hybrid scoped to Power-only on PM100 |
| 16 | Microbenchmark validation side-study | Not Started | 04 | |
| 17 | Naive/trivial baseline predictor | Not Started | 05 (or its own step) | |
| 18 | Rolling-window temporal-drift check (F-DATA) | Not Started | 08 | |
| 19 | Sanity-check assertions on pipeline code | Not Started | 02, 03 | |
| 20 | Feature ablation study alongside SHAP | Not Started | 09 | |

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
