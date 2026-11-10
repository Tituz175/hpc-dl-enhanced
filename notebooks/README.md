# Notebooks

Repository overview, motivation, and results are in the [main README](../README.md). This file covers running the notebooks: order, dependencies, and what you need to execute them.

## Order

Run in sequence. Later notebooks rely on conventions and artifacts from earlier ones.

| # | Notebook | Depends on | Does |
|---|---|---|---|
| 1 | `01_data_acquisition_eda.ipynb` | raw data | Loads both datasets, checks the schema against the loaded frames, sets up the Tier A / Tier B split |
| 2 | `02_feature_engineering.ipynb` | 01 | The four-check feature vetting method, data-quality fixes (sentinel values, one bit-corrupt power value), the Tier A matrix |
| 3 | `03_analytical_baselines.ipynb` | 02 | Hierarchical Roofline (F-DATA), calibrated power model (PM100), and the shared chronological train/test split |
| 4 | `04_microbenchmark_validation.ipynb` | nothing | GPU-microbenchmark validation of the Roofline construction. Self-contained, feeds nothing downstream |
| 5 | `05_classical_ml_baselines.ipynb` | 01 to 03 | F-DATA duration and PM100 power: Random Forest, XGBoost, LightGBM, tuned and evaluated |
| 6 | `05b_classical_ml_memory.ipynb` | 01 to 03 | F-DATA memory and PM100 memory, same method, plus the requested-versus-allocated feature investigation |
| 7 | `05c_classical_ml_power_duration.ipynb` | 01 to 03 | F-DATA power and PM100 duration, the last two of six target/dataset combinations. F-DATA power also gets a calibrated analytical baseline (`src/roofline.py`), F-DATA power's XGBoost needs `tree_method="exact"` for its final fit (a real, root-caused bug in `hist`'s binning on this feature set), and PM100 duration has no analytical baseline (no FLOP/bandwidth fields) |
| 8 | `05d_seed_variance.ipynb` | 05, 05b, 05c | Multi-seed variance: repeats each already-tuned model's final fit across several random seeds to test whether a "best model" claim is robust to seed-to-seed stochasticity. Loads pre-computed results from `data/interim/*.joblib`; see [`scripts/decision6_seed_variance/README.md`](../scripts/decision6_seed_variance/README.md) for how those were produced |
| 9 | `08_evaluation_and_statistics.ipynb` | 05, 05b, 05c, 05d | A user-clustered bootstrap of the R2 difference between each pair of classical models (Random Forest, XGBoost, LightGBM), all six target/dataset combinations, 18 comparisons. Classical models only -- does not cover the deep-learning or hybrid models below. Loads the full sweep's saved results from `scripts/notebook08_paired_bootstrap/run_full_sweep.py` rather than re-running the bootstrap live |
| 10 | `06a_sequence_investigation.ipynb` | 01 to 03 | Jobs-per-user distribution, the sequence-length choice, padding policy, the sequence-construction leakage test, F-DATA/PM100 lag-1 autocorrelation; also holds the deep-learning feature-set explanation the notebooks below repeat a short version of |
| 11 | `06c_dl_power.ipynb` / `06d_dl_duration.ipynb` / `06e_dl_memory.ipynb` | 01 to 03 | The real FNN/LSTM/TCN sweep (via `src/dl_pipeline.py`), one notebook per target pair (power, duration, memory), each covering both F-DATA and PM100. Reuses the already-complete persistent Optuna studies and saved checkpoints -- a normal re-run does zero additional training. Each writes its own combinations' test-prediction artifacts and `current_dim` row to `data/interim/notebook06_results/` (gitignored), shared with 06b and 06f below |
| 12 | `06b_model_architectures.ipynb` | 06c, 06d, 06e (reads their shared `data/interim/notebook06_results/dim_table.csv`) | The FNN/SequenceLSTM/SequenceTCN/`_CausalConvBlock`/FlatDataset/SequenceDataset class definitions as code cells, documented layer by layer, an equivalence check against `src/models.py`, architecture diagrams, the training configuration, and a checkpoint-match check across all 18 (combination, architecture) pairs |
| 13 | `06f_dl_comparison.ipynb` | 06c, 06d, 06e (reads their shared test-prediction artifacts) | The F-DATA duration/memory raw-space collapse diagnostic (including a per-user inspection), the same-features classical-ML control, and the six-combination summary against classical ML |
| 14 | `06g_sequence_tie_variant.ipynb` | 06c, 06d, 06e (compares against their checkpoints/Optuna studies); PM100 only | A PM100-only sequence-construction variant (`tie_policy="skip_back"` in `src/sequences.py`, looks past a tied same-user job instead of excluding it): trains fresh LSTM/TCN under it for all three PM100 combinations, checks whether it closes any of the LSTM/TCN-vs-classical-ML gap from 06f, plus a determinism check and a bounded 3-seed robustness check reusing each pair's own already-found learning rate |
| 15 | `06h_fdata_sequence_tie_variant.ipynb` | 06c, 06d, 06e (same comparison references); F-DATA only | The F-DATA analogue of 06g: one fixed-learning-rate run per pair (lr reused from the original Optuna study, not re-searched, since a from-scratch F-DATA search was estimated at several days) for F-DATA duration/memory/power under `skip_back`, plus a runaway-prediction diagnostic that clips predictions to the training range to check how much of the raw-R2 change is extrapolation rather than a stable fit difference |
| 16 | `08b_dl_vs_classical_paired.ipynb` | `09` (every classical model `08b` uses, saved/refit and R2-verified there first); 06c, 06d, 06e, 06f (best DL model per combo) | The same user-clustered paired bootstrap as `08`, but DL vs classical instead of classical vs classical: for each of all six combinations, the best DL architecture by raw R2 (from `06f`) against that combination's adopted-best classical model. No classical model is refit here -- every one is loaded from `data/interim/`, either the originally-saved fit (`fdata_power`, `pm100_duration`) or the refit `09` already verified against the adopted R2. Reports raw R2, log R2, and MedAE deltas with 95% CIs and the share of resamples where classical wins, for all six combinations |
| 17 | `09_interpretability.ipynb` | 05, 05b, 05c (adopted best models/params); 06f (not required, but the discussion cross-checks against it) | SHAP feature importance (plus gain, side by side) for the adopted-best classical model of all six combinations, top-10 figure and table per combination. `fdata_power` and `pm100_duration` reuse a saved fitted model; `pm100_power` and `pm100_memory_c4` are refit inline (seconds, already-adopted `best_params`, seed 0); `fdata_duration` and `fdata_memory` are refit by `scripts/notebook09_interpretability/refit_and_verify.py`, one at a time under its own `systemd-run --user --scope`, on the full training split, with the fit timed for real. Every refit's test R2 is checked against the adopted value (within 0.001) before SHAP runs on it |

Within notebook 06's parts, 06c/06d/06e must each finish before 06b or 06f run (both read artifacts 06c/06d/06e write); 06c/06d/06e should not run concurrently with each other since each loads the full F-DATA base population. 06a has no dependency on the others and no shared-artifact consumers. 06g and 06h are variant explorations, not a step in the main sequence -- they only need 06c/06d/06e's artifacts to compare against and can run any time after those three finish (06g before or after 06b/06f; 06h likewise).

**06g/06h result CSVs are intentionally outside `results/results_ledger.csv`.** `scripts/build_results_ledger.py` only reads `results/notebook06_sweep_result_*.csv` (the adopted `exclude`-policy sweep, section 5b of that script) and `results/notebook06_same_features_control_*.csv` (section 5c); it has no section for `results/notebook06g_sweep_result_*_skipback.csv` or `results/notebook06h_sweep_result_*_skipback.csv`. This is deliberate, not an oversight: every `_skipback` row already carries `status="diagnostic"` and a `source` field disclosing the fixed/reused learning rate, and 06g/06h are themselves variant checks on the adopted `exclude`-policy sweep, not a replacement for it -- folding them into the ledger would risk them being read as competing, equally-adopted results. They stay as committed CSVs next to the ledger, documented here and cited directly from 06g/06h's own text, rather than merged into the curated table.

Deep learning (06a to 06h), the significance-testing notebook (08, classical models only),
`08b` (DL vs classical, all six combinations), and `09` (interpretability, all six
combinations) are all complete. `07_hybrid_model.ipynb` is not yet started -- scope-limited
to power on PM100 (no analytical baseline exists for PM100 duration or either dataset's
memory target), pending advisor confirmation of that scope.

## Data layout

```
data/raw/fdata/*.parquet
data/raw/pm100/pm100_job_table.parquet
```

Neither dataset ships with the repo. Both are public accounting traces from their HPC centers; the main README names the systems.

## Environment

- Notebooks import from `src/` (`features`, `splits`, `metrics`, `plotting`, `roofline`, `baselines`, `microbench`, `models`, `config`). Run from the project root, or put it on the path.
- `04` needs a CUDA GPU. The rest run on CPU. Full-scale F-DATA wants a lot of RAM: the job-name embedding column alone can need more than 100 GB if loaded naively, so notebook 01 loads every other column on its own and streams the embedding one month at a time.
- `05` and `05b` have a `SKIP_TUNING` flag (`SKIP_TUNING_PM100` for the PM100 sections). When it is set, the notebook reuses recorded best hyperparameters. A fresh Optuna search for a new target runs for hours; reusing stored parameters takes minutes. The committed outputs are real full-scale runs, not skipped ones, and carry no `injected-parameters` cell.
- `05b` Part 2 (PM100 memory) runs in a fresh kernel, separate from Part 1. Chaining long tuning sessions in one process introduced thread-pool contention that distorted the timing numbers, so the two parts are split.
- `05c` and `05d` load pre-computed results rather than tuning inline (`05c`'s own first attempt was killed by a genuine OOM after over 30 hours; the fix restructured tuning into one fresh, short-lived process per model instead of one long-lived kernel accumulating multiple models' state). See [`scripts/decision6_seed_variance/README.md`](../scripts/decision6_seed_variance/README.md) for the underlying pipeline and its `systemd-run --user --scope` launch pattern.

## Figures

Saved to `notebooks/figures/`, named by dataset and target (`pm100_memory_metrics_heatmap.png`). The thesis draft embeds a curated subset; not everything generated here appears in it.

## Additive sections

Notebook 05 Part 1 keeps its original F-DATA duration results and appends a second set with extended user-history window features. Notebook 05b Part 2 keeps its original PM100 memory results and appends a section that adopts the reduced feature set. In both, the earlier results sit next to the revised ones and the reason for the change is written into the notebook.
