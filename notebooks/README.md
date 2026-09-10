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

`06` to `09` (`deep_learning_models`, `hybrid_model`, `evaluation_and_statistics`, `interpretability`) are scaffolded but not yet implemented.

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

## Figures

Saved to `notebooks/figures/`, named by dataset and target (`pm100_memory_metrics_heatmap.png`). The thesis draft embeds a curated subset; not everything generated here appears in it.

## Additive sections

Notebook 05 Part 1 keeps its original F-DATA duration results and appends a second set with extended user-history window features. Notebook 05b Part 2 keeps its original PM100 memory results and appends a section that adopts the reduced feature set. In both, the earlier results sit next to the revised ones and the reason for the change is written into the notebook.
