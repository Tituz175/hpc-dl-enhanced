# Deep Learning Enhanced Prediction of Execution Time, Memory Usage, and Power Consumption in Heterogeneous HPC Systems

**Master's thesis research, Computer Science, New Mexico Highlands University**
Tobi Titus Oyekanmi ([toyekanmi@live.nmhu.edu](mailto:toyekanmi@live.nmhu.edu))

## What this is

A shared HPC scheduler has to place jobs and size their allocations before any of them run. This thesis asks whether machine learning, given only what a user supplies at submission time, predicts a job's execution time, memory use, and power draw more accurately than the analytical models schedulers have traditionally relied on.

Two job-accounting traces anchor the work, chosen because they disagree in ways that matter: one CPU-only system that records hardware performance counters, one CPU-plus-GPU system that does not.

## Datasets

| | F-DATA | PM100 |
|---|---|---|
| System | Fugaku (RIKEN) | Marconi100 (CINECA) |
| Node architecture | A64FX, CPU-only | IBM POWER9 + NVIDIA V100, CPU + GPU |
| Records | 25,866,900 jobs (38 monthly files, March 2021 to April 2024) | 231,238 jobs (May to October 2020) |
| After the completed-jobs filter | 23,308,381 (9.9% removed) | 180,310 (22.0% removed) |
| FLOP / performance-counter fields | yes | none |

Both are publicly released accounting datasets from their respective centers. Neither is included in this repository; the notebook guide gives the expected layout.

The FLOP-data gap is why the two systems get different analytical treatments rather than one treatment applied twice.

## Method

**Tier A / Tier B feature discipline.** Only fields known before a job runs are eligible for prediction: requested resources, queue and priority, the wall-time limit, an anonymized job-name embedding, and a user's own recent job history. Everything measured during or after execution is Tier B, built by a separate function, with an assertion that fails if the two sets ever share a column. That check re-runs after every transformation step, not once at the start.

**Four-check feature vetting.** A candidate feature with unusual missingness is put through four questions before it joins the active set: is the missingness a real optional-flag pattern or just incomplete logging; how large is its effect on each target; does an existing column already carry the same information; and is the effect broad or the signature of a few heavy users. The last question overrode the others once. A flag with the largest raw effect of any candidate, jobs roughly four times longer when it was set, turned out to come almost entirely from one user out of the six who ever set it, and was held out. Running the same check later on features already in use caught a second one: 13 non-zero rows across 180,000 jobs, every one from a single user.

**Analytical baselines.** F-DATA gets a Hierarchical Roofline model built from A64FX peak specifications and the dataset's own FLOP and bandwidth counters. PM100 gets a resource-utilization power model whose form is fixed by hardware reasoning, an idle draw per node plus a linear term per allocated resource, with only the coefficients fit to training data. These are the floors the learned models have to clear.

**Classical ML.** Random Forest, XGBoost, and LightGBM, each given the same 50-trial Optuna budget so the comparison stays fair. Targets are modeled in log space and reported back in real units. The final fit uses the full training split; Random Forest alone is capped at five million rows for that fit, disclosed wherever its numbers appear.

## Results so far

All six target/dataset combinations are done.

| Target | Dataset | Analytical baseline | Best classical ML |
|---|---|---|---|
| Execution time | F-DATA | Roofline, R² = -0.16 | XGBoost, R² = 0.84 |
| Execution time | PM100 | none possible (no FLOP/bandwidth fields) | LightGBM, R² = 0.70 |
| Power | F-DATA | Calibrated model, R² = 0.98 | XGBoost, R² = 0.99 |
| Power | PM100 | Calibrated model, R² = 0.91 | XGBoost, R² = 0.92 &Dagger; |
| Memory | F-DATA | none possible (no physical model for memory) | LightGBM, R² = 0.87 |
| Memory | PM100 | none possible | XGBoost, R² = 0.97 &dagger; |

The best model is not fixed. LightGBM leads F-DATA duration until extended user-history features are added, and then XGBoost takes it (0.82 to 0.84). Of the remaining four: XGBoost leads F-DATA power and PM100 memory, is statistically tied with LightGBM on PM100 power, and LightGBM leads F-DATA memory and PM100 duration, the weakest-performing combination in this thesis (naive baseline R² = 0.41, best model 0.70). Treating "best model" as a property of the configuration rather than of the pipeline is one of the study's own conclusions, reinforced by a multi-seed check (5 random-seed refits per model, 3 for F-DATA Random Forest given its cost): the *top* model's ranking holds up in 5 of 6 combinations, but the runner-up ordering is only reliably established in one.

&dagger; PM100 memory needed an investigation before that number could be trusted. Requested memory matched allocated memory exactly on 94% of test jobs, which let every model return a stored value instead of learning, and then miss on the 6% of jobs where request and allocation diverge, the cases prediction is actually for. Dropping the feature improved XGBoost outright. For Random Forest and LightGBM it was a deliberate trade: the full feature set scored a higher aggregate R² (0.98) but a much worse median error on the divergent jobs, 35 to 44 MB against under 1 MB. A four-feature set was adopted for all three models and re-tuned per model.

&Dagger; XGBoost and LightGBM are statistically tied on PM100 power once seed variance is accounted for (seed-level R² ranges overlap); both clear Random Forest with no overlap.

## Findings worth pulling out

**A hand-built formula and a model that never saw it agreed on the same variable.** The PM100 power model scales its idle-power term by node count, a choice made from hardware reasoning. Feature-importance analysis of the XGBoost model, which was given no such prior, puts requested node count at 61 to 70% of its total signal, ahead of every other feature. Power per node is close to fixed on Marconi100, so node count and power move together, and the two approaches reached it independently. LightGBM, trained separately and structurally different (leaf-wise growth instead of level-wise), lands on the same feature (SHAP 61%, gain-based importance 90%), so this is agreement between two independent model families, not an artifact of picking one to report.

**Roofline's negative R² is not a broken baseline.** Recomputing each job's compute-bound or memory-bound label from Roofline first principles matches F-DATA's own label 99.97% of the time, so the construction is sound. It still scores R² = -0.16 on duration, because the ceiling assumes a job runs at peak rate for its entire wall-clock time and real jobs spend much of theirs on I/O and synchronization. Both statements hold at once.

**A strongly correlated feature can hurt rather than just fail to help.** Requested memory correlates about 0.84 with allocated memory on PM100, yet removing it raised XGBoost's accuracy across the board, not only on the hard cases. The feature let the model substitute lookup for learning so completely that the tuning objective itself was misled, preferring configurations that generalized worse.

**The GPU and memory coefficients in the PM100 power model came out negative.** That is physically backwards. The four allocated-resource predictors correlate 0.85 to 0.97 with each other, because Marconi100 nodes hand out cores, GPUs, and memory in a near-fixed ratio, so ordinary least squares cannot attribute the outcome cleanly to any one of them. The model predicts well; its individual coefficients are reported as fitted, not adjusted to look sensible.

**Independent validation of the Roofline construction.** Notebook 04 times four GPU kernels whose compute-bound or memory-bound regime is known in advance, and checks that the same ceiling formula and classification logic recover it. `vector_add` and `dot_product` classify memory-bound in every run; the largest matrix multiply classifies compute-bound, at 75% of the FP32 peak and 95% of the FP64 peak. This is separate from the 99.97% label agreement above. One check is against a dataset's own labels; the other is against ground truth that does not come from a dataset at all.

**A library-specific bug, not a data problem, on F-DATA power.** XGBoost's histogram-based tree construction (`tree_method="hist"`, its default) catastrophically underfits F-DATA power's feature set (R² ≈ 0.31) while scoring 0.99 with exact splits (`tree_method="exact"`) at the same hyperparameters. Ruled out directly, not assumed: reproducibility, hyperparameter choice, the embedding-PCA columns, and float32/float64 precision were each tested and eliminated in turn. LightGBM's own histogram implementation shows no equivalent failure on the same data. The fix (tune with `hist`, final-fit with `exact`) cost about 59 minutes once; tuning under `exact` for all 50 trials would have cost far more.

**Two fields with different definitions, identical values.** F-DATA's `mszl` ("memory size limit for the job") and `msza` ("memory allocated") are conceptually distinct, a request versus a scheduling outcome, but are byte-identical on every row checked across all 38 months, sentinel and non-sentinel values alike. Among the roughly 2.9 million non-sentinel rows, the limit takes only 21 distinct values (a small set of site-standard tiers, not a per-job custom number) and correlates weakly (0.19) with requested node/core count.

## Repository layout

```
notebooks/   01 to 05d complete (all six target/dataset combinations, plus multi-seed variance);
             06 to 09 scaffolded (deep learning, hybrid, evaluation, interpretability)
src/         features, splits, metrics, plotting, roofline, baselines, microbench, models, config
scripts/     decision6_seed_variance/ (multi-seed refit pipeline, one fresh process per model);
             build_results_ledger.py (regenerates results/, reading only saved joblibs and
             committed notebook outputs, never re-fitting); fdata_power_calibrated_baseline/
results/     results_ledger.csv (every metric, every seed, both datasets), tuning_details.csv,
             seed_variance.csv, investigation_log.md (diagnostics with no persisted artifact)
writing/     thesis draft (build_thesis_doc.py builds thesis_draft_notes.docx) and its figures
data/raw/    datasets go here; not tracked
```

## Running it

The environment is managed with `uv`, pinned to Python 3.12 because `numba`, a SHAP dependency, has no working 3.13 build. `uv sync` installs everything. Notebook execution order and per-notebook requirements are in [`notebooks/README.md`](notebooks/README.md).

## Status

Complete: classical ML (Random Forest, XGBoost, LightGBM) for all six target/dataset combinations, plus a multi-seed variance check on every one of them (5 seeds generally, 3 for F-DATA Random Forest given its cost). Not yet done: paired significance testing across seeds (planned around a user-clustered bootstrap of the R² difference over test-set jobs, not a seed-level Wilcoxon test, which is underpowered by construction at 3 to 5 seeds). Deep learning (feedforward, LSTM, TCN) is scoped and scaffolded but not implemented. The analytical-plus-deep-learning hybrid is scope-limited to power on PM100 (no analytical baseline exists for PM100 duration or either dataset's memory target) and left for later.

## Reference

`EXPERIMENT_TRACKER.md`, kept out of version control, holds the dated log of decisions, dead ends, and methodology notes that the notebooks refer to. Every result quoted above is also in [`results/results_ledger.csv`](results/results_ledger.csv), a long-format table (dataset, target, model, seed, space, metric, value, and source for every row) generated by [`scripts/build_results_ledger.py`](scripts/build_results_ledger.py), which reads only saved model artifacts and committed notebook outputs, never re-fitting anything. [`results/investigation_log.md`](results/investigation_log.md) covers the handful of real diagnostics that produced no artifact precise enough to put in that table. [`scripts/decision6_seed_variance/README.md`](scripts/decision6_seed_variance/README.md) covers the multi-seed pipeline: run order, the `systemd-run` launch pattern this environment requires for genuinely persistent background processes, and where each combination's `best_params` came from.
