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

Four of the six target/dataset combinations are done.

| Target | Dataset | Analytical baseline | Best classical ML |
|---|---|---|---|
| Execution time | F-DATA | Roofline, R² = -0.16 | XGBoost, R² = 0.84 |
| Power | PM100 | Calibrated model, R² = 0.91 | XGBoost, R² = 0.92 |
| Memory | F-DATA | none possible (no physical model for memory) | LightGBM, R² = 0.87 |
| Memory | PM100 | none possible | XGBoost, R² = 0.97 &dagger; |

The best model is not fixed. LightGBM leads F-DATA duration until extended user-history features are added, and then XGBoost takes it (0.82 to 0.84). XGBoost leads both PM100 targets. Treating "best model" as a property of the configuration rather than of the pipeline is one of the study's own conclusions.

&dagger; PM100 memory needed an investigation before that number could be trusted. Requested memory matched allocated memory exactly on 94% of test jobs, which let every model return a stored value instead of learning, and then miss on the 6% of jobs where request and allocation diverge, the cases prediction is actually for. Dropping the feature improved XGBoost outright. For Random Forest and LightGBM it was a deliberate trade: the full feature set scored a higher aggregate R² (0.98) but a much worse median error on the divergent jobs, 35 to 44 MB against under 1 MB. A four-feature set was adopted for all three models and re-tuned per model.

## Findings worth pulling out

**A hand-built formula and a model that never saw it agreed on the same variable.** The PM100 power model scales its idle-power term by node count, a choice made from hardware reasoning. Feature-importance analysis of the XGBoost model, which was given no such prior, puts requested node count at 61 to 70% of its total signal, ahead of every other feature. Power per node is close to fixed on Marconi100, so node count and power move together, and the two approaches reached it independently.

**Roofline's negative R² is not a broken baseline.** Recomputing each job's compute-bound or memory-bound label from Roofline first principles matches F-DATA's own label 99.97% of the time, so the construction is sound. It still scores R² = -0.16 on duration, because the ceiling assumes a job runs at peak rate for its entire wall-clock time and real jobs spend much of theirs on I/O and synchronization. Both statements hold at once.

**A strongly correlated feature can hurt rather than just fail to help.** Requested memory correlates about 0.84 with allocated memory on PM100, yet removing it raised XGBoost's accuracy across the board, not only on the hard cases. The feature let the model substitute lookup for learning so completely that the tuning objective itself was misled, preferring configurations that generalized worse.

**The GPU and memory coefficients in the PM100 power model came out negative.** That is physically backwards. The four allocated-resource predictors correlate 0.85 to 0.97 with each other, because Marconi100 nodes hand out cores, GPUs, and memory in a near-fixed ratio, so ordinary least squares cannot attribute the outcome cleanly to any one of them. The model predicts well; its individual coefficients are reported as fitted, not adjusted to look sensible.

**Independent validation of the Roofline construction.** Notebook 04 times four GPU kernels whose compute-bound or memory-bound regime is known in advance, and checks that the same ceiling formula and classification logic recover it. `vector_add` and `dot_product` classify memory-bound in every run; the largest matrix multiply classifies compute-bound, at 75% of the FP32 peak and 95% of the FP64 peak. This is separate from the 99.97% label agreement above. One check is against a dataset's own labels; the other is against ground truth that does not come from a dataset at all.

## Repository layout

```
notebooks/   01 to 05b complete; 06 to 09 scaffolded (deep learning, hybrid, evaluation, interpretability)
src/         features, splits, metrics, plotting, roofline, baselines, microbench, models, config
writing/     thesis draft (build_thesis_doc.py builds thesis_draft_notes.docx) and its figures
data/raw/    datasets go here; not tracked
```

## Running it

The environment is managed with `uv`, pinned to Python 3.12 because `numba`, a SHAP dependency, has no working 3.13 build. `uv sync` installs everything. Notebook execution order and per-notebook requirements are in [`notebooks/README.md`](notebooks/README.md).

## Status

Complete: classical ML for F-DATA duration and memory, and PM100 power and memory. Not yet done: F-DATA power, and PM100 duration, which has no analytical baseline because PM100 records no FLOP data. Deep learning (feedforward, LSTM, TCN) is scoped and scaffolded but not implemented. The analytical-plus-deep-learning hybrid is scope-limited to power on PM100 and left for later.

## Reference

`EXPERIMENT_TRACKER.md`, kept out of version control, holds the dated log of decisions, dead ends, and methodology notes that the notebooks refer to.
