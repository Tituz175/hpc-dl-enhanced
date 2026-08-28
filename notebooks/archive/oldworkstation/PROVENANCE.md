# Old-workstation notebook archive

Snapshot of notebooks 01-05 as executed on the **original workstation**,
kept for the record before the full-scale re-run on the replacement machine.

- **Archived:** 2026-08-27
- **Source commit:** `12d86c0` (`Add requirements.txt, generated from uv.lock`)
- **Git tag:** `results-oldworkstation-devscale`

These `.ipynb` files and their `.html` renders are the exact outputs that
commit `12d86c0` carries in `notebooks/`. The `.html` copies are here so the
results stay readable without a kernel and independent of future nbformat
or library drift. `figures/` is a copy of `notebooks/figures/` at the same
commit.

## Why superseded

The original workstation could not hold F-DATA's 384-dim embedding matrix
for all ~25.87M rows at once (125 GB RAM), and a single GPU bottlenecks the
deep-learning phase (notebooks 06-07). The replacement machine removes both
limits, so notebooks 01-03 and 05 are being re-run at full F-DATA scale and
notebook 04's Roofline is being re-measured on the new GPU.

| | Original workstation | Replacement workstation |
|---|---|---|
| CPU | Intel i9-10900X, 10c / 20t | Intel i9-10980XE, 18c / 36t |
| RAM | 125 GB | 251 GB |
| GPU | 1x RTX 3090 (24 GB) | 4x RTX 3090 Ti (24 GB each) |

## Scale each notebook ran at here (all dev-scale)

| Notebook | Scale on the original workstation |
|---|---|
| 01 data_acquisition_eda | 5-file random pool, `SAMPLE_SIZE = 5000` |
| 02 feature_engineering | 5-file random pool, `SAMPLE_SIZE = 5000` (Feature Vetting cells already ran on full PM100) |
| 03 analytical_baselines | F-DATA 6 consecutive months; PM100 full |
| 04 microbenchmark_validation | RTX 3090 published peaks: FP32 35.58 TFLOP/s, FP64 0.556 TFLOP/s, BW 936.2 GB/s |
| 05 classical_ml_baselines | F-DATA 6 months, `SAMPLE_SIZE = 1,000,000` train; PM100 full |

The replacement-workstation runs live in `notebooks/` at the current commit;
`EXPERIMENT_TRACKER.md` records the change and the new numbers.
