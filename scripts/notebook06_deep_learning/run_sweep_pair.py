"""
The full deep learning sweep, one (combination, architecture) pair per process. Runs a persistent-
storage Optuna study (n_trials=10, lr log-uniform 1e-5 to 1e-3, the only search dimension),
each trial training up to 20 epochs with early-stopping patience 5 and gradient clipping, on
a chronological tune_train/tune_val split within the combination's train data. After the
study reaches 10 completed trials, the WINNING trial's own saved checkpoint (the model
state at its own best epoch -- literally "the fit at the best hyperparameters", not a
separate refit) is reloaded and evaluated on the real held-out test split, raw and log space,
written to a per-pair results CSV (merged into results/results_ledger.csv separately, to
avoid concurrent-write races across the 4 processes running at once).

LSTM/TCN take BOTH the history sequence AND the current (target) job's own scaled feature
vector -- confirmed as a real bug and fixed (src/models.py's SequenceLSTM/SequenceTCN
docstrings): build_sequence_index only ever gathers strictly-prior jobs, so without this fix
the current job's own resource request/size never reached the sequence models at all.

Persistent storage means a crash leaves a record: completed trials are committed to the
SQLite DB as they finish, so rerunning this exact command resumes from however many trials
already completed rather than starting over.

The data-prep and train/tune/evaluate logic (prepare_combo_data, build_model, run_pair) live
in src/dl_pipeline.py, shared with notebooks 06b/06c/06d/06e -- this script is a thin driver:
load one combination's data, run (or resume) that pair's tuning, write the results row.

Usage: uv run python scripts/notebook06_deep_learning/run_sweep_pair.py <combo> <arch>
  combo: fdata_power | fdata_duration | fdata_memory | pm100_power | pm100_duration | pm100_memory_c4
  arch:  FNN | LSTM | TCN
"""
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.dl_pipeline import prepare_combo_data, run_pair

COMBO = sys.argv[1]
ARCH = sys.argv[2]
assert COMBO in ("fdata_power", "fdata_duration", "fdata_memory",
                  "pm100_power", "pm100_duration", "pm100_memory_c4")
assert ARCH in ("FNN", "LSTM", "TCN")

DEVICE = torch.device("cuda:0")
RESULTS_DIR = ROOT / "results"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [{COMBO}/{ARCH}] {msg}", flush=True)


log("loading data")
t0 = time.perf_counter()
prepared = prepare_combo_data(COMBO, log_fn=log)
log(f"loaded in {time.perf_counter() - t0:.0f}s: tune_train={len(prepared['tt_pos']):,} "
    f"tune_val={len(prepared['tv_pos']):,} test={len(prepared['test_pos']):,}")

result = run_pair(prepared, ARCH, device=DEVICE, log_fn=log, epoch_log_dir=RESULTS_DIR)

log(f"TEST log-space: {result['log_metrics']}")
log(f"TEST raw-space: {result['raw_metrics']}")

dataset, target = {
    "fdata_power": ("fdata", "power"), "fdata_duration": ("fdata", "duration"),
    "fdata_memory": ("fdata", "memory"), "pm100_power": ("pm100", "power"),
    "pm100_duration": ("pm100", "duration"), "pm100_memory_c4": ("pm100", "memory"),
}[COMBO]

rows = []
for space, m in [("log", result["log_metrics"]), ("raw", result["raw_metrics"])]:
    for metric_name, value in m.items():
        rows.append(dict(dataset=dataset, target=target, model=ARCH, variant="optuna_sweep",
                          tree_method=None, fit_rows=result["fit_rows"], seed=0, space=space,
                          metric=metric_name, value=value, status="adopted",
                          source=f"notebook06_sweep:scripts/notebook06_deep_learning/"
                                 f"run_sweep_pair.py (best_trial={result['best_trial']}, "
                                 f"lr={result['best_lr']:.2e}; fit_rows is tune_train, "
                                 f"80% of the real train split -- the winning Optuna trial's "
                                 f"own checkpoint is used directly as the final model, no "
                                 f"separate full-train refit; the deep learning feature set "
                                 f"differs from classical ML's -- epoch-timestamp and "
                                 f"per-split frequency columns dropped, factorize-coded "
                                 f"categoricals replaced with one-hot fit on train+test "
                                 f"combined, see src/dl_data.py)",
                          is_seed_sweep=False))
rows.append(dict(dataset=dataset, target=target, model=ARCH, variant="optuna_sweep",
                  tree_method=None, fit_rows=result["fit_rows"], seed=0, space=None,
                  metric="best_lr", value=result["best_lr"], status="adopted",
                  source="notebook06_sweep", is_seed_sweep=False))

pd.DataFrame(rows).to_csv(RESULTS_DIR / f"notebook06_sweep_result_{COMBO}_{ARCH}.csv", index=False)
log(f"SWEEP PAIR DONE: {COMBO}/{ARCH} -- results saved to "
    f"results/notebook06_sweep_result_{COMBO}_{ARCH}.csv")
