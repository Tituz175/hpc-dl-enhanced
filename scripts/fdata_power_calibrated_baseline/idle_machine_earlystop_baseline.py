"""
Uncontended re-measurement of the two LightGBM fit points from the crash investigation's
per-fit diagnostic (n_estimators=200/num_leaves=100 "mid" and n_estimators=500/num_leaves=255
"expensive"), WITHOUT early stopping, on an otherwise-idle machine -- confirmed via `free -h`
and `ps aux` immediately before launch (idle, no other lgbm/xgb/papermill/isolation/diagnostic
process running, swap empty).

Uses the REAL saved tuning matrices from fdata_power_prepared.joblib (X_tune_train/X_tune_val,
800k/200k rows, 93 columns including the genuine 80-component SBert embedding PCA already fit
on this exact tuning sample) -- not a synthetic reconstruction. A first version of this script
used a zero-filled placeholder for the embedding columns instead of loading them; that was
wrong, not just a disclosed simplification: a constant column gives LightGBM's histogram
builder a single degenerate bin, which is far cheaper to split on than a real embedding
component's actual value distribution across 800k rows, so the zero-filled run's timings were
not a valid stand-in for the real fit's cost and have been discarded (see
results/investigation_log.md section 1 for that correction and what these real-data numbers
replace it with).

Reports elapsed time and peak RSS (resource.getrusage, ru_maxrss).

Run with: uv run python scripts/fdata_power_calibrated_baseline/idle_machine_earlystop_baseline.py
"""
import pathlib
import resource
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np

INTERIM_DIR = ROOT / "data" / "interim"
SEED = 0

t0 = time.perf_counter()

prep = joblib.load(INTERIM_DIR / "fdata_power_prepared.joblib")
X_tune_train, X_tune_val = prep["X_tune_train"], prep["X_tune_val"]
y_tune_train, y_tune_val = prep["y_tune_train"], prep["y_tune_val"]
print(f"[{time.strftime('%H:%M:%S')}] loaded real tuning matrices: "
      f"train={X_tune_train.shape} val={X_tune_val.shape}  "
      f"RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")

results = {}
for label, n_estimators, num_leaves in [("mid", 200, 100), ("expensive", 500, 255)]:
    print(f"[{time.strftime('%H:%M:%S')}] {label} point: n_estimators={n_estimators} ceiling, "
          f"num_leaves={num_leaves}, NO early stopping, idle machine, real embedding data")
    params = dict(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8)
    t_fit0 = time.perf_counter()
    model = lgb.LGBMRegressor(**params, bagging_freq=1, n_jobs=-1, random_state=SEED, verbose=-1)
    model.fit(X_tune_train, y_tune_train)
    elapsed = time.perf_counter() - t_fit0
    pred = model.predict(X_tune_val)
    rmse = float(np.sqrt(np.mean((y_tune_val - pred) ** 2)))
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    per_round = elapsed / n_estimators
    print(f"[{time.strftime('%H:%M:%S')}] {label}: elapsed={elapsed:.2f}s  "
          f"({per_round:.4f}s/round over {n_estimators} rounds, all built -- no early stopping)  "
          f"rmse={rmse:.4f}  peak_rss_so_far={peak_rss_gb:.2f} GB")
    results[label] = dict(n_estimators=n_estimators, num_leaves=num_leaves,
                           elapsed_seconds=elapsed, seconds_per_round=per_round,
                           rmse=rmse, peak_rss_gb_cumulative=peak_rss_gb)

total_elapsed = time.perf_counter() - t0
final_peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"\n[{time.strftime('%H:%M:%S')}] done in {total_elapsed:.1f}s total  "
      f"final peak RSS: {final_peak_rss_gb:.2f} GB")

print("\n=== comparison against the original, contended 2026-09-23 measurement ===")
print(f"  mid:       original=1799.5s (contended)  idle-machine (real data)={results['mid']['elapsed_seconds']:.2f}s")
print(f"  expensive: original=4.0s (early-stopped, not comparable)  "
      f"idle-machine (real data, no early stopping)={results['expensive']['elapsed_seconds']:.2f}s")
print("\nNOTE: this isolates the cost of individual fits at these two hyperparameter points. It "
      "does NOT by itself explain the original crashed run's full multi-hour LightGBM tuning "
      "stage duration -- that would require summing real per-trial costs across however many "
      "Optuna trials LightGBM actually ran before the OOM, which was never logged and is not "
      "reconstructed here. Treat the per-fit numbers above as exactly that: per-fit, not a "
      "reconstruction of the full stage.")

INTERIM_DIR.mkdir(exist_ok=True)
out_path = INTERIM_DIR / "fdata_power_idle_machine_earlystop_baseline.joblib"
joblib.dump(dict(results=results, total_elapsed_seconds=total_elapsed,
                  final_peak_rss_gb=final_peak_rss_gb,
                  data_source="real fdata_power_prepared.joblib X_tune_train/X_tune_val, "
                              "not a synthetic/zero-filled reconstruction"), out_path)
print(f"\nsaved: {out_path}")
