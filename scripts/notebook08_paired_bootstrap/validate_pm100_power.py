"""
Checkpoint script for notebook 08, run before the full 18-comparison sweep as requested:
(1) confirm the sufficient-statistics reconstruction reproduces the real, reported R2 exactly
    on the full (unresampled) test set, for all three PM100 power models;
(2) cross-check the fast (sufficient-statistics) cluster bootstrap against a brute-force
    row-level cluster bootstrap on one comparison, to confirm they agree before trusting the
    fast version on the other 17 comparisons and the larger F-DATA test sets.

PM100 power is all-cheap (all three models' 5-seed refits take seconds each), so this
combination also doubles as the worked example for the "use all five seeds" mechanism: each
seed's fitted model gives one independent (R2_A, R2_B) bootstrap distribution over the SAME
fixed test-set users; the 5 seeds' draws are pooled into one combined distribution, so the
final interval reflects primarily test-set sampling uncertainty while not depending on any
one arbitrarily-chosen seed's specific fit.

Run with: uv run python scripts/notebook08_paired_bootstrap/validate_pm100_power.py
"""
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src import features, splits
from bootstrap_lib import (compute_per_user_stats, r2_from_sufficient_stats,
                            run_cluster_bootstrap, run_rowlevel_bootstrap, top5_user_share)

INTERIM_DIR = ROOT / "data" / "interim"
PM100_PATH = ROOT / "data" / "raw" / "pm100" / "pm100_job_table.parquet"

t0 = time.perf_counter()

# ---- reconstruct test_df to recover user_id, aligned by index with the saved X_test ----
print(f"[{time.strftime('%H:%M:%S')}] reconstructing PM100 test_df for user_id alignment")
pm100 = pd.read_parquet(PM100_PATH)
pm100 = features.filter_completed_jobs(pm100, "pm100")
pm100["power_mean"] = pm100["node_power_consumption"].apply(np.mean)
pm100 = features.add_pm100_derived_indicators(pm100)
pm100 = features.add_user_rolling_stat(pm100, "pm100", "power_mean", window=5)
_, pm100_test_df = splits.chronological_split(pm100, "pm100")

prep = joblib.load(INTERIM_DIR / "pm100_power_prepared.joblib")
X_test = prep["X_test"]
y_test_raw = prep["y_test_raw"]
y_test_log = prep["y_test"]
best_params = prep["best_params"]

assert list(pm100_test_df.index) == list(X_test.index), \
    "reconstructed test_df index does not match saved X_test index -- pipeline mismatch"
user_ids = pm100_test_df.loc[X_test.index, "user_id"].to_numpy()
print(f"[{time.strftime('%H:%M:%S')}] index alignment confirmed exact match, "
      f"{len(np.unique(user_ids))} unique test users across {len(user_ids):,} rows")

# ---- refit all 3 models at seeds 0-4 (all cheap for PM100 power) ----
X_train = prep["X_train"]
y_train_log = prep["y_train"]

predictions_log = {}  # predictions_log[(model, seed)] = array, log space
for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
    params = best_params[model_name]
    for seed in range(5):
        if model_name == "RandomForest":
            m = RandomForestRegressor(**params, n_jobs=-1, random_state=seed)
        elif model_name == "XGBoost":
            m = XGBRegressor(**params, tree_method="hist", n_jobs=-1, random_state=seed)
        else:
            m = LGBMRegressor(**params, bagging_freq=1, n_jobs=-1, random_state=seed, verbose=-1)
        m.fit(X_train, y_train_log)
        predictions_log[(model_name, seed)] = np.asarray(m.predict(X_test), dtype=float)
    print(f"[{time.strftime('%H:%M:%S')}] {model_name}: 5 seeds refit and predicted")

# ============================================================================
# Check 1: sufficient-statistics reconstruction reproduces the real R2 exactly
# ============================================================================
print("\n=== Check 1: sufficient-statistics reconstruction vs. real reported R2 ===")
real_r2_raw = {
    "RandomForest": 0.9018224268573586,  # seed 0, from pm100_power_seedvar_rf.joblib
    "XGBoost": 0.9243268257105238,       # seed 0, from pm100_power_seedvar_xgb.joblib (hist)
    "LightGBM": 0.9237474761737103,      # seed 0, from pm100_power_seedvar_lgbm.joblib
}
all_passed = True
for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
    pred_raw = features.inverse_transform_target(predictions_log[(model_name, 0)])
    stats = compute_per_user_stats(user_ids, y_test_raw, pred_raw, pred_raw)  # A==B here, just need R2_A
    # reconstruct with ALL users at multiplicity 1 (the unresampled full test set)
    n_total = stats["n_u"].sum()
    sy_total = stats["sy_u"].sum()
    sy2_total = stats["sy2_u"].sum()
    ssa_total = stats["ssa_u"].sum()
    reconstructed_r2 = r2_from_sufficient_stats(n_total, sy_total, sy2_total, ssa_total)
    diff = abs(reconstructed_r2 - real_r2_raw[model_name])
    # XGBoost's multi-threaded histogram construction (n_jobs=-1) is not bit-reproducible
    # across runs even at a fixed random_state, so its tolerance is loosened to 1e-6; RF and
    # LightGBM are deterministic given a fixed seed and are held to machine precision (1e-9).
    tol = 1e-6 if model_name == "XGBoost" else 1e-9
    passed = diff < tol
    all_passed &= passed
    print(f"  {model_name}: reconstructed={reconstructed_r2:.12f}  "
          f"real={real_r2_raw[model_name]:.12f}  diff={diff:.3e}  {'PASS' if passed else 'FAIL'}")

print(f"\nCheck 1 overall: {'ALL PASS' if all_passed else 'FAILED -- DO NOT PROCEED'}")
if not all_passed:
    sys.exit(1)

# ============================================================================
# Check 2: fast cluster bootstrap vs. brute-force row-level bootstrap
# ============================================================================
print("\n=== Check 2: fast (sufficient-stats) vs. brute-force row-level cluster bootstrap ===")
print("Comparison: XGBoost vs LightGBM, seed 0, raw space, 2000 iterations each")
pred_a_raw = features.inverse_transform_target(predictions_log[("XGBoost", 0)])
pred_b_raw = features.inverse_transform_target(predictions_log[("LightGBM", 0)])

stats_ab = compute_per_user_stats(user_ids, y_test_raw, pred_a_raw, pred_b_raw)
t_fast0 = time.perf_counter()
fast_r2a, fast_r2b = run_cluster_bootstrap(stats_ab, n_iterations=2000, seed=42)
t_fast = time.perf_counter() - t_fast0
fast_delta = fast_r2a - fast_r2b

t_brute0 = time.perf_counter()
brute_r2a, brute_r2b = run_rowlevel_bootstrap(user_ids, y_test_raw, pred_a_raw, pred_b_raw,
                                               n_iterations=2000, seed=42)
t_brute = time.perf_counter() - t_brute0
brute_delta = brute_r2a - brute_r2b

print(f"  fast (sufficient-stats):  mean(delta)={fast_delta.mean():.6f}  "
      f"std(delta)={fast_delta.std():.6f}  elapsed={t_fast:.2f}s")
print(f"  brute-force (row-level):  mean(delta)={brute_delta.mean():.6f}  "
      f"std(delta)={brute_delta.std():.6f}  elapsed={t_brute:.2f}s")
print(f"  mean(delta) difference: {abs(fast_delta.mean() - brute_delta.mean()):.6f}")
print(f"  std(delta) difference:  {abs(fast_delta.std() - brute_delta.std()):.6f}")
print(f"  speedup: {t_brute / t_fast:.1f}x")

ci_fast = np.percentile(fast_delta, [0.5, 99.5])
ci_brute = np.percentile(brute_delta, [0.5, 99.5])
print(f"  fast 99% CI:       [{ci_fast[0]:.6f}, {ci_fast[1]:.6f}]")
print(f"  brute-force 99% CI: [{ci_brute[0]:.6f}, {ci_brute[1]:.6f}]")

check2_passed = (abs(fast_delta.mean() - brute_delta.mean()) < 0.005 and
                  abs(fast_delta.std() - brute_delta.std()) < 0.005)
print(f"\nCheck 2: {'PASS (agree within 0.005)' if check2_passed else 'FAIL -- investigate before proceeding'}")

# ============================================================================
# Worked example: pooling all 5 seeds (the "use all five seeds" mechanism)
# ============================================================================
print("\n=== Worked example: XGBoost vs LightGBM, all 5 seeds pooled, raw space ===")
pooled_delta = []
for seed in range(5):
    pred_a_raw_s = features.inverse_transform_target(predictions_log[("XGBoost", seed)])
    pred_b_raw_s = features.inverse_transform_target(predictions_log[("LightGBM", seed)])
    stats_s = compute_per_user_stats(user_ids, y_test_raw, pred_a_raw_s, pred_b_raw_s)
    r2a_s, r2b_s = run_cluster_bootstrap(stats_s, n_iterations=2000, seed=100 + seed)
    pooled_delta.append(r2a_s - r2b_s)
    print(f"  seed {seed}: mean(delta)={((r2a_s - r2b_s).mean()):.6f}  "
          f"99% CI=[{np.percentile(r2a_s - r2b_s, 0.5):.6f}, {np.percentile(r2a_s - r2b_s, 99.5):.6f}]")
pooled_delta = np.concatenate(pooled_delta)
pooled_ci = np.percentile(pooled_delta, [0.5, 99.5])
print(f"  POOLED (5x2000=10000 draws): mean(delta)={pooled_delta.mean():.6f}  "
      f"99% CI=[{pooled_ci[0]:.6f}, {pooled_ci[1]:.6f}]")

# ============================================================================
# Verdict rule (applied to this worked example)
# ============================================================================
print("\n=== Verdict rule ===")
print("'supported': interval excludes zero AND the corresponding 05d seed ranges do not overlap.")
print("'supported on this test set, not robust to training seed': interval excludes zero BUT")
print("  the 05d seed ranges DO overlap.")
print("Otherwise: 'not supported'.")

# 05d's actual PM100 power seed ranges, from results/results_ledger.csv (is_seed_sweep=True,
# variant="seed_sweep", tree_method="hist" for XGBoost): XGBoost hist R2 across 5 seeds =
# {0.924327, 0.921704, 0.920704, 0.926036, 0.922032}; LightGBM R2 across 5 seeds =
# {0.923747, 0.927156, 0.928578, 0.911759, 0.931129}.
xgb_range_05d = (0.920704, 0.926036)   # XGBoost hist, 5 seeds, min/max
lgbm_range_05d = (0.911759, 0.931129)  # LightGBM, 5 seeds, min/max
seed_ranges_overlap = not (xgb_range_05d[1] < lgbm_range_05d[0] or lgbm_range_05d[1] < xgb_range_05d[0])

interval_excludes_zero = pooled_ci[0] > 0 or pooled_ci[1] < 0
if interval_excludes_zero and not seed_ranges_overlap:
    verdict = "supported"
elif interval_excludes_zero and seed_ranges_overlap:
    verdict = "supported on this test set, not robust to training seed"
else:
    verdict = "not supported"

print(f"\nXGBoost vs LightGBM, PM100 power, raw space:")
print(f"  pooled 99% CI: [{pooled_ci[0]:.6f}, {pooled_ci[1]:.6f}]  (excludes zero: {interval_excludes_zero})")
print(f"  05d seed ranges: XGBoost {xgb_range_05d}, LightGBM {lgbm_range_05d}  (overlap: {seed_ranges_overlap})")
print(f"  VERDICT: {verdict}")

row_share, ss_share = top5_user_share(stats_ab)
print(f"\nTop-5-user share: {row_share:.2%} of rows, {ss_share:.2%} of sum-of-squares "
      f"(n_users={stats_ab['n_users']}, n_rows={stats_ab['n_rows']:,})")

print(f"\n[{time.strftime('%H:%M:%S')}] total elapsed: {time.perf_counter() - t0:.1f}s")
