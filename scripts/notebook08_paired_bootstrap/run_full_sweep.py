"""
Notebook 08 full 18-comparison sweep: user-clustered paired bootstrap of the R2 difference,
for every pairwise model comparison (RandomForest vs XGBoost, RandomForest vs LightGBM,
XGBoost vs LightGBM), within each of the six target/dataset combinations.

Methodology validated on PM100 power in validate_pm100_power.py before this was written:
sufficient-statistics reconstruction reproduces the real reported R2 exactly (to float
tolerance), and the fast cluster bootstrap agrees with a brute-force row-level cluster
bootstrap to 6 decimal places. Every fresh refit here re-checks its seed-0 R2 against the
real ledger value as it goes, as a running correctness guard, not just a one-time check.

Cost classification (fixed before running, matches the architecture already used for
notebooks 05d's own seed sweep): F-DATA RandomForest (all three targets) and F-DATA power's
XGBoost (tree_method=exact) are "expensive" -- refitting 5 times is not worth it, so these
use the canonical seed=0 model only (loaded from its already-saved *_result.joblib where one
exists, else refit once at seed=0), and any comparison touching one of these models is
restricted to seed=0 for BOTH models in that pair (can't pool a 5-seed model against a
1-seed model without conflating seed-variance into the test-sampling estimate) and flagged
explicitly. Every other model in every other combination is "cheap": refit at seeds 0-4,
each seed's cluster bootstrap (2000 iterations) pooled into one 10,000-draw distribution, per
the approach validated and confirmed with the user (pooling reflects seed-to-seed spread on
top of test-set sampling noise, without anointing one arbitrary seed's fit as canonical).

05d seed ranges below are read fresh from results/results_ledger.csv (raw-space R2,
is_seed_sweep=True, variant="seed_sweep" -- i.e., each combination's ALREADY-ADOPTED XGBoost
tree_method, not the alternate-tree-method or reduced-sample diagnostic variants).
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

from src import features
from bootstrap_lib import (compute_per_user_stats, r2_from_sufficient_stats,
                            run_cluster_bootstrap, top5_user_share)

INTERIM_DIR = ROOT / "data" / "interim"
RESULTS_DIR = ROOT / "results"
N_ITER = 2000
CI_PCT = [0.5, 99.5]  # 99% percentile interval, per spec point 5 (18 comparisons -> 99%)

# 05d's seed ranges (raw space, R2), from results/results_ledger.csv, seed_sweep variant only
SEED_RANGES = {
    ("fdata", "power", "RandomForest"): (0.984739, 0.986353),
    ("fdata", "power", "XGBoost"): (0.988506, 0.990262),
    ("fdata", "power", "LightGBM"): (0.981674, 0.985661),
    ("fdata", "duration", "RandomForest"): (0.820470, 0.822800),
    ("fdata", "duration", "XGBoost"): (0.835891, 0.840090),
    ("fdata", "duration", "LightGBM"): (0.823677, 0.830648),
    ("fdata", "memory", "RandomForest"): (0.814852, 0.824637),
    ("fdata", "memory", "XGBoost"): (0.807001, 0.849787),
    ("fdata", "memory", "LightGBM"): (0.851075, 0.868058),
    ("pm100", "power", "RandomForest"): (0.890953, 0.902963),
    ("pm100", "power", "XGBoost"): (0.920704, 0.926036),
    ("pm100", "power", "LightGBM"): (0.911759, 0.931129),
    ("pm100", "duration", "RandomForest"): (0.601131, 0.616194),
    ("pm100", "duration", "XGBoost"): (0.580415, 0.618513),
    ("pm100", "duration", "LightGBM"): (0.688084, 0.703246),
    ("pm100", "memory", "RandomForest"): (0.949992, 0.951950),
    ("pm100", "memory", "XGBoost"): (0.963078, 0.965823),
    ("pm100", "memory", "LightGBM"): (0.951866, 0.963049),
}
# seed=0 raw-space R2, from the same ledger rows -- used as the running correctness check
# against every fresh refit (and against canonical-model predictions where no refit is done)
REAL_R2_SEED0 = {
    ("fdata", "power", "RandomForest"): 0.985584,
    ("fdata", "power", "XGBoost"): 0.988850,
    ("fdata", "power", "LightGBM"): 0.985491,
    ("fdata", "duration", "RandomForest"): 0.820493,
    ("fdata", "duration", "XGBoost"): 0.835891,
    ("fdata", "duration", "LightGBM"): 0.830648,
    ("fdata", "memory", "RandomForest"): 0.824637,
    ("fdata", "memory", "XGBoost"): 0.807001,
    ("fdata", "memory", "LightGBM"): 0.866805,
    ("pm100", "power", "RandomForest"): 0.901822,
    ("pm100", "power", "XGBoost"): 0.924327,
    ("pm100", "power", "LightGBM"): 0.923747,
    ("pm100", "duration", "RandomForest"): 0.605701,
    ("pm100", "duration", "XGBoost"): 0.618513,
    ("pm100", "duration", "LightGBM"): 0.699380,
    ("pm100", "memory", "RandomForest"): 0.950889,
    ("pm100", "memory", "XGBoost"): 0.965823,
    ("pm100", "memory", "LightGBM"): 0.959930,
}

COMBOS = {
    "pm100_power": dict(
        dataset="pm100", target="power", prepared="pm100_power_prepared.joblib",
        user_ids_key="pm100_power", user_ids_file="notebook08_pm100_user_ids.joblib",
        tree_method_xgb="hist", expensive=set(), canonical={},
    ),
    "pm100_duration": dict(
        dataset="pm100", target="duration", prepared="pm100_duration_prepared.joblib",
        user_ids_key="pm100_duration", user_ids_file="notebook08_pm100_user_ids.joblib",
        tree_method_xgb="hist", expensive=set(), canonical={},
        best_params_source={"RandomForest": "pm100_duration_rf_result.joblib",
                             "XGBoost": "pm100_duration_xgb_result.joblib",
                             "LightGBM": "pm100_duration_lgbm_result.joblib"},
    ),
    "pm100_memory_c4": dict(
        dataset="pm100", target="memory", prepared="pm100_memory_clean4_prepared.joblib",
        user_ids_key="pm100_memory_c4", user_ids_file="notebook08_pm100_user_ids.joblib",
        tree_method_xgb="hist", expensive=set(), canonical={},
    ),
    "fdata_power": dict(
        dataset="fdata", target="power", prepared="fdata_power_prepared.joblib",
        user_ids_key="fdata_power", user_ids_file="notebook08_fdata_user_ids.joblib",
        tree_method_xgb="exact", expensive={"RandomForest", "XGBoost"},
        canonical={"RandomForest": "fdata_power_rf_result.joblib",
                   "XGBoost": "fdata_power_xgb_result.joblib"},
        best_params_source={"RandomForest": "fdata_power_rf_result.joblib",
                             "XGBoost": "fdata_power_xgb_result.joblib",
                             "LightGBM": "fdata_power_lgbm_result.joblib"},
    ),
    "fdata_duration": dict(
        dataset="fdata", target="duration", prepared="fdata_duration_r2_prepared.joblib",
        user_ids_key="fdata_duration", user_ids_file="notebook08_fdata_user_ids.joblib",
        tree_method_xgb="hist", expensive={"RandomForest"}, canonical={},
    ),
    "fdata_memory": dict(
        dataset="fdata", target="memory", prepared="fdata_memory_prepared.joblib",
        user_ids_key="fdata_memory", user_ids_file="notebook08_fdata_user_ids.joblib",
        tree_method_xgb="hist", expensive={"RandomForest"}, canonical={},
    ),
}
PAIRS = [("RandomForest", "XGBoost"), ("RandomForest", "LightGBM"), ("XGBoost", "LightGBM")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_one(model_name, params, X_train, y_train_log, seed, tree_method_xgb, rf_cap):
    if model_name == "RandomForest":
        if rf_cap is not None and rf_cap < len(X_train):
            rows = np.random.RandomState(seed).permutation(len(X_train))[:rf_cap]
            X_fit, y_fit = X_train.iloc[rows], y_train_log[rows]
        else:
            X_fit, y_fit = X_train, y_train_log
        m = RandomForestRegressor(**params, n_jobs=-1, random_state=seed)
    elif model_name == "XGBoost":
        X_fit, y_fit = X_train, y_train_log
        m = XGBRegressor(**params, tree_method=tree_method_xgb, n_jobs=-1, random_state=seed)
    else:
        X_fit, y_fit = X_train, y_train_log
        m = LGBMRegressor(**params, bagging_freq=1, n_jobs=-1, random_state=seed, verbose=-1)
    m.fit(X_fit, y_fit)
    return m


def check_r2(dataset, target, model_name, pred_log, user_ids, y_test_raw):
    pred_raw = features.inverse_transform_target(pred_log)
    stats = compute_per_user_stats(user_ids, y_test_raw, pred_raw, pred_raw)
    recon = r2_from_sufficient_stats(stats["n_u"].sum(), stats["sy_u"].sum(),
                                      stats["sy2_u"].sum(), stats["ssa_u"].sum())
    real = REAL_R2_SEED0[(dataset, target, model_name)]
    diff = abs(recon - real)
    # All three model families use n_jobs=-1 (multi-threaded fit/predict), which is not
    # bit-reproducible run-to-run even at a fixed random_state (parallel reduction order is
    # not fixed) -- confirmed empirically across all three during this sweep, not just
    # XGBoost. 1e-6 is far tighter than any real result would need to move a verdict.
    tol = 1e-6
    status = "OK" if diff < tol else "MISMATCH"
    log(f"    seed0 R2 check [{model_name}]: reconstructed={recon:.6f} real={real:.6f} "
        f"diff={diff:.2e} [{status}]")
    if status == "MISMATCH":
        raise RuntimeError(f"{dataset}_{target} {model_name}: seed0 R2 mismatch, diff={diff:.2e}")


all_rows = []

for combo_name, cfg in COMBOS.items():
    log(f"=== {combo_name} ===")
    prep = joblib.load(INTERIM_DIR / cfg["prepared"])
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train_log = prep["y_train"]
    y_test_raw, y_test_log = prep["y_test_raw"], prep["y_test"]
    if "best_params" in prep:
        best_params = prep["best_params"]
    else:
        best_params = {m: joblib.load(INTERIM_DIR / p)["best_params"]
                        for m, p in cfg["best_params_source"].items()}
    rf_cap = prep.get("rf_cap")
    user_ids_all = joblib.load(INTERIM_DIR / cfg["user_ids_file"])
    user_ids = user_ids_all[cfg["user_ids_key"]]
    assert len(user_ids) == len(X_test), f"{combo_name}: user_ids length mismatch vs X_test"

    predictions_log = {}  # (model, seed) -> log-space predictions array
    for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        if model_name in cfg["expensive"]:
            if model_name in cfg["canonical"]:
                result = joblib.load(INTERIM_DIR / cfg["canonical"][model_name])
                pred = np.asarray(result["model"].predict(X_test), dtype=float)
                log(f"  {model_name}: loaded canonical model, predicted (no refit)")
            else:
                t0 = time.perf_counter()
                m = fit_one(model_name, best_params[model_name], X_train, y_train_log, 0,
                            cfg["tree_method_xgb"], rf_cap)
                pred = np.asarray(m.predict(X_test), dtype=float)
                log(f"  {model_name}: expensive refit at seed=0 done "
                    f"({time.perf_counter() - t0:.0f}s)")
            predictions_log[(model_name, 0)] = pred
            check_r2(cfg["dataset"], cfg["target"], model_name, pred, user_ids, y_test_raw)
        else:
            t0 = time.perf_counter()
            for seed in range(5):
                m = fit_one(model_name, best_params[model_name], X_train, y_train_log, seed,
                            cfg["tree_method_xgb"], rf_cap)
                predictions_log[(model_name, seed)] = np.asarray(m.predict(X_test), dtype=float)
            log(f"  {model_name}: 5 seeds refit ({time.perf_counter() - t0:.0f}s total)")
            check_r2(cfg["dataset"], cfg["target"], model_name, predictions_log[(model_name, 0)],
                      user_ids, y_test_raw)

    stats_top5 = compute_per_user_stats(user_ids, y_test_raw, y_test_raw, y_test_raw)
    row_share, ss_share = top5_user_share(stats_top5)

    for pair_idx, (model_a, model_b) in enumerate(PAIRS):
        pair_expensive = (model_a in cfg["expensive"]) or (model_b in cfg["expensive"])
        seeds_to_use = [0] if pair_expensive else [0, 1, 2, 3, 4]

        for space_idx, space in enumerate(["raw", "log"]):
            y_true = y_test_raw if space == "raw" else y_test_log
            per_seed_mean = []
            all_draws = []
            for seed in seeds_to_use:
                pred_a = predictions_log[(model_a, seed)]
                pred_b = predictions_log[(model_b, seed)]
                if space == "raw":
                    pred_a = features.inverse_transform_target(pred_a)
                    pred_b = features.inverse_transform_target(pred_b)
                stats = compute_per_user_stats(user_ids, y_true, pred_a, pred_b)
                boot_seed = 10_000 * seed + 100 * pair_idx + space_idx
                r2a, r2b = run_cluster_bootstrap(stats, n_iterations=N_ITER, seed=boot_seed)
                delta = r2a - r2b
                per_seed_mean.append(float(delta.mean()))
                all_draws.append(delta)

            pooled = np.concatenate(all_draws)
            ci_lo, ci_hi = np.percentile(pooled, CI_PCT)
            excludes_zero = (ci_lo > 0) or (ci_hi < 0)
            signs = [np.sign(v) for v in per_seed_mean if v != 0]
            sign_disagreement = len(set(signs)) > 1 if len(seeds_to_use) > 1 else False

            range_a = SEED_RANGES[(cfg["dataset"], cfg["target"], model_a)]
            range_b = SEED_RANGES[(cfg["dataset"], cfg["target"], model_b)]
            ranges_overlap = not (range_a[1] < range_b[0] or range_b[1] < range_a[0])

            if excludes_zero and not ranges_overlap:
                verdict = "supported"
            elif excludes_zero and ranges_overlap:
                verdict = "supported on this test set, not robust to training seed"
            else:
                verdict = "not supported"

            all_rows.append(dict(
                combo=combo_name, dataset=cfg["dataset"], target=cfg["target"],
                model_a=model_a, model_b=model_b, space=space,
                expensive_seed0_only=pair_expensive,
                n_seeds_used=len(seeds_to_use),
                mean_delta=float(pooled.mean()), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                excludes_zero=excludes_zero,
                per_seed_mean_delta=per_seed_mean,
                sign_disagreement=sign_disagreement,
                seed_range_a=range_a, seed_range_b=range_b, ranges_overlap=ranges_overlap,
                verdict=verdict,
                n_unique_users=int(stats_top5["n_users"]), n_rows=int(stats_top5["n_rows"]),
                top5_row_share=row_share, top5_ss_share=ss_share,
            ))
            log(f"  {model_a} vs {model_b} [{space}]: mean_delta={pooled.mean():.6f} "
                f"99% CI=[{ci_lo:.6f}, {ci_hi:.6f}] sign_disagree={sign_disagreement} "
                f"verdict='{verdict}'")

    del predictions_log, X_train, X_test, prep
    log(f"=== {combo_name} done ===\n")

results_df = pd.DataFrame(all_rows)
results_df.to_json(RESULTS_DIR / "notebook08_bootstrap_results.json", orient="records", indent=2)
log(f"saved {len(results_df)} rows to {RESULTS_DIR / 'notebook08_bootstrap_results.json'}")
log("FULL SWEEP COMPLETE")
