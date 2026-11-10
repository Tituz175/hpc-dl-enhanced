"""
One-off script: refits the adopted best classical model for ONE F-DATA combination (duration
or memory) on its full training split, times the fit for real (not extrapolated from a
sample), verifies the refit's test R2 matches the already-adopted value within 0.001, and
saves the fitted model object to data/interim/ (gitignored) for notebooks 09 and 08b to load
without refitting again.

Run one combination at a time, each under its own systemd-run --user --scope, per this
project's standing memory-safety convention for F-DATA jobs. Usage:
    python scripts/notebook09_interpretability/refit_and_verify.py fdata_duration
    python scripts/notebook09_interpretability/refit_and_verify.py fdata_memory
"""
import pathlib
import sys
import time

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

ROOT = pathlib.Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT / "data" / "interim"
RESULTS_DIR = ROOT / "results"

CONFIG = {
    "fdata_duration": dict(
        prepared="fdata_duration_r2_prepared.joblib",
        model_name="XGBoost", tree_method="hist",
        out_file="fdata_duration_xgb_result.joblib",
        adopted_r2=0.8359,
    ),
    "fdata_memory": dict(
        prepared="fdata_memory_prepared.joblib",
        model_name="LightGBM", tree_method=None,
        out_file="fdata_memory_lgbm_result.joblib",
        adopted_r2=0.8668,
    ),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(combo):
    cfg = CONFIG[combo]
    prep = joblib.load(INTERIM_DIR / cfg["prepared"])
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train_log, y_test_raw = prep["y_train"], prep["y_test_raw"]
    params = prep["best_params"][cfg["model_name"]]
    log(f"{combo}: loaded {cfg['prepared']}, X_train={len(X_train):,} rows, "
        f"X_test={len(X_test):,} rows, params={params}")

    t0 = time.perf_counter()
    if cfg["model_name"] == "XGBoost":
        model = XGBRegressor(**params, tree_method=cfg["tree_method"], n_jobs=-1, random_state=0)
    else:
        model = LGBMRegressor(**params, bagging_freq=1, n_jobs=-1, random_state=0, verbose=-1)
    model.fit(X_train, y_train_log)
    fit_seconds = time.perf_counter() - t0
    log(f"{combo}: real fit on the FULL {len(X_train):,}-row training split took "
        f"{fit_seconds:.1f}s ({cfg['model_name']}, no sampling, no extrapolation)")

    pred_log = np.asarray(model.predict(X_test), dtype=float)
    pred_raw = np.expm1(pred_log)
    refit_r2 = r2_score(y_test_raw, pred_raw)
    diff = abs(refit_r2 - cfg["adopted_r2"])
    passed = diff < 0.001
    log(f"{combo}: refit test raw R2 = {refit_r2:.6f}, adopted = {cfg['adopted_r2']}, "
        f"diff = {diff:.6f}, within 0.001: {passed}")

    if not passed:
        log(f"{combo}: STOP -- refit R2 does not match the adopted value within 0.001. "
            f"Not saving a model object.")
        sys.exit(1)

    joblib.dump(dict(model_name=cfg["model_name"], model=model, best_params=params,
                      tree_method=cfg["tree_method"], fit_seconds=fit_seconds,
                      refit_test_r2=float(refit_r2), adopted_r2=cfg["adopted_r2"],
                      r2_check_diff=float(diff)),
                INTERIM_DIR / cfg["out_file"])
    log(f"{combo}: saved {cfg['out_file']}")


if __name__ == "__main__":
    main(sys.argv[1])
