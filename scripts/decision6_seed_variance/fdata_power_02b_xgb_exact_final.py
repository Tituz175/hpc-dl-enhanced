import sys, time, pathlib
sys.path.append("..")
import joblib
import numpy as np
from xgboost import XGBRegressor
from src import features, metrics

INTERIM_DIR = pathlib.Path("../data/interim")
prepared = joblib.load(INTERIM_DIR / "fdata_power_prepared.joblib")
X_train, X_test = prepared["X_train"], prepared["X_test"]
y_train, y_test = prepared["y_train"], prepared["y_test"]
y_test_raw = prepared["y_test_raw"]
SEED = prepared["SEED"]

old = joblib.load(INTERIM_DIR / "fdata_power_xgb_result.joblib")
best_params = old["best_params"]
tuning_seconds = old["tuning_seconds"]
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] refitting XGBoost final model with tree_method='exact' "
      f"(hist/approx confirmed broken on this feature set; exact recovers full accuracy -- see tracker)")
print(f"best_params (found via hist-based tuning, reused as-is): {best_params}")

model = XGBRegressor(**best_params, tree_method="exact", n_jobs=-1, random_state=SEED)
t0 = time.perf_counter()
model.fit(X_train, y_train)
fit_seconds = time.perf_counter() - t0
pred_log = model.predict(X_test)
pred_raw = features.inverse_transform_target(pred_log)
results = {
    "log_space": metrics.regression_metrics(y_test, pred_log),
    "raw_space": metrics.regression_metrics(y_test_raw, pred_raw),
}
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] XGBoost (exact) final fit time={fit_seconds:.1f}s")
print(f"    raw_space R2={results['raw_space']['R2']:.4f}  MAPE={results['raw_space']['MAPE']:.2f}")

joblib.dump(dict(
    model_name="XGBoost", best_params=best_params, tuning_seconds=tuning_seconds,
    fit_seconds=fit_seconds, results=results, model=model,
    final_fit_tree_method="exact", tuning_tree_method="hist",
    note="tuning used tree_method=hist (fast, 50 trials); hist/approx confirmed to badly "
         "underfit this feature set at the final-fit scale (R2 ~0.31 vs 0.99), root-caused "
         "to XGBoost's histogram binning specifically (not embedding columns, not float32 "
         "precision, not hyperparameter choice -- all tested directly). Final model refit "
         "with tree_method=exact, same best_params, at a real but one-time ~58min cost.",
), INTERIM_DIR / "fdata_power_xgb_result.joblib")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] saved corrected XGBoost result")
