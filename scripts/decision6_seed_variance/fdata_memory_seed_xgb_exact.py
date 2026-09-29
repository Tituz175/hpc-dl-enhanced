import sys, time, pathlib
sys.path.append("..")
import joblib
import numpy as np
from xgboost import XGBRegressor
from src import features, metrics

INTERIM_DIR = pathlib.Path("../data/interim")
prepared = joblib.load(INTERIM_DIR / "fdata_memory_prepared.joblib")
X_train, X_test = prepared["X_train"], prepared["X_test"]
y_train, y_test_raw = prepared["y_train"], prepared["y_test_raw"]
best_params = prepared["best_params"]["XGBoost"]
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] XGBoost EXACT seed sweep (F-DATA memory), best_params={best_params}, "
      f"tree_method=exact. Escalated from the hist sweep (std=0.0152, and a hist-vs-exact check at "
      f"seed=0 showing a real 0.0354 gap) -- same escalate-when-wide approach used throughout this "
      f"multi-seed variance work (repeat each model's final fit across multiple seeds to test whether "
      f"a 'best model' claim is robust to seed-to-seed stochasticity) -- see README.md in this directory.")

seed_results = {}
for seed in [0, 1, 2, 3, 4]:
    model = XGBRegressor(**best_params, tree_method="exact", n_jobs=-1, random_state=seed)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    pred_log = model.predict(X_test)
    pred_raw = features.inverse_transform_target(pred_log)
    r2 = metrics.regression_metrics(y_test_raw, pred_raw)["R2"]
    seed_results[seed] = dict(R2=r2, fit_seconds=fit_seconds)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] seed={seed}: R2={r2:.4f}  fit_time={fit_seconds:.1f}s", flush=True)

r2_values = [v["R2"] for v in seed_results.values()]
print(f"\nXGBoost EXACT seed sweep (n=5, seeds 0-4): R2 values={[round(v,4) for v in r2_values]}")
print(f"  mean={np.mean(r2_values):.4f}  std={np.std(r2_values):.4f}  range=[{min(r2_values):.4f}, {max(r2_values):.4f}]")

joblib.dump(dict(model_name="XGBoost", seed_results=seed_results, tree_method="exact"),
            INTERIM_DIR / "fdata_memory_seedvar_xgb_exact.joblib")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] saved")
