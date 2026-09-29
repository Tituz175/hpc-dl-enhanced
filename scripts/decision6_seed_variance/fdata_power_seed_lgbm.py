import sys, time, pathlib
sys.path.append("..")
import joblib
import numpy as np
from lightgbm import LGBMRegressor
from src import features, metrics

INTERIM_DIR = pathlib.Path("../data/interim")
prepared = joblib.load(INTERIM_DIR / "fdata_power_prepared.joblib")
X_train, X_test = prepared["X_train"], prepared["X_test"]
y_train, y_test = prepared["y_train"], prepared["y_test"]
y_test_raw = prepared["y_test_raw"]

existing = joblib.load(INTERIM_DIR / "fdata_power_lgbm_result.joblib")
best_params = existing["best_params"]
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] LightGBM seed sweep, best_params={best_params}")

seed_results = {0: dict(R2=existing["results"]["raw_space"]["R2"], fit_seconds=existing["fit_seconds"], reused=True)}
for seed in [1, 2, 3, 4]:
    model = LGBMRegressor(**best_params, bagging_freq=1, n_jobs=-1, random_state=seed, verbose=-1)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    pred_log = model.predict(X_test)
    pred_raw = features.inverse_transform_target(pred_log)
    r2 = metrics.regression_metrics(y_test_raw, pred_raw)["R2"]
    seed_results[seed] = dict(R2=r2, fit_seconds=fit_seconds, reused=False)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] seed={seed}: R2={r2:.4f}  fit_time={fit_seconds:.1f}s")

r2_values = [v["R2"] for v in seed_results.values()]
print(f"\nLightGBM seed sweep (n=5, seeds 0-4): R2 values={[round(v,4) for v in r2_values]}")
print(f"  mean={np.mean(r2_values):.4f}  std={np.std(r2_values):.4f}  range=[{min(r2_values):.4f}, {max(r2_values):.4f}]")

joblib.dump(dict(model_name="LightGBM", seed_results=seed_results),
            INTERIM_DIR / "fdata_power_seedvar_lgbm.joblib")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] saved")
