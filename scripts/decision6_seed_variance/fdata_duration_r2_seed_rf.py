import sys, time, pathlib
sys.path.append("..")
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src import features, metrics

INTERIM_DIR = pathlib.Path("../data/interim")
prepared = joblib.load(INTERIM_DIR / "fdata_duration_r2_prepared.joblib")
X_train, X_test = prepared["X_train"], prepared["X_test"]
y_train, y_test_raw = prepared["y_train"], prepared["y_test_raw"]
rf_cap = prepared["rf_cap"]
best_params = prepared["best_params"]["RandomForest"]
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RF seed sweep (F-DATA duration Result 2), best_params={best_params}, rf_cap={rf_cap:,}")

seed_results = {}
for seed in [0, 1, 2]:
    rf_rows = np.random.RandomState(seed).permutation(len(X_train))[:rf_cap]
    model = RandomForestRegressor(**best_params, n_jobs=-1, random_state=seed)
    t0 = time.perf_counter()
    model.fit(X_train.iloc[rf_rows], y_train[rf_rows])
    fit_seconds = time.perf_counter() - t0
    pred_log = model.predict(X_test)
    pred_raw = features.inverse_transform_target(pred_log)
    r2 = metrics.regression_metrics(y_test_raw, pred_raw)["R2"]
    seed_results[seed] = dict(R2=r2, fit_seconds=fit_seconds)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] seed={seed}: R2={r2:.4f}  fit_time={fit_seconds:.1f}s", flush=True)

r2_values = [v["R2"] for v in seed_results.values()]
print(f"\nRF seed sweep (n=3, seeds 0/1/2): R2 values={[round(v,4) for v in r2_values]}")
print(f"  mean={np.mean(r2_values):.4f}  std={np.std(r2_values):.4f}  range=[{min(r2_values):.4f}, {max(r2_values):.4f}]")

joblib.dump(dict(model_name="RandomForest", seed_results=seed_results, rf_cap=rf_cap),
            INTERIM_DIR / "fdata_duration_r2_seedvar_rf.joblib")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] saved")
