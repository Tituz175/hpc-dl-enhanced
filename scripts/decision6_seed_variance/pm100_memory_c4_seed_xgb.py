import sys, time, pathlib
sys.path.append("..")
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src import features, metrics

INTERIM_DIR = pathlib.Path("../data/interim")
prepared = joblib.load(INTERIM_DIR / "pm100_memory_clean4_prepared.joblib")
X_train, X_test = prepared["X_train"], prepared["X_test"]
y_train, y_test_raw = prepared["y_train"], prepared["y_test_raw"]
best_params = prepared["best_params"]["XGBoost"]

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] XGBoost hist-vs-exact check (PM100 memory clean-4, seed=0), best_params={best_params}")
for tm in ["hist", "exact"]:
    t0 = time.perf_counter()
    m = XGBRegressor(**best_params, tree_method=tm, n_jobs=-1, random_state=0)
    m.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    pred_log = m.predict(X_test)
    pred_raw = features.inverse_transform_target(pred_log)
    r2 = r2_score(y_test_raw, pred_raw)
    print(f"  tree_method={tm}: R2={r2:.4f}  elapsed={elapsed:.1f}s", flush=True)

print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] running full 5-seed sweep for BOTH methods (cheap here, no need to pick one)")
for tm in ["hist", "exact"]:
    seed_results = {}
    for seed in [0, 1, 2, 3, 4]:
        model = XGBRegressor(**best_params, tree_method=tm, n_jobs=-1, random_state=seed)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        pred_log = model.predict(X_test)
        pred_raw = features.inverse_transform_target(pred_log)
        r2 = metrics.regression_metrics(y_test_raw, pred_raw)["R2"]
        seed_results[seed] = dict(R2=r2, fit_seconds=fit_seconds)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] tree_method={tm} seed={seed}: R2={r2:.4f}  fit_time={fit_seconds:.1f}s", flush=True)
    r2_values = [v["R2"] for v in seed_results.values()]
    print(f"\n{tm} seed sweep (n=5): R2 values={[round(v,4) for v in r2_values]}")
    print(f"  mean={np.mean(r2_values):.4f}  std={np.std(r2_values):.4f}  range=[{min(r2_values):.4f}, {max(r2_values):.4f}]")
    joblib.dump(dict(model_name="XGBoost", seed_results=seed_results, tree_method=tm),
                INTERIM_DIR / f"pm100_memory_c4_seedvar_xgb_{tm}.joblib")

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] saved")
