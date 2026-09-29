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

existing = joblib.load(INTERIM_DIR / "fdata_power_xgb_result.joblib")
best_params = existing["best_params"]
SAMPLE_ROWS = 1_000_000
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] XGBoost seed sweep, best_params={best_params}, "
      f"tree_method=exact, {SAMPLE_ROWS:,}-row sample per seed (a disclosed reduced-scale "
      f"cost-control measure -- see this directory's README for the run order); existing "
      f"full-scale R2=0.9888 is untouched and separate.")

seed_results = {}
for seed in [0, 1, 2, 3, 4]:
    idx = np.random.RandomState(seed).permutation(len(X_train))[:SAMPLE_ROWS]
    Xs, ys = X_train.iloc[idx], y_train[idx]
    model = XGBRegressor(**best_params, tree_method="exact", n_jobs=-1, random_state=seed)
    t0 = time.perf_counter()
    model.fit(Xs, ys)
    fit_seconds = time.perf_counter() - t0
    pred_log = model.predict(X_test)
    pred_raw = features.inverse_transform_target(pred_log)
    r2 = metrics.regression_metrics(y_test_raw, pred_raw)["R2"]
    seed_results[seed] = dict(R2=r2, fit_seconds=fit_seconds, reused=False, sample_rows=SAMPLE_ROWS)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] seed={seed}: R2={r2:.4f}  fit_time={fit_seconds:.1f}s")

r2_values = [v["R2"] for v in seed_results.values()]
print(f"\nXGBoost seed sweep (n=5, seeds 0-4, {SAMPLE_ROWS:,}-row sample each): "
      f"R2 values={[round(v,4) for v in r2_values]}")
print(f"  mean={np.mean(r2_values):.4f}  std={np.std(r2_values):.4f}  range=[{min(r2_values):.4f}, {max(r2_values):.4f}]")

joblib.dump(dict(model_name="XGBoost", seed_results=seed_results, sample_rows=SAMPLE_ROWS,
                  tree_method="exact", note="reduced-scale sweep -- superseded by the "
                  "full-scale escalation in fdata_power_seed_xgb_fullscale.py once this "
                  "sweep came back wide enough to warrant it"),
            INTERIM_DIR / "fdata_power_seedvar_xgb.joblib")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] saved")
