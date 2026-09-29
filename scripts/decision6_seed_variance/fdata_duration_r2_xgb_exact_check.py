import sys, time, pathlib
sys.path.append("..")
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src import features

INTERIM_DIR = pathlib.Path("../data/interim")
prepared = joblib.load(INTERIM_DIR / "fdata_duration_r2_prepared.joblib")
X_train, X_test = prepared["X_train"], prepared["X_test"]
y_train, y_test_raw = prepared["y_train"], prepared["y_test_raw"]
best_params = prepared["best_params"]["XGBoost"]

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] F-DATA duration Result 2 XGBoost hist-vs-exact check, best_params={best_params}")
t0 = time.perf_counter()
m = XGBRegressor(**best_params, tree_method="exact", n_jobs=-1, random_state=0)
m.fit(X_train, y_train)
elapsed = time.perf_counter() - t0
pred_log = m.predict(X_test)
pred_raw = features.inverse_transform_target(pred_log)
r2 = r2_score(y_test_raw, pred_raw)
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] tree_method=exact: R2={r2:.4f}  elapsed={elapsed:.1f}s  "
      f"(hist reference, seed=0: R2=0.8359)")

joblib.dump(dict(hist_r2=0.8359, exact_r2=r2, exact_elapsed=elapsed),
            INTERIM_DIR / "fdata_duration_r2_xgb_histexact_check.joblib")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] saved")
