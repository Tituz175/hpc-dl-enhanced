import sys, time, pathlib
sys.path.append("..")
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src import features

INTERIM_DIR = pathlib.Path("../data/interim")
prepared = joblib.load(INTERIM_DIR / "fdata_memory_prepared.joblib")
X_train, X_test = prepared["X_train"], prepared["X_test"]
y_train, y_test_raw = prepared["y_train"], prepared["y_test_raw"]
y_train_raw = features.inverse_transform_target(y_train)
best_params = prepared["best_params"]["XGBoost"]

for seed in [2, 4]:
    print(f"\n{'='*70}\nseed={seed}\n{'='*70}")
    t0 = time.perf_counter()
    m = XGBRegressor(**best_params, tree_method="exact", n_jobs=-1, random_state=seed)
    m.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    booster = m.get_booster()
    print(f"fit time: {elapsed:.1f}s")
    print(f"n_estimators param: {m.n_estimators}  actual boosted rounds: {booster.num_boosted_rounds()}")

    pred_log_train = m.predict(X_train)
    pred_log_test = m.predict(X_test)
    print(f"any NaN in train preds: {np.isnan(pred_log_train).any()}  any inf: {np.isinf(pred_log_train).any()}")
    print(f"any NaN in test preds: {np.isnan(pred_log_test).any()}  any inf: {np.isinf(pred_log_test).any()}")

    train_r2 = r2_score(y_train, pred_log_train)
    test_r2_log = r2_score(prepared["y_test"], pred_log_test)
    pred_raw = features.inverse_transform_target(pred_log_test)
    test_r2_raw = r2_score(y_test_raw, pred_raw)
    print(f"TRAIN log-space R2: {train_r2:.4f}")
    print(f"TEST  log-space R2: {test_r2_log:.4f}")
    print(f"TEST  raw-space  R2: {test_r2_raw:.4f}")

    print(f"pred_log_test stats: mean={pred_log_test.mean():.4f} std={pred_log_test.std():.4f} "
          f"min={pred_log_test.min():.4f} max={pred_log_test.max():.4f}")
    print(f"y_test (log) stats:  mean={prepared['y_test'].mean():.4f} std={prepared['y_test'].std():.4f} "
          f"min={prepared['y_test'].min():.4f} max={prepared['y_test'].max():.4f}")

    # feature importance -- is one seed relying on a wildly different dominant feature?
    gain = m.get_booster().get_score(importance_type="gain")
    top5 = sorted(gain.items(), key=lambda kv: -kv[1])[:5]
    print(f"top-5 features by gain: {top5}")

    joblib.dump(dict(seed=seed, model=m, train_r2=train_r2, test_r2_log=test_r2_log, test_r2_raw=test_r2_raw),
                INTERIM_DIR / f"fdata_memory_xgb_exact_seed{seed}_diagnostic.joblib")

print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] done")
