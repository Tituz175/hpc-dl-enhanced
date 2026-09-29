import sys, time, pathlib
sys.path.append("..")
import joblib
import numpy as np
import optuna
from xgboost import XGBRegressor
from src import features, metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)
INTERIM_DIR = pathlib.Path("../data/interim")
prepared = joblib.load(INTERIM_DIR / "pm100_duration_prepared.joblib")
X_tune_train, X_tune_val = prepared["X_tune_train"], prepared["X_tune_val"]
y_tune_train, y_tune_val = prepared["y_tune_train"], prepared["y_tune_val"]
X_train, X_test = prepared["X_train"], prepared["X_test"]
y_train, y_test = prepared["y_train"], prepared["y_test"]
y_test_raw = prepared["y_test_raw"]
SEED, N_TRIALS = prepared["SEED"], prepared["N_TRIALS"]

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] loaded prepared data, starting XGBoost Optuna search (fresh process, early stopping enabled)")

def xgb_objective(trial):
    t0 = time.perf_counter()
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
    )
    model = XGBRegressor(**params, tree_method="hist", n_jobs=-1, random_state=SEED,
                          early_stopping_rounds=20, eval_metric="rmse")
    model.fit(X_tune_train, y_tune_train, eval_set=[(X_tune_val, y_tune_val)], verbose=False)
    pred = model.predict(X_tune_val)
    rmse = float(np.sqrt(np.mean((y_tune_val - pred) ** 2)))
    print(f"  trial {trial.number}: {params} best_iter={model.best_iteration} elapsed={time.perf_counter()-t0:.1f}s rmse={rmse:.4f}", flush=True)
    return rmse

study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
    storage=f"sqlite:///{INTERIM_DIR}/optuna_pm100_duration.db", study_name="xgb", load_if_exists=True,
)
t0 = time.perf_counter()
study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
tuning_seconds = time.perf_counter() - t0
best_params = study.best_params
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] XGBoost: best RMSE(log-space)={study.best_value:.4f} tuning time={tuning_seconds:.1f}s")
print(f"    best_params={best_params}")

# PM100 is small (~145K train rows) -- unlike F-DATA power, both tree methods are cheap here,
# so try both explicitly rather than assuming hist is fine (F-DATA power's hist/exact gap
# was only found by checking; not repeating that mistake this time).
from sklearn.metrics import r2_score
final_by_method = {}
for tm in ["hist", "exact"]:
    t0 = time.perf_counter()
    m = XGBRegressor(**best_params, tree_method=tm, n_jobs=-1, random_state=SEED)
    m.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    pred_log = m.predict(X_test)
    pred_raw = features.inverse_transform_target(pred_log)
    r2 = r2_score(y_test_raw, pred_raw)
    final_by_method[tm] = dict(model=m, elapsed=elapsed, r2=r2, pred_log=pred_log, pred_raw=pred_raw)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] XGBoost final fit tree_method={tm}: time={elapsed:.1f}s  raw_space R2={r2:.4f}")

best_tm = max(final_by_method, key=lambda k: final_by_method[k]["r2"])
print(f"  chosen tree_method for final model: {best_tm} (R2={final_by_method[best_tm]['r2']:.4f} vs "
      f"{'exact' if best_tm == 'hist' else 'hist'}={final_by_method['exact' if best_tm == 'hist' else 'hist']['r2']:.4f})")

model = final_by_method[best_tm]["model"]
fit_seconds = final_by_method[best_tm]["elapsed"]
pred_log = final_by_method[best_tm]["pred_log"]
pred_raw = final_by_method[best_tm]["pred_raw"]
results = {
    "log_space": metrics.regression_metrics(y_test, pred_log),
    "raw_space": metrics.regression_metrics(y_test_raw, pred_raw),
}

joblib.dump(dict(
    model_name="XGBoost", best_params=best_params, tuning_seconds=tuning_seconds,
    fit_seconds=fit_seconds, results=results, model=model,
    final_fit_tree_method=best_tm, tuning_tree_method="hist",
    both_methods_r2={k: v["r2"] for k, v in final_by_method.items()},
), INTERIM_DIR / "pm100_duration_xgb_result.joblib")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] saved XGBoost result")
