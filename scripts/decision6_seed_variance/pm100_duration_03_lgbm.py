import sys, time, pathlib
sys.path.append("..")
import joblib
import numpy as np
import optuna
import lightgbm as lgb
from lightgbm import LGBMRegressor
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

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] loaded prepared data, starting LightGBM Optuna search (fresh process, early stopping enabled)")

def lgbm_objective(trial):
    t0 = time.perf_counter()
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
    )
    model = LGBMRegressor(**params, bagging_freq=1, n_jobs=-1, random_state=SEED, verbose=-1)
    model.fit(
        X_tune_train, y_tune_train,
        eval_set=[(X_tune_val, y_tune_val)], eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    pred = model.predict(X_tune_val)
    rmse = float(np.sqrt(np.mean((y_tune_val - pred) ** 2)))
    print(f"  trial {trial.number}: {params} best_iter={model.best_iteration_} elapsed={time.perf_counter()-t0:.1f}s rmse={rmse:.4f}", flush=True)
    return rmse

study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
    storage=f"sqlite:///{INTERIM_DIR}/optuna_pm100_duration.db", study_name="lgbm", load_if_exists=True,
)
t0 = time.perf_counter()
study.optimize(lgbm_objective, n_trials=N_TRIALS, show_progress_bar=False)
tuning_seconds = time.perf_counter() - t0
best_params = study.best_params
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] LightGBM: best RMSE(log-space)={study.best_value:.4f} tuning time={tuning_seconds:.1f}s")
print(f"    best_params={best_params}")

model = LGBMRegressor(**best_params, bagging_freq=1, n_jobs=-1, random_state=SEED, verbose=-1)
t0 = time.perf_counter()
model.fit(X_train, y_train)
fit_seconds = time.perf_counter() - t0
pred_log = model.predict(X_test)
pred_raw = features.inverse_transform_target(pred_log)
results = {
    "log_space": metrics.regression_metrics(y_test, pred_log),
    "raw_space": metrics.regression_metrics(y_test_raw, pred_raw),
}
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] LightGBM final fit time={fit_seconds:.1f}s  raw_space R2={results['raw_space']['R2']:.4f}")

joblib.dump(dict(
    model_name="LightGBM", best_params=best_params, tuning_seconds=tuning_seconds,
    fit_seconds=fit_seconds, results=results, model=model,
), INTERIM_DIR / "pm100_duration_lgbm_result.joblib")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] saved LightGBM result")
