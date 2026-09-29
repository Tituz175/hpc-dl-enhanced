import sys, time, pathlib
sys.path.append("..")
import joblib
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from src import features, metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

INTERIM_DIR = pathlib.Path("../data/interim")
prepared = joblib.load(INTERIM_DIR / "fdata_power_prepared.joblib")
X_tune_train, X_tune_val = prepared["X_tune_train"], prepared["X_tune_val"]
y_tune_train, y_tune_val = prepared["y_tune_train"], prepared["y_tune_val"]
X_train, X_test = prepared["X_train"], prepared["X_test"]
y_train, y_test = prepared["y_train"], prepared["y_test"]
y_test_raw = prepared["y_test_raw"]
SEED, N_TRIALS, rf_cap = prepared["SEED"], prepared["N_TRIALS"], prepared["rf_cap"]

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] loaded prepared data, starting RandomForest Optuna search (fresh process)")

def rf_objective(trial):
    t0 = time.perf_counter()
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 250),
        max_depth=trial.suggest_int("max_depth", 4, 18),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_features=trial.suggest_float("max_features", 0.3, 1.0),
    )
    model = RandomForestRegressor(**params, n_jobs=-1, random_state=SEED)
    model.fit(X_tune_train, y_tune_train)
    pred = model.predict(X_tune_val)
    rmse = float(np.sqrt(np.mean((y_tune_val - pred) ** 2)))
    print(f"  trial {trial.number}: {params} elapsed={time.perf_counter()-t0:.1f}s rmse={rmse:.4f}", flush=True)
    return rmse

study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
    storage=f"sqlite:///{INTERIM_DIR}/optuna_fdata_power.db", study_name="rf", load_if_exists=True,
)
t0 = time.perf_counter()
study.optimize(rf_objective, n_trials=N_TRIALS, show_progress_bar=False)
tuning_seconds = time.perf_counter() - t0
best_params = study.best_params
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RandomForest: best RMSE(log-space)={study.best_value:.4f} "
      f"tuning time={tuning_seconds:.1f}s n_trials={N_TRIALS}")
print(f"    best_params={best_params}")

# Final fit on the full (capped) train split, evaluate on full test split
rf_rows = np.random.RandomState(SEED).permutation(len(X_train))[:rf_cap]
model = RandomForestRegressor(**best_params, n_jobs=-1, random_state=SEED)
t0 = time.perf_counter()
model.fit(X_train.iloc[rf_rows], y_train[rf_rows])
fit_seconds = time.perf_counter() - t0
pred_log = model.predict(X_test)
pred_raw = features.inverse_transform_target(pred_log)
results = {
    "log_space": metrics.regression_metrics(y_test, pred_log),
    "raw_space": metrics.regression_metrics(y_test_raw, pred_raw),
}
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RandomForest final fit on {rf_cap:,} rows, time={fit_seconds:.1f}s")
print(f"    raw_space R2={results['raw_space']['R2']:.4f}")

joblib.dump(dict(
    model_name="RandomForest", best_params=best_params, tuning_seconds=tuning_seconds,
    fit_seconds=fit_seconds, results=results, model=model, rf_cap=rf_cap,
), INTERIM_DIR / "fdata_power_rf_result.joblib")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] saved RF result")
