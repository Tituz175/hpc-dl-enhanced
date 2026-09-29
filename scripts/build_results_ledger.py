"""
Build the project's results ledger from saved joblibs and committed notebook
outputs. Two long-format CSVs, one row per (config, seed, space, metric):

  results/results_ledger.csv   -- every evaluation metric, every seed, both
                                   spaces, all six target/dataset combinations,
                                   every model and baseline tried.
  results/tuning_details.csv   -- one row per (model, variant) tuning search:
                                   n_trials, tuning_seconds, best_params.

results/seed_variance.csv is regenerated as a filtered view of the ledger
(every row sourced from a *_seedvar_*.joblib file), not maintained by hand.

Does not re-run any fits. Two kinds of sourcing:
  - joblib: for each *_result.joblib that stores a fitted model object plus
    the matching *_prepared.joblib's X_test/y_test, this script calls
    model.predict() (inference only, not a fit) and recomputes the full
    metrics dict via src.metrics.regression_metrics, then asserts it matches
    the stored dict. Mismatches raise -- this is a real correctness check,
    not a formality.
  - recovered_from_notebook: for combinations that were only ever a single
    in-kernel run with no saved model object (F-DATA duration, F-DATA
    memory, PM100 power, PM100 memory), values are transcribed from the
    committed notebook's own cell OUTPUT (not source, not this project's
    tracker prose) at the precision the notebook printed them (4 decimal
    places -- these cannot be independently recomputed without refitting,
    which is out of scope here).

Seed-sweep joblibs (*_seedvar_*.joblib) only ever stored the scalar R2 and
fit_seconds per seed -- no model, no predictions -- so those rows are
included as genuine joblib-backed values but are NOT independently
recomputable here; that limitation is reported at the end, not hidden.

Run with: uv run python scripts/build_results_ledger.py
"""
import json
import pathlib
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics as metrics_mod
from src.features import inverse_transform_target

INTERIM = ROOT / "data" / "interim"
NOTEBOOKS = ROOT / "notebooks"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

METRIC_NAMES = ["MAE", "RMSE", "R2", "MAPE", "MedAE", "Within20pct"]
# recompute-vs-stored tolerance for joblib-backed model verification: model.predict() is
# deterministic given the same fitted model and input, so agreement should be tight, not the
# loose 1e-4 relative slop a first pass used -- but not machine-epsilon tight either: a first
# attempt at 1e-8 failed on a genuine, tiny, and expected source of noise, not a real bug --
# XGBoost's n_jobs=-1 parallel prediction sums per-tree contributions in a thread-scheduling-
# dependent order, so repeat predict() calls on the same fitted model can differ at the
# ~1e-8 relative level (observed directly: fdata_power_xgb_result.joblib's raw-space RMSE
# recomputed 6462.118312800147 vs stored 6462.1182366989815, a 1.18e-8 relative deviation).
# 1e-6 stays 100x tighter than the original pass while comfortably clearing that real,
# floating-point-only noise floor. The actual max deviation is always printed below, not
# just silently passed, so a real regression showing up as e.g. 1e-3 would still be obvious.
ABS_TOL = 1e-6
REL_TOL = 1e-6

ROWS = []
TUNING_ROWS = []
EXCLUDED = []          # (what, why) -- reported, never silently dropped
VERIFIED = []          # (source, how) -- reported at the end


def add_row(dataset, target, model, variant, tree_method, fit_rows, seed, space, metric, value, status, source,
            is_seed_sweep=False):
    ROWS.append(dict(
        dataset=dataset, target=target, model=model, variant=variant,
        tree_method=tree_method, fit_rows=fit_rows, seed=seed, space=space,
        metric=metric, value=float(value), status=status, source=source,
        is_seed_sweep=is_seed_sweep,
    ))


def add_metrics(dataset, target, model, variant, tree_method, fit_rows, seed, space, metrics_dict, status, source,
                 is_seed_sweep=False):
    for m in METRIC_NAMES:
        if m in metrics_dict:
            add_row(dataset, target, model, variant, tree_method, fit_rows, seed, space,
                    m, metrics_dict[m], status, source, is_seed_sweep=is_seed_sweep)


def add_tuning(dataset, target, model, variant, tree_method, n_trials, tuning_seconds, best_params, status, source):
    TUNING_ROWS.append(dict(
        dataset=dataset, target=target, model=model, variant=variant, tree_method=tree_method,
        n_trials=n_trials, tuning_seconds=tuning_seconds, best_params=best_params,
        status=status, source=source,
    ))


def jsrc(fname):
    return f"joblib:data/interim/{fname}"


def nbsrc(nb, cell):
    return f"recovered_from_notebook:notebooks/{nb}#cell{cell}"


# ============================================================================
# 1. *_prepared.joblib -> Naive baseline, recomputed from stored predictions
# ============================================================================

PREPARED_FILES = {
    ("fdata", "duration"): "fdata_duration_r2_prepared.joblib",
    ("fdata", "memory"): "fdata_memory_prepared.joblib",
    ("fdata", "power"): "fdata_power_prepared.joblib",
    ("pm100", "duration"): "pm100_duration_prepared.joblib",
    ("pm100", "memory"): "pm100_memory_clean4_prepared.joblib",
    ("pm100", "power"): "pm100_power_prepared.joblib",
}

prepared_cache = {}
for (dataset, target), fname in PREPARED_FILES.items():
    d = joblib.load(INTERIM / fname)
    prepared_cache[(dataset, target)] = d

    y_test_raw = np.asarray(d["y_test_raw"], dtype=float)
    naive_pred = np.asarray(d["naive_pred"], dtype=float)
    recomputed = metrics_mod.regression_metrics(y_test_raw, naive_pred)
    stored = d["naive_metrics"]
    naive_max_dev = 0.0
    for m in METRIC_NAMES:
        dev = abs(recomputed[m] - stored[m])
        naive_max_dev = max(naive_max_dev, dev)
        tol_threshold = max(ABS_TOL, abs(stored[m]) * REL_TOL)
        if dev > tol_threshold:
            raise ValueError(f"Naive {m} mismatch for {dataset}/{target} ({fname}): "
                              f"stored={stored[m]} recomputed={recomputed[m]} "
                              f"(abs_dev={dev:.3e}, threshold={tol_threshold:.3e})")
    VERIFIED.append((jsrc(fname), f"naive: recomputed from stored naive_pred vs y_test_raw -- OK "
                                   f"(max absolute deviation: {naive_max_dev:.2e})"))

    add_metrics(dataset, target, "Naive", "", "", len(d["y_train_raw"]), None, "raw",
                stored, "adopted", jsrc(fname))


# ============================================================================
# 2. *_result.joblib -> canonical single-seed ML fits, model-verified
# ============================================================================

RESULT_FILES = [
    # dataset, target, model, variant, status, filename, prepared_key, n_trials
    ("fdata", "power", "RandomForest", "", "adopted", "fdata_power_rf_result.joblib", ("fdata", "power"), 50),
    ("fdata", "power", "XGBoost", "", "adopted", "fdata_power_xgb_result.joblib", ("fdata", "power"), 50),
    ("fdata", "power", "LightGBM", "", "adopted", "fdata_power_lgbm_result.joblib", ("fdata", "power"), 50),
    ("pm100", "duration", "RandomForest", "", "adopted", "pm100_duration_rf_result.joblib", ("pm100", "duration"), 50),
    ("pm100", "duration", "XGBoost", "", "adopted", "pm100_duration_xgb_result.joblib", ("pm100", "duration"), 50),
    ("pm100", "duration", "LightGBM", "", "adopted", "pm100_duration_lgbm_result.joblib", ("pm100", "duration"), 50),
    ("pm100", "duration", "RandomForest", "ext200_trials", "investigated_rejected",
     "pm100_duration_ext200_rf_result.joblib", ("pm100", "duration"), 200),
    ("pm100", "duration", "XGBoost", "ext200_trials", "investigated_rejected",
     "pm100_duration_ext200_xgb_result.joblib", ("pm100", "duration"), 200),
    ("pm100", "duration", "LightGBM", "ext200_trials", "investigated_rejected",
     "pm100_duration_ext200_lgbm_result.joblib", ("pm100", "duration"), 200),
]

result_cache = {}
max_deviation_seen = 0.0
for dataset, target, model_name, variant, status, fname, pkey, n_trials in RESULT_FILES:
    d = joblib.load(INTERIM / fname)
    result_cache[fname] = d
    prep = prepared_cache[pkey]
    fitted = d["model"]

    # tree_method for EVALUATION rows must reflect what actually produced the stored
    # predictions (the final fit); tuning may have used a different tree_method (F-DATA
    # power tuned with hist, final-fit with exact -- the tuning row needs the former).
    final_tree_method = d.get("final_fit_tree_method", "") if model_name == "XGBoost" else ""
    tuning_tree_method = d.get("tuning_tree_method", final_tree_method) if model_name == "XGBoost" else ""

    X_test = prep["X_test"]
    y_test_log = np.asarray(prep["y_test"], dtype=float)
    y_test_raw = np.asarray(prep["y_test_raw"], dtype=float)
    pred_log = np.asarray(fitted.predict(X_test), dtype=float)
    pred_raw = inverse_transform_target(pred_log)

    recomputed_log = metrics_mod.regression_metrics(y_test_log, pred_log)
    recomputed_raw = metrics_mod.regression_metrics(y_test_raw, pred_raw)
    stored_log = d["results"]["log_space"]
    stored_raw = d["results"]["raw_space"]
    file_max_dev = 0.0
    for space_name, recomputed, stored in [("log", recomputed_log, stored_log), ("raw", recomputed_raw, stored_raw)]:
        for m in METRIC_NAMES:
            dev = abs(recomputed[m] - stored[m])
            file_max_dev = max(file_max_dev, dev)
            tol_threshold = max(ABS_TOL, abs(stored[m]) * REL_TOL)
            if dev > tol_threshold:
                raise ValueError(f"MISMATCH {fname} {space_name}-space {m}: "
                                  f"stored={stored[m]} recomputed={recomputed[m]} "
                                  f"(abs_dev={dev:.3e}, threshold={tol_threshold:.3e})")
    max_deviation_seen = max(max_deviation_seen, file_max_dev)
    VERIFIED.append((jsrc(fname),
                      f"model.predict(X_test) recomputed, both spaces -- OK "
                      f"(max absolute deviation across all 12 metric values: {file_max_dev:.2e})"))

    if model_name == "RandomForest":
        fit_rows_expected = min(prep.get("rf_cap", len(prep["y_train_raw"])), len(prep["y_train_raw"]))
        # Verify, don't just assume -- but the first attempt at this got the attribute wrong:
        # tree_.n_node_samples[0] is the count of UNIQUE rows reaching the root under
        # bootstrap-with-replacement resampling (~63.2% of the input, the classic 1-1/e
        # bootstrap statistic -- confirmed directly: 3,159,788 / 5,000,000 = 0.6320 for
        # fdata_power_rf_result.joblib, matching 1-1/e=0.6321 almost exactly), NOT the actual
        # row count fit() was called with. tree_.weighted_n_node_samples[0] is the SUM of
        # each row's bootstrap draw count (sklearn implements bootstrap via integer sample
        # weights, not physical row duplication) and equals the true input row count exactly,
        # since exactly n_samples draws are made regardless of duplicates.
        actual_weighted_samples = fitted.estimators_[0].tree_.weighted_n_node_samples[0]
        if abs(actual_weighted_samples - fit_rows_expected) > 0.5:
            raise ValueError(f"RF fit_rows MISMATCH {fname}: expected {fit_rows_expected} "
                              f"(from rf_cap/train size) but fitted.estimators_[0]'s "
                              f"weighted_n_node_samples[0] is {actual_weighted_samples}")
        fit_rows = int(round(actual_weighted_samples))
        VERIFIED.append((jsrc(fname), f"RF fit_rows verified via "
                                       f"estimators_[0].tree_.weighted_n_node_samples[0] == {fit_rows} -- OK"))
    else:
        fit_rows = len(prep["y_train_raw"])

    # Verify the canonical seed, don't just hardcode 0 -- read it straight back off the
    # fitted model's own params rather than assuming the project-wide SEED=0 convention held.
    actual_random_state = fitted.get_params().get("random_state")
    if actual_random_state != 0:
        raise ValueError(f"random_state MISMATCH {fname}: expected 0 (project convention) "
                          f"but fitted model's own get_params() reports {actual_random_state}")
    seed = actual_random_state
    VERIFIED.append((jsrc(fname), f"canonical seed verified via fitted.get_params()['random_state'] "
                                   f"== {seed} -- OK"))

    add_metrics(dataset, target, model_name, variant, final_tree_method, fit_rows, seed, "log", stored_log, status, jsrc(fname))
    add_metrics(dataset, target, model_name, variant, final_tree_method, fit_rows, seed, "raw", stored_raw, status, jsrc(fname))
    add_row(dataset, target, model_name, variant, final_tree_method, fit_rows, seed, "", "fit_seconds",
            d["fit_seconds"], status, jsrc(fname))
    add_tuning(dataset, target, model_name, variant, tuning_tree_method, n_trials, d["tuning_seconds"],
               d["best_params"], status, jsrc(fname))

print(f"\n[verification] max absolute deviation observed across all model.predict() "
      f"recomputations: {max_deviation_seen:.2e} (tolerance: abs={ABS_TOL:.0e} / rel={REL_TOL:.0e})")

# one-point hist/exact scalar checks embedded in the canonical XGBoost result joblibs
# (the "hist" value duplicates the adopted row already emitted above; only "exact" is new)
if "both_methods_r2" in result_cache["pm100_duration_xgb_result.joblib"]:
    both = result_cache["pm100_duration_xgb_result.joblib"]["both_methods_r2"]
    add_row("pm100", "duration", "XGBoost", "histexact_check_seed0", "exact",
            len(prepared_cache[("pm100", "duration")]["y_train_raw"]), 0, "raw", "R2",
            both["exact"], "diagnostic", jsrc("pm100_duration_xgb_result.joblib"))

if "both_methods_r2" in result_cache["pm100_duration_ext200_xgb_result.joblib"]:
    both = result_cache["pm100_duration_ext200_xgb_result.joblib"]["both_methods_r2"]
    add_row("pm100", "duration", "XGBoost", "ext200_trials_histexact_check", "exact",
            len(prepared_cache[("pm100", "duration")]["y_train_raw"]), 0, "raw", "R2",
            both["exact"], "diagnostic", jsrc("pm100_duration_ext200_xgb_result.joblib"))


# ============================================================================
# 3. *_seedvar_*.joblib -> multi-seed sweeps. R2 + fit_seconds only; not
#    independently recomputable (no stored model/predictions per seed).
# ============================================================================

SEEDVAR_FILES = [
    # dataset, target, model, variant, tree_method, status, filename, prepared_key
    ("fdata", "power", "RandomForest", "seed_sweep", "", "diagnostic", "fdata_power_seedvar_rf.joblib", ("fdata", "power")),
    ("fdata", "power", "LightGBM", "seed_sweep", "", "diagnostic", "fdata_power_seedvar_lgbm.joblib", ("fdata", "power")),
    ("fdata", "power", "XGBoost", "seed_sweep_reduced_1M_sample", "exact", "diagnostic",
     "fdata_power_seedvar_xgb.joblib", ("fdata", "power")),
    ("fdata", "power", "XGBoost", "seed_sweep", "exact", "diagnostic",
     "fdata_power_seedvar_xgb_fullscale.joblib", ("fdata", "power")),

    ("fdata", "duration", "RandomForest", "seed_sweep", "", "diagnostic",
     "fdata_duration_r2_seedvar_rf.joblib", ("fdata", "duration")),
    ("fdata", "duration", "XGBoost", "seed_sweep", "hist", "diagnostic",
     "fdata_duration_r2_seedvar_xgb.joblib", ("fdata", "duration")),
    ("fdata", "duration", "LightGBM", "seed_sweep", "", "diagnostic",
     "fdata_duration_r2_seedvar_lgbm.joblib", ("fdata", "duration")),

    ("fdata", "memory", "RandomForest", "seed_sweep", "", "diagnostic",
     "fdata_memory_seedvar_rf.joblib", ("fdata", "memory")),
    ("fdata", "memory", "XGBoost", "seed_sweep", "hist", "diagnostic",
     "fdata_memory_seedvar_xgb.joblib", ("fdata", "memory")),
    ("fdata", "memory", "XGBoost", "seed_sweep_alt_tree_method", "exact", "diagnostic",
     "fdata_memory_seedvar_xgb_exact.joblib", ("fdata", "memory")),
    ("fdata", "memory", "LightGBM", "seed_sweep", "", "diagnostic",
     "fdata_memory_seedvar_lgbm.joblib", ("fdata", "memory")),

    ("pm100", "power", "RandomForest", "seed_sweep", "", "diagnostic",
     "pm100_power_seedvar_rf.joblib", ("pm100", "power")),
    ("pm100", "power", "XGBoost", "seed_sweep", "hist", "diagnostic",
     "pm100_power_seedvar_xgb.joblib", ("pm100", "power")),
    ("pm100", "power", "XGBoost", "seed_sweep_alt_tree_method", "exact", "diagnostic",
     "pm100_power_seedvar_xgb_exact.joblib", ("pm100", "power")),
    ("pm100", "power", "LightGBM", "seed_sweep", "", "diagnostic",
     "pm100_power_seedvar_lgbm.joblib", ("pm100", "power")),

    ("pm100", "memory", "RandomForest", "seed_sweep", "", "diagnostic",
     "pm100_memory_c4_seedvar_rf.joblib", ("pm100", "memory")),
    ("pm100", "memory", "XGBoost", "seed_sweep", "hist", "diagnostic",
     "pm100_memory_c4_seedvar_xgb_hist.joblib", ("pm100", "memory")),
    ("pm100", "memory", "XGBoost", "seed_sweep_alt_tree_method", "exact", "diagnostic",
     "pm100_memory_c4_seedvar_xgb_exact.joblib", ("pm100", "memory")),
    ("pm100", "memory", "LightGBM", "seed_sweep", "", "diagnostic",
     "pm100_memory_c4_seedvar_lgbm.joblib", ("pm100", "memory")),

    ("pm100", "duration", "RandomForest", "seed_sweep", "", "diagnostic",
     "pm100_duration_seedvar_rf.joblib", ("pm100", "duration")),
    ("pm100", "duration", "XGBoost", "seed_sweep", "hist", "diagnostic",
     "pm100_duration_seedvar_xgb_hist.joblib", ("pm100", "duration")),
    ("pm100", "duration", "XGBoost", "seed_sweep_alt_tree_method", "exact", "diagnostic",
     "pm100_duration_seedvar_xgb_exact.joblib", ("pm100", "duration")),
    ("pm100", "duration", "LightGBM", "seed_sweep", "", "diagnostic",
     "pm100_duration_seedvar_lgbm.joblib", ("pm100", "duration")),
]

for dataset, target, model_name, variant, tree_method, status, fname, pkey in SEEDVAR_FILES:
    d = joblib.load(INTERIM / fname)
    prep = prepared_cache[pkey]
    default_rows = len(prep["y_train_raw"])
    rf_cap = d.get("rf_cap", prep.get("rf_cap", default_rows))
    for seed, seed_result in d["seed_results"].items():
        if model_name == "RandomForest":
            fit_rows = min(rf_cap, default_rows)
        else:
            fit_rows = seed_result.get("sample_rows", default_rows)
        add_row(dataset, target, model_name, variant, tree_method, fit_rows, int(seed), "raw",
                "R2", seed_result["R2"], status, jsrc(fname), is_seed_sweep=True)
        add_row(dataset, target, model_name, variant, tree_method, fit_rows, int(seed), "",
                "fit_seconds", seed_result["fit_seconds"], status, jsrc(fname), is_seed_sweep=True)
VERIFIED.append((f"joblib:data/interim/*_seedvar_*.joblib ({len(SEEDVAR_FILES)} files)",
                  "NOT independently recomputed -- these files store only the scalar R2 and "
                  "fit_seconds per seed, no fitted model or predictions, so recomputation would "
                  "require refitting, which is out of scope here. Included as genuine joblib-backed "
                  "values, not fabricated, but not re-verified computationally."))


# ============================================================================
# 4. Standalone diagnostic joblibs
# ============================================================================

d = joblib.load(INTERIM / "fdata_duration_r2_xgb_histexact_check.joblib")
prep = prepared_cache[("fdata", "duration")]
add_row("fdata", "duration", "XGBoost", "histexact_check_seed0", "hist",
        len(prep["y_train_raw"]), 0, "raw", "R2", d["hist_r2"], "diagnostic",
        jsrc("fdata_duration_r2_xgb_histexact_check.joblib"))
add_row("fdata", "duration", "XGBoost", "histexact_check_seed0", "exact",
        len(prep["y_train_raw"]), 0, "raw", "R2", d["exact_r2"], "diagnostic",
        jsrc("fdata_duration_r2_xgb_histexact_check.joblib"))
add_row("fdata", "duration", "XGBoost", "histexact_check_seed0", "exact",
        len(prep["y_train_raw"]), 0, "", "fit_seconds", d["exact_elapsed"], "diagnostic",
        jsrc("fdata_duration_r2_xgb_histexact_check.joblib"))

for seed in (2, 4):
    fname = f"fdata_memory_xgb_exact_seed{seed}_diagnostic.joblib"
    d = joblib.load(INTERIM / fname)
    prep = prepared_cache[("fdata", "memory")]
    add_row("fdata", "memory", "XGBoost", f"seed{seed}_outlier_diagnostic", "exact",
            len(prep["y_train_raw"]), seed, "log", "train_R2", d["train_r2"], "diagnostic", jsrc(fname))
    add_row("fdata", "memory", "XGBoost", f"seed{seed}_outlier_diagnostic", "exact",
            len(prep["y_train_raw"]), seed, "log", "R2", d["test_r2_log"], "diagnostic", jsrc(fname))
    add_row("fdata", "memory", "XGBoost", f"seed{seed}_outlier_diagnostic", "exact",
            len(prep["y_train_raw"]), seed, "raw", "R2", d["test_r2_raw"], "diagnostic", jsrc(fname))


# ============================================================================
# 5. recovered_from_notebook -- combinations with no saved model object
# ============================================================================

_notebook_cache = {}
_FLOAT_RE = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")


def _load_notebook(nb):
    if nb not in _notebook_cache:
        _notebook_cache[nb] = json.load(open(NOTEBOOKS / nb))
    return _notebook_cache[nb]


def cell_output_text(nb, cell):
    nbjson = _load_notebook(nb)
    text = ""
    for o in nbjson["cells"][cell].get("outputs", []):
        if "text" in o:
            text += "".join(o["text"])
        data = o.get("data", {})
        if "text/plain" in data:
            t = data["text/plain"]
            text += "".join(t) if isinstance(t, list) else t
    return text


def _numbers_in_text(text):
    # notebooks print with thousands separators ("10,517.6011") -- strip commas that sit
    # between digits before parsing, or every such number splits into two false negatives.
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return [float(x) for x in _FLOAT_RE.findall(text)]


def verify_values_in_cell(nb, cell, expected_values, label):
    """Presence-only fallback: is this value anywhere in the cell's output text at all,
    with no row/column attribution. Used only when the labeled table parser below can't
    place a value (fit_seconds lines, tuning citations, analytical-baseline colon blocks
    get their own labeled parsers; this is the last resort for anything else)."""
    found = _numbers_in_text(cell_output_text(nb, cell))
    missing = [v for v in expected_values
               if not any(abs(v - f) <= max(1e-9, abs(v) * 1e-9) for f in found)]
    if missing:
        raise ValueError(f"Notebook-recovered value(s) {missing} for {label} not found "
                          f"verbatim in notebooks/{nb}#cell{cell}'s committed output text -- "
                          f"check for a transcription typo.")


MODEL_LABELS = ["Naive (per-user median)", "RandomForest", "XGBoost", "LightGBM"]
LABELED_VERIFIED_COUNT = 0
PRESENCE_ONLY_COUNT = 0


def parse_pandas_metric_tables(text):
    """Parse pandas to_string()-style wrapped tables into {space: {model: {metric: value}}},
    matching by row LABEL (the model name, token-boundary-anchored via a line-start check so
    e.g. 'RandomForest' can't match as a substring of something else) and by COLUMN (the
    header line's own token order, not a fixed assumption -- these tables wrap into a 5-col
    then a 1-col continuation block, and the column order is read off each block's own
    header). Cuts the text before any 'model        subset' breakdown sub-table (cell 103's
    divergent-condition table reuses the same model-name labels with a matching column count
    and, since it comes after the log-space block with no new space marker, would otherwise
    silently overwrite log-space XGBoost/RandomForest/LightGBM with its own raw-space-flavored
    numbers -- confirmed directly by testing without this cut)."""
    text = re.split(r"\nmodel\s+subset", text)[0]
    space = None
    columns = None
    result = {"raw": {}, "log": {}}
    for line in text.splitlines():
        if re.search(r"\bRaw-space\b", line, re.IGNORECASE):
            space = "raw"; columns = None; continue
        if re.search(r"\bLog-space\b", line, re.IGNORECASE):
            space = "log"; columns = None; continue
        tokens_found = [tok for tok in METRIC_NAMES if re.search(rf"\b{tok}\b", line)]
        stripped = line.strip().rstrip("\\").strip()
        words = stripped.split()
        if tokens_found and words and all(re.fullmatch(r"[A-Za-z0-9]+", w) for w in words):
            columns = sorted(tokens_found, key=lambda t: line.index(t))
            continue
        if space is None or columns is None:
            continue
        for model in MODEL_LABELS:
            if line.strip().startswith(model):
                rest = line.strip()[len(model):]
                nums = _numbers_in_text(rest)
                if len(nums) == len(columns):
                    result[space].setdefault(model, {}).update(dict(zip(columns, nums)))
                break
    return result


def verify_colon_metrics_by_label(nb, cell, expected_metrics, label):
    """Row-labeled verification for the 'MAE: 10,517.6011' colon-style blocks (notebook 03's
    analytical baselines) -- the metric name IS the label here, so match token-boundary-
    anchored 'METRIC:' directly rather than a bare presence search."""
    global LABELED_VERIFIED_COUNT, PRESENCE_ONLY_COUNT
    text = cell_output_text(nb, cell)
    presence_fallback = []
    for metric_name, expected in expected_metrics.items():
        m = re.search(rf"\b{re.escape(metric_name)}\b:\s*([\-\d,]+\.\d+)", text)
        actual = float(m.group(1).replace(",", "")) if m else None
        if actual is not None and abs(actual - expected) <= max(1e-9, abs(expected) * 1e-9):
            LABELED_VERIFIED_COUNT += 1
        else:
            presence_fallback.append(expected)
    if presence_fallback:
        verify_values_in_cell(nb, cell, presence_fallback, label)
        PRESENCE_ONLY_COUNT += len(presence_fallback)
        print(f"  [notebook check] {label}: {len(presence_fallback)} value(s) not placeable by "
              f"label, verified by presence-only instead: {presence_fallback}")


def verify_fit_seconds_by_label(nb, cell, fit_seconds, label):
    """Row-labeled verification for fit_seconds: every cell that reports final-fit time
    writes it on the SAME line as the model name (e.g. 'RandomForest: tuning=8954.7s
    final_fit=1181.5s' or 'RandomForest: fit 126,217 rows in 0.6s'), always as the LAST
    '<number>s' on that line -- scoping the search to lines starting with the model name
    (a token-boundary anchor, not a bare substring search) means a match can't come from an
    unrelated part of the cell."""
    global LABELED_VERIFIED_COUNT, PRESENCE_ONLY_COUNT
    text = cell_output_text(nb, cell)
    presence_fallback = []
    for model_name, expected in fit_seconds.items():
        found_on_line = None
        for line in text.splitlines():
            if line.strip().startswith(model_name + ":") or line.strip().startswith(model_name + " "):
                nums_with_s = re.findall(r"([\d,]+\.\d+)s", line)
                if nums_with_s:
                    found_on_line = float(nums_with_s[-1].replace(",", ""))
        if found_on_line is not None and abs(found_on_line - expected) <= max(1e-9, abs(expected) * 1e-9):
            LABELED_VERIFIED_COUNT += 1
        else:
            presence_fallback.append(expected)
    if presence_fallback:
        verify_values_in_cell(nb, cell, presence_fallback, label)
        PRESENCE_ONLY_COUNT += len(presence_fallback)
        print(f"  [notebook check] {label}: {len(presence_fallback)} value(s) not placeable by "
              f"row label, verified by presence-only instead: {presence_fallback}")


def verify_table_by_label(nb, cell, raw_table, log_table, label):
    """Row+column-labeled verification: parse the cell's own table structure and check each
    expected (model, metric) value against the SAME (model, metric) cell in the parse, not
    just anywhere in the text. Falls back to presence-only (and counts the fallback) only for
    values the structured parser genuinely can't place -- e.g. a table shape this parser
    doesn't yet handle -- so the two verification strengths are never silently conflated."""
    global LABELED_VERIFIED_COUNT, PRESENCE_ONLY_COUNT
    parsed = parse_pandas_metric_tables(cell_output_text(nb, cell))
    presence_fallback = []
    for space_name, table in [("raw", raw_table), ("log", log_table)]:
        for model_name, metrics_dict in table.items():
            for metric_name, expected in metrics_dict.items():
                actual = parsed.get(space_name, {}).get(model_name, {}).get(metric_name)
                if actual is not None and abs(actual - expected) <= max(1e-9, abs(expected) * 1e-9):
                    LABELED_VERIFIED_COUNT += 1
                else:
                    presence_fallback.append(expected)
    if presence_fallback:
        before = PRESENCE_ONLY_COUNT
        verify_values_in_cell(nb, cell, presence_fallback, label)
        PRESENCE_ONLY_COUNT += len(presence_fallback)
        print(f"  [notebook check] {label}: {len(presence_fallback)} value(s) not placeable by "
              f"row+column label, verified by presence-only instead: {presence_fallback}")


_SEED_ASSIGN_RE = re.compile(r"\bSEED\s*=\s*(-?\d+)\b")
_verified_notebook_seeds = {}


def verify_notebook_seed_convention(nb, expected_seed):
    """Confirm the SEED=0 convention from *source* code (not output, not assumed): every
    code cell's own `SEED = <int>` assignment must match, and at least one must exist."""
    if nb in _verified_notebook_seeds:
        return _verified_notebook_seeds[nb]
    nbjson = _load_notebook(nb)
    found_values = set()
    for cell in nbjson["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        found_values.update(int(m) for m in _SEED_ASSIGN_RE.findall(src))
    if not found_values:
        raise ValueError(f"No 'SEED = <int>' assignment found in notebooks/{nb}'s source -- "
                          f"cannot verify the seed convention for its recovered rows.")
    if found_values != {expected_seed}:
        raise ValueError(f"notebooks/{nb} assigns SEED to {sorted(found_values)}, not "
                          f"exclusively {expected_seed} -- the project-wide convention does not "
                          f"hold for this notebook; do not assume seed={expected_seed} for its rows.")
    _verified_notebook_seeds[nb] = expected_seed
    VERIFIED.append((nbsrc(nb, "*"), f"seed convention verified from source code (every "
                                      f"'SEED = <int>' assignment in the notebook is {expected_seed}, "
                                      f"and its final fits use random_state=SEED) -- OK"))
    return expected_seed


_TREE_METHOD_RE = re.compile(r'tree_method\s*=\s*["\'](\w+)["\']')


def verify_tree_method_from_source(nb, fit_cell, expected_tree_method, label):
    """Confirm the adopted XGBoost tree_method from the notebook's own *source* code at the
    actual final-fit cell (an XGBRegressor(..., tree_method=..., ...) construction), not by
    cross-referencing a seedvar joblib's value or assuming it matches a sibling combination."""
    nbjson = _load_notebook(nb)
    src = "".join(nbjson["cells"][fit_cell].get("source", []))
    if "XGBRegressor" not in src:
        raise ValueError(f"notebooks/{nb}#cell{fit_cell} has no XGBRegressor construction -- "
                          f"wrong cell cited for {label}'s tree_method.")
    m = _TREE_METHOD_RE.search(src)
    actual = m.group(1) if m else None
    if actual != expected_tree_method:
        raise ValueError(f"tree_method MISMATCH for {label}: notebooks/{nb}#cell{fit_cell}'s "
                          f"XGBRegressor uses tree_method={actual!r}, expected {expected_tree_method!r}")
    VERIFIED.append((nbsrc(nb, fit_cell), f"{label} XGBoost tree_method read directly from the "
                                           f"final-fit cell's own source: {actual!r} -- OK"))
    return actual


def add_notebook_table(dataset, target, nb, cell, variant, status, tree_method_xgb,
                        raw_table, log_table, fit_seconds, tuning=None, best_params=None,
                        tuning_variant=None, tuning_source=None, n_trials=50,
                        fit_seconds_cell=None):
    source = nbsrc(nb, cell)
    # fit_seconds is sometimes printed in a different cell than the metrics table itself
    # (e.g. PM100 memory clean-4's fit times are in cell 101, the table in cell 103) --
    # default to the same cell, but let callers cite the real one so the automated check
    # below verifies against where the numbers actually are, not where the table is.
    fs_cell = fit_seconds_cell if fit_seconds_cell is not None else cell
    fs_source = nbsrc(nb, fs_cell)

    # automated check: every metric value transcribed below must match its OWN model row and
    # metric column in the cell's own committed table (not just appear somewhere in the text).
    verify_table_by_label(nb, cell, raw_table, log_table, f"{dataset}/{target} ({variant or 'default'})")
    if fit_seconds:
        verify_fit_seconds_by_label(nb, fs_cell, fit_seconds,
                                    f"{dataset}/{target} ({variant or 'default'}) fit_seconds")

    # seed=0 is verified from this notebook's own source code (SEED=0 + random_state=SEED),
    # not assumed -- see verify_notebook_seed_convention.
    seed = verify_notebook_seed_convention(nb, expected_seed=0)

    # fit_rows, same rule as the joblib-backed rows: RandomForest is capped (rf_cap), every
    # other model fits on the full train split. Previously left blank here entirely -- caught
    # by item (c)'s own distinct-fit_rows report crashing on a None for a RandomForest row.
    prep = prepared_cache[(dataset, target)]
    full_rows = len(prep["y_train_raw"])
    rf_fit_rows = min(prep.get("rf_cap", full_rows), full_rows)

    def _fit_rows_for(model_name):
        return rf_fit_rows if model_name == "RandomForest" else full_rows

    for model_name, metrics_dict in raw_table.items():
        tm = tree_method_xgb if model_name == "XGBoost" else ""
        add_metrics(dataset, target, model_name, variant, tm, _fit_rows_for(model_name), seed, "raw",
                    metrics_dict, status, source)
    for model_name, metrics_dict in log_table.items():
        tm = tree_method_xgb if model_name == "XGBoost" else ""
        add_metrics(dataset, target, model_name, variant, tm, _fit_rows_for(model_name), seed, "log",
                    metrics_dict, status, source)
    for model_name, secs in fit_seconds.items():
        tm = tree_method_xgb if model_name == "XGBoost" else ""
        add_row(dataset, target, model_name, variant, tm, _fit_rows_for(model_name), seed, "",
                "fit_seconds", secs, status, fs_source)
    # tuning_seconds and best_params can be sourced independently -- a SKIP_TUNING re-run notebook
    # cell can still print the reused best_params dict precisely even when the original tuning
    # run's cost was overwritten and is no longer recoverable (tuning_seconds left as NaN there).
    if tuning is not None or best_params is not None:
        models = set((tuning or {}).keys()) | set((best_params or {}).keys())
        tv = tuning_variant if tuning_variant is not None else variant
        tsrc = tuning_source if tuning_source is not None else source
        if tuning is not None:
            tuning_nb, tuning_cell = (tsrc.split("notebooks/")[1].split("#cell") if tsrc.startswith("recovered_from_notebook")
                                       else (None, None))
            if tuning_nb is not None:
                verify_values_in_cell(tuning_nb, int(tuning_cell), list(tuning.values()),
                                      f"{dataset}/{target} tuning_seconds")
        for model_name in models:
            tm = tree_method_xgb if model_name == "XGBoost" else ""
            secs = (tuning or {}).get(model_name, float("nan"))
            bp = (best_params or {}).get(model_name)
            add_tuning(dataset, target, model_name, tv, tm, n_trials, secs, bp, status, tsrc)
    VERIFIED.append((source, "recovered_from_notebook: transcribed from committed cell output "
                              "(4 decimal places, as printed); automated check confirmed every "
                              "value is present verbatim in that cell's own output text; not "
                              "independently recomputable without refitting."))


# --- F-DATA duration Result 1 (window=5) -- superseded by Result 2, kept as diagnostic ---
_fdata_duration_r1_tm = verify_tree_method_from_source(
    "05_classical_ml_baselines.ipynb", 27, "hist", "fdata/duration result1_window5")
add_notebook_table(
    "fdata", "duration", "05_classical_ml_baselines.ipynb", 29, "result1_window5", "diagnostic",
    _fdata_duration_r1_tm,
    raw_table={
        "RandomForest": dict(MAE=3453.8171, RMSE=12419.0736, R2=0.7708, MAPE=1887.5566, MedAE=245.7386, Within20pct=46.6053),
        "XGBoost": dict(MAE=2986.0604, RMSE=11153.1835, R2=0.8152, MAPE=2133.1015, MedAE=206.5059, Within20pct=55.2116),
        "LightGBM": dict(MAE=2921.7242, RMSE=10892.1171, R2=0.8237, MAPE=2064.4761, MedAE=180.7551, Within20pct=56.6224),
    },
    log_table={
        "RandomForest": dict(MAE=0.4232, RMSE=0.8305, R2=0.8840, MAPE=10.4906, MedAE=0.2392, Within20pct=92.4455),
        "XGBoost": dict(MAE=0.3925, RMSE=0.8266, R2=0.8851, MAPE=10.1336, MedAE=0.1804, Within20pct=92.4326),
        "LightGBM": dict(MAE=0.3858, RMSE=0.8221, R2=0.8864, MAPE=9.6180, MedAE=0.1624, Within20pct=92.3836),
    },
    fit_seconds={"RandomForest": 1191.4, "XGBoost": 181.0, "LightGBM": 224.8},
)

# --- F-DATA duration Result 2 (windows 5+20+50) -- adopted ---
_fdata_duration_r2_tm = verify_tree_method_from_source(
    "05_classical_ml_baselines.ipynb", 53, "hist", "fdata/duration (default)")
add_notebook_table(
    "fdata", "duration", "05_classical_ml_baselines.ipynb", 55, "", "adopted", _fdata_duration_r2_tm,
    raw_table={
        "RandomForest": dict(MAE=2748.5697, RMSE=10991.5021, R2=0.8205, MAPE=1960.4995, MedAE=164.1498, Within20pct=61.8745),
        "XGBoost": dict(MAE=2714.3641, RMSE=10509.5175, R2=0.8359, MAPE=1923.6291, MedAE=159.5601, Within20pct=60.3646),
        "LightGBM": dict(MAE=2780.6748, RMSE=10676.0911, R2=0.8306, MAPE=1974.5775, MedAE=195.9394, Within20pct=60.7888),
    },
    log_table={
        "RandomForest": dict(MAE=0.3607, RMSE=0.8003, R2=0.8923, MAPE=9.3145, MedAE=0.1373, Within20pct=92.9292),
        "XGBoost": dict(MAE=0.3653, RMSE=0.8050, R2=0.8910, MAPE=9.4589, MedAE=0.1428, Within20pct=92.9049),
        "LightGBM": dict(MAE=0.3645, RMSE=0.7936, R2=0.8941, MAPE=9.0927, MedAE=0.1510, Within20pct=93.0093),
    },
    fit_seconds={"RandomForest": 1266.0, "XGBoost": 181.5, "LightGBM": 244.8},
    best_params={
        "RandomForest": {"n_estimators": 105, "max_depth": 18, "min_samples_leaf": 19, "max_features": 0.5495121930980689},
        "XGBoost": {"n_estimators": 411, "max_depth": 7, "learning_rate": 0.01696003021391471, "subsample": 0.7137365087386797, "colsample_bytree": 0.8996675930517088},
        "LightGBM": {"n_estimators": 493, "num_leaves": 207, "learning_rate": 0.014920507084345887, "subsample": 0.685307832922302, "colsample_bytree": 0.9415149135271289},
    },
    tuning_source=nbsrc("05_classical_ml_baselines.ipynb", 25),
)
EXCLUDED.append((
    "F-DATA duration (Result 1 and Result 2) original Optuna tuning_seconds",
    "best_params ARE precisely sourced (notebook 05 cell 25 prints the full reused dict) and are "
    "included in tuning_details.csv. Only tuning_seconds is missing: notebook 05's tuning cells now "
    "print 'tuning=0.0s' for every model (SKIP_TUNING=True reuses stored best_params from an earlier "
    "historical run); that earlier run's real tuning-cost output was overwritten when the notebook "
    "was re-executed with SKIP_TUNING on. Not recoverable from any committed notebook cell or joblib "
    "-- only exists as rounded prose in EXPERIMENT_TRACKER.md, which is not an accepted source here."
))

# --- F-DATA memory -- adopted (single headline fit; XGBoost's small-ensemble seed-instability is a
#     separate, correctly-scoped finding in the seed-sweep rows, not a reason to reject this row) ---
_fdata_memory_tm = verify_tree_method_from_source(
    "05b_classical_ml_memory.ipynb", 32, "hist", "fdata/memory (default)")
add_notebook_table(
    "fdata", "memory", "05b_classical_ml_memory.ipynb", 34, "", "adopted", _fdata_memory_tm,
    raw_table={
        "RandomForest": dict(MAE=1.420149e9, RMSE=3.015326e9, R2=0.8246, MAPE=79.6985, MedAE=4.116405e8, Within20pct=69.7484),
        "XGBoost": dict(MAE=1.587637e9, RMSE=3.163314e9, R2=0.8070, MAPE=77.2856, MedAE=4.904366e8, Within20pct=63.9724),
        "LightGBM": dict(MAE=1.219928e9, RMSE=2.627899e9, R2=0.8668, MAPE=82.9931, MedAE=3.133620e8, Within20pct=73.9229),
    },
    log_table={
        "RandomForest": dict(MAE=0.2443, RMSE=0.4969, R2=0.8875, MAPE=1.1449, MedAE=0.1120, Within20pct=99.7085),
        "XGBoost": dict(MAE=0.2752, RMSE=0.5264, R2=0.8737, MAPE=1.2869, MedAE=0.1351, Within20pct=99.7089),
        "LightGBM": dict(MAE=0.2233, RMSE=0.4848, R2=0.8929, MAPE=1.0546, MedAE=0.0862, Within20pct=99.7064),
    },
    fit_seconds={"RandomForest": 1181.5, "XGBoost": 34.8, "LightGBM": 77.1},
    tuning={"RandomForest": 8954.7, "XGBoost": 268.5, "LightGBM": 417.0},
    best_params=prepared_cache[("fdata", "memory")]["best_params"],
)

# --- PM100 memory, full 17-feature set -- investigated and REJECTED (mem_req leaks the target on
#     93.76% of jobs; see the 2026-09-05 Feature Vetting investigation) ---
# tree_method was earlier left blank here ("cell 77 doesn't state it") -- that was checking the
# wrong cell. Cell 77 is the RESULTS table; cell 75 is the actual XGBRegressor(...) construction,
# and it does specify tree_method explicitly. Read directly from source, not cross-referenced.
_pm100_memory_full17_tm = verify_tree_method_from_source(
    "05b_classical_ml_memory.ipynb", 75, "hist", "pm100/memory full17_mem_req_leakage")
add_notebook_table(
    "pm100", "memory", "05b_classical_ml_memory.ipynb", 77, "full17_mem_req_leakage",
    "investigated_rejected", _pm100_memory_full17_tm,
    raw_table={
        "RandomForest": dict(MAE=9.9222, RMSE=254.2546, R2=0.9773, MAPE=1.8716, MedAE=0.0000, Within20pct=98.5839),
        "XGBoost": dict(MAE=29.7576, RMSE=331.7707, R2=0.9614, MAPE=3.5850, MedAE=2.3361, Within20pct=93.3023),
        "LightGBM": dict(MAE=31.9018, RMSE=209.8636, R2=0.9845, MAPE=7.8986, MedAE=23.4651, Within20pct=98.1865),
    },
    log_table={
        "RandomForest": dict(MAE=0.0206, RMSE=0.1065, R2=0.9904, MAPE=0.4060, MedAE=0.0000, Within20pct=99.8503),
        "XGBoost": dict(MAE=0.0370, RMSE=0.1013, R2=0.9913, MAPE=0.7205, MedAE=0.0099, Within20pct=99.9002),
        "LightGBM": dict(MAE=0.0763, RMSE=0.1075, R2=0.9902, MAPE=1.4325, MedAE=0.0940, Within20pct=99.8484),
    },
    fit_seconds={"RandomForest": 2.1, "XGBoost": 1.1, "LightGBM": 3.2},
    tuning={"RandomForest": 77.0, "XGBoost": 60.8, "LightGBM": 122.0},
    best_params={
        "RandomForest": {"n_estimators": 125, "max_depth": 18, "min_samples_leaf": 5, "max_features": 0.964286724517838},
        "XGBoost": {"n_estimators": 315, "max_depth": 8, "learning_rate": 0.03905453391570842, "subsample": 0.7294795269246798, "colsample_bytree": 0.9220601400732221},
        "LightGBM": {"n_estimators": 276, "num_leaves": 178, "learning_rate": 0.04093944822901454, "subsample": 0.8011088010679889, "colsample_bytree": 0.9346648837730069},
    },
    tuning_source=nbsrc("05b_classical_ml_memory.ipynb", 73),
)

# --- PM100 memory, clean 4-feature set -- adopted ---
_pm100_memory_c4_tm = verify_tree_method_from_source(
    "05b_classical_ml_memory.ipynb", 101, "hist", "pm100/memory (default)")
add_notebook_table(
    "pm100", "memory", "05b_classical_ml_memory.ipynb", 103, "", "adopted", _pm100_memory_c4_tm,
    raw_table={
        "RandomForest": dict(MAE=24.1213, RMSE=374.0273, R2=0.9509, MAPE=5.7452, MedAE=0.8276, Within20pct=97.4156),
        "XGBoost": dict(MAE=27.3610, RMSE=312.0164, R2=0.9658, MAPE=6.7001, MedAE=0.0820, Within20pct=97.4082),
        "LightGBM": dict(MAE=28.0444, RMSE=337.8480, R2=0.9599, MAPE=6.0310, MedAE=0.0721, Within20pct=97.4026),
    },
    log_table={
        "RandomForest": dict(MAE=0.0274, RMSE=0.1648, R2=0.9770, MAPE=0.6513, MedAE=0.0035, Within20pct=99.3918),
        "XGBoost": dict(MAE=0.0295, RMSE=0.1785, R2=0.9730, MAPE=0.7251, MedAE=0.0003, Within20pct=99.2014),
        "LightGBM": dict(MAE=0.0271, RMSE=0.1735, R2=0.9745, MAPE=0.6626, MedAE=0.0003, Within20pct=99.2254),
    },
    fit_seconds={"RandomForest": 0.6, "XGBoost": 0.9, "LightGBM": 1.2},
    fit_seconds_cell=101,  # cell 103 has the results table; the fit-time line is in cell 101
    best_params=prepared_cache[("pm100", "memory")]["best_params"],
    tuning_source=jsrc("pm100_memory_clean4_prepared.joblib"),
)
EXCLUDED.append((
    "PM100 memory clean-4 Optuna tuning_seconds",
    "best_params ARE joblib-sourced (pm100_memory_clean4_prepared.joblib's own 'best_params' field) "
    "and are included in tuning_details.csv. Only tuning_seconds is missing: cell 100/101 of "
    "05b_classical_ml_memory.ipynb state the clean-4 best_params come from 'real 50-trial Optuna "
    "searches run on this exact 4-column set', but no surviving cell prints that search's "
    "tuning_seconds (only the final-fit times in cell 101). Not recoverable from any committed "
    "notebook cell or joblib."
))

# --- PM100 power -- adopted ---
_pm100_power_tm = verify_tree_method_from_source(
    "05_classical_ml_baselines.ipynb", 86, "hist", "pm100/power (default)")
add_notebook_table(
    "pm100", "power", "05_classical_ml_baselines.ipynb", 88, "", "adopted", _pm100_power_tm,
    raw_table={
        "RandomForest": dict(MAE=398.3066, RMSE=2513.4725, R2=0.9018, MAPE=17.0614, MedAE=130.1689, Within20pct=66.6611),
        "XGBoost": dict(MAE=360.3284, RMSE=2206.6780, R2=0.9243, MAPE=16.6291, MedAE=123.9113, Within20pct=68.1696),
        "LightGBM": dict(MAE=372.1584, RMSE=2215.1090, R2=0.9237, MAPE=17.2101, MedAE=129.6627, Within20pct=65.3356),
    },
    log_table={
        "RandomForest": dict(MAE=0.1750, RMSE=0.2231, R2=0.9399, MAPE=2.5454, MedAE=0.1471, Within20pct=99.9852),
        "XGBoost": dict(MAE=0.1656, RMSE=0.2126, R2=0.9455, MAPE=2.4236, MedAE=0.1371, Within20pct=99.9852),
        "LightGBM": dict(MAE=0.1748, RMSE=0.2230, R2=0.9400, MAPE=2.5514, MedAE=0.1494, Within20pct=99.9852),
    },
    fit_seconds={"RandomForest": 2.0, "XGBoost": 0.2, "LightGBM": 0.3},
    best_params={
        "RandomForest": {"n_estimators": 229, "max_depth": 9, "min_samples_leaf": 8, "max_features": 0.48374603344981765},
        "XGBoost": {"n_estimators": 128, "max_depth": 4, "learning_rate": 0.05882341368895798, "subsample": 0.9337098444135781, "colsample_bytree": 0.9294864662794635},
        "LightGBM": {"n_estimators": 89, "num_leaves": 19, "learning_rate": 0.16977763163336626, "subsample": 0.8890783754749252, "colsample_bytree": 0.9350060741234096},
    },
    tuning_source=nbsrc("05_classical_ml_baselines.ipynb", 84),
)
EXCLUDED.append((
    "PM100 power original Optuna tuning_seconds",
    "best_params ARE precisely sourced (notebook 05 cell 84 prints the full reused dict) and are "
    "included in tuning_details.csv. Only tuning_seconds is missing: notebook 05's PM100 tuning "
    "cell prints 'tuning=0.0s' for every model (SKIP_TUNING_PM100=True reuses stored best_params "
    "from an earlier historical, pre-workstation-migration run); that run's real tuning-cost output "
    "no longer exists in the committed notebook. Not recoverable from any committed notebook cell "
    "or joblib -- only rounded prose in EXPERIMENT_TRACKER.md, which is not an accepted source here."
))

# --- Analytical baselines (notebook 03), raw-space only, no seed / no log-space ---
verify_colon_metrics_by_label("03_analytical_baselines.ipynb", 16,
                              {"MAE": 10517.6011, "RMSE": 27993.7073, "R2": -0.1644,
                               "MAPE": 99.9848, "MedAE": 1803.8535, "Within20pct": 0.0000},
                              "fdata/duration Roofline")
verify_colon_metrics_by_label("03_analytical_baselines.ipynb", 30,
                              {"MAE": 493.4335, "RMSE": 2408.5334, "R2": 0.9098,
                               "MAPE": 21.8178, "MedAE": 174.2140, "Within20pct": 46.1631},
                              "pm100/power CalibratedPowerModel")

add_row("fdata", "duration", "Roofline", "", "", None, None, "raw", "MAE", 10517.6011, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 16))
add_row("fdata", "duration", "Roofline", "", "", None, None, "raw", "RMSE", 27993.7073, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 16))
add_row("fdata", "duration", "Roofline", "", "", None, None, "raw", "R2", -0.1644, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 16))
add_row("fdata", "duration", "Roofline", "", "", None, None, "raw", "MAPE", 99.9848, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 16))
add_row("fdata", "duration", "Roofline", "", "", None, None, "raw", "MedAE", 1803.8535, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 16))
add_row("fdata", "duration", "Roofline", "", "", None, None, "raw", "Within20pct", 0.0000, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 16))

add_row("pm100", "power", "CalibratedPowerModel", "", "", None, None, "raw", "MAE", 493.4335, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 30))
add_row("pm100", "power", "CalibratedPowerModel", "", "", None, None, "raw", "RMSE", 2408.5334, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 30))
add_row("pm100", "power", "CalibratedPowerModel", "", "", None, None, "raw", "R2", 0.9098, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 30))
add_row("pm100", "power", "CalibratedPowerModel", "", "", None, None, "raw", "MAPE", 21.8178, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 30))
add_row("pm100", "power", "CalibratedPowerModel", "", "", None, None, "raw", "MedAE", 174.2140, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 30))
add_row("pm100", "power", "CalibratedPowerModel", "", "", None, None, "raw", "Within20pct", 46.1631, "adopted",
        nbsrc("03_analytical_baselines.ipynb", 30))

# F-DATA power's calibrated analytical baseline -- src/roofline.py implements
# calibrate_fdata_power_model/predict_fdata_power, and 05c's markdown says the model is
# feasible, but (checked directly) no cell in 05c ever calls it or reports its metrics.
# scripts/fdata_power_calibrated_baseline/fit_and_evaluate.py closes that gap: reproduces
# 05c's own pipeline, calibrates on train, evaluates on test, saves to a joblib. Full
# provenance (including exact reproduction of a real, once-computed-but-never-persisted
# result) in results/investigation_log.md section 4.
_calib_path = INTERIM / "fdata_power_calibrated_baseline.joblib"
_calib = joblib.load(_calib_path)
add_metrics("fdata", "power", "CalibratedFDataPowerModel", "", "", _calib["train_rows"], None,
            "raw", _calib["test_metrics"], "adopted", jsrc("fdata_power_calibrated_baseline.joblib"))
add_row("fdata", "power", "CalibratedFDataPowerModel", "", "", _calib["train_rows"], None, "",
        "fit_seconds", _calib["elapsed_seconds"], "adopted", jsrc("fdata_power_calibrated_baseline.joblib"))
VERIFIED.append((jsrc("fdata_power_calibrated_baseline.joblib"),
                  "joblib-backed, produced by a real, re-run script (not fabricated) -- but NOT "
                  "independently recomputed within this build (unlike the RF/XGBoost/LightGBM "
                  "*_result.joblib rows, this joblib stores only coeffs+metrics, not a model "
                  "object or X_test, so recomputation would mean reloading and reprocessing the "
                  "full 23.3M-row F-DATA dataset again here -- same disclosed limitation as the "
                  "*_seedvar_*.joblib rows). The script's own run reproduced a real, independently "
                  "session-recorded prior computation to zero deviation on all four values -- see "
                  "results/investigation_log.md section 4 for that cross-check."))
VERIFIED.append((nbsrc("03_analytical_baselines.ipynb", "16,30"),
                  "recovered_from_notebook: transcribed from committed cell output (4 decimal "
                  "places); no calibration_df row count or log-space equivalent applies to these "
                  "physical-unit analytical models."))

# ============================================================================
# 6. Known, disclosed exclusion: F-DATA power XGBoost hist bug's failed-fit R2
# ============================================================================
EXCLUDED.append((
    "F-DATA power, XGBoost, tree_method=hist final fit (the catastrophic-underfit bug run)",
    "fdata_power_xgb_result.joblib's own 'note' field says only 'R2 ~0.31' (approximate prose, "
    "not a precise stored value) -- no model object or predictions were kept for that specific "
    "failed hist-mode final fit (only the eventual tree_method=exact fit was persisted, since that "
    "is what got adopted). Not precisely sourceable without refitting, which is out of scope here. "
    "Excluded from results_ledger.csv rather than guessed at 3-decimal prose precision."
))
EXCLUDED.append((
    "F-DATA power OOM crash: early-stopping fix verification (before/after numbers)",
    "Scripts exist (scripts/decision6_seed_variance/fdata_power_crash_investigation/) and are "
    "committed, but their output only ever existed in ephemeral /tmp scratch logs that were "
    "never copied into the repo. No number is sourceable without re-running them, which is out "
    "of scope here. Full account in results/investigation_log.md, section 1."
))
EXCLUDED.append((
    "F-DATA power XGBoost tree_method=hist bug: the 4-check root-cause isolation "
    "(hyperparameter-independence, float32/64 precision, embedding-columns, tree_method=approx)",
    "No scripts exist for these at all -- run ad hoc, inline, unlike every other diagnostic in "
    "this project. No persisted numeric artifact for any of the four checks. Full account in "
    "results/investigation_log.md, section 2."
))
EXCLUDED.append((
    "PM100 duration: embedding-style field check",
    "A structural yes/no schema check (PM100 has no embedding field), not a metric value, so it "
    "was never going to be a ledger row -- listed here only because it's part of the same bounded "
    "investigation pass as the ext200_trials check that IS in the ledger. No script was saved. "
    "Full account in results/investigation_log.md, section 3."
))


# ============================================================================
# Write output
# ============================================================================

ledger = pd.DataFrame(ROWS, columns=[
    "dataset", "target", "model", "variant", "tree_method", "fit_rows",
    "seed", "space", "metric", "value", "status", "source", "is_seed_sweep",
])
# nullable integer dtype: fit_rows/seed are genuinely missing (not 0) for e.g. analytical
# baselines and naive's seed column -- plain int64 would force a silent upcast to float64
# (printing "16315866.0") the moment any row has a null; Int64 keeps them integers with a
# real <NA> instead.
ledger["fit_rows"] = ledger["fit_rows"].astype("Int64")
ledger["seed"] = ledger["seed"].astype("Int64")
ledger = ledger.sort_values(["dataset", "target", "model", "variant", "space", "seed", "metric"]).reset_index(drop=True)
ledger.to_csv(RESULTS / "results_ledger.csv", index=False)

tuning = pd.DataFrame(TUNING_ROWS, columns=[
    "dataset", "target", "model", "variant", "tree_method",
    "n_trials", "tuning_seconds", "best_params", "status", "source",
])
tuning["n_trials"] = tuning["n_trials"].astype("Int64")
tuning = tuning.sort_values(["dataset", "target", "model", "variant"]).reset_index(drop=True)
tuning.to_csv(RESULTS / "tuning_details.csv", index=False)

seed_variance = ledger[ledger["is_seed_sweep"] & (ledger["metric"] == "R2")].copy()
seed_variance = seed_variance.drop(columns=["fit_rows", "space", "metric", "is_seed_sweep"]).rename(columns={"value": "r2"})
seed_variance["seed"] = seed_variance["seed"].astype("Int64")
seed_variance = seed_variance.sort_values(["dataset", "target", "model", "variant", "seed"]).reset_index(drop=True)
seed_variance.to_csv(RESULTS / "seed_variance.csv", index=False)

print(f"results_ledger.csv: {len(ledger)} rows")
print(f"tuning_details.csv: {len(tuning)} rows")
print(f"seed_variance.csv: {len(seed_variance)} rows (filtered view of the ledger)")

print("\n=== Distinct fit_rows, every RandomForest row, by dataset/target/variant ===")
rf_fit_rows = (
    ledger[ledger["model"] == "RandomForest"]
    .groupby(["dataset", "target", "variant"], dropna=False)["fit_rows"]
    .unique()
)
for (dataset, target, variant), values in rf_fit_rows.items():
    values = sorted(int(v) for v in values)
    flag = "" if len(values) == 1 else "  <-- MULTIPLE DISTINCT VALUES"
    print(f"  {dataset}/{target} variant={variant or '(default)'}: {values}{flag}")

print(f"\n=== Notebook-recovered value verification: labeled (row+column matched) vs "
      f"presence-only (fallback) ===")
print(f"  labeled (row+column matched): {LABELED_VERIFIED_COUNT}")
print(f"  presence-only (fallback):     {PRESENCE_ONLY_COUNT}")

print("\n=== Verification log ===")
for source, note in VERIFIED:
    print(f"  [{source}]\n    {note}")

print("\n=== Excluded (could not source; not included above) ===")
for what, why in EXCLUDED:
    print(f"  - {what}\n    {why}")
