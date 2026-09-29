"""
Reproduce F-DATA power's calibrated analytical baseline as a real, saved artifact.
notebooks/05c's own markdown says the model is feasible, but no code cell in 05c actually
calls calibrate_fdata_power_model or evaluates it -- checked directly, not assumed. This
script closes that gap with an independently reproducible, saved result. Full provenance,
including the original never-persisted computation this reproduces exactly (coeffs
p_idle=107.98663207789151, gamma=-112.62417703748197, test R2=0.9775433474047817,
MAPE=514.4758112932082), is in results/investigation_log.md, section 4.

Pipeline mirrors notebooks/05c_classical_ml_power_duration.ipynb cells 6-16 exactly (same
load -> filter -> sanitize -> split order), so the resulting train/test row counts should
match notebook 05c's own fdata_power_prepared.joblib (16,315,866 / 6,992,514) as a
consistency check, plus the docstring-required handle_msza_sentinel step for the
calibration's msza predictor specifically.

Run with: uv run python scripts/fdata_power_calibrated_baseline/fit_and_evaluate.py
"""
import glob
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib

from src import features, metrics, roofline, splits

FDATA_DIR = ROOT / "data" / "raw" / "fdata"
INTERIM_DIR = ROOT / "data" / "interim"

t0 = time.perf_counter()

fdata_files = sorted(glob.glob(f"{FDATA_DIR}/*.parquet"))
print(f"[{time.strftime('%H:%M:%S')}] loading {len(fdata_files)} F-DATA months")
fdata = features.load_fdata_no_embedding(fdata_files)
print(f"[{time.strftime('%H:%M:%S')}] raw rows: {len(fdata):,}")

fdata = features.filter_completed_jobs(fdata, "fdata")
print(f"[{time.strftime('%H:%M:%S')}] rows after filter_completed_jobs: {len(fdata):,}")

n_before = len(fdata)
fdata = features.sanitize_fdata_power(fdata)
features.assert_fdata_power_sanitized(fdata)
fdata = fdata.dropna(subset=["avgpcon"])
print(f"[{time.strftime('%H:%M:%S')}] rows after power-sanitize dropna: {len(fdata):,} "
      f"({n_before - len(fdata)} corrupt avgpcon dropped)")

fdata = features.handle_msza_sentinel(fdata)
features.assert_msza_sanitized(fdata)
print(f"[{time.strftime('%H:%M:%S')}] msza_unlimited True: {fdata['msza_unlimited'].sum():,} "
      f"({fdata['msza_unlimited'].mean():.4%})")

train_df, test_df = splits.chronological_split(fdata, "fdata")
print(f"[{time.strftime('%H:%M:%S')}] train={len(train_df):,} test={len(test_df):,} "
      f"(test/train ratio={len(test_df) / len(train_df):.4f})")

print(f"[{time.strftime('%H:%M:%S')}] calibrating on the training-period-only subset (train_df) -- "
      f"test split is never touched by this step")
coeffs = roofline.calibrate_fdata_power_model(train_df)
print(f"coeffs: {coeffs}")

pred_test = roofline.predict_fdata_power(test_df, coeffs)
test_metrics = metrics.regression_metrics(test_df["avgpcon"].to_numpy(dtype=float), pred_test)
print(f"[{time.strftime('%H:%M:%S')}] calibrated model test metrics:")
for k, v in test_metrics.items():
    print(f"  {k}: {v:,.6f}")

elapsed = time.perf_counter() - t0
print(f"\n[{time.strftime('%H:%M:%S')}] done in {elapsed:.1f}s")

ORIGINAL = dict(
    R2=0.9775433474047817, MAPE=514.4758112932082,
    p_idle=107.98663207789151, gamma=-112.62417703748197,
    source="original never-persisted computation -- full provenance in "
           "results/investigation_log.md section 4",
)
print("\n=== comparison against the original never-persisted run ===")
print(f"  R2:    new={test_metrics['R2']:.10f}  original={ORIGINAL['R2']:.10f}  "
      f"diff={abs(test_metrics['R2'] - ORIGINAL['R2']):.3e}")
print(f"  MAPE:  new={test_metrics['MAPE']:.10f}  original={ORIGINAL['MAPE']:.10f}  "
      f"diff={abs(test_metrics['MAPE'] - ORIGINAL['MAPE']):.3e}")
print(f"  p_idle: new={coeffs.p_idle:.10f}  original={ORIGINAL['p_idle']:.10f}  "
      f"diff={abs(coeffs.p_idle - ORIGINAL['p_idle']):.3e}")
print(f"  gamma:  new={coeffs.gamma:.10f}  original={ORIGINAL['gamma']:.10f}  "
      f"diff={abs(coeffs.gamma - ORIGINAL['gamma']):.3e}")

INTERIM_DIR.mkdir(exist_ok=True)
out = dict(
    coeffs=coeffs, test_metrics=test_metrics,
    train_rows=len(train_df), test_rows=len(test_df),
    elapsed_seconds=elapsed,
    original_transcript_comparison=ORIGINAL,
)
out_path = INTERIM_DIR / "fdata_power_calibrated_baseline.joblib"
joblib.dump(out, out_path)
print(f"\nsaved: {out_path}")
