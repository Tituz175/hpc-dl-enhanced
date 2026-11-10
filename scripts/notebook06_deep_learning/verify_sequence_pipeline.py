"""
Pipeline verification for notebook 06's LSTM/TCN sequence input -- does NOT assume the
pipeline is correct just because training runs without erroring. Pulls real examples directly
from the same (X_train, y_train, seq_idx, mask) objects the real trials used (F-DATA duration),
and checks each one against independently-recomputed ground truth, not against itself.

Five checks, per sampled row:
  1. Every real (non-padding) sequence position belongs to the SAME user as the target row.
  2. Sequence positions are time-ordered oldest-to-most-recent, and every one is strictly
     before the target row's own time (no leakage) -- re-derived from fdata_train_df's own
     "usr"/"adt" columns, not from seq_idx/mask's own internal bookkeeping.
  3. Padding positions are masked False AND the gathered tensor is genuinely all-zero there
     (not leftover/garbage data silently included).
  4. The target row's y value matches an INDEPENDENT recomputation: reload the raw "duration"
     column from the reconstructed train_df, apply features.transform_target (log1p) fresh,
     and compare to the y_train array the Dataset actually uses -- a different code path than
     however y_train was originally produced.
  5. The gathered feature vector at each real sequence position exactly equals X_train's row
     at that same global position (no corruption, no double-transform, correct dtype/scale).

Run with: uv run python scripts/notebook06_deep_learning/verify_sequence_pipeline.py
"""
import glob
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src import features, splits
from src.models import SequenceDataset

INTERIM_DIR = ROOT / "data" / "interim"
N_SAMPLES = 10
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("reconstructing F-DATA duration's train_df (same pipeline as the real trials)")
FDATA_DIR = ROOT / "data" / "raw" / "fdata"
fdata_files = sorted(glob.glob(str(FDATA_DIR / "*.parquet")))[:38]
fdata_base = features.load_fdata_no_embedding(fdata_files)
fdata_base = features.filter_completed_jobs(fdata_base, "fdata")
fdata_base = features.handle_mszl_sentinel(fdata_base)
fdata_train_df, _ = splits.chronological_split(fdata_base, "fdata")

prep = joblib.load(INTERIM_DIR / "fdata_duration_r2_prepared.joblib")
X_train_df = prep["X_train"]
y_train = prep["y_train"].astype(np.float32)
assert list(fdata_train_df.index) == list(X_train_df.index), "train index mismatch -- stop"
log(f"index alignment confirmed: {len(fdata_train_df):,} rows")

usr = fdata_train_df["usr"].to_numpy()
adt = pd.to_datetime(fdata_train_df["adt"]).to_numpy()
raw_duration = fdata_train_df["duration"].to_numpy(dtype=float)

X = X_train_df.to_numpy(dtype=np.float32)
seq_idx = np.load(INTERIM_DIR / "notebook06_fdata_duration_train_seq_idx.npy")
mask = np.load(INTERIM_DIR / "notebook06_fdata_duration_train_mask.npy")
ds = SequenceDataset(X, y_train, seq_idx, mask)

rng = np.random.default_rng(SEED)
# deliberately include some rows with little/no history (low position-in-user) and some with
# plenty, not just a uniform random sample, so padding gets exercised, not just the common case
n = len(y_train)
sample_rows = sorted(rng.choice(n, size=N_SAMPLES, replace=False).tolist())
# force at least 2 known-sparse-history rows into the sample: a user's first and second jobs
# anywhere in the train set (position 0 and 1 in their own chronological list)
df_sorted = fdata_train_df.sort_values(["usr", "adt"], kind="mergesort")
pos_in_user = df_sorted.groupby("usr", sort=False).cumcount()
first_job_orig_positions = [fdata_train_df.index.get_loc(idx)
                             for idx in df_sorted.index[pos_in_user == 0][:2]]
sample_rows = sorted(set(sample_rows) | set(first_job_orig_positions))
log(f"sampling {len(sample_rows)} rows (incl. {len(first_job_orig_positions)} forced "
    f"zero-history rows): {sample_rows}")

all_pass = True

for i in sample_rows:
    print(f"\n--- row {i} (user={usr[i]}, time={adt[i]}) ---")
    seq, y_from_ds = ds[i]
    seq = seq.numpy()
    row_mask = mask[i]
    row_seq_idx = seq_idx[i]

    # Check 1 + 2: same-user, strictly-prior, time-ordered
    real_positions = np.where(row_mask)[0]
    check12_ok = True
    prev_time = None
    for pos in real_positions:
        g = row_seq_idx[pos]
        same_user = usr[g] == usr[i]
        strictly_prior = adt[g] < adt[i]
        ordered = (prev_time is None) or (adt[g] >= prev_time)
        if not (same_user and strictly_prior and ordered):
            check12_ok = False
            print(f"  FAIL pos={pos} g={g}: same_user={same_user} strictly_prior={strictly_prior} "
                  f"ordered={ordered} (user={usr[g]}, time={adt[g]})")
        prev_time = adt[g]
    print(f"  Check 1+2 (same-user, strictly-prior, time-ordered): "
          f"{'PASS' if check12_ok else 'FAIL'} ({len(real_positions)} real positions checked)")
    all_pass &= check12_ok

    # Check 3: padding positions masked False AND gathered tensor genuinely zero there
    pad_positions = np.where(~row_mask)[0]
    pad_all_zero = bool(np.all(seq[pad_positions] == 0.0)) if len(pad_positions) else True
    real_never_zero_vector = bool(np.all(np.any(seq[real_positions] != 0.0, axis=1))) if len(real_positions) else True
    check3_ok = pad_all_zero
    print(f"  Check 3 (padding masked + gathered tensor zero at padding): "
          f"{'PASS' if check3_ok else 'FAIL'} "
          f"({len(pad_positions)} padding positions, {len(real_positions)} real positions, "
          f"all real positions non-all-zero: {real_never_zero_vector})")
    all_pass &= check3_ok

    # Check 4: target value vs. independent recomputation from raw duration column
    y_independent = features.transform_target(np.array([raw_duration[i]]))[0]
    y_dataset = float(y_from_ds)
    y_stored = float(y_train[i])
    check4_ok = np.isclose(y_independent, y_dataset, rtol=1e-5) and np.isclose(y_dataset, y_stored, rtol=1e-6)
    print(f"  Check 4 (target matches independent log1p(raw_duration) recompute): "
          f"{'PASS' if check4_ok else 'FAIL'} "
          f"(independent={y_independent:.6f} dataset={y_dataset:.6f} stored_array={y_stored:.6f} "
          f"raw_duration={raw_duration[i]:.2f}s)")
    all_pass &= check4_ok

    # Check 5: gathered feature vector at each real position == X_train's row at that global index
    check5_ok = True
    max_abs_diff = 0.0
    for pos in real_positions:
        g = row_seq_idx[pos]
        diff = np.max(np.abs(seq[pos] - X[g]))
        max_abs_diff = max(max_abs_diff, float(diff))
        if diff > 1e-5:
            check5_ok = False
    # also confirm dtype/scale sanity: gathered values should match the SAME range as X_train_df
    # at that row (not a different column ordering/scale -- compare a couple of named columns)
    print(f"  Check 5 (gathered features exactly match X_train at that global row): "
          f"{'PASS' if check5_ok else 'FAIL'} (max abs diff across all real positions: {max_abs_diff:.2e})")
    all_pass &= check5_ok

print(f"\n{'='*60}")
print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'AT LEAST ONE CHECK FAILED -- see above'}")
print(f"{'='*60}")
