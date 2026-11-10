"""
Shared training/evaluation pipeline for notebook 06's deep-learning sweep (FNN/LSTM/TCN,
all six (dataset, target) combinations), used by both
scripts/notebook06_deep_learning/run_sweep_pair.py (the original detached CLI sweep that
produced every pair's persistent Optuna study and checkpoint) and notebooks
06b_model_architectures/06c_dl_power/06d_dl_duration/06e_dl_memory (which reuse those same
studies/checkpoints live, never retraining from scratch since every pair already has
N_TRIALS completed trials).

Uses src/models.py's FNN/SequenceLSTM/SequenceTCN/FlatDataset/SequenceDataset directly --
notebook 06b keeps its own verbatim copies of these same six classes, purely for inline
documentation, and proves them behaviorally identical to src/models.py's own objects (same
state_dict keys/shapes, same output on a real batch given the same weights). This module has
no notebook-local classes to build on, so it always uses the src/models.py originals -- the
same ones scripts/notebook06_deep_learning/run_sweep_pair.py already imported before this
module existed.
"""
import pathlib
import time

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src import features, metrics, scaling, sequences, splits
from src.dl_data import load_combo
from src.models import FNN, SequenceLSTM, SequenceTCN, FlatDataset, SequenceDataset

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Exactly the original sweep's own budget (set from real single-epoch timing measurements,
# not guessed) -- the only Optuna search dimension across all 18 pairs is the learning rate.
N_TRIALS = 10
MAX_EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 4096

STUDIES_DIR = ROOT / "results" / "optuna_studies"
CKPT_DIR = ROOT / "data" / "interim" / "notebook06_sweep_models"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

ARCHITECTURES = ["FNN", "LSTM", "TCN"]


def prepare_combo_data(combo, log_fn=print, tie_policy="exclude"):
    """Real data-loading + feature-prep pipeline for one (dataset, target) combination --
    the same src.dl_data.load_combo call scripts/notebook06_deep_learning/run_sweep_pair.py
    and every notebook that needs real DL data both go through, so every number produced
    downstream is reproduced live here, not transcribed from a results CSV.

    tie_policy is passed straight through to src.sequences.build_sequence_index -- the
    default ("exclude") reproduces every prior call to this function exactly; "skip_back" is
    the PM100-only sequence-tie variant explored in notebooks/06g_sequence_tie_variant.ipynb
    and does not change this function's default behavior for any other caller."""
    data = load_combo(combo)
    dataset_name = data["dataset_name"]
    X_train_df, X_test_df = data["X_train_df"], data["X_test_df"]
    y_train, y_test_log, y_test_raw = data["y_train"], data["y_test_log"], data["y_test_raw"]
    user_ids_full, times_full, n_train = (data["user_ids_full"], data["times_full"],
                                           data["n_train"])
    n_test = len(X_test_df)

    # tune_train/tune_val split, within the TRAIN portion only -- test data never touches this
    tmp_df = pd.DataFrame({"_u": user_ids_full[:n_train], "_t": times_full[:n_train]})
    time_col_name = {"fdata": "adt", "pm100": "submit_time"}[dataset_name]
    tmp_df = tmp_df.rename(columns={"_t": time_col_name})
    tune_train_df, tune_val_df = splits.chronological_split(tmp_df, dataset_name, train_frac=0.8)
    is_tt = tmp_df.index.isin(tune_train_df.index)
    tt_pos = np.where(is_tt)[0]
    tv_pos = np.where(~is_tt)[0]
    test_pos = np.arange(n_train, n_train + n_test)

    X_full_df = pd.concat([X_train_df, X_test_df], ignore_index=True)
    y_full = np.concatenate([y_train, y_test_log]).astype(np.float32)
    X_tt_df = X_full_df.iloc[tt_pos].reset_index(drop=True)
    scaler_params = scaling.fit_scaler(X_tt_df)
    n_logged = sum(1 for p in scaler_params.values() if p["use_log"])

    # Sequences are built regardless of which architectures are about to run (LSTM and TCN
    # both need them; only FNN does not) -- cheap enough on PM100, and F-DATA pays this
    # cost once per combo here either way.
    t0 = time.perf_counter()
    seq_idx, mask = sequences.build_sequence_index(user_ids_full, times_full, tie_policy=tie_policy)
    X_scaled_full = scaling.apply_scaler(X_full_df, scaler_params)
    seq_seconds = time.perf_counter() - t0

    if log_fn is not None:
        log_fn(f"{combo}: {n_train:,} train / {n_test:,} test rows; scaler log-transformed "
               f"{n_logged}/{len(scaler_params)} columns; sequences built in {seq_seconds:.0f}s, "
               f"fill rate {mask.mean():.2%}")

    return dict(combo=combo, dataset_name=dataset_name, X_full_df=X_full_df, y_full=y_full,
                scaler_params=scaler_params, tt_pos=tt_pos, tv_pos=tv_pos, test_pos=test_pos,
                seq_idx=seq_idx, mask=mask, X_scaled_full=X_scaled_full,
                y_test_log=y_test_log, y_test_raw=y_test_raw,
                user_ids_full=user_ids_full, n_train=n_train)


def build_model(arch, current_dim, history_dim=None):
    if arch == "FNN":
        return FNN(current_dim)
    elif arch == "LSTM":
        return SequenceLSTM(history_dim, current_dim)
    else:
        return SequenceTCN(history_dim, current_dim)


def run_pair(prepared, arch, device=None, log_fn=None, epoch_log_dir=None,
             ckpt_dir=None, study_suffix="", sampler=None):
    """Train/tune (only if the persistent Optuna study is not yet complete) and evaluate one
    (combination, architecture) pair on the real held-out test split.

    ckpt_dir/study_suffix default to None/"" , which reproduce the exact checkpoint path and
    Optuna study name every prior call to this function used (CKPT_DIR, f"{combo}_{arch}") --
    passing them (as notebooks/06g_sequence_tie_variant.ipynb does, ckpt_dir=a separate
    directory, study_suffix="_skipback") trains into a completely separate checkpoint
    directory and a separate Optuna study/storage file, so this never reads, writes, or
    overwrites any of the original 18 pairs' checkpoints or studies.

    sampler defaults to None, which is also optuna.create_study's own default (an unseeded
    TPESampler) -- identical to every prior call. Passing a seeded sampler (e.g.
    optuna.samplers.TPESampler(seed=0)) makes the hyperparameter search itself reproducible.

    Reused, unmodified, by both scripts/notebook06_deep_learning/run_sweep_pair.py (the
    original training driver, which may still run fresh trials for a pair that has not
    reached N_TRIALS completed trials yet) and notebooks 06c/06d/06e (which always find all
    18 pairs' studies already at N_TRIALS/N_TRIALS completed trials in this project, so the
    study.optimize() call below runs zero further training and the final evaluation reloads
    the already-saved `{combo}_{arch}_best.pt` checkpoint directly -- a few seconds of data
    loading and one test-set forward pass, not a fresh training run).

    log_fn: optional callable(str) for per-epoch/per-trial progress messages
    (run_sweep_pair.py passes its own timestamped logger; notebooks leave this as None).
    epoch_log_dir: optional directory to write one per-trial epoch-loss CSV to (matches
    run_sweep_pair.py's own output file naming; only meaningful when fresh trials actually
    run -- never passed by the notebooks, since their studies are always already complete).
    """
    log_fn = log_fn or (lambda msg: None)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    combo = prepared["combo"]
    X_full_df, y_full = prepared["X_full_df"], prepared["y_full"]
    scaler_params = prepared["scaler_params"]
    tt_pos, tv_pos, test_pos = prepared["tt_pos"], prepared["tv_pos"], prepared["test_pos"]
    seq_idx, mask, X_scaled_full = prepared["seq_idx"], prepared["mask"], prepared["X_scaled_full"]
    is_seq = arch != "FNN"
    baseline_mse = float(np.mean((y_full[tt_pos] - y_full[tt_pos].mean()) ** 2))

    def build_loaders(positions_train, positions_val, shuffle_train=True):
        if not is_seq:
            X_t = scaling.apply_scaler(X_full_df.iloc[positions_train].reset_index(drop=True),
                                        scaler_params)
            X_v = scaling.apply_scaler(X_full_df.iloc[positions_val].reset_index(drop=True),
                                        scaler_params)
            train_loader = DataLoader(FlatDataset(X_t, y_full[positions_train]),
                                       batch_size=BATCH_SIZE, shuffle=shuffle_train,
                                       num_workers=0, drop_last=shuffle_train)
            val_loader = DataLoader(FlatDataset(X_v, y_full[positions_val]),
                                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                     drop_last=False)
            return train_loader, val_loader, X_t.shape[1], None
        else:
            full_ds = SequenceDataset(X_scaled_full, y_full, seq_idx, mask)
            train_loader = DataLoader(Subset(full_ds, positions_train), batch_size=BATCH_SIZE,
                                       shuffle=shuffle_train, num_workers=0,
                                       drop_last=shuffle_train)
            val_loader = DataLoader(Subset(full_ds, positions_val), batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=0, drop_last=False)
            current_dim = X_scaled_full.shape[1]
            history_dim = current_dim + 1  # +1 for the padding-mask channel
            return train_loader, val_loader, current_dim, history_dim

    def forward_batch(model, batch):
        if is_seq:
            seq, current, yb = batch
            seq, current, yb = (seq.to(device, non_blocking=True),
                                 current.to(device, non_blocking=True),
                                 yb.to(device, non_blocking=True))
            return model(seq, current), yb
        else:
            xb, yb = batch
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            return model(xb), yb

    _ckpt_dir = ckpt_dir if ckpt_dir is not None else CKPT_DIR
    _ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tmpl = str(_ckpt_dir / f"{combo}_{arch}_trial{{trial}}.pt")
    final_ckpt_path = _ckpt_dir / f"{combo}_{arch}_best.pt"

    def run_one_trial(trial, lr):
        train_loader, val_loader, current_dim, history_dim = build_loaders(tt_pos, tv_pos)
        model = build_model(arch, current_dim, history_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        best_val = float("inf")
        patience_left = PATIENCE
        epoch_log = []
        diverged = False
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            t_ep0 = time.perf_counter()
            train_loss_sum, train_n = 0.0, 0
            for batch in train_loader:
                pred, yb = forward_batch(model, batch)
                opt.zero_grad()
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                train_loss_sum += loss.item() * len(yb)
                train_n += len(yb)
            train_loss = train_loss_sum / train_n

            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    pred, yb = forward_batch(model, batch)
                    val_loss_sum += loss_fn(pred, yb).item() * len(yb)
                    val_n += len(yb)
            val_loss = val_loss_sum / val_n
            t_ep = time.perf_counter() - t_ep0
            improved = val_loss < best_val
            log_fn(f"  trial {trial.number} epoch {epoch:2d}: train_loss={train_loss:.5f} "
                   f"val_loss={val_loss:.5f} lr={lr:.2e} epoch_seconds={t_ep:.1f} "
                   f"{'(best)' if improved else ''}")
            epoch_log.append(dict(trial=trial.number, epoch=epoch, train_loss=train_loss,
                                   val_loss=val_loss, epoch_seconds=t_ep))

            if not np.isfinite(train_loss) or not np.isfinite(val_loss) or val_loss > 1e6 * baseline_mse:
                diverged = True
                log_fn(f"  trial {trial.number} DIVERGED")
                break
            if improved:
                best_val = val_loss
                patience_left = PATIENCE
                torch.save(model.state_dict(), ckpt_tmpl.format(trial=trial.number))
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if epoch_log_dir is not None:
            pd.DataFrame(epoch_log).to_csv(
                pathlib.Path(epoch_log_dir)
                / f"notebook06_sweep_epochlog_{combo}_{arch}_trial{trial.number}.csv",
                index=False)
        if diverged:
            return float("inf")
        return best_val

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        return run_one_trial(trial, lr)

    study_name = f"{combo}_{arch}{study_suffix}"
    storage = f"sqlite:///{STUDIES_DIR / study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage, direction="minimize",
                                 load_if_exists=True, sampler=sampler)
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, N_TRIALS - n_completed)
    log_fn(f"study has {n_completed} completed trials already; running {remaining} more "
           f"(persistent storage: {storage})")
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    best_trial = study.best_trial
    log_fn(f"best trial: #{best_trial.number}, lr={best_trial.params['lr']:.2e}, "
           f"val_loss={best_trial.value:.5f} (ratio to baseline: {best_trial.value / baseline_mse:.4f})")

    # ---- final evaluation on the real held-out test split ----
    _, test_loader, current_dim, history_dim = build_loaders(tt_pos, test_pos, shuffle_train=False)
    model = build_model(arch, current_dim, history_dim).to(device)
    if remaining > 0:
        # Fresh trials just ran in THIS call: the winning trial's own per-trial checkpoint is
        # what is on disk right now. Promote it to the final checkpoint and delete the
        # per-trial checkpoints -- exactly what the original sweep script always did.
        best_ckpt_path = ckpt_tmpl.format(trial=best_trial.number)
        state = torch.load(best_ckpt_path, map_location=device)
        torch.save(state, final_ckpt_path)
        for f in _ckpt_dir.glob(f"{combo}_{arch}_trial*.pt"):
            f.unlink()
    else:
        # Study was already complete before this call: a prior run already promoted the
        # winning trial's checkpoint to the final one and deleted the per-trial files, so
        # that is what gets reloaded here.
        state = torch.load(final_ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds_log = []
    with torch.no_grad():
        for batch in test_loader:
            pred, _ = forward_batch(model, batch)
            preds_log.append(pred.cpu().numpy())
    preds_log = np.concatenate(preds_log)
    preds_raw = features.inverse_transform_target(preds_log)

    log_metrics = metrics.regression_metrics(prepared["y_test_log"], preds_log)
    raw_metrics = metrics.regression_metrics(prepared["y_test_raw"], preds_raw)

    return dict(arch=arch, best_trial=best_trial.number, best_lr=best_trial.params["lr"],
                log_metrics=log_metrics, raw_metrics=raw_metrics,
                preds_log=preds_log, preds_raw=preds_raw,
                y_test_log=prepared["y_test_log"], y_test_raw=prepared["y_test_raw"],
                fit_rows=len(tt_pos))


# ---------------------------------------------------------------------------------------
# Shared on-disk artifacts between notebooks 06c/06d/06e (which compute real results) and
# 06b/06f (which load them back rather than recomputing or reloading F-DATA/PM100
# themselves). Gitignored (data/interim/ as a whole already is) -- not committed.
# ---------------------------------------------------------------------------------------

RESULTS_INTERIM_DIR = ROOT / "data" / "interim" / "notebook06_results"
DIM_TABLE_PATH = RESULTS_INTERIM_DIR / "dim_table.csv"

_ALL_COMBOS = ["fdata_power", "fdata_duration", "fdata_memory",
               "pm100_power", "pm100_duration", "pm100_memory_c4"]


def save_pair_artifacts(combo, arch, prepared, result, out_dir=RESULTS_INTERIM_DIR):
    """Persists one (combination, architecture) pair's real test predictions/targets/user
    ids and metrics so notebook 06f (the comparison/diagnostic notebook) can load them
    without re-running run_pair or reloading F-DATA/PM100 itself. Test-split arrays only --
    never anything from train/tune_train/tune_val."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_user_ids = prepared["user_ids_full"][prepared["n_train"]:]
    np.savez(out_dir / f"{combo}_{arch}_test_predictions.npz",
             preds_log=result["preds_log"], preds_raw=result["preds_raw"],
             y_test_log=result["y_test_log"], y_test_raw=result["y_test_raw"],
             user_ids=test_user_ids)
    meta = dict(combo=combo, arch=arch, best_trial=result["best_trial"],
                best_lr=result["best_lr"], fit_rows=result["fit_rows"],
                **{f"log_{k}": v for k, v in result["log_metrics"].items()},
                **{f"raw_{k}": v for k, v in result["raw_metrics"].items()})
    pd.DataFrame([meta]).to_csv(out_dir / f"{combo}_{arch}_metrics.csv", index=False)


def load_pair_artifacts(combo, arch, out_dir=RESULTS_INTERIM_DIR):
    """Loads back what save_pair_artifacts wrote, as a dict shaped like run_pair's own
    return value (log_metrics/raw_metrics as plain dicts, preds/targets/user_ids as arrays)
    -- so 06f can build its comparison/diagnostic tables the same way whether the DL result
    came from a live run_pair call or a saved artifact."""
    out_dir = pathlib.Path(out_dir)
    pred_path = out_dir / f"{combo}_{arch}_test_predictions.npz"
    meta_path = out_dir / f"{combo}_{arch}_metrics.csv"
    if not pred_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {combo}/{arch} test-prediction artifacts under {out_dir} -- run "
            f"notebook 06c_dl_power.ipynb (power combinations), 06d_dl_duration.ipynb "
            f"(duration combinations), or 06e_dl_memory.ipynb (memory combinations) first, "
            f"whichever one produces {combo}, before this notebook.")
    npz = np.load(pred_path, allow_pickle=True)
    meta = pd.read_csv(meta_path).iloc[0]
    log_metrics = {k[len("log_"):]: meta[k] for k in meta.index if k.startswith("log_")}
    raw_metrics = {k[len("raw_"):]: meta[k] for k in meta.index if k.startswith("raw_")}
    return dict(arch=arch, best_trial=int(meta["best_trial"]), best_lr=float(meta["best_lr"]),
                fit_rows=int(meta["fit_rows"]), log_metrics=log_metrics, raw_metrics=raw_metrics,
                preds_log=npz["preds_log"], preds_raw=npz["preds_raw"],
                y_test_log=npz["y_test_log"], y_test_raw=npz["y_test_raw"],
                user_ids=npz["user_ids"])


def append_dim_row(combo, current_dim, out_dir=RESULTS_INTERIM_DIR):
    """06c/06d/06e each call this once per combination they run, building up the shared
    current_dim table notebook 06b reads -- rather than 06b reloading all six combinations'
    base data itself just to read off a feature count."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dim_table.csv"
    row = pd.DataFrame([dict(combo=combo, current_dim=current_dim, history_dim=current_dim + 1)])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["combo"] != combo]  # idempotent on re-run
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(path, index=False)


def load_dim_table(out_dir=RESULTS_INTERIM_DIR, expected_combos=None):
    """Loaded by notebook 06b. Raises, rather than silently reloading F-DATA/PM100 itself,
    if 06c/06d/06e have not all run yet -- the error names which notebook to run first."""
    out_dir = pathlib.Path(out_dir)
    path = out_dir / "dim_table.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run notebooks 06c_dl_power.ipynb, "
            f"06d_dl_duration.ipynb, and 06e_dl_memory.ipynb first (each writes its own "
            f"combinations' rows into this shared table); do not reload F-DATA/PM100 here "
            f"to work around a missing file.")
    dim_df = pd.read_csv(path).set_index("combo")
    expected_combos = expected_combos or _ALL_COMBOS
    missing = [c for c in expected_combos if c not in dim_df.index]
    if missing:
        raise FileNotFoundError(
            f"dim_table.csv at {path} is missing rows for {missing} -- run whichever of "
            f"06c_dl_power.ipynb / 06d_dl_duration.ipynb / 06e_dl_memory.ipynb produces "
            f"them first; do not reload F-DATA/PM100 here to work around this.")
    return dim_df.loc[expected_combos]
