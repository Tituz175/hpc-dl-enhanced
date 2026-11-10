"""Model wrappers shared across notebooks 05-07.

RF/XGBoost/LightGBM (run as an ablation against each other under the same
tuning budget rather than picking one up front), a plain FNN (kept as
originally proposed, no entity-embedding/FT-Transformer upgrade -- if the
tree ensembles beat it, that's a legitimate literature-consistent finding,
not something to fix by giving the FNN extra architectural help), and
LSTM/TCN (both implemented, not just LSTM) using the inter-job (user
job-history) sequence framing from notebook 06's sequence-construction
investigation -- a job's past SEQ_LEN jobs, chronologically ordered, each
represented by that job's own Tier A feature vector. (An earlier intra-job
framing, using PM100's per-job power time series as LSTM/TCN input, was
considered and dropped: that field is only populated during/after a job's
own execution, so using it to predict that same job's target would be a
Tier B leakage violation.) The Hybrid residual-learning head will live
behind the same common interface once notebook 07 is built.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Regressor": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class TuningBudget:
    """Fixed, documented hyperparameter search budget per model family --
    same budget for every tunable model in the head-to-head comparison.
    This is the CLASSICAL-ML (RF/XGBoost/LightGBM) budget, notebooks
    05/05b/05c -- NOT reused uncritically for the deep-learning models
    (notebook 06), whose per-epoch cost is high enough to need its own,
    empirically-timed budget instead. See DLTuningBudget below."""
    n_trials: int = 50
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass
class DLTuningBudget:
    """Per-architecture Optuna trial budget and max-epoch cap for notebook
    06's FNN/LSTM/TCN, set from real single-epoch timing measurements
    rather than guessed upfront. Populated once that timing is in hand;
    left as None until then so a premature default can't be used by
    accident."""
    n_trials: int | None = None
    max_epochs: int | None = None


# --- FNN -----------------------------------------------------------------

class FNN(nn.Module):
    """Plain feedforward network on the flat Tier A feature set -- the same
    features the classical ML models (05/05b/05c) use, no sequence input.
    Three hidden layers (256/256/128) with ReLU + dropout; a single linear
    output unit for regression in log-space (matching the classical models'
    own target transform)."""

    def __init__(self, input_dim: int, hidden: tuple[int, int, int] = (256, 256, 128),
                 dropout: float = 0.1):
        super().__init__()
        h1, h2, h3 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --- LSTM ------------------------------------------------------------------

class SequenceLSTM(nn.Module):
    """A user's past SEQ_LEN jobs (each job's own Tier A feature vector,
    oldest first, zero-padded at the front for users with less history --
    see src/sequences.py) through a 2-layer LSTM, PLUS the current (target)
    job's own scaled feature vector, concatenated with the LSTM's final
    hidden state before the output head -- i.e. FNN plus history, not
    history alone.

    Found and fixed as a real bug, not a design choice made up front:
    build_sequence_index only ever gathers STRICTLY PRIOR jobs (by
    construction, to avoid leakage -- confirmed directly, sampling 5 real
    rows and checking that a target row's own index never appears in its
    own sequence). That is correct for the sequence itself, but earlier
    versions of this model then saw ONLY that history and nothing else --
    the current job's own resource request, size, and other submission-time
    characteristics never reached the model at all, which is most of what
    FNN uses to get R2=0.905 (log-space, PM100 power) against LSTM-without-
    current-features' R2=0.276 on the identical data. Concatenating the
    current job's own features back in (the history encoding augments them,
    rather than being the model's only information) is the fix."""

    def __init__(self, history_dim: int, current_dim: int, hidden: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(history_dim, hidden, num_layers=num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden + current_dim, 1)

    def forward(self, seq: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        # seq: (batch, seq_len, history_dim); current: (batch, current_dim)
        out, _ = self.lstm(seq)
        last = out[:, -1, :]
        combined = torch.cat([last, current], dim=-1)
        return self.head(combined).squeeze(-1)


# --- TCN ---------------------------------------------------------------

class _CausalConvBlock(nn.Module):
    """One dilated-causal-convolution residual block (Bai et al. 2018's TCN
    pattern, simplified): two causal conv1d + ReLU + dropout layers, plus a
    1x1 residual projection when the channel count changes. Causal padding
    (pad the LEFT only, by (kernel_size-1)*dilation, then trim any
    right-side overhang) keeps position t's output a function of positions
    <= t only -- the same "no future leakage" property src/sequences.py's
    build_sequence_index already enforces at the data level, preserved here
    at the architecture level too."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        # Plain Conv1d (weight_norm tried and removed): an unnormalized first-pass
        # implementation diverged to loss values of 1e13-1e14, weight_norm alone only
        # partially helped (faster recovery from the initial blow-up, but still plateaued
        # ~10 orders of magnitude above baseline after 20 epochs) -- the real root cause was
        # unscaled input features (see src/scaling.py), not the conv layers' own normalization.
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad,
                                dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad,
                                dilation=dilation)
        self.pad = pad
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                            if in_channels != out_channels else None)

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.pad] if self.pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self._trim(self.conv1(x))))
        out = self.dropout(self.relu(self._trim(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SequenceTCN(nn.Module):
    """Same sequence input as SequenceLSTM (batch, seq_len, history_dim),
    transposed to (batch, history_dim, seq_len) for Conv1d. 4 levels of
    dilated causal convolution (dilation 1/2/4/8, two conv layers per level
    at that level's dilation -- receptive field 61 at kernel_size=3,
    computed as 1 + sum over every conv layer of (kernel_size-1)*dilation,
    read directly from the real instantiated model's own Conv1d modules,
    not estimated by hand -- comfortably covers SEQ_LEN=10), channel width
    64 throughout after the first block. PLUS the current job's own scaled
    feature vector, concatenated with the final timestep's channel vector
    before the output head -- same FNN-plus-history fix as SequenceLSTM,
    same reason (see SequenceLSTM's docstring for the full finding)."""

    def __init__(self, history_dim: int, current_dim: int, channels: int = 64, levels: int = 4,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        blocks = []
        in_ch = history_dim
        for level in range(levels):
            blocks.append(_CausalConvBlock(in_ch, channels, kernel_size,
                                            dilation=2 ** level, dropout=dropout))
            in_ch = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels + current_dim, 1)

    def forward(self, seq: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        # seq: (batch, seq_len, history_dim) -> (batch, history_dim, seq_len)
        out = self.blocks(seq.transpose(1, 2))
        last = out[:, :, -1]
        combined = torch.cat([last, current], dim=-1)
        return self.head(combined).squeeze(-1)


# --- Datasets --------------------------------------------------------------

class FlatDataset(Dataset):
    """FNN input: one row's flat Tier A feature vector -> its (log-space)
    target. X is a 2D float32 array (n_rows, n_features)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    """LSTM/TCN input: gathers each row's SEQ_LEN-length sequence on the
    fly from a flat feature table, using the (seq_idx, mask) pair
    src/sequences.build_sequence_index produces -- avoids ever
    materializing the full (n_rows, SEQ_LEN, n_features) tensor at once
    (prohibitive at F-DATA's scale: ~16.3M rows x 10 x ~95 features x
    4 bytes is ~62GB). X stays a plain float32 numpy array on CPU; only
    one batch's worth of gathered sequences is built and moved to GPU at
    a time, by the training loop's own device-transfer step.

    X is expected to already be standardized (src/scaling.py) before
    reaching this class -- after standardization, a real feature value can
    legitimately land near 0, which would be indistinguishable from a
    zero-filled padding position without an explicit signal. The mask is
    therefore appended as an EXTRA channel (1.0 = real position, 0.0 =
    padding) on every timestep, so the sequence's per-timestep dimension is
    n_features + 1, not n_features -- a padded position can never look like
    an average real job just because standardization centers the data near
    zero.

    Also returns the TARGET row's own scaled feature vector (current,
    plain n_features wide, no mask channel) alongside the sequence --
    LSTM/TCN need this directly (see SequenceLSTM's docstring: the
    sequence alone never contains the current job's own features, by
    construction, since it only gathers strictly-prior jobs)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_idx: np.ndarray, mask: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.seq_idx = seq_idx
        self.mask = mask
        self.n_features = self.X.shape[1]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        idx_row = self.seq_idx[idx]        # (SEQ_LEN,), -1 where padded
        mask_row = self.mask[idx]          # (SEQ_LEN,), bool
        seq = np.zeros((len(idx_row), self.n_features + 1), dtype=np.float32)
        real = idx_row[mask_row]
        seq[mask_row, :self.n_features] = self.X[real]
        seq[:, self.n_features] = mask_row.astype(np.float32)  # mask channel: 1.0 real, 0.0 pad
        current = self.X[idx]              # the TARGET row's own scaled features
        return torch.from_numpy(seq), torch.from_numpy(current), self.y[idx]


# RF / XGBoost / LightGBM (notebook 05) get their concrete implementations
# inline in each notebook/script (sklearn/xgboost/lightgbm estimators used
# directly, no wrapper needed). The Hybrid residual-learning head
# (notebook 07) gets its concrete implementation once notebook 06's
# FNN/LSTM/TCN results establish which architecture it builds on.
