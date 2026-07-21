"""Model wrappers shared across notebooks 05-07.

RF/XGBoost/LightGBM (Decision #14), FNN (Decision #12), LSTM/TCN
(Decision #13), and the Hybrid residual-learning head all live behind a
common interface here so notebook 08's evaluation sweep can iterate over
them uniformly.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Regressor": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class TuningBudget:
    """Fixed, documented hyperparameter search budget per model family
    (Decision #5) — same budget for every tunable model in the head-to-head
    comparison."""
    n_trials: int = 50
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


# RF / XGBoost / LightGBM (notebook 05), FNN / LSTM / TCN (notebook 06),
# and the Hybrid model (notebook 07) get their concrete implementations
# once feature engineering (notebook 02) and analytical baselines
# (notebook 03) exist to build on.
