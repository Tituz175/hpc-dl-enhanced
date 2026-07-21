"""Experiment config loading (Decision #11: versioned config per experiment).

Each run should load its hyperparameters/settings from a YAML file under
configs/, and that file should be saved alongside its results so any
result can be traced back to the exact settings that produced it.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file from configs/ by name (e.g. "fnn_fdata.yaml")."""
    path = CONFIGS_DIR / name
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class Seeds:
    """Fixed seeds for reproducibility (Decision #11) and multi-seed
    significance testing (Decision #6)."""
    values: tuple[int, ...] = (0, 1, 2, 3, 4)
