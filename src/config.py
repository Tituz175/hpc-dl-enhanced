"""Experiment config loading.

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
    """Fixed seeds — used both to make individual runs reproducible, and to
    repeat training across multiple seeds so a paired significance test
    (Wilcoxon signed-rank) can back up any claim that one model beats
    another, rather than trusting a single point-estimate difference that
    could just be noise."""
    values: tuple[int, ...] = (0, 1, 2, 3, 4)
