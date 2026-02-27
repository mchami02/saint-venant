"""YAML configuration loader for wavefront learning.

Reads YAML files from the configs/ directory and converts them to the
Python types that consumers expect (dicts, dataclass instances, tuples).
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


_CONFIG_DIR = Path(__file__).parent


def _load_yaml(name: str) -> dict:
    """Read a YAML file from the configs/ directory."""
    path = _CONFIG_DIR / name
    with open(path) as f:
        return yaml.safe_load(f)


# ── TrainingDefaults dataclass ──────────────────────────────────────────


@dataclass(frozen=True)
class TrainingDefaults:
    """Immutable training hyperparameter defaults."""

    # AdamW
    weight_decay: float = 1e-4

    # ReduceLROnPlateau
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_threshold: float = 0.01

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_threshold: float = 0.99

    # Gradient clipping
    grad_clip_max_norm: float = 1.0

    # KL annealing (CVAE)
    kl_warmup_fraction: float = 0.2

    # Periodic plotting
    plot_every_n_epochs: int = 5


# ── Preset loading ──────────────────────────────────────────────────────


def _parse_loss_presets(raw: dict) -> dict:
    """Convert YAML loss_presets to tuple format expected by CombinedLoss.

    YAML format:
        preset_name:
          - name: mse
            weight: 1.0
          - name: kl_divergence
            weight: 1.0
            kwargs:
              free_bits: 0.01

    Python format:
        {"preset_name": [("mse", 1.0), ("kl_divergence", 1.0, {"free_bits": 0.01})]}
    """
    presets = {}
    for preset_name, entries in raw.items():
        preset = []
        for entry in entries:
            if "kwargs" in entry:
                preset.append((entry["name"], entry["weight"], entry["kwargs"]))
            else:
                preset.append((entry["name"], entry["weight"]))
        presets[preset_name] = preset
    return presets


def load_presets() -> dict:
    """Load all preset mappings from presets.yaml.

    Returns a dict with keys:
        model_loss_preset, model_plot_preset, model_transform,
        loss_presets, plot_presets
    """
    data = _load_yaml("presets.yaml")
    data["loss_presets"] = _parse_loss_presets(data["loss_presets"])
    return data


def load_training_defaults() -> TrainingDefaults:
    """Load training hyperparameter defaults from training.yaml."""
    data = _load_yaml("training.yaml")
    return TrainingDefaults(**data["training_defaults"])


def load_cli_defaults() -> dict:
    """Load default CLI argument values from training.yaml."""
    data = _load_yaml("training.yaml")
    return data["cli_defaults"]
