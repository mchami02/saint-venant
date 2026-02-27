"""Configuration package for wavefront learning.

Re-exports all public names from submodules for convenience:
    from configs import MODEL_LOSS_PRESET, TRAINING_DEFAULTS, parse_args
"""

from configs.cli import parse_args
from configs.presets import (
    LOSS_PRESETS,
    MODEL_LOSS_PRESET,
    MODEL_PLOT_PRESET,
    MODEL_TRANSFORM,
    PLOT_PRESETS,
)
from configs.training_defaults import TRAINING_DEFAULTS, TrainingDefaults

__all__ = [
    "LOSS_PRESETS",
    "MODEL_LOSS_PRESET",
    "MODEL_PLOT_PRESET",
    "MODEL_TRANSFORM",
    "PLOT_PRESETS",
    "TRAINING_DEFAULTS",
    "TrainingDefaults",
    "parse_args",
]
