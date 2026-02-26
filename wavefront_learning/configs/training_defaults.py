"""Training hyperparameter defaults loaded from training.yaml.

All existing imports (e.g. ``from configs.training_defaults import TRAINING_DEFAULTS``)
continue to work unchanged.
"""

from configs.loader import TrainingDefaults, load_training_defaults

TRAINING_DEFAULTS = load_training_defaults()

__all__ = ["TrainingDefaults", "TRAINING_DEFAULTS"]
