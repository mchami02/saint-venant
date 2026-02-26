"""Training hyperparameter defaults for wavefront learning.

Centralises the hardcoded training constants that were previously scattered
across train.py and training_loop.py.  All values are exposed via a single
frozen dataclass instance ``TRAINING_DEFAULTS``.
"""

from dataclasses import dataclass


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


TRAINING_DEFAULTS = TrainingDefaults()
