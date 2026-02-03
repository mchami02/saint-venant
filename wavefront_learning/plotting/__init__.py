"""Plotting utilities for wavefront learning results.

This package provides visualization functions for comparing predictions,
plotting trajectories, and logging to W&B.

Modules:
    base: Common setup and helper functions
    grid_plots: Grid comparison and error visualization
    trajectory_plots: Shock trajectory visualization
    hybrid_plots: HybridDeepONet-specific visualization
"""

from .base import (
    _create_comparison_animation,
    _get_colors,
    _get_extent,
    _log_figure,
    _plot_heatmap,
    save_figure,
)
from .grid_plots import (
    plot_comparison_wandb,
    plot_error_map,
    plot_grid_comparison,
    plot_prediction_comparison,
)
from .hybrid_plots import (
    plot_hybrid_predictions_wandb,
)
from .trajectory_plots import (
    plot_existence_heatmap,
    plot_loss_curves,
    plot_sample_predictions,
    plot_shock_trajectories,
    plot_trajectory_on_grid,
    plot_trajectory_on_grid_wandb,
    plot_trajectory_wandb,
    plot_wavefront_trajectory,
)

__all__ = [
    # Base utilities
    "save_figure",
    "_get_extent",
    "_get_colors",
    "_plot_heatmap",
    "_create_comparison_animation",
    "_log_figure",
    # Grid plots
    "plot_prediction_comparison",
    "plot_error_map",
    "plot_comparison_wandb",
    "plot_grid_comparison",
    # Trajectory plots
    "plot_shock_trajectories",
    "plot_existence_heatmap",
    "plot_trajectory_on_grid",
    "plot_trajectory_on_grid_wandb",
    "plot_trajectory_wandb",
    "plot_wavefront_trajectory",
    "plot_loss_curves",
    "plot_sample_predictions",
    # Hybrid plots
    "plot_hybrid_predictions_wandb",
]
