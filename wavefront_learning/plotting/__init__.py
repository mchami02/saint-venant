"""Plotting utilities for wavefront learning results.

This package provides visualization functions for comparing predictions,
plotting trajectories, and logging to W&B.

Modules:
    base: Common setup and helper functions
    grid_plots: Grid comparison and error visualization
    trajectory_plots: Core shock trajectory visualization
    wandb_trajectory_plots: W&B-specific trajectory plots (PLOTS registry)
    hybrid_plots: HybridDeepONet-specific visualization

Main entry point:
    plot_wandb(): Unified plotting function that uses presets
    PLOTS: Registry of available plot functions
    PLOT_PRESETS: Pre-configured plot combinations
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
    plot_ground_truth_wandb,
    plot_prediction_comparison,
)
from .hybrid_plots import (
    plot_hybrid_predictions_wandb,
    plot_mse_error_wandb,
    plot_prediction_with_trajectory_wandb,
    plot_region_weights_wandb,
)
from .trajectory_plots import (
    _compute_acceleration_numpy,
    plot_existence_heatmap,
    plot_loss_curves,
    plot_sample_predictions,
    plot_shock_trajectories,
    plot_trajectory_on_grid,
    plot_wavefront_trajectory,
)
from .wandb_trajectory_plots import (
    plot_existence_wandb,
    plot_grid_with_acceleration_wandb,
    plot_grid_with_trajectory_wandb,
    plot_trajectory_on_grid_wandb,
    plot_trajectory_vs_analytical_wandb,
    plot_trajectory_wandb,
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
    "plot_ground_truth_wandb",
    # Trajectory plots (core)
    "plot_shock_trajectories",
    "plot_existence_heatmap",
    "plot_trajectory_on_grid",
    "plot_wavefront_trajectory",
    "plot_loss_curves",
    "plot_sample_predictions",
    "_compute_acceleration_numpy",
    # Trajectory plots (W&B / PLOTS registry)
    "plot_trajectory_on_grid_wandb",
    "plot_trajectory_wandb",
    "plot_grid_with_trajectory_wandb",
    "plot_grid_with_acceleration_wandb",
    "plot_trajectory_vs_analytical_wandb",
    "plot_existence_wandb",
    # Hybrid plots
    "plot_hybrid_predictions_wandb",
    "plot_prediction_with_trajectory_wandb",
    "plot_mse_error_wandb",
    "plot_region_weights_wandb",
]
