"""Plotting utilities for wavefront learning results.

This package provides visualization functions for comparing predictions,
plotting trajectories, and logging results.

Modules:
    base: Common setup and helper functions
    grid_plots: Grid comparison and error visualization
    trajectory_plots: Core shock trajectory visualization
    wandb_trajectory_plots: Trajectory plots for the PLOTS registry
    hybrid_plots: HybridDeepONet-specific visualization

Main entry point:
    plot(): Unified plotting function that uses presets
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
    plot_comparison,
    plot_error_map,
    plot_grid_comparison,
    plot_ground_truth,
    plot_pde_residual,
    plot_pred,
    plot_prediction_comparison,
)
from .hybrid_plots import (
    plot_hybrid_predictions,
    plot_mse_error,
    plot_pred_traj,
    plot_prediction_with_trajectory_existence,
    plot_region_weights,
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
    plot_existence,
    plot_grid_with_acceleration,
    plot_grid_with_trajectory_existence,
    plot_gt_traj,
    plot_trajectory_vs_analytical,
)
from .charno_plots import (
    plot_charno_decomposition,
    plot_local_densities,
    plot_selection_entropy,
    plot_selection_weights,
    plot_winning_segment,
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
    "plot_comparison",
    "plot_grid_comparison",
    "plot_ground_truth",
    "plot_pde_residual",
    "plot_pred",
    # Trajectory plots (core)
    "plot_shock_trajectories",
    "plot_existence_heatmap",
    "plot_trajectory_on_grid",
    "plot_wavefront_trajectory",
    "plot_loss_curves",
    "plot_sample_predictions",
    "_compute_acceleration_numpy",
    # Trajectory plots (PLOTS registry)
    "plot_grid_with_trajectory_existence",
    "plot_grid_with_acceleration",
    "plot_trajectory_vs_analytical",
    "plot_existence",
    # Hybrid plots
    "plot_hybrid_predictions",
    "plot_prediction_with_trajectory_existence",
    "plot_pred_traj",
    "plot_mse_error",
    "plot_region_weights",
    # Trajectory-only plots (no existence)
    "plot_gt_traj",
    # CharNO diagnostic plots
    "plot_selection_weights",
    "plot_winning_segment",
    "plot_selection_entropy",
    "plot_local_densities",
    "plot_charno_decomposition",
]
