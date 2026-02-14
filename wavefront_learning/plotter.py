"""Plotting factory for wavefront learning.

This module provides:
- PLOTS: Registry of available plot functions
- PLOT_PRESETS: Pre-configured plot combinations for common use cases
- plot(): Main entry point that runs plots based on preset

Each plot function signature:
    fn(traj_data, grid_config) -> list[tuple[str, Figure]]

Returns a list of (log_key, figure) pairs. Logging and cleanup is handled
centrally by plot().
"""

import matplotlib.pyplot as plt
from plotting import (
    _log_figure,
    plot_charno_decomposition,
    plot_existence,
    plot_grid_with_acceleration,
    plot_grid_with_trajectory_existence,
    plot_ground_truth,
    plot_gt_traj,
    plot_local_densities,
    plot_mse_error,
    plot_pde_residual,
    plot_pred,
    plot_pred_traj,
    plot_prediction_with_trajectory_existence,
    plot_region_weights,
    plot_selection_entropy,
    plot_selection_weights,
    plot_trajectory_vs_analytical,
    plot_winning_segment,
)

# Registry of individual plot functions
# Each function signature: fn(traj_data, grid_config) -> list[tuple[str, Figure]]
PLOTS: dict[str, callable] = {
    "ground_truth": plot_ground_truth,
    "pred": plot_pred,
    "grid_with_trajectory_existence": plot_grid_with_trajectory_existence,
    "grid_with_acceleration": plot_grid_with_acceleration,
    "trajectory_vs_analytical": plot_trajectory_vs_analytical,
    "existence": plot_existence,
    "prediction_with_trajectory_existence": plot_prediction_with_trajectory_existence,
    "mse_error": plot_mse_error,
    "pde_residual": plot_pde_residual,
    "region_weights": plot_region_weights,
    "gt_traj": plot_gt_traj,
    "pred_traj": plot_pred_traj,
    "charno_decomposition": plot_charno_decomposition,
    "selection_weights": plot_selection_weights,
    "winning_segment": plot_winning_segment,
    "selection_entropy": plot_selection_entropy,
    "local_densities": plot_local_densities,
}

# Presets for common configurations
# Each preset lists which plots to generate
PLOT_PRESETS: dict[str, list[str]] = {
    "shock_net": [
        "grid_with_trajectory_existence",  # Uses ground truth grid as background
        "grid_with_acceleration",  # Shows acceleration magnitude (shock locations)
    ],
    "hybrid": [
        "pred",  # Predicted grid heatmap
        "prediction_with_trajectory_existence",  # Uses predicted grid as background
        "mse_error",
        "existence",
        "region_weights",
    ],
    "traj_net": [
        "ground_truth",
        "gt_traj",
        "pred_traj",
        "pred",  # Predicted grid heatmap
        "mse_error",
        "pde_residual",
    ],
    "classifier_traj_net": [
        "gt_traj",
        "pred_traj",
        "pred",
        "mse_error",
        "existence",
    ],
    "grid_only": [
        "ground_truth",
        "pred",  # Predicted grid heatmap
        "mse_error",
        "pde_residual",
    ],
    "traj_transformer": [
        "ground_truth",
        "gt_traj",
        "pred_traj",
        "pred",
        "mse_error",
        "pde_residual",
    ],
    "classifier_traj_transformer": [
        "gt_traj",
        "pred_traj",
        "pred",
        "mse_error",
        "existence",
    ],
    "no_traj_transformer": [
        "ground_truth",
        "pred",
        "mse_error",
        "pde_residual",
    ],
    "classifier_all_traj_transformer": [
        "gt_traj",
        "pred_traj",
        "pred",
        "mse_error",
        "existence",
    ],
    "charno": [
        "ground_truth",
        "pred",
        "mse_error",
        "pde_residual",
        "charno_decomposition",
        "selection_weights",
        "winning_segment",
        "selection_entropy",
        "local_densities",
    ],
}


def plot(
    traj_data: dict,
    grid_config: dict,
    logger,
    epoch: int | None,
    mode: str = "val",
    preset: str | None = None,
) -> None:
    """Main plotting entry point.

    Calls plot functions from the registry, then logs the returned figures
    and closes them.

    Args:
        traj_data: Data dict from sample_trajectory_predictions().
        grid_config: Dict with {nx, nt, dx, dt}.
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: "train" or "val".
        preset: Preset name (shock_net, hybrid) or None for auto-detect.
    """
    if logger is None:
        return

    # Auto-detect preset if not specified
    if preset is None:
        preset = "hybrid" if traj_data.get("is_hybrid", False) else "shock_net"

    if preset not in PLOT_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(PLOT_PRESETS.keys())}"
        )

    # Run each plot in the preset
    for plot_name in PLOT_PRESETS[preset]:
        if plot_name in PLOTS:
            try:
                figures = PLOTS[plot_name](traj_data, grid_config)
                for key, fig in figures:
                    _log_figure(logger, f"{mode}/{key}", fig, epoch)
                    plt.close(fig)
            except Exception as e:
                # Log warning but don't fail - some plots may not have required data
                print(f"Warning: Plot '{plot_name}' failed: {e}")
