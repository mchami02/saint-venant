"""Plotting factory for wavefront learning.

This module provides:
- PLOTS: Registry of available plot functions
- PLOT_PRESETS: Pre-configured plot combinations for common use cases
- plot_wandb(): Main entry point that runs plots based on preset

Each plot function signature:
    fn(traj_data, grid_config, logger, epoch, mode, **kwargs) -> None
"""

from plotting import (
    plot_existence_wandb,
    plot_grid_with_acceleration_wandb,
    plot_grid_with_trajectory_existence_wandb,
    plot_ground_truth_wandb,
    plot_gt_traj,
    plot_mse_error_wandb,
    plot_pred_traj,
    plot_prediction_with_trajectory_existence_wandb,
    plot_region_weights_wandb,
    plot_trajectory_vs_analytical_wandb,
)

# Registry of individual plot functions
# Each function signature: fn(traj_data, grid_config, logger, epoch, mode) -> None
PLOTS: dict[str, callable] = {
    "ground_truth": plot_ground_truth_wandb,
    "grid_with_trajectory_existence": plot_grid_with_trajectory_existence_wandb,
    "grid_with_acceleration": plot_grid_with_acceleration_wandb,
    "trajectory_vs_analytical": plot_trajectory_vs_analytical_wandb,
    "existence": plot_existence_wandb,
    "prediction_with_trajectory_existence": plot_prediction_with_trajectory_existence_wandb,
    "mse_error": plot_mse_error_wandb,
    "region_weights": plot_region_weights_wandb,
    "gt_traj": plot_gt_traj,
    "pred_traj": plot_pred_traj,
}

# Presets for common configurations
# Each preset lists which plots to generate
PLOT_PRESETS: dict[str, list[str]] = {
    "shock_net": [
        "grid_with_trajectory_existence",  # Uses ground truth grid as background
        "grid_with_acceleration",  # Shows acceleration magnitude (shock locations)
    ],
    "hybrid": [
        "prediction_with_trajectory_existence",  # Uses predicted grid as background
        "mse_error",
        "existence",
        "region_weights",
    ],
    "traj_net": [
        "gt_traj",  # Uses predicted grid as background
        "pred_traj",
        "mse_error",
    ],
}


def plot_wandb(
    traj_data: dict,
    grid_config: dict,
    logger,
    epoch: int | None,
    mode: str = "val",
    preset: str | None = None,
) -> None:
    """Main plotting entry point.

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
                PLOTS[plot_name](traj_data, grid_config, logger, epoch, mode)
            except Exception as e:
                # Log warning but don't fail - some plots may not have required data
                print(f"Warning: Plot '{plot_name}' failed: {e}")
