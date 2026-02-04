"""Plotting factory for wavefront learning.

This module provides:
- PLOTS: Registry of available plot functions
- PLOT_PRESETS: Pre-configured plot combinations for common use cases
- plot_wandb(): Main entry point that runs plots based on preset

Each plot function signature:
    fn(traj_data, grid_config, logger, epoch, mode, **kwargs) -> None
"""

import matplotlib.pyplot as plt
from plotting.base import _get_colors, _get_extent, _log_figure


def plot_ground_truth_wandb(
    traj_data: dict,
    grid_config: dict,
    logger,
    epoch: int | None,
    mode: str = "val",
) -> None:
    """Plot ground truth grid heatmaps.

    Args:
        traj_data: Dict containing 'grids' of shape (B, nt, nx).
        grid_config: Dict with {nx, nt, dx, dt}.
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
    """
    grids = traj_data["grids"]
    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    extent = _get_extent(nx, nt, dx, dt)

    B = grids.shape[0]
    for b in range(min(B, 3)):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            grids[b],
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        ax.set_title(f"Ground Truth (Sample {b + 1})")
        plt.colorbar(im, ax=ax, label="Density")
        plt.tight_layout()
        _log_figure(logger, f"{mode}/ground_truth_sample_{b + 1}", fig, epoch)
        plt.close(fig)


def plot_grid_with_trajectory_wandb(
    traj_data: dict,
    grid_config: dict,
    logger,
    epoch: int | None,
    mode: str = "val",
) -> None:
    """Plot ground truth grid with predicted trajectory overlay.

    Used for ShockNet where we want to show trajectories on GT grid.

    Args:
        traj_data: Dict containing grids, positions, existence, masks, times.
        grid_config: Dict with {nx, nt, dx, dt}.
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
    """
    grids = traj_data["grids"]
    positions = traj_data["positions"]
    existence = traj_data["existence"]
    masks = traj_data["masks"]
    times = traj_data["times"]

    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    extent = _get_extent(nx, nt, dx, dt)

    if times.ndim > 1:
        times = times[0]

    B = grids.shape[0]
    for b in range(min(B, 3)):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Background: ground truth heatmap
        im = ax.imshow(
            grids[b],
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
            alpha=0.8,
        )
        plt.colorbar(im, ax=ax, label="Density")

        # Overlay: predicted trajectories
        n_disc = int(masks[b].sum())
        colors = _get_colors(n_disc)

        for d in range(n_disc):
            pred_pos = positions[b, d]
            pred_exist = existence[b, d]

            # Plot with varying alpha based on existence
            for t_idx in range(len(times) - 1):
                alpha = float(pred_exist[t_idx])
                ax.plot(
                    [pred_pos[t_idx], pred_pos[t_idx + 1]],  # X = Space
                    [times[t_idx], times[t_idx + 1]],  # Y = Time
                    color=colors[d],
                    alpha=max(0.3, alpha),
                    linewidth=2.5,
                )

        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        ax.set_title(f"GT Grid + Predicted Trajectories (Sample {b + 1})")
        ax.set_xlim(0, nx * dx)
        ax.set_ylim(0, nt * dt)
        plt.tight_layout()
        _log_figure(logger, f"{mode}/grid_with_trajectory_sample_{b + 1}", fig, epoch)
        plt.close(fig)


def plot_trajectory_vs_analytical_wandb(
    traj_data: dict,
    grid_config: dict,  # noqa: ARG001
    logger,
    epoch: int | None,
    mode: str = "val",
) -> None:
    """Plot predicted trajectories vs analytical Rankine-Hugoniot trajectories.

    Args:
        traj_data: Dict containing positions, existence, discontinuities, masks, times.
        grid_config: Dict with {nx, nt, dx, dt} (unused but kept for API consistency).
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
    """
    positions = traj_data["positions"]
    discontinuities = traj_data["discontinuities"]
    masks = traj_data["masks"]
    times = traj_data["times"]

    if times.ndim > 1:
        times = times[0]

    B = positions.shape[0]
    for b in range(min(B, 3)):
        fig, ax = plt.subplots(figsize=(10, 8))

        n_disc = int(masks[b].sum())
        colors = _get_colors(n_disc)

        for d in range(n_disc):
            x_0 = discontinuities[b, d, 0]
            rho_L = discontinuities[b, d, 1]
            rho_R = discontinuities[b, d, 2]

            # Predicted trajectory
            ax.plot(
                times,
                positions[b, d],
                color=colors[d],
                linewidth=2,
                label=f"Pred d={d}",
            )

            # Analytical RH trajectory
            shock_speed = 1.0 - rho_L - rho_R
            analytical = x_0 + shock_speed * times
            ax.plot(
                times,
                analytical,
                color=colors[d],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"RH d={d} (s={shock_speed:.2f})",
            )

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
        ax.fill_between(times, -0.1, 0, color="gray", alpha=0.1)
        ax.fill_between(times, 1, 1.1, color="gray", alpha=0.1)
        ax.set_xlabel("Time t")
        ax.set_ylabel("Position x")
        ax.set_title(f"Predicted vs RH Trajectories (Sample {b + 1})")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _log_figure(
            logger, f"{mode}/trajectory_vs_analytical_sample_{b + 1}", fig, epoch
        )
        plt.close(fig)


def plot_existence_wandb(
    traj_data: dict,
    grid_config: dict,  # noqa: ARG001
    logger,
    epoch: int | None,
    mode: str = "val",
) -> None:
    """Plot existence probability heatmap for all discontinuities.

    Args:
        traj_data: Dict containing existence, masks, times.
        grid_config: Dict with {nx, nt, dx, dt} (unused but kept for API consistency).
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
    """
    existence = traj_data["existence"]
    masks = traj_data["masks"]
    times = traj_data["times"]

    if times.ndim > 1:
        times = times[0]

    B = existence.shape[0]
    for b in range(min(B, 3)):
        n_disc = int(masks[b].sum())
        if n_disc == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        valid_existence = existence[b, :n_disc]

        im = ax.imshow(
            valid_existence,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            extent=[times[0], times[-1], -0.5, n_disc - 0.5],
        )

        ax.set_xlabel("Time t")
        ax.set_ylabel("Discontinuity Index")
        ax.set_title(f"Existence Probability (Sample {b + 1})")
        ax.set_yticks(range(n_disc))
        plt.colorbar(im, ax=ax, label="P(exists)")
        plt.tight_layout()
        _log_figure(logger, f"{mode}/existence_sample_{b + 1}", fig, epoch)
        plt.close(fig)


def plot_prediction_with_trajectory_wandb(
    traj_data: dict,
    grid_config: dict,
    logger,
    epoch: int | None,
    mode: str = "val",
) -> None:
    """Plot predicted grid with trajectory overlay.

    Used for HybridDeepONet where we want to show trajectories on predicted grid.

    Args:
        traj_data: Dict containing output_grid, positions, existence, masks, times.
        grid_config: Dict with {nx, nt, dx, dt}.
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
    """
    if "output_grid" not in traj_data:
        return  # Skip if not a hybrid model

    output_grid = traj_data["output_grid"]
    positions = traj_data["positions"]
    existence = traj_data["existence"]
    masks = traj_data["masks"]
    times = traj_data["times"]

    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    extent = _get_extent(nx, nt, dx, dt)

    if times.ndim > 1:
        times = times[0]

    B = output_grid.shape[0]
    for b in range(min(B, 3)):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Background: predicted grid heatmap
        im = ax.imshow(
            output_grid[b],
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
            alpha=0.8,
        )
        plt.colorbar(im, ax=ax, label="Density")

        # Overlay: predicted trajectories
        n_disc = int(masks[b].sum())
        colors = _get_colors(n_disc)

        for d in range(n_disc):
            valid = existence[b, d] > 0.5
            ax.plot(
                positions[b, d, valid],  # X = Space
                times[valid],  # Y = Time
                color=colors[d],
                linewidth=2,
                linestyle="--",
            )

        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        ax.set_title(f"Predicted Grid + Trajectories (Sample {b + 1})")
        ax.set_xlim(0, nx * dx)
        ax.set_ylim(0, nt * dt)
        plt.tight_layout()
        _log_figure(
            logger, f"{mode}/prediction_with_trajectory_sample_{b + 1}", fig, epoch
        )
        plt.close(fig)


def plot_mse_error_wandb(
    traj_data: dict,
    grid_config: dict,
    logger,
    epoch: int | None,
    mode: str = "val",
) -> None:
    """Plot MSE error heatmap between prediction and ground truth.

    Args:
        traj_data: Dict containing output_grid and grids.
        grid_config: Dict with {nx, nt, dx, dt}.
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
    """
    if "output_grid" not in traj_data:
        return  # Skip if not a hybrid model

    output_grid = traj_data["output_grid"]
    grids = traj_data["grids"]

    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    extent = _get_extent(nx, nt, dx, dt)

    B = output_grid.shape[0]
    for b in range(min(B, 3)):
        error = (output_grid[b] - grids[b]) ** 2

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            error,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="hot",
            vmin=0,
            vmax=0.5,
        )
        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        ax.set_title(f"MSE Error (Sample {b + 1}, mean={error.mean():.4f})")
        plt.colorbar(im, ax=ax, label="Squared Error")
        plt.tight_layout()
        _log_figure(logger, f"{mode}/mse_error_sample_{b + 1}", fig, epoch)
        plt.close(fig)


def plot_region_weights_wandb(
    traj_data: dict,
    grid_config: dict,
    logger,
    epoch: int | None,
    mode: str = "val",
) -> None:
    """Plot region weight visualization for HybridDeepONet.

    Args:
        traj_data: Dict containing region_weights, positions, existence, masks, times.
        grid_config: Dict with {nx, nt, dx, dt}.
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
    """
    if "region_weights" not in traj_data:
        return  # Skip if not a hybrid model

    region_weights = traj_data["region_weights"]
    positions = traj_data["positions"]
    existence = traj_data["existence"]
    masks = traj_data["masks"]
    times = traj_data["times"]

    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    extent = _get_extent(nx, nt, dx, dt)

    if times.ndim > 1:
        times = times[0]

    B = region_weights.shape[0]
    K = region_weights.shape[1]

    for b in range(min(B, 2)):  # Limit to 2 samples for region plots
        n_disc = int(masks[b].sum())
        colors = _get_colors(n_disc)

        # Create subplot grid for regions
        fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
        if K == 1:
            axes = [axes]

        for k in range(K):
            ax = axes[k]
            ax.imshow(
                region_weights[b, k],
                extent=extent,
                aspect="auto",
                origin="lower",
                cmap="Blues",
                vmin=0,
                vmax=1,
            )

            # Overlay trajectories that border this region
            for d in range(n_disc):
                if d == k or d + 1 == k:
                    valid = existence[b, d] > 0.5
                    ax.plot(
                        positions[b, d, valid],
                        times[valid],
                        "--",
                        color=colors[d],
                        linewidth=2,
                    )

            ax.set_xlabel("x")
            ax.set_ylabel("t")
            ax.set_title(f"Region {k}")

        plt.suptitle(f"Region Weights (Sample {b + 1})")
        plt.tight_layout()
        _log_figure(logger, f"{mode}/region_weights_sample_{b + 1}", fig, epoch)
        plt.close(fig)


# Registry of individual plot functions
# Each function signature: fn(traj_data, grid_config, logger, epoch, mode) -> None
PLOTS: dict[str, callable] = {
    "ground_truth": plot_ground_truth_wandb,
    "grid_with_trajectory": plot_grid_with_trajectory_wandb,
    "trajectory_vs_analytical": plot_trajectory_vs_analytical_wandb,
    "existence": plot_existence_wandb,
    "prediction_with_trajectory": plot_prediction_with_trajectory_wandb,
    "mse_error": plot_mse_error_wandb,
    "region_weights": plot_region_weights_wandb,
}

# Presets for common configurations
# Each preset lists which plots to generate
PLOT_PRESETS: dict[str, list[str]] = {
    "shock_net": [
        "ground_truth",
        "grid_with_trajectory",  # Uses ground truth grid as background
        "trajectory_vs_analytical",
        "existence",
    ],
    "hybrid": [
        "ground_truth",
        "prediction_with_trajectory",  # Uses predicted grid as background
        "mse_error",
        "trajectory_vs_analytical",
        "existence",
        "region_weights",
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
