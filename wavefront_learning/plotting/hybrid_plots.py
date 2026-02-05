"""HybridDeepONet-specific plotting functions.

This module provides visualization functions specifically for the HybridDeepONet model,
including region assignments and comprehensive hybrid predictions.

Includes plot functions compatible with the PLOTS registry in plotter.py:
- plot_prediction_with_trajectory_wandb: Predicted grid + trajectory overlay
- plot_mse_error_wandb: MSE error heatmap
- plot_region_weights_wandb: Region weight visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.figure import Figure

from .base import _get_colors, _get_extent, _log_figure


def _create_single_heatmap(
    data: np.ndarray,
    extent: tuple,
    cmap: str = "viridis",
    vmin: float = 0,
    vmax: float = 1,
    title: str = "",
) -> Figure:
    """Create a single heatmap figure for W&B table.

    Args:
        data: 2D array (nt, nx) to plot.
        extent: Extent for imshow [x_min, x_max, t_min, t_max].
        cmap: Colormap name.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        title: Optional title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _create_prediction_with_trajectories(
    prediction: np.ndarray,
    positions: np.ndarray,
    existence: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    extent: tuple,
) -> Figure:
    """Create prediction heatmap with trajectory overlays.

    Args:
        prediction: Prediction grid (nt, nx).
        positions: Predicted positions (D, T).
        existence: Predicted existence (D, T).
        mask: Validity mask (D,).
        times: Query times (T,).
        extent: Extent for imshow.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(
        prediction,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    n_disc = int(mask.sum())
    colors = _get_colors(n_disc)
    for d in range(n_disc):
        valid = existence[d] > 0.5
        ax.plot(positions[d, valid], times[valid], "--", color=colors[d], linewidth=2)

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.tight_layout()
    return fig


def _create_region_with_trajectories(
    region_weight: np.ndarray,
    positions: np.ndarray,
    existence: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    extent: tuple,
    region_idx: int,
) -> Figure:
    """Create region weight heatmap with bordering trajectory overlays.

    Args:
        region_weight: Region weight grid (nt, nx).
        positions: Predicted positions (D, T).
        existence: Predicted existence (D, T).
        mask: Validity mask (D,).
        times: Query times (T,).
        extent: Extent for imshow.
        region_idx: Region index.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(
        region_weight,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    # Overlay trajectories that border this region (d == region_idx or d+1 == region_idx)
    n_disc = int(mask.sum())
    colors = _get_colors(n_disc)
    for d in range(n_disc):
        if d == region_idx or d + 1 == region_idx:
            valid = existence[d] > 0.5
            ax.plot(
                positions[d, valid], times[valid], "--", color=colors[d], linewidth=2
            )

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"Region {region_idx}")
    fig.tight_layout()
    return fig


def _create_comparison_table(
    ground_truths: np.ndarray,
    predictions: np.ndarray,
    positions: np.ndarray,
    existence: np.ndarray,
    masks: np.ndarray,
    region_weights: np.ndarray,
    times: np.ndarray,
    grid_config: dict,
    logger,
    num_samples: int = 5,
    mode: str = "test",
    step: int | None = None,
) -> None:
    """Create a W&B table with comparison images.

    Args:
        ground_truths: Ground truth grids (B, nt, nx).
        predictions: Predicted grids (B, nt, nx).
        positions: Predicted positions (B, D, T).
        existence: Predicted existence (B, D, T).
        masks: Validity masks (B, D).
        region_weights: Region assignments (B, K, nt, nx).
        times: Query times (T,).
        grid_config: Dict with {nx, nt, dx, dt}.
        logger: WandbLogger instance.
        num_samples: Maximum number of samples to include.
        mode: Mode string for logging prefix.
        step: Optional epoch number to log as metric (avoids wandb.watch step conflicts).
    """
    if logger is None or not logger.enabled:
        return

    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    B = min(ground_truths.shape[0], num_samples)
    K = region_weights.shape[1]
    extent = _get_extent(nx, nt, dx, dt)

    # Build column headers
    columns = ["Sample", "Ground Truth", "Prediction", "MSE Error"]
    columns += [f"Region {k}" for k in range(K)]

    data = []
    for b in range(B):
        row = [b]

        # Ground truth
        fig_gt = _create_single_heatmap(ground_truths[b], extent, cmap="viridis")
        row.append(wandb.Image(fig_gt))
        plt.close(fig_gt)

        # Prediction with trajectories
        fig_pred = _create_prediction_with_trajectories(
            predictions[b], positions[b], existence[b], masks[b], times, extent
        )
        row.append(wandb.Image(fig_pred))
        plt.close(fig_pred)

        # MSE error
        error = (predictions[b] - ground_truths[b]) ** 2
        fig_err = _create_single_heatmap(error, extent, cmap="hot", vmin=0, vmax=0.5)
        row.append(wandb.Image(fig_err))
        plt.close(fig_err)

        # Region assignments (with trajectory overlays)
        for k in range(K):
            fig_region = _create_region_with_trajectories(
                region_weights[b, k],
                positions[b],
                existence[b],
                masks[b],
                times,
                extent,
                k,
            )
            row.append(wandb.Image(fig_region))
            plt.close(fig_region)

        data.append(row)

    table = wandb.Table(columns=columns, data=data)
    # Don't use explicit step to avoid conflicts with wandb.watch step counter
    # Include epoch as a metric so it can be used as x-axis in charts
    log_dict = {f"{mode}/hybrid_comparison_table": table}
    if step is not None:
        log_dict["plot_epoch"] = step
    wandb.log(log_dict)


def plot_hybrid_predictions_wandb(
    traj_data: dict,
    grid_config: dict,
    logger,
    epoch: int,
    mode: str = "val",
    use_summary: bool = False,
) -> None:
    """Create comprehensive plots for HybridDeepONet and upload to W&B.

    Logs:
        - {mode}/hybrid_summary: B rows x 3 cols (GT, Pred+traj, MSE Error)
        - test/hybrid_comparison_table: W&B table with GT, Pred, MSE, Region columns

    Args:
        traj_data: Dict containing:
            - grids: Ground truth grids of shape (B, nt, nx).
            - output_grid: Predicted grids of shape (B, nt, nx).
            - positions: Predicted positions of shape (B, D, T).
            - existence: Predicted existence of shape (B, D, T).
            - discontinuities: Initial discontinuities of shape (B, D, 3).
            - masks: Validity masks of shape (B, D).
            - region_densities: Per-region predictions of shape (B, K, nt, nx).
            - region_weights: Region assignments of shape (B, K, nt, nx).
            - times: Query times of shape (T,) or (B, T).
        grid_config: Dict with {nx, nt, dx, dt}.
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
        use_summary: If True, log to summary instead of step-based logging.
    """
    # Extract from traj_data
    ground_truths = traj_data["grids"]
    predictions = traj_data["output_grid"]
    positions = traj_data["positions"]
    existence = traj_data["existence"]
    masks = traj_data["masks"]
    region_weights = traj_data["region_weights"]
    times = traj_data["times"]

    # Extract from grid_config
    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )

    B = positions.shape[0]

    # Handle times shape
    if times.ndim == 1:
        times_1d = times
    else:
        times_1d = times[0]

    extent = _get_extent(nx, nt, dx, dt)

    # Create W&B comparison table
    _create_comparison_table(
        ground_truths,
        predictions,
        positions,
        existence,
        masks,
        region_weights,
        times_1d,
        grid_config,
        logger,
        num_samples=5,
        mode=mode,
        step=epoch,
    )

    # Summary comparison plot: 3 columns (GT, Pred+traj, MSE Error)
    fig, axes = plt.subplots(B, 3, figsize=(15, 5 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for b in range(B):
        n_disc = int(masks[b].sum())
        colors = _get_colors(n_disc)

        # Column 0: Ground truth
        im1 = axes[b, 0].imshow(
            ground_truths[b],
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes[b, 0].set_title(f"Sample {b + 1}: Ground Truth")
        axes[b, 0].set_xlabel("Space x")
        axes[b, 0].set_ylabel("Time t")
        plt.colorbar(im1, ax=axes[b, 0])

        # Column 1: Prediction with predicted trajectories only
        im2 = axes[b, 1].imshow(
            predictions[b],
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        # Add predicted trajectories (no analytical)
        for d in range(n_disc):
            valid = existence[b, d] > 0.5
            axes[b, 1].plot(
                positions[b, d, valid],
                times_1d[valid],
                color=colors[d],
                linewidth=2,
                linestyle="--",
            )
        axes[b, 1].set_title(f"Sample {b + 1}: Prediction + Trajectories")
        axes[b, 1].set_xlabel("Space x")
        axes[b, 1].set_ylabel("Time t")
        plt.colorbar(im2, ax=axes[b, 1])

        # Column 2: MSE Error
        error = (predictions[b] - ground_truths[b]) ** 2
        im3 = axes[b, 2].imshow(
            error,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="hot",
            vmin=0,
            vmax=0.5,
        )
        axes[b, 2].set_title(f"Sample {b + 1}: MSE Error (mean={error.mean():.4f})")
        axes[b, 2].set_xlabel("Space x")
        axes[b, 2].set_ylabel("Time t")
        plt.colorbar(im3, ax=axes[b, 2])

    plt.tight_layout()
    _log_figure(logger, f"{mode}/hybrid_summary", fig, epoch, use_summary)
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
