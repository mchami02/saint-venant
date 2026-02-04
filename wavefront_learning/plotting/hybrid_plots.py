"""HybridDeepONet-specific plotting functions.

This module provides visualization functions specifically for the HybridDeepONet model,
including region assignments and comprehensive hybrid predictions.
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
        data, aspect="auto", origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax
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
        prediction, aspect="auto", origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1
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
        region_weight, aspect="auto", origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=1
    )

    # Overlay trajectories that border this region (d == region_idx or d+1 == region_idx)
    n_disc = int(mask.sum())
    colors = _get_colors(n_disc)
    for d in range(n_disc):
        if d == region_idx or d + 1 == region_idx:
            valid = existence[d] > 0.5
            ax.plot(positions[d, valid], times[valid], "--", color=colors[d], linewidth=2)

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
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    logger,
    num_samples: int = 5,
    mode: str = "test",
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
        nx, nt: Grid dimensions.
        dx, dt: Grid spacing.
        logger: WandbLogger instance.
        num_samples: Maximum number of samples to include.
    """
    if logger is None or not logger.enabled:
        return

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
                region_weights[b, k], positions[b], existence[b], masks[b], times, extent, k
            )
            row.append(wandb.Image(fig_region))
            plt.close(fig_region)

        data.append(row)

    table = wandb.Table(columns=columns, data=data)
    wandb.log({f"{mode}/hybrid_comparison_table": table})


def plot_hybrid_predictions_wandb(
    ground_truths: np.ndarray,
    predictions: np.ndarray,
    positions: np.ndarray,
    existence: np.ndarray,
    discontinuities: np.ndarray,
    masks: np.ndarray,
    region_densities: np.ndarray,
    region_weights: np.ndarray,
    times: np.ndarray,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
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
        ground_truths: Ground truth grids of shape (B, nt, nx).
        predictions: Predicted grids of shape (B, nt, nx).
        positions: Predicted positions of shape (B, D, T).
        existence: Predicted existence of shape (B, D, T).
        discontinuities: Initial discontinuities of shape (B, D, 3).
        masks: Validity masks of shape (B, D).
        region_densities: Per-region predictions of shape (B, K, nt, nx).
        region_weights: Region assignments of shape (B, K, nt, nx).
        times: Query times of shape (T,) or (B, T).
        nx, nt: Grid dimensions.
        dx, dt: Grid spacing.
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
        use_summary: If True, log to summary instead of step-based logging.
    """
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
        nx,
        nt,
        dx,
        dt,
        logger,
        num_samples=5,
        mode=mode,
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
