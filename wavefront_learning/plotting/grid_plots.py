"""Grid comparison plotting functions.

This module provides functions for comparing ground truth vs prediction grids,
error maps, and W&B logging of grid visualizations.

Includes plot functions compatible with the PLOTS registry in plotter.py:
- plot_ground_truth_wandb: Ground truth grid heatmaps
"""

from __future__ import annotations

import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .base import (
    _create_comparison_animation,
    _get_colors,
    _get_extent,
    _log_figure,
)

try:
    from logger import WandbLogger
except ImportError:
    WandbLogger = None


def plot_prediction_comparison(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    grid_config: dict,
    title: str | None = None,
) -> Figure:
    """Plot ground truth vs prediction comparison as heatmaps.

    Args:
        ground_truth: Ground truth array of shape (nt, nx).
        prediction: Prediction array of shape (nt, nx).
        grid_config: Dict with {nx, nt, dx, dt}.
        title: Optional plot title.

    Returns:
        Matplotlib figure with comparison plot.
    """
    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    difference = np.abs(prediction - ground_truth)
    extent = _get_extent(nx, nt, dx, dt)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground Truth
    im1 = axes[0].imshow(
        ground_truth,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[0].set_xlabel("Space x")
    axes[0].set_ylabel("Time t")
    axes[0].set_title("Ground Truth")
    plt.colorbar(im1, ax=axes[0], label="Value")

    # Prediction
    im2 = axes[1].imshow(
        prediction,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1].set_xlabel("Space x")
    axes[1].set_ylabel("Time t")
    axes[1].set_title("Prediction")
    plt.colorbar(im2, ax=axes[1], label="Value")

    # Difference
    im3 = axes[2].imshow(
        difference,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
    )
    axes[2].set_xlabel("Space x")
    axes[2].set_ylabel("Time t")
    axes[2].set_title("Absolute Error")
    plt.colorbar(im3, ax=axes[2], label="Error")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_error_map(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    grid_config: dict,
) -> Figure:
    """Plot spatial-temporal error map.

    Args:
        ground_truth: Ground truth array of shape (nt, nx).
        prediction: Prediction array of shape (nt, nx).
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        Matplotlib figure with error heatmap.
    """
    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    error = np.abs(prediction - ground_truth)
    extent = _get_extent(nx, nt, dx, dt)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(error, extent=extent, aspect="auto", origin="lower", cmap="hot")
    ax.set_xlabel("Space x")
    ax.set_ylabel("Time t")
    ax.set_title("Prediction Error Heatmap")
    plt.colorbar(im, ax=ax, label="Absolute Error")

    plt.tight_layout()
    return fig


def plot_comparison_wandb(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    grid_config: dict,
    logger,
    epoch: int,
    mode: str = "val",
    use_summary: bool = False,
) -> None:
    """Create comparison plots and upload to W&B.

    Args:
        ground_truth: (B, nt, nx) or (nt, nx) array.
        prediction: (B, nt, nx) or (nt, nx) array.
        grid_config: Dict with {nx, nt, dx, dt}.
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
        use_summary: If True, log to summary instead of step-based logging.
    """
    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)

    # Handle 2D input by adding batch dimension
    if ground_truth.ndim == 2:
        ground_truth = ground_truth[np.newaxis, ...]
        prediction = prediction[np.newaxis, ...]

    B = ground_truth.shape[0]
    difference = np.abs(prediction - ground_truth)

    # Create static comparison plots
    fig, axes = plt.subplots(B, 3, figsize=(18, 5 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    extent = _get_extent(nx, nt, dx, dt)

    for b in range(B):
        # Ground Truth
        im1 = axes[b, 0].imshow(
            ground_truth[b],
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="jet",
            vmin=0,
            vmax=1,
        )
        axes[b, 0].set_xlabel("Space x")
        axes[b, 0].set_ylabel("Time t")
        axes[b, 0].set_title(f"Ground Truth (Sample {b + 1})")
        plt.colorbar(im1, ax=axes[b, 0], label="Value")

        # Prediction
        im2 = axes[b, 1].imshow(
            prediction[b],
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="jet",
            vmin=0,
            vmax=1,
        )
        axes[b, 1].set_xlabel("Space x")
        axes[b, 1].set_ylabel("Time t")
        axes[b, 1].set_title(f"Prediction (Sample {b + 1})")
        plt.colorbar(im2, ax=axes[b, 1], label="Value")

        # Difference
        im3 = axes[b, 2].imshow(
            difference[b],
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
        )
        axes[b, 2].set_xlabel("Space x")
        axes[b, 2].set_ylabel("Time t")
        axes[b, 2].set_title(f"Difference (Sample {b + 1})")
        plt.colorbar(im3, ax=axes[b, 2], label="Error")

    plt.tight_layout()
    _log_figure(logger, f"{mode}/comparison_plot", fig, epoch, use_summary)
    plt.close(fig)

    # Create and upload animated GIFs for each sample
    for b in range(B):
        anim, anim_fig = _create_comparison_animation(
            ground_truth[b], prediction[b], nx, nt, dx, dt, sample_idx=b
        )

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            anim.save(tmp.name, writer="pillow", fps=20)
            if use_summary:
                logger.log_summary_video(
                    f"{mode}/animation_sample_{b + 1}", tmp.name, fps=20
                )
            else:
                logger.log_video(
                    tmp.name, f"{mode}/animation_sample_{b + 1}", step=epoch
                )

        plt.close(anim_fig)


def plot_grid_comparison(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    positions: np.ndarray,
    existence: np.ndarray,
    times: np.ndarray,
    grid_config: dict,
    sample_idx: int = 0,
) -> Figure:
    """Plot ground truth vs prediction grid with trajectory overlay.

    Args:
        ground_truth: Ground truth grid of shape (nt, nx).
        prediction: Predicted grid of shape (nt, nx).
        positions: Predicted positions of shape (D, T).
        existence: Predicted existence of shape (D, T).
        times: Query times of shape (T,).
        grid_config: Dict with {nx, nt, dx, dt}.
        sample_idx: Sample index for title.

    Returns:
        Matplotlib figure with comparison plots.
    """
    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    difference = np.abs(prediction - ground_truth)
    extent = _get_extent(nx, nt, dx, dt)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground Truth
    im1 = axes[0].imshow(
        ground_truth,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[0].set_xlabel("Space x")
    axes[0].set_ylabel("Time t")
    axes[0].set_title("Ground Truth")
    plt.colorbar(im1, ax=axes[0], label="Density")

    # Prediction with trajectory overlay
    im2 = axes[1].imshow(
        prediction,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    # Overlay trajectories
    D = positions.shape[0]
    colors = _get_colors(D)
    for d in range(D):
        axes[1].plot(
            positions[d],
            times,
            color=colors[d],
            linewidth=2,
            alpha=0.8,
            linestyle="--",
        )
    axes[1].set_xlabel("Space x")
    axes[1].set_ylabel("Time t")
    axes[1].set_title("Prediction + Trajectories")
    plt.colorbar(im2, ax=axes[1], label="Density")

    # Difference
    im3 = axes[2].imshow(
        difference,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="hot",
        vmin=0,
        vmax=0.5,
    )
    axes[2].set_xlabel("Space x")
    axes[2].set_ylabel("Time t")
    axes[2].set_title("Absolute Error")
    plt.colorbar(im3, ax=axes[2], label="Error")

    fig.suptitle(f"Grid Comparison (Sample {sample_idx + 1})", fontsize=14)
    plt.tight_layout()
    return fig


def plot_ground_truth_wandb(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Plot ground truth grid heatmaps.

    Args:
        traj_data: Dict containing 'grids' of shape (B, nt, nx).
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
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
    figures = []
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
        figures.append((f"ground_truth_sample_{b + 1}", fig))

    return figures
