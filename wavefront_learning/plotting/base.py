"""Base plotting utilities and helpers.

This module provides common setup and helper functions used across all plotting modules.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend (thread-safe, no GUI)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure


def save_figure(fig: Figure, path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to file.

    Args:
        fig: Matplotlib figure to save.
        path: Output file path.
        dpi: Resolution in dots per inch.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _get_extent(nx: int, nt: int, dx: float, dt: float) -> list[float]:
    """Compute extent for imshow from grid parameters.

    Args:
        nx: Number of spatial points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.

    Returns:
        Extent list [x_min, x_max, t_min, t_max].
    """
    return [0, nx * dx, 0, nt * dt]


def _get_colors(n: int) -> np.ndarray:
    """Get a colormap array for n items.

    Args:
        n: Number of colors needed.

    Returns:
        Array of colors from tab10 colormap.
    """
    return plt.cm.tab10(np.linspace(0, 1, max(n, 1)))


def _plot_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    extent: list[float],
    cmap: str = "viridis",
    vmin: float = 0,
    vmax: float = 1,
    label: str = "Value",
    alpha: float = 1.0,
) -> None:
    """Plot a heatmap with colorbar on the given axes.

    Args:
        ax: Matplotlib axes to plot on.
        data: 2D array to plot.
        extent: Extent for imshow [x_min, x_max, y_min, y_max].
        cmap: Colormap name.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        label: Colorbar label.
        alpha: Transparency.
    """
    im = ax.imshow(
        data,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
    )
    plt.colorbar(im, ax=ax, label=label)


def _create_comparison_animation(
    gt: np.ndarray,
    pred: np.ndarray,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    sample_idx: int = 0,
    skip_frames: int = 5,
    fps: int = 20,
):
    """Create an animation showing ground truth vs prediction side by side through time.

    Args:
        gt: Ground truth array (nt, nx).
        pred: Prediction array (nt, nx).
        nx, nt: Grid dimensions.
        dx, dt: Grid spacing.
        sample_idx: Sample index for title.
        skip_frames: Number of frames to skip.
        fps: Frames per second.

    Returns:
        FuncAnimation object and the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.linspace(0, nx * dx, gt.shape[1])

    # Set consistent y-axis limits
    y_min = min(gt.min(), pred.min())
    y_max = max(gt.max(), pred.max())
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1

    # Initialize lines
    (line_gt,) = axes[0].plot(x, gt[0], "b-", linewidth=2)
    (line_pred,) = axes[1].plot(x, pred[0], "r-", linewidth=2)

    for ax, title in zip(
        axes,
        [
            f"Ground Truth (Sample {sample_idx + 1})",
            f"Prediction (Sample {sample_idx + 1})",
        ],
        strict=True,
    ):
        ax.set_xlim(0, nx * dx)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel("Position x")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    time_text = axes[0].text(0.02, 0.95, "", transform=axes[0].transAxes, fontsize=10)
    plt.tight_layout()

    def update(frame):
        t = frame * skip_frames
        if t >= nt:
            t = nt - 1
        line_gt.set_ydata(gt[t])
        line_pred.set_ydata(pred[t])
        time_text.set_text(f"Time: {t * dt:.3f} s (step {t}/{nt})")
        return line_gt, line_pred, time_text

    n_frames = max(1, nt // skip_frames)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    return anim, fig


def _log_figure(
    logger,
    key: str,
    fig: Figure,
    epoch: int,
    use_summary: bool = False,
) -> None:
    """Log a figure to W&B with unified logic.

    Args:
        logger: WandbLogger instance (can be None).
        key: Logging key/name.
        fig: Matplotlib figure to log.
        epoch: Current epoch.
        use_summary: If True, log to summary instead of step-based logging.
    """
    if logger is None:
        return
    if use_summary:
        logger.log_summary_image(key, fig)
    else:
        logger.log_figure(key, fig, step=epoch)
