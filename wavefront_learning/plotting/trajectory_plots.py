"""Shock trajectory plotting functions.

This module provides core functions for visualizing shock trajectories,
existence probabilities, and wavefront detection.

For W&B-specific plotting functions (compatible with plotter.py PLOTS registry),
see wandb_trajectory_plots.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from .base import _get_colors, _get_extent


def _compute_acceleration_numpy(density: np.ndarray, dt: float) -> np.ndarray:
    """Compute temporal acceleration using central finite differences.

    Args:
        density: (B, nt, nx) or (nt, nx) density grid.
        dt: Time step size.

    Returns:
        (B, nt-2, nx) or (nt-2, nx) acceleration for interior time points.
    """
    # Central difference: a = (rho(t+dt) - 2*rho(t) + rho(t-dt)) / dt^2
    if density.ndim == 3:
        return (density[:, 2:, :] - 2 * density[:, 1:-1, :] + density[:, :-2, :]) / (
            dt**2
        )
    else:
        return (density[2:, :] - 2 * density[1:-1, :] + density[:-2, :]) / (dt**2)


def plot_shock_trajectories(
    positions: np.ndarray,
    existence: np.ndarray,
    discontinuities: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    sample_idx: int = 0,
    show_analytical: bool = False,
) -> Figure:
    """Plot predicted shock trajectories vs analytical Rankine-Hugoniot trajectories.

    Args:
        positions: Predicted positions of shape (D, T) for one sample.
        existence: Predicted existence of shape (D, T) for one sample.
        discontinuities: Initial discontinuities of shape (D, 3) with [x_0, rho_L, rho_R].
        mask: Validity mask of shape (D,).
        times: Query times of shape (T,).
        sample_idx: Sample index for title.
        show_analytical: Whether to show analytical RH trajectories.

    Returns:
        Matplotlib figure with trajectory plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    n_disc = int(mask.sum())
    colors = _get_colors(n_disc)

    for d in range(n_disc):
        x_0 = discontinuities[d, 0]
        rho_L = discontinuities[d, 1]
        rho_R = discontinuities[d, 2]

        # Predicted trajectory with existence as alpha
        pred_pos = positions[d]
        pred_exist = existence[d]

        # Plot predicted trajectory with varying alpha based on existence
        for t_idx in range(len(times) - 1):
            alpha = float(pred_exist[t_idx])
            ax.plot(
                [times[t_idx], times[t_idx + 1]],
                [pred_pos[t_idx], pred_pos[t_idx + 1]],
                color=colors[d],
                alpha=max(0.1, alpha),
                linewidth=2,
            )

        # Mark initial position
        ax.scatter([0], [x_0], color=colors[d], s=100, marker="o", zorder=5)

        # Analytical trajectory (Rankine-Hugoniot)
        if show_analytical:
            shock_speed = 1.0 - rho_L - rho_R
            analytical_pos = x_0 + shock_speed * times
            ax.plot(
                times,
                analytical_pos,
                color=colors[d],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"RH d={d} (s={shock_speed:.2f})",
            )

    # Domain boundaries
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax.fill_between(times, -0.1, 0, color="gray", alpha=0.1)
    ax.fill_between(times, 1, 1.1, color="gray", alpha=0.1)

    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Position x", fontsize=12)
    ax.set_title(f"Shock Trajectories (Sample {sample_idx + 1})", fontsize=14)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_existence_heatmap(
    existence: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    sample_idx: int = 0,
) -> Figure:
    """Plot existence probability heatmap for all discontinuities.

    Args:
        existence: Predicted existence of shape (D, T) for one sample.
        mask: Validity mask of shape (D,).
        times: Query times of shape (T,).
        sample_idx: Sample index for title.

    Returns:
        Matplotlib figure with existence heatmap.
    """
    n_disc = int(mask.sum())
    if n_disc == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid discontinuities", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(10, 4))

    # Only show valid discontinuities
    valid_existence = existence[:n_disc]

    im = ax.imshow(
        valid_existence,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        extent=[times[0], times[-1], -0.5, n_disc - 0.5],
    )

    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Discontinuity Index", fontsize=12)
    ax.set_title(f"Existence Probability (Sample {sample_idx + 1})", fontsize=14)
    ax.set_yticks(range(n_disc))
    plt.colorbar(im, ax=ax, label="P(exists)")

    plt.tight_layout()
    return fig


def plot_trajectory_on_grid(
    grid: np.ndarray,
    positions: np.ndarray,
    existence: np.ndarray,
    discontinuities: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    grid_config: dict,
    sample_idx: int = 0,
    show_analytical: bool = False,
) -> Figure:
    """Plot predicted shock trajectories overlaid on solution heatmap.

    Creates a combined plot with:
    - Background: Solution heatmap (nt x nx) with Space on X-axis, Time on Y-axis
    - Foreground: Predicted trajectory lines
    - Dashed lines: Analytical Rankine-Hugoniot trajectories for comparison

    Args:
        grid: Solution grid of shape (nt, nx).
        positions: Predicted positions of shape (D, T) for one sample.
        existence: Predicted existence of shape (D, T) for one sample.
        discontinuities: Initial discontinuities of shape (D, 3) with [x_0, rho_L, rho_R].
        mask: Validity mask of shape (D,).
        times: Query times of shape (T,).
        grid_config: Dict with {nx, nt, dx, dt}.
        sample_idx: Sample index for title.
        show_analytical: Whether to show analytical RH trajectories.

    Returns:
        Matplotlib figure with trajectory overlay on grid.
    """
    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: solution heatmap
    # extent = [x_min, x_max, t_min, t_max] for Space on X, Time on Y
    extent = _get_extent(nx, nt, dx, dt)
    im = ax.imshow(
        grid,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.8,
    )
    plt.colorbar(im, ax=ax, label="Density")

    # Foreground: trajectories
    n_disc = int(mask.sum())
    colors = _get_colors(n_disc)

    for d in range(n_disc):
        x_0 = discontinuities[d, 0]
        rho_L = discontinuities[d, 1]
        rho_R = discontinuities[d, 2]

        # Predicted trajectory with existence as alpha
        pred_pos = positions[d]
        pred_exist = existence[d]

        # Plot predicted trajectory with varying alpha based on existence
        # Note: positions are x-coordinates, times are t-coordinates
        for t_idx in range(len(times) - 1):
            alpha = float(pred_exist[t_idx])
            ax.plot(
                [pred_pos[t_idx], pred_pos[t_idx + 1]],  # X = Space
                [times[t_idx], times[t_idx + 1]],  # Y = Time
                color=colors[d],
                alpha=max(0.3, alpha),
                linewidth=2.5,
            )

        # Mark initial position
        ax.scatter(
            [x_0], [0], color=colors[d], s=100, marker="o", zorder=5, edgecolors="white"
        )

        # Analytical trajectory (Rankine-Hugoniot)
        if show_analytical:
            shock_speed = 1.0 - rho_L - rho_R
            analytical_pos = x_0 + shock_speed * times
            ax.plot(
                analytical_pos,  # X = Space
                times,  # Y = Time
                color=colors[d],
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                label=f"RH d={d} (s={shock_speed:.2f})",
            )

    ax.set_xlabel("Space x", fontsize=12)
    ax.set_ylabel("Time t", fontsize=12)
    ax.set_title(f"Trajectory Overlay (Sample {sample_idx + 1})", fontsize=14)
    ax.set_xlim(0, nx * dx)
    ax.set_ylim(0, nt * dt)
    if n_disc > 0:
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


def plot_wavefront_trajectory(
    prediction: np.ndarray,
    wavefront_positions: np.ndarray | None = None,
    nx: int | None = None,
    nt: int | None = None,
    dx: float | None = None,
    dt: float | None = None,
) -> Figure:
    """Plot wavefront trajectory over time.

    Detects discontinuities and plots their position over time.

    Args:
        prediction: Prediction array of shape (nt, nx).
        wavefront_positions: Optional ground truth wavefront positions.
        nx: Number of spatial points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.

    Returns:
        Matplotlib figure with wavefront trajectory.
    """
    if nt is None:
        nt = prediction.shape[0]
    if nx is None:
        nx = prediction.shape[1]
    if dx is None:
        dx = 1.0 / nx
    if dt is None:
        dt = 1.0 / nt

    fig, ax = plt.subplots(figsize=(8, 6))

    # Detect wavefront positions from prediction (using gradient)
    detected_positions = []
    for t in range(nt):
        grad = np.abs(np.gradient(prediction[t]))
        if grad.max() > 0.1:  # Threshold for detecting discontinuity
            pos = np.argmax(grad) * dx
            detected_positions.append((t * dt, pos))

    if detected_positions:
        times, positions = zip(*detected_positions, strict=True)
        ax.plot(times, positions, "b-", linewidth=2, label="Detected Wavefront")

    if wavefront_positions is not None:
        ax.plot(
            np.arange(nt) * dt,
            wavefront_positions,
            "r--",
            linewidth=2,
            label="True Wavefront",
        )

    ax.set_xlabel("Time t")
    ax.set_ylabel("Position x")
    ax.set_title("Wavefront Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training Progress",
) -> Figure:
    """Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        title: Plot title.

    Returns:
        Matplotlib figure with loss curves.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    return fig


def plot_sample_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 5,
    grid_config: dict | None = None,
) -> list[Figure]:
    """Generate prediction plots for multiple samples.

    Args:
        model: Trained model.
        dataloader: DataLoader to sample from.
        device: Computation device.
        num_samples: Number of samples to plot.
        grid_config: Optional dict with {nx, nt, dx, dt}. If None, inferred from data.

    Returns:
        List of matplotlib figures.
    """
    from .grid_plots import plot_prediction_comparison

    model.eval()
    figures = []
    count = 0

    with torch.no_grad():
        for batch_input, batch_target in dataloader:
            if isinstance(batch_input, dict):
                batch_input = {k: v.to(device) for k, v in batch_input.items()}
            else:
                batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            pred = model(batch_input)

            for i in range(batch_target.shape[0]):
                if count >= num_samples:
                    break

                gt = batch_target[i].squeeze(0).cpu().numpy()
                p = pred[i].squeeze(0).cpu().numpy()

                # Build grid_config if not provided
                if grid_config is not None:
                    _grid_config = grid_config
                else:
                    actual_nt, actual_nx = gt.shape
                    _grid_config = {
                        "nx": actual_nx,
                        "nt": actual_nt,
                        "dx": 1.0 / actual_nx,
                        "dt": 1.0 / actual_nt,
                    }

                fig = plot_prediction_comparison(
                    gt, p, _grid_config, title=f"Sample {count + 1}"
                )
                figures.append(fig)
                count += 1

            if count >= num_samples:
                break

    return figures
