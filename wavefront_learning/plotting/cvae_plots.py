"""CVAE-specific visualization functions.

Provides plots for visualizing multi-sample uncertainty from the
CVAE DeepONet model:
- plot_cvae_samples: Overlay multiple spatial cross-sections from different z-samples
- plot_cvae_uncertainty: Heatmap of pixel-wise std across samples (uncertainty map)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .base import _get_extent, _plot_heatmap


def plot_cvae_samples(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Overlay multiple 1D spatial cross-sections from different z-samples.

    Shows predictions from multiple latent samples at selected time steps,
    with the ground truth as a dashed line.

    Args:
        traj_data: Data dict containing:
            - grids: (num_samples, nt, nx) ground truth
            - cvae_samples: (num_samples, n_z_samples, nt, nx) multi-sample predictions
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    if "cvae_samples" not in traj_data:
        return []

    gt = traj_data["grids"]  # (num_plot_samples, nt, nx)
    samples = traj_data["cvae_samples"]  # (num_plot_samples, n_z_samples, nt, nx)
    dx = grid_config["dx"]

    figures = []
    # Plot first sample in the batch
    gt_i = gt[0]  # (nt, nx)
    samples_i = samples[0]  # (n_z_samples, nt, nx)
    nt, nx = gt_i.shape
    n_z = samples_i.shape[0]

    # Select time steps to visualize
    time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]
    x_vals = np.arange(nx) * dx

    fig, axes = plt.subplots(1, len(time_indices), figsize=(4 * len(time_indices), 3))
    if len(time_indices) == 1:
        axes = [axes]

    for ax, t_idx in zip(axes, time_indices):
        # Plot each z-sample
        for z_idx in range(n_z):
            alpha = 0.4 if n_z > 3 else 0.6
            ax.plot(
                x_vals, samples_i[z_idx, t_idx, :],
                color=f"C{z_idx % 10}", alpha=alpha, linewidth=0.8,
                label=f"z_{z_idx}" if t_idx == time_indices[0] else None,
            )
        # Plot ground truth
        ax.plot(
            x_vals, gt_i[t_idx, :],
            "k--", linewidth=1.5, label="GT" if t_idx == time_indices[0] else None,
        )
        ax.set_title(f"t={t_idx * grid_config['dt']:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")

    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("CVAE Samples at Different Times", fontsize=12)
    fig.tight_layout()
    figures.append(("cvae_samples", fig))

    return figures


def plot_cvae_uncertainty(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Plot pixel-wise std across z-samples as an uncertainty heatmap.

    Args:
        traj_data: Data dict containing:
            - grids: (num_samples, nt, nx) ground truth
            - cvae_samples: (num_samples, n_z_samples, nt, nx) multi-sample predictions
            - cvae_mean: (num_samples, nt, nx) mean prediction
            - cvae_std: (num_samples, nt, nx) std prediction
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    if "cvae_std" not in traj_data:
        return []

    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    extent = _get_extent(nx, nt, dx, dt)
    figures = []

    # Plot first sample
    std_map = traj_data["cvae_std"][0]  # (nt, nx)
    mean_map = traj_data["cvae_mean"][0]  # (nt, nx)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Mean prediction
    _plot_heatmap(
        axes[0], mean_map, extent,
        cmap="viridis", vmin=0, vmax=1, label="Density",
    )
    axes[0].set_title("Mean Prediction")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")

    # Uncertainty (std)
    std_max = max(std_map.max(), 1e-6)
    _plot_heatmap(
        axes[1], std_map, extent,
        cmap="hot", vmin=0, vmax=std_max, label="Std",
    )
    axes[1].set_title("Prediction Uncertainty (Std)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")

    fig.suptitle("CVAE Uncertainty", fontsize=12)
    fig.tight_layout()
    figures.append(("cvae_uncertainty", fig))

    return figures
