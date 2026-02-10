"""Trajectory plotting functions for the PLOTS registry.

This module provides trajectory plotting functions compatible with the
PLOTS registry in plotter.py.

Plot functions compatible with the PLOTS registry:
- plot_grid_with_trajectory_existence: GT grid with predicted trajectory overlay
- plot_trajectory_vs_analytical: Predicted vs RH trajectories
- plot_existence: Existence probability heatmap
- plot_grid_with_acceleration: Grid + trajectory alongside acceleration grid
- plot_gt_traj: GT grid + trajectories (no existence modulation)

All functions return list[tuple[str, Figure]] of (log_key, figure) pairs.
Logging is handled by plot() in plotter.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .base import _get_colors, _get_extent
from .trajectory_plots import (
    _compute_acceleration_numpy,
    plot_existence_heatmap,
    plot_shock_trajectories,
    plot_trajectory_on_grid,
)


def plot_trajectory_on_grid_multi(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Create trajectory-on-grid overlay plots.

    Args:
        traj_data: Dict containing:
            - grids: Solution grids of shape (B, nt, nx).
            - positions: Predicted positions of shape (B, D, T).
            - existence: Predicted existence of shape (B, D, T).
            - discontinuities: Initial discontinuities of shape (B, D, 3).
            - masks: Validity masks of shape (B, D).
            - times: Query times of shape (T,) or (B, T).
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    # Extract from traj_data
    grids = traj_data["grids"]
    positions = traj_data["positions"]
    existence = traj_data["existence"]
    discontinuities = traj_data["discontinuities"]
    masks = traj_data["masks"]
    times = traj_data["times"]

    B = positions.shape[0]

    # Handle times shape
    if times.ndim == 1:
        times_1d = times
    else:
        times_1d = times[0]

    figures = []
    for b in range(min(B, 3)):
        fig = plot_trajectory_on_grid(
            grids[b],
            positions[b],
            existence[b],
            discontinuities[b],
            masks[b],
            times_1d,
            grid_config,
            sample_idx=b,
            show_analytical=False,
        )
        figures.append((f"trajectory_on_grid_sample_{b + 1}", fig))

    return figures


def plot_trajectory(
    traj_data: dict,
    grid_config: dict,  # noqa: ARG001
) -> list[tuple[str, Figure]]:
    """Create trajectory plots.

    Args:
        traj_data: Dict containing:
            - positions: Predicted positions of shape (B, D, T).
            - existence: Predicted existence of shape (B, D, T).
            - discontinuities: Initial discontinuities of shape (B, D, 3).
            - masks: Validity masks of shape (B, D).
            - times: Query times of shape (T,) or (B, T).
        grid_config: Dict with {nx, nt, dx, dt} (unused but kept for API consistency).

    Returns:
        List of (log_key, figure) pairs.
    """
    # Extract from traj_data
    positions = traj_data["positions"]
    existence = traj_data["existence"]
    discontinuities = traj_data["discontinuities"]
    masks = traj_data["masks"]
    times = traj_data["times"]

    B = positions.shape[0]

    # Handle times shape
    if times.ndim == 1:
        times_1d = times
    else:
        times_1d = times[0]  # Assume same times for all samples

    figures = []

    # Create trajectory plots for each sample
    for b in range(min(B, 3)):  # Limit to 3 samples
        # Trajectory plot
        fig_traj = plot_shock_trajectories(
            positions[b],
            existence[b],
            discontinuities[b],
            masks[b],
            times_1d,
            sample_idx=b,
            show_analytical=False,
        )
        figures.append((f"trajectory_sample_{b + 1}", fig_traj))

        # Existence heatmap
        fig_exist = plot_existence_heatmap(
            existence[b],
            masks[b],
            times_1d,
            sample_idx=b,
        )
        figures.append((f"existence_sample_{b + 1}", fig_exist))

    # Create combined summary plot
    fig, axes = plt.subplots(B, 2, figsize=(14, 5 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for b in range(B):
        n_disc = int(masks[b].sum())
        colors = _get_colors(n_disc)

        # Left: trajectories
        ax = axes[b, 0]
        for d in range(n_disc):
            x_0 = discontinuities[b, d, 0]
            rho_L = discontinuities[b, d, 1]
            rho_R = discontinuities[b, d, 2]

            # Predicted
            ax.plot(
                times_1d,
                positions[b, d],
                color=colors[d],
                linewidth=2,
                label=f"Pred d={d}",
            )

            # Analytical
            shock_speed = 1.0 - rho_L - rho_R
            analytical = x_0 + shock_speed * times_1d
            ax.plot(
                times_1d,
                analytical,
                color=colors[d],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Time t")
        ax.set_ylabel("Position x")
        ax.set_title(f"Sample {b + 1}: Trajectories (solid=pred, dashed=RH)")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: existence
        ax = axes[b, 1]
        if n_disc > 0:
            im = ax.imshow(
                existence[b, :n_disc],
                aspect="auto",
                origin="lower",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                extent=[times_1d[0], times_1d[-1], -0.5, n_disc - 0.5],
            )
            plt.colorbar(im, ax=ax, label="P(exists)")
        ax.set_xlabel("Time t")
        ax.set_ylabel("Discontinuity")
        ax.set_title(f"Sample {b + 1}: Existence Probability")

    plt.tight_layout()
    figures.append(("trajectory_summary", fig))

    return figures


def plot_grid_with_trajectory_existence(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Plot ground truth grid with predicted trajectory overlay.

    Used for ShockNet where we want to show trajectories on GT grid.

    Args:
        traj_data: Dict containing grids, positions, existence, masks, times.
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
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
    figures = []
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
        figures.append((f"grid_with_trajectory_sample_{b + 1}", fig))

    return figures


def plot_trajectory_vs_analytical(
    traj_data: dict,
    grid_config: dict,  # noqa: ARG001
) -> list[tuple[str, Figure]]:
    """Plot predicted trajectories vs analytical Rankine-Hugoniot trajectories.

    Args:
        traj_data: Dict containing positions, existence, discontinuities, masks, times.
        grid_config: Dict with {nx, nt, dx, dt} (unused but kept for API consistency).

    Returns:
        List of (log_key, figure) pairs.
    """
    positions = traj_data["positions"]
    discontinuities = traj_data["discontinuities"]
    masks = traj_data["masks"]
    times = traj_data["times"]

    if times.ndim > 1:
        times = times[0]

    B = positions.shape[0]
    figures = []
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
        figures.append((f"trajectory_vs_analytical_sample_{b + 1}", fig))

    return figures


def plot_existence(
    traj_data: dict,
    grid_config: dict,  # noqa: ARG001
) -> list[tuple[str, Figure]]:
    """Plot existence probability heatmap for all discontinuities.

    Args:
        traj_data: Dict containing existence, masks, times.
        grid_config: Dict with {nx, nt, dx, dt} (unused but kept for API consistency).

    Returns:
        List of (log_key, figure) pairs.
    """
    existence = traj_data["existence"]
    masks = traj_data["masks"]
    times = traj_data["times"]

    if times.ndim > 1:
        times = times[0]

    B = existence.shape[0]
    figures = []
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
        figures.append((f"existence_sample_{b + 1}", fig))

    return figures


def plot_gt_traj(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Plot ground truth grid with predicted trajectory overlay (no existence).

    Like plot_grid_with_trajectory_existence but plots full trajectories
    without using existence probabilities to modulate alpha.

    Args:
        traj_data: Dict containing grids, positions, masks, times.
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    grids = traj_data["grids"]
    positions = traj_data["positions"]
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
    figures = []
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
        has_existence = "existence" in traj_data

        for d in range(n_disc):
            if has_existence:
                exist = traj_data["existence"][b, d]
                for t_idx in range(len(times) - 1):
                    if float(exist[t_idx]) >= 0.5:
                        ax.plot(
                            [positions[b, d, t_idx], positions[b, d, t_idx + 1]],
                            [times[t_idx], times[t_idx + 1]],
                            color=colors[d],
                            linewidth=2.5,
                        )
            else:
                ax.plot(
                    positions[b, d],
                    times,
                    color=colors[d],
                    linewidth=2.5,
                )

        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        ax.set_title(f"GT Grid + Predicted Trajectories (Sample {b + 1})")
        ax.set_xlim(0, nx * dx)
        ax.set_ylim(0, nt * dt)
        plt.tight_layout()
        figures.append((f"gt_traj_sample_{b + 1}", fig))

    return figures


def plot_grid_with_acceleration(
    traj_data: dict,
    grid_config: dict,
    accel_threshold: float = 1.0,
) -> list[tuple[str, Figure]]:
    """Plot grid+trajectory alongside acceleration grid.

    Creates a two-column figure:
    - Left: Ground truth grid with predicted trajectory overlay
    - Right: Acceleration magnitude grid showing shock locations

    Note: Acceleration is computed via central difference and has shape (nt-2, nx),
    so it covers the interior time range [dt, (nt-1)*dt].

    Args:
        traj_data: Dict with grids, positions, existence, masks, times.
        grid_config: Dict with {nx, nt, dx, dt}.
        accel_threshold: Threshold for highlighting high acceleration regions.

    Returns:
        List of (log_key, figure) pairs.
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

    # Extent for full grid: [x_min, x_max, t_min, t_max]
    extent_grid = _get_extent(nx, nt, dx, dt)

    # Extent for acceleration (interior times only): [x_min, x_max, dt, (nt-1)*dt]
    extent_accel = [0, nx * dx, dt, (nt - 1) * dt]

    if times.ndim > 1:
        times = times[0]

    B = grids.shape[0]
    figures = []
    for b in range(min(B, 3)):
        # Compute acceleration for this sample
        accel = _compute_acceleration_numpy(grids[b], dt)  # (nt-2, nx)
        accel_mag = np.abs(accel)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Ground truth grid with trajectory overlay
        ax_grid = axes[0]
        im_grid = ax_grid.imshow(
            grids[b],
            extent=extent_grid,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
            alpha=0.8,
        )
        plt.colorbar(im_grid, ax=ax_grid, label="Density")

        # Overlay predicted trajectories
        n_disc = int(masks[b].sum())
        colors = _get_colors(n_disc)

        for d in range(n_disc):
            pred_pos = positions[b, d]
            pred_exist = existence[b, d]

            # Plot with varying alpha based on existence
            for t_idx in range(len(times) - 1):
                alpha = float(pred_exist[t_idx])
                ax_grid.plot(
                    [pred_pos[t_idx], pred_pos[t_idx + 1]],  # X = Space
                    [times[t_idx], times[t_idx + 1]],  # Y = Time
                    color=colors[d],
                    alpha=max(0.3, alpha),
                    linewidth=2.5,
                )

        ax_grid.set_xlabel("Space x")
        ax_grid.set_ylabel("Time t")
        ax_grid.set_title(f"GT Grid + Trajectories (Sample {b + 1})")
        ax_grid.set_xlim(0, nx * dx)
        ax_grid.set_ylim(0, nt * dt)

        # Right: Acceleration magnitude
        ax_accel = axes[1]
        # Use symmetric log scale for better visualization
        vmax = max(accel_mag.max(), accel_threshold * 2)
        im_accel = ax_accel.imshow(
            accel_mag,
            extent=extent_accel,
            aspect="auto",
            origin="lower",
            cmap="hot",
            vmin=0,
            vmax=vmax,
        )
        plt.colorbar(im_accel, ax=ax_accel, label="|Acceleration|")

        # Add contour at threshold level if there are values above threshold
        if accel_mag.max() > accel_threshold:
            # Create coordinate grids for contour
            x_centers = np.linspace(dx / 2, nx * dx - dx / 2, nx)
            t_centers = np.linspace(dt + dt / 2, (nt - 1) * dt - dt / 2, nt - 2)
            ax_accel.contour(
                x_centers,
                t_centers,
                accel_mag,
                levels=[accel_threshold],
                colors="cyan",
                linewidths=1.5,
                linestyles="--",
            )

        ax_accel.set_xlabel("Space x")
        ax_accel.set_ylabel("Time t")
        ax_accel.set_title(
            f"|Acceleration| (Sample {b + 1}, threshold={accel_threshold})"
        )
        ax_accel.set_xlim(0, nx * dx)
        ax_accel.set_ylim(dt, (nt - 1) * dt)

        plt.tight_layout()
        figures.append((f"grid_with_acceleration_sample_{b + 1}", fig))

    return figures
