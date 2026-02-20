"""Wave pattern visualization for WaveFrontModel.

Plot functions compatible with the PLOTS registry:
- plot_wave_pattern: GT grid with wave lines overlay (shocks, rarefactions, spawned)

All functions return list[tuple[str, Figure]] of (log_key, figure) pairs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .base import _get_extent


def plot_wave_pattern(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Plot predicted wave pattern on top of ground truth grid.

    For each sample (up to 3):
    - Background: GT grid heatmap (viridis)
    - Red solid lines: shock waves
    - Blue line bundles: rarefaction fans (N sub-waves)
    - Green dashed lines: spawned waves from interactions
    - Star markers at collision points (origin_t > 0)

    Args:
        traj_data: Dict containing:
            - grids: (B, nt, nx) ground truth grids
            - wave_origins_x: (B, W) wave origin x positions
            - wave_origins_t: (B, W) wave origin times
            - wave_speeds: (B, W) wave speeds
            - wave_active: (B, W) wave activity masks
            - wave_types: (B, W) wave types (0=shock, 1=rarefaction, 2=spawned)
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    grids = traj_data["grids"]
    wave_ox = traj_data["wave_origins_x"]
    wave_ot = traj_data["wave_origins_t"]
    wave_sp = traj_data["wave_speeds"]
    wave_ac = traj_data["wave_active"]
    wave_ty = traj_data["wave_types"]

    nx, nt, dx, dt = (
        grid_config["nx"],
        grid_config["nt"],
        grid_config["dx"],
        grid_config["dt"],
    )
    extent = _get_extent(nx, nt, dx, dt)
    T_max = nt * dt
    X_max = nx * dx

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

        # Overlay wave lines
        W = wave_ox.shape[1]
        collision_xs = []
        collision_ts = []

        for w in range(W):
            activity = float(wave_ac[b, w])
            if activity < 0.05:
                continue

            ox = float(wave_ox[b, w])
            ot = float(wave_ot[b, w])
            sp = float(wave_sp[b, w])
            wtype = float(wave_ty[b, w])

            # Compute wave line endpoints clipped to domain
            # Wave: x(t) = ox + sp * (t - ot)
            # Find t range where x is in [0, X_max]
            t_start = ot
            t_end = T_max

            # Clip to spatial domain
            if abs(sp) > 1e-8:
                t_at_x0 = ot + (0.0 - ox) / sp
                t_at_xmax = ot + (X_max - ox) / sp
                t_bounds = sorted([t_at_x0, t_at_xmax])
                t_start = max(t_start, t_bounds[0])
                t_end = min(t_end, t_bounds[1])

            if t_end <= t_start:
                continue

            x_start = ox + sp * (t_start - ot)
            x_end = ox + sp * (t_end - ot)

            # Clip to domain
            x_start = np.clip(x_start, 0, X_max)
            x_end = np.clip(x_end, 0, X_max)

            # Style by wave type
            if wtype < 0.5:
                # Shock
                color = "red"
                linestyle = "-"
                linewidth = 2.0
            elif wtype < 1.5:
                # Rarefaction
                color = "deepskyblue"
                linestyle = "-"
                linewidth = 1.0
            else:
                # Spawned
                color = "lime"
                linestyle = "--"
                linewidth = 1.5

            ax.plot(
                [x_start, x_end],
                [t_start, t_end],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=min(1.0, activity),
            )

            # Mark collision points (spawned waves with origin_t > 0)
            if ot > 1e-6:
                collision_xs.append(ox)
                collision_ts.append(ot)

        # Draw star markers at collision points
        if collision_xs:
            ax.plot(
                collision_xs,
                collision_ts,
                "*",
                color="yellow",
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=5,
            )

        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        ax.set_title(f"GT Grid + Wave Pattern (Sample {b + 1})")
        ax.set_xlim(0, X_max)
        ax.set_ylim(0, T_max)
        plt.tight_layout()
        figures.append((f"wave_pattern_sample_{b + 1}", fig))

    return figures
