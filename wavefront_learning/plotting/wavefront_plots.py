"""Wave pattern visualization for WaveFrontModel.

Plot functions compatible with the PLOTS registry:
- plot_wave_pattern: GT grid with wave lines overlay (shocks, rarefactions, bent shocks)

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
    - Red solid lines: shock waves (single line)
    - Blue filled region: rarefaction fans (two diverging edges)
    - Lime solid lines: bent shock waves (polynomial curve)
    - Star markers at collision points (origin_t > 0)

    Args:
        traj_data: Dict containing:
            - grids: (B, nt, nx) ground truth grids
            - wave_origins_x: (B, W) wave origin x positions
            - wave_origins_t: (B, W) wave origin times
            - wave_left_speed: (B, W) left edge speeds
            - wave_right_speed: (B, W) right edge speeds
            - wave_active: (B, W) wave activity masks
            - wave_types: (B, W) wave types (0=shock, 1=rarefaction, 2=bent shock)
            - wave_poly_c2: (B, W) bent shock quadratic coeff (optional)
            - wave_poly_c3: (B, W) bent shock cubic coeff (optional)
            - wave_poly_duration: (B, W) bent shock curve duration (optional)
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    grids = traj_data["grids"]
    wave_ox = traj_data["wave_origins_x"]
    wave_ot = traj_data["wave_origins_t"]
    wave_left_sp = traj_data["wave_left_speed"]
    wave_right_sp = traj_data["wave_right_speed"]
    wave_ac = traj_data["wave_active"]
    wave_ty = traj_data["wave_types"]

    # Optional bent shock parameters
    wave_c2 = traj_data.get("wave_poly_c2")
    wave_c3 = traj_data.get("wave_poly_c3")
    wave_dur = traj_data.get("wave_poly_duration")

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
            left_sp = float(wave_left_sp[b, w])
            right_sp = float(wave_right_sp[b, w])
            wtype = float(wave_ty[b, w])

            if wtype < 0.5:
                # --- Shock: single line ---
                _draw_linear_wave(
                    ax, ox, ot, left_sp, T_max, X_max,
                    color="red", linestyle="-", linewidth=2.0,
                    alpha=min(1.0, activity),
                )
            elif wtype < 1.5:
                # --- Rarefaction: two diverging edges with fill ---
                t_pts = np.linspace(ot, T_max, 100)
                x_left = ox + left_sp * (t_pts - ot)
                x_right = ox + right_sp * (t_pts - ot)

                # Clip to domain
                x_left = np.clip(x_left, 0, X_max)
                x_right = np.clip(x_right, 0, X_max)

                ax.fill_betweenx(
                    t_pts, x_left, x_right,
                    alpha=0.15 * min(1.0, activity),
                    color="deepskyblue",
                )
                ax.plot(
                    x_left, t_pts,
                    color="deepskyblue", linestyle="-", linewidth=1.0,
                    alpha=min(1.0, activity),
                )
                ax.plot(
                    x_right, t_pts,
                    color="deepskyblue", linestyle="-", linewidth=1.0,
                    alpha=min(1.0, activity),
                )
            else:
                # --- Bent shock: polynomial curve ---
                c2 = float(wave_c2[b, w]) if wave_c2 is not None else 0.0
                c3 = float(wave_c3[b, w]) if wave_c3 is not None else 0.0
                dur = float(wave_dur[b, w]) if wave_dur is not None else 0.0

                t_pts = np.linspace(ot, T_max, 200)
                dt_pts = t_pts - ot
                c1 = left_sp

                # Curved portion
                pos_curved = ox + c1 * dt_pts + c2 * dt_pts**2 + c3 * dt_pts**3

                if dur > 1e-8:
                    # After curved portion: linear continuation
                    exit_speed = c1 + 2.0 * c2 * dur + 3.0 * c3 * dur**2
                    pos_at_dur = ox + c1 * dur + c2 * dur**2 + c3 * dur**3
                    pos_after = pos_at_dur + exit_speed * (dt_pts - dur)
                    in_curve = dt_pts <= dur
                    x_pts = np.where(in_curve, pos_curved, pos_after)
                else:
                    x_pts = pos_curved

                x_pts = np.clip(x_pts, 0, X_max)

                ax.plot(
                    x_pts, t_pts,
                    color="lime", linestyle="-", linewidth=1.5,
                    alpha=min(1.0, activity),
                )

            # Mark collision points (waves with origin_t > 0)
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


def _draw_linear_wave(
    ax, ox: float, ot: float, speed: float,
    T_max: float, X_max: float,
    **plot_kwargs,
) -> None:
    """Draw a linear wave line clipped to the domain."""
    t_start = ot
    t_end = T_max

    if abs(speed) > 1e-8:
        t_at_x0 = ot + (0.0 - ox) / speed
        t_at_xmax = ot + (X_max - ox) / speed
        t_bounds = sorted([t_at_x0, t_at_xmax])
        t_start = max(t_start, t_bounds[0])
        t_end = min(t_end, t_bounds[1])

    if t_end <= t_start:
        return

    x_start = np.clip(ox + speed * (t_start - ot), 0, X_max)
    x_end = np.clip(ox + speed * (t_end - ot), 0, X_max)

    ax.plot([x_start, x_end], [t_start, t_end], **plot_kwargs)
