"""Shock proximity visualization for ShockAwareDeepONet.

Provides side-by-side comparison of GT and predicted shock proximity fields.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .base import _get_extent


def plot_shock_proximity(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Plot GT vs predicted shock proximity and absolute error.

    Args:
        traj_data: Must contain "shock_proximity" (pred) and
            "shock_proximity_gt" (GT), each of shape (B, nt, nx).
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    pred = traj_data.get("shock_proximity")
    gt = traj_data.get("shock_proximity_gt")
    if pred is None or gt is None:
        return []

    nx = grid_config["nx"]
    nt = grid_config["nt"]
    dx = grid_config["dx"]
    dt = grid_config["dt"]
    extent = _get_extent(nx, nt, dx, dt)

    n_samples = min(len(pred), len(gt), 3)
    figures = []

    for b in range(n_samples):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # GT proximity
        im0 = axes[0].imshow(
            gt[b], aspect="auto", origin="lower", extent=extent,
            cmap="hot", vmin=0, vmax=1,
        )
        axes[0].set_title("GT Proximity")
        axes[0].set_xlabel("Space x")
        axes[0].set_ylabel("Time t")
        plt.colorbar(im0, ax=axes[0])

        # Predicted proximity
        im1 = axes[1].imshow(
            pred[b], aspect="auto", origin="lower", extent=extent,
            cmap="hot", vmin=0, vmax=1,
        )
        axes[1].set_title("Predicted Proximity")
        axes[1].set_xlabel("Space x")
        axes[1].set_ylabel("Time t")
        plt.colorbar(im1, ax=axes[1])

        # Absolute error
        error = np.abs(pred[b] - gt[b])
        im2 = axes[2].imshow(
            error, aspect="auto", origin="lower", extent=extent,
            cmap="hot", vmin=0, vmax=1,
        )
        axes[2].set_title("Absolute Error")
        axes[2].set_xlabel("Space x")
        axes[2].set_ylabel("Time t")
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        figures.append((f"shock_proximity_sample_{b + 1}", fig))

    return figures
