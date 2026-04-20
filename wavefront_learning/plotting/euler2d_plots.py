"""Plotting for 2D Euler fields.

Each tensor is ``(B, 4, nt, ny, nx)``.  We render a horizontal strip of
snapshots at a small number of evenly-spaced timesteps for each of the
four primitive variables ``[rho, u, v, p]``.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


_EULER2D_CHANNELS = [
    ("rho", "density", "viridis"),
    ("u", "u-velocity", "coolwarm"),
    ("v", "v-velocity", "coolwarm"),
    ("p", "pressure", "plasma"),
]


def _snapshot_strip(
    grid: np.ndarray, title_prefix: str, n_snapshots: int = 4
) -> Figure:
    """Render a (4 channels) x (n_snapshots) strip of 2D snapshots.

    ``grid`` is ``(4, nt, ny, nx)``.
    """
    _, nt, _, _ = grid.shape
    ts = np.linspace(0, nt - 1, n_snapshots, dtype=int)
    fig, axes = plt.subplots(
        4, n_snapshots, figsize=(3 * n_snapshots, 10), squeeze=False
    )
    for ch, (name, label, cmap) in enumerate(_EULER2D_CHANNELS):
        for j, t in enumerate(ts):
            ax = axes[ch, j]
            im = ax.imshow(
                grid[ch, t], origin="lower", cmap=cmap, aspect="equal"
            )
            ax.set_title(f"{title_prefix} {name} t={t}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


def plot_euler2d_ground_truth(traj_data: dict, grid_config: dict):
    if "grids" not in traj_data:
        return []
    grids = traj_data["grids"]
    if not isinstance(grids, np.ndarray):
        grids = np.asarray(grids)
    B = grids.shape[0]
    figures = []
    for b in range(min(B, 3)):
        fig = _snapshot_strip(grids[b], title_prefix=f"GT sample {b+1}")
        figures.append((f"euler2d_gt_sample_{b+1}", fig))
    return figures


def plot_euler2d_pred(traj_data: dict, grid_config: dict):
    if "output_grid" not in traj_data:
        return []
    pred = traj_data["output_grid"]
    if not isinstance(pred, np.ndarray):
        pred = np.asarray(pred)
    B = pred.shape[0]
    figures = []
    for b in range(min(B, 3)):
        fig = _snapshot_strip(pred[b], title_prefix=f"Pred sample {b+1}")
        figures.append((f"euler2d_pred_sample_{b+1}", fig))
    return figures


def plot_euler2d_mse_error(traj_data: dict, grid_config: dict):
    if "output_grid" not in traj_data or "grids" not in traj_data:
        return []
    pred = traj_data["output_grid"]
    gt = traj_data["grids"]
    if not isinstance(pred, np.ndarray):
        pred = np.asarray(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.asarray(gt)
    err = (pred - gt) ** 2
    B = err.shape[0]
    figures = []
    for b in range(min(B, 3)):
        fig = _snapshot_strip(err[b], title_prefix=f"MSE sample {b+1}")
        figures.append((f"euler2d_mse_sample_{b+1}", fig))
    return figures
