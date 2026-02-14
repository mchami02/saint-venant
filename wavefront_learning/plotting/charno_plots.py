"""CharNO diagnostic visualization functions.

Provides plots for diagnosing CharNO model internals:
- Selection weights per segment
- Winning segment map with characteristic line overlay
- Selection entropy (uncertainty)
- Local density predictions per segment
- 2x2 decomposition dashboard

All functions follow the standard signature:
    fn(traj_data, grid_config) -> list[tuple[str, Figure]]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from .base import _get_extent, _plot_heatmap


def _valid_segment_count(traj_data: dict, sample: int) -> int:
    """Return number of valid segments for a sample."""
    if "pieces_mask" in traj_data:
        return int(traj_data["pieces_mask"][sample].sum())
    # Fallback: use full K dimension
    return traj_data["selection_weights"].shape[-1]


def plot_selection_weights(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Plot per-segment selection weight heatmaps.

    Shows selection_weights[:, :, k] for each valid segment k.
    Annotates with rho_k if ks available.

    Args:
        traj_data: Dict with 'selection_weights' (B, nt, nx, K).
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    if "selection_weights" not in traj_data:
        return []

    weights = traj_data["selection_weights"]  # (B, nt, nx, K)
    extent = _get_extent(
        grid_config["nx"], grid_config["nt"],
        grid_config["dx"], grid_config["dt"],
    )

    B = weights.shape[0]
    figures = []
    for b in range(min(B, 2)):
        K_valid = _valid_segment_count(traj_data, b)
        ncols = min(K_valid, 6)
        nrows = (K_valid + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False
        )

        for k in range(K_valid):
            row, col = divmod(k, ncols)
            ax = axes[row, col]
            title = f"Segment {k}"
            if "ks" in traj_data:
                rho_k = traj_data["ks"][b, k]
                title += f" (rho={rho_k:.2f})"
            _plot_heatmap(ax, weights[b, :, :, k], extent, cmap="Blues",
                          vmin=0, vmax=1, label="Weight")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("t")

        # Hide unused axes
        for k in range(K_valid, nrows * ncols):
            row, col = divmod(k, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle(f"Selection Weights (Sample {b + 1})", fontsize=12)
        plt.tight_layout()
        figures.append((f"selection_weights_sample_{b + 1}", fig))

    return figures


def plot_winning_segment(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Plot categorical winning segment map with characteristic line overlay.

    Overlays analytical characteristic lines and Rankine-Hugoniot shock lines
    when xs/ks are available.

    Args:
        traj_data: Dict with 'selection_weights' (B, nt, nx, K).
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    if "selection_weights" not in traj_data:
        return []

    weights = traj_data["selection_weights"]  # (B, nt, nx, K)
    nx = grid_config["nx"]
    nt = grid_config["nt"]
    dx = grid_config["dx"]
    dt = grid_config["dt"]
    extent = _get_extent(nx, nt, dx, dt)
    t_max = nt * dt

    B = weights.shape[0]
    figures = []
    for b in range(min(B, 3)):
        K_valid = _valid_segment_count(traj_data, b)
        winner = np.argmax(weights[b], axis=-1)  # (nt, nx)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Discrete colormap for segments
        colors = plt.cm.tab10(np.linspace(0, 1, max(K_valid, 1)))
        cmap = ListedColormap(colors[:K_valid])

        im = ax.imshow(
            winner, extent=extent, aspect="auto", origin="lower",
            cmap=cmap, vmin=-0.5, vmax=K_valid - 0.5,
            interpolation="nearest",
        )
        cbar = plt.colorbar(im, ax=ax, label="Winning Segment")
        cbar.set_ticks(np.arange(K_valid))

        # Overlay characteristic lines and shock lines
        if "xs" in traj_data and "ks" in traj_data:
            xs = traj_data["xs"][b]  # (K+1,)
            ks = traj_data["ks"][b]  # (K,)
            t_line = np.linspace(0, t_max, 200)

            # Characteristic lines from each breakpoint
            for k in range(min(K_valid, len(ks))):
                speed = 1.0 - 2.0 * ks[k]  # Greenshields f'(rho)
                x_char = xs[k] + speed * t_line
                ax.plot(x_char, t_line, "--", color="white", alpha=0.5,
                        linewidth=0.8)
                # Right boundary characteristic
                if k + 1 < len(xs):
                    x_char_r = xs[k + 1] + speed * t_line
                    ax.plot(x_char_r, t_line, "--", color="white",
                            alpha=0.5, linewidth=0.8)

            # Rankine-Hugoniot shock lines between adjacent segments
            for k in range(min(K_valid - 1, len(ks) - 1)):
                if k + 1 < len(xs) and k + 1 < len(ks):
                    shock_speed = 1.0 - ks[k] - ks[k + 1]
                    x_shock = xs[k + 1] + shock_speed * t_line
                    ax.plot(x_shock, t_line, "-", color="red",
                            alpha=0.8, linewidth=1.5,
                            label="RH shock" if k == 0 else None)

            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            if K_valid > 1:
                ax.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        ax.set_title(f"Winning Segment (Sample {b + 1})")
        plt.tight_layout()
        figures.append((f"winning_segment_sample_{b + 1}", fig))

    return figures


def plot_selection_entropy(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Plot selection entropy heatmap.

    H(t,x) = -sum_k(w_k * log(w_k + eps)). High entropy indicates
    uncertain selection; low entropy indicates sharp selection.

    Args:
        traj_data: Dict with 'selection_weights' (B, nt, nx, K).
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    if "selection_weights" not in traj_data:
        return []

    weights = traj_data["selection_weights"]  # (B, nt, nx, K)
    extent = _get_extent(
        grid_config["nx"], grid_config["nt"],
        grid_config["dx"], grid_config["dt"],
    )

    eps = 1e-8
    entropy = -np.sum(weights * np.log(weights + eps), axis=-1)  # (B, nt, nx)

    # Get temperature if available
    temp_str = ""
    if "temperature" in traj_data:
        temp_val = traj_data["temperature"]
        if isinstance(temp_val, (int, float)):
            temp_str = f", tau={temp_val:.4f}"

    B = entropy.shape[0]
    figures = []
    for b in range(min(B, 3)):
        fig, ax = plt.subplots(figsize=(10, 8))
        mean_h = entropy[b].mean()
        _plot_heatmap(ax, entropy[b], extent, cmap="magma",
                      vmin=0, vmax=None, label="Entropy")
        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        ax.set_title(
            f"Selection Entropy (Sample {b + 1}, "
            f"mean={mean_h:.4f}{temp_str})"
        )
        plt.tight_layout()
        figures.append((f"selection_entropy_sample_{b + 1}", fig))

    return figures


def plot_local_densities(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Plot per-segment local density predictions.

    Shows local_rho[:, :, k] for each valid segment k.

    Args:
        traj_data: Dict with 'local_rho' (B, nt, nx, K).
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    if "local_rho" not in traj_data:
        return []

    local_rho = traj_data["local_rho"]  # (B, nt, nx, K)
    extent = _get_extent(
        grid_config["nx"], grid_config["nt"],
        grid_config["dx"], grid_config["dt"],
    )

    B = local_rho.shape[0]
    figures = []
    for b in range(min(B, 2)):
        K_valid = _valid_segment_count(traj_data, b)
        ncols = min(K_valid, 6)
        nrows = (K_valid + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False
        )

        for k in range(K_valid):
            row, col = divmod(k, ncols)
            ax = axes[row, col]
            title = f"Segment {k}"
            if "ks" in traj_data:
                rho_k = traj_data["ks"][b, k]
                title += f" (rho={rho_k:.2f})"
            _plot_heatmap(ax, local_rho[b, :, :, k], extent, cmap="viridis",
                          vmin=0, vmax=1, label="Density")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("t")

        # Hide unused axes
        for k in range(K_valid, nrows * ncols):
            row, col = divmod(k, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle(f"Local Densities (Sample {b + 1})", fontsize=12)
        plt.tight_layout()
        figures.append((f"local_densities_sample_{b + 1}", fig))

    return figures


def plot_charno_decomposition(
    traj_data: dict,
    grid_config: dict,
) -> list[tuple[str, Figure]]:
    """Plot 2x2 decomposition dashboard: GT | Prediction | Winning Segment | Entropy.

    Compact summary for quick diagnosis.

    Args:
        traj_data: Dict with 'grids', 'output_grid', 'selection_weights'.
        grid_config: Dict with {nx, nt, dx, dt}.

    Returns:
        List of (log_key, figure) pairs.
    """
    required = ("grids", "output_grid", "selection_weights")
    if not all(k in traj_data for k in required):
        return []

    grids = traj_data["grids"]
    output_grid = traj_data["output_grid"]
    weights = traj_data["selection_weights"]
    extent = _get_extent(
        grid_config["nx"], grid_config["nt"],
        grid_config["dx"], grid_config["dt"],
    )

    eps = 1e-8
    entropy = -np.sum(weights * np.log(weights + eps), axis=-1)

    B = grids.shape[0]
    figures = []
    for b in range(min(B, 3)):
        K_valid = _valid_segment_count(traj_data, b)
        winner = np.argmax(weights[b], axis=-1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: Ground Truth
        _plot_heatmap(axes[0, 0], grids[b], extent, cmap="viridis",
                      vmin=0, vmax=1, label="Density")
        axes[0, 0].set_title("Ground Truth")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("t")

        # Top-right: Prediction
        _plot_heatmap(axes[0, 1], output_grid[b], extent, cmap="viridis",
                      vmin=0, vmax=1, label="Density")
        axes[0, 1].set_title("Prediction")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("t")

        # Bottom-left: Winning Segment
        colors = plt.cm.tab10(np.linspace(0, 1, max(K_valid, 1)))
        cmap = ListedColormap(colors[:K_valid])
        im_w = axes[1, 0].imshow(
            winner, extent=extent, aspect="auto", origin="lower",
            cmap=cmap, vmin=-0.5, vmax=K_valid - 0.5,
            interpolation="nearest",
        )
        cbar = plt.colorbar(im_w, ax=axes[1, 0], label="Segment")
        cbar.set_ticks(np.arange(K_valid))
        axes[1, 0].set_title("Winning Segment")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("t")

        # Bottom-right: Entropy
        _plot_heatmap(axes[1, 1], entropy[b], extent, cmap="magma",
                      vmin=0, vmax=None, label="Entropy")
        mean_h = entropy[b].mean()
        temp_str = ""
        if "temperature" in traj_data:
            temp_val = traj_data["temperature"]
            if isinstance(temp_val, (int, float)):
                temp_str = f", tau={temp_val:.4f}"
        axes[1, 1].set_title(f"Entropy (mean={mean_h:.4f}{temp_str})")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("t")

        fig.suptitle(f"CharNO Decomposition (Sample {b + 1})", fontsize=13)
        plt.tight_layout()
        figures.append((f"charno_decomposition_sample_{b + 1}", fig))

    return figures
