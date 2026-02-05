"""Plotting and regime analysis for ARZ solutions."""

import numpy as np
import matplotlib.pyplot as plt
from .config import ARZConfig
from .solver import dp_drho, pressure


def compute_regime(
    rho_history: np.ndarray,
    v_history: np.ndarray,
    gamma: float,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Classify each (t,x) as critical (0), subcritical (1), or supercritical (2).
    """
    c_history = np.sqrt(rho_history * dp_drho(rho_history, gamma))
    lambda1 = v_history - c_history
    lambda2 = v_history

    regime = np.zeros_like(v_history, dtype=np.int8)
    regime[(np.abs(lambda1) <= tol) | (np.abs(lambda2) <= tol)] = 0
    regime[(lambda1 < 0) & (lambda2 > 0)] = 1
    regime[(regime != 0) & (regime != 1)] = 2
    return regime


def plot_results(
    rho_history: np.ndarray,
    w_history: np.ndarray,
    v_history: np.ndarray,
    config: ARZConfig,
    title_suffix: str = "",
    show_regime: bool = True,
) -> None:
    """
    Plot spacetime diagrams for ρ, w, v and optionally regime and |v| - c.
    """
    ntime, nx = rho_history.shape
    T = config.dt * (ntime - 1)
    x_left, x_right = config.x[0], config.x[-1]
    gamma = config.gamma

    regime = compute_regime(rho_history, v_history, gamma)
    subfraction = (regime == 1).sum() / regime.size
    superfraction = (regime == 2).sum() / regime.size
    print(f"{title_suffix} Results:")
    print(f"  Subcritical: {subfraction*100:.1f}%  Supercritical: {superfraction*100:.1f}%")

    # Figure 1: ρ, w, v
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
    extent = [x_left, x_right, 0.0, T]

    im0 = axes[0].imshow(rho_history, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    axes[0].set_xlabel("Position x")
    axes[0].set_ylabel("Time t")
    axes[0].set_title(f"ARZ: Density ρ (spacetime){title_suffix}")
    plt.colorbar(im0, ax=axes[0], label="Density ρ")

    im1 = axes[1].imshow(w_history, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    axes[1].set_xlabel("Position x")
    axes[1].set_ylabel("Time t")
    axes[1].set_title(f"ARZ: w (spacetime){title_suffix}")
    plt.colorbar(im1, ax=axes[1], label="w")

    im2 = axes[2].imshow(v_history, aspect="auto", origin="lower", extent=extent, cmap="twilight")
    axes[2].set_xlabel("Position x")
    axes[2].set_ylabel("Time t")
    axes[2].set_title(f"ARZ: Velocity v (spacetime){title_suffix}")
    plt.colorbar(im2, ax=axes[2], label="Velocity v")
    plt.tight_layout()
    plt.show()

    if not show_regime:
        return

    # Figure 2: Regime and |v| - c
    c_history = np.sqrt(rho_history * dp_drho(rho_history, gamma))
    diff = np.abs(v_history) - c_history

    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(
        regime, cmap="RdYlBu_r", aspect="auto", origin="lower", extent=extent, vmin=0, vmax=2
    )
    cbar = plt.colorbar(im, ax=axes[0], ticks=[0, 1, 2])
    cbar.set_ticklabels(["Critical", "Subcritical", "Supercritical"])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title(f"ARZ: Regimes{title_suffix}")

    im_diff = axes[1].imshow(diff, aspect="auto", origin="lower", extent=extent)
    axes[1].contour(
        np.linspace(x_left, x_right, nx),
        np.linspace(0, T, ntime),
        diff,
        levels=[0],
        colors="k",
        linewidths=2,
    )
    plt.colorbar(im_diff, ax=axes[1], label="|v| - c")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title(f"|v| - c{title_suffix}")
    plt.tight_layout()
    plt.show()
