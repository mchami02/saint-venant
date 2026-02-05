"""
ARZ (Aw-Rascle-Zhang) traffic flow model solver.

Usage:
    from arz_solver import ARZConfig, run, plot_results, initial_condition_from_steps

    config = ARZConfig(nx=200, L=1.0, T=1.0, gamma=1.0)
    rho0, w0 = initial_condition_riemann(config.x, rho_left=0.8, rho_right=0.2, v0=0.1)
    rho_hist, w_hist, v_hist = run(rho0, w0, config, bc_type="zero_gradient")
    plot_results(rho_hist, w_hist, v_hist, config, title_suffix=" - zero_gradient")
"""

from .config import ARZConfig
from .solver import run, rusanov_flux, pressure, dp_drho
from .initial_conditions import (
    initial_condition_from_steps,
    initial_condition_riemann,
    initial_condition_three_region,
)
from .plotting import plot_results, compute_regime

__all__ = [
    "ARZConfig",
    "run",
    "plot_results",
    "compute_regime",
    "initial_condition_from_steps",
    "initial_condition_riemann",
    "initial_condition_three_region",
    "rusanov_flux",
    "pressure",
    "dp_drho",
]
