"""2D LWR flux functions, derivatives, critical densities, and CFL utility."""

import torch


def greenshields_flux(rho: torch.Tensor, v_max: float, rho_max: float) -> torch.Tensor:
    """Greenshields flux: f(rho) = v_max * rho * (1 - rho / rho_max)."""
    return v_max * rho * (1.0 - rho / rho_max)


def greenshields_derivative(
    rho: torch.Tensor, v_max: float, rho_max: float
) -> torch.Tensor:
    """Derivative of Greenshields flux: f'(rho) = v_max * (1 - 2*rho / rho_max)."""
    return v_max * (1.0 - 2.0 * rho / rho_max)


def greenshields_critical_density(rho_max: float) -> float:
    """Density maximising Greenshields flux: rho_crit = rho_max / 2."""
    return rho_max / 2.0


def triangular_flux(
    rho: torch.Tensor, v_f: float, w: float, rho_max: float
) -> torch.Tensor:
    """Triangular (two-slope) flux: min(v_f * rho, w * (rho_max - rho))."""
    return torch.minimum(v_f * rho, w * (rho_max - rho))


def triangular_critical_density(v_f: float, w: float, rho_max: float) -> float:
    """Density maximising triangular flux: rho_crit = w * rho_max / (v_f + w)."""
    return w * rho_max / (v_f + w)


def cfl_dt(
    dx: float,
    dy: float,
    max_speed_x: float,
    max_speed_y: float,
    cfl_number: float = 0.45,
) -> float:
    """Compute the maximum stable time step for 2D explicit FV.

    Uses the 2D CFL condition:
        dt * (max_speed_x / dx + max_speed_y / dy) <= cfl_number
    """
    inv_dt = max_speed_x / dx + max_speed_y / dy
    if inv_dt == 0.0:
        return float("inf")
    return cfl_number / inv_dt
