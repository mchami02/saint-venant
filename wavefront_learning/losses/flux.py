"""Centralized flux functions for LWR traffic flow with Greenshields flux.

This module provides flux functions used across multiple loss computations:
- greenshields_flux: f(rho) = rho * (1 - rho)
- greenshields_flux_derivative: f'(rho) = 1 - 2*rho
- compute_shock_speed: s = 1 - rho_L - rho_R (from Rankine-Hugoniot)
"""

import torch


def greenshields_flux(rho: torch.Tensor) -> torch.Tensor:
    """Greenshields flux function: f(rho) = rho * (1 - rho).

    Args:
        rho: Density tensor.

    Returns:
        Flux values with same shape as input.
    """
    return rho * (1.0 - rho)


def greenshields_flux_derivative(rho: torch.Tensor) -> torch.Tensor:
    """Derivative of Greenshields flux: f'(rho) = 1 - 2*rho.

    Args:
        rho: Density tensor.

    Returns:
        Flux derivative values with same shape as input.
    """
    return 1.0 - 2.0 * rho


def compute_shock_speed(rho_L: torch.Tensor, rho_R: torch.Tensor) -> torch.Tensor:
    """Compute shock speed from Rankine-Hugoniot condition for Greenshields flux.

    For Greenshields flux f(rho) = rho * (1 - rho):
        s = [f(rho_R) - f(rho_L)] / [rho_R - rho_L]
          = [rho_R(1-rho_R) - rho_L(1-rho_L)] / [rho_R - rho_L]
          = 1 - rho_L - rho_R

    Args:
        rho_L: Left density values.
        rho_R: Right density values.

    Returns:
        Shock speeds with same shape as inputs.
    """
    return 1.0 - rho_L - rho_R
