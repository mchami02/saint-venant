"""ARZ equation: pressure law, its derivative, and eigenvalues."""

import torch


def pressure(rho: torch.Tensor, gamma: float) -> torch.Tensor:
    """Pressure p(rho) = rho^gamma."""
    return rho.pow(gamma)


def dp_drho(rho: torch.Tensor, gamma: float) -> torch.Tensor:
    """Derivative p'(rho) = gamma * rho^(gamma-1)."""
    if gamma == 1.0:
        return torch.ones_like(rho)
    return gamma * rho.pow(gamma - 1)


def eigenvalues(
    rho: torch.Tensor, v: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues (lam1, lam2) of the ARZ system.

    lam1 = v
    lam2 = v - rho * p'(rho)
    """
    lam1 = v
    lam2 = v - rho * dp_drho(rho, gamma)
    return lam1, lam2
