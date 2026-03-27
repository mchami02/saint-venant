"""Equation of state and variable conversions for the 1D Euler system (PyTorch).

Conservative variables: (rho, rho*u, E)
Primitive variables:    (rho, u, p)

Ideal gas EOS: p = (gamma - 1) * (E - 0.5 * rho * u^2)
"""

import torch


def primitive_to_conservative(
    rho: torch.Tensor,
    u: torch.Tensor,
    p: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert primitive (rho, u, p) to conservative (rho, rho*u, E)."""
    rho_u = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return rho, rho_u, E


def conservative_to_primitive(
    rho: torch.Tensor,
    rho_u: torch.Tensor,
    E: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert conservative (rho, rho*u, E) to primitive (rho, u, p)."""
    u = torch.where(rho > eps, rho_u / rho, torch.zeros_like(rho_u))
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    return rho, u, p


def sound_speed(
    rho: torch.Tensor,
    p: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Speed of sound: c = sqrt(gamma * p / rho)."""
    return torch.sqrt(gamma * p.clamp(min=0.0) / rho.clamp(min=eps))


def pressure_from_conservative(
    rho: torch.Tensor,
    rho_u: torch.Tensor,
    E: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute pressure from conservative variables."""
    u = torch.where(rho > eps, rho_u / rho, torch.zeros_like(rho_u))
    return (gamma - 1.0) * (E - 0.5 * rho * u**2)
