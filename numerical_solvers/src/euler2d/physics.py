"""Equation of state and variable conversions for the 2D Euler system (PyTorch).

Conservative variables: (rho, rho*u, rho*v, E)
Primitive variables:    (rho, u, v, p)

Ideal gas EOS: p = (gamma - 1) * (E - 0.5 * rho * (u^2 + v^2))
"""

import torch


def primitive_to_conservative(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert primitive (rho, u, v, p) to conservative (rho, rho*u, rho*v, E)."""
    rho_u = rho * u
    rho_v = rho * v
    E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return rho, rho_u, rho_v, E


def conservative_to_primitive(
    rho: torch.Tensor,
    rho_u: torch.Tensor,
    rho_v: torch.Tensor,
    E: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert conservative (rho, rho*u, rho*v, E) to primitive (rho, u, v, p)."""
    inv_rho = 1.0 / rho.clamp(min=eps)
    u = torch.where(rho > eps, rho_u * inv_rho, torch.zeros_like(rho_u))
    v = torch.where(rho > eps, rho_v * inv_rho, torch.zeros_like(rho_v))
    p = (gamma - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    return rho, u, v, p


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
    rho_v: torch.Tensor,
    E: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute pressure from conservative variables."""
    inv_rho = 1.0 / rho.clamp(min=eps)
    u = torch.where(rho > eps, rho_u * inv_rho, torch.zeros_like(rho_u))
    v = torch.where(rho > eps, rho_v * inv_rho, torch.zeros_like(rho_v))
    return (gamma - 1.0) * (E - 0.5 * rho * (u * u + v * v))
