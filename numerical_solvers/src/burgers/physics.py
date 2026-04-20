"""Physics helpers for the inviscid Burgers equation (PyTorch).

Scalar conservation law:
    u_t + (u^2 / 2)_x = 0

The conserved variable and the primitive variable are the same scalar ``u``.
"""

import torch


def flux(u: torch.Tensor) -> torch.Tensor:
    """Burgers flux: f(u) = 0.5 * u^2."""
    return 0.5 * u * u


def max_wave_speed(u: torch.Tensor) -> float:
    """Maximum characteristic speed |f'(u)| = |u| over all cells."""
    return u.abs().max().item()
