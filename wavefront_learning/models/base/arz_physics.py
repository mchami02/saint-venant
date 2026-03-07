"""ARZ traffic flow physics as an nn.Module for model serialization.

Wraps the ARZ pressure law and eigenvalue computations so they can be
stored inside model checkpoints (unlike plain functions).

ARZ system (conservative form):
    drho/dt + d(rho*v)/dx = 0           (mass conservation)
    d(rho*w)/dt + d(rho*w*v)/dx = 0     (momentum), w = v + p(rho)

Pressure law: p(rho) = rho^gamma
Eigenvalues:  lam1 = v,  lam2 = v - rho * p'(rho) = v - gamma * rho^gamma
"""

import torch
import torch.nn as nn


class ARZPhysics(nn.Module):
    """ARZ pressure law and eigenvalue computations.

    Args:
        gamma: Pressure exponent (default 1.0).
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def pressure(self, rho: torch.Tensor) -> torch.Tensor:
        """Pressure p(rho) = rho^gamma."""
        return rho.pow(self.gamma)

    def dp_drho(self, rho: torch.Tensor) -> torch.Tensor:
        """Pressure derivative p'(rho) = gamma * rho^(gamma-1)."""
        if self.gamma == 1.0:
            return torch.ones_like(rho)
        return self.gamma * rho.pow(self.gamma - 1)

    def eigenvalues(
        self, rho: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ARZ eigenvalues.

        Args:
            rho: Density (any shape).
            v: Velocity (same shape as rho).

        Returns:
            (lam1, lam2) where lam1 = v, lam2 = v - gamma * rho^gamma.
            lam2 < lam1 for rho > 0.
        """
        lam1 = v
        lam2 = v - rho * self.dp_drho(rho)
        return lam1, lam2

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}"
