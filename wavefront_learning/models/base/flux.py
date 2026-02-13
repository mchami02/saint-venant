"""Pluggable flux interface for scalar conservation laws.

Provides an abstract Flux base class and concrete implementations
for common traffic flow flux functions. Used by CharNO to compute
physics-augmented features (characteristic speeds, shock speeds).
"""

import torch
import torch.nn as nn


class Flux(nn.Module):
    """Abstract flux function for scalar conservation laws.

    A flux f(rho) defines the conservation law:
        d(rho)/dt + d(f(rho))/dx = 0

    Subclasses must implement forward() and derivative().
    The shock_speed() method uses the Rankine-Hugoniot condition by default.
    """

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute flux value f(rho)."""
        raise NotImplementedError

    def derivative(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute flux derivative f'(rho) = characteristic speed."""
        raise NotImplementedError

    def shock_speed(
        self, rho_L: torch.Tensor, rho_R: torch.Tensor
    ) -> torch.Tensor:
        """Compute shock speed from the Rankine-Hugoniot condition.

        s = [f(rho_R) - f(rho_L)] / [rho_R - rho_L]

        Args:
            rho_L: Left state density.
            rho_R: Right state density.

        Returns:
            Shock speed with same shape as inputs.
        """
        eps = 1e-8
        return (self.forward(rho_R) - self.forward(rho_L)) / (
            rho_R - rho_L + eps
        )


class GreenshieldsFlux(Flux):
    """Greenshields flux: f(rho) = rho * (1 - rho).

    Normalized form with v_max = 1, rho_max = 1.
    Characteristic speed: f'(rho) = 1 - 2*rho.
    Shock speed (analytical): s = 1 - rho_L - rho_R.
    """

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        return rho * (1.0 - rho)

    def derivative(self, rho: torch.Tensor) -> torch.Tensor:
        return 1.0 - 2.0 * rho

    def shock_speed(
        self, rho_L: torch.Tensor, rho_R: torch.Tensor
    ) -> torch.Tensor:
        # Analytical closed-form for Greenshields (avoids division)
        return 1.0 - rho_L - rho_R


class TriangularFlux(Flux):
    """Triangular flux: f(rho) = min(v_f * rho, w * (1 - rho)).

    Piecewise linear flux with free-flow speed v_f and backward wave speed w.
    Critical density: rho_crit = w / (v_f + w).

    Args:
        v_f: Free-flow speed (default 1.0).
        w: Backward wave speed (default 1.0).
    """

    def __init__(self, v_f: float = 1.0, w: float = 1.0):
        super().__init__()
        self.v_f = v_f
        self.w = w
        self.rho_crit = w / (v_f + w)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        return torch.min(self.v_f * rho, self.w * (1.0 - rho))

    def derivative(self, rho: torch.Tensor) -> torch.Tensor:
        # Piecewise: v_f if rho < rho_crit, -w if rho > rho_crit
        return torch.where(
            rho < self.rho_crit,
            torch.tensor(self.v_f, device=rho.device, dtype=rho.dtype),
            torch.tensor(-self.w, device=rho.device, dtype=rho.dtype),
        )


# Default flux for the LWR/Greenshields problem
DEFAULT_FLUX = GreenshieldsFlux
