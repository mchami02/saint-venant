"""ARZ (Aw-Rascle-Zhang) physics for neural operator modules.

Provides eigenvalues, shock speeds, and contact speeds for the ARZ system:
    rho_t + (rho * v)_x = 0
    (rho * w)_t + (rho * w * v)_x = 0

where w = v + p(rho) is the Lagrangian marker (Riemann invariant of the
1-family), and p(rho) = rho^gamma is the pressure law.

Eigenvalues:
    lambda_1 = v               (linearly degenerate, contact)
    lambda_2 = v - rho * p'(rho)  (genuinely nonlinear, shock/rarefaction)
"""

import torch
import torch.nn as nn


class ARZPhysics(nn.Module):
    """ARZ equation physics: pressure, eigenvalues, and wave speeds.

    Args:
        gamma: Pressure exponent (p(rho) = rho^gamma). Default 1.0.
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
        """Eigenvalues (lam1, lam2) of the ARZ system.

        Args:
            rho: Density values.
            v: Velocity values.

        Returns:
            (lam1, lam2) where lam1 = v, lam2 = v - rho * p'(rho).
        """
        lam1 = v
        lam2 = v - rho * self.dp_drho(rho)
        return lam1, lam2

    def shock_speed_rh(
        self,
        rho_L: torch.Tensor,
        v_L: torch.Tensor,
        rho_R: torch.Tensor,
        v_R: torch.Tensor,
    ) -> torch.Tensor:
        """Mass-conservation Rankine-Hugoniot shock speed.

        s = (rho_R * v_R - rho_L * v_L) / (rho_R - rho_L)

        Uses sign-preserving denominator clamping to avoid division by zero.

        Args:
            rho_L, v_L: Left state (density, velocity).
            rho_R, v_R: Right state (density, velocity).

        Returns:
            Shock speed tensor with same shape as inputs.
        """
        num = rho_R * v_R - rho_L * v_L
        denom = rho_R - rho_L
        # Sign-preserving clamp: keep sign of denom, ensure |denom| >= eps
        eps = 1e-8
        safe_denom = torch.where(
            denom >= 0,
            denom.clamp(min=eps),
            denom.clamp(max=-eps),
        )
        return num / safe_denom

    def contact_speed(self, v_R: torch.Tensor) -> torch.Tensor:
        """Contact discontinuity speed (1-family): lambda_1(U*) = v_R.

        In the ARZ Riemann problem, the 1-wave is a contact discontinuity
        traveling at speed v_R (the right state velocity).

        Args:
            v_R: Right state velocity.

        Returns:
            Contact speed (= v_R).
        """
        return v_R
