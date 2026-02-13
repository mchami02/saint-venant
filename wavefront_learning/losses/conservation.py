"""Conservation loss for scalar conservation laws.

Enforces that total mass is conserved over time:
    integral rho(t, x) dx ~ constant for all t.

This is a fundamental physical constraint that holds for solutions
of conservation laws with appropriate boundary conditions.
"""

import torch

from .base import BaseLoss


class ConservationLoss(BaseLoss):
    """Penalizes variation of total mass over time.

    Computes the variance of the spatial integral across time steps:
        L = Var_t( sum_x rho(t, x) * dx )

    Args:
        dx: Spatial grid spacing. If None, assumes uniform spacing of 1.
    """

    def __init__(self, dx: float | None = None):
        super().__init__()
        self.dx = dx

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute conservation loss.

        Args:
            input_dict: Input dictionary (unused).
            output_dict: Must contain 'output_grid' of shape (B, 1, nt, nx).
            target: Target grid (unused, conservation is self-supervised).

        Returns:
            Tuple of (loss tensor, components dict).
        """
        pred = output_dict["output_grid"]
        dx = self.dx if self.dx is not None else 1.0

        # Remove channel dim if present: (B, 1, nt, nx) -> (B, nt, nx)
        if pred.dim() == 4:
            pred = pred.squeeze(1)

        # Total mass per time step: sum over spatial dimension
        mass_per_time = torch.sum(pred, dim=-1) * dx  # (B, nt)

        # Variance of mass over time, averaged over batch
        loss = torch.mean(torch.var(mass_per_time, dim=-1))

        return loss, {"conservation": loss.item()}
