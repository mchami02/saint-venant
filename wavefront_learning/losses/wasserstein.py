"""Wasserstein-1 loss for discontinuous solutions.

The W1 (Earth Mover's Distance) metric is mathematically appropriate for
solutions of hyperbolic conservation laws because it penalizes mislocated
shocks linearly in the displacement, rather than quadratically like MSE.

In 1D, W1 has a closed-form: the L1 norm of the cumulative sum of the error.
This is equivalent to the Sobolev W^{-1,1} (Kantorovich-Rubinstein) norm.
"""

import torch

from .base import BaseLoss


class WassersteinLoss(BaseLoss):
    """Wasserstein-1 loss computed per spatial slice.

    For each time step, computes:
        W1(t) = sum_x |cumsum_x(pred - target) * dx| * dx

    Then averages over time steps:
        L = mean_t W1(t)

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
        """Compute Wasserstein-1 loss.

        Args:
            input_dict: Input dictionary (unused).
            output_dict: Must contain 'output_grid' of shape (B, 1, nt, nx).
            target: Target grid of shape (B, 1, nt, nx).

        Returns:
            Tuple of (loss tensor, components dict).
        """
        pred = output_dict["output_grid"]
        dx = self.dx if self.dx is not None else 1.0

        # Remove channel dim if present: (B, 1, nt, nx) -> (B, nt, nx)
        if pred.dim() == 4:
            pred = pred.squeeze(1)
            target = target.squeeze(1)

        # Compute error per spatial slice
        error = pred - target  # (B, nt, nx)

        # Cumulative sum along spatial dimension (last dim)
        cum_error = torch.cumsum(error, dim=-1) * dx  # (B, nt, nx)

        # W1 = L1 norm of the antiderivative, averaged over time and batch
        w1_per_slice = torch.mean(torch.abs(cum_error), dim=-1) * dx  # (B, nt)
        loss = torch.mean(w1_per_slice)

        return loss, {"wasserstein": loss.item()}
