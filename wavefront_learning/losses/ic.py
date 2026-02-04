"""Initial condition loss.

Penalizes deviation from the target initial condition at t=0.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss


class ICLoss(BaseLoss):
    """Loss for matching initial condition at t=0.

    Ensures the predicted grid matches the target at the initial time step.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute initial condition loss.

        Args:
            input_dict: Input dictionary (unused for this loss).
            output_dict: Must contain:
                - 'output_grid': (B, 1, nt, nx) predicted grid
            target: Target grid tensor (B, 1, nt, nx).

        Returns:
            Tuple of (loss tensor, components dict with 'ic' key).
        """
        output_grid = output_dict["output_grid"]

        # Extract IC at t=0: (B, 1, nt, nx) -> (B, nx)
        pred_ic = output_grid[:, 0, 0, :]  # (B, nx)
        true_ic = target[:, 0, 0, :]  # (B, nx)

        loss = F.mse_loss(pred_ic, true_ic)

        return loss, {"ic": loss.item()}
