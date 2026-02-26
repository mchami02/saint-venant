"""Shock proximity loss for ShockAwareDeepONet.

Combines solution MSE with proximity field MSE, weighted by proximity_weight.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss


class ShockProximityLoss(BaseLoss):
    """Combined loss: solution MSE + weighted proximity MSE.

    Args:
        proximity_weight: Weight for the proximity MSE term (default: 0.1).
    """

    def __init__(self, proximity_weight: float = 0.1):
        super().__init__()
        self.proximity_weight = proximity_weight

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            input_dict: Must contain "shock_proximity" ground truth (B, 1, nt, nx).
            output_dict: Must contain "output_grid" and "shock_proximity" predictions.
            target: Ground truth solution grid (B, 1, nt, nx).

        Returns:
            Tuple of (total loss, components dict).
        """
        solution_mse = F.mse_loss(output_dict["output_grid"], target)

        prox_pred = output_dict["shock_proximity"]
        prox_gt = input_dict["shock_proximity"]
        proximity_mse = F.mse_loss(prox_pred, prox_gt)

        total = solution_mse + self.proximity_weight * proximity_mse

        return total, {
            "solution_mse": solution_mse.item(),
            "proximity_mse": proximity_mse.item(),
            "total": total.item(),
        }
