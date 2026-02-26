"""Shock proximity MSE loss.

Computes MSE between predicted and ground truth shock proximity fields.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss


class ShockProximityLoss(BaseLoss):
    """MSE loss on the shock proximity field.

    Compares the model's predicted shock proximity with the ground truth
    proximity precomputed from the Lax entropy condition.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute proximity MSE.

        Args:
            input_dict: Must contain "shock_proximity" GT (B, 1, nt, nx).
            output_dict: Must contain "shock_proximity" prediction (B, 1, nt, nx).
            target: Unused (ground truth solution grid).

        Returns:
            Tuple of (loss, components dict).
        """
        prox_pred = output_dict["shock_proximity"]
        prox_gt = input_dict["shock_proximity"]
        loss = F.mse_loss(prox_pred, prox_gt)

        return loss, {"proximity_mse": loss.item()}
