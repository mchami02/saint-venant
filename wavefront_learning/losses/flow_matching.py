"""Flow matching loss for latent diffusion training."""

import torch
import torch.nn.functional as F

from .base import BaseLoss


class FlowMatchingLoss(BaseLoss):
    """MSE between predicted and target velocity for flow matching.

    Expected output_dict keys:
        - predicted_velocity: (B, latent_dim)
        - target_velocity: (B, latent_dim)
    """

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred_v = output_dict["predicted_velocity"]
        target_v = output_dict["target_velocity"]
        loss = F.mse_loss(pred_v, target_v)
        return loss, {"flow_matching_mse": loss.item()}
