"""MSE loss for grid predictions."""

import torch
import torch.nn.functional as F

from .base import BaseLoss


class MSELoss(BaseLoss):
    """Mean squared error loss for grid predictions.

    Computes MSE between predicted output_grid and target grid.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute MSE loss.

        Args:
            input_dict: Input dictionary (unused for this loss).
            output_dict: Model output dict. Must contain 'output_grid' tensor.
            target: Target grid tensor.

        Returns:
            Tuple of (loss tensor, components dict with 'mse' key).
        """
        output_grid = output_dict["output_grid"]
        loss = F.mse_loss(output_grid, target)

        return loss, {"mse": loss.item()}
