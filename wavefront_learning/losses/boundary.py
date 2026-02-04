"""Boundary loss for shock trajectories.

Penalizes existence of shocks outside the spatial domain.
"""

import torch

from .base import BaseLoss


class BoundaryLoss(BaseLoss):
    """Loss penalizing existence of shocks outside the domain.

    When a shock exits the domain [domain_min, domain_max], its existence
    probability should be 0.

    Args:
        domain_min: Minimum domain boundary (default 0.0).
        domain_max: Maximum domain boundary (default 1.0).
    """

    def __init__(self, domain_min: float = 0.0, domain_max: float = 1.0):
        super().__init__()
        self.domain_min = domain_min
        self.domain_max = domain_max

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute boundary loss.

        Args:
            input_dict: Must contain:
                - 'disc_mask': (B, D) validity mask
            output_dict: Must contain:
                - 'positions': (B, D, T) predicted positions
                - 'existence': (B, D, T) existence probabilities
            target: Target tensor (unused).

        Returns:
            Tuple of (loss tensor, components dict with 'boundary' key).
        """
        predicted_positions = output_dict["positions"]
        predicted_existence = output_dict["existence"]
        mask = input_dict["disc_mask"]

        # Check which positions are outside the domain
        outside = (predicted_positions < self.domain_min) | (
            predicted_positions > self.domain_max
        )
        outside = outside.float()  # (B, D, T)

        # Penalize existence when outside
        penalty = outside * (predicted_existence**2)

        # Apply mask
        mask_exp = mask.unsqueeze(-1)  # (B, D, 1)
        masked_penalty = penalty * mask_exp

        # Average over valid entries
        n_valid = mask.sum() * predicted_positions.shape[-1]
        if n_valid > 0:
            loss = masked_penalty.sum() / n_valid
        else:
            loss = torch.tensor(0.0, device=predicted_positions.device)

        return loss, {"boundary": loss.item()}
