"""Boundary loss for shock trajectories.

Penalizes predicted positions outside the spatial domain.
"""

import torch

from .base import BaseLoss


class BoundaryLoss(BaseLoss):
    """Loss penalizing predicted positions outside the domain.

    Applies a quadratic penalty on how far positions exceed the domain
    boundaries [domain_min, domain_max], regardless of existence.

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
            target: Target tensor (unused).

        Returns:
            Tuple of (loss tensor, components dict with 'boundary' key).
        """
        predicted_positions = output_dict["positions"]
        mask = input_dict["disc_mask"]

        # Quadratic penalty for how far outside the domain
        below = torch.clamp(self.domain_min - predicted_positions, min=0.0)
        above = torch.clamp(predicted_positions - self.domain_max, min=0.0)
        penalty = below**2 + above**2  # (B, D, T)

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
