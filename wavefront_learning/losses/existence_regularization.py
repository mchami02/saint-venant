"""Existence regularization loss.

Prevents existence predictions from collapsing to all 0s or all 1s.
"""

import torch

from .base import BaseLoss


class ExistenceRegularizationLoss(BaseLoss):
    """Regularization to prevent existence from collapsing to 0 or 1.

    Encourages existence predictions to be varied rather than all zeros/ones
    by penalizing deviation from a target mean.

    Args:
        target_mean: Target mean existence value (default 0.5).
    """

    def __init__(self, target_mean: float = 0.5):
        super().__init__()
        self.target_mean = target_mean

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute existence regularization loss.

        Args:
            input_dict: Must contain:
                - 'disc_mask': (B, D) validity mask
            output_dict: Must contain:
                - 'existence': (B, D, T) existence probabilities
            target: Target tensor (unused).

        Returns:
            Tuple of (loss tensor, components dict with 'existence_reg' key).
        """
        predicted_existence = output_dict["existence"]
        mask = input_dict["disc_mask"]

        mask_exp = mask.unsqueeze(-1)  # (B, D, 1)
        masked_exist = predicted_existence * mask_exp

        n_valid = mask.sum() * predicted_existence.shape[-1]
        if n_valid > 0:
            mean_exist = masked_exist.sum() / n_valid
            loss = (mean_exist - self.target_mean) ** 2
        else:
            loss = torch.tensor(0.0, device=predicted_existence.device)

        return loss, {"existence_reg": loss.item()}
