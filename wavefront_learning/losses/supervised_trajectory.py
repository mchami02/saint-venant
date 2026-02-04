"""Supervised trajectory loss when ground truth is available.

Provides direct supervision on trajectory positions and existence.
"""

import torch
import torch.nn as nn

from .base import BaseLoss


class SupervisedTrajectoryLoss(BaseLoss):
    """Supervised loss for trajectory prediction when ground truth is available.

    Can be combined with physics losses for semi-supervised training.

    Args:
        position_weight: Weight for position MSE loss (default 1.0).
        existence_weight: Weight for existence BCE loss (default 1.0).
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        existence_weight: float = 1.0,
    ):
        super().__init__()
        self.position_weight = position_weight
        self.existence_weight = existence_weight
        self.mse = nn.MSELoss(reduction="none")
        self.bce = nn.BCELoss(reduction="none")

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute supervised trajectory loss.

        Args:
            input_dict: Must contain:
                - 'disc_mask': (B, D) validity mask
                - 'target_positions': (B, D, T) ground truth positions (optional)
                - 'target_existence': (B, D, T) ground truth existence (optional)
            output_dict: Must contain:
                - 'positions': (B, D, T) predicted positions
                - 'existence': (B, D, T) existence probabilities
            target: If target_positions/existence not in input_dict, this should
                    be a dict with these keys, or unused.

        Returns:
            Tuple of (loss tensor, components dict).
        """
        positions = output_dict["positions"]
        existence = output_dict["existence"]
        mask = input_dict["disc_mask"]

        # Get target positions/existence from input_dict or target
        if "target_positions" in input_dict:
            target_positions = input_dict["target_positions"]
            target_existence = input_dict["target_existence"]
        elif isinstance(target, dict):
            target_positions = target["positions"]
            target_existence = target["existence"]
        else:
            # Cannot compute supervised loss without targets
            loss = torch.tensor(0.0, device=positions.device)
            return loss, {"position": 0.0, "existence": 0.0}

        # Position MSE loss
        pos_error = self.mse(positions, target_positions)
        mask_exp = mask.unsqueeze(-1)
        pos_loss = (pos_error * mask_exp).sum() / (
            mask.sum() * positions.shape[-1] + 1e-8
        )

        # Existence BCE loss
        exist_error = self.bce(existence, target_existence)
        exist_loss = (exist_error * mask_exp).sum() / (
            mask.sum() * existence.shape[-1] + 1e-8
        )

        total_loss = self.position_weight * pos_loss + self.existence_weight * exist_loss

        components = {
            "position": pos_loss.item(),
            "existence": exist_loss.item(),
        }

        return total_loss, components
