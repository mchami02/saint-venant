"""Collision loss for shock trajectories.

Penalizes simultaneous existence of colliding shocks, encouraging the model
to predict shock merging.
"""

import torch

from .base import BaseLoss


class CollisionLoss(BaseLoss):
    """Loss penalizing simultaneous existence of colliding shocks.

    When two shocks are very close (within threshold), at most one should exist.
    This encourages the model to predict shock merging.

    Args:
        collision_threshold: Distance threshold for collision detection (default 0.02).
    """

    def __init__(self, collision_threshold: float = 0.02):
        super().__init__()
        self.collision_threshold = collision_threshold

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute collision loss.

        Args:
            input_dict: Must contain:
                - 'disc_mask': (B, D) validity mask
            output_dict: Must contain:
                - 'positions': (B, D, T) predicted positions
                - 'existence': (B, D, T) existence probabilities
            target: Target tensor (unused).

        Returns:
            Tuple of (loss tensor, components dict with 'collision' key).
        """
        predicted_positions = output_dict["positions"]
        predicted_existence = output_dict["existence"]
        mask = input_dict["disc_mask"]

        B, D, T = predicted_positions.shape

        if D < 2:
            loss = torch.tensor(0.0, device=predicted_positions.device)
            return loss, {"collision": 0.0}

        # Create pairwise distance matrix
        pos_i = predicted_positions.unsqueeze(2)  # (B, D, 1, T)
        pos_j = predicted_positions.unsqueeze(1)  # (B, 1, D, T)

        # Pairwise distances: (B, D, D, T)
        distances = torch.abs(pos_i - pos_j)

        # Collision indicator: (B, D, D, T)
        colliding = (distances < self.collision_threshold).float()

        # Pairwise existence product: (B, D, D, T)
        exist_i = predicted_existence.unsqueeze(2)  # (B, D, 1, T)
        exist_j = predicted_existence.unsqueeze(1)  # (B, 1, D, T)
        exist_product = exist_i * exist_j

        # Penalty: (B, D, D, T)
        penalty = colliding * exist_product

        # Pairwise mask: both must be valid
        mask_i = mask.unsqueeze(2)  # (B, D, 1)
        mask_j = mask.unsqueeze(1)  # (B, 1, D)
        pair_mask = mask_i * mask_j  # (B, D, D)

        # Apply mask
        masked_penalty = penalty * pair_mask.unsqueeze(-1)  # (B, D, D, T)

        # Only count upper triangle (i < j) to avoid double counting
        triu_mask = torch.triu(
            torch.ones(D, D, device=predicted_positions.device), diagonal=1
        )
        triu_mask = triu_mask.view(1, D, D, 1)  # (1, D, D, 1)

        masked_penalty = masked_penalty * triu_mask

        # Compute total and count
        total_loss = masked_penalty.sum()
        n_pairs = (pair_mask.unsqueeze(-1) * triu_mask).sum() * T

        if n_pairs > 0:
            loss = total_loss / n_pairs
        else:
            loss = total_loss

        return loss, {"collision": loss.item()}
