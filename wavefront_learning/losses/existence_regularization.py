"""Existence regularization loss with IC anchoring.

Enforces that if a discontinuity is predicted to exist at t=0,
its trajectory must start at the actual IC discontinuity position.
"""

import torch

from .base import BaseLoss


class ExistenceRegularizationLoss(BaseLoss):
    """IC anchoring constraint for existence predictions.

    Enforces that predicted trajectories start at the correct IC positions
    when existence at t=0 is high. This creates a soft constraint:
    "if you claim it exists, place it correctly at t=0".

    The loss is weighted by existence probability at t=0, so:
    - High existence (~1) -> full penalty for position errors
    - Low existence (~0) -> position errors are ignored
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute IC anchoring loss.

        Args:
            input_dict: Must contain:
                - 'discontinuities': (B, D, 3) with [x_IC, rho_L, rho_R]
                - 'disc_mask': (B, D) validity mask
            output_dict: Must contain:
                - 'positions': (B, D, T) predicted positions
                - 'existence': (B, D, T) existence probabilities
            target: Target tensor (unused).

        Returns:
            Tuple of (loss tensor, components dict with 'existence_reg' key).
        """
        positions = output_dict["positions"]  # (B, D, T)
        existence = output_dict["existence"]  # (B, D, T)
        discontinuities = input_dict["discontinuities"]  # (B, D, 3)
        mask = input_dict["disc_mask"]  # (B, D)

        # Extract t=0 values
        pred_pos_t0 = positions[:, :, 0]  # (B, D)
        exist_t0 = existence[:, :, 0]  # (B, D)
        ic_pos = discontinuities[:, :, 0]  # (B, D)

        # Position error at t=0, weighted by existence
        pos_error = (pred_pos_t0 - ic_pos) ** 2  # (B, D)
        weighted_error = exist_t0 * pos_error  # (B, D)

        # Apply validity mask
        masked_error = weighted_error * mask

        n_valid = mask.sum()
        if n_valid > 0:
            loss = masked_error.sum() / n_valid
        else:
            loss = torch.tensor(0.0, device=positions.device)

        return loss, {"existence_reg": loss.item()}
