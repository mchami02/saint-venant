"""IC anchoring loss.

Enforces that predicted trajectories start at the actual IC
discontinuity positions. Optionally weighted by existence probability
when the model predicts existence.
"""

import torch

from .base import BaseLoss


class ICAnchoringLoss(BaseLoss):
    """IC anchoring constraint for predicted trajectories.

    Enforces that predicted trajectories start at the correct IC positions.
    If existence probabilities are available in the output, the loss is
    weighted by existence at t=0, creating a soft constraint:
    "if you claim it exists, place it correctly at t=0".

    Without existence:
    - All valid discontinuities are penalized equally for position errors

    With existence:
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
              Optionally contains:
                - 'existence': (B, D, T) existence probabilities
            target: Target tensor (unused).

        Returns:
            Tuple of (loss tensor, components dict with 'ic_anchoring' key).
        """
        positions = output_dict["positions"]  # (B, D, T)
        discontinuities = input_dict["discontinuities"]  # (B, D, 3)
        mask = input_dict["disc_mask"]  # (B, D)

        # Extract t=0 values
        pred_pos_t0 = positions[:, :, 0]  # (B, D)
        ic_pos = discontinuities[:, :, 0]  # (B, D)

        # Position error at t=0
        pos_error = (pred_pos_t0 - ic_pos) ** 2  # (B, D)

        # Weight by existence probability if available
        if "existence" in output_dict:
            exist_t0 = output_dict["existence"][:, :, 0]  # (B, D)
            pos_error = exist_t0 * pos_error

        # Apply validity mask
        masked_error = pos_error * mask

        n_valid = mask.sum()
        if n_valid > 0:
            loss = masked_error.sum() / n_valid
        else:
            loss = torch.tensor(0.0, device=positions.device)

        return loss, {"ic_anchoring": loss.item()}
