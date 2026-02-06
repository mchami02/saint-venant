"""Trajectory regularization loss.

Penalizes erratic trajectory behavior by limiting large spatial jumps
between consecutive timesteps for the same discontinuity.
"""

import torch

from .base import BaseLoss


class RegularizeTrajLoss(BaseLoss):
    """Penalize large spatial jumps between consecutive timesteps.

    For each discontinuity, computes the position difference between
    consecutive time steps and penalizes jumps exceeding a threshold.
    Weighted by existence probability so that non-existent shocks
    are not penalized.

    Loss formula:
        delta = |x(t+1) - x(t)|
        penalty = max(0, delta - max_step)^2
        L = (1/N) * sum_{b,d,t} penalty * e_min(b,d,t) * m(b,d)

    where:
        - e_min = min(existence(t), existence(t+1))
        - m(b,d) = discontinuity validity mask
        - N = number of valid (discontinuity, time-pair) entries

    Args:
        max_step: Maximum allowed position change per timestep before
            penalty applies (default: 0.05).
    """

    def __init__(self, max_step: float = 0.05):
        super().__init__()
        self.max_step = max_step

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute trajectory regularization loss.

        Args:
            input_dict: Must contain:
                - 'disc_mask': (B, D) validity mask
            output_dict: Must contain:
                - 'positions': (B, D, T) predicted positions
                - 'existence': (B, D, T) existence probabilities
            target: Target tensor (unused).

        Returns:
            Tuple of (loss tensor, components dict with 'regularize_traj' key).
        """
        positions = output_dict["positions"]  # (B, D, T)
        existence = output_dict.get("existence")  # (B, D, T) or None
        mask = input_dict["disc_mask"]  # (B, D)

        B, D, T = positions.shape
        device = positions.device

        if T < 2:
            loss = torch.tensor(0.0, device=device)
            return loss, {"regularize_traj": 0.0}

        # Consecutive position differences: (B, D, T-1)
        delta = (positions[:, :, 1:] - positions[:, :, :-1]).abs()

        # Penalize only jumps exceeding the threshold
        excess = torch.clamp(delta - self.max_step, min=0.0)
        penalty = excess**2  # (B, D, T-1)

        # Weight by minimum existence of the consecutive pair (if available)
        if existence is not None:
            exist_min = torch.minimum(
                existence[:, :, :-1], existence[:, :, 1:]
            )  # (B, D, T-1)
            penalty = penalty * exist_min

        # Apply discontinuity mask: (B, D) -> (B, D, 1)
        mask_exp = mask.unsqueeze(-1)
        penalty = penalty * mask_exp

        # Average over valid entries
        n_valid = mask.sum() * (T - 1)
        if n_valid > 0:
            loss = penalty.sum() / n_valid
        else:
            loss = torch.tensor(0.0, device=device)

        return loss, {"regularize_traj": loss.item()}
