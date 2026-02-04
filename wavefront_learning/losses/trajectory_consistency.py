"""Trajectory consistency loss based on Rankine-Hugoniot conditions.

This loss enforces that predicted shock trajectories match the analytical
trajectories derived from the Rankine-Hugoniot jump conditions.
"""

import torch

from .base import BaseLoss
from .flux import compute_shock_speed


def compute_analytical_trajectory(
    x_0: torch.Tensor,
    rho_L: torch.Tensor,
    rho_R: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Compute analytical shock trajectory using Rankine-Hugoniot condition.

    Args:
        x_0: Initial shock positions of shape (B, D).
        rho_L: Left densities of shape (B, D).
        rho_R: Right densities of shape (B, D).
        times: Query times of shape (B, T) or (T,).

    Returns:
        Analytical positions of shape (B, D, T).
    """
    # Compute shock speeds: (B, D)
    speeds = compute_shock_speed(rho_L, rho_R)

    # Handle 1D times
    if times.dim() == 1:
        times = times.unsqueeze(0)  # (1, T)

    # Expand for broadcasting: x_0 (B, D, 1), speeds (B, D, 1), times (B, 1, T)
    x_0 = x_0.unsqueeze(-1)  # (B, D, 1)
    speeds = speeds.unsqueeze(-1)  # (B, D, 1)
    times = times.unsqueeze(1)  # (B, 1, T)

    # Compute trajectory: x(t) = x_0 + s * t
    positions = x_0 + speeds * times  # (B, D, T)

    return positions


class TrajectoryConsistencyLoss(BaseLoss):
    """Loss enforcing Rankine-Hugoniot trajectory consistency.

    The predicted positions should match the analytical trajectory:
        x_pred(t) = x_0 + s * t
    where s = 1 - (rho_L + rho_R).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute trajectory consistency loss.

        Args:
            input_dict: Must contain:
                - 'discontinuities': (B, D, 3) with [x_0, rho_L, rho_R]
                - 't_coords': (B, 1, nt, nx) time coordinates
                - 'disc_mask': (B, D) validity mask
            output_dict: Must contain:
                - 'positions': (B, D, T) predicted positions
            target: Target tensor (unused for this loss).

        Returns:
            Tuple of (loss tensor, components dict with 'trajectory' key).
        """
        predicted_positions = output_dict["positions"]
        discontinuities = input_dict["discontinuities"]
        query_times = input_dict["t_coords"][:, 0, :, 0]  # (B, T)
        mask = input_dict["disc_mask"]

        # Extract initial conditions
        x_0 = discontinuities[..., 0]  # (B, D)
        rho_L = discontinuities[..., 1]  # (B, D)
        rho_R = discontinuities[..., 2]  # (B, D)

        # Compute analytical trajectory
        analytical_positions = compute_analytical_trajectory(
            x_0, rho_L, rho_R, query_times
        )

        # Compute squared error
        error = (predicted_positions - analytical_positions) ** 2  # (B, D, T)

        # Apply mask: only count valid discontinuities
        mask_exp = mask.unsqueeze(-1)  # (B, D, 1)
        masked_error = error * mask_exp

        # Average over valid entries
        n_valid = mask.sum() * query_times.shape[-1]
        if n_valid > 0:
            loss = masked_error.sum() / n_valid
        else:
            loss = torch.tensor(0.0, device=predicted_positions.device)

        return loss, {"trajectory": loss.item()}
