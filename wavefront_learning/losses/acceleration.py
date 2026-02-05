"""Acceleration-based loss for shock detection.

This loss penalizes low existence predictions where the ground truth grid
shows high acceleration, indicating a shock should be present.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss


def compute_acceleration(density: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute temporal acceleration using central finite differences.

    Args:
        density: (B, nt, nx) density grid.
        dt: Time step size.

    Returns:
        (B, nt-2, nx) acceleration for interior time points (indices 1 to nt-2).
    """
    # Central difference: a = (rho(t+dt) - 2*rho(t) + rho(t-dt)) / dt^2
    return (density[:, 2:, :] - 2 * density[:, 1:-1, :] + density[:, :-2, :]) / (
        dt**2
    )


def sample_max_acceleration_at_trajectories(
    acceleration: torch.Tensor,
    positions: torch.Tensor,
    x_coords: torch.Tensor,
    epsilon: float = 0.02,
) -> torch.Tensor:
    """Sample maximum |acceleration| near predicted positions using grid_sample.

    For each predicted trajectory position, samples acceleration at x-epsilon, x,
    and x+epsilon, then takes the max of absolute values.

    Args:
        acceleration: (B, nt_interior, nx) acceleration grid (nt_interior = nt - 2).
        positions: (B, D, T) predicted shock positions.
        x_coords: (B, 1, nt, nx) or (B, nt, nx) spatial coordinates.
        epsilon: Spatial window for sampling near trajectories.

    Returns:
        (B, D, T_interior) max absolute acceleration near each trajectory point,
        where T_interior = T - 2 (matching interior time indices).
    """
    # Handle x_coords shape
    if x_coords.dim() == 4:
        x_coords = x_coords.squeeze(1)  # (B, nt, nx)

    B, nt_interior, nx = acceleration.shape
    D = positions.shape[1]
    T = positions.shape[2]
    T_interior = T - 2  # Interior time points
    device = acceleration.device

    if T_interior <= 0:
        return torch.zeros(B, D, 0, device=device)

    # Get shock positions for interior time points: (B, D, T_interior)
    x_shock = positions[:, :, 1:-1]

    # Create sampling positions: x - epsilon, x, x + epsilon
    x_left = (x_shock - epsilon).clamp(0.0, 1.0)
    x_center = x_shock.clamp(0.0, 1.0)
    x_right = (x_shock + epsilon).clamp(0.0, 1.0)

    # Get x range from coordinates
    x_min = x_coords[:, 0, 0].min()
    x_max = x_coords[:, 0, -1].max()
    x_range = x_max - x_min

    # Normalize x coordinates to [-1, 1] for grid_sample
    x_left_norm = 2.0 * (x_left - x_min) / x_range - 1.0
    x_center_norm = 2.0 * (x_center - x_min) / x_range - 1.0
    x_right_norm = 2.0 * (x_right - x_min) / x_range - 1.0

    # Time coordinates (normalized to [-1, 1]): (T_interior,)
    t_norm = torch.linspace(-1, 1, nt_interior, device=device)
    # Expand for all batches and discontinuities: (B, D, T_interior)
    t_norm_exp = t_norm.unsqueeze(0).unsqueeze(0).expand(B, D, -1)

    # Expand acceleration for batched sampling: (B, 1, nt_interior, nx)
    accel_expanded = acceleration.unsqueeze(1)
    # Replicate for each discontinuity: (B * D, 1, nt_interior, nx)
    accel_flat = accel_expanded.expand(-1, D, -1, -1).reshape(
        B * D, 1, nt_interior, nx
    )

    # Helper function to sample at given x positions
    def sample_at_x(x_norm):
        # Create sampling grid: (B, D, T_interior, 2) -> (B*D, T_interior, 1, 2)
        grid = torch.stack([x_norm, t_norm_exp], dim=-1)  # (B, D, T_interior, 2)
        grid_flat = grid.view(B * D, T_interior, 1, 2)

        # Sample using grid_sample
        sampled = F.grid_sample(
            accel_flat,
            grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B*D, 1, T_interior, 1)

        return sampled.view(B, D, T_interior)

    # Sample at all three positions
    accel_left = sample_at_x(x_left_norm)
    accel_center = sample_at_x(x_center_norm)
    accel_right = sample_at_x(x_right_norm)

    # Take max of absolute values across the three samples
    max_accel = torch.max(
        torch.max(accel_left.abs(), accel_center.abs()), accel_right.abs()
    )

    return max_accel


class AccelerationLoss(BaseLoss):
    """Penalize low existence where ground truth has high acceleration.

    This loss identifies regions in the ground truth where temporal acceleration
    is high (indicating shocks) and penalizes the model for predicting low
    existence at those locations.

    Loss formula:
        L_accel = (1/N) * sum_{b,d,t} 1(|a_near| > tau) * (1 - e_{b,d,t})^2 * m_{b,d}

    where:
        - a_near = max_{|x - x_d(t)| < epsilon} |a(t, x)|
        - e_{b,d,t} = existence probability
        - tau = acceleration threshold
        - m_{b,d} = discontinuity validity mask
        - N = count of high-acceleration points

    Args:
        dt: Time step size for acceleration computation (default: 0.004).
        accel_threshold: Threshold for "high" acceleration (default: 1.0).
        epsilon: Spatial window for sampling near trajectories (default: 0.02).
    """

    def __init__(
        self, dt: float = 0.004, accel_threshold: float = 1.0, epsilon: float = 0.02
    ):
        super().__init__()
        self.dt = dt
        self.accel_threshold = accel_threshold
        self.epsilon = epsilon

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute acceleration loss.

        Args:
            input_dict: Must contain:
                - 'x_coords': (B, 1, nt, nx) spatial coordinates
                - 'disc_mask': (B, D) validity mask
            output_dict: Must contain:
                - 'positions': (B, D, T) predicted shock positions
                - 'existence': (B, D, T) existence probabilities
            target: Target grid (B, 1, nt, nx) or (B, nt, nx).

        Returns:
            Tuple of (loss tensor, components dict with 'acceleration' key).
        """
        positions = output_dict["positions"]
        existence = output_dict["existence"]
        x_coords = input_dict["x_coords"]
        disc_mask = input_dict["disc_mask"]

        B, D, T = positions.shape
        device = positions.device

        # Handle target shape
        if target.dim() == 4:
            target_grid = target.squeeze(1)  # (B, nt, nx)
        else:
            target_grid = target  # (B, nt, nx)

        nt = target_grid.shape[1]

        # Need at least 3 time points for acceleration computation
        if T < 3 or nt < 3:
            loss = torch.tensor(0.0, device=device)
            return loss, {"acceleration": 0.0}

        # Compute acceleration from ground truth
        acceleration = compute_acceleration(target_grid, self.dt)  # (B, nt-2, nx)

        # Sample max acceleration near each predicted trajectory position
        max_accel_near = sample_max_acceleration_at_trajectories(
            acceleration, positions, x_coords, self.epsilon
        )  # (B, D, T-2)

        # Get existence for interior time points
        existence_interior = existence[:, :, 1:-1]  # (B, D, T-2)

        # Apply threshold to identify high-acceleration regions
        high_accel_mask = (max_accel_near > self.accel_threshold).float()

        # Compute penalty: high_accel * (1 - existence)^2
        penalty = high_accel_mask * (1 - existence_interior) ** 2

        # Apply discontinuity mask: (B, D) -> (B, D, 1)
        mask_d = disc_mask.unsqueeze(-1)
        penalty = penalty * mask_d

        # Sum and normalize
        total_penalty = penalty.sum()
        n_high_accel = (high_accel_mask * mask_d).sum().clamp(min=1)

        loss = total_penalty / n_high_accel

        return loss, {"acceleration": loss.item()}
