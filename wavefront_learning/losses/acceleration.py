"""Acceleration-based loss for shock detection.

This loss penalizes low existence predictions where the ground truth grid
shows high acceleration, indicating a shock should be present.

The optional "missed shock" term scans the entire domain for high-acceleration
points and penalizes those not covered by any nearby prediction with high existence.
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
    return (density[:, 2:, :] - 2 * density[:, 1:-1, :] + density[:, :-2, :]) / (dt**2)


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
    accel_flat = accel_expanded.expand(-1, D, -1, -1).reshape(B * D, 1, nt_interior, nx)

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


def compute_trajectory_coverage(
    acceleration: torch.Tensor,
    positions: torch.Tensor,
    existence: torch.Tensor,
    x_coords: torch.Tensor,
    disc_mask: torch.Tensor,
    buffer: float = 0.02,
) -> torch.Tensor:
    """Compute coverage of spatial points by predicted trajectories.

    For each point in the acceleration grid, computes the maximum coverage
    from any predicted trajectory, where coverage is weighted by existence
    and discontinuity mask.

    Args:
        acceleration: (B, nt-2, nx) acceleration grid (for shape reference).
        positions: (B, D, T) predicted shock positions.
        existence: (B, D, T) existence probabilities.
        x_coords: (B, 1, nt, nx) or (B, nt, nx) spatial coordinates.
        disc_mask: (B, D) validity mask for discontinuities.
        buffer: Distance threshold for considering a point "covered".

    Returns:
        (B, nt-2, nx) coverage values in [0, 1] for each spatial point.
    """
    # Handle x_coords shape
    if x_coords.dim() == 4:
        x_coords = x_coords.squeeze(1)  # (B, nt, nx)

    B, nt_interior, nx = acceleration.shape
    D = positions.shape[1]
    device = acceleration.device

    # Extract interior time points to match acceleration grid
    x_coords_interior = x_coords[:, 1:-1, :]  # (B, nt-2, nx)
    positions_interior = positions[:, :, 1:-1]  # (B, D, T-2)
    existence_interior = existence[:, :, 1:-1]  # (B, D, T-2)

    T_interior = positions_interior.shape[2]

    if T_interior == 0 or T_interior != nt_interior:
        # Mismatch in time dimensions, return zero coverage
        return torch.zeros(B, nt_interior, nx, device=device)

    # Expand dimensions for broadcasting
    # positions: (B, D, T) -> (B, D, T, 1)
    # x_coords: (B, nt, nx) -> (B, 1, nt, nx)
    x_shock_exp = positions_interior.unsqueeze(-1)  # (B, D, T-2, 1)
    x_coords_exp = x_coords_interior.unsqueeze(1)  # (B, 1, nt-2, nx)

    # Distance from each shock: (B, D, nt-2, nx)
    distance = torch.abs(x_coords_exp - x_shock_exp)

    # Points within buffer of shock: (B, D, nt-2, nx)
    within_buffer = (distance < buffer).float()

    # Weight by existence: (B, D, T-2) -> (B, D, T-2, 1)
    exist_weight = existence_interior.unsqueeze(-1)  # (B, D, T-2, 1)
    weighted_coverage = within_buffer * exist_weight

    # Apply discontinuity mask: (B, D) -> (B, D, 1, 1)
    dmask = disc_mask.view(B, D, 1, 1)
    weighted_coverage = weighted_coverage * dmask

    # Max coverage across all discontinuities: (B, nt-2, nx)
    coverage = weighted_coverage.max(dim=1)[0]

    return coverage


class AccelerationLoss(BaseLoss):
    """Penalize low existence where ground truth has high acceleration.

    This loss identifies regions in the ground truth where temporal acceleration
    is high (indicating shocks) and penalizes the model for predicting low
    existence at those locations.

    The loss has two components:
    1. Original: Samples acceleration at predicted trajectory positions and
       penalizes low existence where acceleration is high.
    2. Missed shock (optional): Scans the entire domain for high-acceleration
       points and penalizes those not covered by any nearby prediction with
       high existence.

    Loss formulas:
        L_accel = (1/N) * sum_{b,d,t} 1(|a_near| > tau) * (1 - e_{b,d,t})^2 * m_{b,d}

        L_missed = (1/M) * sum_{(b,t,x) in H} (1 - coverage(b,t,x))^2

        L_total = L_accel + w_missed * L_missed

    where:
        - a_near = max_{|x - x_d(t)| < epsilon} |a(t, x)|
        - e_{b,d,t} = existence probability
        - tau = acceleration threshold
        - m_{b,d} = discontinuity validity mask
        - N = count of high-acceleration points at trajectories
        - H = {(b,t,x) : |acceleration(b,t,x)| > tau} (high-acceleration points)
        - coverage(b,t,x) = max_d [existence(b,d,t) * disc_mask(b,d) * 1(|x - x_d(t)| < delta)]
        - M = |H| (count of high-acceleration points in domain)

    Args:
        dt: Time step size for acceleration computation (default: 0.004).
        accel_threshold: Threshold for "high" acceleration (default: 1.0).
        epsilon: Spatial window for sampling near trajectories (default: 0.02).
        missed_shock_weight: Weight for missed shock loss term (default: 0.0, disabled).
        missed_shock_buffer: Buffer distance for coverage computation. If None,
            defaults to epsilon.
    """

    def __init__(
        self,
        dt: float = 0.05,
        accel_threshold: float = 1.0,
        epsilon: float = 0.02,
        missed_shock_weight: float = 0.0,
        missed_shock_buffer: float | None = None,
    ):
        super().__init__()
        self.dt = dt
        self.accel_threshold = accel_threshold
        self.epsilon = epsilon
        self.missed_shock_weight = missed_shock_weight
        self.missed_shock_buffer = (
            missed_shock_buffer if missed_shock_buffer is not None else epsilon
        )

    def _compute_missed_shock_loss(
        self,
        acceleration: torch.Tensor,
        positions: torch.Tensor,
        existence: torch.Tensor,
        x_coords: torch.Tensor,
        disc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute missed shock loss term.

        Scans the entire domain for high-acceleration points and penalizes
        those not covered by any nearby prediction with high existence.

        Args:
            acceleration: (B, nt-2, nx) acceleration grid.
            positions: (B, D, T) predicted shock positions.
            existence: (B, D, T) existence probabilities.
            x_coords: (B, 1, nt, nx) or (B, nt, nx) spatial coordinates.
            disc_mask: (B, D) validity mask.

        Returns:
            Scalar loss tensor.
        """
        # Compute coverage of each spatial point by trajectories
        coverage = compute_trajectory_coverage(
            acceleration,
            positions,
            existence,
            x_coords,
            disc_mask,
            buffer=self.missed_shock_buffer,
        )  # (B, nt-2, nx)

        # Create high-acceleration mask for the entire domain
        high_accel_mask = (acceleration.abs() > self.accel_threshold).float()

        # Compute penalty: high_accel * (1 - coverage)^2
        penalty = high_accel_mask * (1 - coverage) ** 2

        # Sum and normalize by count of high-acceleration points
        total_penalty = penalty.sum()
        n_high_accel = high_accel_mask.sum().clamp(min=1)

        return total_penalty / n_high_accel

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
        existence = output_dict.get(
            "existence", torch.ones_like(positions)
        )
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

        accel_loss = total_penalty / n_high_accel

        components = {"acceleration": accel_loss.item()}
        total_loss = accel_loss

        # Add missed shock loss if enabled
        if self.missed_shock_weight > 0:
            missed_shock_loss = self._compute_missed_shock_loss(
                acceleration, positions, existence, x_coords, disc_mask
            )
            components["missed_shock"] = missed_shock_loss.item()
            total_loss = total_loss + self.missed_shock_weight * missed_shock_loss

        return total_loss, components
