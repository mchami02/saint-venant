"""Rankine-Hugoniot residual loss from sampled densities.

This loss computes the RH residual by sampling densities and checking that
the shock velocity satisfies:
    s = [f(u+) - f(u-)] / [u+ - u-]

Three modes are available:
- "per_region": Sample from per-region density predictions (HybridDeepONet)
- "pred": Sample from the assembled output_grid prediction
- "gt": Sample from the ground truth target grid
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss
from .flux import greenshields_flux


def compute_shock_velocity(positions: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute shock velocity from position trajectory using finite differences.

    Uses central differences for interior points and forward/backward
    differences at boundaries.

    Args:
        positions: Shock positions of shape (B, D, T).
        dt: Time step size.

    Returns:
        Shock velocities of shape (B, D, T).
    """
    T = positions.shape[-1]

    if T < 2:
        return torch.zeros_like(positions)

    velocities = torch.zeros_like(positions)

    # Central differences for interior points
    if T > 2:
        velocities[:, :, 1:-1] = (positions[:, :, 2:] - positions[:, :, :-2]) / (
            2.0 * dt
        )

    # Forward difference for first point
    velocities[:, :, 0] = (positions[:, :, 1] - positions[:, :, 0]) / dt

    # Backward difference for last point
    velocities[:, :, -1] = (positions[:, :, -1] - positions[:, :, -2]) / dt

    return velocities


def sample_density_from_grid(
    grid: torch.Tensor,
    positions: torch.Tensor,
    x_coords: torch.Tensor,
    max_d: int,
    epsilon: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample density values on both sides of shocks from a single grid.

    For each shock d, we sample from the same grid:
    - u_minus: density at x = x_shock - epsilon
    - u_plus: density at x = x_shock + epsilon

    Args:
        grid: Density grid (B, 1, nt, nx) or (B, nt, nx).
        positions: Shock positions (B, D, T).
        x_coords: Spatial coordinates (B, nt, nx) or (B, 1, nt, nx).
        max_d: Maximum discontinuity index to process.
        epsilon: Small offset from shock position for sampling.

    Returns:
        Tuple of (u_minus, u_plus), each of shape (B, max_d, T).
    """
    # Handle grid shape
    if grid.dim() == 4:
        grid = grid.squeeze(1)  # (B, nt, nx)
    B, nt, nx = grid.shape
    device = grid.device

    # Handle x_coords shape
    if x_coords.dim() == 4:
        x_coords = x_coords.squeeze(1)  # (B, nt, nx)

    # Get shock positions for all discontinuities: (B, max_d, T)
    x_shock = positions[:, :max_d, :]

    # Sample positions: (B, max_d, T)
    x_minus = (x_shock - epsilon).clamp(0.0, 1.0)
    x_plus = (x_shock + epsilon).clamp(0.0, 1.0)

    # Get x range from coordinates
    x_min = x_coords[:, 0, 0].min()
    x_max = x_coords[:, 0, -1].max()
    x_range = x_max - x_min

    # Normalize to [-1, 1] for grid_sample: (B, max_d, T)
    x_minus_norm = 2.0 * (x_minus - x_min) / x_range - 1.0
    x_plus_norm = 2.0 * (x_plus - x_min) / x_range - 1.0

    # Time coordinates (normalized to [-1, 1]): (B, T)
    t_norm = torch.linspace(-1, 1, nt, device=device).unsqueeze(0).expand(B, -1)

    # Expand t_norm for all discontinuities: (B, max_d, T)
    t_norm_exp = t_norm.unsqueeze(1).expand(-1, max_d, -1)

    # Expand grid for batched sampling: (B, 1, nt, nx)
    grid_expanded = grid.unsqueeze(1)

    # Replicate grid for each discontinuity: (B * max_d, 1, nt, nx)
    grid_flat = grid_expanded.expand(-1, max_d, -1, -1).reshape(B * max_d, 1, nt, nx)

    # Create sampling grids: (B, max_d, T, 2) -> (B*max_d, T, 1, 2)
    grid_minus = torch.stack([x_minus_norm, t_norm_exp], dim=-1)  # (B, max_d, T, 2)
    grid_minus_flat = grid_minus.view(B * max_d, nt, 1, 2)

    grid_plus = torch.stack([x_plus_norm, t_norm_exp], dim=-1)  # (B, max_d, T, 2)
    grid_plus_flat = grid_plus.view(B * max_d, nt, 1, 2)

    # Batched grid_sample
    u_minus_flat = F.grid_sample(
        grid_flat,
        grid_minus_flat,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B*max_d, 1, nt, 1)

    u_plus_flat = F.grid_sample(
        grid_flat,
        grid_plus_flat,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B*max_d, 1, nt, 1)

    # Reshape back: (B*max_d, 1, nt, 1) -> (B, max_d, T)
    u_minus = u_minus_flat.view(B, max_d, nt)
    u_plus = u_plus_flat.view(B, max_d, nt)

    return u_minus, u_plus


def sample_density_at_shocks_vectorized(
    region_densities: torch.Tensor,
    positions: torch.Tensor,
    x_coords: torch.Tensor,
    max_d: int,
    epsilon: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample density values on both sides of all shocks from per-region predictions.

    For each shock d, we sample:
    - u_minus: density from region d (left of shock) at x = x_shock - epsilon
    - u_plus: density from region d+1 (right of shock) at x = x_shock + epsilon

    Uses bilinear interpolation via grid_sample with batched operations.

    Args:
        region_densities: Per-region densities (B, K, nt, nx).
        positions: Shock positions (B, D, T).
        x_coords: Spatial coordinates (B, nt, nx) or (B, 1, nt, nx).
        max_d: Maximum discontinuity index to process (min(D, K-1)).
        epsilon: Small offset from shock position for sampling.

    Returns:
        Tuple of (u_minus, u_plus), each of shape (B, max_d, T).
    """
    B, K, nt, nx = region_densities.shape
    device = region_densities.device

    # Handle x_coords shape
    if x_coords.dim() == 4:
        x_coords = x_coords.squeeze(1)  # (B, nt, nx)

    # Get shock positions for all discontinuities: (B, max_d, T)
    x_shock = positions[:, :max_d, :]

    # Sample positions: (B, max_d, T)
    x_minus = (x_shock - epsilon).clamp(0.0, 1.0)
    x_plus = (x_shock + epsilon).clamp(0.0, 1.0)

    # Get x range from coordinates
    x_min = x_coords[:, 0, 0].min()
    x_max = x_coords[:, 0, -1].max()
    x_range = x_max - x_min

    # Normalize to [-1, 1] for grid_sample: (B, max_d, T)
    x_minus_norm = 2.0 * (x_minus - x_min) / x_range - 1.0
    x_plus_norm = 2.0 * (x_plus - x_min) / x_range - 1.0

    # Time coordinates (normalized to [-1, 1]): (B, T)
    t_norm = torch.linspace(-1, 1, nt, device=device).unsqueeze(0).expand(B, -1)

    # Expand t_norm for all discontinuities: (B, max_d, T)
    t_norm_exp = t_norm.unsqueeze(1).expand(-1, max_d, -1)

    # Gather left and right regions for each discontinuity
    # For discontinuity d: left = region d, right = region d+1
    d_indices = torch.arange(max_d, device=device)  # (max_d,)

    # Gather left regions: region_densities[:, d, :, :] for each d
    left_indices = d_indices.view(1, max_d, 1, 1).expand(B, -1, nt, nx)
    regions_left = torch.gather(region_densities, 1, left_indices)

    # Gather right regions: region_densities[:, d+1, :, :] for each d
    right_indices = (d_indices + 1).view(1, max_d, 1, 1).expand(B, -1, nt, nx)
    regions_right = torch.gather(region_densities, 1, right_indices)

    # Reshape for batched grid_sample: treat (B, max_d) as batch dimension
    regions_left_flat = regions_left.view(B * max_d, 1, nt, nx)
    regions_right_flat = regions_right.view(B * max_d, 1, nt, nx)

    # Create sampling grids: (B, max_d, T, 2) -> (B*max_d, T, 1, 2)
    grid_minus = torch.stack([x_minus_norm, t_norm_exp], dim=-1)  # (B, max_d, T, 2)
    grid_minus_flat = grid_minus.view(B * max_d, nt, 1, 2)

    grid_plus = torch.stack([x_plus_norm, t_norm_exp], dim=-1)  # (B, max_d, T, 2)
    grid_plus_flat = grid_plus.view(B * max_d, nt, 1, 2)

    # Batched grid_sample
    u_minus_flat = F.grid_sample(
        regions_left_flat,
        grid_minus_flat,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B*max_d, 1, nt, 1)

    u_plus_flat = F.grid_sample(
        regions_right_flat,
        grid_plus_flat,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B*max_d, 1, nt, 1)

    # Reshape back: (B*max_d, 1, nt, 1) -> (B, max_d, T)
    u_minus = u_minus_flat.view(B, max_d, nt)
    u_plus = u_plus_flat.view(B, max_d, nt)

    return u_minus, u_plus


class RHResidualLoss(BaseLoss):
    """Rankine-Hugoniot residual loss.

    Computes R_RH = s * (u+ - u-) - (f(u+) - f(u-)) and minimizes |R_RH|^2.

    Args:
        dt: Time step size for velocity computation.
        epsilon: Offset for density sampling near shocks.
        mode: Density sampling mode:
            - "per_region": Sample from per-region density predictions (requires
              'region_densities' in output_dict). Best for training HybridDeepONet.
            - "pred": Sample from assembled output_grid prediction (requires
              'output_grid' in output_dict). Tests RH on blended prediction.
            - "gt": Sample from ground truth target grid. Tests if predicted
              trajectories match the physics of the ground truth solution.
    """

    def __init__(
        self, dt: float = 0.004, epsilon: float = 0.01, mode: str = "per_region"
    ):
        super().__init__()
        if mode not in ("per_region", "pred", "gt"):
            raise ValueError(f"mode must be 'per_region', 'pred', or 'gt', got {mode}")
        self.dt = dt
        self.epsilon = epsilon
        self.mode = mode

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute RH residual loss.

        Args:
            input_dict: Must contain:
                - 'x_coords': (B, 1, nt, nx) spatial coordinates
                - 'disc_mask': (B, D) validity mask
            output_dict: Must contain:
                - 'positions': (B, D, T) predicted shock positions
                - 'existence': (B, D, T) existence probabilities
                - 'region_densities': (B, K, nt, nx) per-region predictions
                  (required if mode="per_region")
                - 'output_grid': (B, 1, nt, nx) assembled prediction
                  (required if mode="pred")
            target: Target grid (B, 1, nt, nx). Used if mode="gt".

        Returns:
            Tuple of (loss tensor, components dict with 'rh_residual' key).
        """
        positions = output_dict["positions"]
        existence = output_dict["existence"]
        x_coords = input_dict["x_coords"]
        disc_mask = input_dict["disc_mask"]

        B, D, T = positions.shape
        device = positions.device

        # Determine max_d based on mode
        if self.mode == "per_region":
            region_densities = output_dict["region_densities"]
            K = region_densities.shape[1]
            max_d = min(D, K - 1)
        else:
            max_d = D

        if max_d == 0:
            loss = torch.tensor(0.0, device=device)
            return loss, {"rh_residual": 0.0}

        # Compute shock velocities: (B, D, T)
        velocities = compute_shock_velocity(positions, self.dt)

        # Sample densities based on mode
        if self.mode == "per_region":
            u_minus, u_plus = sample_density_at_shocks_vectorized(
                region_densities, positions, x_coords, max_d, self.epsilon
            )
        elif self.mode == "pred":
            output_grid = output_dict["output_grid"]
            u_minus, u_plus = sample_density_from_grid(
                output_grid, positions, x_coords, max_d, self.epsilon
            )
        else:  # mode == "gt"
            u_minus, u_plus = sample_density_from_grid(
                target, positions, x_coords, max_d, self.epsilon
            )

        # Compute flux: (B, max_d, T)
        f_minus = greenshields_flux(u_minus)
        f_plus = greenshields_flux(u_plus)

        # RH residual: s * (u+ - u-) - (f+ - f-)
        v_shock = velocities[:, :max_d, :]  # (B, max_d, T)
        residual = v_shock * (u_plus - u_minus) - (f_plus - f_minus)  # (B, max_d, T)

        # Weight by existence probability: (B, max_d, T)
        exist = existence[:, :max_d, :]
        weighted_residual = exist * residual**2

        # Apply discontinuity mask: (B, max_d) -> (B, max_d, 1)
        mask_d = disc_mask[:, :max_d].unsqueeze(-1)
        weighted_residual = weighted_residual * mask_d

        # Sum and normalize
        total_residual = weighted_residual.sum()
        n_valid = (exist * mask_d).sum().clamp(min=1)

        loss = total_residual / n_valid

        return loss, {"rh_residual": loss.item()}
