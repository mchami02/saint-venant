"""Hybrid loss for HybridDeepONet combining grid, trajectory, and PDE losses.

This module provides the combined loss function for training HybridDeepONet:

1. Grid MSE Loss: Match predicted output_grid to target
2. Rankine-Hugoniot Residual (CORRECTED): Minimize R_RH = ẋ_s(u+ - u-) - (f(u+) - f(u-))
3. PDE Residual: Enforce conservation law in smooth regions
4. Existence Regularization: Prevent existence collapse

The key correction from the original RankineHugoniotLoss is that we no longer
match trajectories to analytical solutions. Instead, we compute the actual
Rankine-Hugoniot residual using the predicted shock velocity and the density
values sampled from region predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.pde_residual import PDEResidualLoss, create_shock_mask


def greenshields_flux(rho: torch.Tensor) -> torch.Tensor:
    """Greenshields flux function: f(ρ) = ρ(1 - ρ).

    Args:
        rho: Density tensor.

    Returns:
        Flux values.
    """
    return rho * (1.0 - rho)


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


def sample_density_at_shocks_vectorized(
    region_densities: torch.Tensor,
    positions: torch.Tensor,
    x_coords: torch.Tensor,
    max_d: int,
    epsilon: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample density values on both sides of all shocks (vectorized).

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
    # Create index tensors
    d_indices = torch.arange(max_d, device=device)  # (max_d,)

    # Gather left regions: region_densities[:, d, :, :] for each d
    # Result shape: (B, max_d, nt, nx)
    left_indices = d_indices.view(1, max_d, 1, 1).expand(B, -1, nt, nx)
    regions_left = torch.gather(region_densities, 1, left_indices)

    # Gather right regions: region_densities[:, d+1, :, :] for each d
    right_indices = (d_indices + 1).view(1, max_d, 1, 1).expand(B, -1, nt, nx)
    regions_right = torch.gather(region_densities, 1, right_indices)

    # Reshape for batched grid_sample: treat (B, max_d) as batch dimension
    # regions: (B, max_d, nt, nx) -> (B*max_d, 1, nt, nx)
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


def rankine_hugoniot_residual(
    positions: torch.Tensor,
    existence: torch.Tensor,
    region_densities: torch.Tensor,
    x_coords: torch.Tensor,
    disc_mask: torch.Tensor,
    dt: float = 0.004,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Compute Rankine-Hugoniot residual loss (vectorized).

    The RH condition states: ẋ_s = [f(u+) - f(u-)] / [u+ - u-]
    Rearranging: R_RH = ẋ_s · (u+ - u-) - (f(u+) - f(u-)) = 0

    This loss minimizes the squared residual weighted by shock existence.

    Args:
        positions: Predicted shock positions (B, D, T).
        existence: Predicted shock existence (B, D, T).
        region_densities: Per-region density predictions (B, K, nt, nx).
        x_coords: Spatial coordinates (B, 1, nt, nx).
        disc_mask: Validity mask for discontinuities (B, D).
        dt: Time step size.
        epsilon: Small offset for density sampling.

    Returns:
        Scalar RH residual loss.
    """
    B, D, T = positions.shape
    K = region_densities.shape[1]  # Number of regions
    device = positions.device

    # Only process discontinuities that have corresponding regions
    # Region d is to the left of shock d, region d+1 is to the right
    max_d = min(D, K - 1)

    if max_d == 0:
        return torch.tensor(0.0, device=device)

    # Compute shock velocities: (B, D, T)
    velocities = compute_shock_velocity(positions, dt)

    # Sample densities on both sides of all shocks (vectorized)
    # u_minus, u_plus: (B, max_d, T)
    u_minus, u_plus = sample_density_at_shocks_vectorized(
        region_densities, positions, x_coords, max_d, epsilon
    )

    # Compute flux: (B, max_d, T)
    f_minus = greenshields_flux(u_minus)
    f_plus = greenshields_flux(u_plus)

    # RH residual: ẋ_s · (u+ - u-) - (f+ - f-)
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

    return total_residual / n_valid


class HybridDeepONetLoss(nn.Module):
    """Combined loss for HybridDeepONet training.

    Combines:
    1. Grid MSE: Match output_grid to target
    2. RH Residual: Enforce Rankine-Hugoniot at shocks (CORRECTED)
    3. Smooth Region Loss: Either PDE residual (unsupervised) or supervised MSE
    4. IC Loss: Match initial condition at t=0 (included in PDE loss)
    5. Existence Regularization: Prevent collapse

    Args:
        grid_weight: Weight for grid MSE loss.
        rh_weight: Weight for Rankine-Hugoniot residual loss.
        smooth_weight: Weight for smooth region loss (PDE residual or supervised).
        reg_weight: Weight for existence regularization.
        ic_weight: Weight for initial condition loss (higher = prioritize IC).
        smooth_loss_type: Type of loss for smooth regions:
            - "pde_residual" (default): Unsupervised physics-informed loss
            - "supervised": MSE between prediction and target in smooth regions
        dt: Time step size.
        dx: Spatial step size.
        shock_buffer: Buffer around shocks for smooth region masking.
        epsilon: Offset for density sampling in RH loss.
    """

    def __init__(
        self,
        grid_weight: float = 1.0,
        rh_weight: float = 1.0,
        smooth_weight: float = 0.1,
        reg_weight: float = 0.01,
        ic_weight: float = 10.0,
        smooth_loss_type: str = "pde_residual",
        dt: float = 0.004,
        dx: float = 0.02,
        shock_buffer: float = 0.05,
        epsilon: float = 0.01,
    ):
        super().__init__()
        self.grid_weight = grid_weight
        self.rh_weight = rh_weight
        self.smooth_weight = smooth_weight
        self.reg_weight = reg_weight
        self.ic_weight = ic_weight
        self.smooth_loss_type = smooth_loss_type
        self.dt = dt
        self.dx = dx
        self.shock_buffer = shock_buffer
        self.epsilon = epsilon

        # PDE residual loss module (includes IC loss) - only used if smooth_loss_type is "pde_residual"
        if smooth_loss_type == "pde_residual":
            self.pde_loss = PDEResidualLoss(
                dt=dt, dx=dx, shock_buffer=shock_buffer, ic_weight=ic_weight
            )
        else:
            self.pde_loss = None

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        batch_input: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss and components.

        Args:
            model_output: Dict containing:
                - 'positions': (B, D, T) predicted positions
                - 'existence': (B, D, T) existence probabilities
                - 'output_grid': (B, 1, nt, nx) assembled grid
                - 'region_densities': (B, K, nt, nx) per-region predictions
                - 'region_weights': (B, K, nt, nx) region assignments
            batch_input: Dict containing:
                - 'discontinuities': (B, D, 3)
                - 'disc_mask': (B, D)
                - 'x_coords': (B, 1, nt, nx)
            target: Target density grid (B, 1, nt, nx).

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        positions = model_output["positions"]
        existence = model_output["existence"]
        output_grid = model_output["output_grid"]
        region_densities = model_output["region_densities"]

        disc_mask = batch_input["disc_mask"]
        x_coords = batch_input["x_coords"]

        # 1. Grid MSE Loss
        loss_grid = F.mse_loss(output_grid, target)

        # 2. Rankine-Hugoniot Residual Loss (CORRECTED)
        loss_rh = rankine_hugoniot_residual(
            positions,
            existence,
            region_densities,
            x_coords,
            disc_mask,
            self.dt,
            self.epsilon,
        )

        # 3. Smooth Region Loss (either PDE residual or supervised MSE)
        if self.smooth_loss_type == "pde_residual":
            # Unsupervised physics-informed loss in smooth regions
            loss_smooth, smooth_components = self.pde_loss(
                output_grid, positions, existence, x_coords, disc_mask, target
            )
            smooth_loss_name = "smooth_pde"
            ic_loss_val = smooth_components["ic_loss"]
        else:
            # Supervised MSE loss in smooth regions only
            # Create mask for smooth regions (1 = smooth, 0 = near shock)
            if x_coords.dim() == 4:
                x_coords_3d = x_coords.squeeze(1)  # (B, nt, nx)
            else:
                x_coords_3d = x_coords

            mask = create_shock_mask(
                positions, existence, x_coords_3d, disc_mask, self.shock_buffer
            )  # (B, nt, nx)

            # Compute masked supervised loss
            output_squeezed = output_grid.squeeze(1)  # (B, nt, nx)
            target_squeezed = target.squeeze(1)  # (B, nt, nx)
            squared_error = (output_squeezed - target_squeezed) ** 2
            masked_error = squared_error * mask
            loss_smooth = masked_error.sum() / mask.sum().clamp(min=1)
            smooth_loss_name = "smooth_supervised"

            # For supervised mode, compute IC loss separately
            pred_ic = output_grid[:, 0, 0, :]  # (B, nx)
            true_ic = target[:, 0, 0, :]  # (B, nx)
            ic_loss_val = F.mse_loss(pred_ic, true_ic).item()

        # 4. Existence Regularization
        # Encourage non-trivial existence predictions
        mask_exp = disc_mask.unsqueeze(-1)  # (B, D, 1)
        masked_exist = existence * mask_exp
        n_valid = disc_mask.sum() * existence.shape[-1]
        if n_valid > 0:
            mean_exist = masked_exist.sum() / n_valid
            loss_reg = (mean_exist - 0.5) ** 2
        else:
            loss_reg = torch.tensor(0.0, device=positions.device)

        # Combine losses
        total_loss = (
            self.grid_weight * loss_grid
            + self.rh_weight * loss_rh
            + self.smooth_weight * loss_smooth
            + self.reg_weight * loss_reg
        )

        components = {
            "total": total_loss.item(),
            "grid": loss_grid.item(),
            "rh_residual": loss_rh.item(),
            smooth_loss_name: loss_smooth.item(),
            "ic_loss": ic_loss_val,
            "existence_reg": loss_reg.item()
            if isinstance(loss_reg, torch.Tensor)
            else loss_reg,
        }

        return total_loss, components


def build_hybrid_loss(args: dict) -> HybridDeepONetLoss:
    """Build HybridDeepONetLoss from configuration dict.

    Args:
        args: Configuration dictionary with optional keys:
            - grid_weight: Weight for grid MSE (default 1.0)
            - rh_weight: Weight for RH residual (default 1.0)
            - smooth_weight: Weight for smooth region loss (default 0.1)
            - reg_weight: Weight for regularization (default 0.01)
            - ic_weight: Weight for IC loss (default 10.0)
            - smooth_loss_type: "pde_residual" or "supervised" (default "pde_residual")
            - dt: Time step size (default 0.004)
            - dx: Spatial step size (default 0.02)
            - shock_buffer: Buffer for smooth region masking (default 0.05)
            - epsilon: Offset for density sampling (default 0.01)

    Returns:
        Configured HybridDeepONetLoss instance.
    """
    return HybridDeepONetLoss(
        grid_weight=args.get("grid_weight", 1.0),
        rh_weight=args.get("rh_weight", 1.0),
        smooth_weight=args.get("smooth_weight", 0.1),
        reg_weight=args.get("reg_weight", 0.01),
        ic_weight=args.get("ic_weight", 10.0),
        smooth_loss_type=args.get("smooth_loss_type", "pde_residual"),
        dt=args.get("dt", 0.004),
        dx=args.get("dx", 0.02),
        shock_buffer=args.get("shock_buffer", 0.05),
        epsilon=args.get("epsilon", 0.01),
    )
