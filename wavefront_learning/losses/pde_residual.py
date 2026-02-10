"""PDE residual loss for smooth regions in LWR traffic flow.

This module provides a physics-informed loss that enforces the conservation
law in smooth regions between shock waves:

    drho/dt + df(rho)/dx = 0

where f(rho) = rho(1 - rho) is the Greenshields flux.

The loss is computed using central finite differences and excludes points
within a buffer distance from predicted shock locations to avoid penalizing
physically valid discontinuities.
"""

import torch

from .base import BaseLoss
from .flux import greenshields_flux


def compute_pde_residual(
    density: torch.Tensor,
    dt: float,
    dx: float,
) -> torch.Tensor:
    """Compute PDE residual using central finite differences.

    Computes R = drho/dt + df(rho)/dx using:
        drho/dt = (rho[t+1] - rho[t-1]) / (2*dt)
        df/dx = (f(rho[x+1]) - f(rho[x-1])) / (2*dx)

    Args:
        density: Density grid of shape (B, nt, nx).
        dt: Time step size.
        dx: Spatial step size.

    Returns:
        PDE residual of shape (B, nt-2, nx-2) for interior points.
    """
    # Compute flux
    flux = greenshields_flux(density)

    # drho/dt using central difference in time
    # (rho[t+1, x] - rho[t-1, x]) / (2*dt)
    drho_dt = (density[:, 2:, 1:-1] - density[:, :-2, 1:-1]) / (2.0 * dt)

    # df/dx using central difference in space
    # (f[t, x+1] - f[t, x-1]) / (2*dx)
    df_dx = (flux[:, 1:-1, 2:] - flux[:, 1:-1, :-2]) / (2.0 * dx)

    # PDE residual: should be zero in smooth regions
    residual = drho_dt + df_dx

    return residual


def create_shock_mask(
    positions: torch.Tensor,
    existence: torch.Tensor,
    x_coords: torch.Tensor,
    disc_mask: torch.Tensor,
    buffer: float = 0.05,
) -> torch.Tensor:
    """Create mask for points away from shocks (vectorized).

    Creates a mask that is 1 for points that are sufficiently far from
    all predicted shock locations (to exclude shock regions from PDE loss).

    Args:
        positions: Predicted shock positions (B, D, T).
        existence: Predicted shock existence (B, D, T).
        x_coords: Spatial coordinates (B, nt, nx) or (B, 1, nt, nx).
        disc_mask: Validity mask for discontinuities (B, D).
        buffer: Buffer distance around shocks to exclude.

    Returns:
        Mask of shape (B, nt, nx) where 1 = valid (away from shock).
    """
    # Handle x_coords shape
    if x_coords.dim() == 4:
        x_coords = x_coords.squeeze(1)  # (B, nt, nx)

    B, nt, nx = x_coords.shape
    D = positions.shape[1]

    # Expand dimensions for broadcasting
    # positions: (B, D, T) -> (B, D, T, 1)
    # x_coords: (B, nt, nx) -> (B, 1, nt, nx)
    x_shock_exp = positions.unsqueeze(-1)  # (B, D, T, 1)
    x_coords_exp = x_coords.unsqueeze(1)  # (B, 1, nt, nx)

    # Distance from each shock: (B, D, nt, nx)
    distance = torch.abs(x_coords_exp - x_shock_exp)

    # Points within buffer of shock: (B, D, nt, nx)
    near_shock = distance < buffer

    # Modulate by existence: (B, D, T) -> (B, D, T, 1)
    exist = (existence > 0.5).unsqueeze(-1)  # (B, D, nt, 1)
    near_shock = near_shock & exist

    # Apply discontinuity mask: (B, D) -> (B, D, 1, 1)
    dmask = disc_mask.view(B, D, 1, 1).bool()
    near_shock = near_shock & dmask

    # Combine: a point is near ANY shock if any d has near_shock=True
    # near_any_shock: (B, nt, nx)
    near_any_shock = near_shock.any(dim=1)

    # Mask is 1 where NOT near any shock
    mask = (~near_any_shock).float()

    return mask


class PDEShockResidualLoss(BaseLoss):
    """PDE residual loss on the ground truth, weighted by distance to shocks.

    Computes drho/dt + df(rho)/dx on the ground truth density (non-zero at
    actual shocks), then weights each cell's squared residual by its distance
    to the nearest predicted discontinuity. Cells near a predicted shock
    contribute little; cells far from any prediction contribute more.

    This provides a smooth gradient signal (vs. a hard binary mask) that
    rewards the model for moving predicted trajectories toward actual shocks.

    Args:
        dt: Time step size.
        dx: Spatial step size.
    """

    def __init__(
        self,
        dt: float = 0.004,
        dx: float = 0.02,
    ):
        super().__init__()
        self.dt = dt
        self.dx = dx

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute distance-weighted PDE residual of the ground truth.

        Args:
            input_dict: Must contain:
                - 'x_coords': (B, 1, nt, nx) spatial coordinates
                - 'disc_mask': (B, D) validity mask
            output_dict: Must contain:
                - 'positions': (B, D, T) predicted shock positions
              Optionally contains:
                - 'existence': (B, D, T) existence probabilities
            target: Ground truth grid (B, 1, nt, nx).

        Returns:
            Tuple of (loss, components_dict) where components_dict
            contains 'pde_shock_residual' and 'total'.
        """
        positions = output_dict["positions"]
        existence = output_dict.get("existence", torch.ones_like(positions))
        x_coords = input_dict["x_coords"]
        disc_mask = input_dict["disc_mask"]

        # Squeeze channel dimension from ground truth
        gt_density = target.squeeze(1)  # (B, nt, nx)

        # Compute PDE residual on GT (interior points only)
        residual = compute_pde_residual(gt_density, self.dt, self.dx)

        # Prepare interior spatial coordinates
        if x_coords.dim() == 4:
            x_coords_3d = x_coords.squeeze(1)
        else:
            x_coords_3d = x_coords
        x_interior = x_coords_3d[:, 1:-1, 1:-1]  # (B, nt-2, nx-2)

        # Interior positions/existence (trim boundary time steps)
        pos_int = positions[:, :, 1:-1]  # (B, D, nt-2)
        exist_int = existence[:, :, 1:-1]  # (B, D, nt-2)

        # Combine disc_mask and existence: if disc doesn't exist, same as shock doesn't exist
        B, D = disc_mask.shape
        combined = disc_mask.view(B, D, 1).float() * exist_int  # (B, D, nt-2)

        # Compute distance from each cell to each predicted position
        # pos_int: (B, D, nt-2) -> (B, D, nt-2, 1)
        # x_interior: (B, nt-2, nx-2) -> (B, 1, nt-2, nx-2)
        dist = torch.abs(
            x_interior.unsqueeze(1) - pos_int.unsqueeze(-1)
        )  # (B, D, nt-2, nx-2)

        # Per cell: min over discontinuities of dist / (combined + eps)
        # Inactive shocks (combined ≈ 0) produce large scores, ignored by min
        eps = 1e-6
        min_score = (dist / (combined.unsqueeze(-1) + eps)).min(dim=1).values  # (B, nt-2, nx-2)

        # Weight PDE residual by min_score
        loss = (residual**2 * min_score).mean()

        components = {
            "pde_shock_residual": loss.item(),
            "total": loss.item(),
        }
        return loss, components


class PDEResidualLoss(BaseLoss):
    """PDE residual loss for conservation law on the predicted grid.

    Enforces drho/dt + df(rho)/dx = 0 using central finite differences
    over all interior points. No shock masking is applied.

    Optionally includes an initial condition (IC) loss that penalizes
    deviation from the target IC at t=0.

    Args:
        dt: Time step size.
        dx: Spatial step size.
        ic_weight: Weight for IC loss (higher = prioritize IC accuracy).
    """

    def __init__(
        self,
        dt: float = 0.004,
        dx: float = 0.02,
        ic_weight: float = 0.0,
    ):
        super().__init__()
        self.dt = dt
        self.dx = dx
        self.ic_weight = ic_weight

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute PDE residual loss with optional IC loss.

        Args:
            input_dict: Not used by this loss.
            output_dict: Must contain:
                - 'output_grid': (B, 1, nt, nx) predicted grid
            target: Target grid (B, 1, nt, nx) for optional IC loss.

        Returns:
            Tuple of (total_loss, components_dict) where components_dict
            contains 'pde_residual', 'ic' (if ic_weight > 0), and 'total'.
        """
        output_grid = output_dict["output_grid"]

        # Squeeze channel dimension
        density = output_grid.squeeze(1)  # (B, nt, nx)

        # Compute PDE residual (interior points only)
        residual = compute_pde_residual(density, self.dt, self.dx)  # (B, nt-2, nx-2)

        pde_loss = (residual**2).mean()

        # Compute IC loss if weight > 0
        components = {
            "pde_residual": pde_loss.item(),
        }

        total_loss = pde_loss

        if self.ic_weight > 0:
            # Extract IC at t=0: (B, 1, nt, nx) -> (B, nx)
            pred_ic = output_grid[:, 0, 0, :]  # (B, nx)
            true_ic = target[:, 0, 0, :]  # (B, nx)
            ic_loss_val = torch.nn.functional.mse_loss(pred_ic, true_ic)
            total_loss = pde_loss + self.ic_weight * ic_loss_val
            components["ic"] = ic_loss_val.item()

        components["total"] = total_loss.item()

        return total_loss, components
