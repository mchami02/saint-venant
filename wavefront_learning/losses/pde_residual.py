"""PDE residual loss for smooth regions in LWR traffic flow.

This module provides a physics-informed loss that enforces the conservation
law in smooth regions between shock waves:

    ∂ρ/∂t + ∂f(ρ)/∂x = 0

where f(ρ) = ρ(1 - ρ) is the Greenshields flux.

The loss is computed using central finite differences and excludes points
within a buffer distance from predicted shock locations to avoid penalizing
physically valid discontinuities.
"""

import torch
import torch.nn as nn


def greenshields_flux(rho: torch.Tensor) -> torch.Tensor:
    """Greenshields flux function: f(ρ) = ρ(1 - ρ).

    Args:
        rho: Density tensor.

    Returns:
        Flux values.
    """
    return rho * (1.0 - rho)


def greenshields_flux_derivative(rho: torch.Tensor) -> torch.Tensor:
    """Derivative of Greenshields flux: f'(ρ) = 1 - 2ρ.

    Args:
        rho: Density tensor.

    Returns:
        Flux derivative values.
    """
    return 1.0 - 2.0 * rho


def compute_pde_residual(
    density: torch.Tensor,
    dt: float,
    dx: float,
) -> torch.Tensor:
    """Compute PDE residual using central finite differences.

    Computes R = ∂ρ/∂t + ∂f(ρ)/∂x using:
        ∂ρ/∂t ≈ (ρ[t+1] - ρ[t-1]) / (2·dt)
        ∂f/∂x ≈ (f(ρ[x+1]) - f(ρ[x-1])) / (2·dx)

    Args:
        density: Density grid of shape (B, nt, nx).
        dt: Time step size.
        dx: Spatial step size.

    Returns:
        PDE residual of shape (B, nt-2, nx-2) for interior points.
    """
    # Compute flux
    flux = greenshields_flux(density)

    # ∂ρ/∂t using central difference in time
    # (rho[t+1, x] - rho[t-1, x]) / (2*dt)
    drho_dt = (density[:, 2:, 1:-1] - density[:, :-2, 1:-1]) / (2.0 * dt)

    # ∂f/∂x using central difference in space
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
    """Create mask for points away from shocks.

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
    device = positions.device

    # Start with all ones (all points valid)
    mask = torch.ones(B, nt, nx, device=device)

    for d in range(D):
        # Shock position at each time: (B, T)
        x_shock = positions[:, d, :]  # (B, nt)

        # Expand for comparison: (B, nt, nx)
        x_shock_exp = x_shock.unsqueeze(-1).expand(-1, -1, nx)

        # Distance from shock: (B, nt, nx)
        distance = torch.abs(x_coords - x_shock_exp)

        # Points within buffer of shock
        near_shock = distance < buffer

        # Modulate by existence: if shock doesn't exist, don't exclude
        exist = existence[:, d, :].unsqueeze(-1) > 0.5  # (B, nt, 1)
        near_shock = near_shock & exist

        # Apply discontinuity mask
        dmask = disc_mask[:, d].view(B, 1, 1).bool()  # (B, 1, 1)
        near_shock = near_shock & dmask

        # Update mask: exclude points near this shock
        mask = mask * (~near_shock).float()

    return mask


class PDEResidualLoss(nn.Module):
    """PDE residual loss for conservation law in smooth regions.

    Enforces ∂ρ/∂t + ∂f(ρ)/∂x = 0 using central finite differences,
    with shock masking to exclude points near predicted discontinuities.

    Args:
        dt: Time step size.
        dx: Spatial step size.
        shock_buffer: Buffer distance around shocks to exclude.
    """

    def __init__(
        self,
        dt: float = 0.004,
        dx: float = 0.02,
        shock_buffer: float = 0.05,
    ):
        super().__init__()
        self.dt = dt
        self.dx = dx
        self.shock_buffer = shock_buffer

    def forward(
        self,
        output_grid: torch.Tensor,
        positions: torch.Tensor,
        existence: torch.Tensor,
        x_coords: torch.Tensor,
        disc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PDE residual loss.

        Args:
            output_grid: Predicted density grid (B, 1, nt, nx).
            positions: Predicted shock positions (B, D, T).
            existence: Predicted shock existence (B, D, T).
            x_coords: Spatial coordinates (B, 1, nt, nx).
            disc_mask: Validity mask for discontinuities (B, D).

        Returns:
            Scalar PDE residual loss.
        """
        # Squeeze channel dimension
        density = output_grid.squeeze(1)  # (B, nt, nx)

        # Compute PDE residual (interior points only)
        residual = compute_pde_residual(density, self.dt, self.dx)  # (B, nt-2, nx-2)

        # Create shock mask for interior points
        # x_coords for interior points
        if x_coords.dim() == 4:
            x_coords_3d = x_coords.squeeze(1)
        else:
            x_coords_3d = x_coords

        # Interior x coordinates
        x_interior = x_coords_3d[:, 1:-1, 1:-1]  # (B, nt-2, nx-2)

        # Create mask (using interior coordinates)
        mask = create_shock_mask(
            positions[:, :, 1:-1],  # Interior time points
            existence[:, :, 1:-1],
            x_interior,
            disc_mask,
            self.shock_buffer,
        )

        # Apply mask and compute mean squared residual
        masked_residual = residual * mask
        n_valid = mask.sum().clamp(min=1)

        loss = (masked_residual**2).sum() / n_valid

        return loss


def pde_residual_loss(
    output_grid: torch.Tensor,
    positions: torch.Tensor,
    existence: torch.Tensor,
    x_coords: torch.Tensor,
    disc_mask: torch.Tensor,
    dt: float = 0.004,
    dx: float = 0.02,
    shock_buffer: float = 0.05,
) -> torch.Tensor:
    """Functional interface for PDE residual loss.

    Args:
        output_grid: Predicted density grid (B, 1, nt, nx).
        positions: Predicted shock positions (B, D, T).
        existence: Predicted shock existence (B, D, T).
        x_coords: Spatial coordinates (B, 1, nt, nx).
        disc_mask: Validity mask for discontinuities (B, D).
        dt: Time step size.
        dx: Spatial step size.
        shock_buffer: Buffer distance around shocks to exclude.

    Returns:
        Scalar PDE residual loss.
    """
    loss_fn = PDEResidualLoss(dt=dt, dx=dx, shock_buffer=shock_buffer)
    return loss_fn(output_grid, positions, existence, x_coords, disc_mask)
