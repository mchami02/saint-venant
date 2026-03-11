"""PDE residual losses for the ARZ traffic flow system.

ARZ system (conservative form):
    drho/dt + d(rho*v)/dx = 0           (mass conservation)
    d(rho*w)/dt + d(rho*w*v)/dx = 0     (momentum), w = v + rho^gamma

Residuals are computed using central finite differences on interior points.
"""

import torch

from .base import BaseLoss


def compute_arz_pde_residual(
    rho: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    dx: float,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ARZ PDE residuals using central finite differences.

    Args:
        rho: Density grid (B, nt, nx).
        v: Velocity grid (B, nt, nx).
        dt: Time step size.
        dx: Spatial step size.
        gamma: Pressure exponent.

    Returns:
        (R1, R2) each of shape (B, nt-2, nx-2) for interior points.
        R1: mass conservation residual.
        R2: momentum conservation residual.
    """
    # Riemann invariant w = v + rho^gamma
    w = v + rho.pow(gamma)

    # Mass flux: rho * v
    mass_flux = rho * v

    # Momentum conservative variable: rho * w
    rho_w = rho * w

    # Momentum flux: rho * w * v
    mom_flux = rho_w * v

    # R1: drho/dt + d(rho*v)/dx
    drho_dt = (rho[:, 2:, 1:-1] - rho[:, :-2, 1:-1]) / (2.0 * dt)
    dmf_dx = (mass_flux[:, 1:-1, 2:] - mass_flux[:, 1:-1, :-2]) / (2.0 * dx)
    R1 = drho_dt + dmf_dx

    # R2: d(rho*w)/dt + d(rho*w*v)/dx
    drho_w_dt = (rho_w[:, 2:, 1:-1] - rho_w[:, :-2, 1:-1]) / (2.0 * dt)
    dmom_dx = (mom_flux[:, 1:-1, 2:] - mom_flux[:, 1:-1, :-2]) / (2.0 * dx)
    R2 = drho_w_dt + dmom_dx

    return R1, R2


class ARZPDEResidualLoss(BaseLoss):
    """PDE residual loss for ARZ conservation laws on the predicted grid.

    Enforces both mass and momentum conservation:
        loss = mean(R1^2 + R2^2)

    Args:
        dt: Time step size.
        dx: Spatial step size.
        gamma: ARZ pressure exponent.
    """

    def __init__(
        self,
        dt: float = 0.004,
        dx: float = 0.02,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.dt = dt
        self.dx = dx
        self.gamma = gamma

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute ARZ PDE residual loss on predicted grid.

        Args:
            input_dict: Not used.
            output_dict: Must contain 'output_grid' of shape (B, 2, nt, nx).
            target: Not used (loss is unsupervised).

        Returns:
            (loss, components_dict).
        """
        output_grid = output_dict["output_grid"]
        rho = output_grid[:, 0]  # (B, nt, nx)
        v = output_grid[:, 1]  # (B, nt, nx)

        R1, R2 = compute_arz_pde_residual(rho, v, self.dt, self.dx, self.gamma)

        pde_loss = (R1.pow(2) + R2.pow(2)).mean()

        return pde_loss, {
            "arz_pde_residual": pde_loss.item(),
            "total": pde_loss.item(),
        }


class ARZPDEShockResidualLoss(BaseLoss):
    """PDE residual loss on ARZ ground truth, weighted by distance to shocks.

    Same distance-weighting pattern as PDEShockResidualLoss but adapted for
    the 2-variable ARZ system. Computes both mass and momentum residuals on
    the ground truth and uses R_combined^2 = R1^2 + R2^2 per cell.

    Args:
        dt: Time step size.
        dx: Spatial step size.
        gamma: ARZ pressure exponent.
        fp_weight: Weight for false-positive shock penalty.
    """

    def __init__(
        self,
        dt: float = 0.004,
        dx: float = 0.02,
        gamma: float = 1.0,
        fp_weight: float = 1.0,
    ):
        super().__init__()
        self.dt = dt
        self.dx = dx
        self.gamma = gamma
        self.fp_weight = fp_weight

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute distance-weighted ARZ PDE residual of the ground truth.

        Args:
            input_dict: Must contain 'x_coords' and 'disc_mask'.
            output_dict: Must contain 'positions', optionally 'existence'.
            target: Ground truth (B, 2, nt, nx).

        Returns:
            (loss, components_dict).
        """
        positions = output_dict["positions"]
        existence = output_dict.get("existence", torch.ones_like(positions))
        x_coords = input_dict["x_coords"]
        disc_mask = input_dict["disc_mask"]

        # Extract rho and v from ground truth
        gt_rho = target[:, 0]  # (B, nt, nx)
        gt_v = target[:, 1]  # (B, nt, nx)

        # Compute PDE residuals on GT
        R1, R2 = compute_arz_pde_residual(
            gt_rho, gt_v, self.dt, self.dx, self.gamma
        )
        # Combined squared residual per cell
        residual_sq = R1.pow(2) + R2.pow(2)  # (B, nt-2, nx-2)

        # Prepare interior spatial coordinates
        if x_coords.dim() == 4:
            x_coords_3d = x_coords.squeeze(1)
        else:
            x_coords_3d = x_coords
        x_interior = x_coords_3d[:, 1:-1, 1:-1]  # (B, nt-2, nx-2)

        # Interior positions/existence
        pos_int = positions[:, :, 1:-1]  # (B, D, nt-2)
        exist_int = existence[:, :, 1:-1]  # (B, D, nt-2)

        B, D = disc_mask.shape
        combined = disc_mask.view(B, D, 1).float() * exist_int  # (B, D, nt-2)

        # Distance from each cell to each predicted position
        dist = torch.abs(
            x_interior.unsqueeze(1) - pos_int.unsqueeze(-1)
        )  # (B, D, nt-2, nx-2)

        eps = 1.0

        # Miss penalty: find nearest predicted shock per cell
        min_score = (dist / (combined.unsqueeze(-1) + eps)).min(dim=1).values
        miss_loss = (residual_sq * min_score).mean()

        # False positive penalty: find nearest high-residual cell per shock
        fp_score = (dist / (residual_sq.unsqueeze(1) + eps)).min(dim=-1).values
        fp_loss = (combined * fp_score).mean()

        loss = miss_loss + self.fp_weight * fp_loss

        return loss, {
            "arz_pde_shock_miss": miss_loss.item(),
            "arz_pde_shock_fp": fp_loss.item(),
            "arz_pde_shock_residual": loss.item(),
            "total": loss.item(),
        }
