"""Entropy condition loss for shock detection in LWR traffic flow.

Uses the Lax entropy condition on the ground truth grid as a threshold-free
boolean shock detector. The entropy condition checks whether characteristic
speeds converge at a cell interface:

    char_left  = f'(u_j)     > shock_speed > f'(u_{j+1}) = char_right

For Greenshields flux f(rho) = rho(1 - rho):
    char_left  = 1 - 2*u_j
    char_right = 1 - 2*u_{j+1}
    shock_speed = 1 - u_j - u_{j+1}

The loss penalizes:
1. Missed shocks: entropy-detected shock interfaces far from any prediction
2. False positives: predicted trajectories far from any entropy-detected shock
"""

import torch

from .base import BaseLoss
from .flux import compute_shock_speed, greenshields_flux_derivative


class EntropyConditionLoss(BaseLoss):
    """Loss based on the Lax entropy condition applied to the GT grid.

    Detects shocks via characteristic speed convergence and penalizes
    missed shocks and false-positive predictions.

    Args:
        dx: Spatial step size (used to compute interface midpoint positions).
        fp_weight: Weight for the false-positive penalty term.
    """

    def __init__(self, dx: float = 0.02, fp_weight: float = 1.0):
        super().__init__()
        self.dx = dx
        self.fp_weight = fp_weight

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute entropy condition loss.

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
            Tuple of (loss, components_dict).
        """
        positions = output_dict["positions"]
        existence = output_dict.get("existence", torch.ones_like(positions))
        x_coords = input_dict["x_coords"]
        disc_mask = input_dict["disc_mask"]

        # Squeeze channel dim from GT: (B, 1, nt, nx) -> (B, nt, nx)
        gt = target.squeeze(1)
        B, nt, nx = gt.shape

        # --- Compute entropy condition at every interface ---
        rho_left = gt[:, :, :-1]  # (B, nt, nx-1)
        rho_right = gt[:, :, 1:]  # (B, nt, nx-1)

        char_left = greenshields_flux_derivative(rho_left)
        char_right = greenshields_flux_derivative(rho_right)
        shock_speed = compute_shock_speed(rho_left, rho_right)

        # Lax entropy condition: char_left > shock_speed > char_right
        is_shock = (char_left > shock_speed) & (shock_speed > char_right)  # (B, nt, nx-1)

        # Jump strength as weighting for miss penalty
        jump_strength = torch.abs(rho_left - rho_right)  # (B, nt, nx-1)

        # --- Interface midpoint x-coordinates ---
        if x_coords.dim() == 4:
            x_coords_3d = x_coords.squeeze(1)  # (B, nt, nx)
        else:
            x_coords_3d = x_coords
        # Midpoints between adjacent cells
        x_mid = 0.5 * (x_coords_3d[:, :, :-1] + x_coords_3d[:, :, 1:])  # (B, nt, nx-1)

        # --- Combine disc_mask and existence ---
        D = disc_mask.shape[1]
        combined = disc_mask.view(B, D, 1).float() * existence  # (B, D, T)

        # --- Distances between predicted positions and interfaces ---
        # positions: (B, D, T) -> (B, D, T, 1)
        # x_mid: (B, nt, nx-1) -> (B, 1, nt, nx-1)
        dist = torch.abs(
            x_mid.unsqueeze(1) - positions.unsqueeze(-1)
        )  # (B, D, nt, nx-1)

        eps = 1.0

        # --- Miss penalty: entropy-detected shocks far from predictions ---
        # For each interface, find nearest active predicted trajectory
        # Inactive predictions (combined ~ 0) get large score via division
        min_dist_to_pred = (
            dist / (combined.unsqueeze(-1) + eps)
        ).min(dim=1).values  # (B, nt, nx-1)

        # Weight by jump strength and mask to shock interfaces only
        is_shock_f = is_shock.float()
        miss_loss = (jump_strength * min_dist_to_pred * is_shock_f).sum() / (
            is_shock_f.sum() + 1.0
        )

        # --- False positive penalty: predictions far from any entropy shock ---
        # For each predicted position, find nearest entropy-detected shock interface
        # Non-shock interfaces (is_shock=False) get large score via division
        fp_dist = (
            dist / (is_shock_f.unsqueeze(1) + eps)
        ).min(dim=-1).values  # (B, D, nt)

        fp_loss = (combined * fp_dist).mean()

        loss = miss_loss + self.fp_weight * fp_loss

        components = {
            "entropy_miss": miss_loss.item(),
            "entropy_fp": fp_loss.item(),
            "entropy": loss.item(),
            "total": loss.item(),
        }
        return loss, components
