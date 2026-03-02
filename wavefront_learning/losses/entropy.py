"""Entropy condition loss for shock detection in LWR traffic flow.

Uses a continuous entropy strength measure derived from the Lax entropy
condition on the ground truth grid. The entropy strength is:

    margin_left  = char_left - shock_speed    (positive when left condition holds)
    margin_right = shock_speed - char_right   (positive when right condition holds)
    entropy_strength = relu(margin_left) * relu(margin_right)

For Greenshields flux, both margins equal (rho_R - rho_L), so
entropy_strength = relu(rho_R - rho_L)^2. This is:
- Zero in smooth regions and at rarefactions (rho_L > rho_R)
- Continuously positive at shock interfaces, proportional to jump strength squared
- Differentiable everywhere

The loss penalizes:
1. Missed shocks: entropy-weighted distance from interfaces to nearest prediction
2. False positives: predicted trajectories far from entropy-strong interfaces
"""

import torch

from .base import BaseLoss
from .flux import compute_shock_speed, greenshields_flux_derivative


class EntropyConditionLoss(BaseLoss):
    """Loss based on continuous entropy strength from the Lax condition on GT.

    Uses continuous entropy strength (product of ReLU margins from the Lax
    entropy condition) instead of a binary shock mask. This provides smooth
    gradients that pull trajectories toward shocks from a distance.

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

        # --- Compute continuous entropy strength at every interface ---
        rho_left = gt[:, :, :-1]  # (B, nt, nx-1)
        rho_right = gt[:, :, 1:]  # (B, nt, nx-1)

        char_left = greenshields_flux_derivative(rho_left)
        char_right = greenshields_flux_derivative(rho_right)
        shock_speed = compute_shock_speed(rho_left, rho_right)

        # Continuous entropy strength: relu(margin_left) * relu(margin_right)
        margin_left = char_left - shock_speed
        margin_right = shock_speed - char_right
        entropy_strength = torch.relu(margin_left) * torch.relu(margin_right)  # (B, nt, nx-1)

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

        # --- Miss penalty: entropy-strong interfaces far from predictions ---
        # For each interface, find nearest active predicted trajectory
        # Inactive predictions (combined ~ 0) get large score via division
        min_dist_to_pred = (
            dist / (combined.unsqueeze(-1) + eps)
        ).min(dim=1).values  # (B, nt, nx-1)

        # Weight by continuous entropy strength and average
        miss_loss = (entropy_strength * min_dist_to_pred).mean()

        # --- False positive penalty: predictions far from entropy-strong interfaces ---
        # For each predicted position, find nearest entropy-strong interface
        # Weak-entropy interfaces get large score via division
        fp_score = (
            dist / (entropy_strength.unsqueeze(1) + eps)
        ).min(dim=-1).values  # (B, D, nt)

        fp_loss = (combined * fp_score).mean()

        loss = miss_loss + self.fp_weight * fp_loss

        components = {
            "entropy_miss": miss_loss.item(),
            "entropy_fp": fp_loss.item(),
            "entropy": loss.item(),
            "total": loss.item(),
        }
        return loss, components
