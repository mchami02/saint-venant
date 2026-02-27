"""MSE loss computed only on non-shock cells.

Shocks are detected on the ground truth grid using the Lax entropy condition
with connected component filtering. This allows the model to focus learning
on smooth regions without being penalized for sharp shock approximations.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss
from .flux import compute_shock_speed, greenshields_flux_derivative


class MSEShockLoss(BaseLoss):
    """MSE loss excluding shock cells detected via the Lax entropy condition.

    Detects shocks on the GT grid using characteristic speed convergence,
    filters small components, expands interface mask to cell mask, and
    computes MSE only on non-shock cells.

    Args:
        dx: Spatial step size (unused, kept for interface compatibility).
        min_component_size: Minimum connected component size to keep as shock.
    """

    def __init__(self, dx: float = 0.02, min_component_size: int = 5):
        super().__init__()
        self.dx = dx
        self.min_component_size = min_component_size

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute MSE loss on non-shock cells.

        Args:
            input_dict: Input dictionary (unused for this loss).
            output_dict: Model output dict. Must contain 'output_grid' tensor.
            target: Ground truth grid (B, 1, nt, nx).

        Returns:
            Tuple of (loss, components_dict).
        """
        output_grid = output_dict["output_grid"]

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
        is_shock = (char_left > shock_speed) & (shock_speed > char_right)

        # Remove small isolated components (noise)
        if self.min_component_size > 0:
            from losses.shock_utils import filter_small_components

            device = is_shock.device
            filtered = []
            for b in range(B):
                mask_np = is_shock[b].cpu().numpy()
                mask_np = filter_small_components(mask_np, self.min_component_size)
                filtered.append(torch.from_numpy(mask_np))
            is_shock = torch.stack(filtered).to(device)

        # Expand interface mask (B, nt, nx-1) to cell mask (B, 1, nt, nx):
        # a cell is shock if either adjacent interface is shock
        pad_left = torch.zeros(B, nt, 1, dtype=torch.bool, device=is_shock.device)
        pad_right = torch.zeros(B, nt, 1, dtype=torch.bool, device=is_shock.device)
        shock_from_left = torch.cat([pad_left, is_shock], dim=2)  # (B, nt, nx)
        shock_from_right = torch.cat([is_shock, pad_right], dim=2)  # (B, nt, nx)
        is_shock_cell = (shock_from_left | shock_from_right).unsqueeze(1)  # (B, 1, nt, nx)

        # Compute MSE only on non-shock cells
        non_shock = ~is_shock_cell
        n_non_shock = non_shock.sum().item()
        n_total = non_shock.numel()
        n_shock = n_total - int(n_non_shock)

        if n_non_shock == 0:
            loss = torch.tensor(0.0, device=target.device, requires_grad=True)
        else:
            diff_sq = (output_grid - target) ** 2
            loss = diff_sq[non_shock].mean()

        return loss, {
            "mse_shock": loss.item(),
            "n_shock_cells": n_shock,
            "n_total_cells": n_total,
        }
