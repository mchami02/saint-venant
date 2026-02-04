"""Grid assembly components for combining region predictions."""

import torch
import torch.nn as nn


class GridAssembler(nn.Module):
    """Assembles full solution grid from region predictions using soft boundaries.

    Uses soft sigmoid boundaries at predicted shock locations to compute
    differentiable region weights. The final grid is a weighted sum of
    region predictions.

    For D discontinuities, there are K = D + 1 regions:
    - Region 0: left of first shock
    - Region k (1 <= k < K-1): between shock k-1 and shock k
    - Region K-1: right of last shock

    Args:
        sigma: Softness parameter for sigmoid boundaries (smaller = sharper).
    """

    def __init__(self, sigma: float = 0.02):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        region_densities: torch.Tensor,
        positions: torch.Tensor,
        existence: torch.Tensor,
        x_coords: torch.Tensor,
        disc_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble grid from region predictions.

        Args:
            region_densities: Predicted densities per region (B, K, nt, nx).
            positions: Predicted shock positions (B, D, T).
            existence: Predicted shock existence (B, D, T).
            x_coords: Spatial coordinates (B, nt, nx) or (B, nx).
            disc_mask: Validity mask for discontinuities (B, D).

        Returns:
            Tuple of:
                - output_grid: Assembled solution grid (B, 1, nt, nx)
                - region_weights: Soft region assignments (B, K, nt, nx)
        """
        B, K, nt, nx = region_densities.shape
        D = K - 1  # Number of potential discontinuities
        device = region_densities.device

        # Ensure x_coords has shape (B, nt, nx)
        if x_coords.dim() == 2:
            # (B, nx) -> (B, 1, nx) -> (B, nt, nx)
            x_coords = x_coords.unsqueeze(1).expand(-1, nt, -1)
        elif x_coords.dim() == 4:
            # (B, 1, nt, nx) -> (B, nt, nx)
            x_coords = x_coords.squeeze(1)

        # Compute soft indicators for being left of each shock
        # left_of_shock[d] = sigmoid((x_shock[d] - x) / sigma)
        # Shape: (B, D, nt, nx)
        left_of_shock = torch.zeros(B, D, nt, nx, device=device)

        for d in range(D):
            # Shock position at each time: (B, T) where T = nt
            x_shock = positions[:, d, :]  # (B, nt)
            # Expand to (B, nt, nx)
            x_shock_exp = x_shock.unsqueeze(-1).expand(-1, -1, nx)

            # Soft indicator: 1 if x < x_shock, 0 if x > x_shock
            # sigmoid((x_shock - x) / sigma)
            indicator = torch.sigmoid((x_shock_exp - x_coords) / self.sigma)

            # Modulate by existence probability
            exist = existence[:, d, :].unsqueeze(-1)  # (B, nt, 1)
            # When shock doesn't exist, indicator should be all 1s (no boundary)
            # indicator_modulated = exist * indicator + (1 - exist) * 1
            indicator = exist * indicator + (1 - exist)

            # Apply discontinuity mask: if discontinuity is not valid, no boundary
            mask_d = disc_mask[:, d].view(B, 1, 1)  # (B, 1, 1)
            indicator = mask_d * indicator + (1 - mask_d)

            left_of_shock[:, d, :, :] = indicator

        # Compute region weights
        # Region k is to the right of shock k-1 and to the left of shock k
        region_weights = torch.zeros(B, K, nt, nx, device=device)

        for k in range(K):
            if k == 0:
                # Region 0: left of first shock (or all if D=0)
                weight = left_of_shock[:, 0, :, :] if D > 0 else torch.ones_like(
                    region_densities[:, 0, :, :]
                )
            elif k == K - 1:
                # Region K-1: right of last shock
                weight = 1.0 - left_of_shock[:, k - 1, :, :]
            else:
                # Region k: between shock k-1 and shock k
                # right of shock k-1 AND left of shock k
                right_of_prev = 1.0 - left_of_shock[:, k - 1, :, :]
                left_of_curr = left_of_shock[:, k, :, :]
                weight = right_of_prev * left_of_curr

            region_weights[:, k, :, :] = weight

        # Normalize weights (they should already sum to ~1, but ensure it)
        weight_sum = region_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        region_weights = region_weights / weight_sum

        # Assemble output grid as weighted sum of region predictions
        output_grid = (region_densities * region_weights).sum(dim=1)  # (B, nt, nx)

        # Add channel dimension
        output_grid = output_grid.unsqueeze(1)  # (B, 1, nt, nx)

        return output_grid, region_weights
