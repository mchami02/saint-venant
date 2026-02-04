"""HybridDeepONet: Combined trajectory and region prediction for wavefront learning.

This module implements a hybrid DeepONet architecture that predicts both:
1. Shock trajectories (positions and existence over time)
2. Density values in each region between shocks

The model uses a shared branch network (DiscontinuityEncoder) with:
- Trajectory trunk (TimeEncoder + TrajectoryDecoder) for shock prediction
- Region trunks (SpaceTimeEncoder + RegionTrunk) for density prediction

The final output grid is assembled by combining region predictions using
soft sigmoid boundaries at predicted shock locations.

Architecture:
    Shared Branch: DiscontinuityEncoder (transformer-based)
    Trajectory Trunk: TimeEncoder + TrajectoryDecoder
    Region Trunks: K = max_disc + 1 RegionTrunks with shared SpaceTimeEncoder
    Grid Assembly: Soft region assignment using sigmoid boundaries
"""

import torch
import torch.nn as nn

from models.region_trunk import RegionTrunkSet, SpaceTimeEncoder
from models.shock_trajectory_net import (
    DiscontinuityEncoder,
    TimeEncoder,
    TrajectoryDecoder,
)


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


class HybridDeepONet(nn.Module):
    """Hybrid DeepONet for combined trajectory and density prediction.

    This model predicts shock trajectories AND the full solution grid by:
    1. Encoding discontinuities using a shared transformer branch
    2. Predicting trajectories using time encoder + decoder
    3. Predicting per-region densities using space-time encoder + region trunks
    4. Assembling the final grid using soft region boundaries

    Args:
        hidden_dim: Hidden dimension for all networks.
        max_discontinuities: Maximum number of discontinuities to handle.
        num_frequencies_t: Number of Fourier frequencies for time encoding.
        num_frequencies_x: Number of Fourier frequencies for space encoding.
        num_disc_layers: Number of transformer layers in discontinuity encoder.
        num_time_layers: Number of MLP layers in time encoder.
        num_coord_layers: Number of MLP layers in coordinate encoder.
        num_res_blocks: Number of residual blocks in decoders.
        num_heads: Number of attention heads in discontinuity encoder.
        dropout: Dropout rate.
        sigma: Softness parameter for region boundaries.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        max_discontinuities: int = 3,
        num_frequencies_t: int = 32,
        num_frequencies_x: int = 16,
        num_disc_layers: int = 2,
        num_time_layers: int = 3,
        num_coord_layers: int = 3,
        num_res_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        sigma: float = 0.02,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_discontinuities = max_discontinuities
        self.num_regions = max_discontinuities + 1

        # Shared branch: discontinuity encoder
        self.branch = DiscontinuityEncoder(
            input_dim=3,  # [x, rho_L, rho_R]
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_disc_layers,
            dropout=dropout,
        )

        # Trajectory trunk: time encoder
        self.time_encoder = TimeEncoder(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_frequencies=num_frequencies_t,
            num_layers=num_time_layers,
        )

        # Trajectory decoder
        self.trajectory_decoder = TrajectoryDecoder(
            branch_dim=hidden_dim,
            trunk_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

        # Region trunk: space-time encoder (shared across regions)
        self.coord_encoder = SpaceTimeEncoder(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_frequencies_t=num_frequencies_t,
            num_frequencies_x=num_frequencies_x,
            num_layers=num_coord_layers,
        )

        # Region trunks: one per region
        self.region_trunks = RegionTrunkSet(
            num_regions=self.num_regions,
            branch_dim=hidden_dim,
            coord_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

        # Grid assembler
        self.grid_assembler = GridAssembler(sigma=sigma)

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch_input: Dict containing:
                - 'discontinuities': (B, D, 3) with [x, rho_L, rho_R]
                - 'disc_mask': (B, D) validity mask
                - 't_coords': (B, 1, nt, nx) time coordinates
                - 'x_coords': (B, 1, nt, nx) space coordinates

        Returns:
            Dict containing:
                - 'positions': (B, D, T) predicted x-positions
                - 'existence': (B, D, T) existence probability
                - 'output_grid': (B, 1, nt, nx) assembled solution grid
                - 'region_densities': (B, K, nt, nx) per-region predictions
                - 'region_weights': (B, K, nt, nx) soft region assignments
        """
        discontinuities = batch_input["discontinuities"]
        disc_mask = batch_input["disc_mask"]
        t_coords = batch_input["t_coords"]  # (B, 1, nt, nx)
        x_coords = batch_input["x_coords"]  # (B, 1, nt, nx)

        B = discontinuities.shape[0]
        nt, nx = t_coords.shape[2], t_coords.shape[3]

        # Squeeze coordinate tensors
        t_coords_3d = t_coords.squeeze(1)  # (B, nt, nx)
        x_coords_3d = x_coords.squeeze(1)  # (B, nt, nx)

        # === SHARED BRANCH ===
        # Encode discontinuities
        branch_emb = self.branch(discontinuities, disc_mask)  # (B, D, hidden_dim)

        # Create pooled branch embedding for region trunks
        # Masked mean pooling
        mask_exp = disc_mask.unsqueeze(-1)  # (B, D, 1)
        branch_sum = (branch_emb * mask_exp).sum(dim=1)  # (B, hidden_dim)
        n_valid = disc_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        branch_pooled = branch_sum / n_valid  # (B, hidden_dim)

        # === TRAJECTORY TRUNK ===
        # Get query times (use first row of t_coords)
        query_times = t_coords_3d[:, :, 0]  # (B, nt)

        # Encode times
        trunk_emb = self.time_encoder(query_times)  # (B, T, hidden_dim)

        # Decode trajectories
        traj_output = self.trajectory_decoder(
            branch_emb, trunk_emb, disc_mask
        )  # {positions, existence}

        # Clamp positions to valid grid domain [0, 1]
        positions = torch.clamp(traj_output["positions"], 0.0, 1.0)
        existence = traj_output["existence"]

        # === REGION TRUNKS ===
        # Encode (t, x) coordinates
        coord_emb = self.coord_encoder(t_coords_3d, x_coords_3d)  # (B, nt, nx, hidden)

        # Predict density for each region
        region_densities = self.region_trunks(
            branch_pooled, coord_emb
        )  # (B, K, nt, nx)

        # === GRID ASSEMBLY ===
        output_grid, region_weights = self.grid_assembler(
            region_densities,
            positions,
            existence,
            x_coords_3d,
            disc_mask,
        )

        return {
            "positions": positions,
            "existence": existence,
            "output_grid": output_grid,
            "region_densities": region_densities,
            "region_weights": region_weights,
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_hybrid_deeponet(args: dict) -> HybridDeepONet:
    """Build HybridDeepONet from configuration dict.

    Args:
        args: Configuration dictionary with optional keys:
            - hidden_dim: Hidden dimension (default 128)
            - max_discontinuities: Max discontinuities (default 3)
            - num_frequencies_t: Time Fourier frequencies (default 32)
            - num_frequencies_x: Space Fourier frequencies (default 16)
            - num_disc_layers: Discontinuity encoder layers (default 2)
            - num_time_layers: Time encoder layers (default 3)
            - num_coord_layers: Coordinate encoder layers (default 3)
            - num_res_blocks: Residual blocks (default 2)
            - num_heads: Attention heads (default 4)
            - dropout: Dropout rate (default 0.1)
            - sigma: Boundary softness (default 0.02)

    Returns:
        Configured HybridDeepONet instance.
    """
    return HybridDeepONet(
        hidden_dim=args.get("hidden_dim", 32),
        max_discontinuities=args.get("max_discontinuities", 10),
        num_frequencies_t=args.get("num_frequencies_t", 8),
        num_frequencies_x=args.get("num_frequencies_x", 8),
        num_disc_layers=args.get("num_disc_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_coord_layers=args.get("num_coord_layers", 2),
        num_res_blocks=args.get("num_res_blocks", 2),
        num_heads=args.get("num_heads", 8),
        dropout=args.get("dropout", 0.05),
        sigma=args.get("sigma", 0.02),
    )
