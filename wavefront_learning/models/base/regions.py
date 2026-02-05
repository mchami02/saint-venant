"""Region trunk components for HybridDeepONet.

This module provides trunk networks for predicting density values in each
region between shock waves. The regions are defined by the predicted shock
trajectories, with K = D + 1 regions for D discontinuities.
"""

import torch
import torch.nn as nn

from .blocks import ResidualBlock


class RegionTrunk(nn.Module):
    """Trunk network for predicting density in a specific region.

    Combines the branch embedding (from DiscontinuityEncoder) with
    space-time coordinate encoding to predict density values.

    Uses bilinear fusion similar to TrajectoryDecoder, then applies
    residual blocks and a final linear layer to produce scalar density.

    Args:
        branch_dim: Dimension of branch (discontinuity) embeddings.
        coord_dim: Dimension of space-time coordinate embeddings.
        hidden_dim: Hidden dimension for the trunk.
        num_res_blocks: Number of residual blocks.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        branch_dim: int = 128,
        coord_dim: int = 128,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Bilinear fusion layer
        self.bilinear = nn.Bilinear(branch_dim, coord_dim, hidden_dim)

        # Linear paths for skip connections
        self.linear_branch = nn.Linear(branch_dim, hidden_dim)
        self.linear_coord = nn.Linear(coord_dim, hidden_dim)

        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)]
        )

        # Output head: predict density value
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Density in [0, 1]
        )

    def forward(
        self,
        branch_emb: torch.Tensor,
        coord_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict density for all (t, x) points using branch embedding.

        Args:
            branch_emb: Pooled branch embedding of shape (B, branch_dim).
            coord_emb: Space-time embeddings of shape (B, nt, nx, coord_dim).

        Returns:
            Predicted density of shape (B, nt, nx).
        """
        B, nt, nx, coord_dim = coord_emb.shape

        # Expand branch embedding to match coordinate grid
        # (B, branch_dim) -> (B, nt, nx, branch_dim)
        branch_exp = branch_emb.unsqueeze(1).unsqueeze(2).expand(-1, nt, nx, -1)

        # Reshape for bilinear: (B*nt*nx, dim)
        branch_flat = branch_exp.reshape(-1, branch_exp.shape[-1])
        coord_flat = coord_emb.reshape(-1, coord_emb.shape[-1])

        # Bilinear fusion + linear paths
        fused = self.bilinear(branch_flat, coord_flat)
        fused = fused + self.linear_branch(branch_flat) + self.linear_coord(coord_flat)
        fused = self.fusion_norm(fused)

        # Residual blocks
        for block in self.res_blocks:
            fused = block(fused)

        # Predict density
        density = self.density_head(fused).squeeze(-1)  # (B*nt*nx,)

        # Reshape back to grid
        density = density.reshape(B, nt, nx)

        return density


class RegionTrunkSet(nn.Module):
    """Set of K region trunks for predicting density in all regions.

    Uses a fixed number of trunks (K = max_discontinuities + 1), where
    each trunk is responsible for predicting density in its region.
    Inactive trunks (for samples with fewer discontinuities) are masked
    during grid assembly.

    Args:
        num_regions: Maximum number of regions (K = max_disc + 1).
        branch_dim: Dimension of branch embeddings.
        coord_dim: Dimension of coordinate embeddings.
        hidden_dim: Hidden dimension for trunks.
        num_res_blocks: Number of residual blocks per trunk.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_regions: int,
        branch_dim: int = 128,
        coord_dim: int = 128,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_regions = num_regions

        # Create K region trunks
        self.trunks = nn.ModuleList(
            [
                RegionTrunk(
                    branch_dim=branch_dim,
                    coord_dim=coord_dim,
                    hidden_dim=hidden_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                )
                for _ in range(num_regions)
            ]
        )

    def forward(
        self,
        branch_emb: torch.Tensor,
        coord_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict density for all regions.

        Args:
            branch_emb: Pooled branch embedding of shape (B, branch_dim).
            coord_emb: Space-time embeddings of shape (B, nt, nx, coord_dim).

        Returns:
            Region densities of shape (B, K, nt, nx).
        """
        B, nt, nx, _ = coord_emb.shape

        # Predict density for each region
        region_densities = []
        for trunk in self.trunks:
            density = trunk(branch_emb, coord_emb)  # (B, nt, nx)
            region_densities.append(density)

        # Stack along region dimension
        return torch.stack(region_densities, dim=1)  # (B, K, nt, nx)
