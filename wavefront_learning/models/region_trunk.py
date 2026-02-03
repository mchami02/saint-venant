"""Region trunk components for HybridDeepONet.

This module provides trunk networks for predicting density values in each
region between shock waves. The regions are defined by the predicted shock
trajectories, with K = D + 1 regions for D discontinuities.

Architecture:
    SpaceTimeEncoder: Encodes (t, x) coordinate pairs using Fourier features
    RegionTrunk: Combines branch embedding with space-time encoding to predict
        density at each (t, x) point within a specific region
"""

import torch
import torch.nn as nn

from models.shock_trajectory_net import FourierFeatures, ResidualBlock


class SpaceTimeEncoder(nn.Module):
    """Encodes (t, x) coordinate pairs using Fourier features.

    Uses separate Fourier encodings for time and space coordinates,
    then concatenates and projects to output dimension.

    Args:
        hidden_dim: Hidden dimension of the MLP.
        output_dim: Output dimension (latent space dimension).
        num_frequencies_t: Number of Fourier frequency bands for time.
        num_frequencies_x: Number of Fourier frequency bands for space.
        num_layers: Number of MLP layers.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_frequencies_t: int = 16,
        num_frequencies_x: int = 16,
        num_layers: int = 3,
    ):
        super().__init__()
        self.fourier_t = FourierFeatures(
            num_frequencies=num_frequencies_t, include_input=True
        )
        self.fourier_x = FourierFeatures(
            num_frequencies=num_frequencies_x, include_input=True
        )

        # Input dimension: fourier_t + fourier_x
        input_dim = self.fourier_t.output_dim + self.fourier_x.output_dim

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(
        self,
        t_coords: torch.Tensor,
        x_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Encode (t, x) coordinate pairs.

        Args:
            t_coords: Time coordinates of shape (B, nt, nx) or (B*nt*nx,).
            x_coords: Space coordinates of shape (B, nt, nx) or (B*nt*nx,).

        Returns:
            Encoded coordinates of shape (B, nt, nx, output_dim) or
            (B*nt*nx, output_dim).
        """
        original_shape = t_coords.shape
        is_3d = t_coords.dim() == 3

        if is_3d:
            B, nt, nx = original_shape
            t_flat = t_coords.reshape(-1)  # (B*nt*nx,)
            x_flat = x_coords.reshape(-1)  # (B*nt*nx,)
        else:
            t_flat = t_coords
            x_flat = x_coords

        # Fourier encode time and space separately
        t_encoded = self.fourier_t(t_flat)  # (B*nt*nx, fourier_t_dim)
        x_encoded = self.fourier_x(x_flat)  # (B*nt*nx, fourier_x_dim)

        # Concatenate and project
        combined = torch.cat([t_encoded, x_encoded], dim=-1)
        output = self.mlp(combined)  # (B*nt*nx, output_dim)

        if is_3d:
            output = output.reshape(B, nt, nx, -1)

        return output


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
