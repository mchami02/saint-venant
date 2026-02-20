"""Decoder modules for wavefront learning."""

import torch
import torch.nn as nn

from .biased_cross_attention import BiasedCrossDecoderLayer
from .blocks import ResidualBlock
from .cross_decoder import CrossDecoderLayer
from .feature_encoders import FourierFeatures


class TrajectoryDecoder(nn.Module):
    """Decodes trajectory predictions from branch and trunk embeddings.

    Uses bilinear fusion to combine discontinuity (branch) and time (trunk)
    embeddings, then applies residual blocks and separate heads for
    position and existence prediction.

    Args:
        branch_dim: Dimension of branch (discontinuity) embeddings.
        trunk_dim: Dimension of trunk (time) embeddings.
        hidden_dim: Hidden dimension for decoder.
        num_res_blocks: Number of residual blocks.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        branch_dim: int = 128,
        trunk_dim: int = 128,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Bilinear fusion layer
        # Maps (branch_dim, trunk_dim) -> hidden_dim
        self.bilinear = nn.Bilinear(branch_dim, trunk_dim, hidden_dim)

        # Also add a linear combination path for better gradient flow
        self.linear_branch = nn.Linear(branch_dim, hidden_dim)
        self.linear_trunk = nn.Linear(trunk_dim, hidden_dim)

        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)
        ])

        # Output heads
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.existence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        branch_emb: torch.Tensor,
        trunk_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode trajectories.

        Args:
            branch_emb: Discontinuity embeddings of shape (B, D, branch_dim).
            trunk_emb: Time embeddings of shape (B, T, trunk_dim).
            mask: Validity mask of shape (B, D).

        Returns:
            Dict with:
                - 'positions': (B, D, T) predicted x-positions for each shock at each time
                - 'existence': (B, D, T) probability that each shock exists at each time
        """
        B, D, _ = branch_emb.shape
        _, T, _ = trunk_emb.shape

        # Expand embeddings for all (discontinuity, time) pairs
        # branch: (B, D, 1, branch_dim) -> (B, D, T, branch_dim)
        # trunk: (B, 1, T, trunk_dim) -> (B, D, T, trunk_dim)
        branch_exp = branch_emb.unsqueeze(2).expand(-1, -1, T, -1)
        trunk_exp = trunk_emb.unsqueeze(1).expand(-1, D, -1, -1)

        # Reshape for bilinear: (B*D*T, dim)
        branch_flat = branch_exp.reshape(-1, branch_exp.shape[-1])
        trunk_flat = trunk_exp.reshape(-1, trunk_exp.shape[-1])

        # Bilinear fusion + linear paths
        fused = self.bilinear(branch_flat, trunk_flat)
        fused = fused + self.linear_branch(branch_flat) + self.linear_trunk(trunk_flat)
        fused = self.fusion_norm(fused)

        # Residual blocks
        for block in self.res_blocks:
            fused = block(fused)

        # Reshape back: (B, D, T, hidden_dim)
        fused = fused.reshape(B, D, T, -1)

        # Apply output heads
        positions = self.position_head(fused).squeeze(-1)  # (B, D, T)
        positions = torch.clamp(positions, 0.0, 1.0)  # Constrain to grid domain
        existence = self.existence_head(fused).squeeze(-1)  # (B, D, T)

        # Mask out invalid discontinuities
        mask_exp = mask.unsqueeze(-1)  # (B, D, 1)
        positions = positions * mask_exp
        existence = existence * mask_exp

        return {
            "positions": positions,
            "existence": existence,
        }


class TrajectoryDecoderTransformer(nn.Module):
    """Decodes trajectory positions using cross-attention.

    Time embeddings (queries) attend to discontinuity embeddings (keys/values),
    then the enriched time features are combined with each discontinuity
    embedding to produce per-discontinuity positions.

    Args:
        hidden_dim: Dimension of embeddings.
        num_cross_layers: Number of cross-attention layers.
        num_attention_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_cross_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_layers = nn.ModuleList(
            [
                CrossDecoderLayer(hidden_dim, num_heads=num_attention_heads)
                for _ in range(num_cross_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.combine_norm = nn.LayerNorm(hidden_dim)

        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        disc_emb: torch.Tensor,
        time_emb: torch.Tensor,
        disc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode trajectory positions via cross-attention.

        Args:
            disc_emb: Discontinuity embeddings (B, D, H).
            time_emb: Time embeddings (B, T, H).
            disc_mask: Validity mask (B, D).

        Returns:
            Predicted positions (B, D, T) clamped to [0, 1].
        """
        B, D, H = disc_emb.shape
        T = time_emb.shape[1]

        # Cross-attention: time queries attend to disc keys/values
        key_padding_mask = ~disc_mask.bool()  # (B, D), True = ignore
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        x = time_emb  # (B, T, H)
        for layer in self.cross_layers:
            x = layer(x, disc_emb, key_padding_mask=key_padding_mask)
        time_enriched = self.final_norm(x)  # (B, T, H)

        # Combine each disc embedding with enriched time embeddings
        disc_exp = disc_emb.unsqueeze(2).expand(-1, -1, T, -1)  # (B, D, T, H)
        time_exp = time_enriched.unsqueeze(1).expand(-1, D, -1, -1)  # (B, D, T, H)
        combined = self.combine_norm(disc_exp + time_exp)  # (B, D, T, H)

        # Position head
        positions = self.position_head(combined).squeeze(-1)  # (B, D, T)
        positions = torch.clamp(positions, 0.0, 1.0)
        positions = positions * disc_mask.unsqueeze(-1)

        return positions


class DensityDecoderTransformer(nn.Module):
    """Decodes density using cross-attention over discontinuity embeddings.

    Encodes spacetime coordinates (with optional boundary positions) using
    Fourier features, then uses cross-attention to fuse with per-discontinuity
    embeddings.

    Args:
        hidden_dim: Hidden dimension.
        num_frequencies_t: Fourier frequencies for time.
        num_frequencies_x: Fourier frequencies for spatial coords.
        num_coord_layers: MLP layers for coordinate encoding.
        num_cross_layers: Number of cross-attention layers.
        num_attention_heads: Number of attention heads.
        dropout: Dropout rate.
        with_boundaries: Whether to include boundary positions in encoding.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_frequencies_t: int = 8,
        num_frequencies_x: int = 8,
        num_coord_layers: int = 2,
        num_cross_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        with_boundaries: bool = True,
        biased: bool = False,
    ):
        super().__init__()
        self.with_boundaries = with_boundaries
        self.biased = biased
        self.num_attention_heads = num_attention_heads

        self.fourier_t = FourierFeatures(num_frequencies=num_frequencies_t)
        self.fourier_x = FourierFeatures(num_frequencies=num_frequencies_x)

        # Input: fourier(t) + fourier(x) [+ fourier(x_left) + fourier(x_right)]
        num_spatial = 3 if with_boundaries else 1
        input_dim = self.fourier_t.output_dim + num_spatial * self.fourier_x.output_dim

        layers = []
        in_dim = input_dim
        for i in range(num_coord_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_coord_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        self.coord_mlp = nn.Sequential(*layers)

        layer_cls = BiasedCrossDecoderLayer if biased else CrossDecoderLayer
        self.cross_layers = nn.ModuleList(
            [
                layer_cls(hidden_dim, num_heads=num_attention_heads)
                for _ in range(num_cross_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        disc_emb: torch.Tensor,
        t_coords: torch.Tensor,
        x_coords: torch.Tensor,
        left_bound: torch.Tensor | None,
        right_bound: torch.Tensor | None,
        disc_mask: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict density via cross-attention with discontinuity embeddings.

        Args:
            disc_emb: Discontinuity embeddings (B, D, H).
            t_coords: Time coordinates (B, nt, nx).
            x_coords: Space coordinates (B, nt, nx).
            left_bound: Left boundary positions (B, nt, nx) or None.
            right_bound: Right boundary positions (B, nt, nx) or None.
            disc_mask: Validity mask (B, D).
            attn_bias: Optional characteristic attention bias (B, nt, nx, D).
                Only used when biased=True.

        Returns:
            Predicted density (B, nt, nx) in [0, 1].
        """
        B, nt, nx = t_coords.shape

        # Fourier encode coordinates
        t_flat = t_coords.reshape(-1)
        x_flat = x_coords.reshape(-1)

        t_enc = self.fourier_t(t_flat)
        x_enc = self.fourier_x(x_flat)

        if self.with_boundaries:
            left_enc = self.fourier_x(left_bound.reshape(-1))
            right_enc = self.fourier_x(right_bound.reshape(-1))
            coord_features = torch.cat([t_enc, x_enc, left_enc, right_enc], dim=-1)
        else:
            coord_features = torch.cat([t_enc, x_enc], dim=-1)

        coord_emb = self.coord_mlp(coord_features)  # (B*nt*nx, H)
        coord_emb = coord_emb.reshape(B, nt * nx, -1)  # (B, Q, H)

        # Prepare attention mask for biased cross-attention
        attn_mask = None
        if self.biased and attn_bias is not None:
            D = disc_emb.shape[1]
            Q = nt * nx
            # (B, nt, nx, D) -> (B, Q, D) -> (B*num_heads, Q, D)
            bias_flat = attn_bias.reshape(B, Q, D)
            attn_mask = (
                bias_flat.unsqueeze(1)
                .expand(-1, self.num_attention_heads, -1, -1)
                .reshape(B * self.num_attention_heads, Q, D)
            )

        # Cross-attention: coord queries attend to disc keys/values
        key_padding_mask = ~disc_mask.bool()  # (B, D)
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        x = coord_emb
        if self.biased:
            for layer in self.cross_layers:
                x = layer(
                    x, disc_emb, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
        else:
            for layer in self.cross_layers:
                x = layer(x, disc_emb, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        # Density head
        density = self.density_head(x).squeeze(-1)  # (B, Q)
        density = torch.clamp(density, 0.0, 1.0)
        return density.reshape(B, nt, nx)
