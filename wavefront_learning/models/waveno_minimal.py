"""WaveNOMinimal: Ablation baseline for WaveNO.

Strips WaveNO down to the essential 5-stage pipeline to test whether
the core mechanism alone is sufficient:

    1. SegmentPhysicsEncoder — encode IC segments with physics features
    2. Fourier query encoding — FourierFeatures(t) + FourierFeatures(x) → MLP
    3. compute_characteristic_bias — backward characteristic foot penalty, no damping
    4. BiasedCrossDecoderLayer — cross-attention with physics bias
    5. Density head — Linear → ReLU → Dropout → Linear, clamped to [0, 1]

Removed vs WaveNO:
    - Self-attention over segments (EncoderLayer)
    - FiLM time conditioning (TimeConditioner)
    - Cross-segment attention per timestep (CrossSegmentAttention)
    - Breakpoint evolution / trajectory prediction
    - Boundary extraction + boundary Fourier features
    - Collision-time damping
    - Classifier head, proximity head, learned collision time

Segment embeddings are static — they don't change with time or interact
with each other. Time dependence enters only through the query encoding
and the characteristic bias.
"""

import torch
import torch.nn as nn

from .base.biased_cross_attention import (
    BiasedCrossDecoderLayer,
    compute_characteristic_bias,
)
from .base.characteristic_features import SegmentPhysicsEncoder
from .base.feature_encoders import FourierFeatures
from .base.flux import DEFAULT_FLUX, Flux


class WaveNOMinimal(nn.Module):
    """Minimal Wavefront Neural Operator (ablation baseline).

    Args:
        hidden_dim: All embedding dimensions.
        num_freq_t: Fourier frequency bands for time in query encoder.
        num_freq_x: Fourier frequency bands for space in query encoder.
        num_seg_frequencies: Fourier frequency bands for segment encoder.
        num_seg_mlp_layers: MLP depth in segment encoder.
        num_cross_layers: Biased cross-attention layers (queries → segments).
        num_heads: Attention heads for cross-attention.
        initial_bias_scale: Initial characteristic bias scale.
        flux: Flux function instance.
        local_features: Include cumulative mass in segment encoder.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_freq_t: int = 8,
        num_freq_x: int = 8,
        num_seg_frequencies: int = 8,
        num_seg_mlp_layers: int = 2,
        num_cross_layers: int = 2,
        num_heads: int = 4,
        initial_bias_scale: float = 5.0,
        flux: Flux | None = None,
        local_features: bool = True,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        flux = flux or DEFAULT_FLUX()
        self.flux = flux

        # Stage 1: Segment encoder (static, no self-attention)
        self.segment_encoder = SegmentPhysicsEncoder(
            hidden_dim=hidden_dim,
            num_frequencies=num_seg_frequencies,
            num_layers=num_seg_mlp_layers,
            flux=flux,
            include_cumulative_mass=local_features,
            dropout=dropout,
        )

        # Stage 2: Query encoder (Fourier + MLP)
        self.fourier_t = FourierFeatures(
            num_frequencies=num_freq_t, include_input=True
        )
        self.fourier_x = FourierFeatures(
            num_frequencies=num_freq_x, include_input=True
        )
        query_input_dim = self.fourier_t.output_dim + self.fourier_x.output_dim
        self.query_mlp = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Stage 3: Characteristic bias scale (no damping)
        self.bias_scale = nn.Parameter(torch.tensor(initial_bias_scale))

        # Stage 4: Biased cross-attention layers
        self.cross_attn_layers = nn.ModuleList(
            [
                BiasedCrossDecoderLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_cross_layers)
            ]
        )

        # Stage 5: Density head
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.density_head[-1].weight)
        nn.init.constant_(self.density_head[-1].bias, 0.5)

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch_input: Dict containing:
                - 'xs': (B, K+1) breakpoint positions
                - 'ks': (B, K) piece values
                - 'pieces_mask': (B, K) validity mask
                - 't_coords': (B, 1, nt, nx) time coordinates
                - 'x_coords': (B, 1, nt, nx) space coordinates

        Returns:
            Dict containing:
                - 'output_grid': (B, 1, nt, nx) predicted density
                - 'characteristic_bias': (B, nt, nx, K) physics bias
        """
        xs = batch_input["xs"]
        ks = batch_input["ks"]
        pieces_mask = batch_input["pieces_mask"]
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, nt, nx)

        B, nt, nx = t_coords.shape
        K = ks.shape[1]

        # Stage 1: Encode segments (static — no self-attention)
        seg_emb = self.segment_encoder(xs, ks, pieces_mask)  # (B, K, H)

        # Stage 2: Query encoding
        t_flat = t_coords.reshape(-1)  # (B*nt*nx,)
        x_flat = x_coords.reshape(-1)  # (B*nt*nx,)
        t_enc = self.fourier_t(t_flat)  # (B*nt*nx, F_t)
        x_enc = self.fourier_x(x_flat)  # (B*nt*nx, F_x)
        query_features = torch.cat([t_enc, x_enc], dim=-1)
        query_emb = self.query_mlp(query_features)  # (B*nt*nx, H)
        query_emb = query_emb.reshape(B, nt, nx, self.hidden_dim)

        # Stage 3: Characteristic attention bias (no damping)
        char_bias = compute_characteristic_bias(
            t_coords,
            x_coords,
            xs,
            ks,
            pieces_mask,
            self.flux,
            self.bias_scale,
            damping_sharpness=None,
        )  # (B, nt, nx, K)

        # Stage 4: Per-time-step biased cross-attention
        # Static segments expanded across time
        kv = seg_emb.unsqueeze(1).expand(B, nt, K, self.hidden_dim)
        kv = kv.reshape(B * nt, K, self.hidden_dim)  # (B*nt, K, H)

        q = query_emb.reshape(B * nt, nx, self.hidden_dim)  # (B*nt, nx, H)
        bias_flat = char_bias.reshape(B * nt, nx, K)  # (B*nt, nx, K)

        # Expand bias for multi-head: (B*nt, nx, K) → (B*nt*heads, nx, K)
        attn_mask = (
            bias_flat.unsqueeze(1)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(B * nt * self.num_heads, nx, K)
        )

        # Key padding mask: (B, K) → (B*nt, K)
        kv_padding_mask = (
            (~pieces_mask.bool())
            .unsqueeze(1)
            .expand(B, nt, K)
            .reshape(B * nt, K)
        )
        all_masked = kv_padding_mask.all(dim=1)
        if all_masked.any():
            kv_padding_mask = kv_padding_mask.clone()
            kv_padding_mask[all_masked] = False

        for layer in self.cross_attn_layers:
            q = layer(
                q, kv, key_padding_mask=kv_padding_mask, attn_mask=attn_mask
            )

        # Stage 5: Density head
        density = self.density_head(q).squeeze(-1)  # (B*nt, nx)
        density = density.clamp(0.0, 1.0)
        output_grid = density.reshape(B, 1, nt, nx)

        return {
            "output_grid": output_grid,
            "characteristic_bias": char_bias,
        }


def build_waveno_minimal(args: dict) -> WaveNOMinimal:
    """Factory function for WaveNOMinimal.

    Args:
        args: Dict or Namespace with model configuration.

    Returns:
        Configured WaveNOMinimal instance.
    """
    if not isinstance(args, dict):
        args = vars(args)

    return WaveNOMinimal(
        hidden_dim=args.get("hidden_dim", 64),
        num_freq_t=args.get("num_freq_t", 8),
        num_freq_x=args.get("num_freq_x", 8),
        num_seg_frequencies=args.get("num_seg_frequencies", 8),
        num_seg_mlp_layers=args.get("num_seg_mlp_layers", 2),
        num_cross_layers=args.get("num_cross_layers", 2),
        num_heads=args.get("num_heads", 4),
        initial_bias_scale=args.get("initial_bias_scale", 5.0),
        local_features=args.get("local_features", True),
        dropout=args.get("dropout", 0.05),
    )
