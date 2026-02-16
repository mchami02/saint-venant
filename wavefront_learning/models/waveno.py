"""WaveNO: Wavefront Neural Operator for hyperbolic conservation laws.

Inspired by wavefront tracking: the solution is determined by wavefronts
emanating from IC boundaries, acting on IC segments. Instead of tracking
wavefronts explicitly (trajectories) or selecting a winning segment (softmin),
WaveNO lets spatial queries discover the relevant segment information via
physics-biased cross-attention.

Architecture:
    1. SegmentPhysicsEncoder + self-attention -> contextualized segment embeddings
    2. TimeConditioner (FiLM) + CrossSegmentAttention -> time-evolved segments
    3. Fourier query encoding -> per-(t, x) query embeddings
    4. Characteristic attention bias -> physics-informed bias (B, nt, nx, K)
    5. Per-time-step biased cross-attention (queries attend to segments)
    6. Density head MLP -> output density

Key advantages over CharNO:
    - Cross-attention replaces softmin (continuous gradient flow, no vanishing)
    - Density decoded from attended features (fused, not decoupled)
    - Single characteristic bias as attention prior, not 8 hand-engineered features
    - No temperature scheduling, no auxiliary selection supervision loss
"""

import torch
import torch.nn as nn

from .base.biased_cross_attention import (
    BiasedCrossDecoderLayer,
    compute_characteristic_bias,
)
from .base.characteristic_features import SegmentPhysicsEncoder, TimeConditioner
from .base.feature_encoders import FourierFeatures
from .base.flux import DEFAULT_FLUX, Flux
from .base.transformer_encoder import EncoderLayer


class CrossSegmentAttention(nn.Module):
    """Lightweight self-attention over the K segment dimension.

    No feedforward network -- just attention + residual + LayerNorm.
    Identical to CharNO's CrossSegmentAttention.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        att = self.attention(x, x, x, key_padding_mask=key_padding_mask)[0]
        return self.norm(x + att)


class WaveNO(nn.Module):
    """Wavefront Neural Operator.

    Args:
        hidden_dim: All embedding dimensions.
        num_freq_t: Fourier frequency bands for time in query encoder.
        num_freq_x: Fourier frequency bands for space in query encoder.
        num_seg_frequencies: Fourier frequency bands for segment encoder.
        num_seg_mlp_layers: MLP depth in segment encoder.
        num_self_attn_layers: Self-attention layers for segment interaction.
        num_cross_layers: Biased cross-attention layers (queries -> segments).
        num_heads: Attention heads (both self and cross).
        num_cross_segment_layers: Cross-segment attention per timestep.
        time_condition: Enable FiLM time conditioning.
        initial_bias_scale: Initial characteristic bias scale.
        initial_damping_sharpness: Initial sharpness for collision-time
            damping of characteristic bias. Controls how quickly the bias
            fades after estimated wave collision time. Higher = sharper
            transition. Default 5.0.
        flux: Flux function instance.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_freq_t: int = 8,
        num_freq_x: int = 8,
        num_seg_frequencies: int = 8,
        num_seg_mlp_layers: int = 2,
        num_self_attn_layers: int = 2,
        num_cross_layers: int = 2,
        num_heads: int = 4,
        num_cross_segment_layers: int = 1,
        time_condition: bool = True,
        initial_bias_scale: float = 10.0,
        initial_damping_sharpness: float = 5.0,
        flux: Flux | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_condition = time_condition
        self.num_heads = num_heads
        flux = flux or DEFAULT_FLUX()
        self.flux = flux

        # === Stage 1: Segment encoder (reused from CharNO) ===
        self.segment_encoder = SegmentPhysicsEncoder(
            hidden_dim=hidden_dim,
            num_frequencies=num_seg_frequencies,
            num_layers=num_seg_mlp_layers,
            flux=flux,
        )

        # Self-attention over segments
        self.self_attn_layers = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, num_heads=num_heads)
                for _ in range(num_self_attn_layers)
            ]
        )

        # === Stage 2: Time conditioning + cross-segment attention ===
        if time_condition:
            self.time_conditioner = TimeConditioner(
                hidden_dim=hidden_dim,
                num_time_frequencies=num_seg_frequencies,
            )

        if num_cross_segment_layers > 0:
            self.cross_segment_layers = nn.ModuleList(
                [
                    CrossSegmentAttention(hidden_dim, num_heads=num_heads)
                    for _ in range(num_cross_segment_layers)
                ]
            )
        else:
            self.cross_segment_layers = nn.ModuleList()

        # === Stage 3: Query encoder (Fourier + MLP) ===
        self.fourier_t = FourierFeatures(
            num_frequencies=num_freq_t, include_input=True
        )
        self.fourier_x = FourierFeatures(
            num_frequencies=num_freq_x, include_input=True
        )
        query_input_dim = self.fourier_t.output_dim + self.fourier_x.output_dim
        self.query_mlp = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # === Stage 4: Characteristic bias scale + collision-time damping ===
        self.bias_scale = nn.Parameter(torch.tensor(initial_bias_scale))
        self.damping_sharpness = nn.Parameter(
            torch.tensor(initial_damping_sharpness)
        )

        # === Stage 5: Biased cross-attention layers ===
        self.cross_attn_layers = nn.ModuleList(
            [
                BiasedCrossDecoderLayer(hidden_dim, num_heads=num_heads)
                for _ in range(num_cross_layers)
            ]
        )

        # === Stage 6: Density head ===
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize last layer near zero for stable start
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
                - 'characteristic_bias': (B, nt, nx, K) physics bias (diagnostic)
        """
        xs = batch_input["xs"]
        ks = batch_input["ks"]
        pieces_mask = batch_input["pieces_mask"]
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, nt, nx)

        B, nt, nx = t_coords.shape
        K = ks.shape[1]

        # === Stage 1: Encode segments ===
        seg_emb = self.segment_encoder(xs, ks, pieces_mask)  # (B, K, H)

        # Self-attention over segments
        key_padding_mask = ~pieces_mask.bool()  # True = padded
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        for layer in self.self_attn_layers:
            seg_emb = layer(seg_emb, key_padding_mask=key_padding_mask)
        seg_emb = seg_emb * pieces_mask.unsqueeze(-1)  # re-zero padded

        # === Stage 2: Time-conditioned segment evolution ===
        t_unique = t_coords[:, :, 0]  # (B, nt)
        if self.time_condition:
            seg_emb_t = self.time_conditioner(seg_emb, t_unique)  # (B, nt, K, H)
        else:
            seg_emb_t = seg_emb.unsqueeze(1).expand(B, nt, K, self.hidden_dim)

        # Cross-segment attention per time step
        if len(self.cross_segment_layers) > 0:
            seg_flat = seg_emb_t.reshape(B * nt, K, -1)
            cs_mask = (
                (~pieces_mask.bool())
                .unsqueeze(1)
                .expand(B, nt, K)
                .reshape(B * nt, K)
            )
            all_masked_cs = cs_mask.all(dim=1)
            if all_masked_cs.any():
                cs_mask = cs_mask.clone()
                cs_mask[all_masked_cs] = False
            for layer in self.cross_segment_layers:
                seg_flat = layer(seg_flat, key_padding_mask=cs_mask)
            seg_emb_t = seg_flat.reshape(B, nt, K, self.hidden_dim)
            seg_emb_t = seg_emb_t * pieces_mask.unsqueeze(1).unsqueeze(-1)

        # === Stage 3: Query encoding ===
        t_flat = t_coords.reshape(-1)  # (B*nt*nx,)
        x_flat = x_coords.reshape(-1)  # (B*nt*nx,)
        t_enc = self.fourier_t(t_flat)  # (B*nt*nx, F_t)
        x_enc = self.fourier_x(x_flat)  # (B*nt*nx, F_x)
        query_features = torch.cat([t_enc, x_enc], dim=-1)  # (B*nt*nx, F_t+F_x)
        query_emb = self.query_mlp(query_features)  # (B*nt*nx, H)
        query_emb = query_emb.reshape(B, nt, nx, self.hidden_dim)  # (B, nt, nx, H)

        # === Stage 4: Characteristic attention bias ===
        char_bias = compute_characteristic_bias(
            t_coords,
            x_coords,
            xs,
            ks,
            pieces_mask,
            self.flux,
            self.bias_scale,
            damping_sharpness=self.damping_sharpness,
        )  # (B, nt, nx, K)

        # === Stage 5: Per-time-step biased cross-attention ===
        # Reshape for batched per-timestep attention
        q = query_emb.reshape(B * nt, nx, self.hidden_dim)  # (B*nt, nx, H)
        kv = seg_emb_t.reshape(B * nt, K, self.hidden_dim)  # (B*nt, K, H)
        bias_flat = char_bias.reshape(B * nt, nx, K)  # (B*nt, nx, K)

        # Expand bias for multi-head: (B*nt, nx, K) -> (B*nt*heads, nx, K)
        attn_mask = (
            bias_flat.unsqueeze(1)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(B * nt * self.num_heads, nx, K)
        )

        # Key padding mask: (B, K) -> (B*nt, K)
        kv_padding_mask = (
            (~pieces_mask.bool())
            .unsqueeze(1)
            .expand(B, nt, K)
            .reshape(B * nt, K)
        )
        all_masked_kv = kv_padding_mask.all(dim=1)
        if all_masked_kv.any():
            kv_padding_mask = kv_padding_mask.clone()
            kv_padding_mask[all_masked_kv] = False

        for layer in self.cross_attn_layers:
            q = layer(
                q, kv, key_padding_mask=kv_padding_mask, attn_mask=attn_mask
            )

        # === Stage 6: Density head ===
        density = self.density_head(q).squeeze(-1)  # (B*nt, nx)
        density = density.clamp(0.0, 1.0)
        output_grid = density.reshape(B, 1, nt, nx)

        return {
            "output_grid": output_grid,
            "characteristic_bias": char_bias,
        }


def build_waveno(args: dict) -> WaveNO:
    """Factory function for WaveNO.

    Args:
        args: Dict or Namespace with model configuration.

    Returns:
        Configured WaveNO instance.
    """
    if not isinstance(args, dict):
        args = vars(args)

    return WaveNO(
        hidden_dim=args.get("hidden_dim", 64),
        num_freq_t=args.get("num_freq_t", 8),
        num_freq_x=args.get("num_freq_x", 8),
        num_seg_frequencies=args.get("num_seg_frequencies", 8),
        num_seg_mlp_layers=args.get("num_seg_mlp_layers", 2),
        num_self_attn_layers=args.get("num_self_attn_layers", 2),
        num_cross_layers=args.get("num_cross_layers", 2),
        num_heads=args.get("num_heads", 4),
        num_cross_segment_layers=args.get("num_cross_segment_layers", 1),
        time_condition=args.get("time_condition", True),
        initial_bias_scale=args.get("initial_bias_scale", 10.0),
        initial_damping_sharpness=args.get("initial_damping_sharpness", 5.0),
    )
