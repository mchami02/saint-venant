"""WaveNOMinimal: Ablation baseline for WaveNO.

Strips WaveNO down to a minimal pipeline to isolate which components
matter most:

    1. SegmentPhysicsEncoder — encode IC segments with physics features
    1b. Self-attention over segments — contextualize embeddings
    2. Fourier query encoding — FourierFeatures(t) + FourierFeatures(x) → MLP
    3. LWRBias — shock/rarefaction-aware attention bias with time damping
    4. BiasedCrossDecoderLayer — cross-attention with physics bias
    5. Density head — Linear → ReLU → Dropout → Linear, clamped to [0, 1]

Removed vs WaveNO:
    - FiLM time conditioning (TimeConditioner)
    - Cross-segment attention per timestep (CrossSegmentAttention)
    - Breakpoint evolution / trajectory prediction
    - Boundary extraction + boundary Fourier features
    - Classifier head, proximity head, learned collision time

Segment embeddings are contextualized (segments know about neighbors)
but static across time. Time dependence enters only through the query
encoding and the characteristic bias.

Ablation flags (use_char_bias, use_film, use_cross_seg_attn)
allow toggling individual components for controlled experiments. See the
build_waveno_ablation_* factory functions for predefined configurations.
"""

import torch
import torch.nn as nn

from .base.biased_cross_attention import BiasedCrossDecoderLayer
from .base.lwr_bias import LWRBias
from .base.characteristic_features import SegmentPhysicsEncoder, TimeConditioner
from .base.feature_encoders import FourierFeatures
from .base.flux import DEFAULT_FLUX, Flux
from .base.transformer_encoder import CrossSegmentAttention, EncoderLayer


class WaveNOMinimal(nn.Module):
    """Minimal Wavefront Neural Operator (ablation baseline).

    Args:
        hidden_dim: All embedding dimensions.
        num_freq_t: Fourier frequency bands for time in query encoder.
        num_freq_x: Fourier frequency bands for space in query encoder.
        num_seg_frequencies: Fourier frequency bands for segment encoder.
        num_seg_mlp_layers: MLP depth in segment encoder.
        num_self_attn_layers: Self-attention layers for segment interaction.
        num_cross_layers: Biased cross-attention layers (queries → segments).
        num_heads: Attention heads for self and cross attention.
        initial_damping_sharpness: Initial sharpness for LWRBias
            collision-time damping (learnable).
        flux: Flux function instance.
        local_features: Include cumulative mass in segment encoder.
        dropout: Dropout rate.
        use_char_bias: Enable LWR attention bias. When False,
            cross-attention is unbiased (attn_mask=None).
        use_film: Enable FiLM time conditioning on segment embeddings.
        use_cross_seg_attn: Enable per-timestep cross-segment attention.
        num_cross_segment_layers: Number of cross-segment attention layers
            (only used when use_cross_seg_attn=True).
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
        initial_damping_sharpness: float = 5.0,
        flux: Flux | None = None,
        use_damping: bool = True,
        local_features: bool = True,
        dropout: float = 0.05,
        use_char_bias: bool = True,
        use_film: bool = False,
        use_cross_seg_attn: bool = False,
        num_cross_segment_layers: int = 1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_char_bias = use_char_bias
        self.use_film = use_film
        self.use_cross_seg_attn = use_cross_seg_attn
        flux = flux or DEFAULT_FLUX()
        self.flux = flux

        # Stage 1: Segment encoder + self-attention
        self.segment_encoder = SegmentPhysicsEncoder(
            hidden_dim=hidden_dim,
            num_frequencies=num_seg_frequencies,
            num_layers=num_seg_mlp_layers,
            flux=flux,
            include_cumulative_mass=local_features,
            dropout=dropout,
        )
        self.self_attn_layers = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_self_attn_layers)
            ]
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

        # Stage 2b: Segment evolution (conditional)
        if use_film:
            self.time_conditioner = TimeConditioner(
                hidden_dim=hidden_dim,
                num_time_frequencies=num_seg_frequencies,
                dropout=dropout,
            )

        if use_cross_seg_attn:
            self.cross_segment_layers = nn.ModuleList(
                [
                    CrossSegmentAttention(
                        hidden_dim, num_heads=num_heads, dropout=dropout
                    )
                    for _ in range(num_cross_segment_layers)
                ]
            )

        # Stage 3: LWR attention bias (conditional)
        if use_char_bias:
            self.lwr_bias = LWRBias(
                initial_damping_sharpness=initial_damping_sharpness,
                flux=flux,
                use_damping=use_damping,
            )

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
                - 'characteristic_bias': (B, nt, nx, K) physics bias (if use_char_bias)
        """
        xs = batch_input["xs"]
        ks = batch_input["ks"]
        pieces_mask = batch_input["pieces_mask"]
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, nt, nx)

        B, nt, nx = t_coords.shape
        K = ks.shape[1]

        # Stage 1: Encode segments + self-attention
        seg_emb = self.segment_encoder(xs, ks, pieces_mask)  # (B, K, H)

        key_padding_mask = ~pieces_mask.bool()  # True = padded
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        for layer in self.self_attn_layers:
            seg_emb = layer(seg_emb, key_padding_mask=key_padding_mask)
        seg_emb = seg_emb * pieces_mask.unsqueeze(-1)  # re-zero padded

        # Stage 2: Query encoding
        t_flat = t_coords.reshape(-1)  # (B*nt*nx,)
        x_flat = x_coords.reshape(-1)  # (B*nt*nx,)
        t_enc = self.fourier_t(t_flat)  # (B*nt*nx, F_t)
        x_enc = self.fourier_x(x_flat)  # (B*nt*nx, F_x)
        query_features = torch.cat([t_enc, x_enc], dim=-1)
        query_emb = self.query_mlp(query_features)  # (B*nt*nx, H)
        query_emb = query_emb.reshape(B, nt, nx, self.hidden_dim)

        # Stage 2b: Segment evolution (FiLM + cross-segment attention)
        if self.use_film:
            t_unique = t_coords[:, :, 0]  # (B, nt)
            kv = self.time_conditioner(seg_emb, t_unique)  # (B, nt, K, H)
        else:
            kv = seg_emb.unsqueeze(1).expand(B, nt, K, self.hidden_dim)

        kv = kv.reshape(B * nt, K, self.hidden_dim)  # (B*nt, K, H)

        if self.use_cross_seg_attn:
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
                kv = layer(kv, key_padding_mask=cs_mask)
            kv = kv.reshape(B, nt, K, self.hidden_dim)
            kv = kv * pieces_mask.unsqueeze(1).unsqueeze(-1)  # re-zero
            kv = kv.reshape(B * nt, K, self.hidden_dim)

        # Stage 3: LWR attention bias (conditional)
        if self.use_char_bias:
            ic_data = {"xs": xs, "ks": ks, "pieces_mask": pieces_mask}
            char_bias = self.lwr_bias(
                ic_data, (t_coords, x_coords)
            )  # (B, nt, nx, K)

            bias_flat = char_bias.reshape(B * nt, nx, K)  # (B*nt, nx, K)
            attn_mask = (
                bias_flat.unsqueeze(1)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(B * nt * self.num_heads, nx, K)
            )
        else:
            char_bias = None
            attn_mask = None

        # Stage 4: Per-time-step cross-attention
        q = query_emb.reshape(B * nt, nx, self.hidden_dim)  # (B*nt, nx, H)

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

        result = {"output_grid": output_grid}
        if char_bias is not None:
            result["characteristic_bias"] = char_bias
        return result


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_waveno_minimal(args: dict) -> WaveNOMinimal:
    """Factory function for WaveNOMinimal (backward compatible).

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
        num_self_attn_layers=args.get("num_self_attn_layers", 2),
        num_cross_layers=args.get("num_cross_layers", 2),
        num_heads=args.get("num_heads", 4),
        initial_damping_sharpness=args.get("initial_damping_sharpness", 5.0),
        local_features=args.get("local_features", True),
        dropout=args.get("dropout", 0.05),
    )


def _build_ablation(args: dict, **flag_overrides) -> WaveNOMinimal:
    """Shared builder for ablation variants."""
    if not isinstance(args, dict):
        args = vars(args)

    kwargs = dict(
        hidden_dim=args.get("hidden_dim", 64),
        num_freq_t=args.get("num_freq_t", 8),
        num_freq_x=args.get("num_freq_x", 8),
        num_seg_frequencies=args.get("num_seg_frequencies", 8),
        num_seg_mlp_layers=args.get("num_seg_mlp_layers", 2),
        num_self_attn_layers=args.get("num_self_attn_layers", 2),
        num_cross_layers=args.get("num_cross_layers", 2),
        num_heads=args.get("num_heads", 4),
        initial_damping_sharpness=args.get("initial_damping_sharpness", 5.0),
        local_features=args.get("local_features", True),
        dropout=args.get("dropout", 0.05),
        num_cross_segment_layers=args.get("num_cross_segment_layers", 1),
    )
    kwargs.update(flag_overrides)
    return WaveNOMinimal(**kwargs)


def build_waveno_ablation(args: dict) -> WaveNOMinimal:
    """Bare minimum: unbiased cross-attention, no extras."""
    return _build_ablation(
        args, use_char_bias=False, use_film=False, use_cross_seg_attn=False
    )


def build_waveno_ablation_bias(args: dict) -> WaveNOMinimal:
    """+ LWR bias (without damping)."""
    return _build_ablation(
        args, use_char_bias=True, use_damping=False, use_film=False, use_cross_seg_attn=False
    )


def build_waveno_ablation_damp(args: dict) -> WaveNOMinimal:
    """+ LWR bias + collision-time damping."""
    return _build_ablation(
        args, use_char_bias=True, use_damping=True, use_film=False, use_cross_seg_attn=False
    )


def build_waveno_ablation_film(args: dict) -> WaveNOMinimal:
    """+ LWR bias + FiLM time conditioning."""
    return _build_ablation(
        args, use_char_bias=True, use_film=True, use_cross_seg_attn=False
    )


def build_waveno_ablation_cross_attn(args: dict) -> WaveNOMinimal:
    """+ LWR bias + cross-segment attention."""
    return _build_ablation(
        args, use_char_bias=True, use_film=False, use_cross_seg_attn=True
    )


def build_waveno_ablation_full(args: dict) -> WaveNOMinimal:
    """All components except trajectories."""
    return _build_ablation(
        args, use_char_bias=True, use_film=True, use_cross_seg_attn=True
    )


def build_waveno_ablation_film_only(args: dict) -> WaveNOMinimal:
    """FiLM only (no LWR bias)."""
    return _build_ablation(
        args, use_char_bias=False, use_film=True, use_cross_seg_attn=False
    )


def build_waveno_ablation_bias_film(args: dict) -> WaveNOMinimal:
    """+ LWR bias (no damping) + FiLM time conditioning."""
    return _build_ablation(
        args, use_char_bias=True, use_damping=False, use_film=True, use_cross_seg_attn=False
    )


def build_waveno_ablation_cross_attn_only(args: dict) -> WaveNOMinimal:
    """Cross-segment attention only (no LWR bias)."""
    return _build_ablation(
        args, use_char_bias=False, use_film=False, use_cross_seg_attn=True
    )
