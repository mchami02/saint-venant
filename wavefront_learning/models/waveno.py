"""WaveNO: Wavefront Neural Operator (default architecture).

The default WaveNO uses PDE bias (without damping) + FiLM time conditioning.
This was identified as the most effective variant in the ablation study.

Architecture:
    1. SegmentPhysicsEncoder — encode IC segments with PDE-specific physics features
    1b. Self-attention over segments — contextualize embeddings
    2. Raw (t, x) query encoding — MLP on raw coordinates
    2b. FiLM time conditioning on segment embeddings (default)
    3. PDEBias — shock/rarefaction-aware attention bias
    4. BiasedCrossDecoderLayer — cross-attention with physics bias
    5. Output head — Linear → ReLU → Dropout → Linear

The model auto-adapts to the equation type via the ``PDE`` instance,
which provides physics features, boundary speeds, and output metadata.
"""

import torch
import torch.nn as nn

from .base.biased_cross_attention import BiasedCrossDecoderLayer
from .base.characteristic_features import SegmentPhysicsEncoder, TimeConditioner
from .base.feature_encoders import FourierFeatures
from .base.pde import PDE, ARZPDE, BurgersPDE, EulerPDE, LWRPDE
from .base.pde_bias import PDEBias
from .base.transformer_encoder import CrossSegmentAttention, EncoderLayer


class WaveNO(nn.Module):
    """Wavefront Neural Operator.

    Args:
        hidden_dim: All embedding dimensions.
        num_seg_frequencies: Fourier frequency bands for segment encoder.
        num_seg_mlp_layers: MLP depth in segment encoder.
        num_self_attn_layers: Self-attention layers for segment interaction.
        num_cross_layers: Biased cross-attention layers (queries → segments).
        num_heads: Attention heads for self and cross attention.
        pde: PDE instance providing boundary speeds for attention bias.
            ``None`` disables physics-informed bias.
        encoder_pde: PDE instance for the segment encoder's physics
            features.  When ``pde=None`` (unbiased mode), this ensures
            the encoder still uses equation-appropriate features.
        initial_damping_sharpness: Initial sharpness for collision-time
            damping (learnable).
        use_damping: Enable collision-time damping in PDEBias.
        local_features: Include cumulative mass in segment encoder.
        dropout: Dropout rate.
        use_film: Enable FiLM time conditioning on segment embeddings.
        use_cross_seg_attn: Enable per-timestep cross-segment attention.
        num_cross_segment_layers: Number of cross-segment attention layers
            (only used when use_cross_seg_attn=True).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_freq_t: int | None = None,
        num_freq_x: int | None = None,
        num_seg_frequencies: int | None = None,
        num_seg_mlp_layers: int = 2,
        num_self_attn_layers: int = 2,
        num_cross_layers: int = 2,
        num_heads: int = 4,
        pde: PDE | None = None,
        encoder_pde: PDE | None = None,
        initial_damping_sharpness: float = 5.0,
        use_damping: bool = True,
        local_features: bool = True,
        dropout: float = 0.05,
        use_film: bool = False,
        use_cross_seg_attn: bool = False,
        num_cross_segment_layers: int = 1,
        output_dim: int | None = None,
        output_clamp: tuple[float, float] | None | str = "auto",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pde = pde
        self.use_film = use_film
        self.use_cross_seg_attn = use_cross_seg_attn

        # Derive output config: explicit > PDE > LWR defaults
        if output_dim is not None:
            self.output_dim = output_dim
        elif pde is not None:
            self.output_dim = pde.output_dim
        else:
            self.output_dim = 1

        if output_clamp != "auto":
            self.output_clamp = output_clamp
        elif pde is not None:
            self.output_clamp = pde.output_clamp
        else:
            self.output_clamp = (0.0, 1.0)

        # Stage 1: Segment encoder + self-attention
        # encoder_pde provides physics features for the segment encoder,
        # independent of whether pde is used for attention bias.
        seg_pde = encoder_pde if encoder_pde is not None else (pde if pde is not None else LWRPDE())
        self.segment_encoder = SegmentPhysicsEncoder(
            hidden_dim=hidden_dim,
            num_frequencies=num_seg_frequencies,
            num_layers=num_seg_mlp_layers,
            pde=seg_pde,
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
        if num_freq_t is not None:
            self.fourier_t = FourierFeatures(
                num_frequencies=num_freq_t, include_input=True
            )
            query_t_dim = self.fourier_t.output_dim
        else:
            self.fourier_t = None
            query_t_dim = 1

        if num_freq_x is not None:
            self.fourier_x = FourierFeatures(
                num_frequencies=num_freq_x, include_input=True
            )
            query_x_dim = self.fourier_x.output_dim
        else:
            self.fourier_x = None
            query_x_dim = 1

        query_input_dim = query_t_dim + query_x_dim
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

        # Stage 3: PDE-aware attention bias (conditional)
        if pde is not None:
            self.pde_bias = PDEBias(
                pde=pde,
                use_damping=use_damping,
                initial_damping_sharpness=initial_damping_sharpness,
            )

        # Stage 4: Biased cross-attention layers
        self.cross_attn_layers = nn.ModuleList(
            [
                BiasedCrossDecoderLayer(
                    hidden_dim, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_cross_layers)
            ]
        )

        # Stage 5: Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        if output_dim == 1:
            nn.init.constant_(self.output_head[-1].bias, 0.5)
        else:
            nn.init.zeros_(self.output_head[-1].bias)

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
                May also contain 'ks_v', 'ks_p' for multi-variable PDEs.

        Returns:
            Dict containing:
                - 'output_grid': (B, output_dim, nt, nx) prediction
                - 'characteristic_bias': (B, nt, nx, K) physics bias (if pde)
        """
        pieces_mask = batch_input["pieces_mask"]
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, nt, nx)

        B, nt, nx = t_coords.shape
        K = pieces_mask.shape[1]

        # Build ic_data for encoder and bias
        ic_data = {
            "xs": batch_input["xs"],
            "ks": batch_input["ks"],
            "pieces_mask": pieces_mask,
        }
        if "ks_v" in batch_input:
            ic_data["ks_v"] = batch_input["ks_v"]
        if "ks_p" in batch_input:
            ic_data["ks_p"] = batch_input["ks_p"]

        # Stage 1: Encode segments + self-attention
        seg_emb = self.segment_encoder(ic_data)  # (B, K, H)

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
        t_enc = (
            self.fourier_t(t_flat)
            if self.fourier_t is not None
            else t_flat.unsqueeze(-1)
        )
        x_enc = (
            self.fourier_x(x_flat)
            if self.fourier_x is not None
            else x_flat.unsqueeze(-1)
        )
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
                (~pieces_mask.bool()).unsqueeze(1).expand(B, nt, K).reshape(B * nt, K)
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

        # Stage 3: PDE-aware attention bias (conditional)
        if self.pde is not None:
            char_bias = self.pde_bias(
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
            (~pieces_mask.bool()).unsqueeze(1).expand(B, nt, K).reshape(B * nt, K)
        )
        all_masked = kv_padding_mask.all(dim=1)
        if all_masked.any():
            kv_padding_mask = kv_padding_mask.clone()
            kv_padding_mask[all_masked] = False

        for layer in self.cross_attn_layers:
            q = layer(q, kv, key_padding_mask=kv_padding_mask, attn_mask=attn_mask)

        # Stage 5: Output head
        out = self.output_head(q)  # (B*nt, nx, output_dim)
        if self.output_clamp is not None:
            out = out.clamp(*self.output_clamp)
        output_grid = out.permute(0, 2, 1).reshape(B, self.output_dim, nt, nx)

        result = {"output_grid": output_grid}
        if char_bias is not None:
            result["characteristic_bias"] = char_bias
        return result


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _build_pde(args: dict) -> PDE | None:
    """Build a PDE instance from args.

    Args:
        args: Dict with at least ``equation`` key.

    Returns:
        PDE instance, or None if bias is disabled.
    """
    eq = args.get("equation", "LWR")
    if eq == "Euler":
        return EulerPDE(gamma=args.get("euler_gamma") or 1.4)
    if eq == "Euler2D":
        from .base.pde import Euler2DPDE
        return Euler2DPDE(gamma=args.get("euler_gamma") or 1.4)
    if eq == "ARZ":
        return ARZPDE(gamma=args.get("gamma", 1.0))
    if eq == "Burgers":
        return BurgersPDE()
    return LWRPDE()


def _build_waveno(args: dict, **flag_overrides) -> WaveNO:
    """Shared builder for WaveNO variants."""
    if not isinstance(args, dict):
        args = vars(args)

    # Build PDE unless overridden to None
    if "pde" in flag_overrides:
        pde = flag_overrides.pop("pde")
    else:
        pde = _build_pde(args)

    # Always build an encoder PDE from the equation so that pde=None
    # variants still get the correct physics features and output config.
    encoder_pde = _build_pde(args)

    kwargs = dict(
        hidden_dim=args.get("hidden_dim", 64),
        num_freq_t=args.get("num_freq_t", None),
        num_freq_x=args.get("num_freq_x", None),
        num_seg_frequencies=args.get("num_seg_frequencies", None),
        num_seg_mlp_layers=args.get("num_seg_mlp_layers", 2),
        num_self_attn_layers=args.get("num_self_attn_layers", 2),
        num_cross_layers=args.get("num_cross_layers", 2),
        num_heads=args.get("num_heads", 4),
        pde=pde,
        encoder_pde=encoder_pde,
        initial_damping_sharpness=args.get("initial_damping_sharpness", 5.0),
        local_features=args.get("local_features", True),
        dropout=args.get("dropout", 0.05),
        num_cross_segment_layers=args.get("num_cross_segment_layers", 1),
        output_dim=encoder_pde.output_dim,
        output_clamp=encoder_pde.output_clamp,
    )
    kwargs.update(flag_overrides)
    return WaveNO(**kwargs)


def build_waveno(args: dict) -> WaveNO:
    """Factory function for WaveNO (bias + FiLM, the default).

    Automatically adapts to the equation type via ``args["equation"]``:
    - LWR: LWR bias, 1-channel output, clamped [0, 1]
    - ARZ: ARZ bias, 2-channel output, no clamp
    - Euler: Euler bias, 3-channel output, no clamp

    Args:
        args: Dict or Namespace with model configuration.

    Returns:
        Configured WaveNO instance.
    """
    return _build_waveno(
        args,
        use_damping=False,
        use_film=True,
        use_cross_seg_attn=False,
    )


def build_waveno_bare(args: dict) -> WaveNO:
    """Bare minimum: unbiased cross-attention, no extras."""
    return _build_waveno(
        args, pde=None, use_film=False, use_cross_seg_attn=False
    )


def build_waveno_bias_only(args: dict) -> WaveNO:
    """+ PDE bias (without damping)."""
    return _build_waveno(
        args,
        use_damping=False,
        use_film=False,
        use_cross_seg_attn=False,
    )


def build_waveno_bias_damp(args: dict) -> WaveNO:
    """+ PDE bias + collision-time damping."""
    return _build_waveno(
        args,
        use_damping=True,
        use_film=False,
        use_cross_seg_attn=False,
    )


def build_waveno_damp(args: dict) -> WaveNO:
    """+ PDE bias + damping + FiLM time conditioning."""
    return _build_waveno(
        args, use_damping=True, use_film=True, use_cross_seg_attn=False
    )


def build_waveno_damp_cross_attn(args: dict) -> WaveNO:
    """+ PDE bias + damping + cross-segment attention."""
    return _build_waveno(
        args, use_damping=True, use_film=False, use_cross_seg_attn=True
    )


def build_waveno_all(args: dict) -> WaveNO:
    """All components except trajectories."""
    return _build_waveno(
        args, use_damping=True, use_film=True, use_cross_seg_attn=True
    )


def build_waveno_film_only(args: dict) -> WaveNO:
    """FiLM only (no PDE bias)."""
    return _build_waveno(
        args, pde=None, use_film=True, use_cross_seg_attn=False
    )


def build_waveno_cross_attn_only(args: dict) -> WaveNO:
    """Cross-segment attention only (no PDE bias)."""
    return _build_waveno(
        args, pde=None, use_film=False, use_cross_seg_attn=True
    )
