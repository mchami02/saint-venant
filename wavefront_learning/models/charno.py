"""CharNO: Characteristic Neural Operator for hyperbolic conservation laws.

Mirrors the Lax-Hopf variational formula: for each query (t, x), select
the initial segment that "wins" the Lax-Hopf minimization, then predict
the local density value. The selection is implemented as a differentiable
softmin over learned scores, converging to the exact Lax-Hopf argmin as
the temperature decreases.

Architecture:
    1. SegmentPhysicsEncoder + self-attention → contextualized segment embeddings
    2. CharacteristicFeatureComputer → per-(query, segment) features
    3. Score MLP + softmin → selection weights (which segment controls each point)
    4. Value MLP + clamp → local density (what value if that segment controls)
    5. Weighted sum → output density ρ(t, x)
"""

import torch
import torch.nn as nn

from .base.characteristic_features import (
    CharacteristicFeatureComputer,
    SegmentPhysicsEncoder,
    TimeConditioner,
)
from .base.flux import DEFAULT_FLUX, Flux
from .base.transformer_encoder import EncoderLayer


class CrossSegmentAttention(nn.Module):
    """Lightweight self-attention over the K segment dimension.

    Unlike EncoderLayer, this has NO feedforward network — just attention
    + residual + LayerNorm. This saves ~60% memory since the FFN's 4x
    expansion on (B*Q, K, D) tensors is the main OOM culprit.
    The downstream score/value MLPs provide sufficient nonlinear capacity.
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


class CharNO(nn.Module):
    """Characteristic Neural Operator.

    Args:
        hidden_dim: Segment embedding dimension.
        char_hidden_dim: Characteristic feature dimension.
        num_frequencies: Fourier frequency bands for segment encoder.
        num_char_frequencies: Fourier frequency bands for characteristic features.
        num_seg_mlp_layers: MLP depth in segment encoder.
        num_self_attn_layers: Self-attention layers for segment interaction.
        num_char_mlp_layers: MLP depth in characteristic feature computer.
        num_score_layers: MLP depth in score network.
        num_local_layers: MLP depth in value network.
        num_heads: Self-attention heads.
        num_cross_segment_layers: Cross-segment attention layers (0 = disabled).
        time_condition: Enable FiLM time conditioning on segment embeddings.
        initial_temperature: Softmin temperature initialization.
        flux: Flux function instance.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        char_hidden_dim: int = 32,
        num_frequencies: int = 8,
        num_char_frequencies: int = 8,
        num_seg_mlp_layers: int = 2,
        num_self_attn_layers: int = 2,
        num_char_mlp_layers: int = 2,
        num_score_layers: int = 2,
        num_local_layers: int = 2,
        num_heads: int = 4,
        num_cross_segment_layers: int = 1,
        time_condition: bool = True,
        initial_temperature: float = 1.0,
        flux: Flux | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.char_hidden_dim = char_hidden_dim
        self.time_condition = time_condition
        flux = flux or DEFAULT_FLUX()

        # Stage 1: Segment encoder
        self.segment_encoder = SegmentPhysicsEncoder(
            hidden_dim=hidden_dim,
            num_frequencies=num_frequencies,
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

        # Stage 2: Characteristic features
        self.char_features = CharacteristicFeatureComputer(
            hidden_dim=char_hidden_dim,
            num_frequencies=num_char_frequencies,
            num_layers=num_char_mlp_layers,
            flux=flux,
        )

        # Stage 2.5: Time-conditioned segment embeddings (FiLM)
        if time_condition:
            self.time_conditioner = TimeConditioner(
                hidden_dim=hidden_dim,
                num_time_frequencies=num_char_frequencies,
            )

        # Stage 2.75: Cross-segment attention over K per time step
        # Operates on (B*nt, K, H) BEFORE spatial expansion — fully vectorized
        if num_cross_segment_layers > 0:
            self.cross_segment_layers = nn.ModuleList(
                [
                    CrossSegmentAttention(hidden_dim, num_heads=num_heads)
                    for _ in range(num_cross_segment_layers)
                ]
            )
        else:
            self.cross_segment_layers = nn.ModuleList()

        # Stage 3: Score network
        combined_dim = hidden_dim + char_hidden_dim
        score_input_dim = combined_dim
        score_layers = []
        in_dim = score_input_dim
        for i in range(num_score_layers):
            out_dim = hidden_dim if i < num_score_layers - 1 else 1
            score_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_score_layers - 1:
                score_layers.append(nn.GELU())
            in_dim = out_dim
        self.score_mlp = nn.Sequential(*score_layers)

        # Learnable temperature (stored as log for positivity)
        self.log_temperature = nn.Parameter(
            torch.tensor(initial_temperature).log()
        )

        # Stage 4: Value network
        value_input_dim = combined_dim
        value_layers = []
        in_dim = value_input_dim
        for i in range(num_local_layers):
            out_dim = hidden_dim if i < num_local_layers - 1 else 1
            value_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_local_layers - 1:
                value_layers.append(nn.GELU())
            in_dim = out_dim
        self.value_mlp = nn.Sequential(*value_layers)

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
                - 'selection_weights': (B, nt, nx, K) segment weights
        """
        xs = batch_input["xs"]
        ks = batch_input["ks"]
        pieces_mask = batch_input["pieces_mask"]
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, nt, nx)

        B, nt, nx = t_coords.shape
        K = ks.shape[1]
        Q = nt * nx

        # === Stage 1: Encode segments ===
        seg_emb = self.segment_encoder(xs, ks, pieces_mask)  # (B, K, H)

        # Self-attention over segments
        key_padding_mask = ~pieces_mask.bool()  # True = padded
        # Handle fully-masked batches
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        for layer in self.self_attn_layers:
            seg_emb = layer(seg_emb, key_padding_mask=key_padding_mask)
        seg_emb = seg_emb * pieces_mask.unsqueeze(-1)  # re-zero padded

        # === Stage 2: Characteristic features ===
        char_feat = self.char_features(
            t_coords, x_coords, xs, ks, pieces_mask
        )  # (B, Q, K, H_char)

        # === Stage 2.5: Time-conditioned segment embeddings ===
        t_unique = t_coords[:, :, 0]  # (B, nt) — same t for all x
        if self.time_condition:
            seg_emb_t = self.time_conditioner(seg_emb, t_unique)  # (B, nt, K, H)
        else:
            seg_emb_t = seg_emb.unsqueeze(1).expand(B, nt, K, self.hidden_dim)

        # === Stage 2.75: Cross-segment attention per time step ===
        # Operates on (B*nt, K, H) — 50x cheaper than (B*Q, K, H+H_char)
        if len(self.cross_segment_layers) > 0:
            seg_flat = seg_emb_t.reshape(B * nt, K, -1)  # (B*nt, K, H)
            cs_mask = (~pieces_mask.bool()).unsqueeze(1).expand(
                B, nt, K
            ).reshape(B * nt, K)
            all_masked_cs = cs_mask.all(dim=1)
            if all_masked_cs.any():
                cs_mask = cs_mask.clone()
                cs_mask[all_masked_cs] = False
            for layer in self.cross_segment_layers:
                seg_flat = layer(seg_flat, key_padding_mask=cs_mask)
            seg_emb_t = seg_flat.reshape(B, nt, K, self.hidden_dim)
            seg_emb_t = seg_emb_t * pieces_mask.unsqueeze(1).unsqueeze(-1)

        # Expand to full query grid: (B, nt, K, H) → (B, Q, K, H)
        seg_emb_exp = (
            seg_emb_t.unsqueeze(2)
            .expand(-1, -1, nx, -1, -1)
            .reshape(B, Q, K, self.hidden_dim)
        )

        # Concatenate with characteristic features
        combined = torch.cat(
            [seg_emb_exp, char_feat], dim=-1
        )  # (B, Q, K, H + H_char)

        # === Stage 3: Score network ===
        # Compute scores
        scores = self.score_mlp(combined).squeeze(-1)  # (B, Q, K)

        # Softmin: softmax of negated scores / temperature
        temperature = self.log_temperature.exp()
        # Mask padded segments with large score so they get ~0 weight
        # Use large finite value instead of inf to avoid NaN on MPS/softmax
        scores = scores.masked_fill(
            ~pieces_mask.unsqueeze(1).bool(), 1e9
        )
        # Scalable softmax: scale by log(K_eff) so selection sharpness
        # adapts to the actual number of segments (fixes attention fading)
        K_eff = pieces_mask.sum(dim=-1, keepdim=True).float().clamp(min=2)  # (B, 1)
        log_K = torch.log(K_eff).unsqueeze(1)  # (B, 1, 1)
        weights = torch.softmax(-scores * log_K / temperature, dim=-1)  # (B, Q, K)

        # === Stage 4: Value network ===
        local_rho = self.value_mlp(combined).squeeze(-1).clamp(0.0, 1.0)  # (B, Q, K)

        # === Stage 5: Output assembly ===
        # Weighted sum over segments
        rho = (weights * local_rho).sum(dim=-1)  # (B, Q)

        # Reshape to grid
        output_grid = rho.reshape(B, 1, nt, nx)

        # Selection weights for interpretability
        selection_weights = weights.reshape(B, nt, nx, K)

        # Local densities per segment for diagnostics
        local_rho_grid = local_rho.reshape(B, nt, nx, K)

        return {
            "output_grid": output_grid,
            "selection_weights": selection_weights,
            "local_rho": local_rho_grid,
            "temperature": temperature,
        }


def build_charno(args: dict) -> CharNO:
    """Factory function for CharNO.

    Args:
        args: Dict or Namespace with model configuration.

    Returns:
        Configured CharNO instance.
    """
    if not isinstance(args, dict):
        args = vars(args)

    return CharNO(
        hidden_dim=args.get("hidden_dim", 64),
        char_hidden_dim=args.get("char_hidden_dim", 32),
        num_frequencies=args.get("num_frequencies", 8),
        num_char_frequencies=args.get("num_char_frequencies", 8),
        num_seg_mlp_layers=args.get("num_seg_mlp_layers", 2),
        num_self_attn_layers=args.get("num_self_attn_layers", 2),
        num_char_mlp_layers=args.get("num_char_mlp_layers", 2),
        num_score_layers=args.get("num_score_layers", 2),
        num_local_layers=args.get("num_local_layers", 2),
        num_heads=args.get("num_heads", 4),
        num_cross_segment_layers=args.get("num_cross_segment_layers", 1),
        time_condition=args.get("time_condition", True),
        initial_temperature=args.get("initial_temperature", 1.0),
    )
