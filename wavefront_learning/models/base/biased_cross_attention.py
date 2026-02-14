"""Biased cross-attention for physics-informed attention mechanisms.

Contains:
- BiasedCrossDecoderLayer: CrossDecoderLayer with additive attention bias (attn_mask).
- compute_characteristic_bias: Backward-characteristic attention bias from IC geometry.
"""

import torch
import torch.nn as nn

from .flux import Flux


class BiasedCrossDecoderLayer(nn.Module):
    """Cross-attention decoder layer with additive attention bias support.

    Identical to CrossDecoderLayer but passes an optional attn_mask to
    nn.MultiheadAttention, enabling physics-informed attention biasing
    (analogous to ALiBi in NLP but using characteristic distance).

    Pre-norm architecture:
        x' = x + MHA(LN(x), LN(z), LN(z), attn_mask=bias)
        x  = x' + MLP(LN(x'))

    Args:
        hidden_dim: Feature dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x, z, key_padding_mask=None, attn_mask=None):
        """Forward pass with optional attention bias.

        Args:
            x: (B, Q, D) query features.
            z: (B, S, D) key/value features (segment tokens).
            key_padding_mask: (B, S) True = ignore.
            attn_mask: (B*num_heads, Q, S) additive bias added to attention
                logits before softmax. Negative values suppress attention.

        Returns:
            Updated query features (B, Q, D).
        """
        x_norm = self.norm_q(x)
        z_norm = self.norm_kv(z)
        att = self.cross_attention(
            x_norm,
            z_norm,
            z_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )[0]
        x = x + att

        ff = self.feedforward(self.norm_ff(x))
        x = x + ff

        return x


def compute_characteristic_bias(
    t_coords: torch.Tensor,
    x_coords: torch.Tensor,
    xs: torch.Tensor,
    ks: torch.Tensor,
    pieces_mask: torch.Tensor,
    flux: Flux,
    scale: nn.Parameter,
) -> torch.Tensor:
    """Compute backward-characteristic attention bias.

    For each query (t, x) and segment k, traces the backward characteristic
    foot y* = x - f'(rho_k) * t and measures how far y* falls outside
    segment k's interval [x_k, x_{k+1}]. Queries in k's characteristic
    cone get bias=0 (full attention); distant queries get large negative
    bias (suppressed attention).

    Uses only flux.derivative() -- works for any flux function.

    Args:
        t_coords: (B, nt, nx) time coordinates.
        x_coords: (B, nt, nx) space coordinates.
        xs: (B, K+1) breakpoint positions.
        ks: (B, K) piece values.
        pieces_mask: (B, K) validity mask.
        flux: Flux instance.
        scale: Learnable scale parameter (initialized ~10).

    Returns:
        Attention bias (B, nt, nx, K), negative values suppress attention.
    """
    B, nt, nx = t_coords.shape
    K = ks.shape[1]

    # Characteristic speed per segment: (B, 1, 1, K)
    lambda_k = flux.derivative(ks).unsqueeze(1).unsqueeze(1)

    # Query coordinates: (B, nt, nx, 1)
    t_exp = t_coords.unsqueeze(-1)
    x_exp = x_coords.unsqueeze(-1)

    # Segment boundaries: (B, 1, 1, K)
    x_left = xs[:, :-1].unsqueeze(1).unsqueeze(1)
    x_right = xs[:, 1:].unsqueeze(1).unsqueeze(1)

    # Backward characteristic foot: y* = x - f'(rho_k) * t
    y_star = x_exp - lambda_k * t_exp  # (B, nt, nx, K)

    # Distance outside segment k's interval
    d_outside = torch.relu(x_left - y_star) + torch.relu(y_star - x_right)

    # Negative bias: 0 inside cone, large negative outside
    bias = -scale.abs() * d_outside.pow(2)

    # Mask padded segments with large negative value
    mask = pieces_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, K)
    bias = bias * mask + (~mask.bool()).float() * (-1e9)

    return bias  # (B, nt, nx, K)
