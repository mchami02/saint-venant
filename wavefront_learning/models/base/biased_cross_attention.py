"""Biased cross-attention for physics-informed attention mechanisms.

Contains:
- BiasedCrossDecoderLayer: CrossDecoderLayer with additive attention bias (attn_mask).
- compute_characteristic_bias: Backward-characteristic attention bias from IC geometry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
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
        self.drop = nn.Dropout(dropout)

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
        x = x + self.drop(att)

        ff = self.feedforward(self.norm_ff(x))
        x = x + self.drop(ff)

        return x


def compute_characteristic_bias(
    t_coords: torch.Tensor,
    x_coords: torch.Tensor,
    xs: torch.Tensor,
    ks: torch.Tensor,
    pieces_mask: torch.Tensor,
    flux: Flux,
    scale: nn.Parameter,
    damping_sharpness: nn.Parameter | None = None,
) -> torch.Tensor:
    """Compute backward-characteristic attention bias.

    For each query (t, x) and segment k, traces the backward characteristic
    foot y* = x - f'(rho_k) * t and measures how far y* falls outside
    segment k's interval [x_k, x_{k+1}]. Queries in k's characteristic
    cone get bias=0 (full attention); distant queries get large negative
    bias (suppressed attention).

    When damping_sharpness is provided, the bias is dampened after each
    segment's estimated collision time. Before collision the bias is full
    (preserves high-res performance); after collision the bias fades so
    learned attention takes over (fixes multi-step generalization).

    Uses only flux.derivative() -- works for any flux function.

    Args:
        t_coords: (B, nt, nx) time coordinates.
        x_coords: (B, nt, nx) space coordinates.
        xs: (B, K+1) breakpoint positions.
        ks: (B, K) piece values.
        pieces_mask: (B, K) validity mask.
        flux: Flux instance.
        scale: Learnable scale parameter (initialized ~10).
        damping_sharpness: Learnable sharpness for collision-time damping.
            None = no damping (backward compat).

    Returns:
        Attention bias (B, nt, nx, K), negative values suppress attention.
    """
    B, nt, nx = t_coords.shape
    K = ks.shape[1]

    # Characteristic speed per segment: (B, K) and (B, 1, 1, K)
    lambda_k_flat = flux.derivative(ks)  # (B, K)
    lambda_k = lambda_k_flat.unsqueeze(1).unsqueeze(1)

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

    # Collision-time damping: fade bias after estimated wave collision
    if damping_sharpness is not None:
        widths = xs[:, 1:] - xs[:, :-1]  # (B, K)

        # Left-neighbor collision time
        lam_left = F.pad(lambda_k_flat[:, :-1], (1, 0))
        lam_left[:, 0] = lambda_k_flat[:, 0]
        dx_left = F.pad(widths[:, :-1], (1, 0))
        dx_left[:, 0] = widths[:, 0]
        speed_diff_left = (lam_left - lambda_k_flat).abs().clamp(min=1e-3)

        # Right-neighbor collision time
        lam_right = F.pad(lambda_k_flat[:, 1:], (0, 1))
        lam_right[:, -1] = lambda_k_flat[:, -1]
        dx_right = F.pad(widths[:, 1:], (0, 1))
        dx_right[:, -1] = widths[:, -1]
        speed_diff_right = (lambda_k_flat - lam_right).abs().clamp(min=1e-3)

        t_coll = torch.minimum(
            dx_left / speed_diff_left,
            dx_right / speed_diff_right,
        )  # (B, K)

        t_coll_exp = t_coll.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, K)
        t_exp_broad = t_coords.unsqueeze(-1)  # (B, nt, nx, 1)
        damping = torch.sigmoid(
            damping_sharpness.abs() * (t_coll_exp - t_exp_broad)
        )  # (B, nt, nx, K) — ~1 before collision, ~0 after

        bias = bias * damping

    # Mask padded segments with large negative value
    mask = pieces_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, K)
    bias = bias * mask + (~mask.bool()).float() * (-1e9)

    return bias  # (B, nt, nx, K)


def compute_discontinuity_characteristic_bias(
    t_coords: torch.Tensor,
    x_coords: torch.Tensor,
    disc_positions: torch.Tensor,
    disc_rhoL: torch.Tensor,
    disc_rhoR: torch.Tensor,
    disc_mask: torch.Tensor,
    flux: Flux,
    scale: nn.Parameter,
    damping_sharpness: nn.Parameter | None = None,
) -> torch.Tensor:
    """Backward-characteristic attention bias for discontinuity-based models.

    For each query (t, x) and discontinuity d at position x_d with states
    (rho_L, rho_R), computes the influence zone from characteristic speeds
    and penalizes queries outside this zone. Queries inside the zone get
    bias=0 (full attention); distant queries get large negative bias
    (suppressed attention).

    Analogous to compute_characteristic_bias but operates on discontinuities
    (D dimension) instead of segments (K dimension). Each discontinuity has
    two states (rho_L, rho_R) producing two characteristic speeds, defining
    an influence zone rather than a single characteristic foot.

    Uses only flux.derivative() -- works for any flux function.

    Args:
        t_coords: (B, nt, nx) time coordinates.
        x_coords: (B, nt, nx) space coordinates.
        disc_positions: (B, D) discontinuity x-positions.
        disc_rhoL: (B, D) left density values.
        disc_rhoR: (B, D) right density values.
        disc_mask: (B, D) validity mask.
        flux: Flux instance.
        scale: Learnable scale parameter (initialized ~5).
        damping_sharpness: Learnable sharpness for collision-time damping.
            None = no damping (backward compat).

    Returns:
        Attention bias (B, nt, nx, D), negative values suppress attention.
    """
    B, nt, nx = t_coords.shape
    D = disc_positions.shape[1]

    # Characteristic speeds per discontinuity: (B, D)
    lambda_L = flux.derivative(disc_rhoL)  # (B, D)
    lambda_R = flux.derivative(disc_rhoR)  # (B, D)
    v_min = torch.minimum(lambda_L, lambda_R)  # (B, D)
    v_max = torch.maximum(lambda_L, lambda_R)  # (B, D)

    # Expand for broadcasting: (B, 1, 1, D)
    x_d = disc_positions.unsqueeze(1).unsqueeze(1)
    v_min_exp = v_min.unsqueeze(1).unsqueeze(1)
    v_max_exp = v_max.unsqueeze(1).unsqueeze(1)

    # Query coordinates: (B, nt, nx, 1)
    t_exp = t_coords.unsqueeze(-1)
    x_exp = x_coords.unsqueeze(-1)

    # Influence zone at time t: [x_d + v_min*t, x_d + v_max*t]
    zone_left = x_d + v_min_exp * t_exp  # (B, nt, nx, D)
    zone_right = x_d + v_max_exp * t_exp  # (B, nt, nx, D)

    # Distance outside influence zone
    d_outside = torch.relu(zone_left - x_exp) + torch.relu(x_exp - zone_right)

    # Negative bias: 0 inside zone, large negative outside
    bias = -scale.abs() * d_outside.pow(2)

    # Collision-time damping: fade bias after adjacent discs' waves interact
    if damping_sharpness is not None:
        # Compute raw gaps between consecutive discontinuities
        raw_gap = disc_positions[:, 1:] - disc_positions[:, :-1]  # (B, D-1)
        # Gaps involving padded discs are meaningless; replace with 1.0
        pair_mask = disc_mask[:, :-1] * disc_mask[:, 1:]  # (B, D-1)
        raw_gap = raw_gap * pair_mask + (1.0 - pair_mask) * 1.0

        # Right collision: disc d's zone right edge meets disc (d+1)'s zone left edge
        # t_coll = (x_{d+1} - x_d) / (v_max_d - v_min_{d+1})
        gap_right = F.pad(raw_gap, (0, 1))  # (B, D)
        gap_right[:, -1] = 1.0  # large gap for rightmost disc
        v_min_right_neighbor = F.pad(v_min[:, 1:], (0, 1))
        v_min_right_neighbor[:, -1] = v_min[:, -1]  # self-pad
        speed_diff_right = (v_max - v_min_right_neighbor).clamp(min=1e-3)
        t_coll_right = gap_right / speed_diff_right

        # Left collision: disc (d-1)'s zone right edge meets disc d's zone left edge
        # t_coll = (x_d - x_{d-1}) / (v_max_{d-1} - v_min_d)
        gap_left = F.pad(raw_gap, (1, 0))  # (B, D)
        gap_left[:, 0] = 1.0  # large gap for leftmost disc
        v_max_left_neighbor = F.pad(v_max[:, :-1], (1, 0))
        v_max_left_neighbor[:, 0] = v_max[:, 0]  # self-pad
        speed_diff_left = (v_max_left_neighbor - v_min).clamp(min=1e-3)
        t_coll_left = gap_left / speed_diff_left

        t_coll = torch.minimum(t_coll_left, t_coll_right).clamp(min=0.0)  # (B, D)

        t_coll_exp = t_coll.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D)
        t_exp_broad = t_coords.unsqueeze(-1)  # (B, nt, nx, 1)
        damping = torch.sigmoid(
            damping_sharpness.abs() * (t_coll_exp - t_exp_broad)
        )  # (B, nt, nx, D) — ~1 before collision, ~0 after

        bias = bias * damping

    # Mask padded discontinuities with large negative value
    mask = disc_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D)
    bias = bias * mask + (~mask.bool()).float() * (-1e9)

    return bias  # (B, nt, nx, D)
