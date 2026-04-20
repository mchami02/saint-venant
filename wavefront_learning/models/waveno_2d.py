"""WaveNO2D family for 2D-spatial wavefront learning.

Architecture (parallels 1D WaveNO but over a Cartesian tile grid):

1. **Segment encoder** — encode each of ``K = Kx * Ky`` tiles from
   ``Euler2DPDE.physics_features`` (``(B, Kx, Ky, 9)``) via an MLP.
2. **Self-attention** over the ``K`` flattened tile embeddings.
3. **Query encoder** — Fourier features on ``(t, x, y)``.
4. **2D attention bias** — :class:`Euler2DBias` with ``(Kx - 1) * Ky``
   x-interface wave cones and ``Kx * (Ky - 1)`` y-interface wave cones.
5. **Biased cross-attention** — queries attend over the ``K`` tiles.
6. **Output head** — linear → output dimension (4 for Euler2D).

Variants:
    - ``WaveNO2D``       : bias on (no damping), FiLM off, default.
    - ``WaveNO2DBare``   : no bias, no FiLM.
    - ``WaveNO2DBiasOnly`` : bias on, FiLM off.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.biased_cross_attention import BiasedCrossDecoderLayer
from .base.euler2d_bias import Euler2DBias
from .base.feature_encoders import FourierFeatures
from .base.pde import Euler2DPDE
from .base.transformer_encoder import EncoderLayer


class _Tile2DEncoder(nn.Module):
    """Simple MLP segment encoder for 2D tiles.

    Consumes the ``(B, Kx, Ky, num_physics_features)`` output of
    ``Euler2DPDE.physics_features`` plus relative tile position/size, and
    returns ``(B, Kx * Ky, hidden_dim)`` embeddings.
    """

    def __init__(
        self,
        pde: Euler2DPDE,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.pde = pde
        # Features: physics (9) + tile center (x, y) + tile size (dx, dy) = 13
        in_dim = pde.num_physics_features + 4
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, ic_data: dict[str, torch.Tensor]) -> torch.Tensor:
        phys = self.pde.physics_features(ic_data)  # (B, Kx, Ky, num_phys)
        xs = ic_data["xs"]    # (B, Kx+1)
        ys = ic_data["ys"]    # (B, Ky+1)
        B, Kx, Ky, _ = phys.shape

        x_center = 0.5 * (xs[:, :-1] + xs[:, 1:])  # (B, Kx)
        y_center = 0.5 * (ys[:, :-1] + ys[:, 1:])  # (B, Ky)
        dx_tile = xs[:, 1:] - xs[:, :-1]           # (B, Kx)
        dy_tile = ys[:, 1:] - ys[:, :-1]           # (B, Ky)

        x_c = x_center[:, :, None].expand(B, Kx, Ky)
        y_c = y_center[:, None, :].expand(B, Kx, Ky)
        dx_c = dx_tile[:, :, None].expand(B, Kx, Ky)
        dy_c = dy_tile[:, None, :].expand(B, Kx, Ky)

        tile_features = torch.cat(
            [phys,
             x_c.unsqueeze(-1), y_c.unsqueeze(-1),
             dx_c.unsqueeze(-1), dy_c.unsqueeze(-1)],
            dim=-1,
        )  # (B, Kx, Ky, in_dim)
        seg = self.mlp(tile_features)                # (B, Kx, Ky, H)
        return seg.reshape(B, Kx * Ky, seg.shape[-1])


class WaveNO2D(nn.Module):
    """2D WaveNO for Euler2D."""

    def __init__(
        self,
        pde: Euler2DPDE | None,
        encoder_pde: Euler2DPDE,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_freq: int = 8,
        num_self_attn_layers: int = 2,
        num_cross_layers: int = 2,
        num_seg_mlp_layers: int = 2,
        dropout: float = 0.05,
        output_dim: int = 4,
        output_clamp: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.pde = pde
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.output_clamp = output_clamp

        self.segment_encoder = _Tile2DEncoder(
            pde=encoder_pde,
            hidden_dim=hidden_dim,
            num_layers=num_seg_mlp_layers,
            dropout=dropout,
        )
        self.self_attn_layers = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_self_attn_layers)
            ]
        )

        self.fourier_t = FourierFeatures(num_frequencies=num_freq, include_input=True)
        self.fourier_x = FourierFeatures(num_frequencies=num_freq, include_input=True)
        self.fourier_y = FourierFeatures(num_frequencies=num_freq, include_input=True)
        q_in = (
            self.fourier_t.output_dim
            + self.fourier_x.output_dim
            + self.fourier_y.output_dim
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(q_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cross_attn_layers = nn.ModuleList(
            [
                BiasedCrossDecoderLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_cross_layers)
            ]
        )

        if pde is not None:
            self.bias_module = Euler2DBias(pde=pde)
        else:
            self.bias_module = None

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, batch_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        xs = batch_input["xs"]
        ys = batch_input["ys"]
        ks = batch_input["ks"]
        ks_u = batch_input["ks_u"]
        ks_v = batch_input["ks_v"]
        ks_p = batch_input["ks_p"]
        pieces_mask = batch_input["pieces_mask"]   # (B, Kx, Ky)
        t_coords = batch_input["t_coords"]         # (B, nt, ny, nx)
        x_coords = batch_input["x_coords"]
        y_coords = batch_input["y_coords"]

        B, nt, ny, nx = t_coords.shape
        Kx, Ky = pieces_mask.shape[1], pieces_mask.shape[2]
        K = Kx * Ky

        ic_data = {
            "xs": xs, "ys": ys,
            "ks": ks, "ks_u": ks_u, "ks_v": ks_v, "ks_p": ks_p,
            "pieces_mask": pieces_mask,
        }

        # Stage 1: segment encoding + self-attention
        seg = self.segment_encoder(ic_data)  # (B, K, H)
        mask_flat = pieces_mask.reshape(B, K)
        key_padding_mask = ~mask_flat.bool()
        for layer in self.self_attn_layers:
            seg = layer(seg, key_padding_mask=key_padding_mask)
        seg = seg * mask_flat.unsqueeze(-1)

        # Stage 2: query encoding
        N = nt * ny * nx
        t_flat = t_coords.reshape(B * N)
        x_flat = x_coords.reshape(B * N)
        y_flat = y_coords.reshape(B * N)
        t_enc = self.fourier_t(t_flat)
        x_enc = self.fourier_x(x_flat)
        y_enc = self.fourier_y(y_flat)
        q = self.query_mlp(torch.cat([t_enc, x_enc, y_enc], dim=-1))
        q = q.reshape(B, N, self.hidden_dim)

        # Stage 3: attention bias (if enabled)
        if self.bias_module is not None:
            bias = self.bias_module(ic_data, (t_coords, x_coords, y_coords))
            # bias: (B, nt, ny, nx, K) → (B, N, K) → expand heads
            bias_flat = bias.reshape(B, N, K)
            attn_mask = (
                bias_flat.unsqueeze(1)
                .expand(B, self.num_heads, N, K)
                .reshape(B * self.num_heads, N, K)
            )
        else:
            attn_mask = None

        # Stage 4: cross-attention (segments are key/value, queries attend)
        kv = seg  # (B, K, H)
        kv_padding_mask = ~mask_flat.bool()
        for layer in self.cross_attn_layers:
            q = layer(q, kv, key_padding_mask=kv_padding_mask, attn_mask=attn_mask)

        # Stage 5: output head → (B, 4, nt, ny, nx)
        out = self.output_head(q)  # (B, N, 4)
        if self.output_clamp is not None:
            out = out.clamp(*self.output_clamp)
        out = out.reshape(B, nt, ny, nx, self.output_dim)
        out = out.permute(0, 4, 1, 2, 3).contiguous()
        return {"output_grid": out}


def _build_euler2d_pde(args: dict) -> Euler2DPDE:
    return Euler2DPDE(gamma=args.get("euler_gamma") or 1.4)


def _build_waveno_2d(args: dict, with_bias: bool) -> WaveNO2D:
    if not isinstance(args, dict):
        args = vars(args)
    encoder_pde = _build_euler2d_pde(args)
    pde = encoder_pde if with_bias else None
    return WaveNO2D(
        pde=pde,
        encoder_pde=encoder_pde,
        hidden_dim=args.get("hidden_dim", 64),
        num_heads=args.get("num_heads", 4),
        num_freq=args.get("num_freq", 8),
        num_self_attn_layers=args.get("num_self_attn_layers", 2),
        num_cross_layers=args.get("num_cross_layers", 2),
        num_seg_mlp_layers=args.get("num_seg_mlp_layers", 2),
        dropout=args.get("dropout", 0.05),
        output_dim=encoder_pde.output_dim,
        output_clamp=encoder_pde.output_clamp,
    )


def build_waveno_2d(args: dict) -> WaveNO2D:
    """Default WaveNO2D (with bias)."""
    return _build_waveno_2d(args, with_bias=True)


def build_waveno_bare_2d(args: dict) -> WaveNO2D:
    """WaveNO2D without bias — plain cross-attention."""
    return _build_waveno_2d(args, with_bias=False)


def build_waveno_bias_only_2d(args: dict) -> WaveNO2D:
    """WaveNO2D with bias only (no FiLM).  Equivalent to the default for 2D."""
    return _build_waveno_2d(args, with_bias=True)
