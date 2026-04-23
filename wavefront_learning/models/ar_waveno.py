"""ARWaveNO: decoder-only autoregressive block predictor.

Takes a grid row at some timestep and predicts the next ``k`` rows in a
single forward pass. Two variants:

- ``ARWaveNOBias``: physics-informed self-attention bias computed from
  the *input state row* at each step (cell-wise Riemann problems between
  adjacent cells).
- ``ARWaveNOBare``: same architecture, no bias.

Architecture:
    1. Segment-encode the current state row as ``nx`` cell tokens.
    2. Fourier-encode the ``k * nx`` output query coordinates as output
       tokens (relative time ``[1..k] * dt``, spatial coords unchanged).
    3. Concatenate into one sequence of length ``N = nx + k * nx``.
    4. Stack of biased self-attention layers. The bias (if enabled)
       populates the ``(out, cell)`` sub-block of the ``(N, N)`` mask;
       elsewhere the mask is zero.
    5. Output head on the last ``k * nx`` tokens → ``(B, output_dim, k, nx)``.

No FiLM, no damping — bias-only or bare, matching the cleanest narrative
for the compute-efficiency ablation.
"""

import torch
import torch.nn as nn

from .base.characteristic_features import SegmentPhysicsEncoder
from .base.feature_encoders import FourierFeatures
from .base.pde import ARZPDE, LWRPDE, PDE, BurgersPDE, EulerPDE
from .base.pde_bias import PDEBias
from .base.self_attention_biased import BiasedSelfAttentionLayer


class ARWaveNO(nn.Module):
    """Decoder-only block predictor with cell-wise Riemann bias.

    Args:
        k: Number of output timesteps per forward pass.
        hidden_dim: Token embedding dimension.
        num_freq_t: Fourier frequencies for time (relative).
        num_freq_x: Fourier frequencies for space.
        num_seg_frequencies: Fourier frequencies for the cell-state encoder.
        num_seg_mlp_layers: MLP depth inside the cell-state encoder.
        num_layers: Self-attention layer depth.
        num_heads: Attention heads.
        pde: PDE instance driving ``PDEBias``. ``None`` → bare (no bias).
        encoder_pde: PDE instance driving the cell-state physics features.
            Defaults to ``pde`` when set; otherwise LWR.
        dropout: Dropout rate.
        output_dim: Number of output channels (defaults to PDE's).
        output_clamp: Optional output clamp range.
    """

    is_autoregressive_block = True

    def __init__(
        self,
        k: int,
        hidden_dim: int = 64,
        num_freq_t: int | None = 8,
        num_freq_x: int | None = 8,
        num_seg_frequencies: int | None = 8,
        num_seg_mlp_layers: int = 2,
        num_layers: int = 4,
        num_heads: int = 4,
        pde: PDE | None = None,
        encoder_pde: PDE | None = None,
        dropout: float = 0.05,
        output_dim: int | None = None,
        output_clamp: tuple[float, float] | None | str = "auto",
    ):
        super().__init__()

        self.k = k
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pde = pde

        # Derive output config from PDE metadata (matches WaveNO's pattern).
        seg_pde = encoder_pde if encoder_pde is not None else (pde or LWRPDE())
        if output_dim is not None:
            self.output_dim = output_dim
        else:
            self.output_dim = seg_pde.output_dim

        if output_clamp != "auto":
            self.output_clamp = output_clamp
        else:
            self.output_clamp = seg_pde.output_clamp

        # Cell-state encoder (reuses segment encoder with nx "segments").
        self.cell_encoder = SegmentPhysicsEncoder(
            hidden_dim=hidden_dim,
            num_frequencies=num_seg_frequencies,
            num_layers=num_seg_mlp_layers,
            pde=seg_pde,
            include_cumulative_mass=True,
            dropout=dropout,
        )

        # Output query encoder: Fourier(t_rel, x) → MLP → hidden_dim.
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

        self.query_mlp = nn.Sequential(
            nn.Linear(query_t_dim + query_x_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Riemann-problem bias (optional).
        if pde is not None:
            self.pde_bias = PDEBias(
                pde=pde,
                use_damping=False,
            )

        # Self-attention stack.
        self.layers = nn.ModuleList(
            [
                BiasedSelfAttentionLayer(
                    hidden_dim, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        # Output head on the output-token slice.
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        if self.output_dim == 1:
            if self.output_clamp is not None:
                lo, hi = self.output_clamp
                init_bias = 0.5 * (lo + hi)
            else:
                init_bias = 0.0
            nn.init.constant_(self.output_head[-1].bias, init_bias)
        else:
            nn.init.zeros_(self.output_head[-1].bias)

    @staticmethod
    def _cell_ic_data(
        state_row: torch.Tensor,
        dx: torch.Tensor,
        nx: int,
    ) -> dict[str, torch.Tensor]:
        """Build an ic_data dict treating each grid cell as its own piece.

        Args:
            state_row: (B, C, nx) current state at ``t_start``.
            dx: (B,) or scalar spatial step.
            nx: Spatial grid size.

        Returns:
            Dict with ``xs`` (B, nx+1), ``ks`` (B, nx), ``pieces_mask``
            (B, nx), and ``ks_v`` / ``ks_p`` for multi-channel PDEs.
        """
        B = state_row.shape[0]
        device = state_row.device
        dtype = state_row.dtype

        # dx may be a scalar tensor per batch element or a single scalar.
        if dx.dim() == 0:
            dx_b = dx.expand(B)
        else:
            dx_b = dx
        dx_b = dx_b.to(dtype=dtype, device=device)

        # Cell edges: (B, nx+1) = dx * [0, 1, ..., nx]
        edge_idx = torch.arange(nx + 1, device=device, dtype=dtype)
        xs = dx_b.unsqueeze(1) * edge_idx.unsqueeze(0)  # (B, nx+1)

        ks = state_row[:, 0, :]  # (B, nx)
        pieces_mask = torch.ones(B, nx, device=device, dtype=dtype)
        ic = {"xs": xs, "ks": ks, "pieces_mask": pieces_mask}
        if state_row.shape[1] >= 2:
            ic["ks_v"] = state_row[:, 1, :]
        if state_row.shape[1] >= 3:
            ic["ks_p"] = state_row[:, 2, :]
        return ic

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Required input keys:
            - ``state_row``: (B, output_dim, nx) state at ``t_start``.
            - ``t_coords``: (B, 1, k, nx) relative times (``[1..k] * dt``).
            - ``x_coords``: (B, 1, k, nx) spatial coords.
            - ``dx``: scalar tensor or (B,) for cell-edge construction.

        Returns:
            ``{"output_grid": (B, output_dim, k, nx),
               "characteristic_bias": (B, k, nx, nx) if pde}``.
        """
        state_row = batch_input["state_row"]  # (B, C, nx)
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, k, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, k, nx)
        dx = batch_input["dx"]

        B, nt, nx = t_coords.shape
        assert nt == self.k, f"t_coords has nt={nt}, expected k={self.k}"

        # Cell-wise ic_data (current-state-as-IC).
        ic_data = self._cell_ic_data(state_row, dx, nx)

        # Stage 1: encode cell tokens from the current state.
        cell_emb = self.cell_encoder(ic_data)  # (B, nx, H)

        # Stage 2: encode output query tokens (k * nx positions).
        Q = self.k * nx
        t_flat = t_coords.reshape(-1)  # (B*k*nx,)
        x_flat = x_coords.reshape(-1)
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
        query_emb = self.query_mlp(query_features).reshape(
            B, Q, self.hidden_dim
        )  # (B, k*nx, H)

        # Stage 3: concatenate into one sequence.
        tokens = torch.cat([cell_emb, query_emb], dim=1)  # (B, nx + k*nx, H)
        N = tokens.shape[1]

        # Stage 4: build the self-attention bias sub-block.
        char_bias = None
        attn_mask = None
        if self.pde is not None:
            char_bias = self.pde_bias(
                ic_data, (t_coords, x_coords)
            )  # (B, k, nx, nx) — (*, cell) per-query bias
            sub = char_bias.reshape(B, Q, nx)  # (B, k*nx, nx)

            # Full (B, N, N) mask — zero everywhere except (out, cell).
            full_mask = torch.zeros(
                B, N, N, device=tokens.device, dtype=sub.dtype
            )
            full_mask[:, nx:, :nx] = sub
            attn_mask = (
                full_mask.unsqueeze(1)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(B * self.num_heads, N, N)
            )

        # Stage 5: self-attention stack.
        x = tokens
        for layer in self.layers:
            x = layer(x, key_padding_mask=None, attn_mask=attn_mask)

        # Stage 6: output head on the last k*nx tokens.
        out_tokens = x[:, nx:, :]  # (B, k*nx, H)
        out = self.output_head(out_tokens)  # (B, k*nx, output_dim)
        if self.output_clamp is not None:
            out = out.clamp(*self.output_clamp)
        output_grid = out.reshape(B, self.k, nx, self.output_dim).permute(
            0, 3, 1, 2
        )  # (B, output_dim, k, nx)

        result = {"output_grid": output_grid}
        if char_bias is not None:
            result["characteristic_bias"] = char_bias
        return result


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _build_pde(args: dict) -> PDE:
    """Mirror of ``models.waveno._build_pde`` — keeps ARWaveNO decoupled."""
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


def _build_ar_waveno(args: dict, **flag_overrides) -> ARWaveNO:
    """Shared builder for ARWaveNO variants."""
    if not isinstance(args, dict):
        args = vars(args)

    if "pde" in flag_overrides:
        pde = flag_overrides.pop("pde")
    else:
        pde = _build_pde(args)

    encoder_pde = _build_pde(args)

    kwargs = dict(
        k=args.get("ar_block_k", 1),
        hidden_dim=args.get("hidden_dim", 64),
        num_freq_t=args.get("num_freq_t", 8),
        num_freq_x=args.get("num_freq_x", 8),
        num_seg_frequencies=args.get("num_seg_frequencies", 8),
        num_seg_mlp_layers=args.get("num_seg_mlp_layers", 2),
        num_layers=args.get("num_ar_layers", 4),
        num_heads=args.get("num_heads", 4),
        pde=pde,
        encoder_pde=encoder_pde,
        dropout=args.get("dropout", 0.05),
        output_dim=encoder_pde.output_dim,
        output_clamp=encoder_pde.output_clamp,
    )
    kwargs.update(flag_overrides)
    return ARWaveNO(**kwargs)


def build_ar_waveno_bias(args: dict) -> ARWaveNO:
    """Factory: ARWaveNO with Riemann-problem bias (from input state row)."""
    return _build_ar_waveno(args)


def build_ar_waveno_bare(args: dict) -> ARWaveNO:
    """Factory: ARWaveNO without bias (ablation baseline)."""
    return _build_ar_waveno(args, pde=None)
