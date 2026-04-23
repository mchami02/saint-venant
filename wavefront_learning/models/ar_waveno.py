"""ARWaveNO: decoder-only autoregressive block predictor.

Takes the last ``k_in`` grid rows ending at ``t_start`` and predicts the
next ``k`` rows in a single forward pass. Two variants:

- ``ARWaveNOBias``: physics-informed self-attention bias computed from
  the **current state row** (last row of the history) at each step
  (cell-wise Riemann problems between adjacent cells).
- ``ARWaveNOBare``: same architecture, no bias.
- ``ARWaveNOLinear``: same architecture, ALiBi-style spatial-distance bias.

Architecture:
    1. Segment-encode each of the ``k_in`` history rows as ``nx`` cell
       tokens (each token is enriched with a Fourier-encoded
       in-history time offset).
    2. Fourier-encode the ``k * nx`` output query coordinates
       (relative time ``[1..k] * dt``) as output tokens.
    3. Concatenate into one sequence of length ``N = k_in * nx + k * nx``.
    4. Stack of biased self-attention layers. The bias (if enabled)
       populates the ``(out, last-hist-row)`` sub-block of the ``(N, N)``
       mask; elsewhere the mask is zero.
    5. Output head on the last ``k * nx`` tokens → ``(B, output_dim, k, nx)``.

No FiLM, no damping — bias-only or bare.
"""

import torch
import torch.nn as nn

from .base.characteristic_features import SegmentPhysicsEncoder
from .base.feature_encoders import FourierFeatures
from .base.linear_bias import LinearBias
from .base.pde import ARZPDE, LWRPDE, PDE, BurgersPDE, EulerPDE
from .base.pde_bias import PDEBias
from .base.self_attention_biased import BiasedSelfAttentionLayer


class ARWaveNO(nn.Module):
    """Decoder-only block predictor with k-step history + cell-wise bias.

    Args:
        k: Number of output timesteps per forward pass (``k_out``).
        k_in: Number of input timesteps (history length). Defaults to
            ``k`` (matched input/output length).
        hidden_dim: Token embedding dimension.
        num_freq_t / num_freq_x / num_seg_frequencies: Fourier bands.
        num_seg_mlp_layers: MLP depth inside the per-row cell encoder.
        num_layers: Self-attention layer depth.
        num_heads: Attention heads.
        pde: PDE instance driving ``PDEBias`` (or ``None`` for bare).
        encoder_pde: PDE for the cell-state physics features (defaults
            to ``pde`` when set, else LWR).
        bias_kind: "pde" | "linear" | "none".
        linear_bias_initial_scale: Initial scale for the linear bias.
        dropout: Dropout rate.
        output_dim: Number of output channels (defaults to PDE's).
        output_clamp: Optional output clamp range.
    """

    is_autoregressive_block = True

    def __init__(
        self,
        k: int,
        k_in: int | None = None,
        hidden_dim: int = 64,
        num_freq_t: int | None = 8,
        num_freq_x: int | None = 8,
        num_seg_frequencies: int | None = 8,
        num_seg_mlp_layers: int = 2,
        num_layers: int = 4,
        num_heads: int = 4,
        pde: PDE | None = None,
        encoder_pde: PDE | None = None,
        bias_kind: str = "pde",
        linear_bias_initial_scale: float = 1.0,
        dropout: float = 0.05,
        output_dim: int | None = None,
        output_clamp: tuple[float, float] | None | str = "auto",
    ):
        super().__init__()

        self.k = k
        self.k_in = k if k_in is None or k_in <= 0 else k_in
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pde = pde
        self.bias_kind = bias_kind

        # Derive output config from PDE metadata.
        seg_pde = encoder_pde if encoder_pde is not None else (pde or LWRPDE())
        if output_dim is not None:
            self.output_dim = output_dim
        else:
            self.output_dim = seg_pde.output_dim

        if output_clamp != "auto":
            self.output_clamp = output_clamp
        else:
            self.output_clamp = seg_pde.output_clamp

        # Per-row cell-state encoder: reused across all k_in history rows.
        self.cell_encoder = SegmentPhysicsEncoder(
            hidden_dim=hidden_dim,
            num_frequencies=num_seg_frequencies,
            num_layers=num_seg_mlp_layers,
            pde=seg_pde,
            include_cumulative_mass=True,
            dropout=dropout,
        )

        # In-history time positional encoding (Fourier on [-(k_in-1)..0]*dt).
        # Added on top of cell_emb to distinguish rows by their age.
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

        # Projects a time-in-history scalar → hidden_dim so it can be
        # added to the per-row cell embeddings.
        self.hist_t_proj = nn.Linear(query_t_dim, hidden_dim)

        # Output query encoder: Fourier(t_rel, x) → MLP → hidden_dim.
        self.query_mlp = nn.Sequential(
            nn.Linear(query_t_dim + query_x_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention bias module (operates on the *current* row only).
        self.bias_module: nn.Module | None
        if bias_kind == "pde" and pde is not None:
            self.bias_module = PDEBias(pde=pde, use_damping=False)
        elif bias_kind == "linear":
            self.bias_module = LinearBias(
                initial_scale=linear_bias_initial_scale
            )
        else:
            self.bias_module = None

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
        """Build an ic_data dict treating each grid cell as a piece.

        Args:
            state_row: (B, C, nx) current state row.
            dx: scalar or (B,) spatial step.
            nx: spatial grid size.
        """
        B = state_row.shape[0]
        device = state_row.device
        dtype = state_row.dtype

        if dx.dim() == 0:
            dx_b = dx.expand(B)
        else:
            dx_b = dx
        dx_b = dx_b.to(dtype=dtype, device=device)

        edge_idx = torch.arange(nx + 1, device=device, dtype=dtype)
        xs = dx_b.unsqueeze(1) * edge_idx.unsqueeze(0)  # (B, nx+1)

        ks = state_row[:, 0, :]
        pieces_mask = torch.ones(B, nx, device=device, dtype=dtype)
        ic = {"xs": xs, "ks": ks, "pieces_mask": pieces_mask}
        if state_row.shape[1] >= 2:
            ic["ks_v"] = state_row[:, 1, :]
        if state_row.shape[1] >= 3:
            ic["ks_p"] = state_row[:, 2, :]
        return ic

    def _encode_history(
        self,
        state_hist: torch.Tensor,
        hist_t_coords: torch.Tensor,
        dx: torch.Tensor,
    ) -> torch.Tensor:
        """Encode k_in history rows into (B, k_in * nx, H) tokens.

        Each row gets its own cell embedding (via the shared
        ``SegmentPhysicsEncoder``) plus a row-level additive time
        positional encoding so the model can tell rows apart by age.

        Args:
            state_hist: (B, C, k_in, nx) history block.
            hist_t_coords: (B, 1, k_in, nx) relative past times.
            dx: (B,) or scalar.

        Returns:
            (B, k_in * nx, H) token tensor.
        """
        B, C, k_in, nx = state_hist.shape
        H = self.hidden_dim

        # Per-row cell embeddings: fold k_in into the batch dim.
        state_flat = state_hist.permute(0, 2, 1, 3).reshape(B * k_in, C, nx)
        if dx.dim() == 0:
            dx_rep = dx.expand(B * k_in)
        else:
            dx_rep = dx.unsqueeze(1).expand(B, k_in).reshape(-1)
        ic_flat = self._cell_ic_data(state_flat, dx_rep, nx)
        cell_emb = self.cell_encoder(ic_flat)  # (B*k_in, nx, H)
        cell_emb = cell_emb.reshape(B, k_in, nx, H)

        # Row-level time positional encoding (uses the t value at column 0).
        t_row = hist_t_coords[:, 0, :, 0]  # (B, k_in)
        t_flat = t_row.reshape(-1)  # (B*k_in,)
        t_enc = (
            self.fourier_t(t_flat)
            if self.fourier_t is not None
            else t_flat.unsqueeze(-1)
        )
        t_emb = self.hist_t_proj(t_enc).reshape(B, k_in, 1, H)

        # Add and flatten time+space into token dim.
        cell_emb = cell_emb + t_emb
        hist_tokens = cell_emb.reshape(B, k_in * nx, H)
        return hist_tokens

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Required keys in ``batch_input``:
            - ``state_hist``: (B, C, k_in, nx) past rows ending at t_start.
            - ``hist_t_coords``: (B, 1, k_in, nx) past-relative times
              ``[-(k_in-1)..0] * dt``.
            - ``t_coords``: (B, 1, k, nx) future-relative times ``[1..k] * dt``.
            - ``x_coords``: (B, 1, k, nx) spatial coords.
            - ``dx``: scalar or (B,) for cell-edge construction.

        Returns:
            ``{"output_grid": (B, output_dim, k, nx),
               "characteristic_bias": (B, k, nx, nx) if bias_module}``.
        """
        state_hist = batch_input["state_hist"]  # (B, C, k_in, nx)
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, k, nx)
        x_coords = batch_input["x_coords"].squeeze(1)
        hist_t_coords = batch_input["hist_t_coords"]  # (B, 1, k_in, nx)
        dx = batch_input["dx"]

        B, nt, nx = t_coords.shape
        assert nt == self.k, f"t_coords has nt={nt}, expected k={self.k}"
        assert state_hist.shape[2] == self.k_in, (
            f"state_hist has k_in={state_hist.shape[2]}, "
            f"expected {self.k_in}"
        )

        # Stage 1: encode k_in history rows → (B, k_in * nx, H) tokens.
        hist_tokens = self._encode_history(state_hist, hist_t_coords, dx)

        # Stage 2: encode output query tokens (k * nx positions).
        Q = self.k * nx
        t_flat = t_coords.reshape(-1)
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

        # Stage 3: concatenate: [hist_tokens, query_emb] → (B, N, H)
        tokens = torch.cat([hist_tokens, query_emb], dim=1)
        N = tokens.shape[1]  # N = k_in*nx + k*nx

        # Stage 4: bias sub-block from the LAST history row → queries.
        # The bias uses the current state (last row) via _cell_ic_data.
        char_bias = None
        attn_mask = None
        if self.bias_module is not None:
            current_state = state_hist[:, :, -1, :]  # (B, C, nx) last hist row
            current_ic = self._cell_ic_data(current_state, dx, nx)
            char_bias = self.bias_module(
                current_ic, (t_coords, x_coords)
            )  # (B, k, nx, nx)
            sub = char_bias.reshape(B, Q, nx)  # (B, k*nx, nx)

            full_mask = torch.zeros(
                B, N, N, device=tokens.device, dtype=sub.dtype
            )
            # Queries (rows) = last k*nx tokens.
            # Keys (cols) = last-hist-row columns = [nx*(k_in-1) : nx*k_in].
            q_start = self.k_in * nx
            k_start = (self.k_in - 1) * nx
            k_end = self.k_in * nx
            full_mask[:, q_start:, k_start:k_end] = sub

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
        out_tokens = x[:, self.k_in * nx :, :]  # (B, k*nx, H)
        out = self.output_head(out_tokens)
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

    # History length: default -> output block length.
    k_out = args.get("ar_block_k", 1)
    k_in_raw = args.get("ar_hist_k", -1)
    k_in = k_out if k_in_raw is None or k_in_raw <= 0 else k_in_raw

    kwargs = dict(
        k=k_out,
        k_in=k_in,
        hidden_dim=args.get("hidden_dim", 64),
        num_freq_t=args.get("num_freq_t", 8),
        num_freq_x=args.get("num_freq_x", 8),
        num_seg_frequencies=args.get("num_seg_frequencies", 8),
        num_seg_mlp_layers=args.get("num_seg_mlp_layers", 2),
        num_layers=args.get("num_ar_layers", 4),
        num_heads=args.get("num_heads", 4),
        pde=pde,
        encoder_pde=encoder_pde,
        bias_kind=args.get("bias_kind", "pde"),
        linear_bias_initial_scale=args.get("linear_bias_initial_scale", 1.0),
        dropout=args.get("dropout", 0.05),
        output_dim=encoder_pde.output_dim,
        output_clamp=encoder_pde.output_clamp,
    )
    kwargs.update(flag_overrides)
    return ARWaveNO(**kwargs)


def build_ar_waveno_bias(args: dict) -> ARWaveNO:
    """Factory: ARWaveNO with Riemann-problem bias (from current state)."""
    return _build_ar_waveno(args, bias_kind="pde")


def build_ar_waveno_bare(args: dict) -> ARWaveNO:
    """Factory: ARWaveNO without bias."""
    return _build_ar_waveno(args, pde=None, bias_kind="none")


def build_ar_waveno_linear(args: dict) -> ARWaveNO:
    """Factory: ARWaveNO with a linear spatial-distance bias."""
    return _build_ar_waveno(args, pde=None, bias_kind="linear")
