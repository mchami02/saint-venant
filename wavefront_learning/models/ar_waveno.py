"""ARWaveNO: decoder-only autoregressive block predictor with history.

Takes the last ``k_in`` grid rows ending at ``t_start`` and predicts the
next ``k`` rows in a single forward pass. The model is residual: it
outputs a delta to add to the last history row.

Architecture:
    1. ``hist_mlp(fourier_t(t) ⊕ fourier_x(x) ⊕ state_value)`` encodes
       each (row, cell) as a hist token.
    2. ``query_mlp(fourier_t(t) ⊕ fourier_x(x))`` encodes each future
       (t, x) as an output-query token.
    3. Concat: tokens = [hist (k_in·nx), out (k·nx)].
    4. Optional Riemann bias placed in the (out × last-hist-row)
       sub-block of the (N, N) self-attention mask.
    5. Self-attention stack.
    6. Residual output: ``pred = state_hist[:,-1,:] + head(tokens)``.

The shared ``fourier_t`` / ``fourier_x`` feature extractors are reused
across history and query tokens so the two pathways align on the same
notion of time/space.
"""

import torch
import torch.nn as nn

from .base.feature_encoders import FourierFeatures
from .base.linear_bias import LinearBias
from .base.pde import ARZPDE, LWRPDE, PDE, BurgersPDE, EulerPDE
from .base.pde_bias import PDEBias
from .base.self_attention_biased import BiasedSelfAttentionLayer


class ARWaveNO(nn.Module):
    """Decoder-only block predictor with k-step history + residual output.

    Args:
        k: Number of output timesteps per forward pass (``k_out``).
        k_in: History length. Defaults to ``k``.
        hidden_dim: Token dim.
        num_freq_t / num_freq_x: Shared Fourier bands for (t, x).
        num_layers: Self-attention depth.
        num_heads: Attention heads.
        pde: PDE instance driving ``PDEBias`` (``None`` → bare).
        encoder_pde: PDE for output metadata (output_dim, clamp).
        bias_kind: "pde" | "linear" | "none".
        linear_bias_initial_scale: Initial scale for the linear bias.
        dropout: Dropout rate.
        output_dim / output_clamp: Optional overrides.
    """

    is_autoregressive_block = True

    def __init__(
        self,
        k: int,
        k_in: int | None = None,
        hidden_dim: int = 64,
        num_freq_t: int | None = 8,
        num_freq_x: int | None = 8,
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
        meta_pde = encoder_pde if encoder_pde is not None else (pde or LWRPDE())
        if output_dim is not None:
            self.output_dim = output_dim
        else:
            self.output_dim = meta_pde.output_dim

        if output_clamp != "auto":
            self.output_clamp = output_clamp
        else:
            self.output_clamp = meta_pde.output_clamp

        # Shared (t, x) feature extractors.
        if num_freq_t is not None:
            self.fourier_t = FourierFeatures(
                num_frequencies=num_freq_t, include_input=True
            )
            t_dim = self.fourier_t.output_dim
        else:
            self.fourier_t = None
            t_dim = 1

        if num_freq_x is not None:
            self.fourier_x = FourierFeatures(
                num_frequencies=num_freq_x, include_input=True
            )
            x_dim = self.fourier_x.output_dim
        else:
            self.fourier_x = None
            x_dim = 1

        # History-token encoder: (t, x, state_value) → H.
        self.hist_mlp = nn.Sequential(
            nn.Linear(t_dim + x_dim + self.output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Query-token encoder: (t, x) → H.
        self.query_mlp = nn.Sequential(
            nn.Linear(t_dim + x_dim, hidden_dim),
            nn.GELU(),
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

        # Residual output head: produces a delta added to the current row.
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )
        # Small-scale init on the final weight (not zero) so that delta
        # starts as a tiny random perturbation rather than identically 0.
        # Zero-weight-init creates a degenerate point: no gradient flows
        # back to the transformer on step 1 (dL/d(pre-head) = W^T @ g = 0),
        # which combined with the small residual-loss signal lets
        # ReduceLROnPlateau kill the learning rate before the transformer
        # starts getting useful gradient. Bias stays at 0 so delta has no
        # systematic offset at init.
        nn.init.xavier_uniform_(
            self.output_head[-1].weight, gain=1e-2
        )
        nn.init.zeros_(self.output_head[-1].bias)

    # -- feature helpers -----------------------------------------------------

    def _tx_features(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Concat Fourier(t) and Fourier(x) as (..., t_dim + x_dim)."""
        t_flat = t.reshape(-1)
        x_flat = x.reshape(-1)
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
        return torch.cat([t_enc, x_enc], dim=-1)  # (Ntokens, t_dim+x_dim)

    def _encode_history(
        self,
        state_hist: torch.Tensor,
        hist_t_coords: torch.Tensor,
        hist_x_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Encode k_in history rows into (B, k_in * nx, H) tokens.

        Input shapes:
            state_hist:      (B, C, k_in, nx)
            hist_t_coords:   (B, 1, k_in, nx)
            hist_x_coords:   (B, 1, k_in, nx)
        """
        B, C, k_in, nx = state_hist.shape
        N = k_in * nx

        tx = self._tx_features(
            hist_t_coords.squeeze(1), hist_x_coords.squeeze(1)
        )  # (B*N, t+x)

        # state: (B, C, k_in, nx) → (B, k_in, nx, C) → (B*N, C)
        state_flat = state_hist.permute(0, 2, 3, 1).reshape(B * N, C)

        features = torch.cat([tx, state_flat], dim=-1)
        return self.hist_mlp(features).reshape(B, N, self.hidden_dim)

    def _encode_queries(
        self,
        t_coords: torch.Tensor,
        x_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Encode future queries into (B, k * nx, H) tokens.

        Input shapes: (B, k, nx) each.
        """
        B = t_coords.shape[0]
        tx = self._tx_features(t_coords, x_coords)
        return self.query_mlp(tx).reshape(B, -1, self.hidden_dim)

    # -- bias helpers --------------------------------------------------------

    @staticmethod
    def _cell_ic_data(
        state_row: torch.Tensor,
        dx: torch.Tensor,
        nx: int,
    ) -> dict[str, torch.Tensor]:
        """Build a cell-wise ic_data dict for PDEBias from a single row."""
        B = state_row.shape[0]
        device = state_row.device
        dtype = state_row.dtype

        if dx.dim() == 0:
            dx_b = dx.expand(B)
        else:
            dx_b = dx
        dx_b = dx_b.to(dtype=dtype, device=device)

        edge_idx = torch.arange(nx + 1, device=device, dtype=dtype)
        xs = dx_b.unsqueeze(1) * edge_idx.unsqueeze(0)

        ks = state_row[:, 0, :]
        pieces_mask = torch.ones(B, nx, device=device, dtype=dtype)
        ic = {"xs": xs, "ks": ks, "pieces_mask": pieces_mask}
        if state_row.shape[1] >= 2:
            ic["ks_v"] = state_row[:, 1, :]
        if state_row.shape[1] >= 3:
            ic["ks_p"] = state_row[:, 2, :]
        return ic

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Required keys:
            state_hist:     (B, C, k_in, nx)
            hist_t_coords:  (B, 1, k_in, nx)
            hist_x_coords:  (B, 1, k_in, nx)
            t_coords:       (B, 1, k, nx)
            x_coords:       (B, 1, k, nx)
            dx:             scalar or (B,)
        """
        state_hist = batch_input["state_hist"]  # (B, C, k_in, nx)
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, k, nx)
        x_coords = batch_input["x_coords"].squeeze(1)
        hist_t_coords = batch_input["hist_t_coords"]
        hist_x_coords = batch_input["hist_x_coords"]
        dx = batch_input["dx"]

        B, nt, nx = t_coords.shape
        assert nt == self.k, f"t_coords has nt={nt}, expected k={self.k}"
        assert state_hist.shape[2] == self.k_in

        # Stage 1–3: encode hist + query tokens and concatenate.
        hist_tokens = self._encode_history(
            state_hist, hist_t_coords, hist_x_coords
        )  # (B, k_in*nx, H)
        query_tokens = self._encode_queries(t_coords, x_coords)  # (B, k*nx, H)
        tokens = torch.cat([hist_tokens, query_tokens], dim=1)  # (B, N, H)
        N = tokens.shape[1]
        Q = self.k * nx

        # Stage 4: bias sub-block from the LAST history row → queries.
        char_bias = None
        attn_mask = None
        if self.bias_module is not None:
            current_state = state_hist[:, :, -1, :]  # (B, C, nx)
            current_ic = self._cell_ic_data(current_state, dx, nx)
            char_bias = self.bias_module(
                current_ic, (t_coords, x_coords)
            )  # (B, k, nx, nx)
            sub = char_bias.reshape(B, Q, nx)

            full_mask = torch.zeros(
                B, N, N, device=tokens.device, dtype=sub.dtype
            )
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

        # Stage 6: residual output — delta added to the last history row.
        out_tokens = x[:, self.k_in * nx :, :]  # (B, k*nx, H)
        delta = self.output_head(out_tokens)  # (B, k*nx, output_dim)
        delta = delta.reshape(B, self.k, nx, self.output_dim).permute(
            0, 3, 1, 2
        )  # (B, output_dim, k, nx)

        last_row = state_hist[:, :, -1:, :]  # (B, C, 1, nx)
        output_grid = last_row.expand(-1, -1, self.k, -1) + delta

        if self.output_clamp is not None:
            output_grid = output_grid.clamp(*self.output_clamp)

        result = {"output_grid": output_grid, "delta": delta}
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

    k_out = args.get("ar_block_k", 1)
    k_in_raw = args.get("ar_hist_k", -1)
    k_in = k_out if k_in_raw is None or k_in_raw <= 0 else k_in_raw

    kwargs = dict(
        k=k_out,
        k_in=k_in,
        hidden_dim=args.get("hidden_dim", 64),
        num_freq_t=args.get("num_freq_t", 8),
        num_freq_x=args.get("num_freq_x", 8),
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
