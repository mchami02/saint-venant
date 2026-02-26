"""AutoregressiveWaveNO: Autoregressive Neural Operator with Trajectory Prediction.

Combines WaveNO's segment encoding and trajectory prediction with autoregressive
time-stepping. Both the spatial state and the shock positions evolve step-by-step,
letting the FNO use shock boundary information and letting trajectories adapt to
the evolving state.

Architecture:
    1. SegmentPhysicsEncoder + self-attention → contextualized segment embeddings
    2. Breakpoint encoder: concat adjacent segment embeddings → breakpoint embeddings
    3. Initialize state from IC grid, positions from discontinuities
    4. Autoregressive rollout (for t = 0..nt-2):
       a. compute_boundaries → left/right boundary channels
       b. FNO(state, dt, left, right) → next state (residual)
       c. Trajectory MLP(bp_emb, fourier(pos), dt) → position delta
       d. Optional teacher forcing on state

Input (via ToGridNoCoords transform):
    - grid_input: (B, 1, nt, nx) — masked IC on grid (only t=0 used)
    - xs, ks, pieces_mask — IC segments for SegmentPhysicsEncoder
    - discontinuities: (B, D, 3) — [x, rho_L, rho_R] for initial positions
    - disc_mask: (B, D) — validity mask
    - x_coords: (B, 1, nt, nx) — spatial grid
    - dt: (B,) — time step scalar

Output: {"output_grid": (B, 1, nt, nx), "positions": (B, D, nt)}
"""

import torch
import torch.nn as nn

from .autoregressive_fno import RealFNO1d
from .base.boundaries import compute_boundaries
from .base.characteristic_features import SegmentPhysicsEncoder
from .base.feature_encoders import FourierFeatures
from .base.flux import DEFAULT_FLUX, Flux
from .base.transformer_encoder import EncoderLayer


class AutoregressiveWaveNO(nn.Module):
    """Autoregressive Neural Operator with segment-aware trajectory prediction.

    Args:
        hidden_dim: Segment/breakpoint embedding dimension.
        num_seg_frequencies: Fourier bands for segment encoder.
        num_seg_mlp_layers: MLP depth in segment encoder.
        num_self_attn_layers: Self-attention layers over segments.
        num_heads: Attention heads.
        num_freq_pos: Fourier bands for position encoding in trajectory MLP.
        fno_hidden: FNO hidden channels.
        fno_modes: FNO Fourier modes.
        fno_layers: FNO spectral conv layers.
        fno_padding: Domain padding for non-periodic BCs.
        dropout: Dropout rate.
        flux: Flux function instance.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_seg_frequencies: int = 8,
        num_seg_mlp_layers: int = 2,
        num_self_attn_layers: int = 2,
        num_heads: int = 4,
        num_freq_pos: int = 8,
        fno_hidden: int = 32,
        fno_modes: int = 16,
        fno_layers: int = 2,
        fno_padding: float = 0.2,
        dropout: float = 0.05,
        flux: Flux | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        flux = flux or DEFAULT_FLUX()

        # === Stage 1: Segment encoder ===
        self.segment_encoder = SegmentPhysicsEncoder(
            hidden_dim=hidden_dim,
            num_frequencies=num_seg_frequencies,
            num_layers=num_seg_mlp_layers,
            flux=flux,
            include_cumulative_mass=True,
            dropout=dropout,
        )

        # Self-attention over segments
        self.self_attn_layers = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_self_attn_layers)
            ]
        )

        # === Stage 2: Breakpoint encoder ===
        # Concat adjacent segment embeddings → breakpoint embedding
        self.bp_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # === Stage 3: FNO for spatial state evolution ===
        # Input channels: state (1) + dt (1) + left_bound (1) + right_bound (1) = 4
        self.fno = RealFNO1d(
            in_channels=4,
            out_channels=1,
            hidden_channels=fno_hidden,
            n_modes=fno_modes,
            n_layers=fno_layers,
            domain_padding=fno_padding,
        )

        # === Stage 4: Trajectory MLP ===
        self.fourier_pos = FourierFeatures(
            num_frequencies=num_freq_pos, include_input=True
        )
        # Input: bp_emb (H) + fourier(pos) (F) + dt (1)
        traj_input_dim = hidden_dim + self.fourier_pos.output_dim + 1
        self.traj_mlp = nn.Sequential(
            nn.Linear(traj_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize final layer near zero for small initial position deltas
        nn.init.zeros_(self.traj_mlp[-1].weight)
        nn.init.zeros_(self.traj_mlp[-1].bias)

        # Teacher forcing support
        self.teacher_forcing_ratio = 0.0

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch_input: Dict containing:
                - 'grid_input': (B, 1, nt, nx) masked IC grid
                - 'xs': (B, K+1) breakpoint positions
                - 'ks': (B, K) piece values
                - 'pieces_mask': (B, K) validity mask
                - 'discontinuities': (B, D, 3) [x, rho_L, rho_R]
                - 'disc_mask': (B, D) discontinuity validity mask
                - 'x_coords': (B, 1, nt, nx) spatial coordinates
                - 'dt': (B,) time step scalar

        Returns:
            Dict containing:
                - 'output_grid': (B, 1, nt, nx) predicted density
                - 'positions': (B, D, nt) breakpoint positions
        """
        grid_input = batch_input["grid_input"]  # (B, 1, nt, nx)
        xs = batch_input["xs"]  # (B, K+1)
        ks = batch_input["ks"]  # (B, K)
        pieces_mask = batch_input["pieces_mask"]  # (B, K)
        discontinuities = batch_input["discontinuities"]  # (B, D, 3)
        disc_mask = batch_input["disc_mask"]  # (B, D)
        x_coords = batch_input["x_coords"]  # (B, 1, nt, nx)
        dt = batch_input["dt"]  # (B,)

        B, _, nt, nx = grid_input.shape
        D = disc_mask.shape[1]

        # === Stage 1: Encode IC segments (once) ===
        seg_emb = self.segment_encoder(xs, ks, pieces_mask)  # (B, K, H)

        key_padding_mask = ~pieces_mask.bool()  # True = padded
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        for layer in self.self_attn_layers:
            seg_emb = layer(seg_emb, key_padding_mask=key_padding_mask)
        seg_emb = seg_emb * pieces_mask.unsqueeze(-1)  # re-zero padded

        # === Stage 2: Breakpoint embeddings (once) ===
        # Concat adjacent segment embeddings for each breakpoint
        # Breakpoint d is between segment d and segment d+1
        K = ks.shape[1]
        # Clamp indices to valid range for padding safety
        left_idx = torch.arange(D, device=ks.device).clamp(max=K - 1)
        right_idx = (torch.arange(D, device=ks.device) + 1).clamp(max=K - 1)
        seg_left = seg_emb[:, left_idx, :]  # (B, D, H)
        seg_right = seg_emb[:, right_idx, :]  # (B, D, H)
        bp_emb = self.bp_encoder(
            torch.cat([seg_left, seg_right], dim=-1)
        )  # (B, D, H)
        bp_emb = bp_emb * disc_mask.unsqueeze(-1)  # zero out invalid

        # === Stage 3: Initialize ===
        state = grid_input[:, :, 0, :]  # (B, 1, nx) — initial condition
        positions = discontinuities[:, :, 0]  # (B, D) — initial positions
        x_grid = x_coords[:, 0, 0, :]  # (B, nx) — spatial grid (same for all t)
        dt_channel = dt.view(B, 1, 1).expand(B, 1, nx)  # (B, 1, nx)

        tf_ratio = self.teacher_forcing_ratio if self.training else 0.0
        target_grid = batch_input.get("target_grid") if tf_ratio > 0 else None

        # === Stage 4: Autoregressive rollout ===
        state_list = [state]
        pos_list = [positions]

        for t in range(nt - 1):
            # a. Compute boundary channels from current positions
            # compute_boundaries expects (B, D, nt) and (B, nt, nx)
            pos_for_bounds = positions.unsqueeze(-1)  # (B, D, 1)
            x_for_bounds = x_grid.unsqueeze(1)  # (B, 1, nx)
            left_bound, right_bound = compute_boundaries(
                pos_for_bounds, x_for_bounds, disc_mask
            )  # each (B, 1, nx)

            # b. FNO step: [state, dt, left, right] → state delta
            fno_in = torch.cat(
                [state, dt_channel, left_bound, right_bound], dim=1
            )  # (B, 4, nx)
            state = (state + self.fno(fno_in).tanh()).clamp(0.0, 1.0)  # (B, 1, nx)

            # c. Trajectory step: predict position delta
            pos_enc = self.fourier_pos(
                positions.reshape(-1)
            ).reshape(B, D, -1)  # (B, D, F)
            dt_exp = dt.view(B, 1, 1).expand(B, D, 1)  # (B, D, 1)
            traj_in = torch.cat([bp_emb, pos_enc, dt_exp], dim=-1)  # (B, D, H+F+1)
            delta_pos = self.traj_mlp(traj_in).squeeze(-1).tanh() * 0.1  # (B, D)
            positions = (positions + delta_pos).clamp(0.0, 1.0) * disc_mask  # (B, D)

            # d. Store
            state_list.append(state)
            pos_list.append(positions)

            # e. Teacher forcing: replace state with GT for next step
            if target_grid is not None and torch.rand(1).item() < tf_ratio:
                state = target_grid[:, :, t + 1, :]

        # === Stage 5: Stack outputs ===
        output_grid = torch.stack(state_list, dim=2)  # (B, 1, nt, nx)
        positions_out = torch.stack(pos_list, dim=2)  # (B, D, nt)

        return {
            "output_grid": output_grid,
            "positions": positions_out,
        }


def build_autoregressive_waveno(args: dict) -> AutoregressiveWaveNO:
    """Factory function for AutoregressiveWaveNO.

    Args:
        args: Dict or Namespace with model configuration.

    Returns:
        Configured AutoregressiveWaveNO instance.
    """
    if not isinstance(args, dict):
        args = vars(args)

    return AutoregressiveWaveNO(
        hidden_dim=args.get("hidden_dim", 64),
        num_seg_frequencies=args.get("num_seg_frequencies", 8),
        num_seg_mlp_layers=args.get("num_seg_mlp_layers", 2),
        num_self_attn_layers=args.get("num_self_attn_layers", 2),
        num_heads=args.get("num_heads", 4),
        num_freq_pos=args.get("num_freq_pos", 8),
        fno_hidden=args.get("fno_hidden", 32),
        fno_modes=args.get("fno_modes", 16),
        fno_layers=args.get("fno_layers", 2),
        fno_padding=args.get("fno_padding", 0.2),
        dropout=args.get("dropout", 0.05),
    )
