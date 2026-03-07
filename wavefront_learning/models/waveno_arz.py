"""WaveNO for ARZ traffic flow system (2-variable: density + velocity).

Same 8-stage architecture as WaveNO with adaptations for the ARZ system:
  1. ARZSegmentPhysicsEncoder (reads ks_rho + ks_v)
  2. TimeConditioner + CrossSegmentAttention (equation-agnostic, reused)
  3. BreakpointEvolution (equation-agnostic, reused)
  4. compute_boundaries (equation-agnostic, reused)
  5. Fourier query encoding (equation-agnostic, reused)
  6. compute_arz_characteristic_bias (two eigenvalues per segment)
  7. BiasedCrossDecoderLayer (equation-agnostic, reused)
  8. Output head: 2-channel [rho, v], no [0,1] clamping

Output: {"output_grid": (B,2,nt,nx), "positions": (B,D,nt),
         "characteristic_bias": (B,nt,nx,K)}
"""

import torch
import torch.nn as nn

from .base.arz_physics import ARZPhysics
from .base.arz_segment_encoder import ARZSegmentPhysicsEncoder
from .base.biased_cross_attention import (
    BiasedCrossDecoderLayer,
    CollisionTimeHead,
    compute_arz_characteristic_bias,
)
from .base.boundaries import compute_boundaries
from .base.breakpoint_evolution import BreakpointEvolution
from .base.characteristic_features import TimeConditioner
from .base.feature_encoders import FourierFeatures
from .base.transformer_encoder import CrossSegmentAttention, EncoderLayer


class WaveNOARZ(nn.Module):
    """WaveNO adapted for the ARZ traffic flow system.

    Args:
        hidden_dim: All embedding dimensions.
        num_freq_t: Fourier frequency bands for time in query encoder.
        num_freq_x: Fourier frequency bands for space in query encoder.
        num_seg_frequencies: Fourier frequency bands for segment encoder.
        num_seg_mlp_layers: MLP depth in segment encoder.
        num_self_attn_layers: Self-attention layers for segment interaction.
        num_cross_layers: Biased cross-attention layers (queries -> segments).
        num_heads: Attention heads (both self and cross).
        num_cross_segment_layers: Cross-segment attention per timestep.
        time_condition: Enable FiLM time conditioning.
        initial_bias_scale: Initial characteristic bias scale.
        initial_damping_sharpness: Initial sharpness for collision-time damping.
        predict_trajectories: If True, predict breakpoint evolution and
            include local boundary features in queries.
        num_traj_cross_layers: Cross-attention layers in breakpoint evolution.
        num_time_layers: MLP layers in breakpoint time encoder.
        num_freq_bound: Fourier frequency bands for boundary features.
        gamma: ARZ pressure exponent.
        learned_collision_time: Use learned collision time instead of analytical.
        dropout: Dropout rate.
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
        num_cross_segment_layers: int = 1,
        time_condition: bool = True,
        initial_bias_scale: float = 5.0,
        initial_damping_sharpness: float = 5.0,
        predict_trajectories: bool = True,
        num_traj_cross_layers: int = 2,
        num_time_layers: int = 2,
        num_freq_bound: int = 8,
        gamma: float = 1.0,
        learned_collision_time: bool = False,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_condition = time_condition
        self.num_heads = num_heads
        self.predict_trajectories = predict_trajectories
        self.learned_collision_time = learned_collision_time

        # ARZ physics module (stored in checkpoint)
        self.arz_physics = ARZPhysics(gamma=gamma)

        # === Learned collision time head ===
        if learned_collision_time:
            self.collision_time_head = CollisionTimeHead(hidden_dim)

        # === Stage 1: ARZ segment encoder ===
        self.segment_encoder = ARZSegmentPhysicsEncoder(
            hidden_dim=hidden_dim,
            num_frequencies=num_seg_frequencies,
            num_layers=num_seg_mlp_layers,
            arz_physics=self.arz_physics,
            dropout=dropout,
        )

        # Self-attention over segments
        self.self_attn_layers = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_self_attn_layers)
            ]
        )

        # === Stage 2: Time conditioning + cross-segment attention ===
        if time_condition:
            self.time_conditioner = TimeConditioner(
                hidden_dim=hidden_dim,
                num_time_frequencies=num_seg_frequencies,
                dropout=dropout,
            )

        if num_cross_segment_layers > 0:
            self.cross_segment_layers = nn.ModuleList(
                [
                    CrossSegmentAttention(
                        hidden_dim, num_heads=num_heads, dropout=dropout
                    )
                    for _ in range(num_cross_segment_layers)
                ]
            )
        else:
            self.cross_segment_layers = nn.ModuleList()

        # === Stage 3: Breakpoint evolution ===
        if predict_trajectories:
            self.breakpoint_evolution = BreakpointEvolution(
                hidden_dim=hidden_dim,
                num_traj_cross_layers=num_traj_cross_layers,
                num_time_layers=num_time_layers,
                num_freq_t=num_freq_t,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.fourier_bound = FourierFeatures(
                num_frequencies=num_freq_bound, include_input=True
            )

        # === Stage 5: Query encoder ===
        self.fourier_t = FourierFeatures(
            num_frequencies=num_freq_t, include_input=True
        )
        self.fourier_x = FourierFeatures(
            num_frequencies=num_freq_x, include_input=True
        )
        query_input_dim = self.fourier_t.output_dim + self.fourier_x.output_dim
        if predict_trajectories:
            query_input_dim += 2 * self.fourier_bound.output_dim
        self.query_mlp = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # === Stage 6: Characteristic bias scale + collision-time damping ===
        self.bias_scale = nn.Parameter(torch.tensor(initial_bias_scale))
        self.damping_sharpness = nn.Parameter(
            torch.tensor(initial_damping_sharpness)
        )

        # === Stage 7: Biased cross-attention layers ===
        self.cross_attn_layers = nn.ModuleList(
            [
                BiasedCrossDecoderLayer(
                    hidden_dim, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_cross_layers)
            ]
        )

        # === Stage 8: Output head (2-channel: rho, v) ===
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.constant_(self.output_head[-1].bias, 0.5)

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass for ARZ system.

        Args:
            batch_input: Dict containing:
                - 'xs': (B, K+1) breakpoint positions
                - 'ks': (B, K) density piece values
                - 'ks_v': (B, K) velocity piece values
                - 'pieces_mask': (B, K) validity mask
                - 'disc_mask': (B, D) discontinuity validity mask
                - 't_coords': (B, 1, nt, nx) time coordinates
                - 'x_coords': (B, 1, nt, nx) space coordinates

        Returns:
            Dict with 'output_grid' (B,2,nt,nx), 'characteristic_bias',
            and optionally 'positions'.
        """
        xs = batch_input["xs"]
        ks_rho = batch_input["ks"]
        ks_v = batch_input["ks_v"]
        pieces_mask = batch_input["pieces_mask"]
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, nt, nx)

        B, nt, nx = t_coords.shape
        K = ks_rho.shape[1]

        # === Stage 1: Encode ARZ segments ===
        seg_emb = self.segment_encoder(
            xs, ks_rho, ks_v, pieces_mask
        )  # (B, K, H)

        # Self-attention over segments
        key_padding_mask = ~pieces_mask.bool()
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        for layer in self.self_attn_layers:
            seg_emb = layer(seg_emb, key_padding_mask=key_padding_mask)
        seg_emb = seg_emb * pieces_mask.unsqueeze(-1)

        # Learned collision time
        seg_t_coll = None
        if self.learned_collision_time:
            seg_t_coll = self.collision_time_head(seg_emb, pieces_mask)

        # === Stage 2: Time-conditioned segment evolution ===
        t_unique = t_coords[:, :, 0]  # (B, nt)
        if self.time_condition:
            seg_emb_t = self.time_conditioner(seg_emb, t_unique)
        else:
            seg_emb_t = seg_emb.unsqueeze(1).expand(B, nt, K, self.hidden_dim)

        # Cross-segment attention per time step
        if len(self.cross_segment_layers) > 0:
            seg_flat = seg_emb_t.reshape(B * nt, K, -1)
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
                seg_flat = layer(seg_flat, key_padding_mask=cs_mask)
            seg_emb_t = seg_flat.reshape(B, nt, K, self.hidden_dim)
            seg_emb_t = seg_emb_t * pieces_mask.unsqueeze(1).unsqueeze(-1)

        # === Stage 3 & 4: Breakpoint evolution + boundaries ===
        positions = None
        if self.predict_trajectories:
            disc_mask = batch_input["disc_mask"]
            positions = self.breakpoint_evolution(
                seg_emb, disc_mask, t_unique
            )  # (B, D, nt)

            left_bound, right_bound = compute_boundaries(
                positions, x_coords, disc_mask
            )

        # === Stage 5: Query encoding ===
        t_flat = t_coords.reshape(-1)
        x_flat = x_coords.reshape(-1)
        t_enc = self.fourier_t(t_flat)
        x_enc = self.fourier_x(x_flat)

        if self.predict_trajectories:
            left_enc = self.fourier_bound(left_bound.reshape(-1))
            right_enc = self.fourier_bound(right_bound.reshape(-1))
            query_features = torch.cat(
                [t_enc, x_enc, left_enc, right_enc], dim=-1
            )
        else:
            query_features = torch.cat([t_enc, x_enc], dim=-1)

        query_emb = self.query_mlp(query_features)
        query_emb = query_emb.reshape(B, nt, nx, self.hidden_dim)

        # === Stage 6: ARZ characteristic attention bias ===
        char_bias = compute_arz_characteristic_bias(
            t_coords,
            x_coords,
            xs,
            ks_rho,
            ks_v,
            pieces_mask,
            self.arz_physics,
            self.bias_scale,
            damping_sharpness=self.damping_sharpness,
            t_coll=seg_t_coll,
        )  # (B, nt, nx, K)

        # === Stage 7: Per-time-step biased cross-attention ===
        q = query_emb.reshape(B * nt, nx, self.hidden_dim)
        kv = seg_emb_t.reshape(B * nt, K, self.hidden_dim)
        bias_flat = char_bias.reshape(B * nt, nx, K)

        attn_mask = (
            bias_flat.unsqueeze(1)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(B * nt * self.num_heads, nx, K)
        )

        kv_padding_mask = (
            (~pieces_mask.bool())
            .unsqueeze(1)
            .expand(B, nt, K)
            .reshape(B * nt, K)
        )
        all_masked_kv = kv_padding_mask.all(dim=1)
        if all_masked_kv.any():
            kv_padding_mask = kv_padding_mask.clone()
            kv_padding_mask[all_masked_kv] = False

        for layer in self.cross_attn_layers:
            q = layer(
                q, kv, key_padding_mask=kv_padding_mask, attn_mask=attn_mask
            )

        # === Stage 8: Output head (2-channel, no clamping) ===
        out = self.output_head(q)  # (B*nt, nx, 2)
        out = out.reshape(B, nt, nx, 2)
        output_grid = out.permute(0, 3, 1, 2)  # (B, 2, nt, nx)

        output = {
            "output_grid": output_grid,
            "characteristic_bias": char_bias,
        }
        if positions is not None:
            output["positions"] = positions

        return output


def build_waveno_arz(args: dict) -> WaveNOARZ:
    """Factory for WaveNOARZ with trajectory prediction."""
    if not isinstance(args, dict):
        args = vars(args)

    return WaveNOARZ(
        hidden_dim=args.get("hidden_dim", 64),
        num_freq_t=args.get("num_freq_t", 8),
        num_freq_x=args.get("num_freq_x", 8),
        num_seg_frequencies=args.get("num_seg_frequencies", 8),
        num_seg_mlp_layers=args.get("num_seg_mlp_layers", 2),
        num_self_attn_layers=args.get("num_self_attn_layers", 2),
        num_cross_layers=args.get("num_cross_layers", 2),
        num_heads=args.get("num_heads", 4),
        num_cross_segment_layers=args.get("num_cross_segment_layers", 1),
        time_condition=args.get("time_condition", True),
        initial_bias_scale=args.get("initial_bias_scale", 5.0),
        initial_damping_sharpness=args.get("initial_damping_sharpness", 5.0),
        predict_trajectories=True,
        num_traj_cross_layers=args.get("num_traj_cross_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_freq_bound=args.get("num_freq_bound", 8),
        gamma=args.get("gamma", 1.0),
        learned_collision_time=args.get("learned_collision_time", False),
        dropout=args.get("dropout", 0.05),
    )


def build_waveno_arz_base(args: dict) -> WaveNOARZ:
    """Factory for WaveNOARZ without trajectory prediction (grid-only)."""
    if not isinstance(args, dict):
        args = vars(args)

    return WaveNOARZ(
        hidden_dim=args.get("hidden_dim", 64),
        num_freq_t=args.get("num_freq_t", 8),
        num_freq_x=args.get("num_freq_x", 8),
        num_seg_frequencies=args.get("num_seg_frequencies", 8),
        num_seg_mlp_layers=args.get("num_seg_mlp_layers", 2),
        num_self_attn_layers=args.get("num_self_attn_layers", 2),
        num_cross_layers=args.get("num_cross_layers", 2),
        num_heads=args.get("num_heads", 4),
        num_cross_segment_layers=args.get("num_cross_segment_layers", 1),
        time_condition=args.get("time_condition", True),
        initial_bias_scale=args.get("initial_bias_scale", 5.0),
        initial_damping_sharpness=args.get("initial_damping_sharpness", 5.0),
        predict_trajectories=False,
        num_traj_cross_layers=args.get("num_traj_cross_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_freq_bound=args.get("num_freq_bound", 8),
        gamma=args.get("gamma", 1.0),
        learned_collision_time=args.get("learned_collision_time", False),
        dropout=args.get("dropout", 0.05),
    )
