"""WaveNO: Wavefront Neural Operator for hyperbolic conservation laws.

Inspired by wavefront tracking: the solution is determined by wavefronts
emanating from IC boundaries, acting on IC segments. Instead of tracking
wavefronts explicitly (trajectories) or selecting a winning segment (softmin),
WaveNO lets spatial queries discover the relevant segment information via
physics-biased cross-attention.

Architecture (with predict_trajectories=True):
    1. SegmentPhysicsEncoder + self-attention -> contextualized segment embeddings
    2. TimeConditioner (FiLM) + CrossSegmentAttention -> time-evolved segments
    3. BreakpointEvolution -> predicted breakpoint positions (B, D, nt)
    4. compute_boundaries -> per-query local boundary positions (x_left, x_right)
    5. Fourier query encoding with local context -> per-(t, x) query embeddings
    6. Characteristic attention bias -> physics-informed bias (B, nt, nx, K)
    7. Per-time-step biased cross-attention (queries attend to segments)
    8. Density head MLP -> output density

Without predict_trajectories, stages 3-4 are skipped and queries encode
only (t, x) as before.

Key advantages over CharNO:
    - Cross-attention replaces softmin (continuous gradient flow, no vanishing)
    - Density decoded from attended features (fused, not decoupled)
    - Single characteristic bias as attention prior, not 8 hand-engineered features
    - No temperature scheduling, no auxiliary selection supervision loss

Key advantage of breakpoint evolution (v2):
    - Local boundary context (x_left, x_right) makes queries K-invariant
    - Generalizes from 4-piece to 10+ piece initial conditions
"""

import torch
import torch.nn as nn

from .base.biased_cross_attention import (
    BiasedCrossDecoderLayer,
    compute_characteristic_bias,
    compute_discontinuity_characteristic_bias,
)
from .base.breakpoint_evolution import BreakpointEvolution
from .base.characteristic_features import (
    DiscontinuityPhysicsEncoder,
    SegmentPhysicsEncoder,
    TimeConditioner,
)
from .base.feature_encoders import DiscontinuityEncoder, FourierFeatures, TimeEncoder
from .base.flux import DEFAULT_FLUX, Flux
from .base.transformer_encoder import EncoderLayer
from .traj_deeponet import compute_boundaries
from .traj_transformer import TrajectoryDecoderTransformer


class CrossSegmentAttention(nn.Module):
    """Lightweight self-attention over the K segment dimension.

    No feedforward network -- just attention + residual + LayerNorm.
    Identical to CharNO's CrossSegmentAttention.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        att = self.attention(x, x, x, key_padding_mask=key_padding_mask)[0]
        return self.norm(x + self.drop(att))


class WaveNO(nn.Module):
    """Wavefront Neural Operator.

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
        initial_damping_sharpness: Initial sharpness for collision-time
            damping of characteristic bias. Controls how quickly the bias
            fades after estimated wave collision time. Higher = sharper
            transition. Default 5.0.
        predict_trajectories: If True, predict breakpoint evolution and
            include local boundary features (x_left, x_right) in queries.
            This makes the model K-invariant for generalization.
        num_traj_cross_layers: Cross-attention layers in breakpoint
            evolution trajectory decoder.
        num_time_layers: MLP layers in breakpoint time encoder.
        num_freq_bound: Fourier frequency bands for boundary features
            in query encoder.
        flux: Flux function instance.
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
        flux: Flux | None = None,
        with_classifier: bool = False,
        local_features: bool = True,
        independent_traj: bool = False,
        use_discontinuities: bool = False,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_condition = time_condition
        self.num_heads = num_heads
        self.predict_trajectories = predict_trajectories
        self.with_classifier = with_classifier
        self.independent_traj = independent_traj
        self.use_discontinuities = use_discontinuities
        flux = flux or DEFAULT_FLUX()
        self.flux = flux

        # === Stage 1: Token encoder ===
        if use_discontinuities:
            self.disc_physics_encoder = DiscontinuityPhysicsEncoder(
                hidden_dim=hidden_dim,
                num_frequencies=num_seg_frequencies,
                num_layers=num_seg_mlp_layers,
                flux=flux,
                dropout=dropout,
            )
        else:
            self.segment_encoder = SegmentPhysicsEncoder(
                hidden_dim=hidden_dim,
                num_frequencies=num_seg_frequencies,
                num_layers=num_seg_mlp_layers,
                flux=flux,
                include_cumulative_mass=local_features,
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
                    CrossSegmentAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
                    for _ in range(num_cross_segment_layers)
                ]
            )
        else:
            self.cross_segment_layers = nn.ModuleList()

        # === Stage 3: Breakpoint evolution (when predict_trajectories) ===
        if predict_trajectories:
            if use_discontinuities or independent_traj:
                # Trajectory decoded directly from disc embeddings
                # (use_discontinuities: disc_physics_encoder embeddings;
                #  independent_traj: separate DiscontinuityEncoder embeddings)
                if independent_traj and not use_discontinuities:
                    self.disc_encoder = DiscontinuityEncoder(
                        input_dim=3,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        num_frequencies=num_seg_frequencies,
                        num_layers=num_seg_mlp_layers,
                        dropout=dropout,
                    )
                self.traj_time_encoder = TimeEncoder(
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_frequencies=num_freq_t,
                    num_layers=num_time_layers,
                    dropout=dropout,
                )
                self.traj_decoder = TrajectoryDecoderTransformer(
                    hidden_dim=hidden_dim,
                    num_cross_layers=num_traj_cross_layers,
                    num_attention_heads=num_heads,
                )
            else:
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

        # === Classifier head (when with_classifier) ===
        if with_classifier:
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        # === Stage 5: Query encoder (Fourier + MLP) ===
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
                BiasedCrossDecoderLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_cross_layers)
            ]
        )

        # === Stage 8: Density head ===
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize last layer near zero for stable start
        nn.init.zeros_(self.density_head[-1].weight)
        nn.init.constant_(self.density_head[-1].bias, 0.5)

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
                - 'disc_mask': (B, D) discontinuity validity mask
                - 'discontinuities': (B, D, 3) discontinuity features
                - 't_coords': (B, 1, nt, nx) time coordinates
                - 'x_coords': (B, 1, nt, nx) space coordinates

        Returns:
            Dict containing:
                - 'output_grid': (B, 1, nt, nx) predicted density
                - 'characteristic_bias': (B, nt, nx, K_or_D) physics bias
                - 'positions': (B, D, nt) breakpoint positions (when predict_trajectories)
        """
        if self.use_discontinuities:
            return self._forward_disc(batch_input)
        return self._forward_seg(batch_input)

    def _forward_seg(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Segment-based forward pass (original WaveNO)."""
        xs = batch_input["xs"]
        ks = batch_input["ks"]
        pieces_mask = batch_input["pieces_mask"]
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, nt, nx)

        B, nt, nx = t_coords.shape
        K = ks.shape[1]

        # === Stage 1: Encode segments ===
        seg_emb = self.segment_encoder(xs, ks, pieces_mask)  # (B, K, H)

        # Self-attention over segments
        key_padding_mask = ~pieces_mask.bool()  # True = padded
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        for layer in self.self_attn_layers:
            seg_emb = layer(seg_emb, key_padding_mask=key_padding_mask)
        seg_emb = seg_emb * pieces_mask.unsqueeze(-1)  # re-zero padded

        # === Stage 2: Time-conditioned segment evolution ===
        t_unique = t_coords[:, :, 0]  # (B, nt)
        if self.time_condition:
            seg_emb_t = self.time_conditioner(seg_emb, t_unique)  # (B, nt, K, H)
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

        # === Stage 3 & 4: Breakpoint evolution + boundary extraction ===
        positions = None
        existence = None
        if self.predict_trajectories:
            disc_mask = batch_input["disc_mask"]  # (B, D)

            if self.independent_traj:
                # Independent trajectory: encode raw discontinuities directly
                discontinuities = batch_input["discontinuities"]  # (B, D, 3)
                disc_emb = self.disc_encoder(discontinuities, disc_mask)  # (B, D, H)
                time_emb = self.traj_time_encoder(t_unique)  # (B, nt, H)
                positions = self.traj_decoder(
                    disc_emb, time_emb, disc_mask
                )  # (B, D, nt)
                bp_emb = disc_emb  # for classifier if needed
            elif self.with_classifier:
                # Need bp_emb for classifier head
                positions, bp_emb = self.breakpoint_evolution(
                    seg_emb, disc_mask, t_unique, return_embeddings=True
                )
            else:
                positions = self.breakpoint_evolution(
                    seg_emb, disc_mask, t_unique
                )  # (B, D, nt)

            # Classifier: filter breakpoints
            if self.with_classifier:
                existence = (
                    self.classifier_head(bp_emb).squeeze(-1) * disc_mask
                )  # (B, D)
                effective_mask = disc_mask * (existence > 0.5).float()
            else:
                effective_mask = disc_mask

            left_bound, right_bound = compute_boundaries(
                positions, x_coords, effective_mask
            )  # each (B, nt, nx)

        # === Stage 5: Query encoding ===
        t_flat = t_coords.reshape(-1)  # (B*nt*nx,)
        x_flat = x_coords.reshape(-1)  # (B*nt*nx,)
        t_enc = self.fourier_t(t_flat)  # (B*nt*nx, F_t)
        x_enc = self.fourier_x(x_flat)  # (B*nt*nx, F_x)

        if self.predict_trajectories:
            left_enc = self.fourier_bound(
                left_bound.reshape(-1)
            )  # (B*nt*nx, F_bound)
            right_enc = self.fourier_bound(
                right_bound.reshape(-1)
            )  # (B*nt*nx, F_bound)
            query_features = torch.cat(
                [t_enc, x_enc, left_enc, right_enc], dim=-1
            )
        else:
            query_features = torch.cat([t_enc, x_enc], dim=-1)

        query_emb = self.query_mlp(query_features)  # (B*nt*nx, H)
        query_emb = query_emb.reshape(B, nt, nx, self.hidden_dim)  # (B, nt, nx, H)

        # === Stage 6: Characteristic attention bias ===
        char_bias = compute_characteristic_bias(
            t_coords,
            x_coords,
            xs,
            ks,
            pieces_mask,
            self.flux,
            self.bias_scale,
            damping_sharpness=self.damping_sharpness,
        )  # (B, nt, nx, K)

        # === Stage 7: Per-time-step biased cross-attention ===
        # Reshape for batched per-timestep attention
        q = query_emb.reshape(B * nt, nx, self.hidden_dim)  # (B*nt, nx, H)
        kv = seg_emb_t.reshape(B * nt, K, self.hidden_dim)  # (B*nt, K, H)
        bias_flat = char_bias.reshape(B * nt, nx, K)  # (B*nt, nx, K)

        # Expand bias for multi-head: (B*nt, nx, K) -> (B*nt*heads, nx, K)
        attn_mask = (
            bias_flat.unsqueeze(1)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(B * nt * self.num_heads, nx, K)
        )

        # Key padding mask: (B, K) -> (B*nt, K)
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

        # === Stage 8: Density head ===
        density = self.density_head(q).squeeze(-1)  # (B*nt, nx)
        density = density.clamp(0.0, 1.0)
        output_grid = density.reshape(B, 1, nt, nx)

        output = {
            "output_grid": output_grid,
            "characteristic_bias": char_bias,
        }
        if positions is not None:
            output["positions"] = positions
        if existence is not None:
            output["existence"] = existence.unsqueeze(-1).expand_as(positions)

        return output

    def _forward_disc(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Discontinuity-based forward pass (WaveNODisc).

        Uses discontinuities as tokens instead of segments. The token
        dimension is D (number of discontinuities) instead of K (number
        of IC pieces).
        """
        discontinuities = batch_input["discontinuities"]  # (B, D, 3)
        disc_mask = batch_input["disc_mask"]  # (B, D)
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, nt, nx)

        B, nt, nx = t_coords.shape
        D = disc_mask.shape[1]

        # === Stage 1: Encode discontinuities ===
        emb = self.disc_physics_encoder(discontinuities, disc_mask)  # (B, D, H)

        # Self-attention over discontinuity tokens
        key_padding_mask = ~disc_mask.bool()  # True = padded
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        for layer in self.self_attn_layers:
            emb = layer(emb, key_padding_mask=key_padding_mask)
        emb = emb * disc_mask.unsqueeze(-1)  # re-zero padded

        # === Stage 2: Time-conditioned disc evolution ===
        t_unique = t_coords[:, :, 0]  # (B, nt)
        if self.time_condition:
            emb_t = self.time_conditioner(emb, t_unique)  # (B, nt, D, H)
        else:
            emb_t = emb.unsqueeze(1).expand(B, nt, D, self.hidden_dim)

        # Cross-token attention per time step
        if len(self.cross_segment_layers) > 0:
            emb_flat = emb_t.reshape(B * nt, D, -1)
            cs_mask = (
                (~disc_mask.bool())
                .unsqueeze(1)
                .expand(B, nt, D)
                .reshape(B * nt, D)
            )
            all_masked_cs = cs_mask.all(dim=1)
            if all_masked_cs.any():
                cs_mask = cs_mask.clone()
                cs_mask[all_masked_cs] = False
            for layer in self.cross_segment_layers:
                emb_flat = layer(emb_flat, key_padding_mask=cs_mask)
            emb_t = emb_flat.reshape(B, nt, D, self.hidden_dim)
            emb_t = emb_t * disc_mask.unsqueeze(1).unsqueeze(-1)

        # === Stage 3 & 4: Trajectory from disc embeddings + boundary extraction ===
        positions = None
        if self.predict_trajectories:
            time_emb = self.traj_time_encoder(t_unique)  # (B, nt, H)
            positions = self.traj_decoder(
                emb, time_emb, disc_mask
            )  # (B, D, nt)

            left_bound, right_bound = compute_boundaries(
                positions, x_coords, disc_mask
            )  # each (B, nt, nx)

        # === Stage 5: Query encoding ===
        t_flat = t_coords.reshape(-1)  # (B*nt*nx,)
        x_flat = x_coords.reshape(-1)  # (B*nt*nx,)
        t_enc = self.fourier_t(t_flat)  # (B*nt*nx, F_t)
        x_enc = self.fourier_x(x_flat)  # (B*nt*nx, F_x)

        if self.predict_trajectories:
            left_enc = self.fourier_bound(
                left_bound.reshape(-1)
            )  # (B*nt*nx, F_bound)
            right_enc = self.fourier_bound(
                right_bound.reshape(-1)
            )  # (B*nt*nx, F_bound)
            query_features = torch.cat(
                [t_enc, x_enc, left_enc, right_enc], dim=-1
            )
        else:
            query_features = torch.cat([t_enc, x_enc], dim=-1)

        query_emb = self.query_mlp(query_features)  # (B*nt*nx, H)
        query_emb = query_emb.reshape(B, nt, nx, self.hidden_dim)  # (B, nt, nx, H)

        # === Stage 6: Discontinuity characteristic attention bias ===
        char_bias = compute_discontinuity_characteristic_bias(
            t_coords,
            x_coords,
            discontinuities[:, :, 0],  # disc positions
            discontinuities[:, :, 1],  # rho_L
            discontinuities[:, :, 2],  # rho_R
            disc_mask,
            self.flux,
            self.bias_scale,
            damping_sharpness=self.damping_sharpness,
        )  # (B, nt, nx, D)

        # === Stage 7: Per-time-step biased cross-attention ===
        q = query_emb.reshape(B * nt, nx, self.hidden_dim)  # (B*nt, nx, H)
        kv = emb_t.reshape(B * nt, D, self.hidden_dim)  # (B*nt, D, H)
        bias_flat = char_bias.reshape(B * nt, nx, D)  # (B*nt, nx, D)

        # Expand bias for multi-head: (B*nt, nx, D) -> (B*nt*heads, nx, D)
        attn_mask = (
            bias_flat.unsqueeze(1)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(B * nt * self.num_heads, nx, D)
        )

        # Key padding mask: (B, D) -> (B*nt, D)
        kv_padding_mask = (
            (~disc_mask.bool())
            .unsqueeze(1)
            .expand(B, nt, D)
            .reshape(B * nt, D)
        )
        all_masked_kv = kv_padding_mask.all(dim=1)
        if all_masked_kv.any():
            kv_padding_mask = kv_padding_mask.clone()
            kv_padding_mask[all_masked_kv] = False

        for layer in self.cross_attn_layers:
            q = layer(
                q, kv, key_padding_mask=kv_padding_mask, attn_mask=attn_mask
            )

        # === Stage 8: Density head ===
        density = self.density_head(q).squeeze(-1)  # (B*nt, nx)
        density = density.clamp(0.0, 1.0)
        output_grid = density.reshape(B, 1, nt, nx)

        output = {
            "output_grid": output_grid,
            "characteristic_bias": char_bias,
        }
        if positions is not None:
            output["positions"] = positions

        return output


def build_waveno(args: dict) -> WaveNO:
    """Factory function for WaveNO.

    Args:
        args: Dict or Namespace with model configuration.

    Returns:
        Configured WaveNO instance.
    """
    if not isinstance(args, dict):
        args = vars(args)

    return WaveNO(
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
        predict_trajectories=args.get("predict_trajectories", True),
        num_traj_cross_layers=args.get("num_traj_cross_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_freq_bound=args.get("num_freq_bound", 8),
        dropout=args.get("dropout", 0.05),
    )


def _build_waveno_base(args: dict, **overrides) -> WaveNO:
    """Shared builder for WaveNO variants."""
    if not isinstance(args, dict):
        args = vars(args)
    kwargs = dict(
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
        predict_trajectories=args.get("predict_trajectories", True),
        num_traj_cross_layers=args.get("num_traj_cross_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_freq_bound=args.get("num_freq_bound", 8),
        dropout=args.get("dropout", 0.05),
    )
    kwargs.update(overrides)
    return WaveNO(**kwargs)


def build_waveno_cls(args: dict) -> WaveNO:
    """WaveNO + classifier head to filter breakpoints."""
    return _build_waveno_base(args, with_classifier=True)


def build_waveno_local(args: dict) -> WaveNO:
    """WaveNO without cumulative mass (N_k) in segment encoder."""
    return _build_waveno_base(args, local_features=False)


def build_waveno_indep_traj(args: dict) -> WaveNO:
    """WaveNO with independent trajectory decoding from raw discontinuities."""
    return _build_waveno_base(args, independent_traj=True)


def build_waveno_disc(args: dict) -> WaveNO:
    """WaveNO with discontinuity-based tokens instead of segments."""
    return _build_waveno_base(args, use_discontinuities=True)
