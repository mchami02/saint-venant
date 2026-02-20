"""TrajTransformer: Transformer-based model for wavefront learning.

This model replaces the bilinear fusion in TrajDeepONet with cross-attention.
Discontinuity embeddings serve as keys/values, while time or spacetime
embeddings serve as queries.

Architecture:
    Branch: DiscontinuityEncoder (Fourier features + MLP) + self-attention
    Trajectory: TimeEncoder -> CrossAttention with disc embeddings -> positions
    Classifier: Simple MLP (no attention) for shock vs rarefaction
    Density: SpaceTime Fourier encoding -> CrossAttention with disc embeddings -> density
"""

import torch
import torch.nn as nn

from .base import (
    DiscontinuityEncoder,
    FourierFeatures,
    TimeEncoder,
)
from .base.biased_cross_attention import (
    compute_discontinuity_characteristic_bias,
)
from .base.boundaries import compute_boundaries
from .base.characteristic_features import TimeConditioner
from .base.cross_decoder import CrossDecoderLayer
from .base.decoders import DensityDecoderTransformer, TrajectoryDecoderTransformer
from .base.flux import DEFAULT_FLUX, Flux
from .base.transformer_encoder import EncoderLayer


class DynamicDensityDecoder(nn.Module):
    """Density decoder with time-varying boundary cross-attention.

    For each time step, spatial query points attend to dynamic boundary tokens
    that encode both wave properties (from disc embeddings) and predicted
    positions at that time step. Uses soft existence weighting for fully
    differentiable gradient flow.

    Args:
        hidden_dim: Hidden dimension.
        num_frequencies_t: Fourier frequencies for time.
        num_frequencies_x: Fourier frequencies for spatial coords and positions.
        num_coord_layers: MLP layers for coordinate encoding.
        num_cross_layers: Number of cross-attention layers.
        num_attention_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_frequencies_t: int = 8,
        num_frequencies_x: int = 8,
        num_coord_layers: int = 2,
        num_cross_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fourier_t = FourierFeatures(num_frequencies=num_frequencies_t)
        self.fourier_x = FourierFeatures(num_frequencies=num_frequencies_x)

        # Coordinate encoding: fourier(t) + fourier(x) -> hidden_dim
        input_dim = self.fourier_t.output_dim + self.fourier_x.output_dim
        layers = []
        in_dim = input_dim
        for i in range(num_coord_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_coord_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        self.coord_mlp = nn.Sequential(*layers)

        # Position projection: fourier(position) -> hidden_dim
        self.position_fourier = FourierFeatures(num_frequencies=num_frequencies_x)
        self.position_proj = nn.Linear(self.position_fourier.output_dim, hidden_dim)

        self.cross_layers = nn.ModuleList(
            [
                CrossDecoderLayer(hidden_dim, num_heads=num_attention_heads)
                for _ in range(num_cross_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        disc_emb: torch.Tensor,
        positions: torch.Tensor,
        existence: torch.Tensor,
        t_coords: torch.Tensor,
        x_coords: torch.Tensor,
        disc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict density via cross-attention with dynamic boundary tokens.

        Args:
            disc_emb: Discontinuity embeddings (B, D, H).
            positions: Predicted trajectory positions (B, D, T).
            existence: Soft existence probabilities (B, D), no threshold.
            t_coords: Time coordinates (B, nt, nx).
            x_coords: Space coordinates (B, nt, nx).
            disc_mask: Validity mask (B, D).

        Returns:
            Predicted density (B, nt, nx) in [0, 1].
        """
        B, D, H = disc_emb.shape
        _, nt, nx = t_coords.shape
        T = positions.shape[2]

        # === Coordinate queries ===
        t_flat = t_coords.reshape(-1)
        x_flat = x_coords.reshape(-1)
        t_enc = self.fourier_t(t_flat)
        x_enc = self.fourier_x(x_flat)
        coord_features = torch.cat([t_enc, x_enc], dim=-1)
        coord_emb = self.coord_mlp(coord_features)  # (B*nt*nx, H)
        coord_emb = coord_emb.reshape(B, nt, nx, H)

        # === Dynamic boundary tokens ===
        # Fourier-encode trajectory positions and project to hidden_dim
        pos_enc = self.position_fourier(positions.reshape(-1))  # (B*D*T, F)
        pos_enc = self.position_proj(pos_enc).reshape(B, D, T, H)  # (B, D, T, H)

        # Combine disc properties with position encoding
        disc_exp = disc_emb.unsqueeze(2).expand(-1, -1, T, -1)  # (B, D, T, H)
        dynamic_emb = disc_exp + pos_enc  # (B, D, T, H)

        # Soft existence weighting (differentiable, no hard threshold)
        effective_weight = (existence * disc_mask).unsqueeze(-1).unsqueeze(2)
        # (B, D, 1, 1) -> broadcast to (B, D, T, H)
        dynamic_emb = dynamic_emb * effective_weight

        # === Per-time-step batched cross-attention ===
        # Queries: (B, nt, nx, H) -> (B*T, nx, H)
        q = coord_emb.reshape(B * nt, nx, H)
        # Keys/Values: (B, D, T, H) -> (B, T, D, H) -> (B*T, D, H)
        kv = dynamic_emb.permute(0, 2, 1, 3).reshape(B * T, D, H)

        # Key padding mask: (B, D) -> (B*T, D)
        key_padding_mask = ~disc_mask.bool()
        key_padding_mask = (
            key_padding_mask.unsqueeze(1).expand(B, T, D).reshape(B * T, D)
        )
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False

        for layer in self.cross_layers:
            q = layer(q, kv, key_padding_mask=key_padding_mask)
        q = self.final_norm(q)

        # Density head
        density = self.density_head(q).squeeze(-1)  # (B*T, nx)
        density = torch.clamp(density, 0.0, 1.0)
        return density.reshape(B, nt, nx)


class TrajTransformer(nn.Module):
    """Transformer-based model for wavefront learning.

    Uses cross-attention instead of bilinear fusion for both trajectory
    decoding and density decoding. Discontinuity embeddings serve as
    keys/values throughout, avoiding the need for branch pooling.

    Args:
        hidden_dim: Hidden dimension for all networks.
        num_frequencies_t: Fourier frequencies for time encoding.
        num_frequencies_x: Fourier frequencies for space encoding.
        num_disc_frequencies: Fourier frequencies for discontinuity encoder.
        num_disc_layers: MLP layers in discontinuity encoder.
        num_time_layers: MLP layers in time encoder.
        num_coord_layers: MLP layers in coordinate encoder.
        num_interaction_layers: Self-attention layers for disc interaction.
        num_traj_cross_layers: Cross-attention layers in trajectory decoder.
        num_density_cross_layers: Cross-attention layers in density decoder.
        num_attention_heads: Number of attention heads.
        dropout: Dropout rate.
        with_traj: If True, predict trajectories and condition density on boundaries.
        classifier: If True, add classifier head for shock vs rarefaction.
        all_boundaries: If True, use DynamicDensityDecoder that cross-attends to
            time-varying boundary tokens (disc embeddings + trajectory positions)
            with soft existence weighting. Forces classifier=True and with_traj=True.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_frequencies_t: int = 8,
        num_frequencies_x: int = 8,
        num_disc_frequencies: int = 8,
        num_disc_layers: int = 2,
        num_time_layers: int = 2,
        num_coord_layers: int = 2,
        num_interaction_layers: int = 2,
        num_traj_cross_layers: int = 2,
        num_density_cross_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        with_traj: bool = True,
        classifier: bool = False,
        all_boundaries: bool = False,
        characteristic_bias: bool = False,
        segment_physics: bool = False,
        film_time: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.all_boundaries = all_boundaries
        self.characteristic_bias = characteristic_bias
        self.segment_physics = segment_physics
        self.film_time = film_time
        self.num_attention_heads = num_attention_heads

        # all_boundaries requires classifier and trajectory prediction
        if all_boundaries:
            classifier = True
            with_traj = True

        self.with_traj = with_traj
        self.has_classifier = classifier

        # Characteristic bias: physics-informed attention in density decoder
        if characteristic_bias or segment_physics:
            self.flux = DEFAULT_FLUX()
        if characteristic_bias:
            self.bias_scale = nn.Parameter(torch.tensor(5.0))
            self.damping_sharpness = nn.Parameter(torch.tensor(5.0))

        # Branch: encode discontinuities
        # With segment_physics: input is [x, rho_L, rho_R, lambda_L, lambda_R, shock_speed]
        disc_input_dim = 6 if segment_physics else 3
        self.branch = DiscontinuityEncoder(
            input_dim=disc_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_frequencies=num_disc_frequencies,
            num_layers=num_disc_layers,
            dropout=dropout,
        )

        # Cross-token self-attention
        self.disc_interaction = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, num_heads=num_attention_heads)
                for _ in range(num_interaction_layers)
            ]
        )

        # Optional classifier: predicts shock (1) vs rarefaction (0) per disc
        if self.has_classifier:
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        # Trajectory decoder (only when with_traj)
        if self.with_traj:
            self.time_encoder = TimeEncoder(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_frequencies=num_frequencies_t,
                num_layers=num_time_layers,
            )

            self.traj_decoder = TrajectoryDecoderTransformer(
                hidden_dim=hidden_dim,
                num_cross_layers=num_traj_cross_layers,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
            )

        # Density decoder
        if self.all_boundaries:
            # Dynamic boundary decoder: cross-attends to time-varying boundary
            # tokens that carry both wave properties and predicted positions.
            self.density_decoder = DynamicDensityDecoder(
                hidden_dim=hidden_dim,
                num_frequencies_t=num_frequencies_t,
                num_frequencies_x=num_frequencies_x,
                num_coord_layers=num_coord_layers,
                num_cross_layers=num_density_cross_layers,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
            )
        else:
            # Standard decoder with optional Fourier-encoded boundary positions.
            self.density_decoder = DensityDecoderTransformer(
                hidden_dim=hidden_dim,
                num_frequencies_t=num_frequencies_t,
                num_frequencies_x=num_frequencies_x,
                num_coord_layers=num_coord_layers,
                num_cross_layers=num_density_cross_layers,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                with_boundaries=with_traj and not film_time,
                biased=characteristic_bias,
            )

        # FiLM time conditioning: modulate disc embeddings per timestep
        if film_time:
            self.time_conditioner = TimeConditioner(
                hidden_dim=hidden_dim,
                num_time_frequencies=num_frequencies_t,
            )

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch_input: Dict containing:
                - 'discontinuities': (B, D, 3) with [x, rho_L, rho_R]
                - 'disc_mask': (B, D) validity mask
                - 't_coords': (B, 1, nt, nx) time coordinates
                - 'x_coords': (B, 1, nt, nx) space coordinates

        Returns:
            Dict containing:
                - 'positions': (B, D, T) predicted trajectory positions
                - 'output_grid': (B, 1, nt, nx) predicted density grid
                - 'existence': (B, D, T) shock/rarefaction probability
                  (only when classifier=True, constant across T)
        """
        discontinuities = batch_input["discontinuities"]
        disc_mask = batch_input["disc_mask"]
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, nt, nx)

        # === CHARACTERISTIC BIAS (optional) ===
        attn_bias = None
        if self.characteristic_bias:
            attn_bias = compute_discontinuity_characteristic_bias(
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

        # === BRANCH ===
        if self.segment_physics:
            # Enrich discontinuities with physics features
            rho_L = discontinuities[:, :, 1]  # (B, D)
            rho_R = discontinuities[:, :, 2]  # (B, D)
            lambda_L = self.flux.derivative(rho_L)  # (B, D)
            lambda_R = self.flux.derivative(rho_R)  # (B, D)
            s = self.flux.shock_speed(rho_L, rho_R)  # (B, D)
            enriched = torch.cat([
                discontinuities,  # [x, rho_L, rho_R]
                lambda_L.unsqueeze(-1),
                lambda_R.unsqueeze(-1),
                s.unsqueeze(-1),
            ], dim=-1)  # (B, D, 6)
            branch_emb = self.branch(enriched, disc_mask)  # (B, D, H)
        else:
            branch_emb = self.branch(discontinuities, disc_mask)  # (B, D, H)

        # === CROSS-DISCONTINUITY INTERACTION ===
        key_padding_mask = ~disc_mask.bool()
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        for layer in self.disc_interaction:
            branch_emb = layer(branch_emb, key_padding_mask=key_padding_mask)
        branch_emb = branch_emb * disc_mask.unsqueeze(-1)  # re-zero padded

        # === CLASSIFIER (optional) ===
        if self.has_classifier:
            existence = (
                self.classifier_head(branch_emb).squeeze(-1) * disc_mask
            )  # (B, D)

        if self.with_traj:
            # === TRAJECTORY ===
            query_times = t_coords[:, :, 0]  # (B, nt)
            time_emb = self.time_encoder(query_times)  # (B, T, H)
            positions = self.traj_decoder(branch_emb, time_emb, disc_mask)  # (B, D, T)

            if self.all_boundaries:
                # === ALL BOUNDARIES PATH ===
                # Dynamic density decoder with soft existence weighting
                # (no hard threshold -- fully differentiable)
                density = self.density_decoder(
                    branch_emb,
                    positions,
                    existence,
                    t_coords,
                    x_coords,
                    disc_mask,
                )
                output_grid = density.unsqueeze(1)  # (B, 1, nt, nx)

                return {
                    "positions": positions,
                    "output_grid": output_grid,
                    "existence": existence.unsqueeze(-1).expand_as(positions),
                }

            # === BOUNDARIES ===
            if self.has_classifier:
                effective_mask = disc_mask * (existence > 0.5).float()
            else:
                effective_mask = disc_mask

            if self.film_time:
                # === FiLM TIME DENSITY PATH ===
                # Time-condition disc embeddings -> per-timestep keys
                query_times = t_coords[:, :, 0]  # (B, nt)
                branch_emb_t = self.time_conditioner(
                    branch_emb, query_times
                )  # (B, nt, D, H)

                # Per-timestep batched density decoding (no boundary features)
                B_size, nt, nx = t_coords.shape
                D = disc_mask.shape[1]
                H = self.hidden_dim

                # Reshape: treat each (batch, timestep) as a separate batch entry
                density = self.density_decoder(
                    branch_emb_t.reshape(B_size * nt, D, H),
                    t_coords.reshape(B_size * nt, 1, nx),
                    x_coords.reshape(B_size * nt, 1, nx),
                    None,
                    None,
                    disc_mask.unsqueeze(1).expand(-1, nt, -1).reshape(B_size * nt, D),
                )  # (B*nt, 1, nx)
                density = density.reshape(B_size, nt, nx)
            else:
                left_bound, right_bound = compute_boundaries(
                    positions, x_coords, effective_mask
                )

                # === DENSITY ===
                density = self.density_decoder(
                    branch_emb,
                    t_coords,
                    x_coords,
                    left_bound,
                    right_bound,
                    disc_mask,
                    attn_bias=attn_bias,
                )  # (B, nt, nx)

            output_grid = density.unsqueeze(1)  # (B, 1, nt, nx)

            output = {
                "positions": positions,
                "output_grid": output_grid,
            }
            if self.has_classifier:
                output["existence"] = existence.unsqueeze(-1).expand_as(positions)
            return output

        # === DENSITY (no trajectory conditioning) ===
        density = self.density_decoder(
            branch_emb, t_coords, x_coords, None, None, disc_mask,
            attn_bias=attn_bias,
        )
        output_grid = density.unsqueeze(1)  # (B, 1, nt, nx)
        return {"output_grid": output_grid}

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_traj_transformer(args: dict) -> TrajTransformer:
    """Build TrajTransformer from configuration dict.

    Args:
        args: Configuration dictionary with optional keys:
            - hidden_dim (default 32)
            - num_frequencies_t (default 8)
            - num_frequencies_x (default 8)
            - num_disc_frequencies (default 8)
            - num_disc_layers (default 2)
            - num_time_layers (default 2)
            - num_coord_layers (default 2)
            - num_interaction_layers (default 2)
            - num_traj_cross_layers (default 2)
            - num_density_cross_layers (default 2)
            - num_attention_heads (default 4)
            - dropout (default 0.0)

    Returns:
        Configured TrajTransformer instance.
    """
    return TrajTransformer(
        hidden_dim=args.get("hidden_dim", 32),
        num_frequencies_t=args.get("num_frequencies_t", 8),
        num_frequencies_x=args.get("num_frequencies_x", 8),
        num_disc_frequencies=args.get("num_disc_frequencies", 8),
        num_disc_layers=args.get("num_disc_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_coord_layers=args.get("num_coord_layers", 2),
        num_interaction_layers=args.get("num_interaction_layers", 2),
        num_traj_cross_layers=args.get("num_traj_cross_layers", 2),
        num_density_cross_layers=args.get("num_density_cross_layers", 2),
        num_attention_heads=args.get("num_attention_heads", 4),
        dropout=args.get("dropout", 0.0),
        with_traj=True,
        classifier=False,
    )


def build_classifier_traj_transformer(args: dict) -> TrajTransformer:
    """Build TrajTransformer with classifier head.

    Args:
        args: Configuration dictionary (same keys as build_traj_transformer).

    Returns:
        Configured TrajTransformer instance with classifier=True.
    """
    return TrajTransformer(
        hidden_dim=args.get("hidden_dim", 32),
        num_frequencies_t=args.get("num_frequencies_t", 8),
        num_frequencies_x=args.get("num_frequencies_x", 8),
        num_disc_frequencies=args.get("num_disc_frequencies", 8),
        num_disc_layers=args.get("num_disc_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_coord_layers=args.get("num_coord_layers", 2),
        num_interaction_layers=args.get("num_interaction_layers", 2),
        num_traj_cross_layers=args.get("num_traj_cross_layers", 2),
        num_density_cross_layers=args.get("num_density_cross_layers", 2),
        num_attention_heads=args.get("num_attention_heads", 4),
        dropout=args.get("dropout", 0.0),
        with_traj=True,
        classifier=True,
    )


def build_no_traj_transformer(args: dict) -> TrajTransformer:
    """Build TrajTransformer without trajectory prediction.

    Density decoder uses only Fourier-encoded (t, x) coordinates with
    cross-attention to discontinuity embeddings. No trajectory or boundary
    conditioning.

    Args:
        args: Configuration dictionary (same keys as build_traj_transformer).

    Returns:
        Configured TrajTransformer with with_traj=False.
    """
    return TrajTransformer(
        hidden_dim=args.get("hidden_dim", 32),
        num_frequencies_t=args.get("num_frequencies_t", 8),
        num_frequencies_x=args.get("num_frequencies_x", 8),
        num_disc_frequencies=args.get("num_disc_frequencies", 8),
        num_disc_layers=args.get("num_disc_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_coord_layers=args.get("num_coord_layers", 2),
        num_interaction_layers=args.get("num_interaction_layers", 2),
        num_traj_cross_layers=args.get("num_traj_cross_layers", 2),
        num_density_cross_layers=args.get("num_density_cross_layers", 2),
        num_attention_heads=args.get("num_attention_heads", 4),
        dropout=args.get("dropout", 0.0),
        with_traj=False,
        classifier=False,
    )


def build_classifier_all_traj_transformer(args: dict) -> TrajTransformer:
    """Build TrajTransformer with classifier and dynamic boundary density decoding.

    Uses DynamicDensityDecoder: for each time step, spatial queries cross-attend
    to boundary tokens enriched with predicted trajectory positions and soft
    existence weighting. Fully differentiable -- no hard threshold or
    compute_boundaries.

    Args:
        args: Configuration dictionary (same keys as build_traj_transformer).

    Returns:
        Configured TrajTransformer with classifier=True and all_boundaries=True.
    """
    return TrajTransformer(
        hidden_dim=args.get("hidden_dim", 32),
        num_frequencies_t=args.get("num_frequencies_t", 8),
        num_frequencies_x=args.get("num_frequencies_x", 8),
        num_disc_frequencies=args.get("num_disc_frequencies", 8),
        num_disc_layers=args.get("num_disc_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_coord_layers=args.get("num_coord_layers", 2),
        num_interaction_layers=args.get("num_interaction_layers", 2),
        num_traj_cross_layers=args.get("num_traj_cross_layers", 2),
        num_density_cross_layers=args.get("num_density_cross_layers", 2),
        num_attention_heads=args.get("num_attention_heads", 4),
        dropout=args.get("dropout", 0.0),
        with_traj=True,
        classifier=True,
        all_boundaries=True,
    )


def build_biased_classifier_traj_transformer(args: dict) -> TrajTransformer:
    """Build ClassifierTrajTransformer with characteristic attention bias.

    Adds physics-informed attention bias to the density decoder's
    cross-attention, based on backward characteristic propagation from
    each discontinuity's influence zone. Improves resolution generalization.

    Args:
        args: Configuration dictionary (same keys as build_traj_transformer).

    Returns:
        Configured TrajTransformer with classifier=True and
        characteristic_bias=True.
    """
    return _build_ctt_base(args, characteristic_bias=True)


def _build_ctt_base(args: dict, **overrides) -> TrajTransformer:
    """Shared builder for ClassifierTrajTransformer variants."""
    kwargs = dict(
        hidden_dim=args.get("hidden_dim", 32),
        num_frequencies_t=args.get("num_frequencies_t", 8),
        num_frequencies_x=args.get("num_frequencies_x", 8),
        num_disc_frequencies=args.get("num_disc_frequencies", 8),
        num_disc_layers=args.get("num_disc_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_coord_layers=args.get("num_coord_layers", 2),
        num_interaction_layers=args.get("num_interaction_layers", 2),
        num_traj_cross_layers=args.get("num_traj_cross_layers", 2),
        num_density_cross_layers=args.get("num_density_cross_layers", 2),
        num_attention_heads=args.get("num_attention_heads", 4),
        dropout=args.get("dropout", 0.0),
        with_traj=True,
        classifier=True,
    )
    kwargs.update(overrides)
    return TrajTransformer(**kwargs)


def build_ctt_biased(args: dict) -> TrajTransformer:
    """CTT + characteristic attention bias (ablation alias)."""
    return _build_ctt_base(args, characteristic_bias=True)


def build_ctt_seg_physics(args: dict) -> TrajTransformer:
    """CTT + physics features (lambda_L, lambda_R, shock_speed) in disc encoder."""
    return _build_ctt_base(args, segment_physics=True)


def build_ctt_film(args: dict) -> TrajTransformer:
    """CTT + FiLM time conditioning on disc embeddings for density decoding."""
    return _build_ctt_base(args, film_time=True)
