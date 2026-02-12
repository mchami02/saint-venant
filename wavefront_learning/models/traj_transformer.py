"""TrajTransformer: Transformer-based model for wavefront learning.

This model replaces the bilinear fusion in TrajDeepONet with cross-attention.
Discontinuity embeddings serve as keys/values, while time or spacetime
embeddings serve as queries.

Architecture:
    Branch: DiscontinuityEncoder (Fourier features + MLP) + self-attention
    Trajectory: TimeEncoder → CrossAttention with disc embeddings → positions
    Classifier: Simple MLP (no attention) for shock vs rarefaction
    Density: SpaceTime Fourier encoding → CrossAttention with disc embeddings → density
"""

import torch
import torch.nn as nn

from .base import (
    DiscontinuityEncoder,
    FourierFeatures,
    TimeEncoder,
)
from .base.cross_decoder import CrossDecoderLayer
from .base.transformer_encoder import EncoderLayer
from .traj_deeponet import compute_boundaries


class TrajectoryDecoderTransformer(nn.Module):
    """Decodes trajectory positions using cross-attention.

    Time embeddings (queries) attend to discontinuity embeddings (keys/values),
    then the enriched time features are combined with each discontinuity
    embedding to produce per-discontinuity positions.

    Args:
        hidden_dim: Dimension of embeddings.
        num_cross_layers: Number of cross-attention layers.
        num_attention_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_cross_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_layers = nn.ModuleList(
            [
                CrossDecoderLayer(hidden_dim, num_heads=num_attention_heads)
                for _ in range(num_cross_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.combine_norm = nn.LayerNorm(hidden_dim)

        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        disc_emb: torch.Tensor,
        time_emb: torch.Tensor,
        disc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode trajectory positions via cross-attention.

        Args:
            disc_emb: Discontinuity embeddings (B, D, H).
            time_emb: Time embeddings (B, T, H).
            disc_mask: Validity mask (B, D).

        Returns:
            Predicted positions (B, D, T) clamped to [0, 1].
        """
        B, D, H = disc_emb.shape
        T = time_emb.shape[1]

        # Cross-attention: time queries attend to disc keys/values
        key_padding_mask = ~disc_mask.bool()  # (B, D), True = ignore
        x = time_emb  # (B, T, H)
        for layer in self.cross_layers:
            x = layer(x, disc_emb, key_padding_mask=key_padding_mask)
        time_enriched = self.final_norm(x)  # (B, T, H)

        # Combine each disc embedding with enriched time embeddings
        disc_exp = disc_emb.unsqueeze(2).expand(-1, -1, T, -1)  # (B, D, T, H)
        time_exp = time_enriched.unsqueeze(1).expand(-1, D, -1, -1)  # (B, D, T, H)
        combined = self.combine_norm(disc_exp + time_exp)  # (B, D, T, H)

        # Position head
        positions = self.position_head(combined).squeeze(-1)  # (B, D, T)
        positions = torch.clamp(positions, 0.0, 1.0)
        positions = positions * disc_mask.unsqueeze(-1)

        return positions


class DensityDecoderTransformer(nn.Module):
    """Decodes density using cross-attention over discontinuity embeddings.

    Encodes spacetime coordinates (with optional boundary positions) using
    Fourier features, then uses cross-attention to fuse with per-discontinuity
    embeddings.

    Args:
        hidden_dim: Hidden dimension.
        num_frequencies_t: Fourier frequencies for time.
        num_frequencies_x: Fourier frequencies for spatial coords.
        num_coord_layers: MLP layers for coordinate encoding.
        num_cross_layers: Number of cross-attention layers.
        num_attention_heads: Number of attention heads.
        dropout: Dropout rate.
        with_boundaries: Whether to include boundary positions in encoding.
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
        with_boundaries: bool = True,
    ):
        super().__init__()
        self.with_boundaries = with_boundaries

        self.fourier_t = FourierFeatures(num_frequencies=num_frequencies_t)
        self.fourier_x = FourierFeatures(num_frequencies=num_frequencies_x)

        # Input: fourier(t) + fourier(x) [+ fourier(x_left) + fourier(x_right)]
        num_spatial = 3 if with_boundaries else 1
        input_dim = self.fourier_t.output_dim + num_spatial * self.fourier_x.output_dim

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
        t_coords: torch.Tensor,
        x_coords: torch.Tensor,
        left_bound: torch.Tensor | None,
        right_bound: torch.Tensor | None,
        disc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict density via cross-attention with discontinuity embeddings.

        Args:
            disc_emb: Discontinuity embeddings (B, D, H).
            t_coords: Time coordinates (B, nt, nx).
            x_coords: Space coordinates (B, nt, nx).
            left_bound: Left boundary positions (B, nt, nx) or None.
            right_bound: Right boundary positions (B, nt, nx) or None.
            disc_mask: Validity mask (B, D).

        Returns:
            Predicted density (B, nt, nx) in [0, 1].
        """
        B, nt, nx = t_coords.shape

        # Fourier encode coordinates
        t_flat = t_coords.reshape(-1)
        x_flat = x_coords.reshape(-1)

        t_enc = self.fourier_t(t_flat)
        x_enc = self.fourier_x(x_flat)

        if self.with_boundaries:
            left_enc = self.fourier_x(left_bound.reshape(-1))
            right_enc = self.fourier_x(right_bound.reshape(-1))
            coord_features = torch.cat([t_enc, x_enc, left_enc, right_enc], dim=-1)
        else:
            coord_features = torch.cat([t_enc, x_enc], dim=-1)

        coord_emb = self.coord_mlp(coord_features)  # (B*nt*nx, H)
        coord_emb = coord_emb.reshape(B, nt * nx, -1)  # (B, Q, H)

        # Cross-attention: coord queries attend to disc keys/values
        key_padding_mask = ~disc_mask.bool()  # (B, D)
        x = coord_emb
        for layer in self.cross_layers:
            x = layer(x, disc_emb, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        # Density head
        density = self.density_head(x).squeeze(-1)  # (B, Q)
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
        all_boundaries: If True, density decoder cross-attends to all non-rarefaction
            boundary embeddings (after self-attention) instead of using left/right
            boundary positions as Fourier features. Forces classifier=True and
            with_traj=True.
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
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.all_boundaries = all_boundaries

        # all_boundaries requires classifier and trajectory prediction
        if all_boundaries:
            classifier = True
            with_traj = True

        self.with_traj = with_traj
        self.has_classifier = classifier

        # Branch: encode discontinuities
        self.branch = DiscontinuityEncoder(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_frequencies=num_disc_frequencies,
            num_layers=num_disc_layers,
            dropout=dropout,
        )

        # Cross-discontinuity self-attention
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

        # Boundary self-attention (only when all_boundaries)
        if self.all_boundaries:
            self.boundary_self_attention = nn.ModuleList(
                [
                    EncoderLayer(hidden_dim, num_heads=num_attention_heads)
                    for _ in range(num_interaction_layers)
                ]
            )

        # Density decoder
        # When all_boundaries, boundary info comes from cross-attention KV,
        # not from Fourier-encoded left/right positions.
        self.density_decoder = DensityDecoderTransformer(
            hidden_dim=hidden_dim,
            num_frequencies_t=num_frequencies_t,
            num_frequencies_x=num_frequencies_x,
            num_coord_layers=num_coord_layers,
            num_cross_layers=num_density_cross_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            with_boundaries=with_traj and not all_boundaries,
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

        # === BRANCH ===
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
            positions = self.traj_decoder(
                branch_emb, time_emb, disc_mask
            )  # (B, D, T)

            if self.all_boundaries:
                # === ALL BOUNDARIES PATH ===
                # Filter to non-rarefaction boundaries
                effective_mask = disc_mask * (existence > 0.5).float()
                boundary_emb = branch_emb * effective_mask.unsqueeze(-1)

                # Self-attention among non-rarefaction boundaries
                boundary_key_mask = ~effective_mask.bool()
                all_masked = boundary_key_mask.all(dim=1)
                if all_masked.any():
                    boundary_key_mask = boundary_key_mask.clone()
                    boundary_key_mask[all_masked] = False
                for layer in self.boundary_self_attention:
                    boundary_emb = layer(
                        boundary_emb, key_padding_mask=boundary_key_mask
                    )
                boundary_emb = boundary_emb * effective_mask.unsqueeze(-1)

                # Density: coord queries cross-attend to boundary embeddings
                density = self.density_decoder(
                    boundary_emb, t_coords, x_coords,
                    None, None, effective_mask,
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
            left_bound, right_bound = compute_boundaries(
                positions, x_coords, effective_mask
            )

            # === DENSITY ===
            density = self.density_decoder(
                branch_emb, t_coords, x_coords,
                left_bound, right_bound, disc_mask,
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
            branch_emb, t_coords, x_coords, None, None, disc_mask
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
    """Build TrajTransformer with classifier and all-boundaries density decoding.

    In this variant, the density decoder cross-attends to all non-rarefaction
    boundary embeddings (after self-attention) instead of using left/right
    boundary positions as Fourier features.

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
