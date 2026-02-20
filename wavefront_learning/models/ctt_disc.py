"""CTTDisc: Classifier TrajTransformer with segment-based tokens.

Standalone model extracted from TrajTransformer(use_segments=True).
Uses segment-based encoding (SegmentPhysicsEncoder) instead of
discontinuity tokens, with BreakpointEvolution for trajectory
prediction and DensityDecoderTransformer for density decoding.

Architecture:
    1. SegmentPhysicsEncoder -> segment embeddings (B, K, H)
    2. EncoderLayer self-attention -> contextualized segment embeddings
    3. BreakpointEvolution -> breakpoint positions (B, D, nt)
    4. Classifier head -> shock vs rarefaction per breakpoint
    5. compute_boundaries -> left/right boundary positions
    6. DensityDecoderTransformer -> density grid (B, nt, nx)
"""

import torch
import torch.nn as nn

from .base.boundaries import compute_boundaries
from .base.breakpoint_evolution import BreakpointEvolution
from .base.characteristic_features import SegmentPhysicsEncoder
from .base.decoders import DensityDecoderTransformer
from .base.flux import DEFAULT_FLUX
from .base.transformer_encoder import EncoderLayer


class CTTDisc(nn.Module):
    """Classifier TrajTransformer with segment-based tokens.

    Args:
        hidden_dim: Hidden dimension for all networks.
        num_frequencies_t: Fourier frequencies for time encoding.
        num_frequencies_x: Fourier frequencies for space encoding.
        num_disc_frequencies: Fourier frequencies for segment encoder.
        num_disc_layers: MLP layers in segment encoder.
        num_time_layers: MLP layers in time encoder.
        num_coord_layers: MLP layers in coordinate encoder.
        num_interaction_layers: Self-attention layers for segment interaction.
        num_traj_cross_layers: Cross-attention layers in trajectory decoder.
        num_density_cross_layers: Cross-attention layers in density decoder.
        num_attention_heads: Number of attention heads.
        dropout: Dropout rate.
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
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.flux = DEFAULT_FLUX()

        # Segment-based encoding
        self.segment_encoder = SegmentPhysicsEncoder(
            hidden_dim=hidden_dim,
            num_frequencies=num_disc_frequencies,
            num_layers=num_disc_layers,
            flux=self.flux,
            dropout=dropout,
        )

        # Cross-token self-attention
        self.disc_interaction = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, num_heads=num_attention_heads)
                for _ in range(num_interaction_layers)
            ]
        )

        # Classifier: predicts shock (1) vs rarefaction (0) per breakpoint
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Trajectory via BreakpointEvolution
        self.breakpoint_evolution = BreakpointEvolution(
            hidden_dim=hidden_dim,
            num_traj_cross_layers=num_traj_cross_layers,
            num_time_layers=num_time_layers,
            num_freq_t=num_frequencies_t,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # Density decoder with boundary conditioning
        self.density_decoder = DensityDecoderTransformer(
            hidden_dim=hidden_dim,
            num_frequencies_t=num_frequencies_t,
            num_frequencies_x=num_frequencies_x,
            num_coord_layers=num_coord_layers,
            num_cross_layers=num_density_cross_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            with_boundaries=True,
            biased=False,
        )

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass using segment-based tokens.

        Args:
            batch_input: Dict containing:
                - 'xs': (B, K+1) breakpoint positions
                - 'ks': (B, K) piece values
                - 'pieces_mask': (B, K) validity mask for segments
                - 'disc_mask': (B, D) validity mask for breakpoints
                - 't_coords': (B, 1, nt, nx) time coordinates
                - 'x_coords': (B, 1, nt, nx) space coordinates

        Returns:
            Dict with positions, output_grid, and existence.
        """
        xs = batch_input["xs"]
        ks = batch_input["ks"]
        pieces_mask = batch_input["pieces_mask"]
        disc_mask = batch_input["disc_mask"]
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, nt, nx)

        # === SEGMENT ENCODING ===
        seg_emb = self.segment_encoder(xs, ks, pieces_mask)  # (B, K, H)

        # === SELF-ATTENTION over segment tokens ===
        key_padding_mask = ~pieces_mask.bool()
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        for layer in self.disc_interaction:
            seg_emb = layer(seg_emb, key_padding_mask=key_padding_mask)
        seg_emb = seg_emb * pieces_mask.unsqueeze(-1)  # re-zero padded

        # === TRAJECTORY via BreakpointEvolution ===
        t_unique = t_coords[:, :, 0]  # (B, nt)

        positions, bp_emb = self.breakpoint_evolution(
            seg_emb, disc_mask, t_unique, return_embeddings=True
        )
        existence = (
            self.classifier_head(bp_emb).squeeze(-1) * disc_mask
        )  # (B, D)

        # === BOUNDARIES ===
        effective_mask = disc_mask * (existence > 0.5).float()

        left_bound, right_bound = compute_boundaries(
            positions, x_coords, effective_mask
        )

        # === DENSITY via cross-attention to segment embeddings ===
        density = self.density_decoder(
            seg_emb,
            t_coords,
            x_coords,
            left_bound,
            right_bound,
            pieces_mask,
        )  # (B, nt, nx)

        output_grid = density.unsqueeze(1)  # (B, 1, nt, nx)

        return {
            "positions": positions,
            "output_grid": output_grid,
            "existence": existence.unsqueeze(-1).expand_as(positions),
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_ctt_disc(args: dict) -> CTTDisc:
    """Build CTTDisc from configuration dict.

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
        Configured CTTDisc instance.
    """
    return CTTDisc(
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
    )
