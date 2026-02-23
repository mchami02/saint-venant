"""TransformerSeg: Segment-based transformer without trajectory prediction.

Combines segment-based input encoding (from CTTSeg) with grid-only output
(from NoTrajTransformer). Uses SegmentPhysicsEncoder for input representation
and DensityDecoderTransformer (without boundaries) for density prediction.

Architecture:
    1. SegmentPhysicsEncoder -> segment embeddings (B, K, H)
    2. EncoderLayer self-attention -> contextualized segment embeddings
    3. DensityDecoderTransformer(with_boundaries=False) -> density grid (B, nt, nx)
"""

import torch
import torch.nn as nn

from .base.characteristic_features import SegmentPhysicsEncoder
from .base.decoders import DensityDecoderTransformer
from .base.flux import DEFAULT_FLUX
from .base.transformer_encoder import EncoderLayer


class TransformerSeg(nn.Module):
    """Segment-based transformer for density prediction (no trajectories).

    Args:
        hidden_dim: Hidden dimension for all networks.
        num_frequencies_t: Fourier frequencies for time encoding.
        num_frequencies_x: Fourier frequencies for space encoding.
        num_disc_frequencies: Fourier frequencies for segment encoder.
        num_disc_layers: MLP layers in segment encoder.
        num_coord_layers: MLP layers in coordinate encoder.
        num_interaction_layers: Self-attention layers for segment interaction.
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
        num_coord_layers: int = 2,
        num_interaction_layers: int = 2,
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

        # Density decoder without boundary conditioning
        self.density_decoder = DensityDecoderTransformer(
            hidden_dim=hidden_dim,
            num_frequencies_t=num_frequencies_t,
            num_frequencies_x=num_frequencies_x,
            num_coord_layers=num_coord_layers,
            num_cross_layers=num_density_cross_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            with_boundaries=False,
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
                - 't_coords': (B, 1, nt, nx) time coordinates
                - 'x_coords': (B, 1, nt, nx) space coordinates

        Returns:
            Dict with output_grid only.
        """
        xs = batch_input["xs"]
        ks = batch_input["ks"]
        pieces_mask = batch_input["pieces_mask"]
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

        # === DENSITY via cross-attention to segment embeddings ===
        density = self.density_decoder(
            seg_emb,
            t_coords,
            x_coords,
            None,
            None,
            pieces_mask,
        )  # (B, nt, nx)

        output_grid = density.unsqueeze(1)  # (B, 1, nt, nx)

        return {"output_grid": output_grid}

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_transformer_seg(args: dict) -> TransformerSeg:
    """Build TransformerSeg from configuration dict.

    Args:
        args: Configuration dictionary with optional keys:
            - hidden_dim (default 32)
            - num_frequencies_t (default 8)
            - num_frequencies_x (default 8)
            - num_disc_frequencies (default 8)
            - num_disc_layers (default 2)
            - num_coord_layers (default 2)
            - num_interaction_layers (default 2)
            - num_density_cross_layers (default 2)
            - num_attention_heads (default 4)
            - dropout (default 0.0)

    Returns:
        Configured TransformerSeg instance.
    """
    return TransformerSeg(
        hidden_dim=args.get("hidden_dim", 32),
        num_frequencies_t=args.get("num_frequencies_t", 8),
        num_frequencies_x=args.get("num_frequencies_x", 8),
        num_disc_frequencies=args.get("num_disc_frequencies", 8),
        num_disc_layers=args.get("num_disc_layers", 2),
        num_coord_layers=args.get("num_coord_layers", 2),
        num_interaction_layers=args.get("num_interaction_layers", 2),
        num_density_cross_layers=args.get("num_density_cross_layers", 2),
        num_attention_heads=args.get("num_attention_heads", 4),
        dropout=args.get("dropout", 0.0),
    )
