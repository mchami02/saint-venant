"""Breakpoint evolution module for WaveNO.

Predicts how IC breakpoints evolve over time using cross-attention.
Adjacent segment pairs are encoded into breakpoint embeddings, which
serve as keys/values for time-query cross-attention to produce
per-breakpoint trajectories.

This gives WaveNO local boundary context (x_left, x_right) at each
query point, making it invariant to the total number of IC segments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_encoders import TimeEncoder


class BreakpointEvolution(nn.Module):
    """Predicts breakpoint positions over time from segment embeddings.

    Creates breakpoint embeddings from adjacent segment pairs, then uses
    cross-attention (time queries -> breakpoint keys/values) to predict
    trajectory positions.

    Args:
        hidden_dim: Embedding dimension (must match seg_emb dimension).
        num_traj_cross_layers: Cross-attention layers in trajectory decoder.
        num_time_layers: MLP layers in time encoder.
        num_freq_t: Fourier frequencies for time encoding.
        num_heads: Attention heads for cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_traj_cross_layers: int = 2,
        num_time_layers: int = 2,
        num_freq_t: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encode adjacent segment pairs into breakpoint embeddings
        self.bp_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encode query times
        self.time_encoder = TimeEncoder(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_frequencies=num_freq_t,
            num_layers=num_time_layers,
            dropout=dropout,
        )

        # Reuse TrajectoryDecoderTransformer architecture inline
        # (cross-attention + additive combination + position head)
        from ..traj_transformer import TrajectoryDecoderTransformer

        self.traj_decoder = TrajectoryDecoderTransformer(
            hidden_dim=hidden_dim,
            num_cross_layers=num_traj_cross_layers,
            num_attention_heads=num_heads,
        )

    def forward(
        self,
        seg_emb: torch.Tensor,
        disc_mask: torch.Tensor,
        t_unique: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict breakpoint positions over time.

        Args:
            seg_emb: Segment embeddings (B, K, H) after self-attention.
            disc_mask: Discontinuity validity mask (B, D) where D = K.
                1 = valid breakpoint between segments i and i+1.
            t_unique: Unique time values (B, nt).
            return_embeddings: If True, also return breakpoint embeddings.

        Returns:
            Predicted breakpoint positions (B, D, nt), clamped to [0, 1].
            If return_embeddings=True, returns (positions, bp_emb) tuple.
        """
        B, K, H = seg_emb.shape
        D = disc_mask.shape[1]  # D = K (max_pieces = max_discontinuities)

        # Create breakpoint embeddings from adjacent segment pairs
        # Pad seg_emb with zeros for the (K+1)-th segment
        seg_padded = F.pad(seg_emb, (0, 0, 0, 1))  # (B, K+1, H)
        left_seg = seg_padded[:, :D]  # (B, D, H)
        right_seg = seg_padded[:, 1 : D + 1]  # (B, D, H)
        bp_input = torch.cat([left_seg, right_seg], dim=-1)  # (B, D, 2H)
        bp_emb = self.bp_encoder(bp_input)  # (B, D, H)
        bp_emb = bp_emb * disc_mask.unsqueeze(-1)  # zero out padded

        # Encode query times
        time_emb = self.time_encoder(t_unique)  # (B, nt, H)

        # Cross-attention: time queries -> breakpoint keys/values
        positions = self.traj_decoder(bp_emb, time_emb, disc_mask)  # (B, D, nt)

        if return_embeddings:
            return positions, bp_emb
        return positions
