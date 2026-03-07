"""Segment encoder for ARZ traffic flow system.

Encodes IC segments with both density (rho) and velocity (v) per segment,
plus ARZ-specific physics features (eigenvalues, pressure, Riemann invariant w).

Same pattern as SegmentPhysicsEncoder in characteristic_features.py but
adapted for the 2-variable ARZ system.
"""

import torch
import torch.nn as nn

from .arz_physics import ARZPhysics
from .feature_encoders import FourierFeatures


class ARZSegmentPhysicsEncoder(nn.Module):
    """Encodes IC segments with ARZ physics-augmented features.

    For each constant piece k with density rho_k and velocity v_k on [xs[k], xs[k+1]]:
      - x_center = (xs[k] + xs[k+1]) / 2    (Fourier-encoded)
      - width = xs[k+1] - xs[k]              (Fourier-encoded)
      - rho_k                                 (density)
      - v_k                                   (velocity)
      - lam1_k = v_k                          (1st eigenvalue)
      - lam2_k = v_k - gamma * rho_k^gamma   (2nd eigenvalue)
      - p_k = rho_k^gamma                    (pressure)
      - w_k = v_k + p_k                      (Riemann invariant)

    Args:
        hidden_dim: Output embedding dimension.
        num_frequencies: Fourier frequency bands for spatial features.
        num_layers: MLP depth.
        arz_physics: ARZPhysics instance.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_frequencies: int = 8,
        num_layers: int = 2,
        arz_physics: ARZPhysics | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.arz_physics = arz_physics or ARZPhysics()

        self.fourier_x = FourierFeatures(
            num_frequencies=num_frequencies, include_input=True
        )

        # Input: fourier(x_center) + fourier(width) + 6 scalar features
        # [rho, v, lam1, lam2, p, w]
        input_dim = 2 * self.fourier_x.output_dim + 6

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(
        self,
        xs: torch.Tensor,
        ks_rho: torch.Tensor,
        ks_v: torch.Tensor,
        pieces_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode IC segments with ARZ physics features.

        Args:
            xs: Breakpoint positions (B, K+1).
            ks_rho: Density piece values (B, K).
            ks_v: Velocity piece values (B, K).
            pieces_mask: Validity mask (B, K), 1 = valid.

        Returns:
            Segment embeddings (B, K, hidden_dim).
        """
        B, K = ks_rho.shape

        # Spatial features
        x_center = (xs[:, :-1] + xs[:, 1:]) / 2  # (B, K)
        width = xs[:, 1:] - xs[:, :-1]  # (B, K)

        # Physics features
        lam1, lam2 = self.arz_physics.eigenvalues(ks_rho, ks_v)  # each (B, K)
        p_k = self.arz_physics.pressure(ks_rho)  # (B, K)
        w_k = ks_v + p_k  # Riemann invariant (B, K)

        # Fourier encode spatial features
        x_center_enc = self.fourier_x(
            x_center.reshape(-1)
        ).reshape(B, K, -1)  # (B, K, F)
        width_enc = self.fourier_x(
            width.reshape(-1)
        ).reshape(B, K, -1)  # (B, K, F)

        # Stack scalar features
        scalar_features = torch.stack(
            [ks_rho, ks_v, lam1, lam2, p_k, w_k], dim=-1
        )  # (B, K, 6)

        # Concatenate all features
        features = torch.cat(
            [x_center_enc, width_enc, scalar_features], dim=-1
        )  # (B, K, 2F+6)

        # MLP projection
        output = self.mlp(features)  # (B, K, H)

        # Zero out padded positions
        output = output * pieces_mask.unsqueeze(-1)

        return output
