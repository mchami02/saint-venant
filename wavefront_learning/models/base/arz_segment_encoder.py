"""ARZ segment encoder: physics-augmented segment features for 2-variable ICs.

Like SegmentPhysicsEncoder but for the ARZ system where each segment has
both density (rho) and velocity (v) values.

Per segment k on [xs[k], xs[k+1]]:
  - x_center, width (Fourier-encoded)
  - rho_k, v_k (raw values)
  - lambda_1_k = v_k (contact eigenvalue)
  - lambda_2_k = v_k - rho_k * p'(rho_k) (GNL eigenvalue)
  - p_k = rho_k^gamma (pressure)
"""

import torch
import torch.nn as nn

from .arz_physics import ARZPhysics
from .feature_encoders import FourierFeatures


class ARZSegmentPhysicsEncoder(nn.Module):
    """Encodes ARZ IC segments with physics-augmented features.

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
        num_frequencies: int | None = 8,
        num_layers: int = 2,
        arz_physics: ARZPhysics | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.physics = arz_physics or ARZPhysics()

        if num_frequencies is not None:
            self.fourier_x = FourierFeatures(
                num_frequencies=num_frequencies, include_input=True
            )
            spatial_dim = self.fourier_x.output_dim
        else:
            self.fourier_x = None
            spatial_dim = 1

        # Input: 2 * spatial_dim (x_center + width) + 5 scalar features
        # [rho, v, lambda_1, lambda_2, pressure]
        num_scalars = 5
        input_dim = 2 * spatial_dim + num_scalars

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
        ks: torch.Tensor,
        ks_v: torch.Tensor,
        pieces_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode ARZ IC segments.

        Args:
            xs: Breakpoint positions (B, K+1).
            ks: Density piece values (B, K).
            ks_v: Velocity piece values (B, K).
            pieces_mask: Validity mask (B, K), 1 = valid.

        Returns:
            Segment embeddings (B, K, hidden_dim).
        """
        B, K = ks.shape

        # Spatial features
        x_center = (xs[:, :-1] + xs[:, 1:]) / 2  # (B, K)
        width = xs[:, 1:] - xs[:, :-1]  # (B, K)

        # Physics features
        lam1, lam2 = self.physics.eigenvalues(ks, ks_v)  # each (B, K)
        p_k = self.physics.pressure(ks)  # (B, K)

        # Encode spatial features
        if self.fourier_x is not None:
            x_center_enc = self.fourier_x(x_center.reshape(-1)).reshape(B, K, -1)
            width_enc = self.fourier_x(width.reshape(-1)).reshape(B, K, -1)
        else:
            x_center_enc = x_center.unsqueeze(-1)
            width_enc = width.unsqueeze(-1)

        # Stack scalar features: [rho, v, lambda_1, lambda_2, pressure]
        scalar_features = torch.stack([ks, ks_v, lam1, lam2, p_k], dim=-1)  # (B, K, 5)
        features = torch.cat(
            [x_center_enc, width_enc, scalar_features], dim=-1
        )

        # MLP projection
        output = self.mlp(features)  # (B, K, H)

        # Zero out padded positions
        output = output * pieces_mask.unsqueeze(-1)

        return output
