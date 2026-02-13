"""Characteristic-space feature modules for CharNO.

Contains:
- SegmentPhysicsEncoder: Augments IC segments with physics features and encodes them.
- CharacteristicFeatureComputer: Computes characteristic-relative coordinates
  for each (query point, segment) pair.
"""

import torch
import torch.nn as nn

from .feature_encoders import FourierFeatures
from .flux import DEFAULT_FLUX, Flux


class SegmentPhysicsEncoder(nn.Module):
    """Encodes IC segments with physics-augmented features.

    For each constant piece k with value rho_k on [xs[k], xs[k+1]]:
      - x_center = (xs[k] + xs[k+1]) / 2
      - width = xs[k+1] - xs[k]
      - rho_k = ks[k]
      - lambda_k = flux.derivative(rho_k)    (characteristic speed)
      - f_k = flux(rho_k)                    (flux value)

    Spatial features (x_center, width) are Fourier-encoded.
    All features are concatenated and projected via MLP.

    Args:
        hidden_dim: Output embedding dimension.
        num_frequencies: Fourier frequency bands for spatial features.
        num_layers: MLP depth.
        flux: Flux function instance.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_frequencies: int = 8,
        num_layers: int = 2,
        flux: Flux | None = None,
    ):
        super().__init__()
        self.flux = flux or DEFAULT_FLUX()

        self.fourier_x = FourierFeatures(
            num_frequencies=num_frequencies, include_input=True
        )

        # Input: fourier(x_center) + fourier(width) + [rho, lambda, flux_val]
        input_dim = 2 * self.fourier_x.output_dim + 3

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(
        self,
        xs: torch.Tensor,
        ks: torch.Tensor,
        pieces_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode IC segments.

        Args:
            xs: Breakpoint positions (B, K+1).
            ks: Piece values (B, K).
            pieces_mask: Validity mask (B, K), 1 = valid.

        Returns:
            Segment embeddings (B, K, hidden_dim).
        """
        B, K = ks.shape

        # Compute physics features
        x_center = (xs[:, :-1] + xs[:, 1:]) / 2  # (B, K)
        width = xs[:, 1:] - xs[:, :-1]  # (B, K)
        lambda_k = self.flux.derivative(ks)  # (B, K)
        flux_k = self.flux(ks)  # (B, K)

        # Fourier encode spatial features
        x_center_enc = self.fourier_x(
            x_center.reshape(-1)
        ).reshape(B, K, -1)  # (B, K, F)
        width_enc = self.fourier_x(
            width.reshape(-1)
        ).reshape(B, K, -1)  # (B, K, F)

        # Concatenate all features
        scalar_features = torch.stack(
            [ks, lambda_k, flux_k], dim=-1
        )  # (B, K, 3)
        features = torch.cat(
            [x_center_enc, width_enc, scalar_features], dim=-1
        )  # (B, K, 2F+3)

        # MLP projection
        output = self.mlp(features)  # (B, K, H)

        # Zero out padded positions
        output = output * pieces_mask.unsqueeze(-1)

        return output


class CharacteristicFeatureComputer(nn.Module):
    """Computes characteristic-relative features for (query, segment) pairs.

    For each query (t, x) and segment k with center x_c, char speed lambda_k:
      - xi_k = (x - x_c) / max(t, eps)           similarity variable
      - char_shift_k = x - x_c - lambda_k * t     characteristic offset
      - dist_left_k = x - (xs[k] + lambda_k * t)  distance to left char boundary
      - dist_right_k = x - (xs[k+1] + lambda_k * t) distance to right char boundary
      - t_val = t                                  time

    Features are Fourier-encoded and projected via MLP.

    Args:
        hidden_dim: Output feature dimension.
        num_frequencies: Fourier frequency bands.
        num_layers: MLP depth.
        flux: Flux function instance.
        eps: Regularization for t=0 division.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_frequencies: int = 8,
        num_layers: int = 2,
        flux: Flux | None = None,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.flux = flux or DEFAULT_FLUX()
        self.eps = eps

        self.fourier = FourierFeatures(
            num_frequencies=num_frequencies, include_input=True
        )

        # 5 features, each Fourier-encoded
        input_dim = 5 * self.fourier.output_dim

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(
        self,
        t_coords: torch.Tensor,
        x_coords: torch.Tensor,
        xs: torch.Tensor,
        ks: torch.Tensor,
        pieces_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute characteristic features for all (query, segment) pairs.

        Args:
            t_coords: Time coordinates (B, nt, nx).
            x_coords: Space coordinates (B, nt, nx).
            xs: Breakpoint positions (B, K+1).
            ks: Piece values (B, K).
            pieces_mask: Validity mask (B, K).

        Returns:
            Characteristic features (B, Q, K, hidden_dim) where Q = nt * nx.
        """
        B, nt, nx = t_coords.shape
        K = ks.shape[1]
        Q = nt * nx

        # Flatten spatial grid
        t_flat = t_coords.reshape(B, Q)  # (B, Q)
        x_flat = x_coords.reshape(B, Q)  # (B, Q)

        # Segment properties
        x_center = (xs[:, :-1] + xs[:, 1:]) / 2  # (B, K)
        x_left = xs[:, :-1]  # (B, K)
        x_right = xs[:, 1:]  # (B, K)
        lambda_k = self.flux.derivative(ks)  # (B, K)

        # Expand for broadcasting: (B, Q, 1) and (B, 1, K)
        t_exp = t_flat.unsqueeze(2)  # (B, Q, 1)
        x_exp = x_flat.unsqueeze(2)  # (B, Q, 1)
        xc_exp = x_center.unsqueeze(1)  # (B, 1, K)
        xl_exp = x_left.unsqueeze(1)  # (B, 1, K)
        xr_exp = x_right.unsqueeze(1)  # (B, 1, K)
        lam_exp = lambda_k.unsqueeze(1)  # (B, 1, K)

        # Compute 5 characteristic features
        t_safe = torch.clamp(t_exp, min=self.eps)
        xi = (x_exp - xc_exp) / t_safe  # similarity variable
        char_shift = x_exp - xc_exp - lam_exp * t_exp  # char offset
        dist_left = x_exp - (xl_exp + lam_exp * t_exp)  # left boundary
        dist_right = x_exp - (xr_exp + lam_exp * t_exp)  # right boundary
        t_feat = t_exp.expand_as(xi)  # time

        # Stack and Fourier-encode each feature separately
        # Shape of each: (B, Q, K)
        # Clamp features to prevent extreme values that cause NaN through
        # Fourier encoding (xi can reach ~1000 at t≈0, producing gradients
        # of magnitude π·128·1000 ≈ 400,000 with num_frequencies=8).
        # Physically meaningful range for xi is [-1, 1] (characteristic speeds);
        # wider clamp preserves information for out-of-range queries.
        feat_clamp = 10.0
        xi = torch.clamp(xi, -feat_clamp, feat_clamp)
        char_shift = torch.clamp(char_shift, -feat_clamp, feat_clamp)
        dist_left = torch.clamp(dist_left, -feat_clamp, feat_clamp)
        dist_right = torch.clamp(dist_right, -feat_clamp, feat_clamp)

        features = [xi, char_shift, dist_left, dist_right, t_feat]

        # Fourier encode each feature and concatenate
        encoded_parts = []
        for feat in features:
            flat = feat.reshape(-1)  # (B*Q*K,)
            enc = self.fourier(flat)  # (B*Q*K, F)
            enc = enc.reshape(B, Q, K, -1)  # (B, Q, K, F)
            encoded_parts.append(enc)

        combined = torch.cat(encoded_parts, dim=-1)  # (B, Q, K, 5*F)

        # MLP projection
        output = self.mlp(combined)  # (B, Q, K, H_char)

        # Zero out padded segments
        output = output * pieces_mask.unsqueeze(1).unsqueeze(-1)

        return output
