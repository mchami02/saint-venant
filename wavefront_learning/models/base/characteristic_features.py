"""Characteristic-space feature modules for CharNO.

Contains:
- SegmentPhysicsEncoder: Augments IC segments with physics features and encodes them.
- CharacteristicFeatureComputer: Computes characteristic-relative coordinates
  for each (query point, segment) pair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        include_cumulative_mass: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.flux = flux or DEFAULT_FLUX()
        self.include_cumulative_mass = include_cumulative_mass

        self.fourier_x = FourierFeatures(
            num_frequencies=num_frequencies, include_input=True
        )

        # Input: fourier(x_center) + fourier(width) + scalar features
        # With cumulative mass: [rho, lambda, flux_val, N_k_normalized] = 4
        # Without: [rho, lambda, flux_val] = 3
        num_scalars = 4 if include_cumulative_mass else 3
        input_dim = 2 * self.fourier_x.output_dim + num_scalars

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
        segments: torch.Tensor,
        segments_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode IC segments.

        Args:
            segments: Segment features (B, S, 3) with [x_start, x_end, rho].
            segments_mask: Validity mask (B, S), 1 = valid.

        Returns:
            Segment embeddings (B, S, hidden_dim).
        """
        B, S, _ = segments.shape

        x_start = segments[:, :, 0]  # (B, S)
        x_end = segments[:, :, 1]  # (B, S)
        rho = segments[:, :, 2]  # (B, S)

        # Compute physics features
        x_center = (x_start + x_end) / 2  # (B, S)
        width = x_end - x_start  # (B, S)
        lambda_k = self.flux.derivative(rho)  # (B, S)
        flux_k = self.flux(rho)  # (B, S)

        # Fourier encode spatial features
        x_center_enc = self.fourier_x(
            x_center.reshape(-1)
        ).reshape(B, S, -1)  # (B, S, F)
        width_enc = self.fourier_x(
            width.reshape(-1)
        ).reshape(B, S, -1)  # (B, S, F)

        # Concatenate all features
        if self.include_cumulative_mass:
            # Cumulative IC integral: N_k = Σ_{j<k} ρ_j·w_j (exclusive prefix sum)
            rho_width = rho * width  # (B, S)
            N_k = torch.cumsum(rho_width, dim=1) - rho_width  # (B, S)
            total_mass = (rho_width * segments_mask).sum(dim=1, keepdim=True).clamp(min=1e-8)
            N_k_normalized = N_k / total_mass  # (B, S) in [0, 1]
            scalar_features = torch.stack(
                [rho, lambda_k, flux_k, N_k_normalized], dim=-1
            )  # (B, S, 4)
        else:
            scalar_features = torch.stack(
                [rho, lambda_k, flux_k], dim=-1
            )  # (B, S, 3)
        features = torch.cat(
            [x_center_enc, width_enc, scalar_features], dim=-1
        )  # (B, S, 2F+3/4)

        # MLP projection
        output = self.mlp(features)  # (B, S, H)

        # Zero out padded positions
        output = output * segments_mask.unsqueeze(-1)

        return output


class DiscontinuityPhysicsEncoder(nn.Module):
    """Encodes IC discontinuities with physics-augmented features.

    For each discontinuity d at (x_d, rho_L, rho_R):
      - x_d: discontinuity position (Fourier-encoded)
      - rho_L, rho_R: left and right density values
      - lambda_L = flux.derivative(rho_L): left characteristic speed
      - lambda_R = flux.derivative(rho_R): right characteristic speed
      - s_d = flux.shock_speed(rho_L, rho_R): Rankine-Hugoniot shock speed

    Spatial position is Fourier-encoded; physics scalars are concatenated raw.
    All features are projected via MLP.

    Args:
        hidden_dim: Output embedding dimension.
        num_frequencies: Fourier frequency bands for position encoding.
        num_layers: MLP depth.
        flux: Flux function instance.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_frequencies: int = 8,
        num_layers: int = 2,
        flux: Flux | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.flux = flux or DEFAULT_FLUX()

        self.fourier_x = FourierFeatures(
            num_frequencies=num_frequencies, include_input=True
        )

        # Input: fourier(x_d) + 5 scalar features
        # [rho_L, rho_R, lambda_L, lambda_R, s_d]
        input_dim = self.fourier_x.output_dim + 5

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
        discontinuities: torch.Tensor,
        disc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode IC discontinuities.

        Args:
            discontinuities: Discontinuity features (B, D, 3) with [x_d, rho_L, rho_R].
            disc_mask: Validity mask (B, D), 1 = valid.

        Returns:
            Discontinuity embeddings (B, D, hidden_dim).
        """
        B, D, _ = discontinuities.shape

        x_d = discontinuities[:, :, 0]  # (B, D)
        rho_L = discontinuities[:, :, 1]  # (B, D)
        rho_R = discontinuities[:, :, 2]  # (B, D)

        # Compute physics features
        lambda_L = self.flux.derivative(rho_L)  # (B, D)
        lambda_R = self.flux.derivative(rho_R)  # (B, D)
        s_d = self.flux.shock_speed(rho_L, rho_R)  # (B, D)

        # Fourier encode position
        x_enc = self.fourier_x(x_d.reshape(-1)).reshape(B, D, -1)  # (B, D, F)

        # Stack scalar physics features
        scalars = torch.stack(
            [rho_L, rho_R, lambda_L, lambda_R, s_d], dim=-1
        )  # (B, D, 5)

        # Concatenate and project
        features = torch.cat([x_enc, scalars], dim=-1)  # (B, D, F+5)
        output = self.mlp(features)  # (B, D, H)

        # Zero out padded positions
        output = output * disc_mask.unsqueeze(-1)

        return output


class CharacteristicFeatureComputer(nn.Module):
    """Computes characteristic-relative features for (query, segment) pairs.

    For each query (t, x) and segment k with center x_c, char speed lambda_k:
      - xi_k = (x - x_c) / max(t, eps)           similarity variable
      - char_shift_k = x - x_c - lambda_k * t     characteristic offset
      - dist_left_k = x - (xs[k] + lambda_k * t)  distance to left char boundary
      - dist_right_k = x - (xs[k+1] + lambda_k * t) distance to right char boundary
      - t_val = t                                  time
      - s_left_k = shock_speed(ks[k-1], ks[k])    left boundary RH speed
      - s_right_k = shock_speed(ks[k], ks[k+1])   right boundary RH speed
      - t_coll_k = min neighbor collision time     estimated wave interaction time

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

        # Learnable per-feature normalization scales (tanh(feat / scale))
        # Order: xi, char_shift, dist_left, dist_right, t, s_left, s_right, t_coll
        self.norm_scales = nn.Parameter(
            torch.tensor([5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 2.0, 5.0])
        )

        # 8 features, each Fourier-encoded
        # Original 5: xi, char_shift, dist_left, dist_right, t
        # New 3: s_left, s_right (boundary shock speeds), t_coll (collision time)
        input_dim = 8 * self.fourier.output_dim

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
        segments: torch.Tensor,
        segments_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute characteristic features for all (query, segment) pairs.

        Args:
            t_coords: Time coordinates (B, nt, nx).
            x_coords: Space coordinates (B, nt, nx).
            segments: Segment features (B, S, 3) with [x_start, x_end, rho].
            segments_mask: Validity mask (B, S).

        Returns:
            Characteristic features (B, Q, S, hidden_dim) where Q = nt * nx.
        """
        B, nt, nx = t_coords.shape
        S = segments.shape[1]
        Q = nt * nx

        # Flatten spatial grid
        t_flat = t_coords.reshape(B, Q)  # (B, Q)
        x_flat = x_coords.reshape(B, Q)  # (B, Q)

        # Segment properties
        x_left = segments[:, :, 0]  # (B, S)
        x_right = segments[:, :, 1]  # (B, S)
        rho = segments[:, :, 2]  # (B, S)
        x_center = (x_left + x_right) / 2  # (B, S)
        lambda_k = self.flux.derivative(rho)  # (B, S)

        # === Inter-segment boundary features ===
        # Shock speeds at left/right boundaries of each segment.
        # For edge segments, pad with self-values so shock_speed(ρ,ρ) = f'(ρ).
        rho_left_pad = F.pad(rho[:, :-1], (1, 0), value=0.0)  # (B, S)
        rho_left_pad[:, 0] = rho[:, 0]  # leftmost: self-pad
        rho_right_pad = F.pad(rho[:, 1:], (0, 1), value=0.0)  # (B, S)
        rho_right_pad[:, -1] = rho[:, -1]  # rightmost: self-pad

        s_left = self.flux.shock_speed(rho_left_pad, rho)  # (B, S)
        s_right = self.flux.shock_speed(rho, rho_right_pad)  # (B, S)

        # Collision time proxy: time for neighboring characteristics to meet
        xc_left_pad = F.pad(x_center[:, :-1], (1, 0), value=0.0)
        xc_left_pad[:, 0] = x_center[:, 0]
        xc_right_pad = F.pad(x_center[:, 1:], (0, 1), value=1.0)
        xc_right_pad[:, -1] = x_center[:, -1]
        lam_left_pad = F.pad(lambda_k[:, :-1], (1, 0), value=0.0)
        lam_left_pad[:, 0] = lambda_k[:, 0]
        lam_right_pad = F.pad(lambda_k[:, 1:], (0, 1), value=0.0)
        lam_right_pad[:, -1] = lambda_k[:, -1]

        eps_coll = 1e-3
        dist_left_nb = (x_center - xc_left_pad).abs()
        dist_right_nb = (xc_right_pad - x_center).abs()
        speed_diff_left = (lam_left_pad - lambda_k).abs().clamp(min=eps_coll)
        speed_diff_right = (lambda_k - lam_right_pad).abs().clamp(min=eps_coll)
        t_coll = torch.minimum(
            dist_left_nb / speed_diff_left,
            dist_right_nb / speed_diff_right,
        )  # (B, S)

        # Expand for broadcasting: (B, Q, 1) and (B, 1, S)
        t_exp = t_flat.unsqueeze(2)  # (B, Q, 1)
        x_exp = x_flat.unsqueeze(2)  # (B, Q, 1)
        xc_exp = x_center.unsqueeze(1)  # (B, 1, S)
        xl_exp = x_left.unsqueeze(1)  # (B, 1, S)
        xr_exp = x_right.unsqueeze(1)  # (B, 1, S)
        lam_exp = lambda_k.unsqueeze(1)  # (B, 1, S)

        # Compute 5 original characteristic features
        t_safe = torch.clamp(t_exp, min=self.eps)
        xi = (x_exp - xc_exp) / t_safe  # similarity variable
        char_shift = x_exp - xc_exp - lam_exp * t_exp  # char offset
        dist_left = x_exp - (xl_exp + lam_exp * t_exp)  # left boundary
        dist_right = x_exp - (xr_exp + lam_exp * t_exp)  # right boundary
        t_feat = t_exp.expand_as(xi)  # time

        # Expand inter-segment features to (B, Q, S)
        s_left_feat = s_left.unsqueeze(1).expand_as(xi)
        s_right_feat = s_right.unsqueeze(1).expand_as(xi)
        t_coll_feat = t_coll.unsqueeze(1).expand_as(xi)

        # Soft normalization: tanh(feat / scale) preserves ordering, maintains
        # gradients everywhere, and maps to [-1, 1]. Replaces hard clamp which
        # destroyed information for distant segments (zero gradient outside range).
        features_raw = [
            xi, char_shift, dist_left, dist_right, t_feat,
            s_left_feat, s_right_feat, t_coll_feat,
        ]
        features = []
        for i, feat in enumerate(features_raw):
            scale = self.norm_scales[i].abs().clamp(min=0.1)
            features.append(torch.tanh(feat / scale))

        # Fourier encode each feature and concatenate
        encoded_parts = []
        for feat in features:
            flat = feat.reshape(-1)  # (B*Q*S,)
            enc = self.fourier(flat)  # (B*Q*S, F)
            enc = enc.reshape(B, Q, S, -1)  # (B, Q, S, F)
            encoded_parts.append(enc)

        combined = torch.cat(encoded_parts, dim=-1)  # (B, Q, S, 8*F)

        # MLP projection
        output = self.mlp(combined)  # (B, Q, S, H_char)

        # Zero out padded segments
        output = output * segments_mask.unsqueeze(1).unsqueeze(-1)

        return output


class TimeConditioner(nn.Module):
    """FiLM-based time conditioning for segment embeddings.

    Modulates static segment embeddings with time-dependent scale and shift:
        seg_emb(t) = gamma(t, seg_emb) * seg_emb + beta(t, seg_emb)

    Initialized near identity (gamma=1, beta=0) so the model starts
    equivalent to the static version and learns time dependence gradually.

    Args:
        hidden_dim: Segment embedding dimension.
        num_time_frequencies: Fourier frequencies for time encoding.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_time_frequencies: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fourier_t = FourierFeatures(
            num_frequencies=num_time_frequencies, include_input=True
        )
        # Input: fourier(t) + seg_emb -> scale (gamma) and shift (beta)
        input_dim = self.fourier_t.output_dim + hidden_dim
        self.film_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        # Initialize near identity: gamma ~ 1, beta ~ 0
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)
        with torch.no_grad():
            self.film_net[-1].bias[:hidden_dim] = 1.0  # gamma = 1

    def forward(
        self,
        seg_emb: torch.Tensor,
        t_unique: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate segment embeddings by time.

        Computes FiLM parameters per unique time step. The caller is
        responsible for expanding to spatial dimensions if needed.

        Args:
            seg_emb: Static segment embeddings (B, K, H).
            t_unique: Unique time values (B, nt), one per time step.

        Returns:
            Time-conditioned segment embeddings (B, nt, K, H).
        """
        B, K, H = seg_emb.shape
        nt = t_unique.shape[1]

        # Fourier-encode unique times: (B, nt) -> (B, nt, F_t)
        t_enc = self.fourier_t(t_unique.reshape(-1)).reshape(B, nt, -1)

        # Expand for (time, segment) pairs — much smaller than (query, segment)
        t_enc_exp = t_enc.unsqueeze(2).expand(-1, -1, K, -1)  # (B, nt, K, F_t)
        seg_emb_t = seg_emb.unsqueeze(1).expand(-1, nt, -1, -1)  # (B, nt, K, H)

        # Concatenate and produce FiLM parameters
        film_input = torch.cat(
            [t_enc_exp, seg_emb_t], dim=-1
        )  # (B, nt, K, F_t + H)
        film_params = self.film_net(film_input)  # (B, nt, K, 2*H)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, nt, K, H)

        # Apply FiLM modulation at per-timestep level
        return gamma * seg_emb_t + beta  # (B, nt, K, H)
