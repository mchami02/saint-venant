"""Encoder modules for wavefront learning.

Contains:
- FourierFeatures: Fourier feature encoding for positional information
- TimeEncoder: Trunk network for encoding query times
- DiscontinuityEncoder: Branch network for encoding discontinuities
- SpaceTimeEncoder: Encoder for (t, x) coordinate pairs
"""

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Fourier feature encoding for positional information.

    Maps scalar inputs to higher-dimensional space using sinusoidal features,
    which helps MLPs learn high-frequency functions.

    Args:
        num_frequencies: Number of frequency bands.
        include_input: Whether to include the original input.
    """

    def __init__(self, num_frequencies: int = 8, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        # Frequencies from 2^0 to 2^(num_frequencies-1)
        frequencies = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer("frequencies", frequencies)

    @property
    def output_dim(self) -> int:
        """Output dimension of the encoding."""
        dim = 2 * self.num_frequencies  # sin and cos
        if self.include_input:
            dim += 1
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier encoding.

        Args:
            x: Input tensor of shape (..., 1) or (...,).

        Returns:
            Encoded tensor of shape (..., output_dim).
        """
        if x.dim() == 1 or x.shape[-1] != 1:
            x = x.unsqueeze(-1)

        # Scale input to [0, 2*pi] range for frequencies
        scaled = x * math.pi * self.frequencies  # (..., num_frequencies)

        # Compute sin and cos features
        features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

        if self.include_input:
            features = torch.cat([x, features], dim=-1)

        return features


class TimeEncoder(nn.Module):
    """Trunk network: encodes query times using Fourier features + MLP.

    Args:
        hidden_dim: Hidden dimension of the MLP.
        output_dim: Output dimension (latent space dimension).
        num_frequencies: Number of Fourier frequency bands.
        num_layers: Number of MLP layers.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_frequencies: int = 32,
        num_layers: int = 3,
    ):
        super().__init__()
        self.fourier = FourierFeatures(num_frequencies=num_frequencies)

        layers = []
        in_dim = self.fourier.output_dim
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Encode query times.

        Args:
            times: Query times of shape (B, T).

        Returns:
            Encoded times of shape (B, T, output_dim).
        """
        # Fourier encode each time
        B, T = times.shape
        times_flat = times.reshape(-1)  # (B*T,)
        encoded = self.fourier(times_flat)  # (B*T, fourier_dim)
        encoded = self.mlp(encoded)  # (B*T, output_dim)
        return encoded.reshape(B, T, -1)  # (B, T, output_dim)


class DiscontinuityEncoder(nn.Module):
    """Branch network: encodes discontinuities using Fourier features + MLP.

    Each discontinuity is processed independently (no cross-attention).
    Handles variable numbers of discontinuities through masking.

    Args:
        input_dim: Dimension of discontinuity features (default 3: x, rho_L, rho_R).
        hidden_dim: Hidden dimension of the MLP.
        output_dim: Output dimension (latent space dimension).
        num_frequencies: Number of Fourier frequency bands for x coordinate.
        num_layers: Number of MLP layers.
        dropout: Dropout rate (unused, kept for API compatibility).
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_frequencies: int = 16,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Fourier features for x coordinate
        self.fourier_x = FourierFeatures(
            num_frequencies=num_frequencies, include_input=True
        )

        # Input dimension: Fourier features for x + remaining features
        mlp_input_dim = self.fourier_x.output_dim + (input_dim - 1)

        # Build MLP layers
        layers = []
        in_dim = mlp_input_dim
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        discontinuities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode discontinuities.

        Args:
            discontinuities: Discontinuity features of shape (B, D, input_dim)
                where first column is x_position, rest are extra features.
            mask: Validity mask of shape (B, D) where 1 = valid, 0 = padding.

        Returns:
            Encoded discontinuities of shape (B, D, output_dim).
        """
        B, D, _ = discontinuities.shape

        # Extract x coordinate and remaining features
        x_coord = discontinuities[:, :, 0]  # (B, D)
        extra_features = discontinuities[:, :, 1:]  # (B, D, input_dim-1)

        # Apply Fourier features to x coordinate
        x_flat = x_coord.reshape(-1)  # (B*D,)
        x_fourier = self.fourier_x(x_flat)  # (B*D, fourier_dim)
        x_fourier = x_fourier.reshape(B, D, -1)  # (B, D, fourier_dim)

        # Concatenate Fourier features with extra features
        features = torch.cat([x_fourier, extra_features], dim=-1)

        # Apply MLP
        output = self.mlp(features)  # (B, D, output_dim)

        # Zero out padded positions
        output = output * mask.unsqueeze(-1)

        return output


class SpaceTimeEncoder(nn.Module):
    """Encodes (t, x) coordinate pairs using Fourier features.

    Uses separate Fourier encodings for time and space coordinates,
    then concatenates and projects to output dimension.

    Args:
        hidden_dim: Hidden dimension of the MLP.
        output_dim: Output dimension (latent space dimension).
        num_frequencies_t: Number of Fourier frequency bands for time.
        num_frequencies_x: Number of Fourier frequency bands for space.
        num_layers: Number of MLP layers.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_frequencies_t: int = 16,
        num_frequencies_x: int = 16,
        num_layers: int = 3,
    ):
        super().__init__()
        self.fourier_t = FourierFeatures(
            num_frequencies=num_frequencies_t, include_input=True
        )
        self.fourier_x = FourierFeatures(
            num_frequencies=num_frequencies_x, include_input=True
        )

        # Input dimension: fourier_t + fourier_x
        input_dim = self.fourier_t.output_dim + self.fourier_x.output_dim

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(
        self,
        t_coords: torch.Tensor,
        x_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Encode (t, x) coordinate pairs.

        Args:
            t_coords: Time coordinates of shape (B, nt, nx) or (B*nt*nx,).
            x_coords: Space coordinates of shape (B, nt, nx) or (B*nt*nx,).

        Returns:
            Encoded coordinates of shape (B, nt, nx, output_dim) or
            (B*nt*nx, output_dim).
        """
        original_shape = t_coords.shape
        is_3d = t_coords.dim() == 3

        if is_3d:
            B, nt, nx = original_shape
            t_flat = t_coords.reshape(-1)  # (B*nt*nx,)
            x_flat = x_coords.reshape(-1)  # (B*nt*nx,)
        else:
            t_flat = t_coords
            x_flat = x_coords

        # Fourier encode time and space separately
        t_encoded = self.fourier_t(t_flat)  # (B*nt*nx, fourier_t_dim)
        x_encoded = self.fourier_x(x_flat)  # (B*nt*nx, fourier_x_dim)

        # Concatenate and project
        combined = torch.cat([t_encoded, x_encoded], dim=-1)
        output = self.mlp(combined)  # (B*nt*nx, output_dim)

        if is_3d:
            output = output.reshape(B, nt, nx, -1)

        return output
