"""VAE encoder for latent diffusion DeepONet.

2D convolutional encoder that maps a solution grid to a latent distribution
parameterized by (mean, logvar).
"""

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """2D convolutional VAE encoder.

    Encodes a solution grid of shape (B, 1, nt, nx) into a latent
    Gaussian distribution parameterized by mean and log-variance vectors.

    Architecture: Conv2d [1→32→64→128] with stride 2, GELU+BatchNorm,
    adaptive average pool, flatten, then two linear heads for mean and logvar.

    Args:
        latent_dim: Dimension of the latent space.
        base_channels: Number of channels in the first conv layer.
    """

    def __init__(self, latent_dim: int = 32, base_channels: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        ch = base_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.GELU(),
            nn.Conv2d(ch * 2, ch * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.fc_mean = nn.Linear(ch * 4, latent_dim)
        self.fc_logvar = nn.Linear(ch * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input grid to latent distribution parameters.

        Args:
            x: Solution grid of shape (B, 1, nt, nx).

        Returns:
            Tuple of (mean, logvar), each of shape (B, latent_dim).
        """
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mean + std * eps.

        Args:
            mean: Mean of shape (B, latent_dim).
            logvar: Log-variance of shape (B, latent_dim).

        Returns:
            Sampled latent vector of shape (B, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps
