"""VAE reconstruction loss with KL divergence and beta warmup."""

import torch
import torch.nn.functional as F

from .base import BaseLoss


class VAEReconstructionLoss(BaseLoss):
    """MSE reconstruction + beta-weighted KL divergence for VAE training.

    Supports linear beta warmup over a configurable number of epochs.

    Args:
        beta: Maximum KL weight.
        beta_warmup_epochs: Number of epochs for linear beta warmup from 0.
    """

    def __init__(self, beta: float = 0.01, beta_warmup_epochs: int = 10):
        super().__init__()
        self.beta = beta
        self.beta_warmup_epochs = beta_warmup_epochs
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for beta warmup schedule."""
        self._epoch = epoch

    @property
    def effective_beta(self) -> float:
        if self.beta_warmup_epochs <= 0:
            return self.beta
        ramp = min(self._epoch / self.beta_warmup_epochs, 1.0)
        return self.beta * ramp

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute VAE loss = MSE + beta * KL.

        Expected output_dict keys:
            - output_grid: (B, 1, nt, nx)
            - z_mean: (B, latent_dim)
            - z_logvar: (B, latent_dim)
        """
        pred = output_dict["output_grid"]
        mean = output_dict["z_mean"]
        logvar = output_dict["z_logvar"]

        recon_mse = F.mse_loss(pred, target)
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        beta = self.effective_beta
        loss = recon_mse + beta * kl

        return loss, {
            "recon_mse": recon_mse.item(),
            "kl": kl.item(),
            "effective_beta": beta,
        }
