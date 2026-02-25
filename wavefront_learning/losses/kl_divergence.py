"""KL divergence loss between two diagonal Gaussians with free bits.

Used by the CVAE DeepONet to regularize the approximate posterior
q(z|a,u) towards the learned prior p(z|a). Supports:
- Per-dimension free bits to prevent posterior collapse
- External beta scheduling via the mutable `beta` attribute
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss


class KLDivergenceLoss(BaseLoss):
    """KL divergence loss between inference and prior Gaussians.

    Computes KL(q(z|a,u) || p(z|a)) for diagonal Gaussians with free bits.
    At test time (when z_mean_q is absent from output_dict), returns zero.

    The `beta` attribute is mutable and should be set externally by the
    training loop for KL annealing.

    Args:
        free_bits: Minimum KL per latent dimension (prevents posterior collapse).
    """

    def __init__(self, free_bits: float = 0.01):
        super().__init__()
        self.free_bits = free_bits
        self.beta = 1.0  # Mutable, set externally for KL annealing

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute KL divergence loss.

        Args:
            input_dict: Input dictionary (unused).
            output_dict: Model output dict. Expected keys:
                - z_mean_p, z_logvar_p: Prior parameters
                - z_mean_q, z_logvar_q: Inference parameters (training only)
            target: Target tensor (unused).

        Returns:
            Tuple of (beta-weighted KL loss, components dict).
        """
        # At test time, inference network is not used
        if "z_mean_q" not in output_dict:
            zero = torch.tensor(0.0, device=target.device)
            return zero, {"kl": 0.0, "kl_raw": 0.0, "kl_beta": self.beta}

        z_mean_q = output_dict["z_mean_q"]
        z_logvar_q = output_dict["z_logvar_q"]
        z_mean_p = output_dict["z_mean_p"]
        z_logvar_p = output_dict["z_logvar_p"]

        # Per-dimension KL: KL(q || p) for diagonal Gaussians
        # = 0.5 * (logvar_p - logvar_q + exp(logvar_q - logvar_p)
        #          + (mean_q - mean_p)^2 / exp(logvar_p) - 1)
        kl_per_dim = 0.5 * (
            z_logvar_p - z_logvar_q
            + torch.exp(z_logvar_q - z_logvar_p)
            + (z_mean_q - z_mean_p).pow(2) / torch.exp(z_logvar_p)
            - 1.0
        )  # (B, latent_dim)

        # Raw KL (for logging): mean over batch, sum over latent dims
        kl_raw = kl_per_dim.mean(dim=0).sum()

        # Free bits: soft clamp per-dimension KL (averaged over batch)
        # softplus(x - λ) + λ ≈ max(x, λ) but is differentiable everywhere
        kl_mean = kl_per_dim.mean(dim=0)  # (latent_dim,)
        kl_soft = F.softplus(kl_mean - self.free_bits) + self.free_bits
        kl_loss = kl_soft.sum()

        # Apply beta weighting
        weighted_loss = self.beta * kl_loss

        return weighted_loss, {
            "kl": kl_loss.item(),
            "kl_raw": kl_raw.item(),
            "kl_beta": self.beta,
        }
