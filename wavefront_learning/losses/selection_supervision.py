"""Selection supervision loss for CharNO.

Provides direct supervision of CharNO's segment selection weights
using soft targets derived from ground truth density values.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss


class SelectionSupervisionLoss(BaseLoss):
    """Direct supervision of CharNO selection weights.

    Computes KL divergence between predicted selection weights and soft
    targets derived from the ground truth. The soft target assigns high
    weight to segments whose value rho_k is close to the ground truth
    density at each (t, x) point.

    Soft target: w_k ~ exp(-|rho_gt - rho_k|^2 / sigma^2)
    Loss: KL(target || predicted)

    Args:
        sigma: Width of the Gaussian kernel for soft target construction.
            Smaller sigma → sharper targets (closer to one-hot).
    """

    def __init__(self, sigma: float = 0.05):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if "selection_weights" not in output_dict:
            return torch.tensor(0.0, device=target.device), {
                "selection_supervision": 0.0,
            }

        weights = output_dict["selection_weights"]  # (B, nt, nx, K)
        ks = input_dict["ks"]  # (B, K)
        pieces_mask = input_dict["pieces_mask"]  # (B, K)

        # Handle target shape: (B, 1, nt, nx) → (B, nt, nx)
        target_grid = target.squeeze(1) if target.dim() == 4 else target

        # Soft targets from GT
        rho_gt = target_grid.unsqueeze(-1)  # (B, nt, nx, 1)
        rho_k = ks.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, K)
        log_targets = -(rho_gt - rho_k).pow(2) / (self.sigma**2)

        # Mask invalid segments
        mask = pieces_mask.unsqueeze(1).unsqueeze(1).bool()  # (B, 1, 1, K)
        log_targets = log_targets.masked_fill(~mask, -1e9)
        target_weights = F.softmax(log_targets, dim=-1)  # (B, nt, nx, K)

        # KL divergence: KL(target || predicted)
        eps = 1e-8
        kl = (
            target_weights
            * (torch.log(target_weights + eps) - torch.log(weights + eps))
        ).sum(dim=-1)
        loss = kl.mean()

        return loss, {"selection_supervision": loss.item()}
