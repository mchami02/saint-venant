"""Cell-average MSE loss for finite-volume-consistent training.

Reshapes model output from (B, 1, nt, nx*k) to (B, 1, nt, nx, k),
averages over the k query points per cell, then computes MSE against
the cell-averaged ground truth from the FV solver.

Falls back to standard MSE if cell_sampling_k is not in input_dict.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss


class CellAverageMSELoss(BaseLoss):
    """MSE loss with cell averaging for FV-consistent training.

    When CellSamplingTransform is active, the model predicts at nx*k
    query points per timestep. This loss reshapes the predictions back
    to (B, 1, nt, nx, k), averages over k, and computes MSE against
    the (B, 1, nt, nx) target from the FV solver.

    If cell_sampling_k is not present in input_dict (e.g. when the
    transform is not used), falls back to standard MSE.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        output_grid = output_dict["output_grid"]

        if "cell_sampling_k" in input_dict:
            k_tensor = input_dict["cell_sampling_k"]
            nx_tensor = input_dict["original_nx"]
            # After collation these may be (B,) tensors; all values are identical
            k = k_tensor.item() if k_tensor.dim() == 0 else k_tensor[0].item()
            nx = nx_tensor.item() if nx_tensor.dim() == 0 else nx_tensor[0].item()
            B, C, nt_exp, nxk = output_grid.shape

            if "original_nt" in input_dict:
                # 2D refinement: (B, C, nt*k_t, nx*k_x) → (B, C, nt, nx)
                nt_tensor = input_dict["original_nt"]
                nt = nt_tensor.item() if nt_tensor.dim() == 0 else nt_tensor[0].item()
                k_t = nt_exp // nt
                k_x = nxk // nx
                output_grid = (
                    output_grid.reshape(B, C, nt, k_t, nx, k_x)
                    .mean(dim=-1)
                    .mean(dim=-2)
                )
            else:
                # Spatial-only: (B, C, nt, nx*k) → (B, C, nt, nx)
                output_grid = output_grid.reshape(B, C, nt_exp, nx, k).mean(dim=-1)

        loss = F.mse_loss(output_grid, target)
        return loss, {"cell_avg_mse": loss.item()}
