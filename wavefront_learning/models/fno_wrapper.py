"""FNO wrapper for wavefront learning.

Thin wrapper around neuralop's FNO that returns a dict output
compatible with the wavefront learning loss/metrics interface.

Input: tensor of shape (B, 3, nt, nx) from ToGridInputTransform
       channels are [ic_masked, t_coords, x_coords]
Output: dict {"output_grid": tensor of shape (B, 1, nt, nx)}
"""

import torch.nn as nn
from neuralop.models import FNO


class FNOWrapper(nn.Module):
    """Wraps neuralop FNO to return dict output."""

    def __init__(self, **fno_kwargs):
        super().__init__()
        self.fno = FNO(**fno_kwargs)

    def forward(self, x):
        out = self.fno(x)
        return {"output_grid": out}


def build_fno(args: dict) -> FNOWrapper:
    """Build FNO model from configuration dict.

    Args:
        args: Configuration dictionary. Supported keys:
            - n_modes_t: Fourier modes in time dimension (default: 16)
            - n_modes_x: Fourier modes in space dimension (default: 8)
            - hidden_channels: FNO hidden width (default: 16)
            - n_layers: Number of FNO layers (default: 2)
    """
    n_modes_t = args.get("n_modes_t", 16)
    n_modes_x = args.get("n_modes_x", 8)
    hidden_channels = args.get("hidden_channels", 16)
    n_layers = args.get("n_layers", 2)

    return FNOWrapper(
        n_modes=(n_modes_t, n_modes_x),
        hidden_channels=hidden_channels,
        in_channels=3,
        out_channels=1,
        n_layers=n_layers,
    )
