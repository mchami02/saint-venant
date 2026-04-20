"""Per-direction WENO-5 reconstruction for 2D Euler.

Re-uses the equation-agnostic 1D WENO-5 kernel from ``src/euler/weno.py``
and wraps it with transpose helpers so it can be applied along either axis.
"""

import torch

from ..euler.weno import weno5_reconstruct


def weno5_x(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """WENO-5 reconstruction along the last axis (x-direction).

    Input shape: (..., ny, nx_ghost). The kernel works on the last axis,
    so we can call it directly.
    """
    return weno5_reconstruct(q)


def weno5_y(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """WENO-5 reconstruction along the second-to-last axis (y-direction)."""
    qT = q.transpose(-1, -2).contiguous()
    vmT, vpT = weno5_reconstruct(qT)
    return vmT.transpose(-1, -2).contiguous(), vpT.transpose(-1, -2).contiguous()
