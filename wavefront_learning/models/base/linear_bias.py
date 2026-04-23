"""Linear spatial-distance attention bias (ALiBi-style).

Drop-in replacement for :class:`PDEBias` that ignores the PDE physics
and penalises attention by raw ``|x_query - x_cell|``. Used as an
ablation: does *any* distance-decaying bias help, or is the Riemann-
problem structure doing the work?

Same ``forward(ic_data, query_points) -> (B, *spatial, K)`` contract as
``PDEBias`` so ``ARWaveNO`` can swap them without other code changes.
"""

import torch
import torch.nn as nn


class LinearBias(nn.Module):
    """Spatial-distance bias on attention logits.

    For each query coordinate ``(t, x)`` and cell token ``k`` (centre
    ``x_c``), computes

        bias(t, x, k) = -|scale| * |x - x_c|

    Independent of ``t``, independent of the PDE, independent of cell
    values. Large negative → suppress attention between far-apart
    query/key positions.

    Args:
        initial_scale: Initial value of the learnable scale. Bias grows
            linearly with distance, so values around 1–10 give a
            reasonable inductive prior for domains in ``[0, 1]``.
    """

    def __init__(self, initial_scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(
        self,
        ic_data: dict[str, torch.Tensor],
        query_points: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        xs = ic_data["xs"]  # (B, K+1)
        pieces_mask = ic_data["pieces_mask"]  # (B, K)
        _, x_coords = query_points  # (B, *spatial)

        # Cell centres.
        x_center = (xs[:, :-1] + xs[:, 1:]) / 2  # (B, K)

        # Match PDEBias broadcast pattern (assumes spatial shape (nt, nx)).
        x_center = x_center[:, None, None, :]  # (B, 1, 1, K)
        x_exp = x_coords.unsqueeze(-1)  # (B, *spatial, 1)

        dist = (x_exp - x_center).abs()  # (B, *spatial, K)
        bias = -self.scale.abs() * dist

        # Mask padded cells with large negative (matches PDEBias).
        mask = pieces_mask[:, None, None, :]
        bias = bias * mask + (~mask.bool()).float() * (-1e9)

        return bias
