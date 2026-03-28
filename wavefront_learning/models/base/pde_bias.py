"""Generic PDE-aware per-segment attention bias.

Computes a physics-informed attention bias for each IC segment using
boundary speeds from a ``PDE`` instance.  The penalty accumulation
pattern is shared across all PDEs:

1. Compute similarity variable ``xi = (x - x_d) / (t + eps)`` at each
   interface.
2. Apply one-sided penalties: ``relu(xi - speed_right)`` for the left
   segment and ``relu(speed_left - xi)`` for the right segment.
3. Accumulate onto K segments with ``F.pad``.
4. Optionally apply collision-time damping (if the PDE supports it).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pde import PDE


class PDEBias(nn.Module):
    """Per-segment attention bias using PDE boundary speeds.

    Args:
        pde: ``PDE`` instance providing ``boundary_speeds`` and
            optionally ``collision_times``.
        use_damping: If True and the PDE provides collision times,
            multiply the bias by a sigmoid that fades it after the
            estimated collision time.
        initial_damping_sharpness: Initial sharpness of the sigmoid
            damping (learnable).
        eps: Small constant added to ``t`` to avoid division by zero.
    """

    def __init__(
        self,
        pde: PDE,
        use_damping: bool = True,
        initial_damping_sharpness: float = 5.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pde = pde
        self.use_damping = use_damping
        self.eps = eps
        if use_damping:
            self.damping_sharpness = nn.Parameter(
                torch.tensor(initial_damping_sharpness)
            )

    def forward(
        self,
        ic_data: dict[str, torch.Tensor],
        query_points: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-segment attention bias.

        Args:
            ic_data: Dictionary with at least ``xs`` (B, K+1),
                ``ks`` (B, K), ``pieces_mask`` (B, K).
            query_points: Tuple ``(t_coords, x_coords)`` each of shape
                ``(B, nt, nx)``.

        Returns:
            Bias tensor ``(B, nt, nx, K)``.  Zero means full attention;
            large negative values suppress attention.
        """
        xs = ic_data["xs"]  # (B, K+1)
        pieces_mask = ic_data["pieces_mask"]  # (B, K)
        t_coords, x_coords = query_points  # each (B, *spatial)

        K = pieces_mask.shape[1]

        # PDE-specific boundary speeds
        speed_right, speed_left = self.pde.boundary_speeds(ic_data)

        # Interface positions
        x_d = xs[:, 1:K]  # (B, K-1)

        # Expand to (B, 1, 1, K-1)
        x_d = x_d[:, None, None, :]
        speed_right = speed_right[:, None, None, :]
        speed_left = speed_left[:, None, None, :]

        t_exp = t_coords.unsqueeze(-1)  # (B, *spatial, 1)
        x_exp = x_coords.unsqueeze(-1)  # (B, *spatial, 1)

        # One-sided penalties in similarity variable
        xi = (x_exp - x_d) / (t_exp + self.eps)  # (B, *spatial, K-1)
        penalty_left_seg = torch.relu(xi - speed_right)
        penalty_right_seg = torch.relu(speed_left - xi)

        # Accumulate onto K segments
        bias = -(
            F.pad(penalty_left_seg, (0, 1))
            + F.pad(penalty_right_seg, (1, 0))
        )  # (B, *spatial, K)

        # Collision-time damping (optional)
        if self.use_damping:
            t_coll = self.pde.collision_times(ic_data)
            if t_coll is not None:
                t_coll = t_coll[:, None, None, :]  # (B, 1, 1, K)
                damping = torch.sigmoid(
                    self.damping_sharpness.abs() * (t_coll - t_exp)
                )
                bias = bias * damping

        # Mask padded segments
        mask = pieces_mask[:, None, None, :]  # (B, 1, 1, K)
        bias = bias * mask + (~mask.bool()).float() * (-1e9)

        return bias
