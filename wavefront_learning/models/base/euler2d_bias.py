"""2D Euler per-tile attention bias.

Analogue of :class:`~models.base.pde_bias.PDEBias` extended to 2D.  The
IC grid is piecewise-constant over a Cartesian ``(Kx, Ky)`` tile grid
with shared ``xs`` / ``ys`` breakpoints.  Each x-interface is a 1D
Riemann problem in ``(rho, u, p)`` along the normal direction, and each
y-interface is a 1D Riemann problem in ``(rho, v, p)`` along the normal
direction — solved by :meth:`Euler2DPDE.boundary_speeds_x` /
``boundary_speeds_y``.

Bias per tile ``(kx, ky)`` sums contributions from its four adjacent
interfaces (left/right x, top/bottom y), each applied in the relevant
self-similar coordinate ``xi = (r - r_d) / (t + eps)`` where ``r`` is
``x`` or ``y``.  Output is flattened to ``(B, nt, ny, nx, Kx*Ky)`` so
downstream attention treats tiles as a flat segment list.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pde import Euler2DPDE


class Euler2DBias(nn.Module):
    """Per-tile 2D attention bias using :class:`Euler2DPDE` wave speeds.

    Args:
        pde: An ``Euler2DPDE`` instance (built with the right gamma).
        eps: Small denominator regulariser in the similarity variable.
    """

    def __init__(self, pde: Euler2DPDE | None = None, eps: float = 1e-6):
        super().__init__()
        self.pde = pde if pde is not None else Euler2DPDE()
        self.eps = eps

    def forward(
        self,
        ic_data: dict[str, torch.Tensor],
        query_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the per-tile 2D attention bias.

        Args:
            ic_data: Dictionary with:
                - ``xs``: ``(B, Kx + 1)`` x-breakpoints.
                - ``ys``: ``(B, Ky + 1)`` y-breakpoints.
                - ``ks``: ``(B, Kx, Ky)`` tile densities.
                - ``ks_u``: ``(B, Kx, Ky)`` tile x-velocities.
                - ``ks_v``: ``(B, Kx, Ky)`` tile y-velocities.
                - ``ks_p``: ``(B, Kx, Ky)`` tile pressures.
                - ``pieces_mask``: ``(B, Kx, Ky)`` validity mask.
            query_points: ``(t_coords, x_coords, y_coords)`` each
                ``(B, nt, ny, nx)``.

        Returns:
            Bias ``(B, nt, ny, nx, Kx * Ky)``.  Zero = full attention.
        """
        xs = ic_data["xs"]      # (B, Kx + 1)
        ys = ic_data["ys"]      # (B, Ky + 1)
        pieces_mask = ic_data["pieces_mask"]  # (B, Kx, Ky)
        t_coords, x_coords, y_coords = query_points

        B, Kx, Ky = pieces_mask.shape
        # ---- x-interface contributions ----
        # speed_*_x: (B, Kx - 1, Ky)
        speed_right_x, speed_left_x = self.pde.boundary_speeds_x(ic_data)

        # Interior x-breakpoints: (B, Kx - 1)
        x_d = xs[:, 1:Kx]

        # Broadcast to (B, nt, ny, nx, Kx - 1, Ky)
        x_d_b = x_d[:, None, None, None, :, None]
        speed_right_x_b = speed_right_x[:, None, None, None, :, :]
        speed_left_x_b = speed_left_x[:, None, None, None, :, :]
        t_exp = t_coords[..., None, None]
        x_exp = x_coords[..., None, None]

        xi_x = (x_exp - x_d_b) / (t_exp + self.eps)
        penalty_left_tile_x = torch.relu(xi_x - speed_right_x_b)
        penalty_right_tile_x = torch.relu(speed_left_x_b - xi_x)

        # Pad along Kx axis: left-seg penalty targets tiles 0..Kx-2
        # (pad right by 1); right-seg penalty targets tiles 1..Kx-1
        # (pad left by 1).  The F.pad order is (last, ..., first).
        # Shape: (B, nt, ny, nx, Kx, Ky).
        bias_x = -(
            F.pad(penalty_left_tile_x, (0, 0, 0, 1))
            + F.pad(penalty_right_tile_x, (0, 0, 1, 0))
        )

        # ---- y-interface contributions ----
        speed_right_y, speed_left_y = self.pde.boundary_speeds_y(ic_data)
        y_d = ys[:, 1:Ky]  # (B, Ky - 1)

        y_d_b = y_d[:, None, None, None, None, :]
        speed_right_y_b = speed_right_y[:, None, None, None, :, :]
        speed_left_y_b = speed_left_y[:, None, None, None, :, :]
        y_exp = y_coords[..., None, None]

        xi_y = (y_exp - y_d_b) / (t_exp + self.eps)
        penalty_left_tile_y = torch.relu(xi_y - speed_right_y_b)
        penalty_right_tile_y = torch.relu(speed_left_y_b - xi_y)

        bias_y = -(
            F.pad(penalty_left_tile_y, (0, 1))
            + F.pad(penalty_right_tile_y, (1, 0))
        )

        bias = bias_x + bias_y  # (B, nt, ny, nx, Kx, Ky)

        # Flatten tile grid to a linear segment dimension.
        B_, nt, ny, nx, Kx_, Ky_ = bias.shape
        bias = bias.reshape(B_, nt, ny, nx, Kx_ * Ky_)

        # Mask padded tiles.
        mask_flat = pieces_mask.reshape(B, Kx * Ky)
        mask_b = mask_flat[:, None, None, None, :]
        bias = bias * mask_b + (~mask_b.bool()).float() * (-1e9)
        return bias
