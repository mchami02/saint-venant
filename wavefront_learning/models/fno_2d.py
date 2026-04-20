"""FNO2D wrapper for 2D-spatial wavefront learning.

Thin wrapper around neuralop's FNO with 3D spectral modes over
``(nt, ny, nx)``.  For Euler2D we have 4 primitive channels
``[rho, u, v, p]`` — the model runs in-place on primitives.

The IC is reconstructed from the tile representation (``xs``, ``ys``,
``ks``, ``ks_u``, ``ks_v``, ``ks_p``) inside :meth:`forward` so no
2D-specific input transform is needed.
"""

import torch
import torch.nn as nn
from neuralop.models import FNO


def _reconstruct_ic_grid_2d(
    xs: torch.Tensor,       # (B, Kx+1)
    ys: torch.Tensor,       # (B, Ky+1)
    ks_list: list[torch.Tensor],  # each (B, Kx, Ky)
    pieces_mask: torch.Tensor,    # (B, Kx, Ky)
    x_coords: torch.Tensor, # (B, nt, ny, nx) cell centres; take t=0 slice
    y_coords: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the 2D IC (B, C, ny, nx) from tile params.

    Returns the IC as a stack of ``C = len(ks_list)`` channels.
    """
    B = xs.shape[0]
    Kx = ks_list[0].shape[1]
    Ky = ks_list[0].shape[2]
    # Take x/y coords at t=0 (any time slice is fine since coords are static)
    x = x_coords[:, 0]  # (B, ny, nx)
    y = y_coords[:, 0]  # (B, ny, nx)

    # Tile indices: for each grid cell find which (kx, ky) tile it lies in.
    # breaks_x interior = xs[:, 1:-1]  shape (B, Kx - 1)
    interior_x = xs[:, 1:-1] if Kx > 1 else xs[:, :0]
    interior_y = ys[:, 1:-1] if Ky > 1 else ys[:, :0]

    # bucketize needs per-batch work; do a loop — this only runs on the IC
    # reconstruction path so the loop is not in a hot kernel.
    ix_all = []
    iy_all = []
    for b in range(B):
        ix_all.append(torch.bucketize(x[b].contiguous(), interior_x[b]))
        iy_all.append(torch.bucketize(y[b].contiguous(), interior_y[b]))
    ix = torch.stack(ix_all, dim=0)  # (B, ny, nx), values in [0, Kx-1]
    iy = torch.stack(iy_all, dim=0)

    # Gather per-channel: ks[b, ix, iy]
    channels = []
    for ks in ks_list:
        # ks: (B, Kx, Ky) — flatten Kx*Ky and linear-index
        flat_idx = ix * Ky + iy  # (B, ny, nx)
        ks_flat = ks.reshape(B, Kx * Ky)
        ch = torch.gather(
            ks_flat, 1, flat_idx.reshape(B, -1)
        ).reshape(B, x.shape[1], x.shape[2])
        channels.append(ch)
    return torch.stack(channels, dim=1)  # (B, C, ny, nx)


class FNO2DWrapper(nn.Module):
    """3D-spectral FNO (time × y × x) for 2D Euler."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        n_modes: tuple[int, int, int] = (8, 8, 8),
        hidden_channels: int = 16,
        n_layers: int = 2,
    ):
        super().__init__()
        self.fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, batch_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        xs = batch_input["xs"]
        ys = batch_input["ys"]
        ks = batch_input["ks"]
        ks_u = batch_input["ks_u"]
        ks_v = batch_input["ks_v"]
        ks_p = batch_input["ks_p"]
        pieces_mask = batch_input["pieces_mask"]
        t_coords = batch_input["t_coords"]  # (B, nt, ny, nx)
        x_coords = batch_input["x_coords"]
        y_coords = batch_input["y_coords"]

        B, nt, ny, nx = t_coords.shape

        ic = _reconstruct_ic_grid_2d(
            xs, ys, [ks, ks_u, ks_v, ks_p], pieces_mask,
            x_coords, y_coords,
        )  # (B, 4, ny, nx)
        # Broadcast IC over time to get (B, 4, nt, ny, nx)
        ic_expanded = ic.unsqueeze(2).expand(B, self.in_channels, nt, ny, nx)
        out = self.fno(ic_expanded)  # (B, 4, nt, ny, nx)
        return {"output_grid": out}


def build_fno_2d(args: dict) -> FNO2DWrapper:
    """Build FNO2D matching the equation's channel count (4 for Euler2D)."""
    # Channel count derived from equation
    eq = args.get("equation", "Euler2D")
    if eq == "Euler2D":
        n_channels = 4
    else:
        n_channels = 4  # default for any 2D equation

    return FNO2DWrapper(
        in_channels=n_channels,
        out_channels=n_channels,
        n_modes=(
            args.get("n_modes_t", 8),
            args.get("n_modes_y", 8),
            args.get("n_modes_x", 8),
        ),
        hidden_channels=args.get("hidden_channels", 16),
        n_layers=args.get("n_layers", 2),
    )
