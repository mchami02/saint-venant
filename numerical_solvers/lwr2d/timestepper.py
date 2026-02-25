"""Time integration and main solve loop for 2D LWR (PyTorch).

Unsplit first-order Godunov finite volume with Forward Euler time stepping.
"""

import torch

from .boundary import apply_ghost_cells_2d
from .flux import godunov_flux_greenshields, godunov_flux_triangular


def solve(
    rho0: torch.Tensor,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    flux_type: str = "greenshields",
    bc_type: str = "zero_gradient",
    v_max_x: float = 1.0,
    v_max_y: float = 1.0,
    rho_max: float = 1.0,
    bc_value: float | None = None,
    v_f_x: float | None = None,
    w_x: float | None = None,
    v_f_y: float | None = None,
    w_y: float | None = None,
) -> torch.Tensor:
    """Run the 2D LWR solver.

    Parameters
    ----------
    rho0 : 2-D tensor (ny, nx) — initial density.
    nx, ny : number of cells in x and y.
    dx, dy : cell widths.
    dt : time step.
    nt : number of time steps.
    flux_type : "greenshields" or "triangular".
    bc_type : "zero_gradient", "periodic", or "dirichlet".
    v_max_x, v_max_y : max speeds in x and y (Greenshields).
    rho_max : maximum density.
    bc_value : constant value for Dirichlet BCs.
    v_f_x, w_x : free-flow speed and jam wave speed in x (triangular).
    v_f_y, w_y : free-flow speed and jam wave speed in y (triangular).

    Returns
    -------
    rho_hist : tensor of shape (nt+1, ny, nx).
    """
    rho = rho0.clone()
    rho_hist = torch.zeros(nt + 1, ny, nx, device=rho.device, dtype=rho.dtype)
    rho_hist[0] = rho

    # Select flux functions per direction
    if flux_type == "greenshields":

        def flux_x(rL, rR):
            return godunov_flux_greenshields(rL, rR, v_max_x, rho_max)

        def flux_y(rL, rR):
            return godunov_flux_greenshields(rL, rR, v_max_y, rho_max)

    elif flux_type == "triangular":
        _vfx = v_f_x if v_f_x is not None else v_max_x
        _wx = w_x if w_x is not None else v_max_x
        _vfy = v_f_y if v_f_y is not None else v_max_y
        _wy = w_y if w_y is not None else v_max_y

        def flux_x(rL, rR):
            return godunov_flux_triangular(rL, rR, _vfx, _wx, rho_max)

        def flux_y(rL, rR):
            return godunov_flux_triangular(rL, rR, _vfy, _wy, rho_max)

    else:
        raise ValueError(f"Unknown flux_type: {flux_type!r}")

    dt_dx = dt / dx
    dt_dy = dt / dy

    for n in range(nt):
        # 1. Apply ghost cells -> (ny+2, nx+2)
        rho_g = apply_ghost_cells_2d(rho, bc_type, n_ghost=1, bc_value=bc_value)

        # 2. X-direction fluxes: nx+1 interfaces for each of ny rows
        #    Left states:  rho_g[:, 0:nx+1]  interior rows only
        #    Right states: rho_g[:, 1:nx+2]
        rho_xL = rho_g[1:-1, :-1]  # (ny, nx+1)
        rho_xR = rho_g[1:-1, 1:]  # (ny, nx+1)
        F = flux_x(rho_xL, rho_xR)  # (ny, nx+1)

        # 3. Y-direction fluxes: ny+1 interfaces for each of nx columns
        rho_yL = rho_g[:-1, 1:-1]  # (ny+1, nx)
        rho_yR = rho_g[1:, 1:-1]  # (ny+1, nx)
        G = flux_y(rho_yL, rho_yR)  # (ny+1, nx)

        # 4. Conservative update
        rho = rho - dt_dx * (F[:, 1:] - F[:, :-1]) - dt_dy * (G[1:, :] - G[:-1, :])

        # 5. Clamp to physical bounds
        rho = rho.clamp(0.0, rho_max)

        rho_hist[n + 1] = rho

    return rho_hist
