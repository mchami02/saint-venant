"""Time integration and main solve loop for the ARZ system (PyTorch)."""

from collections.abc import Callable

import torch

from .boundary import apply_ghost_cells
from .flux import hll, rusanov
from .physics import pressure
from .weno import weno5_reconstruct


def solve(
    rho0: torch.Tensor,
    rho_w0: torch.Tensor,
    *,
    nx: int,
    dx: float,
    dt: float,
    nt: int,
    gamma: float,
    bc_type: str = "zero_gradient",
    flux_type: str = "hll",
    reconstruction: str = "weno5",
    bc_left: tuple[float, float] | None = None,
    bc_right: tuple[float, float] | None = None,
    bc_left_time: Callable[[float], tuple[float, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the ARZ solver.

    Parameters
    ----------
    rho0, rho_w0 : 1-D tensors of length *nx* — initial conservative state.
    nx, dx, dt, nt : grid / time parameters.
    gamma : pressure exponent.
    bc_type : boundary condition type.
    flux_type : "rusanov" or "hll".
    reconstruction : "constant" (1st order) or "weno5" (5th order).
    bc_left, bc_right : static Dirichlet values (rho, v).
    bc_left_time : time-varying left BC callable.

    Returns
    -------
    rho_hist, w_hist, v_hist : tensors of shape (nt+1, nx).
    """
    eps = 1e-12
    use_weno = reconstruction == "weno5"
    n_ghost = 4 if use_weno else 1

    flux_fn = hll if flux_type == "hll" else rusanov

    rho = rho0.clone()
    rho_w = rho_w0.clone()

    rho_hist = torch.zeros(nt + 1, nx, device=rho.device, dtype=rho.dtype)
    w_hist = torch.zeros_like(rho_hist)
    v_hist = torch.zeros_like(rho_hist)

    rho_hist[0] = rho
    w_hist[0] = rho_w / (rho + eps)
    v_hist[0] = w_hist[0] - pressure(rho, gamma)

    # ------------------------------------------------------------------ RHS
    def _compute_rhs(
        rho_loc: torch.Tensor, rho_w_loc: torch.Tensor, t_loc: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rho_g, rho_w_g = apply_ghost_cells(
            rho_loc,
            rho_w_loc,
            bc_type,
            t_loc,
            n_ghost=n_ghost,
            gamma=gamma,
            bc_left=bc_left,
            bc_right=bc_right,
            bc_left_time=bc_left_time,
        )

        if use_weno:
            rho_L, rho_R = weno5_reconstruct(rho_g)
            rho_w_L, rho_w_R = weno5_reconstruct(rho_w_g)
            rho_L = rho_L[: nx + 1]
            rho_R = rho_R[: nx + 1]
            rho_w_L = rho_w_L[: nx + 1]
            rho_w_R = rho_w_R[: nx + 1]
        else:
            # Constant reconstruction: left state = cell i, right = cell i+1
            rho_L = rho_g[: nx + 1]
            rho_R = rho_g[1 : nx + 2]
            rho_w_L = rho_w_g[: nx + 1]
            rho_w_R = rho_w_g[1 : nx + 2]

        f_rho, f_rw = flux_fn(rho_L, rho_w_L, rho_R, rho_w_R, gamma)

        drho = -(1.0 / dx) * (f_rho[1:] - f_rho[:-1])
        drho_w = -(1.0 / dx) * (f_rw[1:] - f_rw[:-1])
        return drho, drho_w

    # --------------------------------------------------------------- march
    for n in range(nt):
        t = n * dt

        if use_weno:
            # SSP-RK3
            k1_rho, k1_rw = _compute_rhs(rho, rho_w, t)
            rho_1 = (rho + dt * k1_rho).clamp(min=0.0)
            rho_w_1 = rho_w + dt * k1_rw

            k2_rho, k2_rw = _compute_rhs(rho_1, rho_w_1, t + dt)
            rho_2 = (0.75 * rho + 0.25 * (rho_1 + dt * k2_rho)).clamp(min=0.0)
            rho_w_2 = 0.75 * rho_w + 0.25 * (rho_w_1 + dt * k2_rw)

            k3_rho, k3_rw = _compute_rhs(rho_2, rho_w_2, t + 0.5 * dt)
            rho = ((1 / 3) * rho + (2 / 3) * (rho_2 + dt * k3_rho)).clamp(min=0.0)
            rho_w = (1 / 3) * rho_w + (2 / 3) * (rho_w_2 + dt * k3_rw)
        else:
            # Forward Euler
            k1_rho, k1_rw = _compute_rhs(rho, rho_w, t)
            rho = (rho + dt * k1_rho).clamp(min=0.0)
            rho_w = rho_w + dt * k1_rw

        w = rho_w / (rho + eps)
        v = w - pressure(rho, gamma)

        rho_hist[n + 1] = rho
        w_hist[n + 1] = w
        v_hist[n + 1] = v

    return rho_hist, w_hist, v_hist
