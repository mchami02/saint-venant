"""Ghost cell boundary conditions for the ARZ system (PyTorch)."""

from collections.abc import Callable

import torch


def apply_ghost_cells(
    rho: torch.Tensor,
    rho_w: torch.Tensor,
    bc_type: str,
    t: float,
    *,
    n_ghost: int,
    gamma: float,
    bc_left: tuple[float, float] | None = None,
    bc_right: tuple[float, float] | None = None,
    bc_left_time: Callable[[float], tuple[float, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build extended state with *n_ghost* ghost cells on each side.

    Parameters
    ----------
    rho, rho_w : 1-D tensors of length nx.
    bc_type : one of "periodic", "inflow_outflow", "time_varying_inflow",
              "dirichlet", "zero_gradient".
    t : current simulation time (used for time-varying BCs).
    n_ghost : number of ghost cells per side (1 for constant, 4 for WENO-5).
    gamma : pressure exponent.
    bc_left, bc_right : (rho, v) pairs for Dirichlet / inflow BCs.
    bc_left_time : callable(t) -> (rho, v) for time-varying inflow.
    """
    device = rho.device
    dtype = rho.dtype
    ng = n_ghost

    def _dirichlet_ghost(
        rho_val: float, v_val: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_val = v_val + rho_val**gamma
        rw_val = rho_val * w_val
        rho_g = torch.full((ng,), rho_val, device=device, dtype=dtype)
        rw_g = torch.full((ng,), rw_val, device=device, dtype=dtype)
        return rho_g, rw_g

    if bc_type == "periodic":
        rho_g = torch.cat([rho[-ng:], rho, rho[:ng]])
        rho_w_g = torch.cat([rho_w[-ng:], rho_w, rho_w[:ng]])

    elif bc_type == "inflow_outflow":
        rho_l, v_l = bc_left if bc_left is not None else (0.5, 1.0)
        rho_gl, rw_gl = _dirichlet_ghost(rho_l, v_l)
        # outflow: zero-gradient on right
        rho_gr = rho[-1:].expand(ng)
        rw_gr = rho_w[-1:].expand(ng)
        rho_g = torch.cat([rho_gl, rho, rho_gr])
        rho_w_g = torch.cat([rw_gl, rho_w, rw_gr])

    elif bc_type == "time_varying_inflow":
        if bc_left_time is not None:
            rho_l_t, v_l_t = bc_left_time(t)
        else:
            rho_l, v_l = bc_left if bc_left is not None else (0.5, 1.0)
            rho_l_t = (
                rho_l
                + 2.0
                * torch.sin(torch.tensor(2.0 * 3.141592653589793 * t / 2.0)).item()
            )
            v_l_t = (
                v_l
                + 0.1
                * torch.sin(torch.tensor(2.0 * 3.141592653589793 * t / 1.5)).item()
            )
        rho_gl, rw_gl = _dirichlet_ghost(rho_l_t, v_l_t)
        rho_gr = rho[-1:].expand(ng)
        rw_gr = rho_w[-1:].expand(ng)
        rho_g = torch.cat([rho_gl, rho, rho_gr])
        rho_w_g = torch.cat([rw_gl, rho_w, rw_gr])

    elif bc_type == "dirichlet":
        rho_l, v_l = bc_left if bc_left is not None else (0.5, 1.0)
        rho_r, v_r = bc_right if bc_right is not None else (0.3, 0.5)
        rho_gl, rw_gl = _dirichlet_ghost(rho_l, v_l)
        rho_gr, rw_gr = _dirichlet_ghost(rho_r, v_r)
        rho_g = torch.cat([rho_gl, rho, rho_gr])
        rho_w_g = torch.cat([rw_gl, rho_w, rw_gr])

    else:
        # zero_gradient (default)
        rho_g = torch.cat([rho[:1].expand(ng), rho, rho[-1:].expand(ng)])
        rho_w_g = torch.cat([rho_w[:1].expand(ng), rho_w, rho_w[-1:].expand(ng)])

    return rho_g, rho_w_g
