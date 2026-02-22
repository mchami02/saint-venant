"""2D LWR traffic flow solver — functional API (PyTorch).

Public API
----------
generate_one : solve a single 2D LWR problem from given initial conditions.
generate_n   : generate *n* samples with random piecewise-constant ICs.
"""

import torch
from tqdm import trange

from .initial_conditions import (
    four_quadrant,
    gaussian_bump,
    random_piecewise,
    riemann_x,
    riemann_y,
)
from .physics import cfl_dt, greenshields_flux, triangular_flux
from .timestepper import solve


# ------------------------------------------------------------------ public
def generate_one(
    rho0: torch.Tensor,
    *,
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
) -> dict[str, torch.Tensor | float | int]:
    """Solve one 2D LWR problem.

    Parameters
    ----------
    rho0 : 2-D tensor (ny, nx) — initial density.
    dx, dy : cell widths in x and y.
    dt : time step.
    nt : number of time steps.
    flux_type : "greenshields" or "triangular".
    bc_type : "zero_gradient", "periodic", or "dirichlet".
    v_max_x, v_max_y : max speeds in x and y (Greenshields).
    rho_max : maximum density.
    bc_value : constant for Dirichlet BCs.
    v_f_x, w_x, v_f_y, w_y : triangular flux parameters per direction.

    Returns
    -------
    dict with keys:
        rho (nt+1, ny, nx), x (nx,), y (ny,), t (nt+1,),
        dx, dy, dt, nt.
    """
    ny, nx = rho0.shape
    x = torch.arange(nx, device=rho0.device, dtype=rho0.dtype) * dx
    y = torch.arange(ny, device=rho0.device, dtype=rho0.dtype) * dy

    rho_hist = solve(
        rho0,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        dt=dt,
        nt=nt,
        flux_type=flux_type,
        bc_type=bc_type,
        v_max_x=v_max_x,
        v_max_y=v_max_y,
        rho_max=rho_max,
        bc_value=bc_value,
        v_f_x=v_f_x,
        w_x=w_x,
        v_f_y=v_f_y,
        w_y=w_y,
    )

    t_arr = torch.arange(nt + 1, device=rho0.device, dtype=rho0.dtype) * dt

    return {
        "rho": rho_hist,
        "x": x,
        "y": y,
        "t": t_arr,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "nt": nt,
    }


def generate_n(
    n: int,
    kx: int,
    ky: int,
    *,
    nx: int = 50,
    ny: int = 50,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    flux_type: str = "greenshields",
    bc_type: str = "zero_gradient",
    v_max_x: float = 1.0,
    v_max_y: float = 1.0,
    rho_max: float = 1.0,
    rho_range: tuple[float, float] = (0.1, 0.9),
    seed: int | None = None,
    show_progress: bool = True,
    device: torch.device | str = "cpu",
    bc_value: float | None = None,
    v_f_x: float | None = None,
    w_x: float | None = None,
    v_f_y: float | None = None,
    w_y: float | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """Generate *n* samples with random kx*ky piecewise-constant ICs.

    Returns
    -------
    dict with keys:
        rho (n, nt+1, ny, nx), x (nx,), y (ny,), t (nt+1,),
        dx, dy, dt, nt.
    """
    device = torch.device(device)
    x = torch.arange(nx, device=device, dtype=torch.float32) * dx
    y = torch.arange(ny, device=device, dtype=torch.float32) * dy

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    rho_all = torch.zeros(n, nt + 1, ny, nx, device=device)

    it = trange(n, desc="LWR2D samples", disable=not show_progress)
    for i in it:
        rho0 = random_piecewise(x, y, kx, ky, rng, rho_range=rho_range)
        result = generate_one(
            rho0,
            dx=dx,
            dy=dy,
            dt=dt,
            nt=nt,
            flux_type=flux_type,
            bc_type=bc_type,
            v_max_x=v_max_x,
            v_max_y=v_max_y,
            rho_max=rho_max,
            bc_value=bc_value,
            v_f_x=v_f_x,
            w_x=w_x,
            v_f_y=v_f_y,
            w_y=w_y,
        )
        rho_all[i] = result["rho"]

    t_arr = torch.arange(nt + 1, device=device, dtype=torch.float32) * dt

    return {
        "rho": rho_all,
        "x": x,
        "y": y,
        "t": t_arr,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "nt": nt,
    }


__all__ = [
    "generate_one",
    "generate_n",
    "greenshields_flux",
    "triangular_flux",
    "cfl_dt",
    "riemann_x",
    "riemann_y",
    "four_quadrant",
    "gaussian_bump",
    "random_piecewise",
]
