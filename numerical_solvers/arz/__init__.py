"""ARZ traffic flow solver — functional API (PyTorch).

Public API
----------
generate_one : solve a single ARZ problem from given initial conditions.
generate_n   : generate *n* samples with random piecewise-constant ICs.
"""

import warnings
from collections.abc import Callable

import numpy as np
import torch
from tqdm import trange

from .initial_conditions import from_steps, random_piecewise, riemann, three_region
from .physics import dp_drho, eigenvalues, pressure
from .timestepper import solve


# ------------------------------------------------------------------ public
def generate_one(
    rho0: torch.Tensor,
    v0: torch.Tensor,
    *,
    dx: float,
    dt: float,
    nt: int,
    gamma: float = 1.0,
    bc_type: str = "zero_gradient",
    flux_type: str = "hll",
    reconstruction: str = "weno5",
    bc_left: tuple[float, float] | None = None,
    bc_right: tuple[float, float] | None = None,
    bc_left_time: Callable[[float], tuple[float, float]] | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """Solve one ARZ problem.

    Parameters
    ----------
    rho0 : 1-D tensor (nx,) — initial density.
    v0   : 1-D tensor (nx,) — initial *velocity* (not w).
    dx : cell width.
    dt : time step.
    nt : number of time steps.
    gamma : pressure exponent.
    bc_type : boundary condition type.
    flux_type : "rusanov" or "hll".
    reconstruction : "constant" or "weno5".
    bc_left, bc_right : static Dirichlet values (rho, v).
    bc_left_time : time-varying left BC callable.

    Returns
    -------
    dict with keys:
        rho (nt+1, nx), v (nt+1, nx), w (nt+1, nx),
        x (nx,), t (nt+1,), dx, dt, nt.
    """
    nx = rho0.shape[0]
    x = torch.arange(nx, device=rho0.device, dtype=rho0.dtype) * dx

    # v -> w -> rho_w  (conservative variable)
    w0 = v0 + pressure(rho0, gamma)
    rho_w0 = rho0 * w0

    rho_hist, w_hist, v_hist = solve(
        rho0,
        rho_w0,
        nx=nx,
        dx=dx,
        dt=dt,
        nt=nt,
        gamma=gamma,
        bc_type=bc_type,
        flux_type=flux_type,
        reconstruction=reconstruction,
        bc_left=bc_left,
        bc_right=bc_right,
        bc_left_time=bc_left_time,
    )

    t_arr = torch.arange(nt + 1, device=rho0.device, dtype=rho0.dtype) * dt

    return {
        "rho": rho_hist,
        "v": v_hist,
        "w": w_hist,
        "x": x,
        "t": t_arr,
        "dx": dx,
        "dt": dt,
        "nt": nt,
    }


def generate_n(
    n: int,
    k: int,
    *,
    nx: int = 200,
    dx: float,
    dt: float,
    nt: int,
    gamma: float = 1.0,
    bc_type: str = "zero_gradient",
    flux_type: str = "hll",
    reconstruction: str = "weno5",
    rho_range: tuple[float, float] = (0.1, 1.0),
    v_range: tuple[float, float] = (0.0, 1.0),
    seed: int | None = None,
    show_progress: bool = True,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | float | int]:
    """Generate *n* samples with random k-piecewise-constant ICs.

    Returns
    -------
    dict with keys:
        rho (n, nt+1, nx), v (n, nt+1, nx), w (n, nt+1, nx),
        x (nx,), t (nt+1,), dx, dt, nt.
    """
    device = torch.device(device)
    x = torch.arange(nx, device=device, dtype=torch.float32) * dx

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    rho_all = torch.zeros(n, nt + 1, nx, device=device)
    v_all = torch.zeros_like(rho_all)
    w_all = torch.zeros_like(rho_all)

    ic_xs_list: list[list[float]] = []
    ic_rho_ks_list: list[list[float]] = []
    ic_v_ks_list: list[list[float]] = []

    max_retries = 10
    rejected = 0

    it = trange(n, desc="ARZ samples", disable=not show_progress)
    for i in it:
        for attempt in range(max_retries):
            rho0, v0, ic_params = random_piecewise(
                x, k, rng, rho_range=rho_range, v_range=v_range
            )
            result = generate_one(
                rho0,
                v0,
                dx=dx,
                dt=dt,
                nt=nt,
                gamma=gamma,
                bc_type=bc_type,
                flux_type=flux_type,
                reconstruction=reconstruction,
            )
            if (
                torch.isfinite(result["rho"]).all()
                and torch.isfinite(result["v"]).all()
                and torch.isfinite(result["w"]).all()
            ):
                ic_xs_list.append(ic_params["xs"])
                ic_rho_ks_list.append(ic_params["rho_ks"])
                ic_v_ks_list.append(ic_params["v_ks"])
                break
            rejected += 1
        else:
            warnings.warn(
                f"Sample {i}: all {max_retries} retries produced NaN/Inf, "
                "using zero-filled sample.",
                stacklevel=2,
            )
            result = {"rho": torch.zeros(nt + 1, nx), "v": torch.zeros(nt + 1, nx), "w": torch.zeros(nt + 1, nx)}
            # Dummy IC params for zero-filled sample
            ic_xs_list.append([0.0] * (k + 1))
            ic_rho_ks_list.append([0.0] * k)
            ic_v_ks_list.append([0.0] * k)

        rho_all[i] = result["rho"]
        v_all[i] = result["v"]
        w_all[i] = result["w"]

    if rejected > 0:
        print(f"Rejected {rejected} NaN/Inf samples during generation.")

    t_arr = torch.arange(nt + 1, device=device, dtype=torch.float32) * dt

    return {
        "rho": rho_all,
        "v": v_all,
        "w": w_all,
        "x": x,
        "t": t_arr,
        "dx": dx,
        "dt": dt,
        "nt": nt,
        "ic_xs": np.array(ic_xs_list),
        "ic_rho_ks": np.array(ic_rho_ks_list),
        "ic_v_ks": np.array(ic_v_ks_list),
    }


__all__ = [
    "generate_one",
    "generate_n",
    "pressure",
    "dp_drho",
    "eigenvalues",
    "from_steps",
    "riemann",
    "three_region",
    "random_piecewise",
]
