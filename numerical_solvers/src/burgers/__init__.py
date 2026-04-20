"""1D inviscid Burgers solver — functional API (PyTorch).

Solves the scalar conservation law:
    u_t + (u^2 / 2)_x = 0

Riemann flux (Godunov with transonic entropy fix) ported from
clawpack/riemann ``burgers_1D_py.py``.

Public API
----------
generate_one : solve a single Burgers problem from a given initial condition.
generate_n   : generate *n* samples with random piecewise-constant ICs.
"""

import warnings

import numpy as np
import torch
from tqdm import tqdm, trange

from .initial_conditions import (
    from_steps,
    random_piecewise,
    random_piecewise_batch,
    riemann,
)
from .physics import flux as physical_flux
from .physics import max_wave_speed
from .timestepper import solve, solve_batch


def generate_one(
    u0: torch.Tensor,
    *,
    dx: float,
    dt: float,
    nt: int,
    bc_type: str = "extrap",
    flux_type: str = "godunov",
    reconstruction: str = "weno5",
    max_value: float | None = None,
    cfl: float | None = None,
) -> dict[str, torch.Tensor | float | int | bool]:
    """Solve one 1D Burgers problem.

    Parameters
    ----------
    u0 : 1-D tensor (nx,) — initial conserved variable.
    dx : cell width.
    dt : output time interval (sub-stepping is adaptive).
    nt : number of output time steps.
    bc_type : "extrap", "periodic", or "wall".
    flux_type : "godunov" or "rusanov".
    reconstruction : "constant" or "weno5".
    max_value : if set, terminate early when |u| exceeds this threshold.
    cfl : CFL number; default 0.9 (weno5) or 0.5 (constant).

    Returns
    -------
    dict with keys:
        u (nt+1, nx), x (nx,), t (nt+1,), dx, dt, nt, valid.
    """
    nx = u0.shape[0]
    x = torch.arange(nx, device=u0.device, dtype=u0.dtype) * dx

    u_hist, valid = solve(
        u0,
        nx=nx,
        dx=dx,
        dt=dt,
        nt=nt,
        bc_type=bc_type,
        flux_type=flux_type,
        reconstruction=reconstruction,
        max_value=max_value,
        cfl=cfl,
    )

    t_arr = torch.arange(nt + 1, device=u0.device, dtype=u0.dtype) * dt

    return {
        "u": u_hist,
        "x": x,
        "t": t_arr,
        "dx": dx,
        "dt": dt,
        "nt": nt,
        "valid": valid,
    }


def generate_n(
    n: int,
    k: int,
    *,
    nx: int = 200,
    dx: float,
    dt: float,
    nt: int,
    bc_type: str = "extrap",
    flux_type: str = "godunov",
    reconstruction: str = "weno5",
    u_range: tuple[float, float] = (-2.0, 2.0),
    max_value: float = 100.0,
    seed: int | None = None,
    show_progress: bool = True,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
    cfl: float | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """Generate *n* samples with random k-piecewise-constant ICs.

    Returns
    -------
    dict with keys:
        u (n, nt+1, nx), x (nx,), t (nt+1,), dx, dt, nt,
        ic_xs (n, k+1), ic_u_ks (n, k).
    """
    device = torch.device(device)
    x = torch.arange(nx, device=device, dtype=torch.float64) * dx

    if reconstruction == "weno5" and k > nx // 8:
        reconstruction = "constant"

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    if batch_size <= 1:
        return _generate_n_sequential(
            n, k, x=x, nx=nx, dx=dx, dt=dt, nt=nt, bc_type=bc_type,
            flux_type=flux_type, reconstruction=reconstruction,
            u_range=u_range, max_value=max_value, rng=rng,
            show_progress=show_progress, device=device, cfl=cfl,
        )

    max_batch_retries = max(10, 3 * ((n + batch_size - 1) // batch_size))
    rejected = 0
    collected = 0

    u_all = torch.zeros(n, nt + 1, nx, device=device, dtype=torch.float64)
    ic_xs_list: list[list[float]] = []
    ic_u_ks_list: list[list[float]] = []

    pbar = tqdm(total=n, desc="Burgers samples", disable=not show_progress)
    attempts = 0

    while collected < n and attempts < max_batch_retries:
        remaining = n - collected
        bs = min(batch_size, remaining * 2)

        u0_batch, ic_params_list = random_piecewise_batch(
            x, k, bs, rng, u_range=u_range,
        )

        u_hist, valid = solve_batch(
            u0_batch,
            nx=nx,
            dx=dx,
            dt=dt,
            nt=nt,
            bc_type=bc_type,
            flux_type=flux_type,
            reconstruction=reconstruction,
            max_value=max_value,
            cfl=cfl,
        )

        good_idx = valid.nonzero(as_tuple=True)[0]
        n_good = len(good_idx)
        n_take = min(n_good, remaining)
        rejected += bs - n_good

        if n_take > 0:
            take_idx = good_idx[:n_take]
            u_all[collected : collected + n_take] = u_hist[take_idx]
            for i in take_idx.tolist():
                ic_xs_list.append(ic_params_list[i]["xs"])
                ic_u_ks_list.append(ic_params_list[i]["u_ks"])
            collected += n_take
            pbar.update(n_take)

        attempts += 1

    pbar.close()

    if collected < n:
        warnings.warn(
            f"Only {collected}/{n} valid samples after {max_batch_retries} "
            "batch retries. Remaining slots are zero-filled.",
            stacklevel=2,
        )
        for _ in range(n - collected):
            ic_xs_list.append([0.0] * (k + 1))
            ic_u_ks_list.append([0.0] * k)

    if rejected > 0:
        print(f"Rejected {rejected} NaN/Inf samples during generation.")

    t_arr = torch.arange(nt + 1, device=device, dtype=torch.float64) * dt

    return {
        "u": u_all,
        "x": x,
        "t": t_arr,
        "dx": dx,
        "dt": dt,
        "nt": nt,
        "ic_xs": np.array(ic_xs_list),
        "ic_u_ks": np.array(ic_u_ks_list),
    }


def _generate_n_sequential(
    n: int,
    k: int,
    *,
    x: torch.Tensor,
    nx: int,
    dx: float,
    dt: float,
    nt: int,
    bc_type: str,
    flux_type: str,
    reconstruction: str,
    u_range: tuple[float, float],
    max_value: float,
    rng: torch.Generator,
    show_progress: bool,
    device: torch.device,
    cfl: float | None = None,
) -> dict[str, torch.Tensor | float | int]:
    u_all = torch.zeros(n, nt + 1, nx, device=device, dtype=torch.float64)
    ic_xs_list: list[list[float]] = []
    ic_u_ks_list: list[list[float]] = []

    max_retries = 10
    rejected = 0

    it = trange(n, desc="Burgers samples", disable=not show_progress)
    for i in it:
        for _ in range(max_retries):
            u0, ic_params = random_piecewise(x, k, rng, u_range=u_range)
            result = generate_one(
                u0,
                dx=dx, dt=dt, nt=nt,
                bc_type=bc_type, flux_type=flux_type,
                reconstruction=reconstruction, max_value=max_value, cfl=cfl,
            )
            if result["valid"]:
                ic_xs_list.append(ic_params["xs"])
                ic_u_ks_list.append(ic_params["u_ks"])
                break
            rejected += 1
        else:
            warnings.warn(
                f"Sample {i}: all {max_retries} retries produced NaN/Inf, "
                "using zero-filled sample.",
                stacklevel=2,
            )
            result = {"u": torch.zeros(nt + 1, nx, dtype=torch.float64)}
            ic_xs_list.append([0.0] * (k + 1))
            ic_u_ks_list.append([0.0] * k)

        u_all[i] = result["u"]

    if rejected > 0:
        print(f"Rejected {rejected} NaN/Inf samples during generation.")

    t_arr = torch.arange(nt + 1, device=device, dtype=torch.float64) * dt

    return {
        "u": u_all,
        "x": x,
        "t": t_arr,
        "dx": dx,
        "dt": dt,
        "nt": nt,
        "ic_xs": np.array(ic_xs_list),
        "ic_u_ks": np.array(ic_u_ks_list),
    }


__all__ = [
    "generate_one",
    "generate_n",
    "solve",
    "solve_batch",
    "random_piecewise",
    "random_piecewise_batch",
    "from_steps",
    "riemann",
    "physical_flux",
    "max_wave_speed",
]
