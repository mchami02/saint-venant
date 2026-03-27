"""1D Euler equations solver — functional API (PyTorch).

Solves the compressible Euler equations:
    rho_t + (rho*u)_x = 0
    (rho*u)_t + (rho*u^2 + p)_x = 0
    E_t + (u*(E + p))_x = 0

with ideal gas EOS: p = (gamma - 1) * (E - 0.5 * rho * u^2).

Riemann solvers ported from clawpack/riemann euler_1D_py.py.

Public API
----------
generate_one : solve a single Euler problem from given initial conditions.
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
    sod,
)
from .physics import (
    conservative_to_primitive,
    pressure_from_conservative,
    primitive_to_conservative,
    sound_speed,
)
from .timestepper import solve, solve_batch


# ------------------------------------------------------------------ public
def generate_one(
    rho0: torch.Tensor,
    u0: torch.Tensor,
    p0: torch.Tensor,
    *,
    dx: float,
    dt: float,
    nt: int,
    gamma: float = 1.4,
    bc_type: str = "extrap",
    flux_type: str = "hllc",
    reconstruction: str = "weno5",
    max_value: float | None = None,
) -> dict[str, torch.Tensor | float | int | bool]:
    """Solve one 1D Euler problem.

    Parameters
    ----------
    rho0 : 1-D tensor (nx,) — initial density.
    u0   : 1-D tensor (nx,) — initial velocity.
    p0   : 1-D tensor (nx,) — initial pressure.
    dx : cell width.
    dt : time step.
    nt : number of time steps.
    gamma : ratio of specific heats.
    bc_type : "extrap", "periodic", or "wall".
    flux_type : "hllc", "hll", or "rusanov".
    reconstruction : "constant" or "weno5".
    max_value : if set, terminate early when any value exceeds this threshold.

    Returns
    -------
    dict with keys:
        rho (nt+1, nx), u (nt+1, nx), p (nt+1, nx),
        x (nx,), t (nt+1,), dx, dt, nt, valid (bool).
    """
    nx = rho0.shape[0]
    x = torch.arange(nx, device=rho0.device, dtype=rho0.dtype) * dx

    # Primitive -> conservative
    _, rho_u0, E0 = primitive_to_conservative(rho0, u0, p0, gamma)

    rho_hist, u_hist, p_hist, valid = solve(
        rho0,
        rho_u0,
        E0,
        nx=nx,
        dx=dx,
        dt=dt,
        nt=nt,
        gamma=gamma,
        bc_type=bc_type,
        flux_type=flux_type,
        reconstruction=reconstruction,
        max_value=max_value,
    )

    t_arr = torch.arange(nt + 1, device=rho0.device, dtype=rho0.dtype) * dt

    return {
        "rho": rho_hist,
        "u": u_hist,
        "p": p_hist,
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
    gamma: float = 1.4,
    bc_type: str = "extrap",
    flux_type: str = "hllc",
    reconstruction: str = "weno5",
    rho_range: tuple[float, float] = (0.1, 2.0),
    u_range: tuple[float, float] = (-2.0, 2.0),
    p_range: tuple[float, float] = (0.1, 5.0),
    max_value: float = 100.0,
    seed: int | None = None,
    show_progress: bool = True,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
) -> dict[str, torch.Tensor | float | int]:
    """Generate *n* samples with random k-piecewise-constant ICs.

    Parameters
    ----------
    batch_size : number of samples solved simultaneously per batch.
        Set to 1 to fall back to sequential generation.

    Returns
    -------
    dict with keys:
        rho (n, nt+1, nx), u (n, nt+1, nx), p (n, nt+1, nx),
        x (nx,), t (nt+1,), dx, dt, nt,
        ic_xs (n, k+1), ic_rho_ks (n, k), ic_u_ks (n, k), ic_p_ks (n, k).
    """
    device = torch.device(device)
    x = torch.arange(nx, device=device, dtype=torch.float64) * dx

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    # Sequential fallback
    if batch_size <= 1:
        return _generate_n_sequential(
            n, k, x=x, nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
            bc_type=bc_type, flux_type=flux_type,
            reconstruction=reconstruction, rho_range=rho_range,
            u_range=u_range, p_range=p_range, max_value=max_value,
            rng=rng, show_progress=show_progress, device=device,
        )

    # Batched generation
    max_batch_retries = 10
    rejected = 0
    collected = 0

    rho_all = torch.zeros(n, nt + 1, nx, device=device, dtype=torch.float64)
    u_all = torch.zeros_like(rho_all)
    p_all = torch.zeros_like(rho_all)
    ic_xs_list: list[list[float]] = []
    ic_rho_ks_list: list[list[float]] = []
    ic_u_ks_list: list[list[float]] = []
    ic_p_ks_list: list[list[float]] = []

    pbar = tqdm(total=n, desc="Euler samples", disable=not show_progress)
    attempts = 0

    while collected < n and attempts < max_batch_retries:
        remaining = n - collected
        bs = min(batch_size, remaining * 2)

        rho0_batch, u0_batch, p0_batch, ic_params_list = random_piecewise_batch(
            x, k, bs, rng,
            rho_range=rho_range, u_range=u_range, p_range=p_range,
        )

        _, rho_u0_batch, E0_batch = primitive_to_conservative(
            rho0_batch, u0_batch, p0_batch, gamma
        )

        rho_hist, u_hist, p_hist, valid = solve_batch(
            rho0_batch,
            rho_u0_batch,
            E0_batch,
            nx=nx,
            dx=dx,
            dt=dt,
            nt=nt,
            gamma=gamma,
            bc_type=bc_type,
            flux_type=flux_type,
            reconstruction=reconstruction,
            max_value=max_value,
        )

        good_idx = valid.nonzero(as_tuple=True)[0]
        n_good = len(good_idx)
        n_take = min(n_good, remaining)
        rejected += bs - n_good

        if n_take > 0:
            take_idx = good_idx[:n_take]
            rho_all[collected : collected + n_take] = rho_hist[take_idx]
            u_all[collected : collected + n_take] = u_hist[take_idx]
            p_all[collected : collected + n_take] = p_hist[take_idx]
            for i in take_idx.tolist():
                ic_xs_list.append(ic_params_list[i]["xs"])
                ic_rho_ks_list.append(ic_params_list[i]["rho_ks"])
                ic_u_ks_list.append(ic_params_list[i]["u_ks"])
                ic_p_ks_list.append(ic_params_list[i]["p_ks"])
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
            ic_rho_ks_list.append([0.0] * k)
            ic_u_ks_list.append([0.0] * k)
            ic_p_ks_list.append([0.0] * k)

    if rejected > 0:
        print(f"Rejected {rejected} NaN/Inf samples during generation.")

    t_arr = torch.arange(nt + 1, device=device, dtype=torch.float64) * dt

    return {
        "rho": rho_all,
        "u": u_all,
        "p": p_all,
        "x": x,
        "t": t_arr,
        "dx": dx,
        "dt": dt,
        "nt": nt,
        "ic_xs": np.array(ic_xs_list),
        "ic_rho_ks": np.array(ic_rho_ks_list),
        "ic_u_ks": np.array(ic_u_ks_list),
        "ic_p_ks": np.array(ic_p_ks_list),
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
    gamma: float,
    bc_type: str,
    flux_type: str,
    reconstruction: str,
    rho_range: tuple[float, float],
    u_range: tuple[float, float],
    p_range: tuple[float, float],
    max_value: float,
    rng: torch.Generator,
    show_progress: bool,
    device: torch.device,
) -> dict[str, torch.Tensor | float | int]:
    """Original sequential generation (used when batch_size <= 1)."""
    rho_all = torch.zeros(n, nt + 1, nx, device=device, dtype=torch.float64)
    u_all = torch.zeros_like(rho_all)
    p_all = torch.zeros_like(rho_all)

    ic_xs_list: list[list[float]] = []
    ic_rho_ks_list: list[list[float]] = []
    ic_u_ks_list: list[list[float]] = []
    ic_p_ks_list: list[list[float]] = []

    max_retries = 10
    rejected = 0

    it = trange(n, desc="Euler samples", disable=not show_progress)
    for i in it:
        for attempt in range(max_retries):
            rho0, u0, p0, ic_params = random_piecewise(
                x, k, rng,
                rho_range=rho_range, u_range=u_range, p_range=p_range,
            )
            result = generate_one(
                rho0, u0, p0,
                dx=dx, dt=dt, nt=nt, gamma=gamma,
                bc_type=bc_type, flux_type=flux_type,
                reconstruction=reconstruction, max_value=max_value,
            )
            if result["valid"]:
                ic_xs_list.append(ic_params["xs"])
                ic_rho_ks_list.append(ic_params["rho_ks"])
                ic_u_ks_list.append(ic_params["u_ks"])
                ic_p_ks_list.append(ic_params["p_ks"])
                break
            rejected += 1
        else:
            warnings.warn(
                f"Sample {i}: all {max_retries} retries produced NaN/Inf, "
                "using zero-filled sample.",
                stacklevel=2,
            )
            result = {
                "rho": torch.zeros(nt + 1, nx, dtype=torch.float64),
                "u": torch.zeros(nt + 1, nx, dtype=torch.float64),
                "p": torch.zeros(nt + 1, nx, dtype=torch.float64),
            }
            ic_xs_list.append([0.0] * (k + 1))
            ic_rho_ks_list.append([0.0] * k)
            ic_u_ks_list.append([0.0] * k)
            ic_p_ks_list.append([0.0] * k)

        rho_all[i] = result["rho"]
        u_all[i] = result["u"]
        p_all[i] = result["p"]

    if rejected > 0:
        print(f"Rejected {rejected} NaN/Inf samples during generation.")

    t_arr = torch.arange(nt + 1, device=device, dtype=torch.float64) * dt

    return {
        "rho": rho_all,
        "u": u_all,
        "p": p_all,
        "x": x,
        "t": t_arr,
        "dx": dx,
        "dt": dt,
        "nt": nt,
        "ic_xs": np.array(ic_xs_list),
        "ic_rho_ks": np.array(ic_rho_ks_list),
        "ic_u_ks": np.array(ic_u_ks_list),
        "ic_p_ks": np.array(ic_p_ks_list),
    }


__all__ = [
    "generate_one",
    "generate_n",
    "solve_batch",
    "random_piecewise_batch",
    "from_steps",
    "riemann",
    "sod",
    "random_piecewise",
    "primitive_to_conservative",
    "conservative_to_primitive",
    "pressure_from_conservative",
    "sound_speed",
]
