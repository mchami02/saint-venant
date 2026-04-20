"""2D inviscid compressible Euler solver — functional API (PyTorch).

Solves the 2D compressible Euler equations:
    rho_t + (rho*u)_x + (rho*v)_y = 0
    (rho*u)_t + (rho*u^2 + p)_x + (rho*u*v)_y = 0
    (rho*v)_t + (rho*u*v)_x + (rho*v^2 + p)_y = 0
    E_t + (u*(E + p))_x + (v*(E + p))_y = 0

with ideal-gas EOS ``p = (gamma - 1) * (E - 0.5 * rho * (u^2 + v^2))``.

Riemann kernels ported from clawpack/riemann ``euler_1D_py.py`` and extended
with a passive tangential momentum scalar following PyClaw's dimensionally
aware approach. Benchmarks use Liska-Wendroff (2003) 2D Riemann
configurations.

Public API
----------
generate_one : solve a single 2D Euler problem from given ICs.
generate_n   : generate *n* samples with random block-constant ICs.
"""

import warnings

import numpy as np
import torch
from tqdm import tqdm, trange

from .initial_conditions import (
    four_quadrant,
    liska_wendroff,
    random_piecewise,
    random_piecewise_batch,
    sod_x,
    sod_y,
)
from .physics import (
    conservative_to_primitive,
    pressure_from_conservative,
    primitive_to_conservative,
    sound_speed,
)
from .timestepper import solve, solve_batch


def generate_one(
    rho0: torch.Tensor,
    u0: torch.Tensor,
    v0: torch.Tensor,
    p0: torch.Tensor,
    *,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    gamma: float = 1.4,
    bc_type: str = "extrap",
    flux_type: str = "hllc",
    reconstruction: str = "constant",
    max_value: float | None = None,
    cfl: float | None = None,
) -> dict[str, torch.Tensor | float | int | bool]:
    """Solve one 2D Euler problem.

    Parameters
    ----------
    rho0, u0, v0, p0 : 2-D tensors (ny, nx) — initial primitive state.
    dx, dy : cell widths.
    dt : output time interval.
    nt : number of output time steps.
    gamma : ratio of specific heats.
    bc_type : "extrap", "periodic", or "wall".
    flux_type : "hllc", "hll", or "rusanov".
    reconstruction : "constant" or "weno5".
    max_value : if set, terminate early when any value exceeds this threshold.
    cfl : CFL number; default 0.5 (weno5) or 0.45 (constant).

    Returns
    -------
    dict with keys:
        rho, u, v, p each of shape (nt+1, ny, nx),
        x (nx,), y (ny,), t (nt+1,), dx, dy, dt, nt, valid.
    """
    ny, nx = rho0.shape
    x = torch.arange(nx, device=rho0.device, dtype=rho0.dtype) * dx
    y = torch.arange(ny, device=rho0.device, dtype=rho0.dtype) * dy

    _, rho_u0, rho_v0, E0 = primitive_to_conservative(rho0, u0, v0, p0, gamma)

    rho_hist, u_hist, v_hist, p_hist, valid = solve(
        rho0, rho_u0, rho_v0, E0,
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, nt=nt, gamma=gamma,
        bc_type=bc_type, flux_type=flux_type, reconstruction=reconstruction,
        max_value=max_value, cfl=cfl,
    )

    t_arr = torch.arange(nt + 1, device=rho0.device, dtype=rho0.dtype) * dt

    return {
        "rho": rho_hist,
        "u": u_hist,
        "v": v_hist,
        "p": p_hist,
        "x": x,
        "y": y,
        "t": t_arr,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "nt": nt,
        "valid": valid,
    }


def generate_n(
    n: int,
    kx: int,
    ky: int,
    *,
    nx: int = 80,
    ny: int = 80,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    gamma: float = 1.4,
    bc_type: str = "extrap",
    flux_type: str = "hllc",
    reconstruction: str = "constant",
    rho_range: tuple[float, float] = (0.1, 2.0),
    u_range: tuple[float, float] = (-2.0, 2.0),
    v_range: tuple[float, float] = (-2.0, 2.0),
    p_range: tuple[float, float] = (0.1, 5.0),
    max_value: float = 100.0,
    seed: int | None = None,
    show_progress: bool = True,
    device: torch.device | str = "cpu",
    batch_size: int = 8,
    cfl: float | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """Generate *n* samples with random block-constant ICs (kx * ky pieces)."""
    device = torch.device(device)
    x = torch.arange(nx, device=device, dtype=torch.float64) * dx
    y = torch.arange(ny, device=device, dtype=torch.float64) * dy

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    if batch_size <= 1:
        return _generate_n_sequential(
            n, kx, ky, x=x, y=y, nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, nt=nt,
            gamma=gamma, bc_type=bc_type, flux_type=flux_type,
            reconstruction=reconstruction,
            rho_range=rho_range, u_range=u_range, v_range=v_range, p_range=p_range,
            max_value=max_value, rng=rng,
            show_progress=show_progress, device=device, cfl=cfl,
        )

    max_batch_retries = max(10, 3 * ((n + batch_size - 1) // batch_size))
    rejected = 0
    collected = 0

    rho_all = torch.zeros(n, nt + 1, ny, nx, device=device, dtype=torch.float64)
    u_all = torch.zeros_like(rho_all)
    v_all = torch.zeros_like(rho_all)
    p_all = torch.zeros_like(rho_all)

    ic_xs_list: list[list[float]] = []
    ic_ys_list: list[list[float]] = []
    ic_rho_list: list[list[list[float]]] = []
    ic_u_list: list[list[list[float]]] = []
    ic_v_list: list[list[list[float]]] = []
    ic_p_list: list[list[list[float]]] = []

    pbar = tqdm(total=n, desc="Euler2D samples", disable=not show_progress)
    attempts = 0

    while collected < n and attempts < max_batch_retries:
        remaining = n - collected
        bs = min(batch_size, remaining * 2)

        rho0_b, u0_b, v0_b, p0_b, params = random_piecewise_batch(
            x, y, kx, ky, bs, rng,
            rho_range=rho_range, u_range=u_range,
            v_range=v_range, p_range=p_range,
        )
        _, ru0_b, rv0_b, E0_b = primitive_to_conservative(
            rho0_b, u0_b, v0_b, p0_b, gamma,
        )

        rho_h, u_h, v_h, p_h, valid = solve_batch(
            rho0_b, ru0_b, rv0_b, E0_b,
            nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, nt=nt, gamma=gamma,
            bc_type=bc_type, flux_type=flux_type,
            reconstruction=reconstruction,
            max_value=max_value, cfl=cfl,
        )

        good_idx = valid.nonzero(as_tuple=True)[0]
        n_good = len(good_idx)
        n_take = min(n_good, remaining)
        rejected += bs - n_good

        if n_take > 0:
            take_idx = good_idx[:n_take]
            rho_all[collected : collected + n_take] = rho_h[take_idx]
            u_all[collected : collected + n_take] = u_h[take_idx]
            v_all[collected : collected + n_take] = v_h[take_idx]
            p_all[collected : collected + n_take] = p_h[take_idx]
            for i in take_idx.tolist():
                ic_xs_list.append(params[i]["xs"])
                ic_ys_list.append(params[i]["ys"])
                ic_rho_list.append(params[i]["rho_ks"])
                ic_u_list.append(params[i]["u_ks"])
                ic_v_list.append(params[i]["v_ks"])
                ic_p_list.append(params[i]["p_ks"])
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
            ic_xs_list.append([0.0] * (kx + 1))
            ic_ys_list.append([0.0] * (ky + 1))
            zeros = [[0.0] * kx for _ in range(ky)]
            ic_rho_list.append(zeros)
            ic_u_list.append(zeros)
            ic_v_list.append(zeros)
            ic_p_list.append(zeros)

    if rejected > 0:
        print(f"Rejected {rejected} NaN/Inf samples during generation.")

    t_arr = torch.arange(nt + 1, device=device, dtype=torch.float64) * dt

    return {
        "rho": rho_all,
        "u": u_all,
        "v": v_all,
        "p": p_all,
        "x": x,
        "y": y,
        "t": t_arr,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "nt": nt,
        "ic_xs": np.array(ic_xs_list),
        "ic_ys": np.array(ic_ys_list),
        "ic_rho_ks": np.array(ic_rho_list),
        "ic_u_ks": np.array(ic_u_list),
        "ic_v_ks": np.array(ic_v_list),
        "ic_p_ks": np.array(ic_p_list),
    }


def _generate_n_sequential(
    n: int, kx: int, ky: int, *,
    x, y, nx, ny, dx, dy, dt, nt, gamma,
    bc_type, flux_type, reconstruction,
    rho_range, u_range, v_range, p_range,
    max_value, rng, show_progress, device, cfl,
):
    rho_all = torch.zeros(n, nt + 1, ny, nx, device=device, dtype=torch.float64)
    u_all = torch.zeros_like(rho_all)
    v_all = torch.zeros_like(rho_all)
    p_all = torch.zeros_like(rho_all)

    ic_xs_list: list[list[float]] = []
    ic_ys_list: list[list[float]] = []
    ic_rho_list: list[list[list[float]]] = []
    ic_u_list: list[list[list[float]]] = []
    ic_v_list: list[list[list[float]]] = []
    ic_p_list: list[list[list[float]]] = []

    max_retries = 10
    rejected = 0

    it = trange(n, desc="Euler2D samples", disable=not show_progress)
    for i in it:
        for _ in range(max_retries):
            rho0, u0, v0, p0, ic = random_piecewise(
                x, y, kx, ky, rng,
                rho_range=rho_range, u_range=u_range,
                v_range=v_range, p_range=p_range,
            )
            result = generate_one(
                rho0, u0, v0, p0,
                dx=dx, dy=dy, dt=dt, nt=nt, gamma=gamma,
                bc_type=bc_type, flux_type=flux_type,
                reconstruction=reconstruction, max_value=max_value, cfl=cfl,
            )
            if result["valid"]:
                ic_xs_list.append(ic["xs"])
                ic_ys_list.append(ic["ys"])
                ic_rho_list.append(ic["rho_ks"])
                ic_u_list.append(ic["u_ks"])
                ic_v_list.append(ic["v_ks"])
                ic_p_list.append(ic["p_ks"])
                break
            rejected += 1
        else:
            warnings.warn(
                f"Sample {i}: all retries produced NaN/Inf, zero-filled.",
                stacklevel=2,
            )
            result = {
                "rho": torch.zeros(nt + 1, ny, nx, dtype=torch.float64),
                "u": torch.zeros(nt + 1, ny, nx, dtype=torch.float64),
                "v": torch.zeros(nt + 1, ny, nx, dtype=torch.float64),
                "p": torch.zeros(nt + 1, ny, nx, dtype=torch.float64),
            }
            ic_xs_list.append([0.0] * (kx + 1))
            ic_ys_list.append([0.0] * (ky + 1))
            zeros = [[0.0] * kx for _ in range(ky)]
            ic_rho_list.append(zeros)
            ic_u_list.append(zeros)
            ic_v_list.append(zeros)
            ic_p_list.append(zeros)

        rho_all[i] = result["rho"]
        u_all[i] = result["u"]
        v_all[i] = result["v"]
        p_all[i] = result["p"]

    if rejected > 0:
        print(f"Rejected {rejected} NaN/Inf samples during generation.")

    t_arr = torch.arange(nt + 1, device=device, dtype=torch.float64) * dt

    return {
        "rho": rho_all,
        "u": u_all,
        "v": v_all,
        "p": p_all,
        "x": x,
        "y": y,
        "t": t_arr,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "nt": nt,
        "ic_xs": np.array(ic_xs_list),
        "ic_ys": np.array(ic_ys_list),
        "ic_rho_ks": np.array(ic_rho_list),
        "ic_u_ks": np.array(ic_u_list),
        "ic_v_ks": np.array(ic_v_list),
        "ic_p_ks": np.array(ic_p_list),
    }


__all__ = [
    "generate_one",
    "generate_n",
    "solve",
    "solve_batch",
    "sod_x",
    "sod_y",
    "four_quadrant",
    "liska_wendroff",
    "random_piecewise",
    "random_piecewise_batch",
    "primitive_to_conservative",
    "conservative_to_primitive",
    "pressure_from_conservative",
    "sound_speed",
]
