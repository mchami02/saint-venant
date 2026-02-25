"""LWR traffic flow solver — functional API (nfv Lax-Hopf).

Public API
----------
generate_one : solve a single LWR problem from piecewise-constant IC params.
generate_n   : generate *n* samples with random piecewise-constant ICs.
"""

import numpy as np
import torch
from nfv.flows import Greenshield
from nfv.initial_conditions import PiecewiseConstant
from nfv.problem import Problem
from nfv.solvers import LaxHopf

from .initial_conditions import (
    PiecewiseRandom,
    from_steps,
    random_piecewise,
    riemann,
)


# ------------------------------------------------------------------ public
def generate_one(
    ks: list[float],
    xs: list[float] | None = None,
    *,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
) -> dict[str, torch.Tensor | float | int]:
    """Solve one LWR problem with a piecewise-constant IC.

    Parameters
    ----------
    ks : list[float]
        Piece values for the piecewise-constant IC.
    xs : list[float] or None
        Breakpoint positions. If None, nfv uses uniform breakpoints.
    nx : number of spatial cells.
    nt : total time points (including IC row).
    dx : cell width.
    dt : time step.

    Returns
    -------
    dict with keys:
        rho (nt, nx), x (nx,), t (nt,), dx, dt, nt.
    """
    ic = PiecewiseConstant(ks, x_noise=False)
    if xs is not None:
        ic.xs = np.asarray(xs, dtype=float)

    problem = Problem(nx=nx, nt=nt, dx=dx, dt=dt, ic=[ic], flow=Greenshield())
    rho = problem.solve(LaxHopf, batch_size=1, dtype=torch.float64)
    rho = rho.squeeze(0)  # (nt, nx)

    x = torch.arange(nx, dtype=torch.float64) * dx
    t = torch.arange(nt, dtype=torch.float64) * dt

    return {"rho": rho, "x": x, "t": t, "dx": dx, "dt": dt, "nt": nt}


def generate_n(
    n: int,
    k: int,
    *,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    rho_range: tuple[float, float] = (0.0, 1.0),
    only_shocks: bool = False,
    seed: int | None = None,
    show_progress: bool = True,
    batch_size: int = 4,
) -> dict[str, torch.Tensor | float | int]:
    """Generate *n* samples with random k-piecewise-constant ICs.

    Parameters
    ----------
    n : number of samples.
    k : number of constant pieces per IC.
    nx : number of spatial cells.
    nt : total time points (including IC row).
    dx : cell width.
    dt : time step.
    rho_range : (min, max) for sampled density values.
    only_shocks : if True, sort each IC's ks ascending (ensures shocks
        for Greenshields flux).
    seed : random seed. If not None, creates a np.random.Generator for
        reproducibility; otherwise uses numpy global state (legacy).
    show_progress : show tqdm progress bar during solve.
    batch_size : batch size for the nfv solver.

    Returns
    -------
    dict with keys:
        rho (n, nt, nx), x (nx,), t (nt,), dx, dt, nt.
    """
    rng = np.random.default_rng(seed) if seed is not None else None

    ics = [
        PiecewiseRandom(
            ks=[
                (rng.random() if rng is not None else np.random.rand())
                * (rho_range[1] - rho_range[0])
                + rho_range[0]
                for _ in range(k)
            ],
            x_noise=False,
            rng=rng,
        )
        for _ in range(n)
    ]

    if only_shocks:
        for ic in ics:
            ic.ks.sort()

    problem = Problem(nx=nx, nt=nt, dx=dx, dt=dt, ic=ics, flow=Greenshield())
    rho = problem.solve(
        LaxHopf,
        batch_size=batch_size,
        dtype=torch.float64,
        progressbar=show_progress,
    )  # (n, nt, nx)

    x = torch.arange(nx, dtype=torch.float64) * dx
    t = torch.arange(nt, dtype=torch.float64) * dt

    return {"rho": rho, "x": x, "t": t, "dx": dx, "dt": dt, "nt": nt}


__all__ = [
    "generate_one",
    "generate_n",
    "riemann",
    "random_piecewise",
    "from_steps",
    "PiecewiseRandom",
]
