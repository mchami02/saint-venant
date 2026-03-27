"""Initial condition generators for the 1D Euler system (PyTorch).

All public functions return (rho0, u0, p0) — primitive variables.
"""

import torch


def _piecewise_constant(
    x: torch.Tensor, steps: list[tuple[float, float]]
) -> torch.Tensor:
    """Build a piecewise-constant profile.

    steps = [(x_end_1, value_1), ...]; value applies on [prev_x, x_end).
    """
    steps = sorted(steps, key=lambda p: p[0])
    out = torch.full_like(x, steps[-1][1])
    for i in range(len(steps) - 2, -1, -1):
        x_end, val = steps[i]
        out = torch.where(x < x_end, torch.full_like(out, val), out)
    return out


def from_steps(
    x: torch.Tensor,
    rho_steps: list[tuple[float, float]],
    u_steps: list[tuple[float, float]] | None = None,
    p_steps: list[tuple[float, float]] | None = None,
    default_u: float = 0.0,
    default_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (rho0, u0, p0) from piecewise-constant step specifications.

    Parameters
    ----------
    x : 1-D tensor — grid points.
    rho_steps : [(x_end, rho_value), ...] — density profile.
    u_steps : [(x_end, u_value), ...] — velocity profile (optional).
    p_steps : [(x_end, p_value), ...] — pressure profile (optional).
    default_u : constant velocity when *u_steps* is None.
    default_p : constant pressure when *p_steps* is None.
    """
    rho0 = _piecewise_constant(x, rho_steps)
    if u_steps is None:
        u0 = torch.full_like(x, default_u)
    else:
        u0 = _piecewise_constant(x, u_steps)
    if p_steps is None:
        p0 = torch.full_like(x, default_p)
    else:
        p0 = _piecewise_constant(x, p_steps)
    return rho0, u0, p0


def riemann(
    x: torch.Tensor,
    rho_left: float = 1.0,
    rho_right: float = 0.125,
    u_left: float = 0.0,
    u_right: float = 0.0,
    p_left: float = 1.0,
    p_right: float = 0.1,
    x_split: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-region Riemann problem."""
    sentinel = x.max().item() + 1.0
    return from_steps(
        x,
        rho_steps=[(x_split, rho_left), (sentinel, rho_right)],
        u_steps=[(x_split, u_left), (sentinel, u_right)],
        p_steps=[(x_split, p_left), (sentinel, p_right)],
    )


def sod(
    x: torch.Tensor,
    x_split: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classic Sod shock tube problem.

    Left:  (rho, u, p) = (1.0, 0.0, 1.0)
    Right: (rho, u, p) = (0.125, 0.0, 0.1)
    """
    return riemann(
        x,
        rho_left=1.0,
        rho_right=0.125,
        u_left=0.0,
        u_right=0.0,
        p_left=1.0,
        p_right=0.1,
        x_split=x_split,
    )


def random_piecewise(
    x: torch.Tensor,
    k: int,
    rng: torch.Generator,
    rho_range: tuple[float, float] = (0.1, 2.0),
    u_range: tuple[float, float] = (-2.0, 2.0),
    p_range: tuple[float, float] = (0.1, 5.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Generate a random k-piecewise-constant (rho0, u0, p0).

    Parameters
    ----------
    x : 1-D tensor — grid points.
    k : number of constant pieces.
    rng : PyTorch random generator for reproducibility.
    rho_range, u_range, p_range : (min, max) for sampled values.

    Returns
    -------
    rho0 : 1-D tensor — initial density.
    u0 : 1-D tensor — initial velocity.
    p0 : 1-D tensor — initial pressure.
    ic_params : dict with keys "xs", "rho_ks", "u_ks", "p_ks".
    """
    nx = len(x)
    x_min, x_max = x.min().item(), x.max().item()
    dx = (x[1] - x[0]).item() if nx > 1 else 0.0

    n_breaks = k - 1
    if n_breaks > nx:
        raise ValueError(
            f"Cannot place {n_breaks} breakpoints in {nx} cells "
            "(need n_breaks <= nx)"
        )

    # Pick n_breaks distinct cells, place one breakpoint within each
    cell_indices = torch.randperm(nx, generator=rng)[:n_breaks].sort().values
    offsets = torch.rand(n_breaks, generator=rng)
    breaks = ((cell_indices + offsets) * dx + x_min).tolist()
    # Append a sentinel past the right boundary
    breaks.append(x_max + 1.0)

    rho_lo, rho_hi = rho_range
    u_lo, u_hi = u_range
    p_lo, p_hi = p_range

    rho_vals = torch.rand(k, generator=rng) * (rho_hi - rho_lo) + rho_lo
    u_vals = torch.rand(k, generator=rng) * (u_hi - u_lo) + u_lo
    p_vals = torch.rand(k, generator=rng) * (p_hi - p_lo) + p_lo

    rho_steps = [(b, rv.item()) for b, rv in zip(breaks, rho_vals, strict=False)]
    u_steps = [(b, uv.item()) for b, uv in zip(breaks, u_vals, strict=False)]
    p_steps = [(b, pv.item()) for b, pv in zip(breaks, p_vals, strict=False)]

    # Build IC params: actual domain boundaries (not sentinel)
    domain_right = x_max + dx
    ic_params = {
        "xs": [x_min] + breaks[:-1] + [domain_right],
        "rho_ks": rho_vals.tolist(),
        "u_ks": u_vals.tolist(),
        "p_ks": p_vals.tolist(),
    }

    rho0, u0, p0 = from_steps(x, rho_steps=rho_steps, u_steps=u_steps, p_steps=p_steps)
    return rho0, u0, p0, ic_params
