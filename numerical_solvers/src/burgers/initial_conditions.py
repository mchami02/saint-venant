"""Initial condition generators for the inviscid Burgers equation (PyTorch)."""

import torch


def _piecewise_constant(x: torch.Tensor, steps: list[tuple[float, float]]) -> torch.Tensor:
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
    u_steps: list[tuple[float, float]],
) -> torch.Tensor:
    """Build u0 from piecewise-constant step specifications.

    u_steps = [(x_end, u_value), ...] — value applies on [prev_x, x_end).
    """
    return _piecewise_constant(x, u_steps)


def riemann(
    x: torch.Tensor,
    u_left: float = 1.0,
    u_right: float = 0.0,
    x_split: float = 0.5,
) -> torch.Tensor:
    """Two-region Riemann problem for Burgers."""
    sentinel = x.max().item() + 1.0
    return from_steps(x, [(x_split, u_left), (sentinel, u_right)])


def random_piecewise(
    x: torch.Tensor,
    k: int,
    rng: torch.Generator,
    u_range: tuple[float, float] = (-2.0, 2.0),
) -> tuple[torch.Tensor, dict]:
    """Generate a random k-piecewise-constant u0 profile.

    Returns
    -------
    u0 : 1-D tensor — initial conserved variable.
    ic_params : dict with keys "xs" (k+1 breakpoints including domain
        boundaries) and "u_ks" (k piece values).
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

    cell_indices = torch.randperm(nx, generator=rng)[:n_breaks].sort().values
    offsets = torch.rand(n_breaks, generator=rng)
    breaks = ((cell_indices + offsets) * dx + x_min).tolist()
    breaks.append(x_max + 1.0)  # sentinel past right boundary

    u_lo, u_hi = u_range
    u_vals = torch.rand(k, generator=rng) * (u_hi - u_lo) + u_lo

    u_steps = [(b, uv.item()) for b, uv in zip(breaks, u_vals, strict=False)]

    domain_right = x_max + dx
    ic_params = {
        "xs": [x_min] + breaks[:-1] + [domain_right],
        "u_ks": u_vals.tolist(),
    }
    u0 = from_steps(x, u_steps)
    return u0, ic_params


def random_piecewise_batch(
    x: torch.Tensor,
    k: int,
    n: int,
    rng: torch.Generator,
    u_range: tuple[float, float] = (-2.0, 2.0),
) -> tuple[torch.Tensor, list[dict]]:
    """Generate *n* random k-piecewise-constant u0 profiles as a batch."""
    us: list[torch.Tensor] = []
    params: list[dict] = []
    for _ in range(n):
        u0, ic = random_piecewise(x, k, rng, u_range=u_range)
        us.append(u0)
        params.append(ic)
    return torch.stack(us), params
