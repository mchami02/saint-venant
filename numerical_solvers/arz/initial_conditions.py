"""Initial condition generators for the ARZ system (PyTorch).

All public functions return (rho0, v0) — physical velocity, not w.
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
    v_steps: list[tuple[float, float]] | None = None,
    default_v: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (rho0, v0) from piecewise-constant step specifications.

    Parameters
    ----------
    x : 1-D tensor — grid points.
    rho_steps : [(x_end, rho_value), ...] — density profile.
    v_steps : [(x_end, v_value), ...] — velocity profile (optional).
    default_v : constant velocity when *v_steps* is None.
    """
    rho0 = _piecewise_constant(x, rho_steps)
    if v_steps is None:
        v0 = torch.full_like(x, default_v)
    else:
        v0 = _piecewise_constant(x, v_steps)
    return rho0, v0


def riemann(
    x: torch.Tensor,
    rho_left: float = 0.8,
    rho_right: float = 0.2,
    v0: float = 0.1,
    x_split: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-region Riemann problem: (rho_left, v0) | (rho_right, v0)."""
    return from_steps(
        x,
        rho_steps=[(x_split, rho_left), (x.max().item() + 1.0, rho_right)],
        default_v=v0,
    )


def three_region(
    x: torch.Tensor,
    rho_left: float = 0.3,
    rho_mid: float = 0.8,
    rho_right: float = 0.2,
    v0: float = 0.1,
    x1: float = 0.2,
    x2: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Three-region piecewise-constant IC."""
    return from_steps(
        x,
        rho_steps=[(x1, rho_left), (x2, rho_mid), (x.max().item() + 1.0, rho_right)],
        default_v=v0,
    )


def random_piecewise(
    x: torch.Tensor,
    k: int,
    rng: torch.Generator,
    rho_range: tuple[float, float] = (0.1, 1.0),
    v_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random k-piecewise-constant (rho0, v0).

    Parameters
    ----------
    x : 1-D tensor — grid points.
    k : number of constant pieces.
    rng : PyTorch random generator for reproducibility.
    rho_range, v_range : (min, max) for sampled values.
    """
    x_min, x_max = x.min().item(), x.max().item()
    L = x_max - x_min

    # Random breakpoints (sorted, within domain)
    breaks = torch.rand(k - 1, generator=rng) * L + x_min
    breaks = breaks.sort().values.tolist()
    # Append a sentinel past the right boundary
    breaks.append(x_max + 1.0)

    rho_lo, rho_hi = rho_range
    v_lo, v_hi = v_range

    rho_vals = torch.rand(k, generator=rng) * (rho_hi - rho_lo) + rho_lo
    v_vals = torch.rand(k, generator=rng) * (v_hi - v_lo) + v_lo

    rho_steps = [(b, rv.item()) for b, rv in zip(breaks, rho_vals, strict=False)]
    v_steps = [(b, vv.item()) for b, vv in zip(breaks, v_vals, strict=False)]

    return from_steps(x, rho_steps=rho_steps, v_steps=v_steps)
