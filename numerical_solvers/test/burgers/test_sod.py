"""Shock speed (Rankine-Hugoniot) test for Burgers."""

import torch

from numerical_solvers.src.burgers import generate_one, riemann


def _shock_position(u: torch.Tensor, x: torch.Tensor, uL: float, uR: float) -> float:
    """Locate the mid-point of the transition from uL to uR."""
    mid = 0.5 * (uL + uR)
    # First index where u crosses below mid (for uL > uR case)
    if uL > uR:
        idx = (u < mid).nonzero()
    else:
        idx = (u > mid).nonzero()
    if idx.numel() == 0:
        return float("nan")
    return x[idx[0]].item()


def test_shock_speed_rankine_hugoniot():
    """Shock from uL=1 → uR=0 must propagate at (uL+uR)/2 = 0.5."""
    nx = 400
    dx = 1.0 / nx
    dt = 0.001
    nt = 300
    x = torch.arange(nx, dtype=torch.float64) * dx
    x_split = 0.3
    u0 = riemann(x, u_left=1.0, u_right=0.0, x_split=x_split)

    result = generate_one(
        u0, dx=dx, dt=dt, nt=nt,
        flux_type="godunov", reconstruction="constant", bc_type="extrap",
    )
    assert result["valid"]

    # Analytical shock position at t = nt*dt
    t_final = nt * dt
    expected = x_split + 0.5 * t_final  # shock speed = (1+0)/2

    x_shock = _shock_position(result["u"][-1], x, 1.0, 0.0)
    # Within dx of the analytical position
    assert abs(x_shock - expected) < 2 * dx, (
        f"shock at {x_shock:.4f}, expected {expected:.4f}"
    )
