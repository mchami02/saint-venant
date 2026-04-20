"""CFL sub-stepping sanity checks for 2D Euler."""

import torch

from numerical_solvers.src.euler2d import generate_one, liska_wendroff


def test_cfl_halving_does_not_blow_up():
    nx = ny = 30
    dx = dy = 1.0 / nx
    dt = 0.05; nt = 4
    x = torch.arange(nx, dtype=torch.float64) * dx
    y = torch.arange(ny, dtype=torch.float64) * dy
    rho0, u0, v0, p0 = liska_wendroff(x, y, config=3)

    r1 = generate_one(rho0, u0, v0, p0, dx=dx, dy=dy, dt=dt, nt=nt, cfl=0.45)
    r2 = generate_one(rho0, u0, v0, p0, dx=dx, dy=dy, dt=dt, nt=nt, cfl=0.2)
    assert r1["valid"] and r2["valid"]
    # Both should be reasonable — they need not match exactly because sub-stepping
    # changes. Just assert no NaNs and comparable density ranges.
    assert torch.isfinite(r1["rho"]).all() and torch.isfinite(r2["rho"]).all()
    assert abs(r1["rho"].max().item() - r2["rho"].max().item()) < 0.5


def test_large_dt_is_substepped():
    """A very large requested output dt should not break the solver — it
    just sub-steps more internally."""
    nx = ny = 30
    dx = dy = 1.0 / nx
    dt = 0.5; nt = 2  # Much larger than CFL allows per step
    x = torch.arange(nx, dtype=torch.float64) * dx
    y = torch.arange(ny, dtype=torch.float64) * dy
    rho0, u0, v0, p0 = liska_wendroff(x, y, config=3)
    result = generate_one(rho0, u0, v0, p0, dx=dx, dy=dy, dt=dt, nt=nt)
    assert result["valid"]
