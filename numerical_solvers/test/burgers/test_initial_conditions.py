"""Tests for Burgers initial condition generators."""

import torch

from numerical_solvers.src.burgers.initial_conditions import (
    from_steps,
    random_piecewise,
    random_piecewise_batch,
    riemann,
)


def test_riemann_basic():
    x = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    u0 = riemann(x, u_left=2.0, u_right=-1.0, x_split=0.5)
    assert (u0[x < 0.5] == 2.0).all()
    assert (u0[x >= 0.5] == -1.0).all()


def test_from_steps_multiple():
    x = torch.linspace(0.0, 1.0, 201, dtype=torch.float64)
    u = from_steps(x, [(0.3, 1.0), (0.7, -1.0), (2.0, 2.0)])
    assert (u[x < 0.3] == 1.0).all()
    assert (u[(x >= 0.3) & (x < 0.7)] == -1.0).all()
    assert (u[x >= 0.7] == 2.0).all()


def test_random_piecewise_k_pieces():
    rng = torch.Generator()
    rng.manual_seed(0)
    x = torch.arange(100, dtype=torch.float64) * 0.01
    u0, ic = random_piecewise(x, k=5, rng=rng, u_range=(-1.5, 1.5))
    assert u0.shape == (100,)
    assert len(ic["xs"]) == 6  # k+1 breakpoints
    assert len(ic["u_ks"]) == 5
    assert all(-1.5 <= v <= 1.5 for v in ic["u_ks"])


def test_random_piecewise_batch_shapes():
    rng = torch.Generator()
    rng.manual_seed(0)
    x = torch.arange(80, dtype=torch.float64) * 0.01
    us, params = random_piecewise_batch(x, k=3, n=4, rng=rng)
    assert us.shape == (4, 80)
    assert len(params) == 4
    assert all(len(p["u_ks"]) == 3 for p in params)
