"""Batch-solver equivalence tests for Burgers."""

import torch

from numerical_solvers.src.burgers import generate_one
from numerical_solvers.src.burgers.initial_conditions import random_piecewise_batch
from numerical_solvers.src.burgers.timestepper import solve_batch


def test_batch_equals_sequential():
    rng = torch.Generator()
    rng.manual_seed(7)
    nx, dx, dt, nt = 60, 0.01, 0.002, 20
    x = torch.arange(nx, dtype=torch.float64) * dx
    u0_batch, _ = random_piecewise_batch(x, k=3, n=4, rng=rng, u_range=(-1.0, 1.0))

    # Batch solve
    u_hist_b, valid_b = solve_batch(
        u0_batch, nx=nx, dx=dx, dt=dt, nt=nt,
        bc_type="periodic", flux_type="godunov", reconstruction="constant",
    )
    assert valid_b.all()

    # Per-sample sequential solve
    for i in range(4):
        res = generate_one(
            u0_batch[i], dx=dx, dt=dt, nt=nt,
            bc_type="periodic", flux_type="godunov", reconstruction="constant",
        )
        torch.testing.assert_close(u_hist_b[i], res["u"], atol=1e-12, rtol=0)
