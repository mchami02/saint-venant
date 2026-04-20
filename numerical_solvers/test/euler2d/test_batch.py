"""Batch-solver equivalence tests for 2D Euler."""

import torch

from numerical_solvers.src.euler2d import generate_one
from numerical_solvers.src.euler2d.initial_conditions import random_piecewise_batch
from numerical_solvers.src.euler2d.physics import primitive_to_conservative
from numerical_solvers.src.euler2d.timestepper import solve_batch


def test_batch_equals_sequential():
    rng = torch.Generator(); rng.manual_seed(11)
    nx = ny = 20
    dx = dy = 1.0 / nx
    dt = 0.001; nt = 10
    x = torch.arange(nx, dtype=torch.float64) * dx
    y = torch.arange(ny, dtype=torch.float64) * dy
    rhos, us, vs, ps, _ = random_piecewise_batch(
        x, y, kx=2, ky=2, n=3, rng=rng,
        rho_range=(0.5, 1.5), u_range=(-0.3, 0.3),
        v_range=(-0.3, 0.3), p_range=(0.5, 1.5),
    )
    _, rus, rvs, Es = primitive_to_conservative(rhos, us, vs, ps, 1.4)

    rho_b, u_b, v_b, p_b, valid = solve_batch(
        rhos, rus, rvs, Es,
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, nt=nt, gamma=1.4,
        bc_type="extrap", flux_type="hllc", reconstruction="constant",
    )
    assert valid.all()

    for i in range(3):
        res = generate_one(
            rhos[i], us[i], vs[i], ps[i],
            dx=dx, dy=dy, dt=dt, nt=nt,
            flux_type="hllc", reconstruction="constant", bc_type="extrap",
        )
        torch.testing.assert_close(rho_b[i], res["rho"], atol=1e-12, rtol=0)
        torch.testing.assert_close(u_b[i], res["u"], atol=1e-12, rtol=0)
        torch.testing.assert_close(v_b[i], res["v"], atol=1e-12, rtol=0)
        torch.testing.assert_close(p_b[i], res["p"], atol=1e-12, rtol=0)
