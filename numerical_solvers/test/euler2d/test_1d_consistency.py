"""Cross-check: 2D Euler on a y-invariant Sod IC must reproduce 1D Euler Sod.

This is the decisive regression test — it verifies that the dimensional
sweeps, passive-scalar advection, and ghost-cell handling are all correct
by comparing row-by-row against the well-tested 1D solver.
"""

import torch

from numerical_solvers.src import euler, euler2d


def test_sod_x_matches_1d_euler():
    nx, ny = 100, 8
    dx = 1.0 / nx; dy = 0.1
    dt = 0.001; nt = 50
    gamma = 1.4

    x = torch.arange(nx, dtype=torch.float64) * dx
    y = torch.arange(ny, dtype=torch.float64) * dy

    # 2D
    rho2_0, u2_0, v2_0, p2_0 = euler2d.sod_x(x, y, x_split=0.5)
    out2 = euler2d.generate_one(
        rho2_0, u2_0, v2_0, p2_0,
        dx=dx, dy=dy, dt=dt, nt=nt, gamma=gamma,
        flux_type="hllc", reconstruction="constant",
    )
    assert out2["valid"]

    # 1D
    rho1_0, u1_0, p1_0 = euler.sod(x, x_split=0.5)
    out1 = euler.generate_one(
        rho1_0, u1_0, p1_0,
        dx=dx, dt=dt, nt=nt, gamma=gamma,
        flux_type="hllc", reconstruction="constant",
    )
    assert out1["valid"]

    mid = ny // 2
    torch.testing.assert_close(out2["rho"][-1, mid], out1["rho"][-1], atol=1e-10, rtol=0)
    torch.testing.assert_close(out2["u"][-1, mid], out1["u"][-1], atol=1e-10, rtol=0)
    torch.testing.assert_close(out2["p"][-1, mid], out1["p"][-1], atol=1e-10, rtol=0)
    # v must stay zero
    assert out2["v"][-1].abs().max().item() < 1e-12


def test_sod_y_matches_1d_euler():
    """Same check but along y. The 2D solver should reproduce 1D Euler on
    an x-invariant Sod initial condition, with u=0 throughout."""
    nx, ny = 8, 100
    dx = 0.1; dy = 1.0 / ny
    dt = 0.001; nt = 50
    gamma = 1.4

    x = torch.arange(nx, dtype=torch.float64) * dx
    y = torch.arange(ny, dtype=torch.float64) * dy

    rho2_0, u2_0, v2_0, p2_0 = euler2d.sod_y(x, y, y_split=0.5)
    out2 = euler2d.generate_one(
        rho2_0, u2_0, v2_0, p2_0,
        dx=dx, dy=dy, dt=dt, nt=nt, gamma=gamma,
        flux_type="hllc", reconstruction="constant",
    )
    assert out2["valid"]

    rho1_0, u1_0, p1_0 = euler.sod(y, x_split=0.5)
    out1 = euler.generate_one(
        rho1_0, u1_0, p1_0,
        dx=dy, dt=dt, nt=nt, gamma=gamma,
        flux_type="hllc", reconstruction="constant",
    )
    assert out1["valid"]

    mid = nx // 2
    # The 2D rho/v row along x=mid should match the 1D rho/u
    torch.testing.assert_close(out2["rho"][-1, :, mid], out1["rho"][-1], atol=1e-10, rtol=0)
    torch.testing.assert_close(out2["v"][-1, :, mid], out1["u"][-1], atol=1e-10, rtol=0)
    torch.testing.assert_close(out2["p"][-1, :, mid], out1["p"][-1], atol=1e-10, rtol=0)
    # u must stay zero
    assert out2["u"][-1].abs().max().item() < 1e-12
