"""Symmetry tests for 2D Euler: rotational / reflection invariance."""

import torch

from numerical_solvers.src.euler2d import generate_one


def test_xy_reflection_of_sod_x_equals_sod_y():
    """Running Sod along x on a square grid must equal running Sod along y,
    under the axis swap (rho[x,y] <-> rho[y,x], u <-> v)."""
    n = 40
    dx = dy = 1.0 / n
    dt = 0.001; nt = 20
    x = torch.arange(n, dtype=torch.float64) * dx
    y = torch.arange(n, dtype=torch.float64) * dy

    from numerical_solvers.src.euler2d import sod_x, sod_y

    rho_x, u_x, v_x, p_x = sod_x(x, y, x_split=0.5)
    out_x = generate_one(
        rho_x, u_x, v_x, p_x,
        dx=dx, dy=dy, dt=dt, nt=nt,
        flux_type="hllc", reconstruction="constant",
    )
    rho_y, u_y, v_y, p_y = sod_y(x, y, y_split=0.5)
    out_y = generate_one(
        rho_y, u_y, v_y, p_y,
        dx=dx, dy=dy, dt=dt, nt=nt,
        flux_type="hllc", reconstruction="constant",
    )

    # Swap axes: out_x['rho'][t, j, i] should equal out_y['rho'][t, i, j]
    atol = 1e-10
    torch.testing.assert_close(
        out_x["rho"][-1].transpose(-1, -2), out_y["rho"][-1], atol=atol, rtol=0,
    )
    torch.testing.assert_close(
        out_x["u"][-1].transpose(-1, -2), out_y["v"][-1], atol=atol, rtol=0,
    )
    torch.testing.assert_close(
        out_x["v"][-1].transpose(-1, -2), out_y["u"][-1], atol=atol, rtol=0,
    )
    torch.testing.assert_close(
        out_x["p"][-1].transpose(-1, -2), out_y["p"][-1], atol=atol, rtol=0,
    )
