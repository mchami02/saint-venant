"""Tests for symmetry properties of the 2D LWR solver."""

import torch

from numerical_solvers.lwr2d import generate_one
from numerical_solvers.lwr2d.initial_conditions import gaussian_bump


class TestDiagonalSymmetry:
    """Gaussian bump with v_max_x == v_max_y on a square grid: rho[t] ~ rho[t].T."""

    def test_symmetric_bump(self):
        n = 40
        dx = dy = 0.025
        dt = 0.005
        nt = 30

        x = torch.arange(n, dtype=torch.float32) * dx
        y = torch.arange(n, dtype=torch.float32) * dy
        rho0 = gaussian_bump(x, y, rho_bg=0.1, rho_peak=0.7, sigma=0.1)

        # Verify IC is symmetric
        torch.testing.assert_close(rho0, rho0.T, atol=1e-7, rtol=0.0)

        result = generate_one(
            rho0,
            dx=dx,
            dy=dy,
            dt=dt,
            nt=nt,
            bc_type="zero_gradient",
            v_max_x=1.0,
            v_max_y=1.0,
        )
        rho = result["rho"]

        for t_idx in range(nt + 1):
            torch.testing.assert_close(rho[t_idx], rho[t_idx].T, atol=1e-6, rtol=1e-6)


class TestUniformStaysUniform:
    """A spatially uniform IC should remain uniform for all time."""

    def test_uniform(self):
        nx, ny = 20, 20
        dx = dy = 0.05
        dt = 0.01
        nt = 20
        rho_val = 0.4

        rho0 = torch.full((ny, nx), rho_val)
        result = generate_one(rho0, dx=dx, dy=dy, dt=dt, nt=nt, bc_type="periodic")
        rho = result["rho"]

        expected = torch.full_like(rho, rho_val)
        torch.testing.assert_close(rho, expected, atol=1e-7, rtol=0.0)
