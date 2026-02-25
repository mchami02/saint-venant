"""Tests that 2D solver reduces to 1D behaviour for 1D initial conditions."""

import torch

from numerical_solvers.lwr2d import generate_one
from numerical_solvers.lwr2d.initial_conditions import riemann_x, riemann_y


class TestXRiemannConsistency:
    """x-Riemann IC (uniform in y): all rows must be identical at every step."""

    def test_all_rows_identical(self):
        nx, ny = 40, 20
        dx = dy = 0.025
        dt = 0.005
        nt = 40

        x = torch.arange(nx, dtype=torch.float32) * dx
        y = torch.arange(ny, dtype=torch.float32) * dy
        rho0 = riemann_x(x, y, rho_left=0.8, rho_right=0.2)

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
        rho = result["rho"]  # (nt+1, ny, nx)

        # Every row should equal the first row at every timestep
        for t_idx in range(nt + 1):
            row0 = rho[t_idx, 0, :]
            for j in range(1, ny):
                torch.testing.assert_close(rho[t_idx, j, :], row0, atol=1e-6, rtol=1e-6)


class TestYRiemannConsistency:
    """y-Riemann IC (uniform in x): all columns must be identical."""

    def test_all_columns_identical(self):
        nx, ny = 20, 40
        dx = dy = 0.025
        dt = 0.005
        nt = 40

        x = torch.arange(nx, dtype=torch.float32) * dx
        y = torch.arange(ny, dtype=torch.float32) * dy
        rho0 = riemann_y(x, y, rho_bottom=0.7, rho_top=0.3)

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
        rho = result["rho"]  # (nt+1, ny, nx)

        # Every column should equal the first column at every timestep
        for t_idx in range(nt + 1):
            col0 = rho[t_idx, :, 0]
            for i in range(1, nx):
                torch.testing.assert_close(rho[t_idx, :, i], col0, atol=1e-6, rtol=1e-6)
