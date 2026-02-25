"""Tests for CFL utility and solver stability."""

import torch

from numerical_solvers.lwr2d import generate_one
from numerical_solvers.lwr2d.initial_conditions import four_quadrant
from numerical_solvers.lwr2d.physics import cfl_dt


class TestCflDt:
    """Verify cfl_dt returns correct values."""

    def test_basic(self):
        dt = cfl_dt(dx=0.1, dy=0.1, max_speed_x=1.0, max_speed_y=1.0, cfl_number=0.5)
        # dt * (1/0.1 + 1/0.1) = dt * 20 = 0.5  =>  dt = 0.025
        assert abs(dt - 0.025) < 1e-12

    def test_anisotropic(self):
        dt = cfl_dt(dx=0.05, dy=0.1, max_speed_x=2.0, max_speed_y=1.0, cfl_number=0.45)
        # dt * (2/0.05 + 1/0.1) = dt * (40 + 10) = dt * 50 = 0.45
        # => dt = 0.009
        assert abs(dt - 0.009) < 1e-12

    def test_zero_speed(self):
        dt = cfl_dt(dx=0.1, dy=0.1, max_speed_x=0.0, max_speed_y=0.0)
        assert dt == float("inf")


class TestSolverStability:
    """Solver at CFL-limited dt should produce bounded, reasonable output."""

    def test_four_quadrant_stable(self):
        nx, ny = 30, 30
        dx = dy = 1.0 / 30
        v_max = 1.0
        rho_max = 1.0
        dt = cfl_dt(dx, dy, v_max, v_max, cfl_number=0.45)
        nt = 50

        x = torch.arange(nx, dtype=torch.float32) * dx
        y = torch.arange(ny, dtype=torch.float32) * dy
        rho0 = four_quadrant(x, y, rho_bl=0.8, rho_br=0.3, rho_tl=0.5, rho_tr=0.1)

        result = generate_one(
            rho0,
            dx=dx,
            dy=dy,
            dt=dt,
            nt=nt,
            bc_type="zero_gradient",
            v_max_x=v_max,
            v_max_y=v_max,
            rho_max=rho_max,
        )
        rho = result["rho"]

        # All values within [0, rho_max]
        assert rho.min() >= 0.0
        assert rho.max() <= rho_max
        # No NaN or Inf
        assert torch.isfinite(rho).all()
