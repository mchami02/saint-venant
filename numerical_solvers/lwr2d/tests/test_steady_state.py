"""Tests that known steady states remain unchanged."""

import torch

from numerical_solvers.lwr2d import generate_one


class TestSteadyStates:
    """rho=0, rho=rho_max, and constant rho should be steady states."""

    def _run_steady(self, rho_val: float, rho_max: float = 1.0):
        nx, ny = 20, 20
        dx = dy = 0.05
        dt = 0.01
        nt = 20

        rho0 = torch.full((ny, nx), rho_val)
        result = generate_one(
            rho0,
            dx=dx,
            dy=dy,
            dt=dt,
            nt=nt,
            bc_type="periodic",
            rho_max=rho_max,
        )
        rho = result["rho"]
        expected = torch.full_like(rho, rho_val)
        torch.testing.assert_close(rho, expected, atol=1e-7, rtol=0.0)

    def test_rho_zero(self):
        self._run_steady(0.0)

    def test_rho_max(self):
        self._run_steady(1.0)

    def test_rho_constant_mid(self):
        self._run_steady(0.5)

    def test_rho_constant_custom_rho_max(self):
        self._run_steady(2.5, rho_max=5.0)
