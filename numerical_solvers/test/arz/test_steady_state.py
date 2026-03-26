"""Tests for ARZ steady state preservation."""

import pytest
import torch

from numerical_solvers.arz import generate_one


class TestUniformSteady:
    @pytest.mark.parametrize("bc_type", ["periodic", "zero_gradient"])
    def test_uniform_density_velocity_steady(self, bc_type):
        """Uniform rho and v should remain constant."""
        nx, dx, dt, nt = 32, 0.05, 0.005, 20
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type=bc_type, flux_type="hll", reconstruction="constant",
        )
        torch.testing.assert_close(
            result["rho"], rho0.unsqueeze(0).expand(nt + 1, -1), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            result["v"], v0.unsqueeze(0).expand(nt + 1, -1), atol=1e-6, rtol=1e-6
        )

    def test_zero_density_steady(self):
        nx, dx, dt, nt = 32, 0.05, 0.005, 20
        rho0 = torch.zeros(nx)
        v0 = torch.zeros(nx)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="periodic", flux_type="hll", reconstruction="constant",
        )
        torch.testing.assert_close(result["rho"], torch.zeros(nt + 1, nx), atol=1e-10, rtol=0)
