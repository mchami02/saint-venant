"""Tests for Euler steady state preservation."""

import pytest
import torch

from numerical_solvers.src.euler import generate_one


class TestUniformSteady:
    def test_stationary_uniform(self):
        """(rho=1, u=0, p=1) with periodic BC stays constant."""
        nx, dx, dt, nt = 32, 0.05, 0.002, 20
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="periodic", flux_type="hllc", reconstruction="constant",
        )
        torch.testing.assert_close(result["rho"], rho0.unsqueeze(0).expand(nt + 1, -1), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result["u"], u0.unsqueeze(0).expand(nt + 1, -1), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result["p"], p0.unsqueeze(0).expand(nt + 1, -1), atol=1e-10, rtol=1e-10)

    def test_uniform_moving_periodic(self):
        """Uniform flow (rho=1, u=0.5, p=1) with periodic BC stays uniform."""
        nx, dx, dt, nt = 32, 0.05, 0.002, 20
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.full((nx,), 0.5, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="periodic", flux_type="hllc", reconstruction="constant",
        )
        torch.testing.assert_close(result["rho"], rho0.unsqueeze(0).expand(nt + 1, -1), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result["u"], u0.unsqueeze(0).expand(nt + 1, -1), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result["p"], p0.unsqueeze(0).expand(nt + 1, -1), atol=1e-10, rtol=1e-10)
