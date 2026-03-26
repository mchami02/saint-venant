"""Tests for ARZ solver stability."""

import pytest
import torch

from numerical_solvers.arz import generate_one
from numerical_solvers.arz.initial_conditions import riemann


class TestNoNaN:
    def test_riemann_constant(self):
        nx, dx, dt, nt = 32, 0.05, 0.005, 20
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type="hll", reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["v"]).all()

    def test_riemann_weno5(self):
        nx, dx, dt, nt = 32, 0.05, 0.005, 20
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type="hll", reconstruction="weno5",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


class TestDensityNonNegative:
    def test_density_nonneg(self):
        nx, dx, dt, nt = 32, 0.05, 0.005, 20
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.9, rho_right=0.1)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type="hll", reconstruction="constant",
        )
        assert (result["rho"] >= -1e-10).all(), f"Negative density: {result['rho'].min()}"


class TestMaxValueTermination:
    def test_max_value_triggers(self):
        nx, dx, dt, nt = 32, 0.05, 0.005, 20
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type="hll", reconstruction="constant",
            max_value=0.01,
        )
        assert not result["valid"]


class TestAllCombosStable:
    @pytest.mark.parametrize("flux_type", ["rusanov", "hll"])
    @pytest.mark.parametrize("bc_type", ["zero_gradient", "periodic"])
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    def test_stable(self, flux_type, bc_type, reconstruction):
        nx, dx, dt, nt = 32, 0.05, 0.002, 15
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.5, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type=bc_type, flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"], f"Failed: {flux_type}/{bc_type}/{reconstruction}"
        assert torch.isfinite(result["rho"]).all()
