"""Tests for Euler solver stability."""

import pytest
import torch

from numerical_solvers.euler import generate_one
from numerical_solvers.euler.initial_conditions import sod


class TestNoNaN:
    def test_sod_constant(self):
        nx, dx, dt, nt = 32, 0.05, 0.002, 20
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type="hllc", reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["u"]).all()
        assert torch.isfinite(result["p"]).all()

    def test_sod_weno5(self):
        nx, dx, dt, nt = 32, 0.05, 0.002, 20
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type="hllc", reconstruction="weno5",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


class TestDensityNonNegative:
    def test_density_nonneg(self):
        nx, dx, dt, nt = 32, 0.05, 0.002, 20
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type="hllc", reconstruction="constant",
        )
        assert (result["rho"] >= -1e-10).all(), f"Negative density: {result['rho'].min()}"


class TestMaxValueTermination:
    def test_max_value_triggers(self):
        nx, dx, dt, nt = 32, 0.05, 0.002, 20
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type="hllc", reconstruction="constant",
            max_value=0.01,
        )
        assert not result["valid"]


class TestAllCombosStable:
    @pytest.mark.parametrize("flux_type", ["hllc", "hll", "rusanov"])
    @pytest.mark.parametrize("bc_type", ["extrap", "periodic"])
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    def test_stable(self, flux_type, bc_type, reconstruction):
        nx, dx, dt, nt = 32, 0.05, 0.001, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type=bc_type, flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"], f"Failed: {flux_type}/{bc_type}/{reconstruction}"
        assert torch.isfinite(result["rho"]).all()

    def test_wall_bc_stable(self):
        nx, dx, dt, nt = 32, 0.05, 0.001, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="wall", flux_type="hllc", reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
