"""Tests for ARZ wave interaction scenarios (multiple colliding waves)."""

import pytest
import torch

from numerical_solvers.arz import generate_one
from numerical_solvers.arz.initial_conditions import from_steps, three_region


@pytest.fixture(
    params=[
        ("rusanov", "constant"),
        ("hll", "constant"),
        ("hll", "weno5"),
    ],
    ids=["rusanov-const", "hll-const", "hll-weno5"],
)
def flux_recon(request):
    return request.param


class TestThreeRegionCollision:
    """Three-region ICs where two waves propagate and collide."""

    def test_converging_density_jump(self, flux_recon):
        """High-low-high density: two waves propagate inward and collide."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 60, 0.025, 0.002, 80
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = three_region(
            x, rho_left=0.8, rho_mid=0.1, rho_right=0.8,
            v0=0.3, x1=0.3, x2=0.7,
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type=flux_type,
            reconstruction=reconstruction,
        )
        assert result["valid"], f"Solver failed for {flux_type}/{reconstruction}"
        assert torch.isfinite(result["rho"]).all(), "NaN/Inf after wave collision"
        assert (result["rho"] >= -1e-6).all(), f"Negative density: {result['rho'].min()}"

    def test_diverging_density_jump(self, flux_recon):
        """Low-high-low density: rarefaction-like waves interact."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 60, 0.025, 0.002, 80
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = three_region(
            x, rho_left=0.2, rho_mid=0.9, rho_right=0.2,
            v0=0.2, x1=0.3, x2=0.7,
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type=flux_type,
            reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["v"]).all()


class TestMultiRegionCollision:
    """Five or more regions: complex wave interactions."""

    def test_five_region_alternating(self, flux_recon):
        """Alternating high-low density with uniform velocity."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 80, 0.02, 0.002, 100
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = from_steps(
            x,
            rho_steps=[
                (0.3, 0.8), (0.6, 0.2), (0.9, 0.7), (1.2, 0.15), (2.0, 0.6),
            ],
            default_v=0.25,
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type=flux_type,
            reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()

    def test_counter_propagating_velocity(self, flux_recon):
        """Two regions with opposing velocities: head-on wave collision."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 60, 0.025, 0.002, 80
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = from_steps(
            x,
            rho_steps=[(0.75, 0.5), (2.0, 0.5)],
            v_steps=[(0.75, 0.8), (2.0, 0.1)],
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type=flux_type,
            reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["v"]).all()


class TestPeriodicWaveCollision:
    """Waves wrap around and collide under periodic BCs."""

    def test_periodic_collision(self, flux_recon):
        """A wave wraps around and collides with itself."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 60, 0.025, 0.002, 120
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = from_steps(
            x,
            rho_steps=[(0.5, 0.8), (2.0, 0.2)],
            default_v=0.3,
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="periodic", flux_type=flux_type,
            reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        # Mass must be conserved through collisions
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(
            mass, mass[0].expand_as(mass), atol=1e-4, rtol=1e-4,
        )
