"""Tests for Euler wave interaction scenarios (multiple colliding waves)."""

import pytest
import torch

from numerical_solvers.euler import generate_one
from numerical_solvers.euler.initial_conditions import from_steps, riemann
from numerical_solvers.euler.physics import primitive_to_conservative


@pytest.fixture(
    params=[
        ("hllc", "constant"),
        ("hll", "constant"),
        ("rusanov", "constant"),
        ("hllc", "weno5"),
    ],
    ids=["hllc-const", "hll-const", "rusanov-const", "hllc-weno5"],
)
def flux_recon(request):
    return request.param


class TestBlastWave:
    """Blast wave problems: two strong shocks collide."""

    def test_woodward_colella_blast(self, flux_recon):
        """Simplified blast wave: high-low-high pressure, stationary gas.

        Two strong shocks propagate inward from pressure jumps and collide.
        This is one of the hardest tests for Euler solvers.
        """
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 80, 0.0125, 0.0005, 100
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.3, 1.0), (0.7, 1.0), (2.0, 1.0)],
            u_steps=[(2.0, 0.0)],
            p_steps=[(0.3, 10.0), (0.7, 0.1), (2.0, 10.0)],
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type,
            reconstruction=reconstruction,
        )
        assert result["valid"], f"Blast wave failed: {flux_type}/{reconstruction}"
        assert torch.isfinite(result["rho"]).all(), "NaN/Inf in blast wave"
        assert (result["rho"] >= -1e-6).all(), f"Negative density: {result['rho'].min()}"


class TestThreeRegionCollision:
    """Three-region ICs producing converging or diverging waves."""

    def test_converging_flows(self, flux_recon):
        """Two regions flowing toward each other: head-on collision."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 60, 0.025, 0.001, 80
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.75, 1.0), (2.0, 1.0)],
            u_steps=[(0.75, 1.0), (2.0, -1.0)],
            p_steps=[(2.0, 1.0)],
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type,
            reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["u"]).all()
        assert torch.isfinite(result["p"]).all()

    def test_high_low_high_density(self, flux_recon):
        """High-low-high density: compression waves collide in the middle."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 60, 0.025, 0.001, 80
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.4, 2.0), (0.8, 0.2), (2.0, 2.0)],
            u_steps=[(2.0, 0.0)],
            p_steps=[(0.4, 2.0), (0.8, 0.2), (2.0, 2.0)],
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type,
            reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()


class TestMultiRegionInteraction:
    """Many regions creating complex wave interaction patterns."""

    def test_five_region_alternating(self, flux_recon):
        """Five alternating pressure regions: many shocks and rarefactions interact."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 80, 0.0125, 0.0005, 100
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[
                (0.2, 1.0), (0.4, 0.5), (0.6, 1.5), (0.8, 0.3), (2.0, 1.0),
            ],
            u_steps=[(2.0, 0.0)],
            p_steps=[
                (0.2, 5.0), (0.4, 0.5), (0.6, 3.0), (0.8, 0.2), (2.0, 2.0),
            ],
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type,
            reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["p"]).all()


class TestPeriodicWaveCollision:
    """Waves wrap around and collide under periodic BCs."""

    def test_periodic_collision_conservation(self, flux_recon):
        """Shock wraps around and collides; conservation must hold."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 60, 0.025, 0.001, 100
        gamma = 1.4
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = riemann(
            x, rho_left=1.0, rho_right=0.5,
            u_left=0.5, u_right=-0.5,
            p_left=1.0, p_right=0.5,
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=gamma,
            bc_type="periodic", flux_type=flux_type,
            reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

        # Mass conservation through collisions
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(
            mass, mass[0].expand_as(mass), atol=1e-7, rtol=1e-7,
        )

        # Momentum conservation through collisions
        momentum = (result["rho"] * result["u"]).sum(dim=-1) * dx
        torch.testing.assert_close(
            momentum, momentum[0].expand_as(momentum), atol=1e-7, rtol=1e-7,
        )

        # Energy conservation through collisions
        rho, u, p = result["rho"], result["u"], result["p"]
        E = p / (gamma - 1.0) + 0.5 * rho * u**2
        energy = E.sum(dim=-1) * dx
        torch.testing.assert_close(
            energy, energy[0].expand_as(energy), atol=1e-7, rtol=1e-7,
        )


class TestSodDoubleRiemann:
    """Two Sod-like interfaces creating 6 waves that interact."""

    def test_double_sod(self, flux_recon):
        """Two Sod interfaces close together: waves from each interact."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 100, 0.01, 0.0005, 80
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.3, 1.0), (0.7, 0.125), (2.0, 1.0)],
            u_steps=[(2.0, 0.0)],
            p_steps=[(0.3, 1.0), (0.7, 0.1), (2.0, 1.0)],
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type,
            reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()
