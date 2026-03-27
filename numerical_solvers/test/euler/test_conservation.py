"""Tests for Euler conservation of mass, momentum, and energy."""

import pytest
import torch

from numerical_solvers.src.euler import generate_one
from numerical_solvers.src.euler.initial_conditions import riemann
from numerical_solvers.src.euler.physics import primitive_to_conservative


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


class TestConservationPeriodic:
    def test_mass_conserved(self, flux_recon):
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 50, 0.02, 0.001, 30
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=0.5, u_left=0.0, u_right=0.0, p_left=1.0, p_right=0.5)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="periodic", flux_type=flux_type, reconstruction=reconstruction,
        )
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-8, rtol=1e-8)

    def test_momentum_conserved(self, flux_recon):
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 50, 0.02, 0.001, 30
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=0.5, u_left=0.0, u_right=0.0, p_left=1.0, p_right=0.5)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="periodic", flux_type=flux_type, reconstruction=reconstruction,
        )
        momentum = (result["rho"] * result["u"]).sum(dim=-1) * dx
        torch.testing.assert_close(momentum, momentum[0].expand_as(momentum), atol=1e-8, rtol=1e-8)

    def test_energy_conserved(self, flux_recon):
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 50, 0.02, 0.001, 30
        gamma = 1.4
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=0.5, u_left=0.0, u_right=0.0, p_left=1.0, p_right=0.5)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=gamma,
            bc_type="periodic", flux_type=flux_type, reconstruction=reconstruction,
        )
        rho = result["rho"]
        u = result["u"]
        p = result["p"]
        E = p / (gamma - 1.0) + 0.5 * rho * u**2
        energy = E.sum(dim=-1) * dx
        torch.testing.assert_close(energy, energy[0].expand_as(energy), atol=1e-8, rtol=1e-8)


class TestMassBoundedExtrap:
    def test_mass_bounded(self):
        nx, dx, dt, nt = 32, 0.05, 0.002, 20
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=0.125)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type="hllc", reconstruction="constant",
        )
        mass = result["rho"].sum(dim=-1) * dx
        assert mass[-1] <= mass[0] * 2.0, "Mass grew unreasonably with extrap BC"
