"""Tests for ARZ mass and rho*w conservation."""

import pytest
import torch

from numerical_solvers.arz import generate_one
from numerical_solvers.arz.initial_conditions import riemann
from numerical_solvers.arz.physics import pressure


@pytest.fixture(
    params=[
        ("rusanov", "constant"),
        ("hll", "constant"),
        ("rusanov", "weno5"),
        ("hll", "weno5"),
    ],
    ids=["rusanov-const", "hll-const", "rusanov-weno5", "hll-weno5"],
)
def flux_recon(request):
    return request.param


class TestMassConservationPeriodic:
    def test_mass_conserved(self, flux_recon):
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 40, 0.025, 0.002, 30
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2, x_split=0.5)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="periodic", flux_type=flux_type, reconstruction=reconstruction,
        )
        mass = result["rho"].sum(dim=-1) * dx  # (nt+1,)
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-4, rtol=1e-4)

    def test_rho_w_conserved(self, flux_recon):
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 40, 0.025, 0.002, 30
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2, x_split=0.5)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="periodic", flux_type=flux_type, reconstruction=reconstruction,
        )
        rho_w = result["rho"] * result["w"]
        rho_w_total = rho_w.sum(dim=-1) * dx
        torch.testing.assert_close(rho_w_total, rho_w_total[0].expand_as(rho_w_total), atol=1e-4, rtol=1e-4)


class TestMassBoundedZeroGrad:
    def test_mass_bounded(self):
        nx, dx, dt, nt = 32, 0.05, 0.005, 20
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type="hll", reconstruction="constant",
        )
        mass = result["rho"].sum(dim=-1) * dx
        assert mass[-1] <= mass[0] * 2.0, "Mass grew unreasonably with zero-gradient BC"
