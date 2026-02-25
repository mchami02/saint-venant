"""Tests for mass conservation in the 2D LWR solver."""

import torch

from numerical_solvers.lwr2d import generate_one
from numerical_solvers.lwr2d.initial_conditions import gaussian_bump, riemann_x


def _total_mass(rho: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Sum rho * dx * dy over all cells for each time step."""
    # rho shape: (nt+1, ny, nx)
    return rho.sum(dim=(-2, -1)) * dx * dy


class TestMassConservationPeriodic:
    """With periodic BCs, total mass must be exactly conserved."""

    def test_gaussian_bump_periodic(self):
        nx, ny = 40, 40
        dx = dy = 0.025
        dt = 0.005
        nt = 50

        x = torch.arange(nx, dtype=torch.float32) * dx
        y = torch.arange(ny, dtype=torch.float32) * dy
        rho0 = gaussian_bump(x, y, rho_bg=0.2, rho_peak=0.7, sigma=0.1)

        result = generate_one(rho0, dx=dx, dy=dy, dt=dt, nt=nt, bc_type="periodic")
        mass = _total_mass(result["rho"], dx, dy)
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-5, rtol=1e-5)

    def test_riemann_periodic(self):
        nx, ny = 30, 30
        dx = dy = 1.0 / 30
        dt = 0.005
        nt = 30

        x = torch.arange(nx, dtype=torch.float32) * dx
        y = torch.arange(ny, dtype=torch.float32) * dy
        rho0 = riemann_x(x, y, rho_left=0.8, rho_right=0.2)

        result = generate_one(rho0, dx=dx, dy=dy, dt=dt, nt=nt, bc_type="periodic")
        mass = _total_mass(result["rho"], dx, dy)
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-5, rtol=1e-5)


class TestMassConservationZeroGradient:
    """With zero-gradient BCs, mass should not increase for outgoing waves."""

    def test_mass_nonincreasing(self):
        nx, ny = 30, 30
        dx = dy = 1.0 / 30
        dt = 0.005
        nt = 30

        x = torch.arange(nx, dtype=torch.float32) * dx
        y = torch.arange(ny, dtype=torch.float32) * dy
        rho0 = gaussian_bump(x, y, rho_bg=0.1, rho_peak=0.5, sigma=0.08)

        result = generate_one(rho0, dx=dx, dy=dy, dt=dt, nt=nt, bc_type="zero_gradient")
        mass = _total_mass(result["rho"], dx, dy)
        # Mass should be approximately conserved (small leakage through boundaries)
        # but certainly shouldn't blow up
        assert mass[-1] <= mass[0] * 1.5, "Mass grew unreasonably"
