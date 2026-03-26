"""Tests for Euler boundary conditions."""

import pytest
import torch

from numerical_solvers.euler.boundary import apply_ghost_cells


@pytest.fixture(params=[1, 4], ids=["ng1", "ng4"])
def n_ghost(request):
    return request.param


class TestExtrapBC:
    def test_output_length(self, n_ghost):
        nx = 16
        rho = torch.rand(nx, dtype=torch.float64)
        rho_u = torch.rand(nx, dtype=torch.float64)
        E = torch.rand(nx, dtype=torch.float64)
        rho_g, rho_u_g, E_g = apply_ghost_cells(rho, rho_u, E, "extrap", n_ghost=n_ghost)
        assert rho_g.shape[0] == nx + 2 * n_ghost

    def test_left_copies(self, n_ghost):
        nx = 16
        rho = torch.arange(1, nx + 1, dtype=torch.float64)
        rho_u = rho * 0.5
        E = rho * 2.0
        rho_g, rho_u_g, E_g = apply_ghost_cells(rho, rho_u, E, "extrap", n_ghost=n_ghost)
        for i in range(n_ghost):
            assert rho_g[i].item() == rho[0].item()
            assert rho_u_g[i].item() == rho_u[0].item()
            assert E_g[i].item() == E[0].item()

    def test_right_copies(self, n_ghost):
        nx = 16
        rho = torch.arange(1, nx + 1, dtype=torch.float64)
        rho_u = rho * 0.5
        E = rho * 2.0
        rho_g, rho_u_g, E_g = apply_ghost_cells(rho, rho_u, E, "extrap", n_ghost=n_ghost)
        for i in range(n_ghost):
            assert rho_g[-(i + 1)].item() == rho[-1].item()

    def test_interior_unchanged(self, n_ghost):
        nx = 16
        rho = torch.rand(nx, dtype=torch.float64)
        rho_u = torch.rand(nx, dtype=torch.float64)
        E = torch.rand(nx, dtype=torch.float64)
        rho_g, _, _ = apply_ghost_cells(rho, rho_u, E, "extrap", n_ghost=n_ghost)
        torch.testing.assert_close(rho_g[n_ghost:-n_ghost], rho)


class TestPeriodicBC:
    def test_wraparound(self, n_ghost):
        nx = 16
        rho = torch.arange(nx, dtype=torch.float64)
        rho_u = rho * 2.0
        E = rho * 3.0
        rho_g, rho_u_g, E_g = apply_ghost_cells(rho, rho_u, E, "periodic", n_ghost=n_ghost)
        # Left ghosts = last n_ghost of interior
        torch.testing.assert_close(rho_g[:n_ghost], rho[-n_ghost:])
        # Right ghosts = first n_ghost of interior
        torch.testing.assert_close(rho_g[-n_ghost:], rho[:n_ghost])


class TestWallBC:
    def test_density_mirrored(self, n_ghost):
        nx = 16
        rho = torch.arange(1, nx + 1, dtype=torch.float64)
        rho_u = torch.ones(nx, dtype=torch.float64)
        E = torch.ones(nx, dtype=torch.float64) * 2.5
        rho_g, _, _ = apply_ghost_cells(rho, rho_u, E, "wall", n_ghost=n_ghost)
        # Left ghosts: rho[:ng].flip(0)
        expected_left = rho[:n_ghost].flip(0)
        torch.testing.assert_close(rho_g[:n_ghost], expected_left)

    def test_momentum_flipped(self, n_ghost):
        nx = 16
        rho = torch.ones(nx, dtype=torch.float64)
        rho_u = torch.arange(1, nx + 1, dtype=torch.float64)
        E = torch.ones(nx, dtype=torch.float64) * 2.5
        _, rho_u_g, _ = apply_ghost_cells(rho, rho_u, E, "wall", n_ghost=n_ghost)
        # Left ghosts: -rho_u[:ng].flip(0)
        expected_left = -rho_u[:n_ghost].flip(0)
        torch.testing.assert_close(rho_u_g[:n_ghost], expected_left)
        # Right ghosts: -rho_u[-ng:].flip(0)
        expected_right = -rho_u[-n_ghost:].flip(0)
        torch.testing.assert_close(rho_u_g[-n_ghost:], expected_right)

    def test_energy_mirrored(self, n_ghost):
        nx = 16
        rho = torch.ones(nx, dtype=torch.float64)
        rho_u = torch.ones(nx, dtype=torch.float64)
        E = torch.arange(1, nx + 1, dtype=torch.float64)
        _, _, E_g = apply_ghost_cells(rho, rho_u, E, "wall", n_ghost=n_ghost)
        expected_left = E[:n_ghost].flip(0)
        torch.testing.assert_close(E_g[:n_ghost], expected_left)
