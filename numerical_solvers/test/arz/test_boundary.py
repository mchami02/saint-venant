"""Tests for ARZ boundary conditions."""

import pytest
import torch

from numerical_solvers.arz.boundary import apply_ghost_cells
from numerical_solvers.arz.physics import pressure


@pytest.fixture(params=[1, 4], ids=["ng1", "ng4"])
def n_ghost(request):
    return request.param


class TestPeriodicBC:
    def test_output_length(self, n_ghost):
        nx = 16
        rho = torch.rand(nx)
        rho_w = torch.rand(nx)
        rho_g, rho_w_g = apply_ghost_cells(rho, rho_w, "periodic", t=0.0, n_ghost=n_ghost, gamma=1.0)
        assert rho_g.shape[0] == nx + 2 * n_ghost

    def test_wraparound(self, n_ghost):
        nx = 16
        rho = torch.arange(nx, dtype=torch.float32)
        rho_w = torch.arange(nx, dtype=torch.float32) * 2
        rho_g, rho_w_g = apply_ghost_cells(rho, rho_w, "periodic", t=0.0, n_ghost=n_ghost, gamma=1.0)
        # Left ghosts = last n_ghost of interior
        torch.testing.assert_close(rho_g[:n_ghost], rho[-n_ghost:])
        # Right ghosts = first n_ghost of interior
        torch.testing.assert_close(rho_g[-n_ghost:], rho[:n_ghost])


class TestZeroGradientBC:
    def test_left_copies(self, n_ghost):
        nx = 16
        rho = torch.arange(1, nx + 1, dtype=torch.float32)
        rho_w = torch.arange(1, nx + 1, dtype=torch.float32) * 0.5
        rho_g, rho_w_g = apply_ghost_cells(rho, rho_w, "zero_gradient", t=0.0, n_ghost=n_ghost, gamma=1.0)
        for i in range(n_ghost):
            assert rho_g[i].item() == rho[0].item()
            assert rho_w_g[i].item() == rho_w[0].item()

    def test_right_copies(self, n_ghost):
        nx = 16
        rho = torch.arange(1, nx + 1, dtype=torch.float32)
        rho_w = torch.arange(1, nx + 1, dtype=torch.float32) * 0.5
        rho_g, rho_w_g = apply_ghost_cells(rho, rho_w, "zero_gradient", t=0.0, n_ghost=n_ghost, gamma=1.0)
        for i in range(n_ghost):
            assert rho_g[-(i + 1)].item() == rho[-1].item()

    def test_interior_unchanged(self, n_ghost):
        nx = 16
        rho = torch.rand(nx)
        rho_w = torch.rand(nx)
        rho_g, _ = apply_ghost_cells(rho, rho_w, "zero_gradient", t=0.0, n_ghost=n_ghost, gamma=1.0)
        torch.testing.assert_close(rho_g[n_ghost:-n_ghost], rho)


class TestDirichletBC:
    def test_left_values(self, n_ghost):
        nx = 16
        gamma = 1.0
        rho = torch.rand(nx)
        rho_w = torch.rand(nx)
        rho_l, v_l = 0.6, 0.4
        rho_g, rho_w_g = apply_ghost_cells(
            rho, rho_w, "dirichlet", t=0.0, n_ghost=n_ghost, gamma=gamma,
            bc_left=(rho_l, v_l), bc_right=(0.3, 0.5),
        )
        w_l = v_l + rho_l**gamma
        rw_l = rho_l * w_l
        for i in range(n_ghost):
            assert abs(rho_g[i].item() - rho_l) < 1e-6
            assert abs(rho_w_g[i].item() - rw_l) < 1e-6

    def test_right_values(self, n_ghost):
        nx = 16
        gamma = 1.0
        rho = torch.rand(nx)
        rho_w = torch.rand(nx)
        rho_r, v_r = 0.3, 0.5
        rho_g, rho_w_g = apply_ghost_cells(
            rho, rho_w, "dirichlet", t=0.0, n_ghost=n_ghost, gamma=gamma,
            bc_left=(0.5, 1.0), bc_right=(rho_r, v_r),
        )
        w_r = v_r + rho_r**gamma
        rw_r = rho_r * w_r
        for i in range(n_ghost):
            assert abs(rho_g[-(i + 1)].item() - rho_r) < 1e-6
            assert abs(rho_w_g[-(i + 1)].item() - rw_r) < 1e-6


class TestInflowOutflowBC:
    def test_left_dirichlet_right_extrapolation(self):
        nx = 16
        gamma = 1.0
        ng = 1
        rho = torch.arange(1, nx + 1, dtype=torch.float32) * 0.05
        rho_w = rho * 0.3
        rho_g, rho_w_g = apply_ghost_cells(
            rho, rho_w, "inflow_outflow", t=0.0, n_ghost=ng, gamma=gamma,
        )
        # Right ghost = zero-gradient (last interior value)
        assert rho_g[-1].item() == rho[-1].item()
        # Left ghost = Dirichlet with default (0.5, 1.0)
        rho_l, v_l = 0.5, 1.0
        w_l = v_l + rho_l**gamma
        rw_l = rho_l * w_l
        assert abs(rho_g[0].item() - rho_l) < 1e-6
        assert abs(rho_w_g[0].item() - rw_l) < 1e-6


class TestTimeVaryingInflowBC:
    def test_callable_produces_different_values(self):
        nx = 16
        gamma = 1.0
        ng = 1
        rho = torch.rand(nx)
        rho_w = torch.rand(nx)

        def bc_fn(t):
            return (0.5 + t, 1.0 - t)

        rho_g1, _ = apply_ghost_cells(
            rho, rho_w, "time_varying_inflow", t=0.0, n_ghost=ng, gamma=gamma, bc_left_time=bc_fn,
        )
        rho_g2, _ = apply_ghost_cells(
            rho, rho_w, "time_varying_inflow", t=0.5, n_ghost=ng, gamma=gamma, bc_left_time=bc_fn,
        )
        # Left ghost should differ at different times
        assert rho_g1[0].item() != rho_g2[0].item()
