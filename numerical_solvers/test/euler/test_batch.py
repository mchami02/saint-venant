"""Tests for batch-parallel Euler solver and batch-compatible helpers."""

import pytest
import torch

from numerical_solvers.src.euler import generate_n, solve_batch
from numerical_solvers.src.euler.boundary import apply_ghost_cells
from numerical_solvers.src.euler.initial_conditions import (
    random_piecewise,
    random_piecewise_batch,
    sod,
)
from numerical_solvers.src.euler.physics import primitive_to_conservative
from numerical_solvers.src.euler.timestepper import solve
from numerical_solvers.src.euler.weno import weno5_reconstruct


# ------------------------------------------------------------------ WENO batch
class TestWENOBatch:
    def test_batch_output_shapes(self):
        v = torch.rand(3, 16, dtype=torch.float64)
        v_minus, v_plus = weno5_reconstruct(v)
        assert v_minus.shape == (3, 9)
        assert v_plus.shape == (3, 9)

    def test_batch_matches_sequential(self):
        torch.manual_seed(0)
        v_batch = torch.rand(5, 20, dtype=torch.float64)
        vm_batch, vp_batch = weno5_reconstruct(v_batch)

        for i in range(5):
            vm_single, vp_single = weno5_reconstruct(v_batch[i])
            torch.testing.assert_close(vm_batch[i], vm_single)
            torch.testing.assert_close(vp_batch[i], vp_single)


# -------------------------------------------------------------- boundary batch
class TestBoundaryBatch:
    @pytest.fixture(params=[1, 4])
    def n_ghost(self, request):
        return request.param

    def _make_batch(self, B=4, nx=16):
        torch.manual_seed(42)
        rho = torch.rand(B, nx, dtype=torch.float64) + 0.1
        rho_u = rho * (torch.rand(B, nx, dtype=torch.float64) - 0.5)
        E = rho * 2.5 + 0.5 * rho_u**2 / rho  # valid total energy
        return rho, rho_u, E

    def test_extrap_shapes(self, n_ghost):
        rho, rho_u, E = self._make_batch()
        B, nx = rho.shape
        rho_g, rho_u_g, E_g = apply_ghost_cells(
            rho, rho_u, E, "extrap", n_ghost=n_ghost,
        )
        assert rho_g.shape == (B, nx + 2 * n_ghost)
        assert rho_u_g.shape == (B, nx + 2 * n_ghost)
        assert E_g.shape == (B, nx + 2 * n_ghost)

    def test_periodic_shapes(self, n_ghost):
        rho, rho_u, E = self._make_batch()
        B, nx = rho.shape
        rho_g, _, _ = apply_ghost_cells(
            rho, rho_u, E, "periodic", n_ghost=n_ghost,
        )
        assert rho_g.shape == (B, nx + 2 * n_ghost)

    def test_wall_shapes(self, n_ghost):
        rho, rho_u, E = self._make_batch()
        B, nx = rho.shape
        rho_g, rho_u_g, E_g = apply_ghost_cells(
            rho, rho_u, E, "wall", n_ghost=n_ghost,
        )
        assert rho_g.shape == (B, nx + 2 * n_ghost)

    def test_interior_preserved(self, n_ghost):
        rho, rho_u, E = self._make_batch()
        ng = n_ghost
        for bc in ["extrap", "periodic", "wall"]:
            rho_g, rho_u_g, E_g = apply_ghost_cells(
                rho, rho_u, E, bc, n_ghost=ng,
            )
            torch.testing.assert_close(rho_g[:, ng:-ng], rho)
            torch.testing.assert_close(rho_u_g[:, ng:-ng], rho_u)
            torch.testing.assert_close(E_g[:, ng:-ng], E)

    def test_batch_matches_sequential(self, n_ghost):
        rho, rho_u, E = self._make_batch(B=3)
        for bc in ["extrap", "periodic", "wall"]:
            rho_g_b, ru_g_b, E_g_b = apply_ghost_cells(
                rho, rho_u, E, bc, n_ghost=n_ghost,
            )
            for i in range(3):
                rho_g_s, ru_g_s, E_g_s = apply_ghost_cells(
                    rho[i], rho_u[i], E[i], bc, n_ghost=n_ghost,
                )
                torch.testing.assert_close(rho_g_b[i], rho_g_s)
                torch.testing.assert_close(ru_g_b[i], ru_g_s)
                torch.testing.assert_close(E_g_b[i], E_g_s)


# ----------------------------------------------------------- solve_batch
class TestSolveBatch:
    @pytest.fixture(params=["constant", "weno5"])
    def reconstruction(self, request):
        return request.param

    def _make_sod_batch(self, B=4, nx=32):
        x = torch.arange(nx, dtype=torch.float64) * 0.05
        rho0, u0, p0 = sod(x)
        rho0_b = rho0.unsqueeze(0).expand(B, -1).clone()
        u0_b = u0.unsqueeze(0).expand(B, -1).clone()
        p0_b = p0.unsqueeze(0).expand(B, -1).clone()
        return rho0_b, u0_b, p0_b, nx

    def test_output_shapes(self, reconstruction):
        B, nx, nt = 4, 32, 10
        gamma = 1.4
        rho0_b, u0_b, p0_b, _ = self._make_sod_batch(B, nx)
        _, rho_u0_b, E0_b = primitive_to_conservative(rho0_b, u0_b, p0_b, gamma)

        rho_h, u_h, p_h, valid = solve_batch(
            rho0_b, rho_u0_b, E0_b,
            nx=nx, dx=0.05, dt=0.001, nt=nt, gamma=gamma,
            reconstruction=reconstruction,
        )
        assert rho_h.shape == (B, nt + 1, nx)
        assert u_h.shape == (B, nt + 1, nx)
        assert p_h.shape == (B, nt + 1, nx)
        assert valid.shape == (B,)

    def test_batch_matches_sequential(self, reconstruction):
        B, nx, nt = 3, 32, 15
        gamma, dx, dt = 1.4, 0.05, 0.001
        rho0_b, u0_b, p0_b, _ = self._make_sod_batch(B, nx)
        _, rho_u0_b, E0_b = primitive_to_conservative(rho0_b, u0_b, p0_b, gamma)

        rho_h, u_h, p_h, valid = solve_batch(
            rho0_b, rho_u0_b, E0_b,
            nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
            reconstruction=reconstruction,
        )

        # Compare against single solve
        rho0_s = rho0_b[0]
        _, rho_u0_s, E0_s = primitive_to_conservative(rho0_s, u0_b[0], p0_b[0], gamma)
        rho_s, u_s, p_s, valid_s = solve(
            rho0_s, rho_u0_s, E0_s,
            nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
            reconstruction=reconstruction,
        )

        assert valid[0].item() == valid_s
        torch.testing.assert_close(rho_h[0], rho_s, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(u_h[0], u_s, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(p_h[0], p_s, atol=1e-10, rtol=1e-10)

    def test_different_ics_in_batch(self):
        nx, nt = 32, 10
        gamma, dx, dt = 1.4, 0.05, 0.001
        x = torch.arange(nx, dtype=torch.float64) * dx

        rng = torch.Generator().manual_seed(123)
        N = 5
        rho0_list, rho_u0_list, E0_list = [], [], []
        for _ in range(N):
            rho0, u0, p0, _ = random_piecewise(x, 2, rng)
            _, rho_u0, E0 = primitive_to_conservative(rho0, u0, p0, gamma)
            rho0_list.append(rho0)
            rho_u0_list.append(rho_u0)
            E0_list.append(E0)

        rho0_b = torch.stack(rho0_list)
        rho_u0_b = torch.stack(rho_u0_list)
        E0_b = torch.stack(E0_list)

        rho_h, u_h, p_h, valid = solve_batch(
            rho0_b, rho_u0_b, E0_b,
            nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
            reconstruction="constant",
        )

        for i in range(N):
            rho_s, u_s, p_s, valid_s = solve(
                rho0_list[i], rho_u0_list[i], E0_list[i],
                nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
                reconstruction="constant",
            )
            assert valid[i].item() == valid_s
            if valid_s:
                torch.testing.assert_close(rho_h[i], rho_s, atol=1e-10, rtol=1e-10)

    def test_single_sample_batch(self):
        nx, nt = 32, 10
        gamma, dx, dt = 1.4, 0.05, 0.001
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        _, rho_u0, E0 = primitive_to_conservative(rho0, u0, p0, gamma)

        rho_h, u_h, p_h, valid = solve_batch(
            rho0.unsqueeze(0), rho_u0.unsqueeze(0), E0.unsqueeze(0),
            nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
            reconstruction="constant",
        )
        assert rho_h.shape == (1, nt + 1, nx)
        assert valid.shape == (1,)

    def test_all_bc_types(self):
        B, nx, nt = 2, 32, 5
        gamma, dx, dt = 1.4, 0.05, 0.001
        rho0_b, u0_b, p0_b, _ = self._make_sod_batch(B, nx)
        _, rho_u0_b, E0_b = primitive_to_conservative(rho0_b, u0_b, p0_b, gamma)

        for bc in ["extrap", "periodic", "wall"]:
            rho_h, _, _, valid = solve_batch(
                rho0_b, rho_u0_b, E0_b,
                nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
                bc_type=bc, reconstruction="constant",
            )
            assert rho_h.shape == (B, nt + 1, nx)


# -------------------------------------------------- random_piecewise_batch
class TestRandomPiecewiseBatch:
    def test_output_shapes(self):
        x = torch.arange(50, dtype=torch.float64) * 0.02
        rng = torch.Generator().manual_seed(42)
        rho0, u0, p0, params = random_piecewise_batch(x, 3, 10, rng)
        assert rho0.shape == (10, 50)
        assert u0.shape == (10, 50)
        assert p0.shape == (10, 50)
        assert len(params) == 10
        assert len(params[0]["xs"]) == 4  # k+1

    def test_values_in_range(self):
        x = torch.arange(50, dtype=torch.float64) * 0.02
        rng = torch.Generator().manual_seed(42)
        rho0, u0, p0, _ = random_piecewise_batch(
            x, 3, 20, rng,
            rho_range=(0.2, 0.8), u_range=(-1.0, 1.0), p_range=(0.5, 2.0),
        )
        assert rho0.min() >= 0.2 - 1e-6
        assert rho0.max() <= 0.8 + 1e-6
        assert u0.min() >= -1.0 - 1e-6
        assert u0.max() <= 1.0 + 1e-6
        assert p0.min() >= 0.5 - 1e-6
        assert p0.max() <= 2.0 + 1e-6


# -------------------------------------------------------- generate_n batch
class TestGenerateNBatch:
    def test_shapes(self):
        result = generate_n(
            6, 2, nx=32, dx=0.05, dt=0.001, nt=10, seed=42,
            show_progress=False, batch_size=4,
        )
        assert result["rho"].shape == (6, 11, 32)
        assert result["u"].shape == (6, 11, 32)
        assert result["p"].shape == (6, 11, 32)
        assert result["ic_xs"].shape[0] == 6
        assert result["ic_rho_ks"].shape[0] == 6
        assert result["ic_u_ks"].shape[0] == 6
        assert result["ic_p_ks"].shape[0] == 6

    def test_all_finite(self):
        result = generate_n(
            10, 2, nx=32, dx=0.05, dt=0.001, nt=10, seed=42,
            show_progress=False, batch_size=8,
        )
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["u"]).all()
        assert torch.isfinite(result["p"]).all()

    def test_return_keys(self):
        result = generate_n(
            4, 2, nx=32, dx=0.05, dt=0.001, nt=10, seed=42,
            show_progress=False, batch_size=4,
        )
        expected_keys = {
            "rho", "u", "p", "x", "t", "dx", "dt", "nt",
            "ic_xs", "ic_rho_ks", "ic_u_ks", "ic_p_ks",
        }
        assert set(result.keys()) == expected_keys

    def test_sequential_fallback(self):
        result = generate_n(
            4, 2, nx=32, dx=0.05, dt=0.001, nt=10, seed=42,
            show_progress=False, batch_size=1,
        )
        assert result["rho"].shape == (4, 11, 32)
        assert torch.isfinite(result["rho"]).all()

    def test_weno5_batch(self):
        result = generate_n(
            4, 2, nx=32, dx=0.05, dt=0.0005, nt=10, seed=42,
            show_progress=False, batch_size=4, reconstruction="weno5",
        )
        assert result["rho"].shape == (4, 11, 32)
        assert torch.isfinite(result["rho"]).all()

    def test_batch_size_larger_than_n(self):
        result = generate_n(
            3, 2, nx=32, dx=0.05, dt=0.001, nt=10, seed=42,
            show_progress=False, batch_size=64,
        )
        assert result["rho"].shape == (3, 11, 32)
