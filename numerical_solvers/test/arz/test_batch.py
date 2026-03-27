"""Tests for batch-parallel ARZ solver and batch-compatible helpers."""

import pytest
import torch

from numerical_solvers.src.arz import generate_n, generate_one, solve_batch
from numerical_solvers.src.arz.boundary import apply_ghost_cells
from numerical_solvers.src.arz.initial_conditions import (
    random_piecewise,
    random_piecewise_batch,
    riemann,
)
from numerical_solvers.src.arz.physics import pressure
from numerical_solvers.src.arz.timestepper import solve
from numerical_solvers.src.arz.weno import weno5_reconstruct


# ------------------------------------------------------------------ WENO batch
class TestWENOBatch:
    def test_batch_output_shapes(self):
        v = torch.rand(3, 16)
        v_minus, v_plus = weno5_reconstruct(v)
        assert v_minus.shape == (3, 9)
        assert v_plus.shape == (3, 9)

    def test_batch_matches_sequential(self):
        torch.manual_seed(0)
        v_batch = torch.rand(5, 20)
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
        rho = torch.rand(B, nx) + 0.1
        rho_w = rho * (torch.rand(B, nx) * 0.5 + 0.1)
        return rho, rho_w

    def test_zero_gradient_shapes(self, n_ghost):
        rho, rho_w = self._make_batch()
        B, nx = rho.shape
        rho_g, rho_w_g = apply_ghost_cells(
            rho, rho_w, "zero_gradient", 0.0, n_ghost=n_ghost, gamma=1.0,
        )
        assert rho_g.shape == (B, nx + 2 * n_ghost)
        assert rho_w_g.shape == (B, nx + 2 * n_ghost)

    def test_periodic_shapes(self, n_ghost):
        rho, rho_w = self._make_batch()
        B, nx = rho.shape
        rho_g, rho_w_g = apply_ghost_cells(
            rho, rho_w, "periodic", 0.0, n_ghost=n_ghost, gamma=1.0,
        )
        assert rho_g.shape == (B, nx + 2 * n_ghost)

    def test_dirichlet_shapes(self, n_ghost):
        rho, rho_w = self._make_batch()
        B, nx = rho.shape
        rho_g, rho_w_g = apply_ghost_cells(
            rho, rho_w, "dirichlet", 0.0,
            n_ghost=n_ghost, gamma=1.0,
            bc_left=(0.5, 0.2), bc_right=(0.3, 0.1),
        )
        assert rho_g.shape == (B, nx + 2 * n_ghost)

    def test_inflow_outflow_shapes(self, n_ghost):
        rho, rho_w = self._make_batch()
        B, nx = rho.shape
        rho_g, rho_w_g = apply_ghost_cells(
            rho, rho_w, "inflow_outflow", 0.0,
            n_ghost=n_ghost, gamma=1.0,
            bc_left=(0.5, 0.2),
        )
        assert rho_g.shape == (B, nx + 2 * n_ghost)

    def test_interior_preserved(self, n_ghost):
        rho, rho_w = self._make_batch()
        ng = n_ghost
        for bc in ["zero_gradient", "periodic", "dirichlet", "inflow_outflow"]:
            kwargs = {"n_ghost": ng, "gamma": 1.0}
            if bc == "dirichlet":
                kwargs["bc_left"] = (0.5, 0.2)
                kwargs["bc_right"] = (0.3, 0.1)
            elif bc == "inflow_outflow":
                kwargs["bc_left"] = (0.5, 0.2)
            rho_g, _ = apply_ghost_cells(rho, rho_w, bc, 0.0, **kwargs)
            torch.testing.assert_close(rho_g[:, ng:-ng], rho)

    def test_batch_matches_sequential(self, n_ghost):
        rho, rho_w = self._make_batch(B=3)
        for bc in ["zero_gradient", "periodic"]:
            rho_g_batch, rw_g_batch = apply_ghost_cells(
                rho, rho_w, bc, 0.0, n_ghost=n_ghost, gamma=1.0,
            )
            for i in range(3):
                rho_g_single, rw_g_single = apply_ghost_cells(
                    rho[i], rho_w[i], bc, 0.0, n_ghost=n_ghost, gamma=1.0,
                )
                torch.testing.assert_close(rho_g_batch[i], rho_g_single)
                torch.testing.assert_close(rw_g_batch[i], rw_g_single)


# ----------------------------------------------------------- solve_batch
class TestSolveBatch:
    @pytest.fixture(params=["constant", "weno5"])
    def reconstruction(self, request):
        return request.param

    def _make_riemann_batch(self, B=4, nx=32):
        """Create B identical Riemann problems stacked as a batch."""
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, v0=0.1)
        rho0_batch = rho0.unsqueeze(0).expand(B, -1).clone()
        v0_batch = v0.unsqueeze(0).expand(B, -1).clone()
        return rho0_batch, v0_batch, nx

    def test_output_shapes(self, reconstruction):
        B, nx, nt = 4, 32, 10
        rho0_batch, v0_batch, _ = self._make_riemann_batch(B, nx)
        gamma = 1.0
        w0 = v0_batch + pressure(rho0_batch, gamma)
        rho_w0 = rho0_batch * w0

        rho_h, w_h, v_h, valid = solve_batch(
            rho0_batch, rho_w0,
            nx=nx, dx=0.05, dt=0.005, nt=nt, gamma=gamma,
            reconstruction=reconstruction,
        )
        assert rho_h.shape == (B, nt + 1, nx)
        assert w_h.shape == (B, nt + 1, nx)
        assert v_h.shape == (B, nt + 1, nx)
        assert valid.shape == (B,)

    def test_batch_matches_sequential(self, reconstruction):
        """Batch of N identical ICs should match N sequential solves."""
        B, nx, nt = 3, 32, 15
        gamma = 1.0
        dx, dt = 0.05, 0.005
        rho0_batch, v0_batch, _ = self._make_riemann_batch(B, nx)
        w0 = v0_batch + pressure(rho0_batch, gamma)
        rho_w0 = rho0_batch * w0

        rho_h, w_h, v_h, valid = solve_batch(
            rho0_batch, rho_w0,
            nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
            reconstruction=reconstruction,
        )

        # Compare against single solve
        rho0_single = rho0_batch[0]
        w0_single = v0_batch[0] + pressure(rho0_single, gamma)
        rho_w0_single = rho0_single * w0_single
        rho_s, w_s, v_s, valid_s = solve(
            rho0_single, rho_w0_single,
            nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
            reconstruction=reconstruction,
        )

        assert valid[0].item() == valid_s
        torch.testing.assert_close(rho_h[0], rho_s, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(w_h[0], w_s, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v_h[0], v_s, atol=1e-6, rtol=1e-6)

    def test_different_ics_in_batch(self):
        """Batch with different ICs should match individual solves."""
        nx, nt = 32, 10
        gamma, dx, dt = 1.0, 0.05, 0.005
        x = torch.arange(nx, dtype=torch.float32) * dx

        rng = torch.Generator().manual_seed(123)
        N = 5
        rho0_list, rho_w0_list = [], []
        for _ in range(N):
            rho0, v0, _ = random_piecewise(x, 2, rng, rho_range=(0.1, 1.0), v_range=(0.0, 0.5))
            w0 = v0 + pressure(rho0, gamma)
            rho0_list.append(rho0)
            rho_w0_list.append(rho0 * w0)

        rho0_batch = torch.stack(rho0_list)
        rho_w0_batch = torch.stack(rho_w0_list)

        rho_h, w_h, v_h, valid = solve_batch(
            rho0_batch, rho_w0_batch,
            nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
            reconstruction="constant",
        )

        for i in range(N):
            rho_s, w_s, v_s, valid_s = solve(
                rho0_list[i], rho_w0_list[i],
                nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
                reconstruction="constant",
            )
            assert valid[i].item() == valid_s
            if valid_s:
                torch.testing.assert_close(rho_h[i], rho_s, atol=1e-6, rtol=1e-6)

    def test_mixed_validity(self):
        """Batch where some samples blow up and others stay valid."""
        nx, nt = 32, 20
        gamma, dx, dt = 1.0, 0.05, 0.005
        x = torch.arange(nx, dtype=torch.float32) * dx

        # Stable IC
        rho_good = torch.full((nx,), 0.5)
        v_good = torch.full((nx,), 0.1)
        w_good = v_good + pressure(rho_good, gamma)
        rho_w_good = rho_good * w_good

        # Unstable IC (extreme values that should blow up with tight max_value)
        rho_bad = torch.full((nx,), 5.0)
        v_bad = torch.full((nx,), 5.0)
        w_bad = v_bad + pressure(rho_bad, gamma)
        rho_w_bad = rho_bad * w_bad

        rho0 = torch.stack([rho_good, rho_bad, rho_good])
        rho_w0 = torch.stack([rho_w_good, rho_w_bad, rho_w_good])

        rho_h, w_h, v_h, valid = solve_batch(
            rho0, rho_w0,
            nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
            reconstruction="constant", max_value=2.0,
        )

        # Good samples should be valid, bad sample might not be
        assert valid[0].item() is True
        assert valid[2].item() is True

    def test_single_sample_batch(self):
        """Batch of 1 should work."""
        nx, nt = 32, 10
        gamma, dx, dt = 1.0, 0.05, 0.005
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x)
        w0 = v0 + pressure(rho0, gamma)
        rho_w0 = rho0 * w0

        rho_h, w_h, v_h, valid = solve_batch(
            rho0.unsqueeze(0), rho_w0.unsqueeze(0),
            nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
            reconstruction="constant",
        )
        assert rho_h.shape == (1, nt + 1, nx)
        assert valid.shape == (1,)

    def test_time_varying_inflow_raises(self):
        rho0 = torch.rand(2, 16)
        rho_w0 = torch.rand(2, 16)
        with pytest.raises(ValueError, match="time_varying_inflow"):
            solve_batch(
                rho0, rho_w0,
                nx=16, dx=0.1, dt=0.01, nt=5, gamma=1.0,
                bc_type="time_varying_inflow",
            )

    def test_all_bc_types(self):
        """All supported BC types work in batch mode."""
        nx, nt = 32, 5
        gamma, dx, dt = 1.0, 0.05, 0.005
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x)
        w0 = v0 + pressure(rho0, gamma)
        rho_w0 = rho0 * w0
        rho0_batch = rho0.unsqueeze(0).expand(2, -1).clone()
        rho_w0_batch = rho_w0.unsqueeze(0).expand(2, -1).clone()

        for bc in ["zero_gradient", "periodic", "dirichlet", "inflow_outflow"]:
            kwargs = {}
            if bc == "dirichlet":
                kwargs["bc_left"] = (0.5, 0.1)
                kwargs["bc_right"] = (0.3, 0.1)
            elif bc == "inflow_outflow":
                kwargs["bc_left"] = (0.5, 0.1)
            rho_h, _, _, valid = solve_batch(
                rho0_batch, rho_w0_batch,
                nx=nx, dx=dx, dt=dt, nt=nt, gamma=gamma,
                bc_type=bc, reconstruction="constant", **kwargs,
            )
            assert rho_h.shape == (2, nt + 1, nx)


# -------------------------------------------------- random_piecewise_batch
class TestRandomPiecewiseBatch:
    def test_output_shapes(self):
        x = torch.arange(50, dtype=torch.float32) * 0.02
        rng = torch.Generator().manual_seed(42)
        rho0, v0, params = random_piecewise_batch(x, 3, 10, rng)
        assert rho0.shape == (10, 50)
        assert v0.shape == (10, 50)
        assert len(params) == 10
        assert len(params[0]["xs"]) == 4  # k+1

    def test_values_in_range(self):
        x = torch.arange(50, dtype=torch.float32) * 0.02
        rng = torch.Generator().manual_seed(42)
        rho0, v0, _ = random_piecewise_batch(
            x, 3, 20, rng, rho_range=(0.2, 0.8), v_range=(0.1, 0.5),
        )
        assert rho0.min() >= 0.2 - 1e-6
        assert rho0.max() <= 0.8 + 1e-6
        assert v0.min() >= 0.1 - 1e-6
        assert v0.max() <= 0.5 + 1e-6


# -------------------------------------------------------- generate_n batch
class TestGenerateNBatch:
    def test_shapes(self):
        result = generate_n(
            6, 2, nx=32, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, batch_size=4,
        )
        assert result["rho"].shape == (6, 11, 32)
        assert result["v"].shape == (6, 11, 32)
        assert result["w"].shape == (6, 11, 32)
        assert result["ic_xs"].shape[0] == 6
        assert result["ic_rho_ks"].shape[0] == 6
        assert result["ic_v_ks"].shape[0] == 6

    def test_all_finite(self):
        result = generate_n(
            10, 2, nx=32, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, batch_size=8,
        )
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["v"]).all()

    def test_return_keys(self):
        result = generate_n(
            4, 2, nx=32, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, batch_size=4,
        )
        expected_keys = {"rho", "v", "w", "x", "t", "dx", "dt", "nt", "ic_xs", "ic_rho_ks", "ic_v_ks"}
        assert set(result.keys()) == expected_keys

    def test_sequential_fallback(self):
        """batch_size=1 should produce valid results."""
        result = generate_n(
            4, 2, nx=32, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, batch_size=1,
        )
        assert result["rho"].shape == (4, 11, 32)
        assert torch.isfinite(result["rho"]).all()

    def test_weno5_batch(self):
        result = generate_n(
            4, 2, nx=32, dx=0.05, dt=0.002, nt=10, seed=42,
            show_progress=False, batch_size=4, reconstruction="weno5",
        )
        assert result["rho"].shape == (4, 11, 32)
        assert torch.isfinite(result["rho"]).all()

    def test_batch_size_larger_than_n(self):
        """batch_size > n should still work."""
        result = generate_n(
            3, 2, nx=32, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, batch_size=64,
        )
        assert result["rho"].shape == (3, 11, 32)
