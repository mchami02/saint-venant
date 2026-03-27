"""Tests for LWR solver with many-piece initial conditions (k >= 5).

Verifies that the solver handles complex piecewise-constant ICs correctly:
output shapes, density bounds, no NaN/Inf, and reproducibility.
"""

import numpy as np
import pytest
import torch

from numerical_solvers.src.lwr import generate_n, generate_one


# ============================================================================
# Hand-crafted many-piece ICs via generate_one
# ============================================================================


class TestManyPieceGenerateOne:
    """generate_one with hand-crafted ICs having 5+ pieces."""

    def test_staircase_ascending_5_pieces(self):
        """Monotone ascending staircase: all rarefaction interfaces."""
        ks = [0.1, 0.3, 0.5, 0.7, 0.9]
        xs = [0, 0.2, 0.4, 0.6, 0.8, 1]
        result = generate_one(ks, xs, nx=100, nt=100, dx=0.01, dt=0.002)
        rho = result["rho"]
        assert torch.isfinite(rho).all()
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10

    def test_staircase_descending_5_pieces(self):
        """Monotone descending staircase: all shock interfaces."""
        ks = [0.9, 0.7, 0.5, 0.3, 0.1]
        xs = [0, 0.2, 0.4, 0.6, 0.8, 1]
        result = generate_one(ks, xs, nx=100, nt=100, dx=0.01, dt=0.002)
        rho = result["rho"]
        assert torch.isfinite(rho).all()
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10

    def test_alternating_high_low_8_pieces(self):
        """Alternating 0.9/0.1: maximum wave interactions."""
        ks = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
        xs = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        result = generate_one(ks, xs, nx=160, nt=200, dx=0.00625, dt=0.001)
        rho = result["rho"]
        assert torch.isfinite(rho).all()
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10

    def test_clustered_breakpoints_6_pieces(self):
        """Breakpoints clustered near center: tightly packed discontinuities."""
        ks = [0.2, 0.8, 0.1, 0.9, 0.3, 0.7]
        xs = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1]
        result = generate_one(ks, xs, nx=100, nt=100, dx=0.01, dt=0.002)
        rho = result["rho"]
        assert torch.isfinite(rho).all()
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10

    def test_near_critical_density_7_pieces(self):
        """All pieces near rho=0.5 (critical density / max flux)."""
        ks = [0.45, 0.55, 0.48, 0.52, 0.49, 0.51, 0.50]
        xs = [0, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 1]
        result = generate_one(ks, xs, nx=100, nt=100, dx=0.01, dt=0.002)
        rho = result["rho"]
        assert torch.isfinite(rho).all()
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10

    def test_extreme_contrast_10_pieces(self):
        """10 pieces alternating between near-zero and near-max density."""
        ks = [0.01, 0.99, 0.02, 0.98, 0.01, 0.99, 0.02, 0.98, 0.01, 0.99]
        n_pieces = len(ks)
        xs = [i / n_pieces for i in range(n_pieces + 1)]
        result = generate_one(ks, xs, nx=200, nt=200, dx=0.005, dt=0.001)
        rho = result["rho"]
        assert torch.isfinite(rho).all()
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10

    def test_single_narrow_pulse_9_pieces(self):
        """Low background with a narrow high-density pulse in the middle."""
        ks = [0.1, 0.1, 0.1, 0.1, 0.95, 0.1, 0.1, 0.1, 0.1]
        xs = [0, 0.1, 0.2, 0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 1]
        result = generate_one(ks, xs, nx=100, nt=150, dx=0.01, dt=0.002)
        rho = result["rho"]
        assert torch.isfinite(rho).all()
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10

    def test_long_time_evolution_6_pieces(self):
        """Run long enough for all waves to interact multiple times."""
        ks = [0.8, 0.2, 0.7, 0.3, 0.6, 0.4]
        xs = [0, 0.15, 0.3, 0.5, 0.65, 0.85, 1]
        result = generate_one(ks, xs, nx=100, nt=500, dx=0.01, dt=0.002)
        rho = result["rho"]
        assert torch.isfinite(rho).all()
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10


# ============================================================================
# generate_n with many pieces
# ============================================================================


class TestManyPieceGenerateN:
    """generate_n with higher piece counts."""

    @pytest.mark.parametrize("k", [5, 8, 10, 15])
    def test_shapes(self, k):
        """Output shapes are correct for various k values."""
        n, nx, nt = 4, 60, 30
        result = generate_n(
            n, k, nx=nx, nt=nt, dx=1.0 / nx, dt=0.005,
            seed=42, show_progress=False,
        )
        assert result["rho"].shape == (n, nt, nx)
        assert result["ic_xs"].shape == (n, k + 1)
        assert result["ic_ks"].shape == (n, k)

    @pytest.mark.parametrize("k", [5, 8, 10, 15])
    def test_density_bounds(self, k):
        """Density stays in [0, 1] for many-piece random ICs."""
        n, nx, nt = 6, 60, 50
        result = generate_n(
            n, k, nx=nx, nt=nt, dx=1.0 / nx, dt=0.005,
            seed=42, show_progress=False,
        )
        rho = result["rho"]
        assert torch.isfinite(rho).all(), f"NaN/Inf for k={k}"
        assert rho.min() >= -1e-10, f"Negative density for k={k}: {rho.min()}"
        assert rho.max() <= 1.0 + 1e-10, f"Density > 1 for k={k}: {rho.max()}"

    @pytest.mark.parametrize("k", [5, 10])
    def test_only_shocks(self, k):
        """only_shocks=True sorts ks ascending, all results remain valid."""
        n, nx, nt = 4, 60, 50
        result = generate_n(
            n, k, nx=nx, nt=nt, dx=1.0 / nx, dt=0.005,
            only_shocks=True, seed=42, show_progress=False,
        )
        rho = result["rho"]
        assert torch.isfinite(rho).all()
        # Verify ks are sorted ascending for each sample
        for i in range(n):
            ks = result["ic_ks"][i]
            assert all(ks[j] <= ks[j + 1] for j in range(len(ks) - 1))

    @pytest.mark.parametrize("k", [5, 10])
    def test_reproducibility(self, k):
        """Same seed produces identical results for many-piece ICs."""
        kwargs = dict(nx=40, nt=20, dx=0.025, dt=0.005, seed=99, show_progress=False)
        r1 = generate_n(3, k, **kwargs)
        r2 = generate_n(3, k, **kwargs)
        torch.testing.assert_close(r1["rho"], r2["rho"])
        np.testing.assert_array_equal(r1["ic_ks"], r2["ic_ks"])
        np.testing.assert_array_equal(r1["ic_xs"], r2["ic_xs"])

    def test_breakpoints_ordered(self):
        """Breakpoints are strictly increasing for all samples."""
        n, k, nx, nt = 8, 10, 80, 20
        result = generate_n(
            n, k, nx=nx, nt=nt, dx=1.0 / nx, dt=0.005,
            seed=42, show_progress=False,
        )
        for i in range(n):
            xs = result["ic_xs"][i]
            assert len(xs) == k + 1
            assert xs[0] == pytest.approx(0.0)
            assert xs[-1] == pytest.approx(1.0)
            for j in range(len(xs) - 1):
                assert xs[j] < xs[j + 1], f"Breakpoints not sorted: sample {i}"

    def test_ic_values_in_range(self):
        """Piece values respect the specified rho_range."""
        n, k = 6, 10
        rho_range = (0.1, 0.8)
        result = generate_n(
            n, k, nx=60, nt=20, dx=1.0 / 60, dt=0.005,
            rho_range=rho_range, seed=42, show_progress=False,
        )
        for i in range(n):
            ks = result["ic_ks"][i]
            assert all(rho_range[0] <= v <= rho_range[1] for v in ks)
