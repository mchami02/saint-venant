"""Tests for LWR initial condition generators."""

import numpy as np
import pytest

from numerical_solvers.src.lwr.initial_conditions import (
    PiecewiseRandom,
    from_steps,
    random_piecewise,
    riemann,
)


class TestRiemann:
    def test_default_values(self):
        ks, xs = riemann()
        assert ks == [0.8, 0.2]
        assert xs == [0, 0.5, 1]

    def test_custom_values(self):
        ks, xs = riemann(0.3, 0.9, 0.7)
        assert ks == [0.3, 0.9]
        assert xs == [0, 0.7, 1]


class TestFromSteps:
    def test_passthrough(self):
        ks, xs = from_steps([0.1, 0.5, 0.9], [0, 0.3, 0.7, 1])
        assert ks == [0.1, 0.5, 0.9]
        assert xs == [0, 0.3, 0.7, 1]

    def test_none_xs(self):
        ks, xs = from_steps([0.1, 0.5])
        assert ks == [0.1, 0.5]
        assert xs is None


class TestRandomPiecewise:
    def test_shapes_k3(self):
        rng = np.random.default_rng(42)
        ks, xs = random_piecewise(3, rng=rng, nx=50)
        assert len(ks) == 3
        assert len(xs) == 4  # k + 1

    def test_boundaries(self):
        rng = np.random.default_rng(42)
        ks, xs = random_piecewise(3, rng=rng, nx=50)
        assert xs[0] == 0.0
        assert xs[-1] == 1.0

    def test_values_in_range(self):
        rng = np.random.default_rng(42)
        ks, xs = random_piecewise(5, rng=rng, rho_range=(0.2, 0.8), nx=50)
        for val in ks:
            assert 0.2 <= val <= 0.8

    def test_breakpoints_sorted(self):
        rng = np.random.default_rng(42)
        ks, xs = random_piecewise(5, rng=rng, nx=50)
        for i in range(len(xs) - 1):
            assert xs[i] < xs[i + 1]

    def test_reproducibility(self):
        ks1, xs1 = random_piecewise(3, rng=np.random.default_rng(99), nx=50)
        ks2, xs2 = random_piecewise(3, rng=np.random.default_rng(99), nx=50)
        assert ks1 == ks2
        assert xs1 == xs2

    def test_k1_no_internal_breakpoints(self):
        rng = np.random.default_rng(42)
        ks, xs = random_piecewise(1, rng=rng)
        assert len(ks) == 1
        assert xs == [0.0, 1.0]

    def test_raises_without_nx(self):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="nx is required"):
            random_piecewise(3, rng=rng)


class TestPiecewiseRandom:
    def test_one_per_cell(self):
        rng = np.random.default_rng(42)
        ic = PiecewiseRandom([0.1, 0.2, 0.3, 0.4, 0.5], x_noise=False, rng=rng, nx=20)
        # n_breaks = 4, should have 6 entries in xs (0, b1, b2, b3, b4, 1)
        assert len(ic.xs) == 6
        # All internal breakpoints in distinct cells
        internal = ic.xs[1:-1]
        cell_indices = np.floor(internal * 20).astype(int)
        assert len(set(cell_indices)) == len(cell_indices)

    def test_k1_trivial(self):
        ic = PiecewiseRandom([0.5], x_noise=False)
        np.testing.assert_array_equal(ic.xs, [0.0, 1.0])

    def test_raises_too_many_breaks(self):
        with pytest.raises(ValueError, match="Cannot place"):
            PiecewiseRandom([0.1] * 6, x_noise=False, nx=3)

    def test_raises_without_nx(self):
        with pytest.raises(ValueError, match="nx is required"):
            PiecewiseRandom([0.1, 0.2], x_noise=False)
