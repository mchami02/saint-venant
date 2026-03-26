"""Tests for Euler initial condition generators."""

import pytest
import torch

from numerical_solvers.euler.initial_conditions import (
    from_steps,
    random_piecewise,
    riemann,
    sod,
)


class TestSod:
    def test_canonical_values(self):
        x = torch.arange(40, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x)
        left = x < 0.5
        right = x >= 0.5
        torch.testing.assert_close(rho0[left], torch.full((left.sum(),), 1.0, dtype=torch.float64))
        torch.testing.assert_close(rho0[right], torch.full((right.sum(),), 0.125, dtype=torch.float64))
        torch.testing.assert_close(u0, torch.zeros_like(u0))
        torch.testing.assert_close(p0[left], torch.full((left.sum(),), 1.0, dtype=torch.float64))
        torch.testing.assert_close(p0[right], torch.full((right.sum(),), 0.1, dtype=torch.float64))

    def test_custom_split(self):
        x = torch.arange(20, dtype=torch.float64) * 0.05
        rho0, _, _ = sod(x, x_split=0.3)
        left = x < 0.3
        right = x >= 0.3
        torch.testing.assert_close(rho0[left], torch.full((left.sum(),), 1.0, dtype=torch.float64))
        torch.testing.assert_close(rho0[right], torch.full((right.sum(),), 0.125, dtype=torch.float64))


class TestRiemann:
    def test_custom_values(self):
        x = torch.arange(20, dtype=torch.float64) * 0.05
        rho0, u0, p0 = riemann(x, rho_left=2.0, rho_right=0.5, u_left=1.0, u_right=-1.0, p_left=3.0, p_right=0.5)
        left = x < 0.5
        right = x >= 0.5
        torch.testing.assert_close(rho0[left], torch.full((left.sum(),), 2.0, dtype=torch.float64))
        torch.testing.assert_close(u0[left], torch.full((left.sum(),), 1.0, dtype=torch.float64))
        torch.testing.assert_close(p0[right], torch.full((right.sum(),), 0.5, dtype=torch.float64))


class TestFromSteps:
    def test_rho_only(self):
        x = torch.arange(20, dtype=torch.float64) * 0.05
        rho0, u0, p0 = from_steps(x, rho_steps=[(2.0, 0.5)])
        torch.testing.assert_close(u0, torch.zeros_like(u0))
        torch.testing.assert_close(p0, torch.ones_like(p0))

    def test_all_fields(self):
        x = torch.arange(20, dtype=torch.float64) * 0.05
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.5, 1.0), (2.0, 0.5)],
            u_steps=[(0.5, 0.0), (2.0, 1.0)],
            p_steps=[(0.5, 2.0), (2.0, 0.5)],
        )
        left = x < 0.5
        torch.testing.assert_close(rho0[left], torch.full((left.sum(),), 1.0, dtype=torch.float64))
        torch.testing.assert_close(u0[left], torch.full((left.sum(),), 0.0, dtype=torch.float64))
        torch.testing.assert_close(p0[left], torch.full((left.sum(),), 2.0, dtype=torch.float64))


class TestRandomPiecewise:
    def test_shapes(self):
        x = torch.arange(32, dtype=torch.float64) * 0.05
        rng = torch.Generator().manual_seed(42)
        rho0, u0, p0, ic_params = random_piecewise(x, 3, rng)
        assert rho0.shape == (32,)
        assert u0.shape == (32,)
        assert p0.shape == (32,)

    def test_values_in_ranges(self):
        x = torch.arange(32, dtype=torch.float64) * 0.05
        rng = torch.Generator().manual_seed(42)
        rho_range = (0.2, 1.5)
        u_range = (-1.0, 1.0)
        p_range = (0.5, 3.0)
        rho0, u0, p0, _ = random_piecewise(x, 4, rng, rho_range=rho_range, u_range=u_range, p_range=p_range)
        assert rho0.min() >= rho_range[0] - 1e-6
        assert rho0.max() <= rho_range[1] + 1e-6
        assert u0.min() >= u_range[0] - 1e-6
        assert u0.max() <= u_range[1] + 1e-6
        assert p0.min() >= p_range[0] - 1e-6
        assert p0.max() <= p_range[1] + 1e-6

    def test_ic_params_keys(self):
        x = torch.arange(32, dtype=torch.float64) * 0.05
        rng = torch.Generator().manual_seed(42)
        _, _, _, ic_params = random_piecewise(x, 3, rng)
        assert set(ic_params.keys()) == {"xs", "rho_ks", "u_ks", "p_ks"}
        assert len(ic_params["xs"]) == 4  # k + 1
        assert len(ic_params["rho_ks"]) == 3
        assert len(ic_params["u_ks"]) == 3
        assert len(ic_params["p_ks"]) == 3

    def test_reproducibility(self):
        x = torch.arange(32, dtype=torch.float64) * 0.05
        rho1, u1, p1, _ = random_piecewise(x, 3, torch.Generator().manual_seed(99))
        rho2, u2, p2, _ = random_piecewise(x, 3, torch.Generator().manual_seed(99))
        torch.testing.assert_close(rho1, rho2)
        torch.testing.assert_close(u1, u2)
        torch.testing.assert_close(p1, p2)

    def test_raises_too_many_breaks(self):
        x = torch.arange(5, dtype=torch.float64) * 0.05
        rng = torch.Generator().manual_seed(42)
        with pytest.raises(ValueError, match="Cannot place"):
            random_piecewise(x, 10, rng)
