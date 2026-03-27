"""Tests for ARZ initial condition generators."""

import torch

from numerical_solvers.src.arz.initial_conditions import (
    from_steps,
    random_piecewise,
    riemann,
    three_region,
)


class TestRiemann:
    def test_shapes(self):
        x = torch.arange(32, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x)
        assert rho0.shape == (32,)
        assert v0.shape == (32,)

    def test_left_right_values(self):
        x = torch.arange(32, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, x_split=0.5)
        left_mask = x < 0.5
        right_mask = x >= 0.5
        torch.testing.assert_close(rho0[left_mask], torch.full((left_mask.sum(),), 0.8))
        torch.testing.assert_close(rho0[right_mask], torch.full((right_mask.sum(),), 0.2))

    def test_uniform_velocity(self):
        x = torch.arange(32, dtype=torch.float32) * 0.05
        _, v0 = riemann(x, v0=0.3)
        torch.testing.assert_close(v0, torch.full_like(v0, 0.3))


class TestThreeRegion:
    def test_structure(self):
        x = torch.arange(50, dtype=torch.float32) * 0.02
        rho0, v0 = three_region(x, rho_left=0.3, rho_mid=0.8, rho_right=0.2, x1=0.2, x2=0.5)
        left = x < 0.2
        mid = (x >= 0.2) & (x < 0.5)
        right = x >= 0.5
        torch.testing.assert_close(rho0[left], torch.full((left.sum(),), 0.3))
        torch.testing.assert_close(rho0[mid], torch.full((mid.sum(),), 0.8))
        torch.testing.assert_close(rho0[right], torch.full((right.sum(),), 0.2))


class TestFromSteps:
    def test_with_velocity(self):
        x = torch.arange(20, dtype=torch.float32) * 0.05
        rho0, v0 = from_steps(
            x,
            rho_steps=[(0.5, 0.8), (2.0, 0.3)],
            v_steps=[(0.5, 0.1), (2.0, 0.9)],
        )
        left = x < 0.5
        torch.testing.assert_close(rho0[left], torch.full((left.sum(),), 0.8))
        torch.testing.assert_close(v0[left], torch.full((left.sum(),), 0.1))

    def test_default_velocity(self):
        x = torch.arange(20, dtype=torch.float32) * 0.05
        _, v0 = from_steps(x, rho_steps=[(2.0, 0.5)], default_v=0.3)
        torch.testing.assert_close(v0, torch.full_like(v0, 0.3))


class TestRandomPiecewise:
    def test_shapes(self):
        x = torch.arange(32, dtype=torch.float32) * 0.05
        rng = torch.Generator().manual_seed(42)
        rho0, v0, ic_params = random_piecewise(x, 3, rng)
        assert rho0.shape == (32,)
        assert v0.shape == (32,)

    def test_values_in_range(self):
        x = torch.arange(32, dtype=torch.float32) * 0.05
        rng = torch.Generator().manual_seed(42)
        rho0, v0, _ = random_piecewise(x, 4, rng, rho_range=(0.2, 0.8), v_range=(0.1, 0.5))
        assert rho0.min() >= 0.2 - 1e-6
        assert rho0.max() <= 0.8 + 1e-6
        assert v0.min() >= 0.1 - 1e-6
        assert v0.max() <= 0.5 + 1e-6

    def test_ic_params_structure(self):
        x = torch.arange(32, dtype=torch.float32) * 0.05
        rng = torch.Generator().manual_seed(42)
        _, _, ic_params = random_piecewise(x, 3, rng)
        assert "xs" in ic_params
        assert "rho_ks" in ic_params
        assert "v_ks" in ic_params
        assert len(ic_params["xs"]) == 4  # k + 1
        assert len(ic_params["rho_ks"]) == 3
        assert len(ic_params["v_ks"]) == 3

    def test_reproducibility(self):
        x = torch.arange(32, dtype=torch.float32) * 0.05
        rho1, v1, _ = random_piecewise(x, 3, torch.Generator().manual_seed(99))
        rho2, v2, _ = random_piecewise(x, 3, torch.Generator().manual_seed(99))
        torch.testing.assert_close(rho1, rho2)
        torch.testing.assert_close(v1, v2)

    def test_raises_too_many_breaks(self):
        x = torch.arange(5, dtype=torch.float32) * 0.05
        rng = torch.Generator().manual_seed(42)
        import pytest
        with pytest.raises(ValueError, match="Cannot place"):
            random_piecewise(x, 10, rng)
